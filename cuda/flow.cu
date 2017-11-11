#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "flow.h"
#include "flow.k"
#include "../../comms.h"

// Solve a single timestep on the given mesh
void solve_hydro_2d(
    Mesh* mesh, int tt, double* P, double* rho, double* rho_old, 
    double* e, double* u, double* v, double* rho_u, double* rho_v, 
    double* Qxx, double* Qyy, double* F_x, double* F_y, double* uF_x, 
    double* uF_y, double* vF_x, double* vF_y, double* reduce_array)
{
  if(mesh->rank == MASTER) {
    printf("Timestep:        %.12e\n", mesh->dt, mesh->dt_h);
  }

  int nblocks = ceil(mesh->local_nx*mesh->local_ny/(double)NTHREADS);
  START_PROFILING(&compute_profile);
  equation_of_state<<<nblocks, NTHREADS>>>(
      mesh->local_nx, mesh->local_ny, P, rho, e);
  gpu_check(cudaDeviceSynchronize());
  STOP_PROFILING(&compute_profile, "equation_of_state");

  nblocks = ceil((mesh->local_nx+1)*(mesh->local_ny+1)/(double)NTHREADS);
  START_PROFILING(&compute_profile);
  pressure_acceleration<<<nblocks, NTHREADS>>>(
      mesh->local_nx, mesh->local_ny, mesh->pad, mesh->dt, rho_u, rho_v, 
      u, v, P, rho, mesh->edgedx, mesh->edgedy, 
      mesh->celldx, mesh->celldy);
  gpu_check(cudaDeviceSynchronize());
  STOP_PROFILING(&compute_profile, "pressure_acceleration");

  handle_boundary_2d(mesh->local_nx+1, mesh->local_ny, mesh, u, INVERT_X, PACK);
  handle_boundary_2d(mesh->local_nx, mesh->local_ny+1, mesh, v, INVERT_Y, PACK);

  artificial_viscosity(
      mesh->local_nx, mesh->local_ny, mesh, mesh->dt, Qxx, Qyy, 
      u, v, rho_u, rho_v, rho, 
      mesh->edgedx, mesh->edgedy, mesh->celldx, mesh->celldy);
  gpu_check(cudaDeviceSynchronize());

  nblocks = ceil(mesh->local_nx*mesh->local_ny/(double)NTHREADS);
  START_PROFILING(&compute_profile);
  shock_heating_and_work<<<nblocks, NTHREADS>>>(
      mesh->local_nx, mesh->local_ny, mesh->pad, mesh->dt_h, e, P, u, 
      v, rho, Qxx, Qyy, mesh->celldx, mesh->celldy);
  gpu_check(cudaDeviceSynchronize());
  STOP_PROFILING(&compute_profile, "shock_heating_and_work");

  handle_boundary_2d(mesh->local_nx, mesh->local_ny, mesh, e, NO_INVERT, PACK);

  set_timestep(
      mesh->local_nx, mesh->local_ny, Qxx, Qyy, rho, 
      e, mesh, reduce_array, tt == 0, mesh->celldx, mesh->celldy);

  // Perform advection
  advect_mass_and_energy(
      mesh->local_nx, mesh->local_ny, mesh, tt, mesh->dt, mesh->dt_h, rho, e, rho_old, F_x, F_y, 
      uF_x, uF_y, u, v, mesh->edgedx, mesh->edgedy, mesh->celldx, mesh->celldy);
  gpu_check(cudaDeviceSynchronize());

  advect_momentum(
      mesh->local_nx, mesh->local_ny, tt, mesh, mesh->dt_h, mesh->dt, u, v, 
      uF_x, uF_y, vF_x, vF_y, rho_u, rho_v, rho, F_x, F_y, 
      mesh->edgedx, mesh->edgedy, mesh->celldx, mesh->celldy);
  gpu_check(cudaDeviceSynchronize());
}

// Calculate the pressure from GAMma law equation of state
void artificial_viscosity(
    const int nx, const int ny, Mesh* mesh, const double dt, double* Qxx, 
    double* Qyy, double* u, double* v, double* rho_u, double* rho_v, const double* rho, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  int nblocks = ceil(nx*ny/(double)NTHREADS);
  START_PROFILING(&compute_profile);
  calc_viscous_stresses<<<nblocks, NTHREADS>>>(
      nx, ny, mesh->pad, dt, Qxx, Qyy, u, v, rho_u, rho_v, rho, 
      edgedx, edgedy, celldx, celldy);
  gpu_check(cudaDeviceSynchronize());
  STOP_PROFILING(&compute_profile, "artificial_viscosity");

  handle_boundary_2d(nx, ny, mesh, Qxx, NO_INVERT, PACK);
  handle_boundary_2d(nx, ny, mesh, Qyy, NO_INVERT, PACK);

  nblocks = ceil((nx+1)*(ny+1)/(double)NTHREADS);
  START_PROFILING(&compute_profile);
  viscous_acceleration<<<nblocks, NTHREADS>>>(
      nx, ny, mesh->pad, dt, Qxx, Qyy, u, v, rho_u, rho_v, rho, 
      edgedx, edgedy, celldx, celldy);
  gpu_check(cudaDeviceSynchronize());
  STOP_PROFILING(&compute_profile, "artificial_viscosity");

  handle_boundary_2d(nx+1, ny, mesh, u, INVERT_X, PACK);
  handle_boundary_2d(nx, ny+1, mesh, v, INVERT_Y, PACK);
}

// Calculates the timestep from the current state
void set_timestep(
    const int nx, const int ny, double* Qxx, double* Qyy, 
    const double* rho, const double* e, Mesh* mesh, double* reduce_array, 
    const int first_step, const double* celldx, const double* celldy)
{
  int nblocks = ceil((nx+1)*(ny+1)/(double)NTHREADS);
  START_PROFILING(&compute_profile);
  calc_min_timestep<<<nblocks, NTHREADS>>>(
      nx, ny, mesh->pad, mesh->max_dt, Qxx, Qyy, rho, e, reduce_array, 
      first_step, celldx, celldy);
  gpu_check(cudaDeviceSynchronize());
  STOP_PROFILING(&compute_profile, "calc_min_timestep");

  START_PROFILING(&comms_profile);
  double local_min_dt;
  finish_min_reduce(nblocks, reduce_array, &local_min_dt);

  // Ensure that the timestep does not jump too far from one step to the next
  double global_min_dt = reduce_all_min(local_min_dt);
  const double final_min_dt = min(global_min_dt, C_M*mesh->dt_h);
  mesh->dt = 0.5*(C_T*final_min_dt + mesh->dt_h);
  mesh->dt_h = (first_step) ? mesh->dt : C_T*final_min_dt;
  gpu_check(cudaDeviceSynchronize());
  STOP_PROFILING(&comms_profile, "finish_min_reduce");
}

// Perform advection with monotonicity improvement
void advect_mass_and_energy(
    const int nx, const int ny, Mesh* mesh, const int tt, const double dt,
    const double dt_h, double* rho, double* e, double* rho_old, double* F_x, double* F_y, 
    double* eF_x, double* eF_y, const double* u, const double* v, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  int nblocks = ceil(nx*ny/(double)NTHREADS);
  START_PROFILING(&compute_profile);
  store_old_rho<<<nblocks, NTHREADS>>>(nx, ny, mesh->pad, rho, rho_old);
  gpu_check(cudaDeviceSynchronize());
  STOP_PROFILING(&compute_profile, "store_old_rho");

  if(tt % 2 == 0) {
    mass_and_energy_x_advection(
        nx, ny, 1, mesh, dt, dt_h, rho, rho_old, e, u, F_x, eF_x, 
        celldx, edgedx, celldy, edgedy);
    mass_and_energy_y_advection(
        nx, ny, 0, mesh, dt, dt_h, rho, rho_old, e, v, F_y, eF_y, 
        celldx, edgedx, celldy, edgedy);
  }
  else {
    mass_and_energy_y_advection(
        nx, ny, 1, mesh, dt, dt_h, rho, rho_old, e, v, F_y, eF_y, 
        celldx, edgedx, celldy, edgedy);
    mass_and_energy_x_advection(
        nx, ny, 0, mesh, dt, dt_h, rho, rho_old, e, u, F_x, eF_x, 
        celldx, edgedx, celldy, edgedy);
  }
}

// Advect energy and mass in the x direction
void mass_and_energy_x_advection(
    const int nx, const int ny, const int first, Mesh* mesh, const double dt, 
    const double dt_h, double* rho, double* rho_old, double* e, const double* u, 
    double* F_x, double* eF_x, const double* celldx, const double* edgedx, 
    const double* celldy, const double* edgedy)
{
  int nblocks = ceil((nx+1)*ny/(double)NTHREADS);
  START_PROFILING(&compute_profile);
  calc_x_mass_and_energy_flux<<<nblocks, NTHREADS>>>(
      nx, ny, first, mesh->pad, dt, dt_h, rho, rho_old, e, u, 
      F_x, eF_x, celldx, edgedx, celldy, edgedy);
  gpu_check(cudaDeviceSynchronize());
  STOP_PROFILING(&compute_profile, "advect_mass_and_energy");

  handle_boundary_2d(nx+1, ny, mesh, F_x, INVERT_X, PACK);

  nblocks = ceil(nx*ny/(double)NTHREADS);
  START_PROFILING(&compute_profile);
  advect_mass_and_energy_in_x<<<nblocks, NTHREADS>>>(
      nx, ny, first, mesh->pad, dt, dt_h, rho, rho_old, e, u, 
      F_x, eF_x, celldx, edgedx, celldy, edgedy);
  gpu_check(cudaDeviceSynchronize());
  STOP_PROFILING(&compute_profile, "advect_mass_and_energy");

  handle_boundary_2d(nx, ny, mesh, rho, NO_INVERT, PACK);
  handle_boundary_2d(nx, ny, mesh, e, NO_INVERT, PACK);
}

// Advect energy and mass in the y direction
void mass_and_energy_y_advection(
    const int nx, const int ny, const int first, Mesh* mesh, const double dt,
    const double dt_h, double* rho, double* rho_old, double* e, const double* v, 
    double* F_y, double* eF_y, const double* celldx, const double* edgedx, 
    const double* celldy, const double* edgedy)
{
  int nblocks = ceil(nx*(ny+1)/(double)NTHREADS);
  START_PROFILING(&compute_profile);
  calc_y_mass_and_energy_flux<<<nblocks, NTHREADS>>>(
      nx, ny, first, mesh->pad, dt, dt_h, rho, rho_old, e, v, 
      F_y, eF_y, celldx, edgedx, celldy, edgedy);
  gpu_check(cudaDeviceSynchronize());
  STOP_PROFILING(&compute_profile, "advect_mass_and_energy");

  handle_boundary_2d(nx, ny+1, mesh, F_y, INVERT_Y, PACK);

  nblocks = ceil(nx*ny/(double)NTHREADS);
  START_PROFILING(&compute_profile);
  advect_mass_and_energy_in_y<<<nblocks, NTHREADS>>>(
      nx, ny, first, mesh->pad, dt, dt_h, rho, rho_old, e, v, 
      F_y, eF_y, celldx, edgedx, celldy, edgedy);
  gpu_check(cudaDeviceSynchronize());
  STOP_PROFILING(&compute_profile, "advect_mass_and_energy");

  handle_boundary_2d(nx, ny, mesh, rho, NO_INVERT, PACK);
  handle_boundary_2d(nx, ny, mesh, e, NO_INVERT, PACK);
}

// Advect momentum according to the velocity
void advect_momentum(
    const int nx, const int ny, const int tt, Mesh* mesh, const double dt_h, 
    const double dt, double* u, double* v, double* uF_x, double* uF_y, 
    double* vF_x, double* vF_y, double* rho_u, double* rho_v, 
    const double* rho, const double* F_x, const double* F_y, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  int nblocks = 0;
  if(tt % 2) {
    nblocks = ceil(nx*ny/(double)NTHREADS);
    START_PROFILING(&compute_profile);
    ux_momentum_flux<<<nblocks, NTHREADS>>>(
        nx, ny, mesh->pad, dt_h, dt, u, v, uF_x, rho_u, rho, F_x, edgedx, edgedy, celldx, celldy);
    gpu_check(cudaDeviceSynchronize());
    STOP_PROFILING(&compute_profile, "advect momentum");

    handle_boundary_2d(nx, ny, mesh, uF_x, NO_INVERT, PACK);

    nblocks = ceil((nx+1)*ny/(double)NTHREADS);
    START_PROFILING(&compute_profile);
    advect_rho_u_and_u_in_x<<<nblocks, NTHREADS>>>(
        nx, ny, tt, mesh->pad, dt_h, dt, u, v, uF_x, uF_y, 
        vF_x, vF_y, rho_u, rho_v, rho, F_x, F_y, edgedx, edgedy, celldx, celldy);
    gpu_check(cudaDeviceSynchronize());
    STOP_PROFILING(&compute_profile, "advect momentum");

    handle_boundary_2d(nx+1, ny, mesh, u, INVERT_X, PACK);

    nblocks = ceil((nx+1)*(ny+1)/(double)NTHREADS);
    START_PROFILING(&compute_profile);
    uy_momentum_flux<<<nblocks, NTHREADS>>>(
        nx, ny, mesh->pad, dt_h, dt, u, v, uF_y, rho_u, rho, F_y, edgedx, edgedy, celldx, celldy);
    gpu_check(cudaDeviceSynchronize());
    STOP_PROFILING(&compute_profile, "advect momentum");

    handle_boundary_2d(nx+1, ny+1, mesh, uF_y, NO_INVERT, PACK);

    nblocks = ceil((nx+1)*ny/(double)NTHREADS);
    START_PROFILING(&compute_profile);
    advect_rho_u_in_y<<<nblocks, NTHREADS>>>(
        nx, ny, tt, mesh->pad, dt_h, dt, u, v, uF_x, uF_y, vF_x, vF_y, rho_u, rho_v, 
        rho, F_x, F_y, edgedx, edgedy, celldx, celldy);
    gpu_check(cudaDeviceSynchronize());
    STOP_PROFILING(&compute_profile, "advect momentum");

    nblocks = ceil((nx+1)*(ny+1)/(double)NTHREADS);
    START_PROFILING(&compute_profile);
    vx_momentum_flux<<<nblocks, NTHREADS>>>(
        nx, ny, mesh->pad, dt_h, dt, u, v, vF_x, rho_v, rho, F_x, edgedx, edgedy, celldx, celldy);
    gpu_check(cudaDeviceSynchronize());
    STOP_PROFILING(&compute_profile, "advect momentum");

    handle_boundary_2d(nx+1, ny+1, mesh, vF_x, NO_INVERT, PACK);

    nblocks = ceil(nx*(ny+1)/(double)NTHREADS);
    START_PROFILING(&compute_profile);
    advect_rho_v_and_v_in_x<<<nblocks, NTHREADS>>>(
        nx, ny, mesh->pad, dt_h, dt, u, v, vF_x, rho_v, rho, F_x, 
        edgedx, edgedy, celldx, celldy);
    gpu_check(cudaDeviceSynchronize());
    STOP_PROFILING(&compute_profile, "advect momentum");

    handle_boundary_2d(nx, ny+1, mesh, v, INVERT_Y, PACK);

    nblocks = ceil(nx*ny/(double)NTHREADS);
    START_PROFILING(&compute_profile);
    vy_momentum_flux<<<nblocks, NTHREADS>>>(
        nx, ny, mesh->pad, dt_h, dt, u, v, vF_y, rho_v, rho, F_y, edgedx, edgedy, celldx, celldy);
    gpu_check(cudaDeviceSynchronize());
    STOP_PROFILING(&compute_profile, "advect momentum");

    handle_boundary_2d(nx, ny, mesh, vF_y, NO_INVERT, PACK);

    nblocks = ceil(nx*(ny+1)/(double)NTHREADS);
    START_PROFILING(&compute_profile);
    advect_rho_v_in_y<<<nblocks, NTHREADS>>>(
        nx, ny, mesh->pad, dt_h, dt, u, v, vF_y, rho_v, rho, F_y, 
        edgedx, edgedy, celldx, celldy);
    gpu_check(cudaDeviceSynchronize());
    STOP_PROFILING(&compute_profile, "advect momentum");
  }
  else {
    nblocks = ceil((nx+1)*(ny+1)/(double)NTHREADS);
    START_PROFILING(&compute_profile);
    uy_momentum_flux<<<nblocks, NTHREADS>>>(
        nx, ny, mesh->pad, dt_h, dt, u, v, uF_y, rho_u, rho, F_y, edgedx, edgedy, celldx, celldy);
    gpu_check(cudaDeviceSynchronize());
    STOP_PROFILING(&compute_profile, "advect momentum");

    handle_boundary_2d(nx+1, ny+1, mesh, uF_y, NO_INVERT, PACK);

    nblocks = ceil((nx+1)*ny/(double)NTHREADS);
    START_PROFILING(&compute_profile);
    advect_rho_u_and_u_in_y<<<nblocks, NTHREADS>>>(
        nx, ny, tt, mesh->pad, dt_h, dt, u, v, uF_x, uF_y, vF_x, vF_y, rho_u, 
        rho_v, rho, F_x, F_y, edgedx, edgedy, celldx, celldy);
    gpu_check(cudaDeviceSynchronize());
    STOP_PROFILING(&compute_profile, "advect momentum");

    handle_boundary_2d(nx+1, ny, mesh, u, INVERT_X, PACK);

    nblocks = ceil(nx*ny/(double)NTHREADS);
    START_PROFILING(&compute_profile);
    ux_momentum_flux<<<nblocks, NTHREADS>>>(
        nx, ny, mesh->pad, dt_h, dt, u, v, uF_x, rho_u, rho, 
        F_x, edgedx, edgedy, celldx, celldy);
    gpu_check(cudaDeviceSynchronize());
    STOP_PROFILING(&compute_profile, "advect momentum");

    handle_boundary_2d(nx, ny, mesh, uF_x, NO_INVERT, PACK);

    nblocks = ceil((nx+1)*ny/(double)NTHREADS);
    START_PROFILING(&compute_profile);
    advect_rho_u_in_x<<<nblocks, NTHREADS>>>(
        nx, ny, tt, mesh->pad, dt_h, dt, u, v, uF_x, uF_y, vF_x, vF_y, rho_u, rho_v, 
        rho, F_x, F_y, edgedx, edgedy, celldx, celldy);
    gpu_check(cudaDeviceSynchronize());
    STOP_PROFILING(&compute_profile, "advect momentum");

    nblocks = ceil(nx*ny/(double)NTHREADS);
    START_PROFILING(&compute_profile);
    vy_momentum_flux<<<nblocks, NTHREADS>>>(
        nx, ny, mesh->pad, dt_h, dt, u, v, vF_y, rho_v, rho, F_y, edgedx, edgedy, celldx, celldy);
    gpu_check(cudaDeviceSynchronize());
    STOP_PROFILING(&compute_profile, "advect momentum");

    handle_boundary_2d(nx, ny, mesh, vF_y, NO_INVERT, PACK);

    nblocks = ceil(nx*(ny+1)/(double)NTHREADS);
    START_PROFILING(&compute_profile);
    advect_rho_v_and_v_in_y<<<nblocks, NTHREADS>>>(
        nx, ny, mesh->pad, dt_h, dt, u, v, vF_y, rho_v, rho, F_y, 
        edgedx, edgedy, celldx, celldy);
    gpu_check(cudaDeviceSynchronize());
    STOP_PROFILING(&compute_profile, "advect momentum");

    handle_boundary_2d(nx, ny+1, mesh, v, INVERT_Y, PACK);

    nblocks = ceil((nx+1)*(ny+1)/(double)NTHREADS);
    START_PROFILING(&compute_profile);
    vx_momentum_flux<<<nblocks, NTHREADS>>>(
        nx, ny, mesh->pad, dt_h, dt, u, v, vF_x, rho_v, rho, F_x, edgedx, edgedy, celldx, celldy);
    gpu_check(cudaDeviceSynchronize());
    STOP_PROFILING(&compute_profile, "advect momentum");

    handle_boundary_2d(nx+1, ny+1, mesh, vF_x, NO_INVERT, PACK);

    nblocks = ceil(nx*(ny+1)/(double)NTHREADS);
    START_PROFILING(&compute_profile);
    advect_rho_v_in_x<<<nblocks, NTHREADS>>>(
        nx, ny, mesh->pad, dt_h, dt, u, v, vF_x, rho_v, rho, 
        F_x, edgedx, edgedy, celldx, celldy);
    gpu_check(cudaDeviceSynchronize());
    STOP_PROFILING(&compute_profile, "advect momentum");
  }
}

// Prints some conservation values
void print_conservation(
    const int nx, const int ny, double* rho, double* e, double* reduce_array, Mesh* mesh) 
{
  START_PROFILING(&compute_profile);
  int nblocks = ceil(nx*ny/(double)NTHREADS);
  calc_mass_sum<<<nblocks, NTHREADS>>>( 
      nx, ny, mesh->pad, rho, reduce_array);
  gpu_check(cudaDeviceSynchronize());
  STOP_PROFILING(&compute_profile, __func__);

  START_PROFILING(&comms_profile);
  double local_mass_tot = 0.0;
  finish_sum_reduce(nblocks, reduce_array, &local_mass_tot);

  double global_mass_tot = reduce_to_master(local_mass_tot);

  if(mesh->rank == MASTER) {
    printf("Total mass:    %.12e\n", global_mass_tot);
  }
  STOP_PROFILING(&comms_profile, "finish_sum_reduce");
}

