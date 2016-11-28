#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "main.h"
#include "profiler.h"

#define ind0 (ii*nx + jj)
#define ind1 (ii*(nx+1) + jj)

struct Profile compute_profiler = {0};
struct Profile comms_profiler = {0};

int main(int argc, char** argv)
{
  if(argc != 4) {
    printf("usage: ./hydro.exe <local_nx> <local_y> <niters>\n");
    exit(1);
  }

  // Store the dimensions of the mesh
  Mesh mesh = {0};
  State state = {0};
  mesh.global_nx = atoi(argv[1]);
  mesh.global_ny = atoi(argv[2]);
  mesh.local_nx = atoi(argv[1]) + 2*PAD;
  mesh.local_ny = atoi(argv[2]) + 2*PAD;
  mesh.rank = MASTER;
  mesh.nranks = 1;
  mesh.niters = atoi(argv[3]);

  initialise_comms(argc, argv, &mesh);
  initialise_mesh(&mesh);
  initialise_state(&state, &mesh);

  set_timestep(
      mesh.local_nx, mesh.local_ny, state.Qxx, state.Qyy, state.rho, state.u, 
      state.v, state.e, &mesh, 0, mesh.edgedx, mesh.edgedy, mesh.celldx, mesh.celldy);

  // Prepare for solve
  struct Profile w = {0};
  double elapsed_sim_time = 0.0;

  // Main timestep loop
  int tt;
  for(tt = 0; tt < mesh.niters; ++tt) {

    if(mesh.rank == MASTER) 
      printf("Iteration %d\n", tt+1);

    START_PROFILING(&w);

    equation_of_state(
        mesh.local_nx, mesh.local_ny, state.P, state.rho, state.e);

    lagrangian_step(
        mesh.local_nx, mesh.local_ny, &mesh, mesh.dt, state.rho_u, state.rho_v, 
        state.u, state.v, state.P, state.rho, mesh.edgedx, mesh.edgedy, 
        mesh.celldx, mesh.celldy);

    artificial_viscosity(
        mesh.local_nx, mesh.local_ny, &mesh, mesh.dt, state.Qxx, state.Qyy, 
        state.u, state.v, state.rho_u, state.rho_v, state.rho, 
        mesh.edgedx, mesh.edgedy, mesh.celldx, mesh.celldy);

    set_timestep(
        mesh.local_nx, mesh.local_ny, state.Qxx, state.Qyy, state.rho, 
        state.u, state.v, state.e, &mesh, tt == 0, 
        mesh.edgedx, mesh.edgedy, mesh.celldx, mesh.celldy);

    if(mesh.rank == MASTER)
      printf("dt %.12e dt_h %.12e\n", mesh.dt, mesh.dt_h);

    shock_heating_and_work(
        mesh.local_nx, mesh.local_ny, &mesh, mesh.dt_h, state.e, state.P, state.u, 
        state.v, state.rho, state.Qxx, state.Qyy, mesh.edgedx, mesh.edgedy, 
        mesh.celldx, mesh.celldy);

    // Perform advection
    advect_mass(
        mesh.local_nx, mesh.local_ny, &mesh, tt, mesh.dt_h, state.rho, 
        state.rho_old, state.F_x, state.F_y, state.u, state.v, mesh.edgedx, 
        mesh.edgedy, mesh.celldx, mesh.celldy);

    advect_momentum(
        mesh.local_nx, mesh.local_ny, &mesh, mesh.dt_h, mesh.dt, state.u, state.v, state.slope_x1, 
        state.slope_y1, state.mF_x, state.mF_y, state.rho_u, state.rho_v, 
        state.rho, state.F_x, state.F_y, mesh.edgedx, mesh.edgedy, mesh.celldx, mesh.celldy);

    advect_energy(
        mesh.local_nx, mesh.local_ny, &mesh, mesh.dt_h, mesh.dt, state.e, 
        state.F_x, state.F_y, state.u, state.v, state.rho_old, state.rho,
        mesh.edgedx, mesh.edgedy, mesh.celldx, mesh.celldy);

    STOP_PROFILING(&w, "wallclock");

    elapsed_sim_time += mesh.dt;
    if(elapsed_sim_time >= SIM_END) {
      if(mesh.rank == MASTER)
        printf("reached end of simulation time\n");
      break;
    }

    print_conservation(mesh.local_nx, mesh.local_ny, &state, &mesh);

    if(mesh.rank == MASTER) {
      printf("simulation time: %.4lf(s)\n", elapsed_sim_time);
    }
  }

  double global_wallclock = 0.0;
  if(tt > 0) {
#ifdef MPI
    struct ProfileEntry pe = profiler_get_profile_entry(&w, "wallclock");
    MPI_Reduce(&pe.time, &global_wallclock, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
#endif
  }

  if(mesh.rank == MASTER) {
    PRINT_PROFILING_RESULTS(&compute_profiler);
    PRINT_PROFILING_RESULTS(&comms_profiler);
    printf("Wallclock %.2fs, Elapsed Simulation Time %.4fs\n", global_wallclock, elapsed_sim_time);
  }

  write_all_ranks_to_visit(
      mesh.global_nx, mesh.global_ny, mesh.local_nx, mesh.local_ny, mesh.x_off, 
      mesh.y_off, mesh.rank, mesh.nranks, state.rho, "density", 0, elapsed_sim_time);
  write_all_ranks_to_visit(
      mesh.global_nx, mesh.global_ny, mesh.local_nx, mesh.local_ny, mesh.x_off, 
      mesh.y_off, mesh.rank, mesh.nranks, state.e, "energy", 0, elapsed_sim_time);
  write_all_ranks_to_visit(
      mesh.global_nx+1, mesh.global_ny, mesh.local_nx+1, mesh.local_ny, mesh.x_off, 
      mesh.y_off, mesh.rank, mesh.nranks, state.u, "u", 0, elapsed_sim_time);
  write_all_ranks_to_visit(
      mesh.global_nx, mesh.global_ny+1, mesh.local_nx, mesh.local_ny+1, mesh.x_off, 
      mesh.y_off, mesh.rank, mesh.nranks, state.v, "v", 0, elapsed_sim_time);

  finalise_state(&state);
  finalise_mesh(&mesh);

  return 0;
}

void initialise_comms(
    int argc, char** argv, Mesh* mesh)
{
  for(int ii = 0; ii < NNEIGHBOURS; ++ii) {
    mesh->neighbours[ii] = EDGE;
  }

#ifdef MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mesh->rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mesh->nranks);

  decompose_2d_cartesian(
      mesh->rank, mesh->nranks, mesh->global_nx, mesh->global_ny,
      mesh->neighbours, &mesh->local_nx, &mesh->local_ny, &mesh->x_off, &mesh->y_off);

  // Add on the halo padding to the local mesh
  mesh->local_nx += 2*PAD;
  mesh->local_ny += 2*PAD;
#endif 

  if(mesh->rank == MASTER)
    printf("Problem dimensions %dx%d for %d iterations.\n", 
        mesh->global_nx, mesh->global_ny, mesh->niters);
}

// Calculate the pressure from GAMma law equation of state
void equation_of_state(
    const int nx, const int ny, double* P, const double* rho, const double* e)
{
  START_PROFILING(&compute_profiler);

#pragma omp parallel for
  for(int ii = 0; ii < ny; ++ii) {
#pragma omp simd
    for(int jj = 0; jj < nx; ++jj) {
      // Only invoke simple GAMma law at the moment
      P[ind0] = (GAM - 1.0)*rho[ind0]*e[ind0];
    }
  }

  STOP_PROFILING(&compute_profiler, __func__);
}

// Calculates the timestep from the cuyyent state
void set_timestep(
    const int nx, const int ny, double* Qxx, double* Qyy, const double* rho, 
    const double* u, const double* v, const double* e, Mesh* mesh, const int first_step,
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  double local_min_dt = MAX_DT;

  START_PROFILING(&compute_profiler);
  // Check the minimum timestep from the sound speed in the nx and ny directions
#pragma omp parallel for reduction(min: local_min_dt)
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd 
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      double thread_min_dt = 1.0;

      // Constrain based on the artificial viscous stresses
      const double xmask = (Qxx[ind0] != 0.0) ? 0.0 : MAX_DT;
      thread_min_dt = min(thread_min_dt, xmask+0.25*edgedx[jj]*sqrt(rho[ind0]/Qxx[ind0]));
      const double ymask = (Qyy[ind0] != 0.0) ? 0.0 : MAX_DT;
      thread_min_dt = min(thread_min_dt, ymask+0.25*edgedy[ii]*sqrt(rho[ind0]/Qyy[ind0]));

      // Constrain based on the sound speed within the system
      const double c_s = sqrt(GAM*(GAM - 1.0)*e[ind0]);
      thread_min_dt = min(thread_min_dt, (celldx[jj]/(fabs(u[ind1]) + c_s)));
      thread_min_dt = min(thread_min_dt, (celldy[ii]/(fabs(v[ind0]) + c_s)));
      local_min_dt = min(local_min_dt, thread_min_dt);
    }
  }
  STOP_PROFILING(&compute_profiler, __func__);

  double global_min_dt = local_min_dt;

#ifdef MPI
  START_PROFILING(&comms_profiler);
  MPI_Allreduce(&local_min_dt, &global_min_dt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  STOP_PROFILING(&comms_profiler, "reductions");
#endif

  // Ensure that the timestep does not jump too far from one step to the next
  const double final_min_dt = min(global_min_dt, C_M*mesh->dt_h);
  mesh->dt = 0.5*(C_T*final_min_dt + mesh->dt_h);
  mesh->dt_h = (first_step) ? mesh->dt : C_T*final_min_dt;
}

// Calculate change in momentum caused by pressure gradients, and then extract
// the velocities using edge centered density approximations
void lagrangian_step(
    const int nx, const int ny, Mesh* mesh, const double dt, double* rho_u, 
    double* rho_v, double* u, double* v, const double* P, const double* rho,
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  START_PROFILING(&compute_profiler);

#pragma omp parallel for
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      // Update the momenta using the pressure gradients
      rho_u[ind1] -= dt*(P[ind0] - P[ind0-1])/edgedx[jj];
      rho_v[ind0] -= dt*(P[ind0] - P[ind0-nx])/edgedy[ii];

      // Calculate the zone edge centered density
      const double rho_edge_x = 
        (rho[ind0]*celldx[jj]*celldy[ii] + rho[ind0-1]*celldx[jj - 1]*celldy[ii])/ 
        (2.0*edgedx[jj]*celldy[ii]);
      const double rho_edge_y = 
        (rho[ind0]*celldx[jj]*celldy[ii] + rho[ind0-nx]*celldx[jj]*celldy[ii - 1])/ 
        (2.0*celldx[jj]*edgedy[ii]);

      // Find the velocities from the momenta and edge centered mass densities
      u[ind1] = (rho_edge_x == 0.0) ? 0.0 : rho_u[ind1] / rho_edge_x;
      v[ind0] = (rho_edge_y == 0.0) ? 0.0 : rho_v[ind0] / rho_edge_y;
    }
  }

  STOP_PROFILING(&compute_profiler, __func__);

  handle_boundary(nx+1, ny, mesh, u, INVERT_X, PACK);
  handle_boundary(nx, ny+1, mesh, v, INVERT_Y, PACK);
}

void artificial_viscosity(
    const int nx, const int ny, Mesh* mesh, const double dt, double* Qxx, 
    double* Qyy, double* u, double* v, double* rho_u, double* rho_v, const double* rho, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  START_PROFILING(&compute_profiler);

  // Calculate the artificial viscous stresses
#pragma omp parallel for 
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      const double u_diff = (u[ind1+1] - u[ind1]);
      const double v_diff = (v[ind0+nx] - v[ind0]);
      Qxx[ind0] = (u_diff >= 0.0) ? 0.0 : C_Q*rho[ind0]*u_diff*u_diff;
      Qyy[ind0] = (v_diff >= 0.0) ? 0.0 : C_Q*rho[ind0]*v_diff*v_diff;
    }
  }

  STOP_PROFILING(&compute_profiler, __func__);

  // TODO: WE SHOULDN'T FILL IN GAPS HERE RIGHT?
  handle_boundary(nx, ny, mesh, Qxx, NO_INVERT, PACK);
  handle_boundary(nx, ny, mesh, Qyy, NO_INVERT, PACK);

  START_PROFILING(&compute_profiler);

  // Update the momenta by the artificial viscous stresses
#pragma omp parallel for
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      rho_u[ind1] -= dt*(Qxx[ind0] - Qxx[ind0-1])/edgedx[jj];
      rho_v[ind0] -= dt*(Qyy[ind0] - Qyy[ind0-nx])/celldy[ii];

      // Calculate the zone edge centered density
      const double rho_edge_x = 
        (rho[ind0]*celldx[jj]*celldy[ii] + rho[ind0-1]*celldx[jj-1]*celldy[ii]) / 
        (2.0*edgedx[jj]*celldy[ii]);
      const double rho_edge_y = 
        (rho[ind0]*celldx[jj]*celldy[ii] + rho[ind0-nx]*celldx[jj]*celldy[ii-1]) / 
        (2.0*celldx[jj]*edgedy[ii]);

      // Find the velocities from the momenta and edge centered mass densities
      u[ind1] = (rho_edge_x == 0.0) ? 0.0 : rho_u[ind1] / rho_edge_x;
      v[ind0] = (rho_edge_y == 0.0) ? 0.0 : rho_v[ind0] / rho_edge_y;
    }
  }
  STOP_PROFILING(&compute_profiler, __func__);

  handle_boundary(nx+1, ny, mesh, u, INVERT_X, PACK);
  handle_boundary(nx, ny+1, mesh, v, INVERT_Y, PACK);
}

// Calculates the work done due to forces within the element
void shock_heating_and_work(
    const int nx, const int ny, Mesh* mesh, const double dt_h, double* e, 
    const double* P, const double* u, const double* v, const double* rho, 
    const double* Qxx, const double* Qyy, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  START_PROFILING(&compute_profiler);

#pragma omp parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      if(rho[ind0] == 0.0) {
        e[ind0] = 0.0;
        continue;
      }

      const double div_v = 
        (u[ind1+1] - u[ind1])/celldx[jj] + (v[ind0+nx] - v[ind0])/celldy[ii];
      const double div_v_dt = div_v*dt_h;

      const double rho_c = rho[ind0]/(1.0 + div_v_dt);

      /// A working formulation that is second order in time for Pressure!?
      const double e_c = e[ind0] - (P[ind0]*div_v_dt)/rho[ind0];
      const double work = 0.5*div_v_dt*(P[ind0] + (GAM-1.0)*e_c*rho_c)/rho[ind0];

      // Calculate the heating due to shock
      double shock_heating = (dt_h*
        (Qxx[ind0]*(u[ind1+1] - u[ind1])/(celldx[jj]*rho_c)+
         Qyy[ind0]*(v[ind0+nx] - v[ind0])/(celldy[ii]*rho_c)));

      e[ind0] -= (work + shock_heating);
    }
  }

  STOP_PROFILING(&compute_profiler, __func__);

  handle_boundary(nx, ny, mesh, e, NO_INVERT, PACK);
}

// Perform advection with monotonicity improvement
void advect_mass(
    const int nx, const int ny, Mesh* mesh, const int tt, const double dt_h, 
    double* rho, double* rho_old, double* F_x, double* F_y, const double* u, 
    const double* v, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  // Store the current value of rho
  START_PROFILING(&compute_profiler);

#pragma omp parallel for
  for(int ii = 0; ii < nx*ny; ++ii) {
    rho_old[ii] = rho[ii];
  }

  // Compute the mass fluxes along the x edges
  // In the ghost cells flux is left as 0.0
#pragma omp parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      const double rho_x_diff = (rho[ind0]-rho[ind0-1]);
      const double rho_y_diff = (rho[ind0]-rho[ind0-nx]);

      // Van leer limiter
      double limiterx = 0.0;
      if(rho_x_diff) {
        const double smoothness = (u[ind1] >= 0.0) 
          ? (rho[ind0-1]-rho[ind0-2])/rho_x_diff
          : (rho[ind0+1]-rho[ind0])/rho_x_diff;
        limiterx = (smoothness + fabs(smoothness))/(1.0+fabs(smoothness));
      }

      // Van leer limiter
      double limitery = 0.0;
      if(rho_y_diff) {
        const double smoothness = (v[ind0] >= 0.0) 
          ? (rho[ind0-nx]-rho[ind0-2*nx])/rho_y_diff
          : (rho[ind0+nx]-rho[ind0])/rho_y_diff;
        limitery = (smoothness + fabs(smoothness))/(1.0+fabs(smoothness));
      }

      // Calculate the flux
      const double rho_x_upwind = (u[ind1] >= 0.0) ? rho[ind0-1] : rho[ind0];
      F_x[ind1] = (u[ind1]*rho_x_upwind+
          0.5*fabs(u[ind1])*(1.0-fabs((u[ind1]*dt_h)/celldx[jj]))*limiterx*rho_x_diff);
      const double rho_y_upwind = (v[ind0] >= 0.0) ? rho[ind0-nx] : rho[ind0];
      F_y[ind0] = (v[ind0]*rho_y_upwind+
          0.5*fabs(v[ind0])*(1.0-fabs((v[ind0]*dt_h)/celldy[ii]))*limitery*rho_y_diff);
    }
  }
  STOP_PROFILING(&compute_profiler, "advect_mass");

  handle_boundary(nx+1, ny, mesh, F_x, INVERT_X, PACK);
  handle_boundary(nx, ny+1, mesh, F_y, INVERT_Y, PACK);

  // Calculate the new density values
  START_PROFILING(&compute_profiler);
#pragma omp parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      rho[ind0] -= dt_h*
        (edgedx[jj+1]*F_x[ind1+1] - edgedx[jj]*F_x[ind1]+
         edgedy[ii+1]*F_y[ind0+nx] - edgedy[ii]*F_y[ind0])/
        (celldx[jj]*celldy[ii]);
    }
  }
  STOP_PROFILING(&compute_profiler, "advect_mass");
  handle_boundary(nx, ny, mesh, rho, NO_INVERT, PACK);
}

// Advect momentum according to the velocity
void advect_momentum(
    const int nx, const int ny, Mesh* mesh, const double dt_h, const double dt, 
    double* u, double* v, double* uF_x, double* uF_y, double* vF_x, 
    double* vF_y, double* rho_u, double* rho_v, const double* rho, 
    const double* F_x, const double* F_y, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  // Calculate the cell centered x momentum fluxes in the x direction
#pragma omp parallel for
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      const double invdx = 1.0/edgedx[jj];
      const double invdy = 1.0/edgedy[ii];

      // Construct all required slopes for the stencil to interpolate the
      // velocities at the cell centers
      const double S_L_0 = invdx*(u[ind1] - u[ind1-1]);
      const double S_C_0 = invdx*(u[ind1+1] - u[ind1]);
      const double S_R_0 = invdx*(u[ind1+2] - u[ind1+1]);
      const double S_D_1 = invdy*(u[ind1-(nx+1)] - u[ind1-2*(nx+1)]);
      const double S_C_1 = invdy*(u[ind1] - u[ind1-(nx+1)]);
      const double S_U_1 = invdy*(u[ind1+(nx+1)] - u[ind1]);

      // Construct the fluxes
      const double f_x = edgedx[jj]*0.5*(F_x[ind1] + F_x[ind1+1]); 
      const double f_y = edgedy[ii]*0.5*(F_y[ind0] + F_y[ind0-1]);
      const double u_cell_x = 0.5*(u[ind1]+u[ind1+1]);
      const double v_cell_y = 0.5*(v[ind0]+v[ind0-1]);

      // Construct the fluxes from the slopes
      uF_x[ind0] = f_x*((u_cell_x >= 0.0) 
          ? u[ind1] + 0.5*minmod(S_L_0, S_C_0)*(edgedx[jj]+u_cell_x*dt)
          : u[ind1+1] - 0.5*minmod(S_C_0, S_R_0)*(edgedx[jj]-u_cell_x*dt));
      uF_y[ind1] = f_y*((v_cell_y >= 0.0)
          ? u[ind1-(nx+1)] + 0.5*minmod(S_D_1, S_C_1)*(edgedy[ii]+v_cell_y*dt)
          : u[ind1] - 0.5*minmod(S_C_1, S_U_1)*(edgedy[ii]-v_cell_y*dt));
    }
  }
  STOP_PROFILING(&compute_profiler, __func__);

  handle_boundary(nx, ny, mesh, uF_x, NO_INVERT, PACK);
  handle_boundary(nx+1, ny+1, mesh, uF_y, NO_INVERT, PACK);

  // Calculate the axial momentum
  START_PROFILING(&compute_profiler);
#pragma omp parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      rho_u[ind1] -= dt_h*
        ((uF_x[ind0] - uF_x[ind0-1])/(edgedx[jj]*celldy[ii]) +
         (uF_y[ind1+(nx+1)] - uF_y[ind1])/(celldx[jj]*edgedy[ii]));
    }
  }

  /// ny DIMENSION ADVECTION
  // Calculate the corner centered y momentum fluxes in the x direction
  // Calculate the cell centered y momentum fluxes in the y direction
#pragma omp parallel for
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      const double invdx = 1.0/edgedx[jj];
      const double invdy = 1.0/edgedy[ii];

      const double S_L_0 = invdx*(v[ind0-1] - v[ind0-2]);
      const double S_C_0 = invdx*(v[ind0] - v[ind0-1]);
      const double S_R_0 = invdx*(v[ind0+1] - v[ind0]);
      const double S_D_1 = invdy*(v[ind0] - v[ind0-nx]);
      const double S_C_1 = invdy*(v[ind0+nx] - v[ind0]);
      const double S_U_1 = invdy*(v[ind0+2*nx] - v[ind0+nx]);

      const double f_x = celldx[jj]*0.5*(F_x[ind1] + F_x[ind1-(nx+1)]);
      const double f_y = celldy[ii]*0.5*(F_y[ind0] + F_y[ind0+nx]);
      const double u_cell_x = 0.5*(u[ind1]+u[ind1-(nx+1)]);
      const double v_cell_y = 0.5*(v[ind0]+v[ind0+nx]);

      vF_x[ind1] = f_x*((u_cell_x >= 0.0) 
          ? v[ind0-1] + 0.5*minmod(S_L_0, S_C_0)*(edgedx[jj]+u_cell_x*dt)
          : v[ind0] - 0.5*minmod(S_C_0, S_R_0)*(edgedx[jj]-u_cell_x*dt));
      vF_y[ind0] = f_y*((v_cell_y >= 0.0)
          ? v[ind0] + 0.5*minmod(S_D_1, S_C_1)*(edgedy[ii]+v_cell_y*dt)
          : v[ind0+nx] - 0.5*minmod(S_C_1, S_U_1)*(edgedy[ii]-v_cell_y*dt));
    }
  }
  STOP_PROFILING(&compute_profiler, __func__);

  handle_boundary(nx+1, ny+1, mesh, vF_x, NO_INVERT, PACK);
  handle_boundary(nx, ny, mesh, vF_y, NO_INVERT, PACK);

  START_PROFILING(&compute_profiler);
#pragma omp parallel for
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      rho_v[ind0] -= dt_h*(
          (vF_x[ind1+1] - vF_x[ind1])/(edgedx[jj]*celldy[ii]) +
          (vF_y[ind0] - vF_y[ind0-nx])/(celldx[jj]*edgedy[ii]));
    }
  }
  STOP_PROFILING(&compute_profiler, __func__);
}

// Perform advection of internal energy
void advect_energy(
    const int nx, const int ny, Mesh* mesh, const double dt_h, const double dt, 
    double* e, double* F_x, double* F_y, const double* u, const double* v, 
    const double* rho_old, const double* rho, const double* edgedx, 
    const double* edgedy, const double* celldx, const double* celldy)
{
  START_PROFILING(&compute_profiler);

  // Calculate the zone edge centered energies, and flux
#pragma omp parallel for
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      // Use MC limiter to get slope of energy
      const double invdx = 1.0/edgedx[jj];
      const double invdy = 1.0/edgedy[ii];
      const double a_x_0 = 0.5*invdx*(e[ind0]-e[ind0-2]);
      const double b_x_0 = 2.0*invdx*(e[ind0-1]-e[ind0-2]);
      const double c_x_0 = 2.0*invdx*(e[ind0]-e[ind0-1]);
      const double a_x_1 = 0.5*invdx*(e[ind0+1]-e[ind0-1]);
      const double b_x_1 = 2.0*invdx*(e[ind0]-e[ind0-1]);
      const double c_x_1 = 2.0*invdx*(e[ind0+1]-e[ind0]);
      const double a_y_0 = 0.5*invdy*(e[ind0]-e[ind0-2*nx]);
      const double b_y_0 = 2.0*invdy*(e[ind0-nx]-e[ind0-2*nx]);
      const double c_y_0 = 2.0*invdy*(e[ind0]-e[ind0-nx]);
      const double a_y_1 = 0.5*invdy*(e[ind0+nx]-e[ind0-nx]);
      const double b_y_1 = 2.0*invdy*(e[ind0]-e[ind0-nx]);
      const double c_y_1 = 2.0*invdy*(e[ind0+nx]-e[ind0]);

      // Calculate the interpolated densities
      const double edge_e_x = (u[ind1] > 0.0)
        ? e[ind0-1] + 0.5*minmod(minmod(a_x_0, b_x_0), c_x_0)*(celldx[jj-1] - u[ind1]*dt)
        : e[ind0] - 0.5*minmod(minmod(a_x_1, b_x_1), c_x_1)*(celldx[jj] + u[ind1]*dt);
      const double edge_e_y = (v[ind0] > 0.0)
        ? e[ind0-nx] + 0.5*minmod(minmod(a_y_0, b_y_0), c_y_0)*(celldy[ii-1] - v[ind0]*dt)
        : e[ind0] - 0.5*minmod(minmod(a_y_1, b_y_1), c_y_1)*(celldy[ii] + v[ind0]*dt);

      // Update the fluxes to now include the contribution from energy
      F_x[ind1] = edgedx[jj]*edge_e_x*F_x[ind1]; 
      F_y[ind0] = edgedy[ii]*edge_e_y*F_y[ind0]; 
    }
  }

  // Calculate the new energy values
#pragma omp parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) { 
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      e[ind0] = (rho[ind0] == 0.0) ? 0.0 : 
        (rho_old[ind0]*e[ind0] - (dt_h/(celldx[jj]*celldy[ii]))*
         (F_x[ind1+1] - F_x[ind1] + F_y[ind0+nx] - F_y[ind0]))/rho[ind0];
    }
  }

  STOP_PROFILING(&compute_profiler, __func__);

  handle_boundary(nx, ny, mesh, e, NO_INVERT, PACK);
}

// Enforce reflective boundary conditions on the problem state
void handle_boundary(
    const int nx, const int ny, Mesh* mesh, double* arr, 
    const int invert, const int pack)
{
  START_PROFILING(&comms_profiler);

#ifdef MPI
  int nmessages = 0;
  MPI_Request out_req[NNEIGHBOURS];
  MPI_Request in_req[NNEIGHBOURS];
#endif

  double x_inversion_coeff = (invert == INVERT_X) ? -1.0 : 1.0;

  if(mesh->neighbours[WEST] == EDGE) {
    // reflect at the west
#pragma omp parallel for collapse(2)
    for(int ii = 0; ii < ny; ++ii) {
      for(int dd = 0; dd < PAD; ++dd) {
        arr[ii*nx + (PAD - 1 - dd)] = x_inversion_coeff*arr[ii*nx + (PAD + dd)];
      }
    }
  }
#ifdef MPI
  else if(pack) {
#pragma omp parallel for collapse(2)
    for(int ii = 0; ii < ny; ++ii) {
      for(int dd = 0; dd < PAD; ++dd) {
        mesh->west_buffer_out[ii*PAD+dd] = arr[(ii*nx)+(PAD+dd)];
      }
    }

    MPI_Isend(mesh->west_buffer_out, ny*PAD, MPI_DOUBLE,
        mesh->neighbours[WEST], 3, MPI_COMM_WORLD, &out_req[WEST]);
    MPI_Irecv(mesh->west_buffer_in, ny*PAD, MPI_DOUBLE, 
        mesh->neighbours[WEST], 2, MPI_COMM_WORLD, &in_req[nmessages++]);
  }
#endif

  // Reflect at the east
  if(mesh->neighbours[EAST] == EDGE) {
#pragma omp parallel for collapse(2)
    for(int ii = 0; ii < ny; ++ii) {
      for(int dd = 0; dd < PAD; ++dd) {
        arr[ii*nx + (nx - PAD + dd)] = x_inversion_coeff*arr[ii*nx + (nx - 1 - PAD - dd)];
      }
    }
  }
#ifdef MPI
  else if(pack) {
#pragma omp parallel for collapse(2)
    for(int ii = 0; ii < ny; ++ii) {
      for(int dd = 0; dd < PAD; ++dd) {
        mesh->east_buffer_out[ii*PAD+dd] = arr[(ii*nx)+(nx-2*PAD+dd)];
      }
    }

    MPI_Isend(mesh->east_buffer_out, ny*PAD, MPI_DOUBLE, 
        mesh->neighbours[EAST], 2, MPI_COMM_WORLD, &out_req[EAST]);
    MPI_Irecv(mesh->east_buffer_in, ny*PAD, MPI_DOUBLE,
        mesh->neighbours[EAST], 3, MPI_COMM_WORLD, &in_req[nmessages++]);
  }
#endif

  double y_inversion_coeff = (invert == INVERT_Y) ? -1.0 : 1.0;

  // Reflect at the north
  if(mesh->neighbours[NORTH] == EDGE) {
#pragma omp parallel for collapse(2)
    for(int dd = 0; dd < PAD; ++dd) {
      for(int jj = 0; jj < nx; ++jj) {
        arr[(ny - PAD + dd)*nx + jj] = y_inversion_coeff*arr[(ny - 1 - PAD - dd)*nx + jj];
      }
    }
  }
#ifdef MPI
  else if(pack) {
#pragma omp parallel for collapse(2)
    for(int dd = 0; dd < PAD; ++dd) {
      for(int jj = 0; jj < nx; ++jj) {
        mesh->north_buffer_out[dd*nx+jj] = arr[(ny-2*PAD+dd)*nx+jj];
      }
    }

    MPI_Isend(mesh->north_buffer_out, nx*PAD, MPI_DOUBLE, 
        mesh->neighbours[NORTH], 1, MPI_COMM_WORLD, &out_req[NORTH]);
    MPI_Irecv(mesh->north_buffer_in, nx*PAD, MPI_DOUBLE,
        mesh->neighbours[NORTH], 0, MPI_COMM_WORLD, &in_req[nmessages++]);
  }
#endif

  // reflect at the south
  if(mesh->neighbours[SOUTH] == EDGE) {
#pragma omp parallel for collapse(2)
    for(int dd = 0; dd < PAD; ++dd) {
      for(int jj = 0; jj < nx; ++jj) {
        arr[(PAD - 1 - dd)*nx + jj] = y_inversion_coeff*arr[(PAD + dd)*nx + jj];
      }
    }
  }
#ifdef MPI
  else if (pack) {
#pragma omp parallel for collapse(2)
    for(int dd = 0; dd < PAD; ++dd) {
      for(int jj = 0; jj < nx; ++jj) {
        mesh->south_buffer_out[dd*nx+jj] = arr[(PAD+dd)*nx+jj];
      }
    }

    MPI_Isend(mesh->south_buffer_out, nx*PAD, MPI_DOUBLE, 
        mesh->neighbours[SOUTH], 0, MPI_COMM_WORLD, &out_req[SOUTH]);
    MPI_Irecv(mesh->south_buffer_in, nx*PAD, MPI_DOUBLE,
        mesh->neighbours[SOUTH], 1, MPI_COMM_WORLD, &in_req[nmessages++]);
  }
#endif

  // Unpack the buffers
#ifdef MPI
  if(pack) {
    MPI_Waitall(nmessages, in_req, MPI_STATUSES_IGNORE);

    if(mesh->neighbours[NORTH] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int dd = 0; dd < PAD; ++dd) {
        for(int jj = 0; jj < nx; ++jj) {
          arr[(ny-PAD+dd)*nx+jj] = mesh->north_buffer_in[dd*nx+jj];
        }
      }
    }

    if(mesh->neighbours[SOUTH] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int dd = 0; dd < PAD; ++dd) {
        for(int jj = 0; jj < nx; ++jj) {
          arr[dd*nx + jj] = mesh->south_buffer_in[dd*nx+jj];
        }
      }
    }

    if(mesh->neighbours[WEST] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int ii = 0; ii < ny; ++ii) {
        for(int dd = 0; dd < PAD; ++dd) {
          arr[ii*nx + dd] = mesh->west_buffer_in[ii*PAD+dd];
        }
      }
    }

    if(mesh->neighbours[EAST] != EDGE) {
#pragma omp parallel for collapse(2)
      for(int ii = 0; ii < ny; ++ii) {
        for(int dd = 0; dd < PAD; ++dd) {
          arr[ii*nx + (nx-PAD+dd)] = mesh->east_buffer_in[ii*PAD+dd];
        }
      }
    }
  }
#endif

  STOP_PROFILING(&comms_profiler, __func__);
}

// Initialise the state for the problem
void initialise_state(
    State* state, Mesh* mesh)
{
  // Allocate memory for all state variables
  state->rho = (double*)_mm_malloc(sizeof(double)*mesh->local_nx*mesh->local_ny, VEC_ALIGN);
  state->rho_old = (double*)_mm_malloc(sizeof(double)*mesh->local_nx*mesh->local_ny, VEC_ALIGN);
  state->P = (double*)_mm_malloc(sizeof(double)*mesh->local_nx*mesh->local_ny, VEC_ALIGN);
  state->e = (double*)_mm_malloc(sizeof(double)*mesh->local_nx*mesh->local_ny, VEC_ALIGN);
  state->rho_u = (double*)_mm_malloc(sizeof(double)*(mesh->local_nx+1)*(mesh->local_ny+1), VEC_ALIGN);
  state->rho_v = (double*)_mm_malloc(sizeof(double)*(mesh->local_nx+1)*(mesh->local_ny+1), VEC_ALIGN);
  state->u = (double*)_mm_malloc(sizeof(double)*(mesh->local_nx+1)*(mesh->local_ny+1), VEC_ALIGN);
  state->v = (double*)_mm_malloc(sizeof(double)*(mesh->local_nx+1)*(mesh->local_ny+1), VEC_ALIGN);
  state->F_x = (double*)_mm_malloc(sizeof(double)*(mesh->local_nx+1)*(mesh->local_ny+1), VEC_ALIGN);
  state->F_y = (double*)_mm_malloc(sizeof(double)*(mesh->local_nx+1)*(mesh->local_ny+1), VEC_ALIGN);
  state->mF_x = (double*)_mm_malloc(sizeof(double)*(mesh->local_nx+1)*(mesh->local_ny+1), VEC_ALIGN);
  state->mF_y = (double*)_mm_malloc(sizeof(double)*(mesh->local_nx+1)*(mesh->local_ny+1), VEC_ALIGN);
  state->slope_x0 = (double*)_mm_malloc(sizeof(double)*mesh->local_nx*mesh->local_ny, VEC_ALIGN);
  state->slope_y0 = (double*)_mm_malloc(sizeof(double)*mesh->local_nx*mesh->local_ny, VEC_ALIGN);
  state->slope_x1 = (double*)_mm_malloc(sizeof(double)*(mesh->local_nx+1)*(mesh->local_ny+1), VEC_ALIGN);
  state->slope_y1 = (double*)_mm_malloc(sizeof(double)*(mesh->local_nx+1)*(mesh->local_ny+1), VEC_ALIGN);
  state->Qxx = (double*)_mm_malloc(sizeof(double)*(mesh->local_nx+1)*(mesh->local_ny+1), VEC_ALIGN);
  state->Qyy = (double*)_mm_malloc(sizeof(double)*(mesh->local_nx+1)*(mesh->local_ny+1), VEC_ALIGN);

  // Initialise all of the memory to default values
#pragma omp parallel for
  for(int ii = 0; ii < mesh->local_nx*mesh->local_ny; ++ii) {
    state->rho[ii] = 0.0;
    state->rho_old[ii] = 0.0;
    state->P[ii] = 0.0;
    state->e[ii] = 0.0;
    state->slope_x0[ii] = 0.0;
    state->slope_y0[ii] = 0.0;
  }

#pragma omp parallel for
  for(int ii = 0; ii < (mesh->local_nx+1)*(mesh->local_ny+1); ++ii) {
    state->F_x[ii] = 0.0;
    state->F_y[ii] = 0.0;
    state->mF_x[ii] = 0.0;
    state->mF_y[ii] = 0.0;
    state->Qxx[ii] = 0.0;
    state->Qyy[ii] = 0.0;
    state->slope_x1[ii] = 0.0;
    state->slope_y1[ii] = 0.0;
    state->rho_u[ii] = 0.0;
    state->u[ii] = 0.0;
    state->v[ii] = 0.0;
    state->rho_v[ii] = 0.0;
  }

  // Initialise the entire local mesh
  for(int ii = 0; ii < mesh->local_ny; ++ii) {
    for(int jj = 0; jj < mesh->local_nx; ++jj) {
      state->rho[ii*mesh->local_nx+jj] = 0.125;
      state->e[ii*mesh->local_nx+jj] = 2.0;
    }
  }

  printf("rank %d nx %d ny %d x_off %d y_off %d global_nx %d global_ny %d\n", 
      mesh->rank, mesh->local_nx, mesh->local_ny, mesh->x_off, mesh->y_off,
      mesh->global_nx, mesh->global_ny);

  // Introduce a problem
  for(int ii = 0; ii < mesh->local_ny; ++ii) {
    for(int jj = 0; jj < mesh->local_nx; ++jj) {
      // CENTER SQUARE TEST
      const int dist = 100;
      if(jj+mesh->x_off-PAD >= mesh->global_nx/2-dist && 
          jj+mesh->x_off-PAD < mesh->global_nx/2+dist && 
          ii+mesh->y_off-PAD >= mesh->global_ny/2-dist && 
          ii+mesh->y_off-PAD < mesh->global_ny/2+dist) {
        state->rho[ii*mesh->local_nx+jj] = 1.0;
        state->e[ii*mesh->local_nx+jj] = 2.5;
      }
#if 0
      if(jj+mesh->x_off <= (mesh->global_nx/2+PAD)) {
        state->rho[ii*mesh->local_nx+jj] = 1.0;
        state->e[ii*mesh->local_nx+jj] = 2.5;
      }
#endif // if 0
#if 0
      if(ii <= mesh->local_ny/2) {
        state->rho[ii*mesh->local_nx+jj] = 1.0;
        state->e[ii*mesh->local_nx+jj] = 2.5;
      }
#endif // if 0
#if 0
      if(ii > mesh->local_ny/2) {
        state->rho[ii*mesh->local_nx+jj] = 1.0;
        state->e[ii*mesh->local_nx+jj] = 2.5;
      }
#endif // if 0
#if 0
      if(jj > mesh->local_nx/2) {
        state->rho[ii*mesh->local_nx+jj] = 1.0;
        state->e[ii*mesh->local_nx+jj] = 2.5;
      }
#endif // if 0
    }
  }
}

// Initialise the mesh describing variables
void initialise_mesh(
    Mesh* mesh)
{
  mesh->edgedx = (double*)_mm_malloc(sizeof(double)*mesh->local_nx+1, VEC_ALIGN);
  mesh->celldx = (double*)_mm_malloc(sizeof(double)*mesh->local_nx, VEC_ALIGN);
  mesh->edgedy = (double*)_mm_malloc(sizeof(double)*mesh->local_ny+1, VEC_ALIGN);
  mesh->celldy = (double*)_mm_malloc(sizeof(double)*mesh->local_ny, VEC_ALIGN);
  mesh->dt = 0.01*C_T*MAX_DT;
  mesh->dt_h = 0.01*C_T*MAX_DT;

  // Simple uniform rectilinear initialisation
  for(int ii = 0; ii < mesh->local_ny+1; ++ii) {
    mesh->edgedy[ii] = 10.0 / (mesh->global_ny);
  }
  for(int ii = 0; ii < mesh->local_ny; ++ii) {
    mesh->celldy[ii] = 10.0 / (mesh->global_ny);
  }
  for(int ii = 0; ii < mesh->local_nx+1; ++ii) {
    mesh->edgedx[ii] = 10.0 / (mesh->global_nx);
  }
  for(int ii = 0; ii < mesh->local_nx; ++ii) {
    mesh->celldx[ii] = 10.0 / (mesh->global_nx);
  }

  mesh->north_buffer_out 
    = (double*)malloc(sizeof(double)*(mesh->local_nx+1)*PAD*NVARS_TO_COMM);
  mesh->east_buffer_out  
    = (double*)malloc(sizeof(double)*(mesh->local_ny+1)*PAD*NVARS_TO_COMM);
  mesh->south_buffer_out 
    = (double*)malloc(sizeof(double)*(mesh->local_nx+1)*PAD*NVARS_TO_COMM);
  mesh->west_buffer_out  
    = (double*)malloc(sizeof(double)*(mesh->local_ny+1)*PAD*NVARS_TO_COMM);
  mesh->north_buffer_in  
    = (double*)malloc(sizeof(double)*(mesh->local_nx+1)*PAD*NVARS_TO_COMM);
  mesh->east_buffer_in   
    = (double*)malloc(sizeof(double)*(mesh->local_ny+1)*PAD*NVARS_TO_COMM);
  mesh->south_buffer_in  
    = (double*)malloc(sizeof(double)*(mesh->local_nx+1)*PAD*NVARS_TO_COMM);
  mesh->west_buffer_in   
    = (double*)malloc(sizeof(double)*(mesh->local_ny+1)*PAD*NVARS_TO_COMM);
}

// Prints some conservation values
void print_conservation(
    const int nx, const int ny, State* state, Mesh* mesh) {
  double mass_tot = 0.0;
  double energy_tot = 0.0;
#pragma omp parallel for reduction(+:mass_tot, energy_tot)
  for(int ii = PAD; ii < ny-PAD; ++ii) {
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      mass_tot += state->rho[ind0];
      energy_tot += state->rho[ind0]*state->e[ind0];
    }
  }

  double global_mass_tot = mass_tot;
  double global_energy_tot = energy_tot;

#ifdef MPI
  MPI_Reduce(&mass_tot, &global_mass_tot, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
  MPI_Reduce(&energy_tot, &global_energy_tot, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
#endif

  if(mesh->rank == MASTER) {
    printf("total mass: %.12e\n", global_mass_tot);
    printf("total energy: %.12e\n", global_energy_tot);
  }
}

// Deallocate all of the state memory
void finalise_state(State* state)
{
  _mm_free(state->F_x);
  _mm_free(state->F_y);
  _mm_free(state->rho);
  _mm_free(state->rho_old);
  _mm_free(state->slope_x0);
  _mm_free(state->slope_y0);
  _mm_free(state->slope_x1);
  _mm_free(state->slope_y1);
  _mm_free(state->u);
  _mm_free(state->v);
  _mm_free(state->P);
  _mm_free(state->e);
}

// Deallocate all of the mesh memory
void finalise_mesh(Mesh* mesh)
{
  _mm_free(mesh->edgedy);
  _mm_free(mesh->celldy);
  _mm_free(mesh->edgedx);
  _mm_free(mesh->celldx);
}

