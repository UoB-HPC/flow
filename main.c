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
        mesh.local_nx, mesh.local_ny, &mesh, mesh.dt, state.e, state.P, state.u, 
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
        mesh.local_nx, mesh.local_ny, &mesh, mesh.dt_h, mesh.dt, state.e, state.slope_x0, 
        state.slope_y0, state.F_x, state.F_y, state.u, state.v, state.rho_old, state.rho,
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

  char visit_name[256];
  sprintf(visit_name, "density%d", mesh.rank);

  write_all_ranks_to_visit(
      mesh.global_nx, mesh.global_ny, mesh.local_nx, mesh.local_ny, mesh.x_off, 
      mesh.y_off, mesh.rank, mesh.nranks, state.rho, visit_name, tt, elapsed_sim_time);

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

      // TODO: possible DBZ
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

  mesh->dt = 0.5*(C_T*global_min_dt + mesh->dt_h);
  mesh->dt_h = (first_step) ? mesh->dt : C_T*global_min_dt;
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
      rho_u[ind1] -= dt*(P[ind0] - P[ind0-1])/(edgedx[jj]);
      rho_v[ind0] -= dt*(P[ind0] - P[ind0-nx])/(edgedy[ii]);

      // Calculate the zone edge centered density
      const double rho_edge_x = 
        (rho[ind0]*celldx[jj]*celldy[ii] + 
         rho[ind0-1]*celldx[jj - 1]*celldy[ii]) / 
        (2.0*edgedx[jj]*celldy[ii]);
      const double rho_edge_y = 
        (rho[ind0]*celldx[jj]*celldy[ii] + 
         rho[ind0-nx]*celldx[jj]*celldy[ii - 1]) / 
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
      rho_u[ind1] -= (dt/(edgedx[jj]*celldy[ii]))*
        (celldx[jj]*Qxx[ind0] - celldx[jj-1]*Qxx[ind0-1]);
      rho_v[ind0] -= (dt/(edgedx[jj]*celldy[ii]))*
        (celldy[ii]*Qyy[ind0] - celldy[ii-1]*Qyy[ind0-nx]);

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
    const int nx, const int ny, Mesh* mesh, const double dt, double* e, 
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

      // Calculate the heating due to shock
      double shock_heating = (rho[ind0] == 0.0) ? 0.0 : (dt*
          (Qxx[ind0]*(edgedx[jj+1]*u[ind1+1] - edgedx[jj]*u[ind1])+
           Qyy[ind0]*(edgedy[ii+1]*v[ind0+nx] - edgedy[ii]*v[ind0]))/
          (celldx[jj]*celldy[ii]*rho[ind0]));

      const double div_v = 
        (edgedx[jj + 1]*u[ind1+1] - edgedx[jj]*u[ind1]+
         edgedy[ii + 1]*v[ind0+nx] - edgedy[ii]*v[ind0])/ 
        (celldx[jj]*celldy[ii]);
      const double div_v_dt = div_v*dt;

      /// A working formulation that is second order in time for Pressure!?
      const double e_C = e[ind0] - (P[ind0]*div_v_dt)/rho[ind0];
      const double rho_C = rho[ind0]/(1.0 + div_v_dt);
      const double work = 0.5*div_v_dt*(P[ind0] + (GAM-1.0)*e_C*rho_C)/rho[ind0];

      e[ind0] -= (shock_heating + work);
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

  STOP_PROFILING(&compute_profiler, __func__);

  // Perform the dimensional splitting on the mass flux, with a the alternating
  // fix for asymmetries
  if(tt % 2 == 0) {
    x_mass_flux(nx, ny, mesh, dt_h, rho, u, F_x, celldx, edgedx, celldy, edgedy);
    y_mass_flux(nx, ny, mesh, dt_h, rho, v, F_y, celldx, edgedx, celldy, edgedy);
  }
  else {
    y_mass_flux(nx, ny, mesh, dt_h, rho, v, F_y, celldx, edgedx, celldy, edgedy);
    x_mass_flux(nx, ny, mesh, dt_h, rho, u, F_x, celldx, edgedx, celldy, edgedy);
  }

  handle_boundary(nx, ny, mesh, rho, NO_INVERT, PACK);
}

// Calculate the flux in the x direction
void x_mass_flux(
    const int nx, const int ny, Mesh* mesh, const double dt_h, double* rho, 
    const double* u, double* F_x, const double* celldx, const double* edgedx, 
    const double* celldy, const double* edgedy)
{
  START_PROFILING(&compute_profiler);

  // Compute the mass fluxes along the x edges
  // In the ghost cells flux is left as 0.0
#pragma omp parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      const double rho_diff = (rho[ind0]-rho[ind0-1]);

      // Van leer limiter
      double limiter = 0.0;
      if(rho_diff) {
        const double smoothness = (u[ind1] >= 0.0) 
          ? (rho[ind0-1]-rho[ind0-2])/rho_diff
          : (rho[ind0+1]-rho[ind0])/rho_diff;
        limiter = (smoothness + fabs(smoothness))/(1.0+fabs(smoothness));
      }

      // Calculate the flux
      const double rho_upwind = (u[ind1] >= 0.0) ? rho[ind0-1] : rho[ind0];
      F_x[ind1] = (u[ind1]*rho_upwind+
        0.5*fabs(u[ind1])*(1.0-fabs((u[ind1]*dt_h)/celldx[jj]))*limiter*rho_diff);
    }
  }

  STOP_PROFILING(&compute_profiler, "advect_mass");

  handle_boundary(nx+1, ny, mesh, F_x, INVERT_X, PACK);

  START_PROFILING(&compute_profiler);

  // Calculate the new density values
#pragma omp parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      rho[ind0] -= dt_h*
        (edgedx[jj+1]*F_x[ind1+1] - edgedx[jj]*F_x[ind1])/ 
        (celldx[jj]*celldy[ii]);
    }
  }

  STOP_PROFILING(&compute_profiler, "advect_mass");
}

// Calculate the flux in the y direction
void y_mass_flux(
    const int nx, const int ny, Mesh* mesh, const double dt_h, double* rho, 
    const double* v, double* F_y, const double* celldx, const double* edgedx, 
    const double* celldy, const double* edgedy)
{
  // Compute the mass flux along the y edges
  // In the ghost cells flux is left as 0.0
  START_PROFILING(&compute_profiler);
#pragma omp parallel for
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      const double rho_diff = (rho[ind0]-rho[ind0-nx]);

      // Van leer limiter
      double limiter = 0.0;
      if(rho_diff) {
        const double smoothness = (v[ind0] >= 0.0) 
          ? (rho[ind0-nx]-rho[ind0-nx])/rho_diff
          : (rho[ind0+nx]-rho[ind0])/rho_diff;
        limiter = (smoothness + fabs(smoothness))/(1.0+fabs(smoothness));
      }

      // Calculate the flux
      const double rho_upwind = (v[ind0] >= 0.0) ? rho[ind0-nx] : rho[ind0];
      F_y[ind0] = (v[ind0]*rho_upwind+
        0.5*fabs(v[ind0])*(1.0-fabs((v[ind0]*dt_h)/celldx[jj]))*limiter*rho_diff);
    }
  }
  STOP_PROFILING(&compute_profiler, "advect_mass");

  handle_boundary(nx, ny+1, mesh, F_y, INVERT_Y, PACK);

  // Calculate the new density values
  START_PROFILING(&compute_profiler);
#pragma omp parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      rho[ind0] -= dt_h*
        (edgedy[ii+1]*F_y[ind0+nx] - edgedy[ii]*F_y[ind0])/
        (celldx[jj]*celldy[ii]);
    }
  }
  STOP_PROFILING(&compute_profiler, "advect_mass");
}

// Advect momentum according to the velocity
void advect_momentum(
    const int nx, const int ny, Mesh* mesh, const double dt_h, const double dt, 
    double* u, double* v, double* slope_x, double* slope_y, double* mF_x, 
    double* mF_y, double* rho_u, double* rho_v, const double* rho, 
    const double* F_x, const double* F_y, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  /// nx DIMENSION ADVECTION
  START_PROFILING(&compute_profiler);
#pragma omp parallel for
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      const double S_L = (u[ind1] - u[ind1-1])/edgedx[jj];
      const double S_R = (u[ind1+1] - u[ind1])/edgedx[jj+1];
      const double S_D = (u[ind1] - u[ind1-(nx+1)])/edgedy[ii];
      const double S_U = (u[ind1+(nx+1)] - u[ind1])/edgedy[ii+1];
      slope_x[ind1] = 
        ((S_L >= 0.0) ^ (S_R < 0.0)) ? (fabs(S_L) < fabs(S_R) ? S_L : S_R) : (0.0);
      slope_y[ind0] =
        ((S_D >= 0.0) ^ (S_U < 0.0)) ? (fabs(S_D) < fabs(S_U) ? S_D : S_U) : (0.0);
    }
  }

  // Calculate the cell centered x momentum fluxes in the x direction
#pragma omp parallel for
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      const double f_x = edgedy[ii]*0.5*(F_x[ind1] + F_x[ind1+1]); 
      const double f_y = edgedx[jj]*0.5*(F_y[ind0] + F_y[ind0-1]);

      const double u_cell_x = 0.5*(u[ind1]+u[ind1+1]);
      const double v_cell_y = 0.5*(v[ind0]+v[ind0-1]);

      mF_x[ind0] = f_x*((u_cell_x >= 0.0) 
          ? u[ind1] + 0.5*slope_x[ind1]*(edgedx[jj]+u_cell_x*dt)
          : u[ind1+1] - 0.5*slope_x[ind1+1]*(edgedx[jj]-u_cell_x*dt));
      mF_y[ind1] = f_y*((v_cell_y >= 0.0)
          ? u[ind1-(nx+1)] + 0.5*slope_y[ind0-nx]*(edgedx[jj]+v_cell_y*dt)
          : u[ind1] - 0.5*slope_y[ind0]*(edgedx[jj]-v_cell_y*dt));
    }
  }
  STOP_PROFILING(&compute_profiler, __func__);

  handle_boundary(nx, ny, mesh, mF_x, NO_INVERT, PACK);
  handle_boundary(nx+1, ny+1, mesh, mF_y, NO_INVERT, PACK);

  // Calculate the axial momentum
  START_PROFILING(&compute_profiler);
#pragma omp parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      rho_u[ind1] -= dt_h*
        ((mF_x[ind0] - mF_x[ind0-1])/(edgedx[jj]*celldy[ii]) +
         (mF_y[ind1+(nx+1)] - mF_y[ind1])/(celldx[jj]*edgedy[ii]));
    }
  }

  /// ny DIMENSION ADVECTION

  // Calculate the cell centered monotonic slopes for u and v
#pragma omp parallel for
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      const double S_L = (v[ind0] - v[ind0-1])/edgedx[jj];
      const double S_R = (v[ind0+1] - v[ind0])/edgedx[jj+1];
      const double S_D = (v[ind0] - v[ind0-nx])/edgedy[ii];
      const double S_U = (v[ind0+nx] - v[ind0])/edgedy[ii+1];
      slope_x[ind1] = ((S_L >= 0.0) ^ (S_R < 0.0)) 
        ? (fabs(S_L) < fabs(S_R) ? S_L : S_R) : (0.0);
      slope_y[ind0] = ((S_D >= 0.0) ^ (S_U < 0.0)) 
        ? (fabs(S_D) < fabs(S_U) ? S_D : S_U) : (0.0);
    }
  }

  // Calculate the corner centered y momentum fluxes in the x direction
  // Calculate the cell centered y momentum fluxes in the y direction
#pragma omp parallel for
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      const double f_x = celldy[ii]*0.5*(F_x[ind1] + F_x[ind1-(nx+1)]);
      const double f_y = celldx[jj]*0.5*(F_y[ind0] + F_y[ind0+nx]);

      const double u_cell_x = 0.5*(u[ind1]+u[ind1-(nx+1)]);
      const double v_cell_y = 0.5*(v[ind0]+v[ind0+nx]);

      mF_y[ind0] = f_y*((v_cell_y >= 0.0)
          ? v[ind0] + 0.5*slope_y[ind0]*(edgedx[jj]+v_cell_y*dt)
          : v[ind0+nx] - 0.5*slope_y[ind0+nx]*(edgedx[jj]-v_cell_y*dt));
      mF_x[ind1] = f_x*((u_cell_x >= 0.0) 
          ? v[ind0-1] + 0.5*slope_x[ind1-1]*(edgedx[jj]+u_cell_x*dt)
          : v[ind0] - 0.5*slope_x[ind1]*(edgedx[jj]-u_cell_x*dt));
    }
  }
  STOP_PROFILING(&compute_profiler, __func__);

  handle_boundary(nx+1, ny+1, mesh, mF_x, NO_INVERT, PACK);
  handle_boundary(nx, ny, mesh, mF_y, NO_INVERT, PACK);

  START_PROFILING(&compute_profiler);
#pragma omp parallel for
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      rho_v[ind0] -= dt_h*(
          (mF_x[ind1+1] - mF_x[ind1])/(edgedx[jj]*celldy[ii]) +
          (mF_y[ind0] - mF_y[ind0-nx])/(celldx[jj]*edgedy[ii]));
    }
  }
  STOP_PROFILING(&compute_profiler, __func__);
}

// Perform advection of internal energy
void advect_energy(
    const int nx, const int ny, Mesh* mesh, const double dt_h, const double dt, 
    double* e, double* slope_x, double* slope_y, double* F_x, double* F_y, 
    const double* u, const double* v, const double* rho_old, const double* rho,
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  START_PROFILING(&compute_profiler);

#pragma omp parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      // Use MC limiter to get slope of energy
      const double a_x = (e[ind0+1]-e[ind0-1])/(2.0*edgedx[jj]);
      const double b_x = 2.0*(e[ind0]-e[ind0-1])/edgedx[jj];
      const double c_x = 2.0*(e[ind0+1]-e[ind0])/edgedx[jj];
      slope_x[ind0] = minmod(minmod(a_x, b_x), c_x);

      // Use MC limiter to get slope of energy
      const double a_y = (e[ind0+nx]-e[ind0-nx])/(2.0*edgedx[jj]);
      const double b_y = 2.0*(e[ind0]-e[ind0-nx])/edgedx[jj];
      const double c_y = 2.0*(e[ind0+nx]-e[ind0])/edgedx[jj];
      slope_y[ind0] = minmod(minmod(a_y, b_y), c_y);

#if 0
      const double ie_edge_d = 
        (celldy[ii]*e[ind0-nx] + celldy[ii - 1]*e[ind0])/(celldy[ii] + celldy[ii - 1]);
      const double ie_edge_u = 
        (celldy[ii + 1]*e[ind0] + celldy[ii]*e[ind0+nx])/(celldy[ii + 1] + celldy[ii]);
      const double ie_y_max = max(e[ind0-nx], max(e[ind0], e[ind0+nx]));
      const double ie_y_min = min(e[ind0-nx], min(e[ind0], e[ind0+nx]));
      const double s1_y = min(ie_y_max - e[ind0], e[ind0] - ie_y_min) / (celldy[ii] / 2.0);
      const double s2_y = (ie_edge_u - ie_edge_d) / celldy[ii];
      slope_y[ind0] = (s2_y != 0.0) ? (s2_y / fabs(s2_y))*min(fabs(s2_y), s1_y) : 0.0;
#endif // if 0
    }
  }

  // Calculate the zone edge centered energies, and flux
#pragma omp parallel for
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      // Calculate the interpolated densities
      const double edge_e_x = (u[ind1] > 0.0)
        ? e[ind0-1] + 0.5*u[ind1]*(dt/edgedx[jj])*(edgedx[jj]-u[ind1]*dt)*(slope_x[ind0]-slope_x[ind0-1])
        : e[ind0] - 0.5*u[ind1]*(dt/edgedx[jj])*(edgedx[jj]-u[ind1]*dt)*(slope_x[ind0]-slope_x[ind0-1]);

      const double edge_e_y = (v[ind1] > 0.0)
        ? e[ind0-nx] + 0.5*v[ind1]*(dt/edgedy[ii])*(edgedy[ii]-v[ind1]*dt)*(slope_y[ind0]-slope_y[ind0-1])
        : e[ind0] - 0.5*v[ind1]*(dt/edgedy[ii])*(edgedy[ii]-v[ind1]*dt)*(slope_y[ind0]-slope_y[ind0-1]);

#if 0
      const double edge_e_y = (v[ind0] > 0.0)
        ? e[ind0-nx] + 0.5*slope_y[ind0-nx]*(celldy[ii-1] - v[ind0]*dt)
        : e[ind0] - 0.5*slope_y[ind0]*(celldy[ii] + v[ind0]*dt);
#endif // if 0

      // Update the fluxes to now include the contribution from energy
      F_x[ind1] = edgedy[ii]*edge_e_x*F_x[ind1]; 
      F_y[ind0] = edgedx[jj]*edge_e_y*F_y[ind0]; 
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
#if 0
      // CENTER SQUARE TEST
      const int dist = 100;
      if(jj+mesh->x_off-PAD >= mesh->global_nx/2-dist && 
          jj+mesh->x_off-PAD < mesh->global_nx/2+dist && 
          ii+mesh->y_off-PAD >= mesh->global_ny/2-dist && 
          ii+mesh->y_off-PAD < mesh->global_ny/2+dist) {
        state->rho[ii*mesh->local_nx+jj] = 1.0;
        state->e[ii*mesh->local_nx+jj] = 2.5;
      }
#endif // if 0
      if(jj <= mesh->local_nx/2) {
        state->rho[ii*mesh->local_nx+jj] = 1.0;
        state->e[ii*mesh->local_nx+jj] = 2.5;
      }
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
  mesh->dt = 0.5*C_T*MAX_DT;
  mesh->dt_h = 0.5*C_T*MAX_DT;

  // Simple uniform rectilinear initialisation
  for(int ii = 0; ii < mesh->local_ny+1; ++ii) {
    mesh->edgedy[ii] = 10.0 / (mesh->global_ny-2*PAD);
  }
  for(int ii = 0; ii < mesh->local_ny; ++ii) {
    mesh->celldy[ii] = 10.0 / (mesh->global_ny-2*PAD);
  }
  for(int ii = 0; ii < mesh->local_nx+1; ++ii) {
    mesh->edgedx[ii] = 10.0 / (mesh->global_nx-2*PAD);
  }
  for(int ii = 0; ii < mesh->local_nx; ++ii) {
    mesh->celldx[ii] = 10.0 / (mesh->global_nx-2*PAD);
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

