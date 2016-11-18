#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "main.h"
#include "profiler.h"

#define ind0 (ii*nx + jj)
#define ind1 (ii*(nx+1) + jj)

int main(int argc, char** argv)
{
  if(argc != 4) {
    printf("usage: ./hydro.exe <local_nx> <local_y> <niters>\n");
    exit(1);
  }

  // Store the dimensions of the mesh
  Mesh mesh = {0};
  State state = {0};
  mesh.global_nx = atoi(argv[1]) + 2*PAD;
  mesh.global_ny = atoi(argv[2]) + 2*PAD;
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
  struct Profile p = {0};
  struct Profile w = {0};
  double elapsed_sim_time = 0.0;

  // Main timestep loop
  int tt;
  for(tt = 0; tt < mesh.niters; ++tt) {

    if(mesh.rank == MASTER) 
      printf("Iteration %d\n", tt+1);

    START_PROFILING(&w);

    START_PROFILING(&p);
    equation_of_state(
        mesh.local_nx, mesh.local_ny, state.P, state.rho, state.e);
    STOP_PROFILING(&p, "equation_of_state");

    START_PROFILING(&p);
    lagrangian_step(
        mesh.local_nx, mesh.local_ny, &mesh, mesh.dt, state.rho_u, state.rho_v, 
        state.u, state.v, state.P, state.rho, mesh.edgedx, mesh.edgedy, 
        mesh.celldx, mesh.celldy);
    STOP_PROFILING(&p, "lagrangian_step");

    START_PROFILING(&p);
    artificial_viscosity(
        mesh.local_nx, mesh.local_ny, &mesh, mesh.dt, state.Qxx, state.Qyy, 
        state.u, state.v, state.rho_u, state.rho_v, state.rho, 
        mesh.edgedx, mesh.edgedy, mesh.celldx, mesh.celldy);
    STOP_PROFILING(&p, "artificial_viscosity");

    START_PROFILING(&p);
    set_timestep(
        mesh.local_nx, mesh.local_ny, state.Qxx, state.Qyy, state.rho, 
        state.u, state.v, state.e, &mesh, tt == 0, 
        mesh.edgedx, mesh.edgedy, mesh.celldx, mesh.celldy);
    STOP_PROFILING(&p, "set_timestep");

    if(mesh.rank == MASTER)
      printf("dt %.12e dt_h %.12e\n", mesh.dt, mesh.dt_h);

    START_PROFILING(&p);
    shock_heating_and_work(
        mesh.local_nx, mesh.local_ny, &mesh, mesh.dt, state.e, state.P, state.u, 
        state.v, state.rho, state.Qxx, state.Qyy, mesh.edgedx, mesh.edgedy, 
        mesh.celldx, mesh.celldy);
    STOP_PROFILING(&p, "shock_heating_and_work");

    // Perform advection
    START_PROFILING(&p);
    advect_mass(
        mesh.local_nx, mesh.local_ny, &mesh, tt, mesh.dt_h, state.rho, 
        state.rho_old, state.F_x, state.F_y, state.u, state.v, mesh.edgedx, 
        mesh.edgedy, mesh.celldx, mesh.celldy);
    STOP_PROFILING(&p, "advect_mass");

    START_PROFILING(&p);
    advect_momentum(
        mesh.local_nx, mesh.local_ny, &mesh, mesh.dt_h, mesh.dt, state.u, state.v, state.slope_x1, 
        state.slope_y1, state.mF_x, state.mF_y, state.rho_u, state.rho_v, 
        state.rho, state.F_x, state.F_y, mesh.edgedx, mesh.edgedy, mesh.celldx, mesh.celldy);
    STOP_PROFILING(&p, "advect_momentum");

    START_PROFILING(&p);
    advect_energy(
        mesh.local_nx, mesh.local_ny, &mesh, mesh.dt_h, mesh.dt, state.e, state.slope_x0, 
        state.slope_y0, state.F_x, state.F_y, state.u, state.v, state.rho_old, state.rho,
        mesh.edgedx, mesh.edgedy, mesh.celldx, mesh.celldy);
    STOP_PROFILING(&p, "advect_energy");

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

#if 0
    if(tt%VISIT_STEP == 0) 
      write_to_visit(mesh.local_nx, mesh.local_ny, mesh.x_off, 
          mesh.y_off, state.rho, "wet_density", tt, elapsed_sim_time);
#endif // if 0
  }

  double global_wallclock = 0.0;
  if(tt > 0) {
    struct ProfileEntry pe = profiler_get_profile_entry(&w, "wallclock");
    global_wallclock = pe.time;
  }

#ifdef MPI
  /// TODO: Can't quite remember whether this is a legitimate thing to do or
  //not, do we end up reading and writing to the same location in random order?
  MPI_Allreduce(&global_wallclock, &global_wallclock, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  if(mesh.rank == MASTER) {
    PRINT_PROFILING_RESULTS(&p);

    printf("Wallclock %.2fs, Elapsed Simulation Time %.4fs\n", 
        global_wallclock, elapsed_sim_time);
  }

  char visit_name[256];
  sprintf(visit_name, "density_%d", mesh.rank);
  write_to_visit(mesh.local_nx, mesh.local_ny, mesh.x_off, 
      mesh.y_off, state.rho, visit_name, tt, elapsed_sim_time);

  finalise_state(&state);
  finalise_mesh(&mesh);

  return 0;
}

#ifdef MPI
static inline void initialise_comms(
    int argc, char** argv, Mesh* mesh)
{
  for(int ii = 0; ii < NNEIGHBOURS; ++ii) {
    mesh->neighbours[ii] = EDGE;
  }

#ifdef MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mesh->rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mesh->nranks);

  mesh->local_nx = 0;
  mesh->local_ny = 0;
  decompose_ranks(mesh);

  // Add on the halo padding to the local mesh
  mesh->local_nx += 2*PAD;
  mesh->local_ny += 2*PAD;

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
#endif 

  if(mesh->rank == MASTER)
    printf("Problem dimensions %dx%d for %d iterations.\n", 
        mesh->global_nx, mesh->global_ny, mesh->niters);
}

// Decomposes the ranks, potentially load balancing and minimising the
// ratio of perimeter to area
static inline void decompose_ranks(Mesh* mesh) 
{
  int ranks_x = 0;
  int ranks_y = 0;
  int found_even = 0;
  float min_ratio = 0.0f;

  // Determine decomposition that minimises perimeter to area ratio
  for(int ff = 1; ff <= sqrt(mesh->nranks); ++ff) {
    if(mesh->nranks % ff) continue;
    // If load balance is preferred then prioritise even split over ratio
    // Test if this split evenly decomposes into the mesh
    const int even_split_ff_x = 
      (mesh->global_nx % ff == 0 && mesh->global_ny % (mesh->nranks/ff) == 0);
    const int even_split_ff_y = 
      (mesh->global_nx % (mesh->nranks/ff) == 0 && mesh->global_ny % ff == 0);
    const int new_ranks_x = even_split_ff_x ? ff : mesh->nranks/ff;
    const int new_ranks_y = even_split_ff_x ? mesh->nranks/ff : ff;
    const int is_even = even_split_ff_x || even_split_ff_y;
    found_even |= (LOAD_BALANCE && is_even);

    const float potential_ratio = 
      (2*(new_ranks_x+new_ranks_y))/(float)(new_ranks_x*new_ranks_y);

    // Update if we minimise the ratio further, only if we don't care about load
    // balancing or have found an even split
    if((found_even <= is_even) && (min_ratio == 0.0f || potential_ratio < min_ratio)) {
      min_ratio = potential_ratio;
      // If we didn't find even split, prefer longer mesh edge on x dimension
      ranks_x = (!found_even && new_ranks_x > new_ranks_y) ? new_ranks_y : new_ranks_x;
      ranks_y = (!found_even && new_ranks_x > new_ranks_y) ? new_ranks_x : new_ranks_y;
    }
  }

  // Calculate the offsets up until our rank, and then fetch rank dimensions
  int x_off = 0;
  const int x_rank = (mesh->rank%ranks_x);
  for(int xx = 0; xx <= x_rank; ++xx) {
    mesh->x_off = x_off;
    const int x_floor = mesh->global_nx/ranks_x;
    const int x_pad_req = (mesh->global_nx != (mesh->x_off + (ranks_x-xx)*x_floor));
    mesh->local_nx = x_pad_req ? x_floor+1 : x_floor;
    x_off += mesh->local_nx;
  }
  int y_off = 0;
  const int y_rank = (mesh->rank/ranks_x);
  for(int yy = 0; yy <= y_rank; ++yy) {
    mesh->y_off = y_off;
    const int y_floor = mesh->global_ny/ranks_y;
    const int y_pad_req = (mesh->global_ny != (mesh->y_off + (ranks_y-yy)*y_floor));
    mesh->local_ny = y_pad_req ? y_floor+1 : y_floor;
    y_off += mesh->local_ny;
  }

  // Calculate the surrounding ranks
  mesh->neighbours[NORTH] = (y_rank < ranks_y-1) ? mesh->rank+ranks_x : EDGE;
  mesh->neighbours[EAST] = (x_rank < ranks_x-1) ? mesh->rank+1 : EDGE;
  mesh->neighbours[SOUTH] = (y_rank > 0) ? mesh->rank-ranks_x : EDGE;
  mesh->neighbours[WEST] = (x_rank > 0) ? mesh->rank-1 : EDGE;

  printf("rank %d neighbours %d %d %d %d\n",
      mesh->rank, mesh->neighbours[NORTH], mesh->neighbours[EAST], 
      mesh->neighbours[SOUTH], mesh->neighbours[WEST]);
}
#endif

// Calculate the pressure from GAMma law equation of state
static inline void equation_of_state(
    const int nx, const int ny, double* P, const double* rho, const double* e)
{
#pragma omp parallel for
  for(int ii = 0; ii < ny; ++ii) {
    for(int jj = 0; jj < nx; ++jj) {
      // Only invoke simple GAMma law at the moment
      P[ind0] = (GAM - 1.0)*rho[ind0]*e[ind0];
    }
  }
}

// Calculates the timestep from the cuyyent state
static inline void set_timestep(
    const int nx, const int ny, double* Qxx, double* Qyy, const double* rho, 
    const double* u, const double* v, const double* e, Mesh* mesh, const int first_step,
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  double local_min_dt = MAX_DT;

  // Check the minimum timestep from the sound speed in the nx and ny directions
#pragma omp parallel for reduction(min: local_min_dt)
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      // Constrain based on the artificial viscous stresses
      if(Qxx[ind0] != 0.0)
        local_min_dt = min(local_min_dt, 0.25*edgedx[jj]*sqrt(rho[ind0]/Qxx[ind0]));
      if(Qyy[ind0] != 0.0)
        local_min_dt = min(local_min_dt, 0.25*edgedy[ii]*sqrt(rho[ind0]/Qyy[ind0]));

      // Constrain based on the sound speed within the system
      const double c_s = sqrt(GAM*(GAM - 1.0)*e[ind0]);

      // TODO: possible DBZ
      local_min_dt = min(local_min_dt, (celldx[jj]/(fabs(u[ind1]) + c_s)));
      local_min_dt = min(local_min_dt, (celldy[ii]/(fabs(v[ind0]) + c_s)));
    }
  }

  double global_min_dt = local_min_dt;

#ifdef MPI
  MPI_Allreduce(&local_min_dt, &global_min_dt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif

  mesh->dt = 0.5*(C_T*global_min_dt + mesh->dt_h);
  mesh->dt_h = (first_step) ? mesh->dt : C_T*global_min_dt;
}

// Calculate change in momentum caused by pressure gradients, and then extract
// the velocities using edge centered density approximations
static inline void lagrangian_step(
    const int nx, const int ny, Mesh* mesh, const double dt, double* rho_u, 
    double* rho_v, double* u, double* v, const double* P, const double* rho,
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
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

  handle_boundary(nx+1, ny, mesh, u, INVERT_X, PACK);
  handle_boundary(nx, ny+1, mesh, v, INVERT_Y, PACK);
}

static inline void artificial_viscosity(
    const int nx, const int ny, Mesh* mesh, const double dt, double* Qxx, 
    double* Qyy, double* u, double* v, double* rho_u, double* rho_v, const double* rho, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
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

  handle_boundary(nx, ny, mesh, Qxx, NO_INVERT, NO_PACK);
  handle_boundary(nx, ny, mesh, Qyy, NO_INVERT, NO_PACK);

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

  handle_boundary(nx+1, ny, mesh, u, INVERT_X, PACK);
  handle_boundary(nx, ny+1, mesh, v, INVERT_Y, PACK);
}

// Calculates the work done due to forces within the element
static inline void shock_heating_and_work(
    const int nx, const int ny, Mesh* mesh, const double dt, double* e, 
    const double* P, const double* u, const double* v, const double* rho, 
    const double* Qxx, const double* Qyy, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
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

  handle_boundary(nx, ny, mesh, e, NO_INVERT, PACK);
}

// Perform advection with monotonicity improvement
static inline void advect_mass(
    const int nx, const int ny, Mesh* mesh, const int tt, const double dt_h, 
    double* rho, double* rho_old, double* F_x, double* F_y, const double* u, 
    const double* v, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  // Store the current value of rho
#pragma omp parallel for
  for(int ii = 0; ii < nx*ny; ++ii) {
    rho_old[ii] = rho[ii];
  }

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
static inline void x_mass_flux(
    const int nx, const int ny, Mesh* mesh, const double dt_h, double* rho, 
    const double* u, double* F_x, const double* celldx, const double* edgedx, 
    const double* celldy, const double* edgedy)
{
  // Compute the mass fluxes along the x edges
  // In the ghost cells flux is left as 0.0
#pragma omp parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      double rx_denom = (rho[ind0]-rho[ind0-1]);
      double rx = 0.0;
      if(rx_denom) {
        rx = (u[ind1] >= 0.0) 
          ? (rho[ind0-1] - rho[ind0-2])/rx_denom
          : (rho[ind0+1] - rho[ind0])/rx_denom;
      }

      //const double limiterx = max(0.0, max(min(1.0, 2.0*rx), min(2.0, rx)));
      const double limiterx = (rx + fabs(rx))/(1.0 + fabs(rx));
      const double hsx = (u[ind1] >= 0.0) ? 1.0 : -1.0;
      F_x[ind1] = 
        0.5*u[ind1]*((1.0+hsx)*rho[ind0-1]+(1.0-hsx)*rho[ind0])+
        0.5*fabs(u[ind1])*(1.0-fabs((u[ind1]*dt_h)/celldx[jj]))*
        limiterx*rx_denom;
    }
  }

  handle_boundary(nx+1, ny, mesh, F_x, INVERT_X, NO_PACK);

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
}

// Calculate the flux in the y direction
static inline void y_mass_flux(
    const int nx, const int ny, Mesh* mesh, const double dt_h, double* rho, 
    const double* v, double* F_y, const double* celldx, const double* edgedx, 
    const double* celldy, const double* edgedy)
{
  // Compute the mass flux along the y edges
  // In the ghost cells flux is left as 0.0
#pragma omp parallel for
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      double ry_denom = (rho[ind0]-rho[ind0-nx]);
      double ry = 0.0;
      if(ry_denom) {
        ry = (v[ind0] >= 0.0) 
          ? (rho[ind0-nx] - rho[ind0-2*nx])/ry_denom 
          : (rho[ind0+nx] - rho[ind0])/ry_denom;
      }

      //const double limitery = max(0.0, max(min(1.0, 2.0*ry), min(2.0, ry)));
      const double limitery = (ry + fabs(ry))/(1.0 + fabs(ry));
      const double hsy = (v[ind0] >= 0.0) ? 1.0 : -1.0;
      F_y[ind0] =
        0.5*v[ind0]*((1.0+hsy)*rho[ind0-nx]+(1.0-hsy)*rho[ind0])+
        0.5*fabs(v[ind0])*(1.0-fabs((v[ind0]*dt_h)/celldx[jj]))*
        limitery*ry_denom;
    }
  }

  handle_boundary(nx, ny+1, mesh, F_y, INVERT_Y, NO_PACK);

  // Calculate the new density values
#pragma omp parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      rho[ind0] -= dt_h*
        (edgedy[ii+1]*F_y[ind0+nx] - edgedy[ii]*F_y[ind0])/
        (celldx[jj]*celldy[ii]);
    }
  }
}

// Advect momentum according to the velocity
static inline void advect_momentum(
    const int nx, const int ny, Mesh* mesh, const double dt_h, const double dt, 
    double* u, double* v, double* slope_x, double* slope_y, double* mF_x, 
    double* mF_y, double* rho_u, double* rho_v, const double* rho, 
    const double* F_x, const double* F_y, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  /// nx DIMENSION ADVECTION

#pragma omp parallel for
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      const double u_x_max = max(u[ind1-1], max(u[ind1], u[ind1+1]));
      const double u_x_min = min(u[ind1-1], min(u[ind1], u[ind1+1]));
      const double u_y_max = max(u[ind1-(nx+1)], max(u[ind1], u[ind1+(nx+1)]));
      const double u_y_min = min(u[ind1-(nx+1)], min(u[ind1], u[ind1+(nx+1)]));

      // Construct absolute value of slope S_L or S_R
      const double s1_x = min(u_x_max - u[ind1], u[ind1] - u_x_min) / (celldx[jj] / 2.0);
      const double s1_y = min(u_y_max - u[ind1], u[ind1] - u_y_min) / (celldy[ii] / 2.0);

      const double u_edge_l = 
        (edgedx[jj]*u[ind1-1] + edgedx[jj-1]*u[ind1])/(edgedx[jj] + edgedx[jj - 1]);
      const double u_edge_r = 
        (edgedx[jj+1]*u[ind1] + edgedx[jj]*u[ind1+1])/(edgedx[jj+1] + edgedx[jj]);
      const double u_edge_d = 
        (edgedy[ii]*u[ind1 - (nx+1)] + edgedy[ii - 1]*u[ind1])/(edgedy[ii] + edgedy[ii - 1]);
      const double u_edge_u = 
        (edgedy[ii + 1]*u[ind1] + edgedy[ii]*u[ind1 + (nx+1)])/(edgedy[ii + 1] + edgedy[ii]);

      // Construct the slope
      const double s2_x = (u_edge_r - u_edge_l) / edgedx[jj];
      const double s2_y = (u_edge_u - u_edge_d) / edgedy[ii];

      // Define the zone centered slope (culling 0's)
      slope_x[ind1] = (s2_x != 0.0) ? (s2_x / fabs(s2_x))*min(fabs(s2_x), s1_x) : 0.0;
      slope_y[ind0] = (s2_y != 0.0) ? (s2_y / fabs(s2_y))*min(fabs(s2_y), s1_y) : 0.0;
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

  handle_boundary(nx, ny, mesh, mF_x, NO_INVERT, NO_PACK);
  handle_boundary(nx+1, ny+1, mesh, mF_y, NO_INVERT, NO_PACK);

  // Calculate the axial momentum
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

#pragma omp parallel for
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      // Calculate the maximum and minimum neighbouring density
      const double v_x_max = max(v[ind0-1], max(v[ind0], v[ind0+1]));
      const double v_x_min = min(v[ind0-1], min(v[ind0], v[ind0+1]));
      const double v_y_max = max(v[ind0-nx], max(v[ind0], v[ind0+nx]));
      const double v_y_min = min(v[ind0-nx], min(v[ind0], v[ind0+nx]));

      // Construct absolute value of slope S_L or S_R
      const double s1_x = min(v_x_max - v[ind0], v[ind0] - v_x_min) / (celldx[jj] / 2.0);
      const double s1_y = min(v_y_max - v[ind0], v[ind0] - v_y_min) / (celldy[ii] / 2.0);

      // Calculate the density interpolated from the zone center to zone boundary
      const double v_edge_l = 
        (celldx[jj]*v[ind0 - 1] + celldx[jj - 1]*v[ind0])/(celldx[jj] + celldx[jj - 1]);
      const double v_edge_r = 
        (celldx[jj + 1]*v[ind0] + celldx[jj]*v[ind0 + 1])/(celldx[jj + 1] + celldx[jj]);
      const double v_edge_d = 
        (celldy[ii]*v[ind0 - nx] + celldy[ii - 1]*v[ind0])/(celldy[ii] + celldy[ii - 1]);
      const double v_edge_u = 
        (celldy[ii + 1]*v[ind0] + celldy[ii]*v[ind0 + nx])/(celldy[ii + 1] + celldy[ii]);

      // Construct the slope
      const double s2_x = (v_edge_r - v_edge_l) / celldx[jj];
      const double s2_y = (v_edge_u - v_edge_d) / celldy[ii];

      // Define the zone centered slope (culling 0's)
      slope_x[ind1] = (s2_x != 0.0) ? (s2_x / fabs(s2_x))*min(fabs(s2_x), s1_x) : 0.0;
      slope_y[ind0] = (s2_y != 0.0) ? (s2_y / fabs(s2_y))*min(fabs(s2_y), s1_y) : 0.0;
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

  handle_boundary(nx+1, ny+1, mesh, mF_x, NO_INVERT, NO_PACK);
  handle_boundary(nx, ny, mesh, mF_y, NO_INVERT, NO_PACK);

#pragma omp parallel for
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      rho_v[ind0] -= dt_h*(
          (mF_x[ind1+1] - mF_x[ind1])/(edgedx[jj]*celldy[ii]) +
          (mF_y[ind0] - mF_y[ind0-nx])/(celldx[jj]*edgedy[ii]));
    }
  }
}

// Perform advection of internal energy
static inline void advect_energy(
    const int nx, const int ny, Mesh* mesh, const double dt_h, const double dt, 
    double* e, double* slope_x, double* slope_y, double* F_x, double* F_y, 
    const double* u, const double* v, const double* rho_old, const double* rho,
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
#pragma omp parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      // Calculate the maximum and minimum neighbouring density
      const double ie_x_max = max(e[ind0-1], max(e[ind0], e[ind0+1]));
      const double ie_x_min = min(e[ind0-1], min(e[ind0], e[ind0+1]));
      const double ie_y_max = max(e[ind0-nx], max(e[ind0], e[ind0+nx]));
      const double ie_y_min = min(e[ind0-nx], min(e[ind0], e[ind0+nx]));

      // Construct absolute value of slope S_L or S_R
      const double s1_x = min(ie_x_max - e[ind0], e[ind0] - ie_x_min) / (celldx[jj] / 2.0);
      const double s1_y = min(ie_y_max - e[ind0], e[ind0] - ie_y_min) / (celldy[ii] / 2.0);

      // Calculate the density interpolated from the zone center to zone boundary
      const double ie_edge_l = 
        (celldx[jj]*e[ind0-1] + celldx[jj - 1]*e[ind0])/(celldx[jj] + celldx[jj - 1]);
      const double ie_edge_r = 
        (celldx[jj + 1]*e[ind0] + celldx[jj]*e[ind0+1])/(celldx[jj + 1] + celldx[jj]);
      const double ie_edge_d = 
        (celldy[ii]*e[ind0-nx] + celldy[ii - 1]*e[ind0])/(celldy[ii] + celldy[ii - 1]);
      const double ie_edge_u = 
        (celldy[ii + 1]*e[ind0] + celldy[ii]*e[ind0+nx])/(celldy[ii + 1] + celldy[ii]);

      // Construct the slope
      const double s2_x = (ie_edge_r - ie_edge_l) / celldx[jj];
      const double s2_y = (ie_edge_u - ie_edge_d) / celldy[ii];

      // Define the zone centered slope (culling 0's)
      slope_x[ind0] = (s2_x != 0.0) ? (s2_x / fabs(s2_x))*min(fabs(s2_x), s1_x) : 0.0;
      slope_y[ind0] = (s2_y != 0.0) ? (s2_y / fabs(s2_y))*min(fabs(s2_y), s1_y) : 0.0;
    }
  }

  // Calculate the zone edge centered energies, and flux
#pragma omp parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      // Calculate the interpolated densities
      const double edge_e_x = (u[ind1] > 0.0)
        ? e[ind0-1] + 0.5*slope_x[ind0-1]*(celldx[jj-1] - u[ind1]*dt)
        : e[ind0] - 0.5*slope_x[ind0]*(celldx[jj] + u[ind1]*dt);
      const double edge_e_y = (v[ind0] > 0.0)
        ? e[ind0-nx] + 0.5*slope_y[ind0-nx]*(celldy[ii-1] - v[ind0]*dt)
        : e[ind0] - 0.5*slope_y[ind0]*(celldy[ii] + v[ind0]*dt);

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

  handle_boundary(nx, ny, mesh, e, NO_INVERT, PACK);
}

// Enforce reflective boundary conditions on the problem state
static inline void handle_boundary(
    const int nx, const int ny, Mesh* mesh, double* arr, 
    const int invert, const int pack)
{
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
        const int out_index = ii*PAD+dd;
        mesh->west_buffer_out[out_index] = arr[(ii*nx)+(PAD+dd)];
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
        const int out_index = ii*PAD+dd;
        mesh->east_buffer_out[out_index] = arr[(ii*nx)+(nx-2*PAD+dd)];
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
        const int out_index = dd*nx+jj;
        mesh->north_buffer_out[out_index] = arr[(ny-2*PAD+dd)*nx+jj];
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
        const int out_index = dd*nx+jj;
        mesh->south_buffer_out[out_index] = arr[(PAD+dd)*nx+jj];
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
}

// Initialise the state for the problem
static inline void initialise_state(
    State* state, Mesh* mesh)
{
  // Allocate memory for all state variables
  state->rho = (double*)malloc(sizeof(double)*mesh->local_nx*mesh->local_ny);
  state->rho_old = (double*)malloc(sizeof(double)*mesh->local_nx*mesh->local_ny);
  state->P = (double*)malloc(sizeof(double)*mesh->local_nx*mesh->local_ny);
  state->e = (double*)malloc(sizeof(double)*mesh->local_nx*mesh->local_ny);
  state->rho_u = (double*)malloc(sizeof(double)*(mesh->local_nx+1)*(mesh->local_ny+1));
  state->rho_v = (double*)malloc(sizeof(double)*(mesh->local_nx+1)*(mesh->local_ny+1));
  state->u = (double*)malloc(sizeof(double)*(mesh->local_nx+1)*(mesh->local_ny+1));
  state->v = (double*)malloc(sizeof(double)*(mesh->local_nx+1)*(mesh->local_ny+1));
  state->F_x = (double*)malloc(sizeof(double)*(mesh->local_nx+1)*(mesh->local_ny+1));
  state->F_y = (double*)malloc(sizeof(double)*(mesh->local_nx+1)*(mesh->local_ny+1));
  state->mF_x = (double*)malloc(sizeof(double)*(mesh->local_nx+1)*(mesh->local_ny+1));
  state->mF_y = (double*)malloc(sizeof(double)*(mesh->local_nx+1)*(mesh->local_ny+1));
  state->slope_x0 = (double*)malloc(sizeof(double)*mesh->local_nx*mesh->local_ny);
  state->slope_y0 = (double*)malloc(sizeof(double)*mesh->local_nx*mesh->local_ny);
  state->slope_x1 = (double*)malloc(sizeof(double)*(mesh->local_nx+1)*(mesh->local_ny+1));
  state->slope_y1 = (double*)malloc(sizeof(double)*(mesh->local_nx+1)*(mesh->local_ny+1));
  state->Qxx = (double*)malloc(sizeof(double)*(mesh->local_nx+1)*(mesh->local_ny+1));
  state->Qyy = (double*)malloc(sizeof(double)*(mesh->local_nx+1)*(mesh->local_ny+1));

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
      if(jj+mesh->x_off < mesh->global_nx/2) { //LEFT SOD
        state->rho[ii*mesh->local_nx+jj] = 1.0;
        state->e[ii*mesh->local_nx+jj] = 2.5;
      }
#endif // if 0
      if(ii+mesh->y_off < mesh->global_ny/2) { //BOTTOM SOD
        state->rho[ii*mesh->local_nx+jj] = 1.0;
        state->e[ii*mesh->local_nx+jj] = 2.5;
      }

#if 0
      // BLUE HOLE TEST 
      const int m = 258;
      const int o = 10400;
      if((ii - m)*(ii - m) + (jj - m)*(jj - m) > o) {
        state->rho[ii*mesh->local_nx+jj] = 1.0;
        state->e[ii*mesh->local_nx+jj] = 2.5;
      }
#endif // if 0

#if 0
      // CENTER SQUARE TEST
      const int dist = 20;
      if(jj+mesh->x_off-PAD >= mesh->global_nx/2-dist && 
          jj+mesh->x_off-PAD < mesh->global_nx/2+dist && 
          ii+mesh->y_off-PAD >= mesh->global_ny/2-dist && 
          ii+mesh->y_off-PAD < mesh->global_ny/2+dist) {
        state->rho[ii*mesh->local_nx+jj] = 1.0;
        state->e[ii*mesh->local_nx+jj] = 2.5;
      }
#endif // if 0
    }
  }

#if 0
  // TODO: FIX THIS - CURRENTLY ASSUMING THAT THERE IS NO VELOCITY

  // Calculate the zone edge centered density
  const double rho_edge_x = 
    (state->rho[ind0]*mesh->celldx[jj]*mesh->celldy[ii] + 
     state->rho[ind0-1]*mesh->celldx[jj - 1]*mesh->celldy[ii]) / 
    (2.0*mesh->edgedx[jj]*mesh->celldy[ii]);
  const double rho_edge_y = 
    (state->rho[ind0]*mesh->celldx[jj]*mesh->celldy[ii] + 
     state->rho[ind0-mesh->local_nx]*mesh->celldx[jj]*mesh->celldy[ii - 1]) / 
    (2.0*mesh->celldx[jj]*mesh->edgedy[ii]);

  // Find the velocities from the momenta and edge centered mass densities
  state->rho_u[ind1] = state->u[ind1] * rho_edge_x;
  state->rho_v[ind0] = state->v[ind0] * rho_edge_y;
#endif // if 0
}

// Initialise the mesh describing variables
static inline void initialise_mesh(
    Mesh* mesh)
{
  mesh->edgedx = (double*)malloc(sizeof(double)*mesh->local_nx+1);
  mesh->celldx = (double*)malloc(sizeof(double)*mesh->local_nx);
  mesh->edgedy = (double*)malloc(sizeof(double)*mesh->local_ny+1);
  mesh->celldy = (double*)malloc(sizeof(double)*mesh->local_ny);
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
}

// Write out data for visualisation in visit
static inline void write_to_visit(
    const int nx, const int ny, const int x_off, const int y_off, 
    const double* data, const char* name, const int step, const double time)
{
  char bovname[256];
  char datname[256];
  sprintf(bovname, "%s%d.bov", name, step);
  sprintf(datname, "%s%d.dat", name, step);

  FILE* bovfp = fopen(bovname, "w");

  if(!bovfp) {
    printf("Could not open file %s\n", bovname);
    exit(1);
  }

  fprintf(bovfp, "TIME: %.4f\n", time);
  fprintf(bovfp, "DATA_FILE: %s\n", datname);
  fprintf(bovfp, "DATA_SIZE: %d %d 1\n", nx, ny);
  fprintf(bovfp, "DATA_FORMAT: DOUBLE\n");
  fprintf(bovfp, "VARIABLE: density");
  fprintf(bovfp, "DATA_ENDIAN: LITTLE\n");
  fprintf(bovfp, "CENTERING: zone\n");

#ifdef MPI
  fprintf(bovfp, "BRICK_ORIGIN: %f %f 0.\n", (float)x_off, (float)y_off);
#else
  fprintf(bovfp, "BRICK_ORIGIN: 0. 0. 0.\n");
#endif

  fprintf(bovfp, "BRICK_SIZE: %d %d 1\n", nx, ny);

  fclose(bovfp);

  FILE* datfp = fopen(datname, "wb");
  if(!datfp) {
    printf("Could not open file %s\n", datname);
    exit(1);
  }

  fwrite(data, nx*ny, sizeof(data), datfp);
  fclose(datfp);
}

// Prints some conservation values
static inline void print_conservation(
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
  MPI_Allreduce(&mass_tot, &global_mass_tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&energy_tot, &global_energy_tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  if(mesh->rank == MASTER) {
    printf("total mass: %.12e\n", global_mass_tot);
    printf("total energy: %.12e\n", global_energy_tot);
  }
}

// Deallocate all of the state memory
static inline void finalise_state(State* state)
{
  free(state->F_x);
  free(state->F_y);
  free(state->rho);
  free(state->rho_old);
  free(state->slope_x0);
  free(state->slope_y0);
  free(state->slope_x1);
  free(state->slope_y1);
  free(state->u);
  free(state->v);
  free(state->P);
  free(state->e);
}

// Deallocate all of the mesh memory
static inline void finalise_mesh(Mesh* mesh)
{
  free(mesh->edgedy);
  free(mesh->celldy);
  free(mesh->edgedx);
  free(mesh->celldx);
}


#if 0
/// TODO: What happens about the slopes defined on the halo regions, 
// is it OK that they are simply set to 0???
#pragma omp parallel for
for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
  for(int jj = PAD; jj < nx-PAD; ++jj) {
    // Calculate the maximum and minimum neighbouring density
    const double rho_x_max = max(rho[ind0-1], max(rho[ind0], rho[ind0+1]));
    const double rho_x_min = min(rho[ind0-1], min(rho[ind0], rho[ind0+1]));
    const double rho_y_max = max(rho[ind0-nx], max(rho[ind0], rho[ind0+nx]));
    const double rho_y_min = min(rho[ind0-nx], min(rho[ind0], rho[ind0+nx]));

    // Construct absolute value of slope S_L or S_R
    const double s1_x = 
      min(rho_x_max - rho[ind0], rho[ind0] - rho_x_min) / (celldx[jj] / 2.0);
    const double s1_y = 
      min(rho_y_max - rho[ind0], rho[ind0] - rho_y_min) / (celldy[ii] / 2.0);

    // Calculate the density interpolated from the zone center to zone boundary
    const double rho_edge_l = 
      (celldx[jj]*rho[ind0-1] + celldx[jj - 1]*rho[ind0])/(celldx[jj] + celldx[jj - 1]);
    const double rho_edge_r = 
      (celldx[jj + 1]*rho[ind0] + celldx[jj]*rho[ind0+1])/(celldx[jj + 1] + celldx[jj]);
    const double rho_edge_d = 
      (celldy[ii]*rho[ind0-nx] + celldy[ii - 1]*rho[ind0])/(celldy[ii] + celldy[ii - 1]);
    const double rho_edge_u = 
      (celldy[ii + 1]*rho[ind0] + celldy[ii]*rho[ind0+nx])/(celldy[ii + 1] + celldy[ii]);

    // Construct the slope
    const double s2_x = (rho_edge_r - rho_edge_l) / celldx[jj];
    const double s2_y = (rho_edge_u - rho_edge_d) / celldy[ii];

    // Define the zone centered slope (culling 0's)
    slope_x[ind0] = (s2_x != 0.0) ? (s2_x / fabs(s2_x))*min(fabs(s2_x), s1_x) : 0.0;
    slope_y[ind0] = (s2_y != 0.0) ? (s2_y / fabs(s2_y))*min(fabs(s2_y), s1_y) : 0.0;
  }
}

// Calculate the zone edge centered energies, and flux
#pragma omp parallel for
for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
  for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
    // Calculate the interpolated densities
    const double edge_rho_x = (u[ind1] >= 0.0)
      ? rho[ind0-1] + 0.5*slope_x[ind0-1]*(celldx[jj-1] - u[ind1]*dt_h)
      : rho[ind0] - 0.5*slope_x[ind0]*(celldx[jj] + u[ind1]*dt_h);
    F_x[ind1] = edge_rho_x*u[ind1]; 
  }
}

// Calculate the zone edge centered energies, and flux
#pragma omp parallel for
for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
  for(int jj = PAD; jj < nx-PAD; ++jj) {
    // Calculate the interpolated densities
    const double edge_rho_y = (v[ind0] >= 0.0)
      ? rho[ind0-nx] + 0.5*slope_y[ind0-nx]*(celldy[ii-1] - v[ind0]*dt_h)
      : rho[ind0] - 0.5*slope_y[ind0]*(celldy[ii] + v[ind0]*dt_h);
    F_y[ind0] = edge_rho_y*v[ind0]; 
  }
}
#endif // if 0

#if 0
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
#endif // if 0

#if 0
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
#endif // if 0

#if 0
#pragma omp parallel for
for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
  for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
    // Calculate the zone edge centered density
    const double rho_edge_x = 
      (rho[ind0]*celldx[jj]*celldy[ii] + rho[ind0-1]*celldx[jj - 1]*celldy[ii]) / 
      (2.0*edgedx[jj]*celldy[ii]);
    const double rho_edge_y = 
      (rho[ind0]*celldy[ii]*celldx[jj] + rho[ind0-nx]*celldy[ii - 1]*celldx[jj]) / 
      (2.0*edgedx[jj]*celldy[ii]);
    u[ind1] = (rho_edge_x == 0.0) ? 0.0 : rho_u[ind1] / rho_edge_x;
    v[ind0] = (rho_edge_y == 0.0) ? 0.0 : rho_v[ind0] / rho_edge_y;
  }
}

handle_boundary(nx+1, ny, neighbours, u, INVERT_X);
handle_boundary(nx, ny+1, neighbours, v, INVERT_Y);
#endif // if 0



#if 0
if(jj >= nx/2) { //RIGHT SOD
  state->rho[ind0] = 1.0;
  state->e[ind0] = 2.5;
}
#endif // if 0

#if 0
if(ii < ny/2) { //UP SOD
  state->rho[ind0] = 1.0;
  state->e[ind0] = 2.5;
}
#endif // if 0

#if 0
if(ii >= ny/2) { //DOWN SOD
  state->rho[ind0] = 1.0;
  state->e[ind0] = 2.5;
}
#endif // if 0

#if 0
// Batches the halos up into buffers, communicates and then unwraps them
static inline void communicate_halos(
    const int nx, const int ny, Mesh* mesh, double* rho, double* rho_u, 
    double* rho_v, double* e)
{
  int nmessages = 0;
  MPI_Request out_req[NNEIGHBOURS];
  MPI_Request in_req[NNEIGHBOURS];

  // TODO: Currently overestimating the amount of data to send
  const int message_length_x = (nx+1)*PAD*NVARS_TO_COMM; 
  const int message_length_y = (ny+1)*PAD*NVARS_TO_COMM;

  if(mesh->neighbours[SOUTH] != EDGE) {
#pragma omp parallel for collapse(2)
    for(int dd = 0; dd < PAD; ++dd) {
      for(int jj = 0; jj < nx; ++jj) {
        const int out_index = dd*nx+jj;
        mesh->south_buffer_out[(RHO_OFF*(nx+1)*PAD)+out_index]   = rho  [(PAD+dd)*nx+jj];
        mesh->south_buffer_out[(E_OFF*(nx+1)*PAD)+out_index]     = e    [(PAD+dd)*nx+jj];
        mesh->south_buffer_out[(RHO_V_OFF*(nx+1)*PAD)+out_index] = rho_v[(PAD+dd)*nx+jj];
      }
    }

#pragma omp parallel for collapse(2)
    for(int dd = 0; dd < PAD; ++dd) {
      for(int jj = 0; jj < (nx+1); ++jj) {
        const int out_index = dd*(nx+1)+jj;
        mesh->south_buffer_out[(RHO_U_OFF*(nx+1)*PAD)+out_index] = rho_u[(PAD+dd)*(nx+1)+jj];
      }
    }

    MPI_Isend(mesh->south_buffer_out, message_length_x, MPI_DOUBLE, 
        mesh->neighbours[SOUTH], 0, MPI_COMM_WORLD, &out_req[SOUTH]);
    MPI_Irecv(mesh->south_buffer_in, message_length_x, MPI_DOUBLE,
        mesh->neighbours[SOUTH], 1, MPI_COMM_WORLD, &in_req[nmessages++]);
  }

  if(mesh->neighbours[NORTH] != EDGE) {
#pragma omp parallel for collapse(2)
    for(int dd = 0; dd < PAD; ++dd) {
      for(int jj = 0; jj < nx; ++jj) {
        const int out_index = dd*nx+jj;
        mesh->north_buffer_out[(RHO_OFF*(nx+1)*PAD)+out_index]   = rho  [(ny-2*PAD+dd)*nx+jj];
        mesh->north_buffer_out[(E_OFF*(nx+1)*PAD)+out_index]     = e    [(ny-2*PAD+dd)*nx+jj];
        mesh->north_buffer_out[(RHO_V_OFF*(nx+1)*PAD)+out_index] = rho_v[((ny+1)-2*PAD+dd)*nx+jj];
      }
    }
#pragma omp parallel for collapse(2)
    for(int dd = 0; dd < PAD; ++dd) {
      for(int jj = 0; jj < (nx+1); ++jj) {
        const int out_index = dd*(nx+1)+jj;
        mesh->north_buffer_out[(RHO_U_OFF*(nx+1)*PAD)+out_index] = rho_u[(ny-2*PAD+dd)*(nx+1)+jj];
      }
    }

    MPI_Isend(mesh->north_buffer_out, message_length_x, MPI_DOUBLE, 
        mesh->neighbours[NORTH], 1, MPI_COMM_WORLD, &out_req[NORTH]);
    MPI_Irecv(mesh->north_buffer_in, message_length_x, MPI_DOUBLE,
        mesh->neighbours[NORTH], 0, MPI_COMM_WORLD, &in_req[nmessages++]);
  }

  if(mesh->neighbours[EAST] != EDGE) {
#pragma omp parallel for collapse(2)
    for(int ii = 0; ii < ny; ++ii) {
      for(int dd = 0; dd < PAD; ++dd) {
        const int out_index = ii*PAD+dd;
        mesh->east_buffer_out[out_index+(RHO_OFF*ny*PAD)]   = rho  [(ii*nx)+(nx-2*PAD+dd)];
        mesh->east_buffer_out[out_index+(E_OFF*ny*PAD)]     = e    [(ii*nx)+(nx-2*PAD+dd)];
        mesh->east_buffer_out[out_index+(RHO_U_OFF*ny*PAD)] = rho_u[(ii*(nx+1))+((nx+1)-2*PAD+dd)];
      }
    }
    for(int ii = 0; ii < (ny+1); ++ii) {
      for(int dd = 0; dd < PAD; ++dd) {
        const int out_index = ii*PAD+dd;
        mesh->east_buffer_out[out_index+(RHO_V_OFF*ny*PAD)] = rho_v[(ii*nx)+(nx-2*PAD+dd)];
      }
    }

    MPI_Isend(mesh->east_buffer_out, message_length_y, MPI_DOUBLE, 
        mesh->neighbours[EAST], 2, MPI_COMM_WORLD, &out_req[EAST]);
    MPI_Irecv(mesh->east_buffer_in, message_length_y, MPI_DOUBLE,
        mesh->neighbours[EAST], 3, MPI_COMM_WORLD, &in_req[nmessages++]);
  }

  if(mesh->neighbours[WEST] != EDGE) {
    for(int ii = 0; ii < ny; ++ii) {
      for(int dd = 0; dd < PAD; ++dd) {
        const int out_index = ii*PAD+dd;
        mesh->west_buffer_out[out_index+(RHO_OFF*ny*PAD)]   = rho  [(ii*nx)+(PAD+dd)];
        mesh->west_buffer_out[out_index+(E_OFF*ny*PAD)]     = e    [(ii*nx)+(PAD+dd)];
        mesh->west_buffer_out[out_index+(RHO_U_OFF*ny*PAD)] = rho_u[(ii*(nx+1))+(PAD+dd)];
      }
    }
    for(int ii = 0; ii < (ny+1); ++ii) {
      for(int dd = 0; dd < PAD; ++dd) {
        const int out_index = ii*PAD+dd;
        mesh->west_buffer_out[out_index+(RHO_V_OFF*ny*PAD)] = rho_v[(ii*nx)+(PAD+dd)];
      }
    }

    MPI_Isend(mesh->west_buffer_out, message_length_y, MPI_DOUBLE,
        mesh->neighbours[WEST], 3, MPI_COMM_WORLD, &out_req[WEST]);
    MPI_Irecv(mesh->west_buffer_in, message_length_y, MPI_DOUBLE, 
        mesh->neighbours[WEST], 2, MPI_COMM_WORLD, &in_req[nmessages++]);
  }

  MPI_Waitall(nmessages, in_req, MPI_STATUSES_IGNORE);

  if(mesh->neighbours[NORTH] != EDGE) {
#pragma omp parallel for collapse(2)
    for(int dd = 0; dd < PAD; ++dd) {
      for(int jj = 0; jj < nx; ++jj) {
        const int in_index = dd*nx+jj;
        rho  [(ny-PAD+dd)*nx+jj] = mesh->north_buffer_in[(RHO_OFF*(nx+1)*PAD)+in_index];
        e    [(ny-PAD+dd)*nx+jj] = mesh->north_buffer_in[(E_OFF*(nx+1)*PAD)+in_index];
        rho_v[((ny+1)-PAD+dd)*nx+jj] = mesh->north_buffer_in[(RHO_V_OFF*(nx+1)*PAD)+in_index];
      }
    }
#pragma omp parallel for collapse(2)
    for(int dd = 0; dd < PAD; ++dd) {
      for(int jj = 0; jj < (nx+1); ++jj) {
        const int in_index = dd*(nx+1)+jj;
        rho_u[(ny-PAD+dd)*(nx+1)+jj] = mesh->north_buffer_in[(RHO_U_OFF*(nx+1)*PAD)+in_index];
      }
    }
  }

  if(mesh->neighbours[SOUTH] != EDGE) {
#pragma omp parallel for collapse(2)
    for(int dd = 0; dd < PAD; ++dd) {
      for(int jj = 0; jj < nx; ++jj) {
        const int in_index = dd*nx+jj;
        rho  [dd*nx + jj] = mesh->south_buffer_in[(RHO_OFF*(nx+1)*PAD)+in_index];
        e    [dd*nx + jj] = mesh->south_buffer_in[(E_OFF*(nx+1)*PAD)+in_index];
        rho_v[dd*nx + jj] = mesh->south_buffer_in[(RHO_V_OFF*(nx+1)*PAD)+in_index];
      }
    }

#pragma omp parallel for collapse(2)
    for(int dd = 0; dd < PAD; ++dd) {
      for(int jj = 0; jj < (nx+1); ++jj) {
        const int in_index = dd*(nx+1)+jj;
        rho_u[dd*(nx+1) + jj] = mesh->south_buffer_in[(RHO_U_OFF*(nx+1)*PAD)+in_index];
      }
    }
  }

#if 0
  if(mesh->neighbours[WEST] != EDGE) {
#pragma omp parallel for collapse(2)
    for(int ii = 0; ii < ny; ++ii) {
      for(int dd = 0; dd < PAD; ++dd) {
        const int in_index = ii*PAD+dd;
        rho  [ii*nx + dd] = mesh->west_buffer_in[(RHO_OFF*ny*PAD) + in_index];
        e    [ii*nx + dd] = mesh->west_buffer_in[(E_OFF*ny*PAD) + in_index];
        rho_u[ii*(nx+1) + dd] = mesh->west_buffer_in[(RHO_U_OFF*ny*PAD) + in_index];
      }
    }

#pragma omp parallel for collapse(2)
    for(int ii = 0; ii < (ny+1); ++ii) {
      for(int dd = 0; dd < PAD; ++dd) {
        const int in_index = ii*PAD+dd;
        rho_v[ii*nx + dd] = mesh->west_buffer_in[(RHO_V_OFF*ny*PAD) + in_index];
      }
    }
  }

  if(mesh->neighbours[EAST] != EDGE) {
#pragma omp parallel for collapse(2)
    for(int ii = 0; ii < ny; ++ii) {
      for(int dd = 0; dd < PAD; ++dd) {
        const int in_index = ii*PAD+dd;
        rho  [ii*nx + (nx-PAD+dd)] = mesh->east_buffer_in[(RHO_OFF*ny*PAD)+in_index];
        e    [ii*nx + (nx-PAD+dd)] = mesh->east_buffer_in[(E_OFF*ny*PAD)+in_index];
        rho_u[ii*(nx+1) + ((nx+1)-PAD+dd)] = mesh->east_buffer_in[(RHO_U_OFF*ny*PAD)+in_index];
      }
    }

#pragma omp parallel for collapse(2)
    for(int ii = 0; ii < (ny+1); ++ii) {
      for(int dd = 0; dd < PAD; ++dd) {
        const int in_index = ii*PAD+dd;
        rho_v[ii*nx + (nx-PAD+dd)] = mesh->east_buffer_in[(RHO_V_OFF*ny*PAD)+in_index];
      }
    }
  }
#endif // if 0
}

#endif // if 0
