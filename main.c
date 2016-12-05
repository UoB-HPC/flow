#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "wet.h"
#include "../mesh.h"
#include "../state.h"
#include "../comms.h"

int main(int argc, char** argv)
{
  if(argc != 4) {
    printf("usage: ./wet.exe <local_nx> <local_y> <niters>\n");
    exit(1);
  }

  // Store the dimensions of the mesh
  Mesh mesh = {0};
  mesh.global_nx = atoi(argv[1]);
  mesh.global_ny = atoi(argv[2]);
  mesh.local_nx = atoi(argv[1]) + 2*PAD;
  mesh.local_ny = atoi(argv[2]) + 2*PAD;
  mesh.rank = MASTER;
  mesh.nranks = 1;
  mesh.niters = atoi(argv[3]);

  init_mpi(argc, argv, &mesh.rank, &mesh.nranks);
  initialise_comms(&mesh);
  initialise_mesh(&mesh);

  State state = {0};
  initialise_state(
      mesh.global_nx, mesh.global_ny, mesh.local_nx, mesh.local_ny, 
      mesh.x_off, mesh.y_off, &state);

  set_timestep(
      mesh.local_nx, mesh.local_ny, state.Qxx, state.Qyy, state.rho, 
      state.e, &mesh, 0, mesh.celldx, mesh.celldy);

#if 0
  write_all_ranks_to_visit(
      mesh.global_nx+2*PAD, mesh.global_ny+2*PAD, mesh.local_nx, mesh.local_ny, mesh.x_off, 
      mesh.y_off, mesh.rank, mesh.nranks, mesh.neighbours, state.rho, "initial_density", 0, 0.0);
#endif // if 0

  // Prepare for solve
  struct Profile wallclock = {0};
  double elapsed_sim_time = 0.0;

  // Main timestep loop
  int tt;
  for(tt = 0; tt < mesh.niters; ++tt) {

    if(mesh.rank == MASTER) 
      printf("Iteration %d\n", tt+1);

    START_PROFILING(&wallclock);

    solve_hydro(
        &mesh, tt, state.P, state.rho, state.rho_old, state.e, state.u, 
        state.v, state.rho_u, state.rho_v, state.Qxx, state.Qyy, state.F_x, 
        state.F_y, state.uF_x, state.uF_y, state.vF_x, state.vF_y);

    STOP_PROFILING(&wallclock, "wallclock");

    print_conservation(mesh.local_nx, mesh.local_ny, state.rho, state.e, &mesh);

    if(mesh.rank == MASTER) {
      printf("simulation time: %.4lf(s)\n", elapsed_sim_time);
    }

    elapsed_sim_time += mesh.dt;
    if(elapsed_sim_time >= SIM_END) {
      if(mesh.rank == MASTER)
        printf("reached end of simulation time\n");
      break;
    }
  }

  double global_wallclock = 0.0;
  if(tt > 0) {
#ifdef MPI
    struct ProfileEntry pe = profiler_get_profile_entry(&wallclock, "wallclock");
    MPI_Reduce(&pe.time, &global_wallclock, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
#endif
  }

  if(mesh.rank == MASTER) {
    PRINT_PROFILING_RESULTS(&compute_profile);
    PRINT_PROFILING_RESULTS(&comms_profile);
    printf("Wallclock %.2fs, Elapsed Simulation Time %.4fs\n", global_wallclock, elapsed_sim_time);
  }

  fetch_data(mesh.local_nx, mesh.local_ny, state.rho);
  write_all_ranks_to_visit(
      mesh.global_nx+2*PAD, mesh.global_ny+2*PAD, mesh.local_nx, mesh.local_ny, mesh.x_off, 
      mesh.y_off, mesh.rank, mesh.nranks, mesh.neighbours, state.rho, "density", 0, elapsed_sim_time);

  finalise_state(&state);
  finalise_mesh(&mesh);

  return 0;
}

