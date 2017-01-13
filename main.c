#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "wet_interface.h"
#include "wet_data.h"
#include "../mesh.h"
#include "../shared_data.h"
#include "../comms.h"

int main(int argc, char** argv)
{
  // Store the dimensions of the mesh
  Mesh mesh = {0};
  mesh.global_nx = atoi(argv[1]);
  mesh.global_ny = atoi(argv[2]);
  mesh.local_nx = atoi(argv[1]) + 2*PAD;
  mesh.local_ny = atoi(argv[2]) + 2*PAD;
  mesh.rank = MASTER;
  mesh.nranks = 1;
  mesh.niters = atoi(argv[3]);

  initialise_mpi(argc, argv, &mesh.rank, &mesh.nranks);

  if(argc != 4 && mesh.rank == MASTER) {
    printf("usage: ./wet.exe <local_nx> <local_y> <niters>\n");
    exit(1);
  }

  initialise_comms(&mesh);
  initialise_devices(mesh.rank);
  initialise_mesh_2d(&mesh);

  SharedData shared_data = {0};
  initialise_shared_data_2d(
      mesh.global_nx, mesh.global_ny, mesh.local_nx, mesh.local_ny, 
      mesh.x_off, mesh.y_off, &shared_data);

  WetData wet_data = {0};
  initialise_wet_data_2d(mesh.local_nx, mesh.local_ny, &wet_data);

  set_timestep(
      mesh.local_nx, mesh.local_ny, shared_data.Qxx, shared_data.Qyy, 
      shared_data.rho, shared_data.e, &mesh, shared_data.reduce_array, 
      0, mesh.celldx, mesh.celldy);

#if 0
  write_all_ranks_to_visit(
      mesh.global_nx+2*PAD, mesh.global_ny+2*PAD, mesh.local_nx, mesh.local_ny, 
      mesh.x_off, mesh.y_off, mesh.rank, mesh.nranks, mesh.neighbours, 
      shared_data.rho, "initial_density", 0, 0.0);
#endif // if 0

  // Prepare for solve
  struct Profile wallclock = {0};
  double elapsed_sim_time = 0.0;

  START_PROFILING(&wallclock);

  // Main timestep loop
  int tt;
  for(tt = 0; tt < mesh.niters; ++tt) {

    if(mesh.rank == MASTER) 
      printf("Iteration %d\n", tt+1);

    solve_hydro_2d(
        &mesh, tt, shared_data.P, shared_data.rho, shared_data.rho_old, 
        shared_data.e, shared_data.u, shared_data.v, wet_data.rho_u, 
        wet_data.rho_v, shared_data.Qxx, shared_data.Qyy, wet_data.F_x, 
        wet_data.F_y, wet_data.uF_x, wet_data.uF_y, wet_data.vF_x, 
        wet_data.vF_y, shared_data.reduce_array);

    print_conservation(
        mesh.local_nx, mesh.local_ny, shared_data.rho, shared_data.e, 
        shared_data.reduce_array, &mesh);

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

  barrier();

  STOP_PROFILING(&wallclock, "wallclock");

  if(mesh.rank == MASTER) {
    struct ProfileEntry pe = profiler_get_profile_entry(&wallclock, "wallclock");
    PRINT_PROFILING_RESULTS(&compute_profile);
    PRINT_PROFILING_RESULTS(&comms_profile);
    printf("Wallclock %.4fs, Elapsed Simulation Time %.4fs\n", 
        pe.time, elapsed_sim_time);
  }

  write_all_ranks_to_visit(
      mesh.global_nx+2*PAD, mesh.global_ny+2*PAD, mesh.local_nx, mesh.local_ny, 
      mesh.x_off, mesh.y_off, mesh.rank, mesh.nranks, mesh.neighbours, 
      shared_data.rho, "density", 0, elapsed_sim_time);

  finalise_shared_data(&shared_data);
  finalise_mesh(&mesh);

  return 0;
}

