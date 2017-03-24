#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "wet_interface.h"
#include "wet_data.h"
#include "../mesh.h"
#include "../shared_data.h"
#include "../comms.h"
#include "../params.h"

int main(int argc, char** argv)
{
  if(argc != 2) {
    TERMINATE("usage: ./wet <parameter_filename>\n");
  }

  // Store the dimensions of the mesh
  Mesh mesh = {0};
  const char* wet_params = argv[1];
  mesh.global_nx = get_int_parameter("global_nx", wet_params);
  mesh.global_ny = get_int_parameter("global_ny", wet_params);
  mesh.niters = get_int_parameter("niters", wet_params);
  mesh.local_nx = mesh.global_nx+2*PAD;
  mesh.local_ny = mesh.global_ny+2*PAD;
  mesh.width = get_double_parameter("width", ARCH_ROOT_PARAMS);
  mesh.height = get_double_parameter("height", ARCH_ROOT_PARAMS);
  mesh.max_dt = get_double_parameter("max_dt", ARCH_ROOT_PARAMS);
  mesh.sim_end = get_double_parameter("sim_end", ARCH_ROOT_PARAMS);
  mesh.dt = C_T*mesh.max_dt;
  mesh.dt_h = C_T*mesh.max_dt;
  mesh.rank = MASTER;
  mesh.nranks = 1;

  initialise_mpi(argc, argv, &mesh.rank, &mesh.nranks);
  initialise_comms(&mesh);
  initialise_devices(mesh.rank);
  initialise_mesh_2d(&mesh);

  SharedData shared_data = {0};
  initialise_shared_data_2d(
      mesh.global_nx, mesh.global_ny, mesh.local_nx, mesh.local_ny, 
      mesh.x_off, mesh.y_off, mesh.width, mesh.height,
      wet_params, mesh.edgex, mesh.edgey, &shared_data);

  WetData wet_data = {0};
  initialise_wet_data_2d(mesh.local_nx, mesh.local_ny, &wet_data);

  set_timestep(
      mesh.local_nx, mesh.local_ny, shared_data.Qxx, shared_data.Qyy, 
      shared_data.rho, shared_data.e, &mesh, shared_data.reduce_array0, 
      0, mesh.celldx, mesh.celldy);

  // Prepare for solve
  double wallclock = 0.0;
  double elapsed_sim_time = 0.0;

  // Main timestep loop
  int tt;
  for(tt = 0; tt < mesh.niters; ++tt) {

    if(mesh.rank == MASTER) {
      printf("Iteration %d\n", tt+1);
    }

    double w0 = omp_get_wtime();

    solve_hydro_2d(
        &mesh, tt, shared_data.P, shared_data.rho, shared_data.rho_old, 
        shared_data.e, shared_data.u, shared_data.v, wet_data.rho_u, 
        wet_data.rho_v, shared_data.Qxx, shared_data.Qyy, wet_data.F_x, 
        wet_data.F_y, wet_data.uF_x, wet_data.uF_y, wet_data.vF_x, 
        wet_data.vF_y, shared_data.reduce_array0);

    print_conservation(
        mesh.local_nx, mesh.local_ny, shared_data.rho, shared_data.e, 
        shared_data.reduce_array0, &mesh);

    if(mesh.rank == MASTER) {
      printf("simulation time: %.4lf(s)\n", elapsed_sim_time);
    }

    wallclock += omp_get_wtime()-w0;
    elapsed_sim_time += mesh.dt;
    if(elapsed_sim_time >= mesh.sim_end) {
      if(mesh.rank == MASTER)
        printf("reached end of simulation time\n");
      break;
    }
  }

  barrier();

  if(mesh.rank == MASTER) {
    PRINT_PROFILING_RESULTS(&compute_profile);
    PRINT_PROFILING_RESULTS(&comms_profile);
    printf("Wallclock %.4fs, Elapsed Simulation Time %.4fs\n", 
        wallclock, elapsed_sim_time);
  }

  write_all_ranks_to_visit(
      mesh.global_nx+2*PAD, mesh.global_ny+2*PAD, mesh.local_nx, mesh.local_ny, 
      mesh.x_off, mesh.y_off, mesh.rank, mesh.nranks, mesh.neighbours, 
      shared_data.rho, "density", 0, elapsed_sim_time);

  finalise_shared_data(&shared_data);
  finalise_mesh(&mesh);

  return 0;
}

