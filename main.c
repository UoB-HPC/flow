#include "../comms.h"
#include "../mesh.h"
#include "../params.h"
#include "../shared_data.h"
#include "flow_data.h"
#include "flow_interface.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  if (argc != 2) {
    TERMINATE("usage: ./flow <parameter_filename>\n");
  }

  // Store the dimensions of the mesh
  Mesh mesh;
  const char* flow_params = argv[1];
  mesh.global_nx = get_int_parameter("nx", flow_params);
  mesh.global_ny = get_int_parameter("ny", flow_params);
  mesh.niters = get_int_parameter("iterations", flow_params);
  mesh.pad = 2;
  mesh.local_nx = mesh.global_nx + 2 * mesh.pad;
  mesh.local_ny = mesh.global_ny + 2 * mesh.pad;
  mesh.width = get_double_parameter("width", ARCH_ROOT_PARAMS);
  mesh.height = get_double_parameter("height", ARCH_ROOT_PARAMS);
  mesh.max_dt = get_double_parameter("max_dt", ARCH_ROOT_PARAMS);
  mesh.sim_end = get_double_parameter("sim_end", ARCH_ROOT_PARAMS);
  mesh.dt = C_T * get_double_parameter("dt", flow_params);
  mesh.dt_h = mesh.dt;
  mesh.rank = MASTER;
  mesh.nranks = 1;

  const int visit_dump = get_int_parameter("visit_dump", flow_params);

  initialise_mpi(argc, argv, &mesh.rank, &mesh.nranks);
  initialise_comms(&mesh);
  initialise_devices(mesh.rank);
  initialise_mesh_2d(&mesh);

  int nthreads = 0;
#pragma omp parallel
  { nthreads = omp_get_num_threads(); }

  if (mesh.rank == MASTER) {
    printf("Number of ranks: %d\n", mesh.nranks);
    printf("Number of threads: %d\n", nthreads);
  }

  SharedData shared_data;
  initialise_shared_data_2d(mesh.local_nx, mesh.local_ny, mesh.pad, mesh.width,
                            mesh.height, flow_params, mesh.edgex, mesh.edgey,
                            &shared_data);

  handle_boundary_2d(mesh.local_nx, mesh.local_ny, &mesh, shared_data.density,
                     NO_INVERT, PACK);
  handle_boundary_2d(mesh.local_nx, mesh.local_ny, &mesh, shared_data.energy,
                     NO_INVERT, PACK);

  FlowData flow_data;
  initialise_flow_data_2d(mesh.local_nx, mesh.local_ny, &flow_data);

  set_timestep(mesh.local_nx, mesh.local_ny, shared_data.Qxx, shared_data.Qyy,
               shared_data.density, shared_data.energy, &mesh,
               shared_data.reduce_array0, 0, mesh.celldx, mesh.celldy);

  // Prepare for solve
  double wallclock = 0.0;
  double elapsed_sim_time = 0.0;

  // Main timestep loop
  int tt;
  for (tt = 0; tt < mesh.niters; ++tt) {

    if (mesh.rank == MASTER) {
      printf("\nIteration %d\n", tt + 1);
    }

    double w0 = omp_get_wtime();

    solve_hydro_2d(&mesh, tt, shared_data.pressure, shared_data.density,
                   shared_data.density_old, shared_data.energy, shared_data.u,
                   shared_data.v, flow_data.momentum_x, flow_data.momentum_y,
                   shared_data.Qxx, shared_data.Qyy, flow_data.mass_flux_x,
                   flow_data.mass_flux_y, flow_data.momentum_x_flux_x,
                   flow_data.momentum_x_flux_y, flow_data.momentum_y_flux_x,
                   flow_data.momentum_y_flux_y, shared_data.reduce_array0);

    print_conservation(mesh.local_nx, mesh.local_ny, shared_data.density,
                       shared_data.energy, shared_data.reduce_array0, &mesh);

    wallclock += omp_get_wtime() - w0;

    elapsed_sim_time += mesh.dt;
    if (elapsed_sim_time >= mesh.sim_end) {
      if (mesh.rank == MASTER) {
        printf("reached end of simulation time\n");
      }
      break;
    }

    if (mesh.rank == MASTER) {
      printf("Simulation time: %.4lfs\nWallclock:       %.4lfs\n",
             elapsed_sim_time, wallclock);
    }

    if (visit_dump) {
      write_all_ranks_to_visit(
          mesh.global_nx + 2 * mesh.pad, mesh.global_ny + 2 * mesh.pad,
          mesh.local_nx, mesh.local_ny, mesh.pad, mesh.x_off, mesh.y_off,
          mesh.rank, mesh.nranks, mesh.neighbours, shared_data.density,
          "density", tt, elapsed_sim_time);
    }
  }

  barrier();

  validate(mesh.local_nx, mesh.local_ny, mesh.pad, flow_params, mesh.rank,
           shared_data.density, shared_data.energy);

  if (mesh.rank == MASTER) {
    PRINT_PROFILING_RESULTS(&compute_profile);
    PRINT_PROFILING_RESULTS(&comms_profile);
    printf("Wallclock %.4fs, Elapsed Simulation Time %.4fs\n", wallclock,
           elapsed_sim_time);
  }

  if (visit_dump) {
    write_all_ranks_to_visit(
        mesh.global_nx + 2 * mesh.pad, mesh.global_ny + 2 * mesh.pad,
        mesh.local_nx, mesh.local_ny, mesh.pad, mesh.x_off, mesh.y_off,
        mesh.rank, mesh.nranks, mesh.neighbours, shared_data.density, "density",
        0, elapsed_sim_time);
  }

  finalise_shared_data(&shared_data);
  finalise_mesh(&mesh);

  return 0;
}

// Validates the results of the simulation
void validate(const int nx, const int ny, const int pad,
              const char* params_filename, const int rank, double* density,
              double* energy) {
  double* h_energy;
  double* h_density;
  allocate_host_data(&h_energy, nx * ny);
  allocate_host_data(&h_density, nx * ny);
  copy_buffer(nx * ny, &energy, &h_energy, RECV);
  copy_buffer(nx * ny, &density, &h_density, RECV);

  double local_density_total = 0.0;
  double local_energy_total = 0.0;

#pragma omp parallel for reduction(+ : local_density_total, local_energy_total)
  for (int ii = pad; ii < ny - pad; ++ii) {
    for (int jj = pad; jj < nx - pad; ++jj) {
      const int index = (ii * nx) + (jj);
      local_density_total += h_density[(index)];
      local_energy_total += h_energy[(index)];
    }
  }

  double global_density_total = reduce_all_sum(local_density_total);
  double global_energy_total = reduce_all_sum(local_energy_total);

  if (rank != MASTER) {
    return;
  }

  int nresults = 0;
  char* keys = (char*)malloc(sizeof(char) * MAX_KEYS * (MAX_STR_LEN + 1));
  double* values = (double*)malloc(sizeof(double) * MAX_KEYS);
  if (!get_key_value_parameter(params_filename, FLOW_TESTS, keys, values,
                               &nresults)) {
    printf("Warning. Test entry was not found, could NOT validate.\n");
    return;
  }

  double expected_energy;
  double expected_density;
  if (strmatch(&(keys[0]), "energy")) {
    expected_energy = values[0];
    expected_density = values[1];
  } else {
    expected_energy = values[1];
    expected_density = values[0];
  }

  printf("\nExpected energy  %.12e, result was %.12e.\n", expected_energy,
         global_energy_total);
  printf("Expected density %.12e, result was %.12e.\n", expected_density,
         global_density_total);

  const int pass = within_tolerance(expected_energy, global_energy_total,
                                    VALIDATE_TOLERANCE) &&
                   within_tolerance(expected_density, global_density_total,
                                    VALIDATE_TOLERANCE);

  if (pass) {
    printf("PASSED validation.\n");
  } else {
    printf("FAILED validation.\n");
  }

  free(keys);
  free(values);
  deallocate_host_data(h_energy);
  deallocate_host_data(h_density);
}
