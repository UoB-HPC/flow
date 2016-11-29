#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "wet.h"

extern struct Profile compute_profile;
extern struct Profile comms_profile;

int main(int argc, char** argv)
{
  if(argc != 4) {
    printf("usage: ./wet.exe <local_nx> <local_y> <niters>\n");
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
  struct Profile wallclock = {0};
  double elapsed_sim_time = 0.0;

  // Main timestep loop
  int tt;
  for(tt = 0; tt < mesh.niters; ++tt) {

    if(mesh.rank == MASTER) 
      printf("Iteration %d\n", tt+1);

    START_PROFILING(&wallclock);

    solve(
        mesh, tt == 0, state.P, state.rho, state.rho_old, state.e, state.u, 
        state.v, state.rho_u, state.rho_v, state.Qxx, state.Qyy, state.F_x, 
        state.F_y, state.uF_x, state.uF_y, state.vF_x, state.vF_y);

    STOP_PROFILING(&wallclock, "wallclock");

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
    struct ProfileEntry pe = profiler_get_profile_entry(&wallclock, "wallclock");
    MPI_Reduce(&pe.time, &global_wallclock, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
#endif
  }

  if(mesh.rank == MASTER) {
    PRINT_PROFILING_RESULTS(&compute_profile);
    PRINT_PROFILING_RESULTS(&comms_profile);
    printf("Wallclock %.2fs, Elapsed Simulation Time %.4fs\n", global_wallclock, elapsed_sim_time);
  }

  write_all_ranks_to_visit(
      mesh.global_nx+2*PAD, mesh.global_ny+2*PAD, mesh.local_nx, mesh.local_ny, mesh.x_off, 
      mesh.y_off, mesh.rank, mesh.nranks, state.rho, "density", 0, elapsed_sim_time);
  write_all_ranks_to_visit(
      mesh.global_nx+2*PAD, mesh.global_ny+2*PAD, mesh.local_nx, mesh.local_ny, mesh.x_off, 
      mesh.y_off, mesh.rank, mesh.nranks, state.e, "energy", 0, elapsed_sim_time);
  write_all_ranks_to_visit(
      mesh.global_nx+1+2*PAD, mesh.global_ny+2*PAD, mesh.local_nx+1, mesh.local_ny, mesh.x_off, 
      mesh.y_off, mesh.rank, mesh.nranks, state.u, "u", 0, elapsed_sim_time);
  write_all_ranks_to_visit(
      mesh.global_nx+2*PAD, mesh.global_ny+1+2*PAD, mesh.local_nx, mesh.local_ny+1, mesh.x_off, 
      mesh.y_off, mesh.rank, mesh.nranks, state.v, "v", 0, elapsed_sim_time);

  finalise_state(&state);
  finalise_mesh(&mesh);

  return 0;
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
#if 0
      // OFF CENTER SQUARE TEST
      const int dist = 100;
      if(jj+mesh->x_off-PAD >= mesh->global_nx/4-dist && 
          jj+mesh->x_off-PAD < mesh->global_nx/4+dist && 
          ii+mesh->y_off-PAD >= mesh->global_ny/2-dist && 
          ii+mesh->y_off-PAD < mesh->global_ny/2+dist) {
        state->rho[ii*mesh->local_nx+jj] = 1.0;
        state->e[ii*mesh->local_nx+jj] = 2.5;
      }
#endif // if 0
      if(jj+mesh->x_off < (mesh->global_nx/2+2*PAD)) {
        state->rho[ii*mesh->local_nx+jj] = 1.0;
        state->e[ii*mesh->local_nx+jj] = 2.5;
      }
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

