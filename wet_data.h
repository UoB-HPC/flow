#pragma once

typedef struct {
  // Wet-specific state
  double* rho_u;    // Momentum in the x direction
  double* rho_v;    // Momentum in the y direction

  double* F_x;      // Mass flux in the x direction
  double* F_y;      // Mass flux in the y direction

  double* uF_x;     // Momentum in the x direction flux in the x direction 
  double* uF_y;     // Momentum in the x direction flux in the y direction

  double* vF_x;     // Momentum in the y direction flux in the x direction
  double* vF_y;     // Momentum in the y direction flux in the y direction

  double* wF_x;     // Momentum in the z direction flux in the x direction
  double* wF_y;     // Momentum in the z direction flux in the y direction
} WetData;

// Initialises the state variables for two dimensional applications
void initialise_wet_data_2d(
    const int local_nx, const int local_ny, WetData* wet_data);
void deallocate_wet_data_2d(
    WetData* wet_data);

#if 0
// TODO: Make Wet 3d
double* rho_u;    // Momentum in the x direction
double* rho_v;    // Momentum in the y direction
double* rho_w;    // Momentum in the z direction

double* F_x;      // Mass flux in the x direction
double* F_y;      // Mass flux in the y direction
double* F_z;      // Mass flux in the z direction

double* uF_x;     // Momentum in the x direction flux in the x direction 
double* uF_y;     // Momentum in the x direction flux in the y direction
double* uF_z;     // Momentum in the x direction flux in the z direction

double* vF_x;     // Momentum in the y direction flux in the x direction
double* vF_y;     // Momentum in the y direction flux in the y direction
double* vF_z;     // Momentum in the y direction flux in the z direction

double* wF_x;     // Momentum in the z direction flux in the x direction
double* wF_y;     // Momentum in the z direction flux in the y direction
double* wF_z;     // Momentum in the z direction flux in the z direction

// Wet-specific state
allocate_data(&state->rho_u, (local_nx+1)*(local_ny+1)*(local_nz+1));
allocate_data(&state->rho_v, (local_nx+1)*(local_ny+1)*(local_nz+1));
allocate_data(&state->rho_w, (local_nx+1)*(local_ny+1)*(local_nz+1));
allocate_data(&state->F_x, (local_nx+1)*(local_ny+1)*(local_nz+1));
allocate_data(&state->F_y, (local_nx+1)*(local_ny+1)*(local_nz+1));
allocate_data(&state->F_z, (local_nx+1)*(local_ny+1)*(local_nz+1));
allocate_data(&state->uF_x, (local_nx+1)*(local_ny+1)*(local_nz+1));
allocate_data(&state->uF_y, (local_nx+1)*(local_ny+1)*(local_nz+1));
allocate_data(&state->uF_z, (local_nx+1)*(local_ny+1)*(local_nz+1));
allocate_data(&state->vF_x, (local_nx+1)*(local_ny+1)*(local_nz+1));
allocate_data(&state->vF_y, (local_nx+1)*(local_ny+1)*(local_nz+1));
allocate_data(&state->vF_z, (local_nx+1)*(local_ny+1)*(local_nz+1));
allocate_data(&state->wF_x, (local_nx+1)*(local_ny+1)*(local_nz+1));
allocate_data(&state->wF_y, (local_nx+1)*(local_ny+1)*(local_nz+1));
allocate_data(&state->wF_z, (local_nx+1)*(local_ny+1)*(local_nz+1));
allocate_data(&state->reduce_array, (local_nx+1)*(local_ny+1)*(local_nz+1));

#endif // if 0

