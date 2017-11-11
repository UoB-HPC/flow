#ifndef __FLOWHDR
#define __FLOWHDR

#pragma once

#define C_T 0.3 // Safety factor for setting timestep
#define VALIDATE_TOLERANCE 1.0e-5
#define ARCH_ROOT_PARAMS "../arch.params"
#define FLOW_PARAMS "flow.params"
#define FLOW_TESTS "flow.tests"

typedef struct {
  // Flow-specific state
  double* momentum_x; // Momentum in the x direction
  double* momentum_y; // Momentum in the y direction

  double* mass_flux_x; // Mass flux in the x direction
  double* mass_flux_y; // Mass flux in the y direction

  double* uF_x; // Momentum in the x direction flux in the x direction
  double* uF_y; // Momentum in the x direction flux in the y direction

  double* vF_x; // Momentum in the y direction flux in the x direction
  double* vF_y; // Momentum in the y direction flux in the y direction

  double* wF_x; // Momentum in the z direction flux in the x direction
  double* wF_y; // Momentum in the z direction flux in the y direction
} FlowData;

// Initialises the state variables for two dimensional applications
void initialise_flow_data_2d(const int local_nx, const int local_ny,
                             FlowData* flow_data);
void deallocate_flow_data_2d(FlowData* flow_data);

// Validates the results of the simulation
void validate(const int nx, const int ny, const int pad,
              const char* params_filename, const int rank, double* density,
              double* energy);

#endif
