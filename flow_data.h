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
  double* momentum_x;        // Momentum in the x dimension
  double* momentum_y;        // Momentum in the y dimension
  double* mass_flux_x;       // Mass flux in the x dimension
  double* mass_flux_y;       // Mass flux in the y dimension
  double* momentum_x_flux_x; // x momentum flux in the x dimension
  double* momentum_x_flux_y; // x momentum flux in the y dimension
  double* momentum_y_flux_x; // y momentum flux in the x dimension
  double* momentum_y_flux_y; // y momentum flux in the y dimension
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
