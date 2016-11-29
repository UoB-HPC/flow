#include "../shared.h"
#include "../mesh.h"

// Controllable parameters for the application
#define GAM 1.4
#define C_Q 2.5 // Suggested between 1.0 and 2.0
#define C_T 0.5
#define C_M (1.5/C_T)
#define SIM_END 2.5

enum { NO_INVERT, INVERT_X, INVERT_Y };

/// Problem state
typedef struct
{
  // Density (cell centered)
  double* rho;
  double* rho_old;

  // Pressure and internal energy, (cell centered)
  double* P;
  double* e;

  // Momenta (edge centered)
  double* rho_u;
  double* rho_v;

  // Fluid velocities (edge centered)
  double* v;
  double* u;

  // Mass fluxes, stored to avoid repetetive recomputation
  double* F_x;
  double* F_y;
  double* uF_x;
  double* uF_y;
  double* vF_x;
  double* vF_y;

  // Interpolated velocity values
  double* Qxx;
  double* Qyy;
} State;

// Constitutes an individual solve of a mesh
void solve_hydro(
    Mesh* mesh, int first_step, double* P, double* rho, double* rho_old, 
    double* e, double* u, double*v , double* rho_u, double* rho_v, 
    double* Qxx, double* Qyy, double* F_x, double* F_y, double* uF_x, 
    double* uF_y, double* vF_x, double* vF_y);

void equation_of_state(
    const int nx, const int ny, double* P, const double* rho, const double* e);

void set_timestep(
    const int nx, const int ny, double* Qxx, double* Qyy, const double* rho, 
    const double* u, const double* v, const double* e, Mesh* mesh, const int first_step,
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy);

void lagrangian_step(
    const int nx, const int ny, Mesh* mesh, const double dt, double* rho_u, 
    double* rho_v, double* u, double* v, const double* P, const double* rho,
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy);

void artificial_viscosity(
    const int nx, const int ny, Mesh* mesh, const double dt, double* Qxx, 
    double* Qyy, double* u, double* v, double* rho_u, double* rho_v, const double* rho, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy);

// Calculates the work done due to forces within the element
void shock_heating_and_work(
    const int nx, const int ny, Mesh* mesh, const double dt_h, double* e, 
    const double* P, const double* u, const double* v, const double* rho, 
    const double* Qxx, const double* Qyy, const double* celldx, const double* celldy);

// Perform advection with monotonicity improvement
void advect_mass(
    const int nx, const int ny, Mesh* mesh, const double dt_h, double* rho, 
    double* rho_old, double* F_x, double* F_y, const double* u, const double* v, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy);

// Advect momentum according to the velocity
void advect_momentum(
    const int nx, const int ny, Mesh* mesh, const double dt_h, const double dt, 
    double* u, double* v, double* slope_x, double* slope_y, double* mF_x, 
    double* mF_y, double* rho_u, double* rho_v, const double* rho, 
    const double* F_x, const double* F_y, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy);

// Perform advection of internal energy
void advect_energy(
    const int nx, const int ny, Mesh* mesh, const double dt_h, const double dt, 
    double* e, double* F_x, double* F_y, 
    const double* u, const double* v, const double* rho_old, const double* rho,
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy);

// Enforce reflective boundary conditions on the problem state
void handle_boundary(
    const int nx, const int ny, Mesh* mesh, 
    double* arr, const int invert, const int pack);

// Write out data for visualisation in visit
// Prints some conservation values
void print_conservation(
    const int nx, const int ny, State* state, Mesh* mesh);

