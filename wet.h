#include "../shared.h"
#include "../mesh.h"
#include "../state.h"

// Controllable parameters for the application
#define GAM 1.4
#define C_Q 2.0
#define C_M (1.5/C_T)

// Constitutes an individual solve of a mesh
void solve_hydro(
    Mesh* mesh, int tt, double* P, double* rho, double* rho_old, 
    double* e, double* u, double*v , double* rho_u, double* rho_v, 
    double* Qxx, double* Qyy, double* F_x, double* F_y, double* uF_x, 
    double* uF_y, double* vF_x, double* vF_y);

void equation_of_state(
    const int nx, const int ny, double* P, const double* rho, const double* e);

void set_timestep(
    const int nx, const int ny, double* Qxx, double* Qyy, const double* rho, 
    const double* e, Mesh* mesh, const int first_step,
    const double* celldx, const double* celldy);

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
void advect_mass_and_energy(
    const int nx, const int ny, Mesh* mesh, const int tt, const double dt_h, 
    double* rho, double* e, double* rho_old, double* F_x, double* F_y, 
    double* eF_x, double* eF_y, const double* u, const double* v, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy);

// Calculate the flux in the x direction
void x_mass_and_energy_flux(
    const int nx, const int ny, Mesh* mesh, const double dt_h, double* rho, 
    const double* u, double* F_x, const double* celldx, const double* edgedx, 
    const double* celldy, const double* edgedy);

// Calculate the flux in the y direction
void y_mass_and_energy_flux(
    const int nx, const int ny, Mesh* mesh, const double dt_h, double* rho, 
    const double* v, double* F_y, const double* celldx, const double* edgedx, 
    const double* celldy, const double* edgedy);

void x_energy_flux(
    const int nx, const int ny, const double dt_h, double* rho, 
    const double* u, double* e, const double* rho_old, double* F_x, double* eF_y,
    const double* celldx, const double* edgedx, 
    const double* celldy, const double* edgedy);

void y_energy_flux(
    const int nx, const int ny, const double dt_h, double* rho, 
    const double* v, double* e, const double* rho_old, double* F_y, double* eF_y, 
    const double* celldx, const double* edgedx, 
    const double* celldy, const double* edgedy);

// Advect momentum according to the velocity
void advect_momentum(
    const int nx, const int ny, Mesh* mesh, const double dt_h, const double dt, 
    double* u, double* v, double* slope_x, double* slope_y, double* mF_x, 
    double* mF_y, double* rho_u, double* rho_v, const double* rho, 
    const double* F_x, const double* F_y, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy);

// Write out data for visualisation in visit
// Prints some conservation values
void print_conservation(
    const int nx, const int ny, State* state, Mesh* mesh);

