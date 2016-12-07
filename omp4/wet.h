#include "../../mesh.h"

// Calculate the pressure from GAMma law equation of state
void equation_of_state(
    const int nx, const int ny, double* P, const double* rho, const double* e);

// Calculate change in momentum caused by pressure gradients, and then extract
// the velocities using edge centered density approximations
void pressure_acceleration(
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
    const int nx, const int ny, Mesh* mesh, const int tt, const double dt,
    const double dt_h, double* rho, double* e, double* rho_old, double* F_x, double* F_y, 
    double* eF_x, double* eF_y, const double* u, const double* v, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy);

void ux_momentum_flux(
    const int nx, const int ny, Mesh* mesh, const double dt_h, const double dt, 
    double* u, double* v, double* uF_x, double* rho_u, const double* rho, const double* F_x, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy);

void uy_momentum_flux(
    const int nx, const int ny, Mesh* mesh, const double dt_h, const double dt, 
    double* u, double* v, double* uF_y, double* rho_u, const double* rho, 
    const double* F_y, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy);

void vx_momentum_flux(
    const int nx, const int ny, Mesh* mesh, const double dt_h, const double dt, 
    const double* u, double* v, double* vF_x, double* rho_v, const double* rho, 
    const double* F_x, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy);

void vy_momentum_flux(
    const int nx, const int ny, Mesh* mesh, const double dt_h, const double dt, 
    double* u, double* v, double* vF_y, double* rho_v, const double* rho, const double* F_y, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy);

// Advect momentum according to the velocity
void advect_momentum(
    const int nx, const int ny, const int tt, Mesh* mesh, const double dt_h, 
    const double dt, double* u, double* v, double* uF_x, double* uF_y, 
    double* vF_x, double* vF_y, double* rho_u, double* rho_v, 
    const double* rho, const double* F_x, const double* F_y, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy);

