#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "../flow_interface.h"
#include "../../cuda/shared.h"
#include "../../cuda/reduction.k"

__global__ void equation_of_state(
    const int nx, const int ny, double* P, const double* rho, const double* e)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  if(gid >= nx*ny) return;
  const int jj = (gid % (nx));
  const int ii = (gid / (nx));

  // Only invoke simple GAMma law at the moment
  P[(ii*nx+jj)] = (GAM - 1.0)*rho[(ii*nx+jj)]*e[(ii*nx+jj)];
}

// Calculate change in momentum caused by pressure gradients, and then extract
// the velocities using edge centered density approximations
__global__ void pressure_acceleration(
    const int nx, const int ny, Mesh* mesh, const double dt, double* rho_u, 
    double* rho_v, double* u, double* v, const double* P, const double* rho,
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int jj = (gid % (nx+1));
  const int ii = (gid / (nx+1));

  if(ii < PAD || jj < PAD || jj >= (nx+1)-PAD || ii >= (ny+1)-PAD) 
    return;

  // Update the momenta using the pressure gradients
  rho_u[(ii*(nx+1)+jj)] -= dt*(P[(ii*nx+jj)] - P[(ii*nx+jj)-1])/edgedx[jj];
  rho_v[(ii*nx+jj)] -= dt*(P[(ii*nx+jj)] - P[(ii*nx+jj)-nx])/edgedy[ii];

  // Calculate the zone edge centered density
  const double rho_edge_x = 
    (rho[(ii*nx+jj)]*celldx[jj]*celldy[ii] + rho[(ii*nx+jj)-1]*celldx[jj - 1]*celldy[ii])/ 
    (2.0*edgedx[jj]*celldy[ii]);
  const double rho_edge_y = 
    (rho[(ii*nx+jj)]*celldx[jj]*celldy[ii] + rho[(ii*nx+jj)-nx]*celldx[jj]*celldy[ii - 1])/ 
    (2.0*celldx[jj]*edgedy[ii]);

  // Find the velocities from the momenta and edge centered mass densities
  u[(ii*(nx+1)+jj)] = (rho_edge_x == 0.0) ? 0.0 : rho_u[(ii*(nx+1)+jj)] / rho_edge_x;
  v[(ii*nx+jj)] = (rho_edge_y == 0.0) ? 0.0 : rho_v[(ii*nx+jj)] / rho_edge_y;
}

__global__ void calc_viscous_stresses(
    const int nx, const int ny, Mesh* mesh, const double dt, double* Qxx, 
    double* Qyy, double* u, double* v, double* rho_u, double* rho_v, const double* rho, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int jj = (gid % (nx));
  const int ii = (gid / (nx));

  if(ii < PAD || jj < PAD || jj >= nx-PAD || ii >= ny-PAD) 
    return;

  const double u_i = min(0.0, u[(ii*(nx+1)+jj)+1] - u[(ii*(nx+1)+jj)]);
  const double u_ii = 0.5*(
      fabs(min(0.0, (u[(ii*(nx+1)+jj)+2]-u[(ii*(nx+1)+jj)+1])) - min(0.0, (u[(ii*(nx+1)+jj)+1]-u[(ii*(nx+1)+jj)]))) + 
      fabs(min(0.0, (u[(ii*(nx+1)+jj)+1]-u[(ii*(nx+1)+jj)])) - min(0.0, (u[(ii*(nx+1)+jj)]-u[(ii*(nx+1)+jj)-1]))));
  const double v_i = min(0.0, v[(ii*nx+jj)+nx] - v[(ii*nx+jj)]);
  const double v_ii = 0.5*(
      fabs(min(0.0, (v[(ii*nx+jj)+2*nx]-v[(ii*nx+jj)+nx])) - min(0.0, (v[(ii*nx+jj)+nx]-v[(ii*nx+jj)]))) + 
      fabs(min(0.0, (v[(ii*nx+jj)+nx]-v[(ii*nx+jj)])) - min(0.0, (v[(ii*nx+jj)]-v[(ii*nx+jj)-nx]))));
  Qxx[(ii*nx+jj)] = -C_Q*rho[(ii*nx+jj)]*u_i*u_ii;
  Qyy[(ii*nx+jj)] = -C_Q*rho[(ii*nx+jj)]*v_i*v_ii;
}

__global__ void viscous_acceleration(
    const int nx, const int ny, Mesh* mesh, const double dt, double* Qxx, 
    double* Qyy, double* u, double* v, double* rho_u, double* rho_v, const double* rho, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int jj = (gid % (nx+1));
  const int ii = (gid / (nx+1));

  if(ii < PAD || jj < PAD || jj >= (nx+1)-PAD || ii >= (ny+1)-PAD) 
    return;

  rho_u[(ii*(nx+1)+jj)] -= dt*(Qxx[(ii*nx+jj)] - Qxx[(ii*nx+jj)-1])/celldx[jj];
  rho_v[(ii*nx+jj)] -= dt*(Qyy[(ii*nx+jj)] - Qyy[(ii*nx+jj)-nx])/celldy[ii];

  // Calculate the zone edge centered density
  const double rho_edge_x = 
    (rho[(ii*nx+jj)]*celldx[jj]*celldy[ii] + rho[(ii*nx+jj)-1]*celldx[jj-1]*celldy[ii]) / 
    (2.0*edgedx[jj]*celldy[ii]);
  const double rho_edge_y = 
    (rho[(ii*nx+jj)]*celldx[jj]*celldy[ii] + rho[(ii*nx+jj)-nx]*celldx[jj]*celldy[ii-1]) / 
    (2.0*celldx[jj]*edgedy[ii]);

  // Find the velocities from the momenta and edge centered mass densities
  u[(ii*(nx+1)+jj)] = (rho_edge_x == 0.0) ? 0.0 : rho_u[(ii*(nx+1)+jj)] / rho_edge_x;
  v[(ii*nx+jj)] = (rho_edge_y == 0.0) ? 0.0 : rho_v[(ii*nx+jj)] / rho_edge_y;
}

// Calculates the work done due to forces within the element
__global__ void shock_heating_and_work(
    const int nx, const int ny, Mesh* mesh, const double dt_h, double* e, 
    const double* P, const double* u, const double* v, const double* rho, 
    const double* Qxx, const double* Qyy, const double* celldx, const double* celldy)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int jj = (gid % (nx));
  const int ii = (gid / (nx));

  if(ii < PAD || jj < PAD || jj >= nx-PAD || ii >= ny-PAD) 
    return;

  const double div_vel_x = (u[(ii*(nx+1)+jj)+1] - u[(ii*(nx+1)+jj)])/celldx[jj];
  const double div_vel_y = (v[(ii*nx+jj)+nx] - v[(ii*nx+jj)])/celldy[ii];
  const double div_vel_dt = (div_vel_x + div_vel_y)*dt_h;
  const double e_q = e[(ii*nx+jj)] - dt_h*(Qxx[(ii*nx+jj)]*div_vel_x + Qyy[(ii*nx+jj)]*div_vel_y)/rho[(ii*nx+jj)];

  /// A working formulation that is second order in time for Pressure!?
  const double rho_c = rho[(ii*nx+jj)]/(1.0 + div_vel_dt);
  const double e_c = e_q - (P[(ii*nx+jj)]*div_vel_dt)/rho[(ii*nx+jj)];
  const double work = 0.5*div_vel_dt*(P[(ii*nx+jj)] + (GAM-1.0)*e_c*rho_c)/rho[(ii*nx+jj)];
  e[(ii*nx+jj)] = (rho[(ii*nx+jj)] == 0.0) ? 0.0 : e_q-work;
}

__global__ void store_old_rho(
    const int nx, const int ny, double* rho, double* rho_old)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int jj = (gid % (nx));
  const int ii = (gid / (nx));

  if(ii < PAD || jj < PAD || jj >= nx-PAD || ii >= ny-PAD) 
    return;

  rho_old[(ii*nx+jj)] = rho[(ii*nx+jj)];
}

// Calculate the flux in the x direction
__global__ void calc_x_mass_and_energy_flux(
    const int nx, const int ny, const int first, Mesh* mesh, const double dt, 
    const double dt_h, double* rho, double* rho_old, double* e, const double* u, 
    double* F_x, double* eF_x, const double* celldx, const double* edgedx, 
    const double* celldy, const double* edgedy)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int jj = (gid % (nx+1));
  const int ii = (gid / (nx+1));

  if(ii < PAD || jj < PAD || jj >= (nx+1)-PAD || ii >= ny-PAD) 
    return;

  // Interpolate to make second order in time
  const double invdx = 1.0/edgedx[jj];
  const double suc0 = 0.5*invdx*(u[(ii*(nx+1)+jj)+1]-u[(ii*(nx+1)+jj)-1]);
  const double sur0 = 2.0*invdx*(u[(ii*(nx+1)+jj)]-u[(ii*(nx+1)+jj)-1]);
  const double sul0 = 2.0*invdx*(u[(ii*(nx+1)+jj)+1]-u[(ii*(nx+1)+jj)]);
  const double u_tc = u[(ii*(nx+1)+jj)] - 0.5*u[(ii*(nx+1)+jj)]*dt*minmod(suc0, minmod(sur0, sul0));

  // Van leer limiter
  double limiter = 0.0;
  const double rho_diff = (rho[(ii*nx+jj)]-rho[(ii*nx+jj)-1]);
  if(rho_diff) {
    const double smoothness = (u_tc >= 0.0) 
      ? (rho[(ii*nx+jj)-1]-rho[(ii*nx+jj)-2])/rho_diff
      : (rho[(ii*nx+jj)+1]-rho[(ii*nx+jj)])/rho_diff;
    limiter = (smoothness + fabs(smoothness))/(1.0+fabs(smoothness));
  }

  // Calculate the flux
  const double rho_upwind = (u_tc >= 0.0) ? rho[(ii*nx+jj)-1] : rho[(ii*nx+jj)];
  F_x[(ii*(nx+1)+jj)] = (u_tc*rho_upwind+
      0.5*fabs(u_tc)*(1.0-fabs((u_tc*dt_h)/celldx[jj]))*limiter*rho_diff);

  // Use MC limiter to get slope of energy
  const double a_x_0 = 0.5*invdx*(e[(ii*nx+jj)]-e[(ii*nx+jj)-2]);
  const double b_x_0 = 2.0*invdx*(e[(ii*nx+jj)-1]-e[(ii*nx+jj)-2]);
  const double c_x_0 = 2.0*invdx*(e[(ii*nx+jj)]-e[(ii*nx+jj)-1]);
  const double a_x_1 = 0.5*invdx*(e[(ii*nx+jj)+1]-e[(ii*nx+jj)-1]);
  const double b_x_1 = 2.0*invdx*(e[(ii*nx+jj)]-e[(ii*nx+jj)-1]);
  const double c_x_1 = 2.0*invdx*(e[(ii*nx+jj)+1]-e[(ii*nx+jj)]);

  // Calculate the interpolated densities
  const double edge_e_x = (u_tc > 0.0)
    ? e[(ii*nx+jj)-1] + 0.5*minmod(minmod(a_x_0, b_x_0), c_x_0)*(celldx[jj-1] - u_tc*dt_h)
    : e[(ii*nx+jj)] - 0.5*minmod(minmod(a_x_1, b_x_1), c_x_1)*(celldx[jj] + u_tc*dt_h);

  // Update the fluxes to now include the contribution from energy
  eF_x[(ii*(nx+1)+jj)] = edgedy[ii]*edge_e_x*F_x[(ii*(nx+1)+jj)]; 
}

// Advect mass and energy in the x direction
__global__ void advect_mass_and_energy_in_x(
    const int nx, const int ny, const int first, Mesh* mesh, const double dt, 
    const double dt_h, double* rho, double* rho_old, double* e, const double* u, 
    double* F_x, double* eF_x, const double* celldx, const double* edgedx, 
    const double* celldy, const double* edgedy)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int jj = (gid % (nx));
  const int ii = (gid / (nx));

  if(ii < PAD || jj < PAD || jj >= nx-PAD || ii >= ny-PAD) 
    return;

  rho[(ii*nx+jj)] -= dt_h*
    (edgedy[ii+1]*F_x[(ii*(nx+1)+jj)+1] - edgedy[ii]*F_x[(ii*(nx+1)+jj)])/ 
    (celldx[jj]*celldy[ii]);
  const double rho_e = (rho_old[(ii*nx+jj)]*e[(ii*nx+jj)] - 
      (dt_h*(eF_x[(ii*(nx+1)+jj)+1] - eF_x[(ii*(nx+1)+jj)]))/(celldx[jj]*celldy[ii]));
  e[(ii*nx+jj)] = (first) 
    ? (rho_old[(ii*nx+jj)] == 0.0) ? 0.0 : rho_e/rho_old[(ii*nx+jj)]
    : (rho[(ii*nx+jj)] == 0.0) ? 0.0 : rho_e/rho[(ii*nx+jj)];
}

__global__ void calc_y_mass_and_energy_flux(
    const int nx, const int ny, const int first, Mesh* mesh, const double dt,
    const double dt_h, double* rho, double* rho_old, double* e, const double* v, 
    double* F_y, double* eF_y, const double* celldx, const double* edgedx, 
    const double* celldy, const double* edgedy)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int jj = (gid % (nx));
  const int ii = (gid / (nx));

  if(ii < PAD || jj < PAD || jj >= nx-PAD || ii >= (ny+1)-PAD) 
    return;

  // Compute the mass flux along the y edges
  // In the ghost cells flux is left as 0.0
  // Interpolate the velocity to make second order in time
  const double invdy = 1.0/edgedy[ii];
  const double svc0 = 0.5*invdy*(v[(ii*nx+jj)+nx]-v[(ii*nx+jj)-nx]);
  const double svr0 = 2.0*invdy*(v[(ii*nx+jj)]-v[(ii*nx+jj)-nx]);
  const double svl0 = 2.0*invdy*(v[(ii*nx+jj)+nx]-v[(ii*nx+jj)]);
  const double v_tc = v[(ii*nx+jj)] - 0.5*v[(ii*nx+jj)]*dt*minmod(svc0, minmod(svr0, svl0));

  // Van leer limiter
  const double rho_diff = (rho[(ii*nx+jj)]-rho[(ii*nx+jj)-nx]);
  double limiter = 0.0;
  if(rho_diff) {
    const double smoothness = (v[(ii*nx+jj)] >= 0.0) 
      ? (rho[(ii*nx+jj)-nx]-rho[(ii*nx+jj)-2*nx])/rho_diff
      : (rho[(ii*nx+jj)+nx]-rho[(ii*nx+jj)])/rho_diff;
    limiter = (smoothness + fabs(smoothness))/(1.0+fabs(smoothness));
  }

  // Calculate the flux
  const double rho_upwind = (v_tc >= 0.0) ? rho[(ii*nx+jj)-nx] : rho[(ii*nx+jj)];
  F_y[(ii*nx+jj)] = (v_tc*rho_upwind+
      0.5*fabs(v_tc)*(1.0-fabs((v_tc*dt_h)/celldy[ii]))*limiter*rho_diff);

  // Use MC limiter to get slope of energy
  const double a_y_0 = 0.5*invdy*(e[(ii*nx+jj)]-e[(ii*nx+jj)-2*nx]);
  const double b_y_0 = 2.0*invdy*(e[(ii*nx+jj)-nx]-e[(ii*nx+jj)-2*nx]);
  const double c_y_0 = 2.0*invdy*(e[(ii*nx+jj)]-e[(ii*nx+jj)-nx]);
  const double a_y_1 = 0.5*invdy*(e[(ii*nx+jj)+nx]-e[(ii*nx+jj)-nx]);
  const double b_y_1 = 2.0*invdy*(e[(ii*nx+jj)]-e[(ii*nx+jj)-nx]);
  const double c_y_1 = 2.0*invdy*(e[(ii*nx+jj)+nx]-e[(ii*nx+jj)]);

  const double edge_e_y = (v_tc > 0.0)
    ? e[(ii*nx+jj)-nx] + 0.5*minmod(minmod(a_y_0, b_y_0), c_y_0)*(celldy[ii-1] - v_tc*dt_h)
    : e[(ii*nx+jj)] - 0.5*minmod(minmod(a_y_1, b_y_1), c_y_1)*(celldy[ii] + v_tc*dt_h);

  // Update the fluxes to now include the contribution from energy
  eF_y[(ii*nx+jj)] = edgedx[jj]*edge_e_y*F_y[(ii*nx+jj)]; 
}

__global__ void advect_mass_and_energy_in_y(
    const int nx, const int ny, const int first, Mesh* mesh, const double dt,
    const double dt_h, double* rho, double* rho_old, double* e, const double* v, 
    double* F_y, double* eF_y, const double* celldx, const double* edgedx, 
    const double* celldy, const double* edgedy)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int jj = (gid % (nx));
  const int ii = (gid / (nx));

  if(ii < PAD || jj < PAD || jj >= nx-PAD || ii >= ny-PAD) 
    return;

  rho[(ii*nx+jj)] -= dt_h*
    (edgedx[jj+1]*F_y[(ii*nx+jj)+nx] - edgedx[jj]*F_y[(ii*nx+jj)])/
    (celldx[jj]*celldy[ii]);
  const double rho_e = (rho_old[(ii*nx+jj)]*e[(ii*nx+jj)] - 
      (dt_h*(eF_y[(ii*nx+jj)+nx] - eF_y[(ii*nx+jj)]))/(celldx[jj]*celldy[ii]));
  e[(ii*nx+jj)] = (first) 
    ? (rho_old[(ii*nx+jj)] == 0.0) ? 0.0 : rho_e/rho_old[(ii*nx+jj)]
    : (rho[(ii*nx+jj)] == 0.0) ? 0.0 : rho_e/rho[(ii*nx+jj)];
}

__global__ void ux_momentum_flux(
    const int nx, const int ny, Mesh* mesh, const double dt_h, const double dt, 
    double* u, double* v, double* uF_x, double* rho_u, const double* rho, const double* F_x, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int jj = (gid % (nx));
  const int ii = (gid / (nx));

  if(ii < PAD || jj < PAD || jj >= nx-PAD || ii >= ny-PAD) 
    return;

  // Calculate the cell centered x momentum fluxes in the x direction
  // Use MC limiter to get slope of velocity
  const double invdx = 1.0/edgedx[jj];
  const double a_x_0 = 0.5*invdx*(u[(ii*(nx+1)+jj)+1]-u[(ii*(nx+1)+jj)-1]);
  const double b_x_0 = 2.0*invdx*(u[(ii*(nx+1)+jj)]-u[(ii*(nx+1)+jj)-1]);
  const double c_x_0 = 2.0*invdx*(u[(ii*(nx+1)+jj)+1]-u[(ii*(nx+1)+jj)]);
  const double a_x_1 = 0.5*invdx*(u[(ii*(nx+1)+jj)+2]-u[(ii*(nx+1)+jj)]);
  const double b_x_1 = 2.0*invdx*(u[(ii*(nx+1)+jj)+1]-u[(ii*(nx+1)+jj)]);
  const double c_x_1 = 2.0*invdx*(u[(ii*(nx+1)+jj)+2]-u[(ii*(nx+1)+jj)+1]);

  // Calculate the interpolated densities
  const double u_cell_x = 0.5*(u[(ii*(nx+1)+jj)]+u[(ii*(nx+1)+jj)+1]);
  const double f_x = edgedy[ii]*0.5*(F_x[(ii*(nx+1)+jj)] + F_x[(ii*(nx+1)+jj)+1]); 
  const double u_cell_x_interp = (u_cell_x > 0.0)
    ? u[(ii*(nx+1)+jj)] + 0.5*minmod(minmod(a_x_0, b_x_0), c_x_0)*(celldx[jj-1] - u_cell_x*dt_h)
    : u[(ii*(nx+1)+jj)+1] - 0.5*minmod(minmod(a_x_1, b_x_1), c_x_1)*(celldx[jj] + u_cell_x*dt_h);
  uF_x[(ii*nx+jj)] = f_x*u_cell_x_interp;
}


__global__ void advect_rho_u_in_x(
    const int nx, const int ny, const int tt, Mesh* mesh, const double dt_h, 
    const double dt, double* u, double* v, double* uF_x, double* uF_y, 
    double* vF_x, double* vF_y, double* rho_u, double* rho_v, 
    const double* rho, const double* F_x, const double* F_y, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int jj = (gid % (nx+1));
  const int ii = (gid / (nx+1));

  if(ii < PAD || jj < PAD || jj >= (nx+1)-PAD || ii >= ny-PAD) 
    return;

  rho_u[(ii*(nx+1)+jj)] -= dt_h*(uF_x[(ii*nx+jj)] - uF_x[(ii*nx+jj)-1])/(edgedx[jj]*celldy[ii]);
}

__global__ void advect_rho_u_and_u_in_x(
    const int nx, const int ny, const int tt, Mesh* mesh, const double dt_h, 
    const double dt, double* u, double* v, double* uF_x, double* uF_y, 
    double* vF_x, double* vF_y, double* rho_u, double* rho_v, 
    const double* rho, const double* F_x, const double* F_y, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int jj = (gid % (nx+1));
  const int ii = (gid / (nx+1));

  if(ii < PAD || jj < PAD || jj >= (nx+1)-PAD || ii >= ny-PAD) 
    return;

  rho_u[(ii*(nx+1)+jj)] -= dt_h*(uF_x[(ii*nx+jj)] - uF_x[(ii*nx+jj)-1])/(edgedx[jj]*celldy[ii]);
  const double rho_edge_x = 
    (rho[(ii*nx+jj)]*celldx[jj]*celldy[ii] + rho[(ii*nx+jj)-1]*celldx[jj - 1]*celldy[ii])/ 
    (2.0*edgedx[jj]*celldy[ii]);
  u[(ii*(nx+1)+jj)] = (rho_edge_x == 0.0) ? 0.0 : rho_u[(ii*(nx+1)+jj)] / rho_edge_x;
}

__global__ void advect_rho_u_in_y(
    const int nx, const int ny, const int tt, Mesh* mesh, const double dt_h, 
    const double dt, double* u, double* v, double* uF_x, double* uF_y, 
    double* vF_x, double* vF_y, double* rho_u, double* rho_v, 
    const double* rho, const double* F_x, const double* F_y, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int jj = (gid % (nx+1));
  const int ii = (gid / (nx+1));
  if(ii < PAD || jj < PAD || jj >= (nx+1)-PAD || ii >= ny-PAD) 
    return;

  rho_u[(ii*(nx+1)+jj)] -= dt_h*(uF_y[(ii*(nx+1)+jj)+(nx+1)] - uF_y[(ii*(nx+1)+jj)])/(celldx[jj]*edgedy[ii]);
}

__global__ void advect_rho_u_and_u_in_y(
    const int nx, const int ny, const int tt, Mesh* mesh, const double dt_h, 
    const double dt, double* u, double* v, double* uF_x, double* uF_y, 
    double* vF_x, double* vF_y, double* rho_u, double* rho_v, 
    const double* rho, const double* F_x, const double* F_y, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int jj = (gid % (nx+1));
  const int ii = (gid / (nx+1));

  if(ii < PAD || jj < PAD || jj >= (nx+1)-PAD || ii >= ny-PAD) 
    return;

  rho_u[(ii*(nx+1)+jj)] -= dt_h*(uF_y[(ii*(nx+1)+jj)+(nx+1)] - uF_y[(ii*(nx+1)+jj)])/(celldx[jj]*edgedy[ii]);
  const double rho_edge_x = 
    (rho[(ii*nx+jj)]*celldx[jj]*celldy[ii] + rho[(ii*nx+jj)-1]*celldx[jj - 1]*celldy[ii])/ 
    (2.0*edgedx[jj]*celldy[ii]);
  u[(ii*(nx+1)+jj)] = (rho_edge_x == 0.0) ? 0.0 : rho_u[(ii*(nx+1)+jj)] / rho_edge_x;
}

__global__ void uy_momentum_flux(
    const int nx, const int ny, Mesh* mesh, const double dt_h, const double dt, 
    double* u, double* v, double* uF_y, double* rho_u, const double* rho, 
    const double* F_y, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int jj = (gid % (nx+1));
  const int ii = (gid / (nx+1));

  if(ii < PAD || jj < PAD || jj >= (nx+1)-PAD || ii >= (ny+1)-PAD) 
    return;

  // Use MC limiter to get slope of velocity
  const double invdy = 1.0/edgedy[ii];
  const double a_y_0 = 0.5*invdy*(u[(ii*(nx+1)+jj)]-u[(ii*(nx+1)+jj)-2*(nx+1)]);
  const double b_y_0 = 2.0*invdy*(u[(ii*(nx+1)+jj)-(nx+1)]-u[(ii*(nx+1)+jj)-2*(nx+1)]);
  const double c_y_0 = 2.0*invdy*(u[(ii*(nx+1)+jj)]-u[(ii*(nx+1)+jj)-(nx+1)]);
  const double a_y_1 = 0.5*invdy*(u[(ii*(nx+1)+jj)+(nx+1)]-u[(ii*(nx+1)+jj)-(nx+1)]);
  const double b_y_1 = 2.0*invdy*(u[(ii*(nx+1)+jj)]-u[(ii*(nx+1)+jj)-(nx+1)]);
  const double c_y_1 = 2.0*invdy*(u[(ii*(nx+1)+jj)+(nx+1)]-u[(ii*(nx+1)+jj)]);
  const double v_cell_y = 0.5*(v[(ii*nx+jj)-1]+v[(ii*nx+jj)]);

  const double f_y = edgedx[jj]*0.5*(F_y[(ii*nx+jj)] + F_y[(ii*nx+jj)-1]);
  const double u_corner_y = (v_cell_y > 0.0)
    ? u[(ii*(nx+1)+jj)-(nx+1)] + 0.5*minmod(minmod(a_y_0, b_y_0), c_y_0)*(celldy[ii-1] - v_cell_y*dt_h)
    : u[(ii*(nx+1)+jj)] - 0.5*minmod(minmod(a_y_1, b_y_1), c_y_1)*(celldy[ii] + v_cell_y*dt_h);
  uF_y[(ii*(nx+1)+jj)] = f_y*u_corner_y;
}

__global__ void vx_momentum_flux(
    const int nx, const int ny, Mesh* mesh, const double dt_h, const double dt, 
    const double* u, double* v, double* vF_x, double* rho_v, const double* rho, 
    const double* F_x, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int jj = (gid % (nx+1));
  const int ii = (gid / (nx+1));

  if(ii < PAD || jj < PAD || jj >= (nx+1)-PAD || ii >= (ny+1)-PAD) {
    return;
  }

  // Use MC limiter to get slope of velocity
  const double invdx = 1.0/edgedx[jj];
  const double a_x_0 = 0.5*invdx*(v[(ii*nx+jj)]-v[(ii*nx+jj)-2]);
  const double b_x_0 = 2.0*invdx*(v[(ii*nx+jj)-1]-v[(ii*nx+jj)-2]);
  const double c_x_0 = 2.0*invdx*(v[(ii*nx+jj)]-v[(ii*nx+jj)-1]);
  const double a_x_1 = 0.5*invdx*(v[(ii*nx+jj)+1]-v[(ii*nx+jj)-1]);
  const double b_x_1 = 2.0*invdx*(v[(ii*nx+jj)]-v[(ii*nx+jj)-1]);
  const double c_x_1 = 2.0*invdx*(v[(ii*nx+jj)+1]-v[(ii*nx+jj)]);

  // Calculate the interpolated densities
  const double f_x = celldy[ii]*0.5*(F_x[(ii*(nx+1)+jj)] + F_x[(ii*(nx+1)+jj)-(nx+1)]);
  const double u_cell_x = 0.5*(u[(ii*(nx+1)+jj)]+u[(ii*(nx+1)+jj)-(nx+1)]);
  const double v_cell_x_interp = (u_cell_x > 0.0)
    ? v[(ii*nx+jj)-1] + 0.5*minmod(minmod(a_x_0, b_x_0), c_x_0)*(celldx[jj-1] - u_cell_x*dt_h)
    : v[(ii*nx+jj)] - 0.5*minmod(minmod(a_x_1, b_x_1), c_x_1)*(celldx[jj] + u_cell_x*dt_h);
  vF_x[(ii*(nx+1)+jj)] = f_x*v_cell_x_interp;
}

__global__ void advect_rho_v_and_v_in_y(
    const int nx, const int ny, Mesh* mesh, const double dt_h, const double dt, 
    double* u, double* v, double* vF_y, double* rho_v, const double* rho, const double* F_y, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int jj = (gid % (nx));
  const int ii = (gid / (nx));

  if(ii < PAD || jj < PAD || jj >= nx-PAD || ii >= (ny+1)-PAD) 
    return;

  rho_v[(ii*nx+jj)] -= dt_h*(vF_y[(ii*nx+jj)] - vF_y[(ii*nx+jj)-nx])/(celldx[jj]*edgedy[ii]);
  const double rho_edge_y = 
    (rho[(ii*nx+jj)]*celldx[jj]*celldy[ii] + rho[(ii*nx+jj)-nx]*celldx[jj]*celldy[ii - 1])/ 
    (2.0*celldx[jj]*edgedy[ii]);
  v[(ii*nx+jj)] = (rho_edge_y == 0.0) ? 0.0 : rho_v[(ii*nx+jj)] / rho_edge_y;
}

__global__ void advect_rho_v_and_v_in_x(
    const int nx, const int ny, Mesh* mesh, const double dt_h, const double dt, 
    const double* u, double* v, double* vF_x, double* rho_v, const double* rho, 
    const double* F_x, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int jj = (gid % (nx));
  const int ii = (gid / (nx));

  if(ii < PAD || jj < PAD || jj >= nx-PAD || ii >= (ny+1)-PAD) 
    return;

  rho_v[(ii*nx+jj)] -= dt_h*(vF_x[(ii*(nx+1)+jj)+1] - vF_x[(ii*(nx+1)+jj)])/(edgedx[jj]*celldy[ii]);
  const double rho_edge_y = 
    (rho[(ii*nx+jj)]*celldx[jj]*celldy[ii] + rho[(ii*nx+jj)-nx]*celldx[jj]*celldy[ii - 1])/ 
    (2.0*celldx[jj]*edgedy[ii]);
  v[(ii*nx+jj)] = (rho_edge_y == 0.0) ? 0.0 : rho_v[(ii*nx+jj)] / rho_edge_y;
}

__global__ void advect_rho_v_in_x(
    const int nx, const int ny, Mesh* mesh, const double dt_h, const double dt, 
    const double* u, double* v, double* vF_x, double* rho_v, const double* rho, 
    const double* F_x, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int jj = (gid % (nx));
  const int ii = (gid / (nx));

  if(ii < PAD || jj < PAD || jj >= nx-PAD || ii >= (ny+1)-PAD) 
    return;

  rho_v[(ii*nx+jj)] -= dt_h*(vF_x[(ii*(nx+1)+jj)+1] - vF_x[(ii*(nx+1)+jj)])/(edgedx[jj]*celldy[ii]);
}

__global__ void vy_momentum_flux(
    const int nx, const int ny, Mesh* mesh, const double dt_h, const double dt, 
    double* u, double* v, double* vF_y, double* rho_v, const double* rho, const double* F_y, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int jj = (gid % (nx));
  const int ii = (gid / (nx));

  if(ii < PAD || jj < PAD || jj >= nx-PAD || ii >= ny-PAD) 
    return;

  // Use MC limiter to get slope of velocity
  const double invdy = 1.0/edgedy[ii];
  const double a_y_0 = 0.5*invdy*(v[(ii*nx+jj)+nx]-v[(ii*nx+jj)-nx]);
  const double b_y_0 = 2.0*invdy*(v[(ii*nx+jj)]-v[(ii*nx+jj)-nx]);
  const double c_y_0 = 2.0*invdy*(v[(ii*nx+jj)+nx]-v[(ii*nx+jj)]);
  const double a_y_1 = 0.5*invdy*(v[(ii*nx+jj)+2*nx]-v[(ii*nx+jj)]);
  const double b_y_1 = 2.0*invdy*(v[(ii*nx+jj)+nx]-v[(ii*nx+jj)]);
  const double c_y_1 = 2.0*invdy*(v[(ii*nx+jj)+2*nx]-v[(ii*nx+jj)+nx]);

  const double f_y = celldx[jj]*0.5*(F_y[(ii*nx+jj)] + F_y[(ii*nx+jj)+nx]);
  const double v_cell_y = 0.5*(v[(ii*nx+jj)]+v[(ii*nx+jj)+nx]);
  const double v_cell_y_interp = (v_cell_y > 0.0)
    ? v[(ii*nx+jj)] + 0.5*minmod(minmod(a_y_0, b_y_0), c_y_0)*(celldy[ii-1] - v_cell_y*dt_h)
    : v[(ii*nx+jj)+nx] - 0.5*minmod(minmod(a_y_1, b_y_1), c_y_1)*(celldy[ii] + v_cell_y*dt_h);
  vF_y[(ii*nx+jj)] = f_y*v_cell_y_interp;
}

__global__ void advect_rho_v_in_y(
    const int nx, const int ny, Mesh* mesh, const double dt_h, const double dt, 
    double* u, double* v, double* vF_y, double* rho_v, const double* rho, const double* F_y, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int jj = (gid % (nx));
  const int ii = (gid / (nx));

  if(ii < PAD || jj < PAD || jj >= nx-PAD || ii >= (ny+1)-PAD) 
    return;

  rho_v[(ii*nx+jj)] -= dt_h*(vF_y[(ii*nx+jj)] - vF_y[(ii*nx+jj)-nx])/(celldx[jj]*edgedy[ii]);
}

__global__ void calc_min_timestep(
    const int nx, const int ny, const double max_dt, double* Qxx, double* Qyy, 
    const double* rho, const double* e, Mesh* mesh, double* reduce_array, 
    const int first_step, const double* celldx, const double* celldy)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int jj = (gid % (nx));
  const int ii = (gid / (nx));

  const int tid = threadIdx.x;

  __shared__ double sdata[NTHREADS];
  sdata[tid] = max_dt;

  if(ii >= PAD && ii < ny-PAD && jj >= PAD && jj < nx-PAD) {
    // Constrain based on the sound speed within the system
    const double c_s = sqrt(GAM*(GAM - 1.0)*e[(ii*nx+jj)]);
    const double thread_min_dt_x = celldx[jj]/sqrt(c_s*c_s + 2.0*Qxx[(ii*nx+jj)]/rho[(ii*nx+jj)]);
    const double thread_min_dt_y = celldy[ii]/sqrt(c_s*c_s + 2.0*Qyy[(ii*nx+jj)]/rho[(ii*nx+jj)]);
    sdata[tid] = min(thread_min_dt_x, thread_min_dt_y);
  }
  __syncthreads();

  min_reduce_in_shared<double, NTHREADS>(tid, sdata);
  if (tid == 0) 
    reduce_array[blockIdx.x] = sdata[0];
}

__global__ void calc_mass_sum(
    const int nx, const int ny, const double* rho, double* reduce_array)
{
  const int gid = threadIdx.x+blockIdx.x*blockDim.x;
  const int jj = (gid % (nx));
  const int ii = (gid / (nx));
  const int tid = threadIdx.x;

  __shared__ double sdata[NTHREADS];
  sdata[tid] = 0.0;

  if(ii >= PAD && ii < ny-PAD && jj >= PAD && jj < nx-PAD) {
    sdata[tid] = rho[(ii*nx+jj)];
  }

  __syncthreads();

  sum_reduce_in_shared<double, NTHREADS>(tid, sdata);
  if (tid == 0) 
    reduce_array[blockIdx.x] = sdata[0];
}

