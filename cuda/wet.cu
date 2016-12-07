#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "../wet.h"
#include "../../comms.h"

#define NBLOCKS 128

#define set_cuda_indices(padx) \
  const int gid = threadIdx.x+blockIdx.x*blockDim.x; \
const int jj = (gid % (nx+padx));\
const int ii = (gid / (nx+padx));

#define ind0 (ii*nx + jj)
#define ind1 (ii*(nx+1) + jj)

// Solve a single timestep on the given mesh
void solve_hydro(
    Mesh* mesh, int tt, double* P, double* rho, double* rho_old, 
    double* e, double* u, double* v, double* rho_u, double* rho_v, 
    double* Qxx, double* Qyy, double* F_x, double* F_y, double* uF_x, 
    double* uF_y, double* vF_x, double* vF_y, double* min_timesteps)
{
  if(mesh->rank == MASTER)
    printf("dt %.12e dt_h %.12e\n", mesh->dt, mesh->dt_h);

  int nthreads_per_block = ceil(mesh->local_nx*mesh->local_ny/(double)NBLOCKS);
  equation_of_state<<<nthreads_per_block, NBLOCKS>>>(
      mesh->local_nx, mesh->local_ny, P, rho, e);

  nthreads_per_block = ceil((mesh->local_nx+1)*(mesh->local_ny+1)/(double)NBLOCKS);
  pressure_acceleration<<<nthreads_per_block, NBLOCKS>>>(
      mesh->local_nx, mesh->local_ny, mesh, mesh->dt, rho_u, rho_v, 
      u, v, P, rho, mesh->edgedx, mesh->edgedy, 
      mesh->celldx, mesh->celldy);

  handle_boundary(mesh->local_nx+1, mesh->local_ny, mesh, u, INVERT_X, PACK);
  handle_boundary(mesh->local_nx, mesh->local_ny+1, mesh, v, INVERT_Y, PACK);

  artificial_viscosity(
      mesh->local_nx, mesh->local_ny, mesh, mesh->dt, Qxx, Qyy, 
      u, v, rho_u, rho_v, rho, 
      mesh->edgedx, mesh->edgedy, mesh->celldx, mesh->celldy);

  nthreads_per_block = ceil(mesh->local_nx*mesh->local_ny/(double)NBLOCKS);
  shock_heating_and_work<<<nthreads_per_block, NBLOCKS>>>(
      mesh->local_nx, mesh->local_ny, mesh, mesh->dt_h, e, P, u, 
      v, rho, Qxx, Qyy, mesh->celldx, mesh->celldy);

  handle_boundary(mesh->local_nx, mesh->local_ny, mesh, e, NO_INVERT, PACK);

  set_timestep(
      mesh->local_nx, mesh->local_ny, Qxx, Qyy, rho, 
      e, mesh, min_timesteps, tt == 0, mesh->celldx, mesh->celldy);

  // Perform advection
  advect_mass_and_energy(
      mesh->local_nx, mesh->local_ny, mesh, tt, mesh->dt, mesh->dt_h, rho, e, rho_old, F_x, F_y, 
      uF_x, uF_y, u, v, mesh->edgedx, mesh->edgedy, mesh->celldx, mesh->celldy);

  advect_momentum(
      mesh->local_nx, mesh->local_ny, tt, mesh, mesh->dt_h, mesh->dt, u, v, 
      uF_x, uF_y, vF_x, vF_y, rho_u, rho_v, rho, F_x, F_y, 
      mesh->edgedx, mesh->edgedy, mesh->celldx, mesh->celldy);
}

// Calculate the pressure from GAMma law equation of state
__global__ void equation_of_state(
    const int nx, const int ny, double* P, const double* rho, const double* e)
{
  set_cuda_indices(0);

  // Only invoke simple GAMma law at the moment
  P[ind0] = (GAM - 1.0)*rho[ind0]*e[ind0];
}

// Calculate change in momentum caused by pressure gradients, and then extract
// the velocities using edge centered density approximations
__global__ void pressure_acceleration(
    const int nx, const int ny, Mesh* mesh, const double dt, double* rho_u, 
    double* rho_v, double* u, double* v, const double* P, const double* rho,
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  set_cuda_indices(1);

  if(ii < PAD || jj < PAD || jj >= (nx+1)-PAD || ii >= (ny+1)-PAD) 
    return;

  // Update the momenta using the pressure gradients
  rho_u[ind1] -= dt*(P[ind0] - P[ind0-1])/edgedx[jj];
  rho_v[ind0] -= dt*(P[ind0] - P[ind0-nx])/edgedy[ii];

  // Calculate the zone edge centered density
  const double rho_edge_x = 
    (rho[ind0]*celldx[jj]*celldy[ii] + rho[ind0-1]*celldx[jj - 1]*celldy[ii])/ 
    (2.0*edgedx[jj]*celldy[ii]);
  const double rho_edge_y = 
    (rho[ind0]*celldx[jj]*celldy[ii] + rho[ind0-nx]*celldx[jj]*celldy[ii - 1])/ 
    (2.0*celldx[jj]*edgedy[ii]);

  // Find the velocities from the momenta and edge centered mass densities
  u[ind1] = (rho_edge_x == 0.0) ? 0.0 : rho_u[ind1] / rho_edge_x;
  v[ind0] = (rho_edge_y == 0.0) ? 0.0 : rho_v[ind0] / rho_edge_y;
}

void artificial_viscosity(
    const int nx, const int ny, Mesh* mesh, const double dt, double* Qxx, 
    double* Qyy, double* u, double* v, double* rho_u, double* rho_v, const double* rho, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  int nthreads_per_block = ceil(nx*ny/(double)NBLOCKS);
  calc_viscous_stresses<<<nthreads_per_block, NBLOCKS>>>(
      nx, ny, mesh, dt, Qxx, Qyy, u, v, rho_u, rho_v, rho, 
      edgedx, edgedy, celldx, celldy);

  handle_boundary(nx, ny, mesh, Qxx, NO_INVERT, PACK);
  handle_boundary(nx, ny, mesh, Qyy, NO_INVERT, PACK);

  nthreads_per_block = ceil((nx+1)*(ny+1)/(double)NBLOCKS);
  viscous_acceleration<<<nthreads_per_block, NBLOCKS>>>(
      nx, ny, mesh, dt, Qxx, Qyy, u, v, rho_u, rho_v, rho, 
      edgedx, edgedy, celldx, celldy);

  handle_boundary(nx+1, ny, mesh, u, INVERT_X, PACK);
  handle_boundary(nx, ny+1, mesh, v, INVERT_Y, PACK);
}

__global__ void calc_viscous_stresses(
    const int nx, const int ny, Mesh* mesh, const double dt, double* Qxx, 
    double* Qyy, double* u, double* v, double* rho_u, double* rho_v, const double* rho, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  set_cuda_indices(0);

  if(ii < PAD || jj < PAD || jj >= nx-PAD || ii >= ny-PAD) 
    return;

  const double u_i = min(0.0, u[ind1+1] - u[ind1]);
  const double u_ii = 0.5*(
      fabs(min(0.0, (u[ind1+2]-u[ind1+1])) - min(0.0, (u[ind1+1]-u[ind1]))) + 
      fabs(min(0.0, (u[ind1+1]-u[ind1])) - min(0.0, (u[ind1]-u[ind1-1]))));
  const double v_i = min(0.0, v[ind0+nx] - v[ind0]);
  const double v_ii = 0.5*(
      fabs(min(0.0, (v[ind0+2*nx]-v[ind0+nx])) - min(0.0, (v[ind0+nx]-v[ind0]))) + 
      fabs(min(0.0, (v[ind0+nx]-v[ind0])) - min(0.0, (v[ind0]-v[ind0-nx]))));
  Qxx[ind0] = -C_Q*rho[ind0]*u_i*u_ii;
  Qyy[ind0] = -C_Q*rho[ind0]*v_i*v_ii;
}

__global__ void viscous_acceleration(
    const int nx, const int ny, Mesh* mesh, const double dt, double* Qxx, 
    double* Qyy, double* u, double* v, double* rho_u, double* rho_v, const double* rho, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  set_cuda_indices(1);

  if(ii < PAD || jj < PAD || jj >= (nx+1)-PAD || ii >= (ny+1)-PAD) 
    return;

  rho_u[ind1] -= dt*(Qxx[ind0] - Qxx[ind0-1])/celldx[jj];
  rho_v[ind0] -= dt*(Qyy[ind0] - Qyy[ind0-nx])/celldy[ii];

  // Calculate the zone edge centered density
  const double rho_edge_x = 
    (rho[ind0]*celldx[jj]*celldy[ii] + rho[ind0-1]*celldx[jj-1]*celldy[ii]) / 
    (2.0*edgedx[jj]*celldy[ii]);
  const double rho_edge_y = 
    (rho[ind0]*celldx[jj]*celldy[ii] + rho[ind0-nx]*celldx[jj]*celldy[ii-1]) / 
    (2.0*celldx[jj]*edgedy[ii]);

  // Find the velocities from the momenta and edge centered mass densities
  u[ind1] = (rho_edge_x == 0.0) ? 0.0 : rho_u[ind1] / rho_edge_x;
  v[ind0] = (rho_edge_y == 0.0) ? 0.0 : rho_v[ind0] / rho_edge_y;
}

// Calculates the work done due to forces within the element
__global__ void shock_heating_and_work(
    const int nx, const int ny, Mesh* mesh, const double dt_h, double* e, 
    const double* P, const double* u, const double* v, const double* rho, 
    const double* Qxx, const double* Qyy, const double* celldx, const double* celldy)
{
  set_cuda_indices(0);

  if(ii < PAD || jj < PAD || jj >= nx-PAD || ii >= ny-PAD) 
    return;

  const double div_vel_x = (u[ind1+1] - u[ind1])/celldx[jj];
  const double div_vel_y = (v[ind0+nx] - v[ind0])/celldy[ii];
  const double div_vel_dt = (div_vel_x + div_vel_y)*dt_h;
  const double e_q = e[ind0] - dt_h*(Qxx[ind0]*div_vel_x + Qyy[ind0]*div_vel_y)/rho[ind0];

  /// A working formulation that is second order in time for Pressure!?
  const double rho_c = rho[ind0]/(1.0 + div_vel_dt);
  const double e_c = e_q - (P[ind0]*div_vel_dt)/rho[ind0];
  const double work = 0.5*div_vel_dt*(P[ind0] + (GAM-1.0)*e_c*rho_c)/rho[ind0];
  e[ind0] = (rho[ind0] == 0.0) ? 0.0 : e_q-work;
}

// Calculates the timestep from the current state
void set_timestep(
    const int nx, const int ny, double* Qxx, double* Qyy, const double* rho, 
    const double* e, Mesh* mesh, double* min_timesteps, const int first_step,
    const double* celldx, const double* celldy)
{
  double local_min_dt = MAX_DT;

  int nthreads_per_block = ceil((nx+1)*(ny+1)/(double)NBLOCKS);
  calc_min_timestep<NBLOCKS><<<nthreads_per_block, NBLOCKS>>>(
      nx, ny, Qxx, Qyy, rho, e, mesh, min_timesteps, first_step, celldx, celldy);

  // TODO: This is not right, it doesn't reduce all values
  while(nthreads_per_block > 1) {
    nthreads_per_block = ceil(nthreads_per_block/(double)NBLOCKS);
    min_reduce<NBLOCKS><<<nthreads_per_block, NBLOCKS>>>(min_timesteps, min_timesteps);
  }

  sync_data(1, min_timesteps, &local_min_dt, RECV);

  // Ensure that the timestep does not jump too far from one step to the next
  double global_min_dt = reduce_all_min(local_min_dt);
  const double final_min_dt = min(global_min_dt, C_M*mesh->dt_h);
  mesh->dt = 0.5*(C_T*final_min_dt + mesh->dt_h);
  mesh->dt_h = (first_step) ? mesh->dt : C_T*final_min_dt;
}

  template <unsigned int block_size>
__global__ void calc_min_timestep(
    const int nx, const int ny, double* Qxx, double* Qyy, const double* rho, 
    const double* e, Mesh* mesh, double* min_timesteps, const int first_step,
    const double* celldx, const double* celldy)
{
  set_cuda_indices(0);

  // Constrain based on the sound speed within the system
  const double c_s = sqrt(GAM*(GAM - 1.0)*e[ind0]);
  const double thread_min_dt_x = celldx[jj]/sqrt(c_s*c_s + 2.0*Qxx[ind0]/rho[ind0]);
  const double thread_min_dt_y = celldy[ii]/sqrt(c_s*c_s + 2.0*Qyy[ind0]/rho[ind0]);
  const double thread_min_dt = min(thread_min_dt_x, thread_min_dt_y);

  __shared__ double sdata[block_size];
  const int tid = threadIdx.x;
  sdata[tid] = min(MAX_DT, thread_min_dt);
  __syncthreads();

  min_reduce_in_shared<NBLOCKS>(tid, sdata);
  if (tid == 0) min_timesteps[blockIdx.x] = sdata[0];
}

// Perform advection with monotonicity improvement
void advect_mass_and_energy(
    const int nx, const int ny, Mesh* mesh, const int tt, const double dt,
    const double dt_h, double* rho, double* e, double* rho_old, double* F_x, double* F_y, 
    double* eF_x, double* eF_y, const double* u, const double* v, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  int nthreads_per_block = ceil(nx*ny/(double)NBLOCKS);
  store_old_rho<<<nthreads_per_block, NBLOCKS>>>(nx, ny, rho, rho_old);

  if(tt % 2 == 0) {
    mass_and_energy_x_advection(
        nx, ny, 1, mesh, dt, dt_h, rho, rho_old, e, u, F_x, eF_x, 
        celldx, edgedx, celldy, edgedy);
    mass_and_energy_y_advection(
        nx, ny, 0, mesh, dt, dt_h, rho, rho_old, e, v, F_y, eF_y, 
        celldx, edgedx, celldy, edgedy);
  }
  else {
    mass_and_energy_y_advection(
        nx, ny, 1, mesh, dt, dt_h, rho, rho_old, e, v, F_y, eF_y, 
        celldx, edgedx, celldy, edgedy);
    mass_and_energy_x_advection(
        nx, ny, 0, mesh, dt, dt_h, rho, rho_old, e, u, F_x, eF_x, 
        celldx, edgedx, celldy, edgedy);
  }
}

__global__ void store_old_rho(
    const int nx, const int ny, double* rho, double* rho_old)
{
  set_cuda_indices(0);

  if(ii < PAD || jj < PAD || jj >= nx-PAD || ii >= ny-PAD) 
    return;

  rho_old[ind0] = rho[ind0];
}

// Calculate the flux in the x direction
__global__ void calc_x_mass_and_energy_flux(
    const int nx, const int ny, const int first, Mesh* mesh, const double dt, 
    const double dt_h, double* rho, double* rho_old, double* e, const double* u, 
    double* F_x, double* eF_x, const double* celldx, const double* edgedx, 
    const double* celldy, const double* edgedy)
{
  set_cuda_indices(1);

  if(ii < PAD || jj < PAD || jj >= (nx+1)-PAD || ii >= ny-PAD) 
    return;

  // Interpolate to make second order in time
  const double invdx = 1.0/edgedx[jj];
  const double suc0 = 0.5*invdx*(u[ind1+1]-u[ind1-1]);
  const double sur0 = 2.0*invdx*(u[ind1]-u[ind1-1]);
  const double sul0 = 2.0*invdx*(u[ind1+1]-u[ind1]);
  const double u_tc = u[ind1] - 0.5*u[ind1]*dt*minmod(suc0, minmod(sur0, sul0));

  // Van leer limiter
  double limiter = 0.0;
  const double rho_diff = (rho[ind0]-rho[ind0-1]);
  if(rho_diff) {
    const double smoothness = (u_tc >= 0.0) 
      ? (rho[ind0-1]-rho[ind0-2])/rho_diff
      : (rho[ind0+1]-rho[ind0])/rho_diff;
    limiter = (smoothness + fabs(smoothness))/(1.0+fabs(smoothness));
  }

  // Calculate the flux
  const double rho_upwind = (u_tc >= 0.0) ? rho[ind0-1] : rho[ind0];
  F_x[ind1] = (u_tc*rho_upwind+
      0.5*fabs(u_tc)*(1.0-fabs((u_tc*dt_h)/celldx[jj]))*limiter*rho_diff);

  // Use MC limiter to get slope of energy
  const double a_x_0 = 0.5*invdx*(e[ind0]-e[ind0-2]);
  const double b_x_0 = 2.0*invdx*(e[ind0-1]-e[ind0-2]);
  const double c_x_0 = 2.0*invdx*(e[ind0]-e[ind0-1]);
  const double a_x_1 = 0.5*invdx*(e[ind0+1]-e[ind0-1]);
  const double b_x_1 = 2.0*invdx*(e[ind0]-e[ind0-1]);
  const double c_x_1 = 2.0*invdx*(e[ind0+1]-e[ind0]);

  // Calculate the interpolated densities
  const double edge_e_x = (u_tc > 0.0)
    ? e[ind0-1] + 0.5*minmod(minmod(a_x_0, b_x_0), c_x_0)*(celldx[jj-1] - u_tc*dt_h)
    : e[ind0] - 0.5*minmod(minmod(a_x_1, b_x_1), c_x_1)*(celldx[jj] + u_tc*dt_h);

  // Update the fluxes to now include the contribution from energy
  eF_x[ind1] = edgedy[ii]*edge_e_x*F_x[ind1]; 
}

// Advect mass and energy in the x direction
__global__ void advect_mass_and_energy_in_x(
    const int nx, const int ny, const int first, Mesh* mesh, const double dt, 
    const double dt_h, double* rho, double* rho_old, double* e, const double* u, 
    double* F_x, double* eF_x, const double* celldx, const double* edgedx, 
    const double* celldy, const double* edgedy)
{
  set_cuda_indices(0);

  if(ii < PAD || jj < PAD || jj >= nx-PAD || ii >= ny-PAD) 
    return;

  rho[ind0] -= dt_h*
    (edgedy[ii+1]*F_x[ind1+1] - edgedy[ii]*F_x[ind1])/ 
    (celldx[jj]*celldy[ii]);
  const double rho_e = (rho_old[ind0]*e[ind0] - 
      (dt_h*(eF_x[ind1+1] - eF_x[ind1]))/(celldx[jj]*celldy[ii]));
  e[ind0] = (first) 
    ? (rho_old[ind0] == 0.0) ? 0.0 : rho_e/rho_old[ind0]
    : (rho[ind0] == 0.0) ? 0.0 : rho_e/rho[ind0];
}

// Advect energy and mass in the y direction
void mass_and_energy_y_advection(
    const int nx, const int ny, const int first, Mesh* mesh, const double dt,
    const double dt_h, double* rho, double* rho_old, double* e, const double* v, 
    double* F_y, double* eF_y, const double* celldx, const double* edgedx, 
    const double* celldy, const double* edgedy)
{
  int nthreads_per_block = ceil(nx*(ny+1)/(double)NBLOCKS);
  calc_y_mass_and_energy_flux<<<nthreads_per_block, NBLOCKS>>>(
      nx, ny, first, mesh, dt, dt_h, rho, rho_old, e, v, 
      F_y, eF_y, celldx, edgedx, celldy, edgedy);

  handle_boundary(nx, ny+1, mesh, F_y, INVERT_Y, PACK);

  nthreads_per_block = ceil(nx*ny/(double)NBLOCKS);
  advect_mass_and_energy_in_y<<<nthreads_per_block, NBLOCKS>>>(
      nx, ny, first, mesh, dt, dt_h, rho, rho_old, e, v, 
      F_y, eF_y, celldx, edgedx, celldy, edgedy);

  handle_boundary(nx, ny, mesh, rho, NO_INVERT, PACK);
  handle_boundary(nx, ny, mesh, e, NO_INVERT, PACK);
}

__global__ void calc_y_mass_and_energy_flux(
    const int nx, const int ny, const int first, Mesh* mesh, const double dt,
    const double dt_h, double* rho, double* rho_old, double* e, const double* v, 
    double* F_y, double* eF_y, const double* celldx, const double* edgedx, 
    const double* celldy, const double* edgedy)
{
  set_cuda_indices(0);

  if(ii < PAD || jj < PAD || jj >= nx-PAD || ii >= (ny+1)-PAD) 
    return;

  // Compute the mass flux along the y edges
  // In the ghost cells flux is left as 0.0
  // Interpolate the velocity to make second order in time
  const double invdy = 1.0/edgedy[ii];
  const double svc0 = 0.5*invdy*(v[ind0+nx]-v[ind0-nx]);
  const double svr0 = 2.0*invdy*(v[ind0]-v[ind0-nx]);
  const double svl0 = 2.0*invdy*(v[ind0+nx]-v[ind0]);
  const double v_tc = v[ind0] - 0.5*v[ind0]*dt*minmod(svc0, minmod(svr0, svl0));

  // Van leer limiter
  const double rho_diff = (rho[ind0]-rho[ind0-nx]);
  double limiter = 0.0;
  if(rho_diff) {
    const double smoothness = (v[ind0] >= 0.0) 
      ? (rho[ind0-nx]-rho[ind0-2*nx])/rho_diff
      : (rho[ind0+nx]-rho[ind0])/rho_diff;
    limiter = (smoothness + fabs(smoothness))/(1.0+fabs(smoothness));
  }

  // Calculate the flux
  const double rho_upwind = (v_tc >= 0.0) ? rho[ind0-nx] : rho[ind0];
  F_y[ind0] = (v_tc*rho_upwind+
      0.5*fabs(v_tc)*(1.0-fabs((v_tc*dt_h)/celldy[ii]))*limiter*rho_diff);

  // Use MC limiter to get slope of energy
  const double a_y_0 = 0.5*invdy*(e[ind0]-e[ind0-2*nx]);
  const double b_y_0 = 2.0*invdy*(e[ind0-nx]-e[ind0-2*nx]);
  const double c_y_0 = 2.0*invdy*(e[ind0]-e[ind0-nx]);
  const double a_y_1 = 0.5*invdy*(e[ind0+nx]-e[ind0-nx]);
  const double b_y_1 = 2.0*invdy*(e[ind0]-e[ind0-nx]);
  const double c_y_1 = 2.0*invdy*(e[ind0+nx]-e[ind0]);

  const double edge_e_y = (v_tc > 0.0)
    ? e[ind0-nx] + 0.5*minmod(minmod(a_y_0, b_y_0), c_y_0)*(celldy[ii-1] - v_tc*dt_h)
    : e[ind0] - 0.5*minmod(minmod(a_y_1, b_y_1), c_y_1)*(celldy[ii] + v_tc*dt_h);

  // Update the fluxes to now include the contribution from energy
  eF_y[ind0] = edgedx[jj]*edge_e_y*F_y[ind0]; 
}

__global__ void advect_mass_and_energy_in_y(
    const int nx, const int ny, const int first, Mesh* mesh, const double dt,
    const double dt_h, double* rho, double* rho_old, double* e, const double* v, 
    double* F_y, double* eF_y, const double* celldx, const double* edgedx, 
    const double* celldy, const double* edgedy)
{
  set_cuda_indices(0);

  if(ii < PAD || jj < PAD || jj >= nx-PAD || ii >= ny-PAD) 
    return;

  rho[ind0] -= dt_h*
    (edgedx[jj+1]*F_y[ind0+nx] - edgedx[jj]*F_y[ind0])/
    (celldx[jj]*celldy[ii]);
  const double rho_e = (rho_old[ind0]*e[ind0] - 
      (dt_h*(eF_y[ind0+nx] - eF_y[ind0]))/(celldx[jj]*celldy[ii]));
  e[ind0] = (first) 
    ? (rho_old[ind0] == 0.0) ? 0.0 : rho_e/rho_old[ind0]
    : (rho[ind0] == 0.0) ? 0.0 : rho_e/rho[ind0];
}

// Advect energy and mass in the x direction
void mass_and_energy_x_advection(
    const int nx, const int ny, const int first, Mesh* mesh, const double dt, 
    const double dt_h, double* rho, double* rho_old, double* e, const double* u, 
    double* F_x, double* eF_x, const double* celldx, const double* edgedx, 
    const double* celldy, const double* edgedy)
{
  int nthreads_per_block = ceil((nx+1)*ny/(double)NBLOCKS);
  calc_x_mass_and_energy_flux<<<nthreads_per_block, NBLOCKS>>>(
      nx, ny, first, mesh, dt, dt_h, rho, rho_old, e, u, 
      F_x, eF_x, celldx, edgedx, celldy, edgedy);

  handle_boundary(nx+1, ny, mesh, F_x, INVERT_X, PACK);

  nthreads_per_block = ceil(nx*ny/(double)NBLOCKS);
  advect_mass_and_energy_in_x<<<nthreads_per_block, NBLOCKS>>>(
      nx, ny, first, mesh, dt, dt_h, rho, rho_old, e, u, 
      F_x, eF_x, celldx, edgedx, celldy, edgedy);

  handle_boundary(nx, ny, mesh, rho, NO_INVERT, PACK);
  handle_boundary(nx, ny, mesh, e, NO_INVERT, PACK);
}

__global__ void ux_momentum_flux(
    const int nx, const int ny, Mesh* mesh, const double dt_h, const double dt, 
    double* u, double* v, double* uF_x, double* rho_u, const double* rho, const double* F_x, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  set_cuda_indices(0);

  if(ii < PAD || jj < PAD || jj >= nx-PAD || ii >= ny-PAD) 
    return;

  // Calculate the cell centered x momentum fluxes in the x direction
  // Use MC limiter to get slope of velocity
  const double invdx = 1.0/edgedx[jj];
  const double a_x_0 = 0.5*invdx*(u[ind1+1]-u[ind1-1]);
  const double b_x_0 = 2.0*invdx*(u[ind1]-u[ind1-1]);
  const double c_x_0 = 2.0*invdx*(u[ind1+1]-u[ind1]);
  const double a_x_1 = 0.5*invdx*(u[ind1+2]-u[ind1]);
  const double b_x_1 = 2.0*invdx*(u[ind1+1]-u[ind1]);
  const double c_x_1 = 2.0*invdx*(u[ind1+2]-u[ind1+1]);

  // Calculate the interpolated densities
  const double u_cell_x = 0.5*(u[ind1]+u[ind1+1]);
  const double f_x = edgedy[ii]*0.5*(F_x[ind1] + F_x[ind1+1]); 
  const double u_cell_x_interp = (u_cell_x > 0.0)
    ? u[ind1] + 0.5*minmod(minmod(a_x_0, b_x_0), c_x_0)*(celldx[jj-1] - u_cell_x*dt_h)
    : u[ind1+1] - 0.5*minmod(minmod(a_x_1, b_x_1), c_x_1)*(celldx[jj] + u_cell_x*dt_h);
  uF_x[ind0] = f_x*u_cell_x_interp;
}


__global__ void advect_rho_u_in_x(
    const int nx, const int ny, const int tt, Mesh* mesh, const double dt_h, 
    const double dt, double* u, double* v, double* uF_x, double* uF_y, 
    double* vF_x, double* vF_y, double* rho_u, double* rho_v, 
    const double* rho, const double* F_x, const double* F_y, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  set_cuda_indices(1);

  if(ii < PAD || jj < PAD || jj >= (nx+1)-PAD || ii >= ny-PAD) 
    return;

  rho_u[ind1] -= dt_h*(uF_x[ind0] - uF_x[ind0-1])/(edgedx[jj]*celldy[ii]);
}

__global__ void advect_rho_u_and_u_in_x(
    const int nx, const int ny, const int tt, Mesh* mesh, const double dt_h, 
    const double dt, double* u, double* v, double* uF_x, double* uF_y, 
    double* vF_x, double* vF_y, double* rho_u, double* rho_v, 
    const double* rho, const double* F_x, const double* F_y, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  set_cuda_indices(1);

  if(ii < PAD || jj < PAD || jj >= (nx+1)-PAD || ii >= ny-PAD) 
    return;

  rho_u[ind1] -= dt_h*(uF_x[ind0] - uF_x[ind0-1])/(edgedx[jj]*celldy[ii]);
  const double rho_edge_x = 
    (rho[ind0]*celldx[jj]*celldy[ii] + rho[ind0-1]*celldx[jj - 1]*celldy[ii])/ 
    (2.0*edgedx[jj]*celldy[ii]);
  u[ind1] = (rho_edge_x == 0.0) ? 0.0 : rho_u[ind1] / rho_edge_x;
}

__global__ void advect_rho_u_in_y(
    const int nx, const int ny, const int tt, Mesh* mesh, const double dt_h, 
    const double dt, double* u, double* v, double* uF_x, double* uF_y, 
    double* vF_x, double* vF_y, double* rho_u, double* rho_v, 
    const double* rho, const double* F_x, const double* F_y, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  set_cuda_indices(1);

  if(ii < PAD || jj < PAD || jj >= (nx+1)-PAD || ii >= ny-PAD) 
    return;

  rho_u[ind1] -= dt_h*(uF_y[ind1+(nx+1)] - uF_y[ind1])/(celldx[jj]*edgedy[ii]);
}

__global__ void advect_rho_u_and_u_in_y(
    const int nx, const int ny, const int tt, Mesh* mesh, const double dt_h, 
    const double dt, double* u, double* v, double* uF_x, double* uF_y, 
    double* vF_x, double* vF_y, double* rho_u, double* rho_v, 
    const double* rho, const double* F_x, const double* F_y, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  set_cuda_indices(1);

  if(ii < PAD || jj < PAD || jj >= (nx+1)-PAD || ii >= ny-PAD) 
    return;

  rho_u[ind1] -= dt_h*(uF_y[ind1+(nx+1)] - uF_y[ind1])/(celldx[jj]*edgedy[ii]);
  const double rho_edge_x = 
    (rho[ind0]*celldx[jj]*celldy[ii] + rho[ind0-1]*celldx[jj - 1]*celldy[ii])/ 
    (2.0*edgedx[jj]*celldy[ii]);
  u[ind1] = (rho_edge_x == 0.0) ? 0.0 : rho_u[ind1] / rho_edge_x;
}

__global__ void uy_momentum_flux(
    const int nx, const int ny, Mesh* mesh, const double dt_h, const double dt, 
    double* u, double* v, double* uF_y, double* rho_u, const double* rho, 
    const double* F_y, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  set_cuda_indices(1);

  if(ii < PAD || jj < PAD || jj >= (nx+1)-PAD || ii >= (ny+1)-PAD) 
    return;

  // Use MC limiter to get slope of velocity
  const double invdy = 1.0/edgedy[ii];
  const double a_y_0 = 0.5*invdy*(u[ind1]-u[ind1-2*(nx+1)]);
  const double b_y_0 = 2.0*invdy*(u[ind1-(nx+1)]-u[ind1-2*(nx+1)]);
  const double c_y_0 = 2.0*invdy*(u[ind1]-u[ind1-(nx+1)]);
  const double a_y_1 = 0.5*invdy*(u[ind1+(nx+1)]-u[ind1-(nx+1)]);
  const double b_y_1 = 2.0*invdy*(u[ind1]-u[ind1-(nx+1)]);
  const double c_y_1 = 2.0*invdy*(u[ind1+(nx+1)]-u[ind1]);
  const double v_cell_y = 0.5*(v[ind0-1]+v[ind0]);

  const double f_y = edgedx[jj]*0.5*(F_y[ind0] + F_y[ind0-1]);
  const double u_corner_y = (v_cell_y > 0.0)
    ? u[ind1-(nx+1)] + 0.5*minmod(minmod(a_y_0, b_y_0), c_y_0)*(celldy[ii-1] - v_cell_y*dt_h)
    : u[ind1] - 0.5*minmod(minmod(a_y_1, b_y_1), c_y_1)*(celldy[ii] + v_cell_y*dt_h);
  uF_y[ind1] = f_y*u_corner_y;
}

__global__ void vx_momentum_flux(
    const int nx, const int ny, Mesh* mesh, const double dt_h, const double dt, 
    const double* u, double* v, double* vF_x, double* rho_v, const double* rho, 
    const double* F_x, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  set_cuda_indices(1);

  if(ii < PAD || jj < PAD || jj >= (nx+1)-PAD || ii >= (ny+1)-PAD) 
    return;

  // Use MC limiter to get slope of velocity
  const double invdx = 1.0/edgedx[jj];
  const double a_x_0 = 0.5*invdx*(v[ind0]-v[ind0-2]);
  const double b_x_0 = 2.0*invdx*(v[ind0-1]-v[ind0-2]);
  const double c_x_0 = 2.0*invdx*(v[ind0]-v[ind0-1]);
  const double a_x_1 = 0.5*invdx*(v[ind0+1]-v[ind0-1]);
  const double b_x_1 = 2.0*invdx*(v[ind0]-v[ind0-1]);
  const double c_x_1 = 2.0*invdx*(v[ind0+1]-v[ind0]);

  // Calculate the interpolated densities
  const double f_x = celldy[ii]*0.5*(F_x[ind1] + F_x[ind1-(nx+1)]);
  const double u_cell_x = 0.5*(u[ind1]+u[ind1-(nx+1)]);
  const double v_cell_x_interp = (u_cell_x > 0.0)
    ? v[ind0-1] + 0.5*minmod(minmod(a_x_0, b_x_0), c_x_0)*(celldx[jj-1] - u_cell_x*dt_h)
    : v[ind0] - 0.5*minmod(minmod(a_x_1, b_x_1), c_x_1)*(celldx[jj] + u_cell_x*dt_h);
  vF_x[ind1] = f_x*v_cell_x_interp;
}

__global__ void advect_rho_v_and_v_in_y(
    const int nx, const int ny, Mesh* mesh, const double dt_h, const double dt, 
    double* u, double* v, double* vF_y, double* rho_v, const double* rho, const double* F_y, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  set_cuda_indices(0);

  if(ii < PAD || jj < PAD || jj >= nx-PAD || ii >= (ny+1)-PAD) 
    return;

  rho_v[ind0] -= dt_h*(vF_y[ind0] - vF_y[ind0-nx])/(celldx[jj]*edgedy[ii]);
  const double rho_edge_y = 
    (rho[ind0]*celldx[jj]*celldy[ii] + rho[ind0-nx]*celldx[jj]*celldy[ii - 1])/ 
    (2.0*celldx[jj]*edgedy[ii]);
  v[ind0] = (rho_edge_y == 0.0) ? 0.0 : rho_v[ind0] / rho_edge_y;
}

__global__ void advect_rho_v_and_v_in_x(
    const int nx, const int ny, Mesh* mesh, const double dt_h, const double dt, 
    const double* u, double* v, double* vF_x, double* rho_v, const double* rho, 
    const double* F_x, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  set_cuda_indices(0);

  if(ii < PAD || jj < PAD || jj >= nx-PAD || ii >= (ny+1)-PAD) 
    return;

  rho_v[ind0] -= dt_h*(vF_x[ind1+1] - vF_x[ind1])/(edgedx[jj]*celldy[ii]);
  const double rho_edge_y = 
    (rho[ind0]*celldx[jj]*celldy[ii] + rho[ind0-nx]*celldx[jj]*celldy[ii - 1])/ 
    (2.0*celldx[jj]*edgedy[ii]);
  v[ind0] = (rho_edge_y == 0.0) ? 0.0 : rho_v[ind0] / rho_edge_y;
}

__global__ void advect_rho_v_in_x(
    const int nx, const int ny, Mesh* mesh, const double dt_h, const double dt, 
    const double* u, double* v, double* vF_x, double* rho_v, const double* rho, 
    const double* F_x, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  set_cuda_indices(0);

  if(ii < PAD || jj < PAD || jj >= nx-PAD || ii >= (ny+1)-PAD) 
    return;

  rho_v[ind0] -= dt_h*(vF_x[ind1+1] - vF_x[ind1])/(edgedx[jj]*celldy[ii]);
}

__global__ void vy_momentum_flux(
    const int nx, const int ny, Mesh* mesh, const double dt_h, const double dt, 
    double* u, double* v, double* vF_y, double* rho_v, const double* rho, const double* F_y, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  set_cuda_indices(0);

  if(ii < PAD || jj < PAD || jj >= nx-PAD || ii >= ny-PAD) 
    return;

  // Use MC limiter to get slope of velocity
  const double invdy = 1.0/edgedy[ii];
  const double a_y_0 = 0.5*invdy*(v[ind0+nx]-v[ind0-nx]);
  const double b_y_0 = 2.0*invdy*(v[ind0]-v[ind0-nx]);
  const double c_y_0 = 2.0*invdy*(v[ind0+nx]-v[ind0]);
  const double a_y_1 = 0.5*invdy*(v[ind0+2*nx]-v[ind0]);
  const double b_y_1 = 2.0*invdy*(v[ind0+nx]-v[ind0]);
  const double c_y_1 = 2.0*invdy*(v[ind0+2*nx]-v[ind0+nx]);

  const double f_y = celldx[jj]*0.5*(F_y[ind0] + F_y[ind0+nx]);
  const double v_cell_y = 0.5*(v[ind0]+v[ind0+nx]);
  const double v_cell_y_interp = (v_cell_y > 0.0)
    ? v[ind0] + 0.5*minmod(minmod(a_y_0, b_y_0), c_y_0)*(celldy[ii-1] - v_cell_y*dt_h)
    : v[ind0+nx] - 0.5*minmod(minmod(a_y_1, b_y_1), c_y_1)*(celldy[ii] + v_cell_y*dt_h);
  vF_y[ind0] = f_y*v_cell_y_interp;
}

__global__ void advect_rho_v_in_y(
    const int nx, const int ny, Mesh* mesh, const double dt_h, const double dt, 
    double* u, double* v, double* vF_y, double* rho_v, const double* rho, const double* F_y, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  set_cuda_indices(0);

  if(ii < PAD || jj < PAD || jj >= nx-PAD || ii >= (ny+1)-PAD) 
    return;

  rho_v[ind0] -= dt_h*(vF_y[ind0] - vF_y[ind0-nx])/(celldx[jj]*edgedy[ii]);
}

// Advect momentum according to the velocity
void advect_momentum(
    const int nx, const int ny, const int tt, Mesh* mesh, const double dt_h, 
    const double dt, double* u, double* v, double* uF_x, double* uF_y, 
    double* vF_x, double* vF_y, double* rho_u, double* rho_v, 
    const double* rho, const double* F_x, const double* F_y, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  int nthreads_per_block = 0;
  if(tt % 2) {
    nthreads_per_block = ceil(nx*ny/(double)NBLOCKS);
    ux_momentum_flux<<<nthreads_per_block, NBLOCKS>>>(
        nx, ny, mesh, dt_h, dt, u, v, uF_x, rho_u, rho, F_x, edgedx, edgedy, celldx, celldy);
    handle_boundary(nx, ny, mesh, uF_x, NO_INVERT, PACK);

    nthreads_per_block = ceil((nx+1)*ny/(double)NBLOCKS);
    advect_rho_u_and_u_in_x<<<nthreads_per_block, NBLOCKS>>>(
        nx, ny, tt, mesh, dt_h, dt, u, v, uF_x, uF_y, 
        vF_x, vF_y, rho_u, rho_v, rho, F_x, F_y, edgedx, edgedy, celldx, celldy);
    handle_boundary(nx+1, ny, mesh, u, INVERT_X, PACK);

    nthreads_per_block = ceil((nx+1)*(ny+1)/(double)NBLOCKS);
    uy_momentum_flux<<<nthreads_per_block, NBLOCKS>>>(
        nx, ny, mesh, dt_h, dt, u, v, uF_y, rho_u, rho, F_y, edgedx, edgedy, celldx, celldy);
    handle_boundary(nx+1, ny+1, mesh, uF_y, NO_INVERT, PACK);

    nthreads_per_block = ceil((nx+1)*ny/(double)NBLOCKS);
    advect_rho_u_in_y<<<nthreads_per_block, NBLOCKS>>>(
        nx, ny, tt, mesh, dt_h, dt, u, v, uF_x, uF_y, vF_x, vF_y, rho_u, rho_v, 
        rho, F_x, F_y, edgedx, edgedy, celldx, celldy);

    nthreads_per_block = ceil((nx+1)*(ny+1)/(double)NBLOCKS);
    vx_momentum_flux<<<nthreads_per_block, NBLOCKS>>>(
        nx, ny, mesh, dt_h, dt, u, v, vF_x, rho_v, rho, F_x, edgedx, edgedy, celldx, celldy);
    handle_boundary(nx+1, ny+1, mesh, vF_x, NO_INVERT, PACK);

    nthreads_per_block = ceil(nx*(ny+1)/(double)NBLOCKS);
    advect_rho_v_and_v_in_x<<<nthreads_per_block, NBLOCKS>>>(
        nx, ny, mesh, dt_h, dt, u, v, vF_x, rho_v, rho, F_x, 
        edgedx, edgedy, celldx, celldy);
    handle_boundary(nx, ny+1, mesh, v, INVERT_Y, PACK);

    nthreads_per_block = ceil(nx*ny/(double)NBLOCKS);
    vy_momentum_flux<<<nthreads_per_block, NBLOCKS>>>(
        nx, ny, mesh, dt_h, dt, u, v, vF_y, rho_v, rho, F_y, edgedx, edgedy, celldx, celldy);
    handle_boundary(nx, ny, mesh, vF_y, NO_INVERT, PACK);

    nthreads_per_block = ceil(nx*(ny+1)/(double)NBLOCKS);
    advect_rho_v_in_y<<<nthreads_per_block, NBLOCKS>>>(
        nx, ny, mesh, dt_h, dt, u, v, vF_y, rho_v, rho, F_y, 
        edgedx, edgedy, celldx, celldy);
  }
  else {
    nthreads_per_block = ceil((nx+1)*(ny+1)/(double)NBLOCKS);
    uy_momentum_flux<<<nthreads_per_block, NBLOCKS>>>(
        nx, ny, mesh, dt_h, dt, u, v, uF_y, rho_u, rho, F_y, edgedx, edgedy, celldx, celldy);
    handle_boundary(nx+1, ny+1, mesh, uF_y, NO_INVERT, PACK);

    nthreads_per_block = ceil((nx+1)*ny/(double)NBLOCKS);
    advect_rho_u_and_u_in_y<<<nthreads_per_block, NBLOCKS>>>(
        nx, ny, tt, mesh, dt_h, dt, u, v, uF_x, uF_y, vF_x, vF_y, rho_u, 
        rho_v, rho, F_x, F_y, edgedx, edgedy, celldx, celldy);
    handle_boundary(nx+1, ny, mesh, u, INVERT_X, PACK);

    nthreads_per_block = ceil(nx*ny/(double)NBLOCKS);
    ux_momentum_flux<<<nthreads_per_block, NBLOCKS>>>(
        nx, ny, mesh, dt_h, dt, u, v, uF_x, rho_u, rho, 
        F_x, edgedx, edgedy, celldx, celldy);
    handle_boundary(nx, ny, mesh, uF_x, NO_INVERT, PACK);

    nthreads_per_block = ceil((nx+1)*ny/(double)NBLOCKS);
    advect_rho_u_in_y<<<nthreads_per_block, NBLOCKS>>>(
        nx, ny, tt, mesh, dt_h, dt, u, v, uF_x, uF_y, vF_x, vF_y, rho_u, rho_v, 
        rho, F_x, F_y, edgedx, edgedy, celldx, celldy);

    nthreads_per_block = ceil(nx*ny/(double)NBLOCKS);
    vy_momentum_flux<<<nthreads_per_block, NBLOCKS>>>(
        nx, ny, mesh, dt_h, dt, u, v, vF_y, rho_v, rho, F_y, edgedx, edgedy, celldx, celldy);
    handle_boundary(nx, ny, mesh, vF_y, NO_INVERT, PACK);

    nthreads_per_block = ceil(nx*(ny+1)/(double)NBLOCKS);
    advect_rho_v_and_v_in_y<<<nthreads_per_block, NBLOCKS>>>(
        nx, ny, mesh, dt_h, dt, u, v, vF_y, rho_v, rho, F_y, 
        edgedx, edgedy, celldx, celldy);
    handle_boundary(nx, ny+1, mesh, v, INVERT_Y, PACK);

    nthreads_per_block = ceil((nx+1)*(ny+1)/(double)NBLOCKS);
    vx_momentum_flux<<<nthreads_per_block, NBLOCKS>>>(
        nx, ny, mesh, dt_h, dt, u, v, vF_x, rho_v, rho, F_x, edgedx, edgedy, celldx, celldy);

    nthreads_per_block = ceil(nx*(ny+1)/(double)NBLOCKS);
    advect_rho_v_in_x<<<nthreads_per_block, NBLOCKS>>>(
        nx, ny, mesh, dt_h, dt, u, v, vF_x, rho_v, rho, 
        F_x, edgedx, edgedy, celldx, celldy);
  }
}


// Prints some conservation values
void print_conservation(
    const int nx, const int ny, double* rho, double* e, Mesh* mesh) 
{
  double mass_tot = 0.0;
  double energy_tot = 0.0;
#pragma omp parallel for reduction(+:mass_tot, energy_tot)
  for(int ii = PAD; ii < ny-PAD; ++ii) {
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      mass_tot += rho[ind0];
      energy_tot += rho[ind0]*e[ind0];
    }
  }

  double global_mass_tot = mass_tot;
  double global_energy_tot = energy_tot;

#ifdef MPI
  MPI_Reduce(&mass_tot, &global_mass_tot, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
  MPI_Reduce(&energy_tot, &global_energy_tot, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
#endif

  if(mesh->rank == MASTER) {
    printf("total mass: %.12e\n", global_mass_tot);
    printf("total energy: %.12e\n", global_energy_tot);
  }
}

// http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
  template <unsigned int block_size>
__global__ void min_reduce(
    double* data, double* result)
{
  __shared__ double sdata[block_size];
  const int ind = blockIdx.x*block_size + threadIdx.x;
  sdata[threadIdx.x] = data[ind];
  __syncthreads();

  min_reduce_in_shared<block_size>(threadIdx.x, sdata);
  if (threadIdx.x == 0) result[blockIdx.x] = sdata[0];
}

  template <unsigned int block_size>
__device__ void min_reduce_in_shared(
    const int tid, double* sdata)
{
  if (block_size >= 512) { 
    if (tid < 256) { sdata[tid] = min(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
  if (block_size >= 256) { 
    if (tid < 128) { sdata[tid] = min(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
  if (block_size >= 128) { 
    if (tid < 64) { sdata[tid] = min(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
  if (tid < 32) {
    if (block_size >= 64) sdata[tid] = min(sdata[tid], sdata[tid + 32]);
    if (block_size >= 32) sdata[tid] = min(sdata[tid], sdata[tid + 16]);
    if (block_size >= 16) sdata[tid] = min(sdata[tid], sdata[tid + 8]);
    if (block_size >= 8) sdata[tid] = min(sdata[tid], sdata[tid + 4]);
    if (block_size >= 4) sdata[tid] = min(sdata[tid], sdata[tid + 2]);
    if (block_size >= 2) sdata[tid] = min(sdata[tid], sdata[tid + 1]);
  }
}

