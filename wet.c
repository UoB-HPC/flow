#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "wet.h"
#include "../comms.h"

#define ind0 (ii*nx + jj)
#define ind1 (ii*(nx+1) + jj)

// Solve a single timestep on the given mesh
void solve_hydro(
    Mesh* mesh, int tt, double* P, double* rho, double* rho_old, 
    double* e, double* u, double* v, double* rho_u, double* rho_v, 
    double* Qxx, double* Qyy, double* F_x, double* F_y, double* uF_x, 
    double* uF_y, double* vF_x, double* vF_y)
{
  if(mesh->rank == MASTER)
    printf("dt %.12e dt_h %.12e\n", mesh->dt, mesh->dt_h);

  equation_of_state(
      mesh->local_nx, mesh->local_ny, P, rho, e);

  lagrangian_step(
      mesh->local_nx, mesh->local_ny, mesh, mesh->dt, rho_u, rho_v, 
      u, v, P, rho, mesh->edgedx, mesh->edgedy, 
      mesh->celldx, mesh->celldy);

  artificial_viscosity(
      mesh->local_nx, mesh->local_ny, mesh, mesh->dt, Qxx, Qyy, 
      u, v, rho_u, rho_v, rho, 
      mesh->edgedx, mesh->edgedy, mesh->celldx, mesh->celldy);

  shock_heating_and_work(
      mesh->local_nx, mesh->local_ny, mesh, mesh->dt_h, e, P, u, 
      v, rho, Qxx, Qyy, mesh->celldx, mesh->celldy);

  set_timestep(
      mesh->local_nx, mesh->local_ny, Qxx, Qyy, rho, 
      e, mesh, tt == 0, mesh->celldx, mesh->celldy);

  // Perform advection
  advect_mass_and_energy(
      mesh->local_nx, mesh->local_ny, mesh, tt, mesh->dt_h, rho, rho_old, F_x, F_y, 
      u, v, mesh->edgedx, mesh->edgedy, mesh->celldx, mesh->celldy);

  advect_momentum(
      mesh->local_nx, mesh->local_ny, mesh, mesh->dt_h, mesh->dt, u, v, uF_x, 
      uF_y, vF_x, vF_y, rho_u, rho_v, 
      rho, F_x, F_y, mesh->edgedx, mesh->edgedy, mesh->celldx, mesh->celldy);

  advect_energy(
      mesh->local_nx, mesh->local_ny, mesh, mesh->dt_h, mesh->dt, e, 
      F_x, F_y, u, v, rho_old, rho,
      mesh->edgedx, mesh->edgedy, mesh->celldx, mesh->celldy);
}

// Calculate the pressure from GAMma law equation of state
void equation_of_state(
    const int nx, const int ny, double* P, const double* rho, const double* e)
{
  START_PROFILING(&compute_profile);

#pragma omp parallel for
  for(int ii = 0; ii < ny; ++ii) {
#pragma omp simd
    for(int jj = 0; jj < nx; ++jj) {
      // Only invoke simple GAMma law at the moment
      P[ind0] = (GAM - 1.0)*rho[ind0]*e[ind0];
    }
  }

  STOP_PROFILING(&compute_profile, __func__);
}

// Calculates the timestep from the current state
void set_timestep(
    const int nx, const int ny, double* Qxx, double* Qyy, const double* rho, 
    const double* e, Mesh* mesh, const int first_step,
    const double* celldx, const double* celldy)
{
  double local_min_dt = MAX_DT;

  START_PROFILING(&compute_profile);
  // Check the minimum timestep from the sound speed in the nx and ny directions
#pragma omp parallel for reduction(min: local_min_dt)
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd 
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      // Constrain based on the sound speed within the system
      const double c_s = sqrt(GAM*(GAM - 1.0)*e[ind0]);
      const double thread_min_dt_x = celldx[jj]/sqrt(c_s*c_s + 2.0*Qxx[ind0]/rho[ind0]);
      const double thread_min_dt_y = celldy[ii]/sqrt(c_s*c_s + 2.0*Qyy[ind0]/rho[ind0]);
      const double thread_min_dt = min(thread_min_dt_x, thread_min_dt_y);
      local_min_dt = min(local_min_dt, thread_min_dt);
    }
  }
  STOP_PROFILING(&compute_profile, __func__);

  double global_min_dt = reduce_all_min(local_min_dt);

  // Ensure that the timestep does not jump too far from one step to the next
  const double final_min_dt = min(global_min_dt, C_M*mesh->dt_h);
  mesh->dt = 0.5*(C_T*final_min_dt + mesh->dt_h);
  mesh->dt_h = (first_step) ? mesh->dt : C_T*final_min_dt;
}

// Calculate change in momentum caused by pressure gradients, and then extract
// the velocities using edge centered density approximations
void lagrangian_step(
    const int nx, const int ny, Mesh* mesh, const double dt, double* rho_u, 
    double* rho_v, double* u, double* v, const double* P, const double* rho,
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  START_PROFILING(&compute_profile);

#pragma omp parallel for
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
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
  }

  STOP_PROFILING(&compute_profile, __func__);

  handle_boundary(nx+1, ny, mesh, u, INVERT_X, PACK);
  handle_boundary(nx, ny+1, mesh, v, INVERT_Y, PACK);
}

void artificial_viscosity(
    const int nx, const int ny, Mesh* mesh, const double dt, double* Qxx, 
    double* Qyy, double* u, double* v, double* rho_u, double* rho_v, const double* rho, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  START_PROFILING(&compute_profile);

  // Calculate the artificial viscous stresses
  // PLPC Hydro Paper
#pragma omp parallel for 
  for(int ii = PAD-1; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD-1; jj < nx-PAD; ++jj) {
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
  }

  STOP_PROFILING(&compute_profile, __func__);

  handle_boundary(nx, ny, mesh, Qxx, NO_INVERT, PACK);
  handle_boundary(nx, ny, mesh, Qyy, NO_INVERT, PACK);

  START_PROFILING(&compute_profile);

  // Update the momenta by the artificial viscous stresses
#pragma omp parallel for
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      rho_u[ind1] -= dt*(Qxx[ind0] - Qxx[ind0-1])/edgedx[jj];
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
  }
  STOP_PROFILING(&compute_profile, __func__);

  handle_boundary(nx+1, ny, mesh, u, INVERT_X, PACK);
  handle_boundary(nx, ny+1, mesh, v, INVERT_Y, PACK);
}

// Calculates the work done due to forces within the element
void shock_heating_and_work(
    const int nx, const int ny, Mesh* mesh, const double dt_h, double* e, 
    const double* P, const double* u, const double* v, const double* rho, 
    const double* Qxx, const double* Qyy, const double* celldx, const double* celldy)
{
  START_PROFILING(&compute_profile);

#pragma omp parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      if(rho[ind0] == 0.0) {
        e[ind0] = 0.0;
        continue;
      }

      const double div_vel_x = (u[ind1+1] - u[ind1])/celldx[jj];
      const double div_vel_y = (v[ind0+nx] - v[ind0])/celldy[ii];
      const double div_vel_dt = (div_vel_x + div_vel_y)*dt_h;

      /// A working formulation that is second order in time for Pressure!?
      const double rho_c = rho[ind0]/(1.0 + div_vel_dt);
      const double e_c = e[ind0] - (P[ind0]*div_vel_dt)/rho[ind0];
      const double work = 0.5*div_vel_dt*(P[ind0] + (GAM-1.0)*e_c*rho_c)/rho[ind0];
      const double shock_heating = dt_h*(Qxx[ind0]*div_vel_x + Qyy[ind0]*div_vel_y)/rho_c;
      e[ind0] -= (work + shock_heating);
    }
  }

  STOP_PROFILING(&compute_profile, __func__);

  handle_boundary(nx, ny, mesh, e, NO_INVERT, PACK);
}

// Perform advection with monotonicity improvement
void advect_mass_and_energy(
    const int nx, const int ny, Mesh* mesh, const int tt, const double dt_h, 
    double* rho, double* rho_old, double* F_x, double* F_y, const double* u, const double* v, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
#pragma omp parallel for
  for(int ii = 0; ii < nx*ny; ++ii) {
    rho_old[ii] = rho[ii];
  }

  // Perform the dimensional splitting on the mass flux, with a the alternating
  // fix for asymmetries
  if(tt % 2) {
    x_mass_and_energy_flux(
        nx, ny, mesh, dt_h, rho, u, F_x, celldx, edgedx, celldy, edgedy);
    y_mass_and_energy_flux(
        nx, ny, mesh, dt_h, rho, v, F_y, celldx, edgedx, celldy, edgedy);
  }
  else {
    y_mass_and_energy_flux(
        nx, ny, mesh, dt_h, rho, v, F_y, celldx, edgedx, celldy, edgedy);
    x_mass_and_energy_flux(
        nx, ny, mesh, dt_h, rho, u, F_x, celldx, edgedx, celldy, edgedy);
  }

  handle_boundary(nx, ny, mesh, rho, NO_INVERT, PACK);
}

// Calculate the flux in the x direction
void x_mass_and_energy_flux(
    const int nx, const int ny, Mesh* mesh, const double dt_h, double* rho, 
    const double* u, double* F_x, const double* celldx, const double* edgedx, 
    const double* celldy, const double* edgedy)
{
  // Compute the mass fluxes along the x edges
  // In the ghost cells flux is left as 0.0
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      const double rho_diff = (rho[ind0]-rho[ind0-1]);

      // Van leer limiter
      double limiter = 0.0;
      if(rho_diff) {
        const double smoothness = (u[ind1] >= 0.0) 
          ? (rho[ind0-1]-rho[ind0-2])/rho_diff
          : (rho[ind0+1]-rho[ind0])/rho_diff;
        limiter = (smoothness + fabs(smoothness))/(1.0+fabs(smoothness));
      }

      // Calculate the flux
      const double rho_upwind = (u[ind1] >= 0.0) ? rho[ind0-1] : rho[ind0];
      F_x[ind1] = (u[ind1]*rho_upwind+
          0.5*fabs(u[ind1])*(1.0-fabs((u[ind1]*dt_h)/celldx[jj]))*limiter*rho_diff);
    }
  }
  STOP_PROFILING(&compute_profile, "advect_mass_and_energy");

  handle_boundary(nx+1, ny, mesh, F_x, INVERT_X, PACK);

  // Calculate the new density values
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      rho[ind0] -= dt_h*
        (edgedx[jj+1]*F_x[ind1+1] - edgedx[jj]*F_x[ind1])/ 
        (celldx[jj]*celldy[ii]);
    }
  }
  STOP_PROFILING(&compute_profile, "advect_mass_and_energy");
}

// Calculate the flux in the y direction
void y_mass_and_energy_flux(
    const int nx, const int ny, Mesh* mesh, const double dt_h, double* rho, 
    const double* v, double* F_y, const double* celldx, const double* edgedx, 
    const double* celldy, const double* edgedy)
{
  // Compute the mass flux along the y edges
  // In the ghost cells flux is left as 0.0
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      const double rho_diff = (rho[ind0]-rho[ind0-nx]);

      // Van leer limiter
      double limiter = 0.0;
      if(rho_diff) {
        const double smoothness = (v[ind0] >= 0.0) 
          ? (rho[ind0-nx]-rho[ind0-2*nx])/rho_diff
          : (rho[ind0+nx]-rho[ind0])/rho_diff;
        limiter = (smoothness + fabs(smoothness))/(1.0+fabs(smoothness));
      }

      // Calculate the flux
      const double rho_upwind = (v[ind0] >= 0.0) ? rho[ind0-nx] : rho[ind0];
      F_y[ind0] = (v[ind0]*rho_upwind+
          0.5*fabs(v[ind0])*(1.0-fabs((v[ind0]*dt_h)/celldx[jj]))*limiter*rho_diff);
    }
  }
  STOP_PROFILING(&compute_profile, "advect_mass_and_energy");

  handle_boundary(nx, ny+1, mesh, F_y, INVERT_Y, PACK);

  // Calculate the new density values
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      rho[ind0] -= dt_h*
        (edgedy[ii+1]*F_y[ind0+nx] - edgedy[ii]*F_y[ind0])/
        (celldx[jj]*celldy[ii]);
    }
  }
  STOP_PROFILING(&compute_profile, "advect_mass_and_energy");
}

// Advect momentum according to the velocity
void advect_momentum(
    const int nx, const int ny, Mesh* mesh, const double dt_h, const double dt, 
    double* u, double* v, double* uF_x, double* uF_y, double* vF_x, 
    double* vF_y, double* rho_u, double* rho_v, const double* rho, 
    const double* F_x, const double* F_y, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  // Calculate the cell centered x momentum fluxes in the x direction
#pragma omp parallel for
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      const double invdx = 1.0/edgedx[jj];
      const double invdy = 1.0/edgedy[ii];

      // Construct all required slopes for the stencil to interpolate the
      // velocities at the cell centers
      const double S_L_0 = invdx*(u[ind1] - u[ind1-1]);
      const double S_C_0 = invdx*(u[ind1+1] - u[ind1]);
      const double S_R_0 = invdx*(u[ind1+2] - u[ind1+1]);
      const double S_D_1 = invdy*(u[ind1-(nx+1)] - u[ind1-2*(nx+1)]);
      const double S_C_1 = invdy*(u[ind1] - u[ind1-(nx+1)]);
      const double S_U_1 = invdy*(u[ind1+(nx+1)] - u[ind1]);

      // Construct the fluxes
      const double f_x = edgedy[ii]*0.5*(F_x[ind1] + F_x[ind1+1]); 
      const double f_y = edgedx[jj]*0.5*(F_y[ind0] + F_y[ind0-1]);
      const double u_cell_x = 0.5*(u[ind1]+u[ind1+1]);
      const double v_cell_y = 0.5*(v[ind0]+v[ind0-1]);

      // Construct the fluxes from the slopes
      uF_x[ind0] = f_x*((u_cell_x >= 0.0) 
          ? u[ind1] + 0.5*minmod(S_L_0, S_C_0)*(edgedx[jj]+u_cell_x*dt)
          : u[ind1+1] - 0.5*minmod(S_C_0, S_R_0)*(edgedx[jj]-u_cell_x*dt));
      uF_y[ind1] = f_y*((v_cell_y >= 0.0)
          ? u[ind1-(nx+1)] + 0.5*minmod(S_D_1, S_C_1)*(edgedy[ii]+v_cell_y*dt)
          : u[ind1] - 0.5*minmod(S_C_1, S_U_1)*(edgedy[ii]-v_cell_y*dt));
    }
  }
  STOP_PROFILING(&compute_profile, __func__);

  handle_boundary(nx, ny, mesh, uF_x, NO_INVERT, PACK);
  handle_boundary(nx+1, ny+1, mesh, uF_y, NO_INVERT, PACK);

  // Calculate the axial momentum
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      rho_u[ind1] -= dt_h*
        ((uF_x[ind0] - uF_x[ind0-1])/(edgedx[jj]*celldy[ii]) +
         (uF_y[ind1+(nx+1)] - uF_y[ind1])/(celldx[jj]*edgedy[ii]));
    }
  }

  /// ny DIMENSION ADVECTION
  // Calculate the corner centered y momentum fluxes in the x direction
  // Calculate the cell centered y momentum fluxes in the y direction
#pragma omp parallel for
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      const double invdx = 1.0/edgedx[jj];
      const double invdy = 1.0/edgedy[ii];

      const double S_L_0 = invdx*(v[ind0-1] - v[ind0-2]);
      const double S_C_0 = invdx*(v[ind0] - v[ind0-1]);
      const double S_R_0 = invdx*(v[ind0+1] - v[ind0]);
      const double S_D_1 = invdy*(v[ind0] - v[ind0-nx]);
      const double S_C_1 = invdy*(v[ind0+nx] - v[ind0]);
      const double S_U_1 = invdy*(v[ind0+2*nx] - v[ind0+nx]);

      const double f_x = celldy[ii]*0.5*(F_x[ind1] + F_x[ind1-(nx+1)]);
      const double f_y = celldx[jj]*0.5*(F_y[ind0] + F_y[ind0+nx]);
      const double u_cell_x = 0.5*(u[ind1]+u[ind1-(nx+1)]);
      const double v_cell_y = 0.5*(v[ind0]+v[ind0+nx]);

      vF_x[ind1] = f_x*((u_cell_x >= 0.0) 
          ? v[ind0-1] + 0.5*minmod(S_L_0, S_C_0)*(edgedx[jj]+u_cell_x*dt)
          : v[ind0] - 0.5*minmod(S_C_0, S_R_0)*(edgedx[jj]-u_cell_x*dt));
      vF_y[ind0] = f_y*((v_cell_y >= 0.0)
          ? v[ind0] + 0.5*minmod(S_D_1, S_C_1)*(edgedy[ii]+v_cell_y*dt)
          : v[ind0+nx] - 0.5*minmod(S_C_1, S_U_1)*(edgedy[ii]-v_cell_y*dt));
    }
  }
  STOP_PROFILING(&compute_profile, __func__);

  handle_boundary(nx+1, ny+1, mesh, vF_x, NO_INVERT, PACK);
  handle_boundary(nx, ny, mesh, vF_y, NO_INVERT, PACK);

  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      rho_v[ind0] -= dt_h*(
          (vF_x[ind1+1] - vF_x[ind1])/(edgedx[jj]*celldy[ii]) +
          (vF_y[ind0] - vF_y[ind0-nx])/(celldx[jj]*edgedy[ii]));
    }
  }
  STOP_PROFILING(&compute_profile, __func__);
}

// Perform advection of internal energy
void advect_energy(
    const int nx, const int ny, Mesh* mesh, const double dt_h, const double dt, 
    double* e, double* F_x, double* F_y, const double* u, const double* v, 
    const double* rho_old, const double* rho, const double* edgedx, 
    const double* edgedy, const double* celldx, const double* celldy)
{
  START_PROFILING(&compute_profile);

  // Calculate the zone edge centered energies, and flux
#pragma omp parallel for
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      // Use MC limiter to get slope of energy
      const double invdx = 1.0/edgedx[jj];
      const double a_x_0 = 0.5*invdx*(e[ind0]-e[ind0-2]);
      const double b_x_0 = 2.0*invdx*(e[ind0-1]-e[ind0-2]);
      const double c_x_0 = 2.0*invdx*(e[ind0]-e[ind0-1]);
      const double a_x_1 = 0.5*invdx*(e[ind0+1]-e[ind0-1]);
      const double b_x_1 = 2.0*invdx*(e[ind0]-e[ind0-1]);
      const double c_x_1 = 2.0*invdx*(e[ind0+1]-e[ind0]);

      // Calculate the interpolated densities
      const double edge_e_x = (u[ind1] > 0.0)
        ? e[ind0-1] + 0.5*minmod(minmod(a_x_0, b_x_0), c_x_0)*(celldx[jj-1] - u[ind1]*dt)
        : e[ind0] - 0.5*minmod(minmod(a_x_1, b_x_1), c_x_1)*(celldx[jj] + u[ind1]*dt);

      // Update the fluxes to now include the contribution from energy
      F_x[ind1] = edgedy[ii]*edge_e_x*F_x[ind1]; 
    }
  }

  // Calculate the new energy values
#pragma omp parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) { 
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      e[ind0] = (rho[ind0] == 0.0) ? 0.0 : 
        (rho_old[ind0]*e[ind0] - (dt_h/(celldx[jj]*celldy[ii]))*
         (F_x[ind1+1] - F_x[ind1]))/rho_old[ind0];
    }
  }

  // Calculate the zone edge centered energies, and flux
#pragma omp parallel for
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      // Use MC limiter to get slope of energy
      const double invdy = 1.0/edgedy[ii];
      const double a_y_0 = 0.5*invdy*(e[ind0]-e[ind0-2*nx]);
      const double b_y_0 = 2.0*invdy*(e[ind0-nx]-e[ind0-2*nx]);
      const double c_y_0 = 2.0*invdy*(e[ind0]-e[ind0-nx]);
      const double a_y_1 = 0.5*invdy*(e[ind0+nx]-e[ind0-nx]);
      const double b_y_1 = 2.0*invdy*(e[ind0]-e[ind0-nx]);
      const double c_y_1 = 2.0*invdy*(e[ind0+nx]-e[ind0]);

      const double edge_e_y = (v[ind0] > 0.0)
        ? e[ind0-nx] + 0.5*minmod(minmod(a_y_0, b_y_0), c_y_0)*(celldy[ii-1] - v[ind0]*dt)
        : e[ind0] - 0.5*minmod(minmod(a_y_1, b_y_1), c_y_1)*(celldy[ii] + v[ind0]*dt);

      // Update the fluxes to now include the contribution from energy
      F_y[ind0] = edgedx[jj]*edge_e_y*F_y[ind0]; 
    }
  }

  // Calculate the new energy values
#pragma omp parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) { 
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      e[ind0] = (rho[ind0] == 0.0) ? 0.0 : 
        (rho_old[ind0]*e[ind0] - (dt_h/(celldx[jj]*celldy[ii]))*
         (F_y[ind0+nx] - F_y[ind0]))/rho[ind0];
    }
  }

  STOP_PROFILING(&compute_profile, __func__);

  handle_boundary(nx, ny, mesh, e, NO_INVERT, PACK);
}

// Prints some conservation values
void print_conservation(
    const int nx, const int ny, State* state, Mesh* mesh) {
  double mass_tot = 0.0;
  double energy_tot = 0.0;
#pragma omp parallel for reduction(+:mass_tot, energy_tot)
  for(int ii = PAD; ii < ny-PAD; ++ii) {
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      mass_tot += state->rho[ind0];
      energy_tot += state->rho[ind0]*state->e[ind0];
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

