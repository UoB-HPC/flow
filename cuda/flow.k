#include "../../cuda/reduction.k"
#include "../../cuda/shared.h"
#include "../flow_interface.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Calculates the ideal gas equation of state
__global__ void equation_of_state(const int nx, const int ny, double* pressure,
                                  const double* density, const double* energy) {

  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid >= nx * ny)
    return;
  const int jj = (gid % (nx));
  const int ii = (gid / (nx));

  // Only invoke simple GAMma law at the moment
  pressure[(ii * nx + jj)] =
      (GAM - 1.0) * density[(ii * nx + jj)] * energy[(ii * nx + jj)];
}

// Calculate change in momentum caused by pressure gradients, and then extract
// the velocities using edge centered density approximations
__global__ void
pressure_acceleration(const int nx, const int ny, const int pad,
                      const double dt, double* momentum_x, double* momentum_y,
                      double* velocity_x, double* velocity_y,
                      const double* pressure, const double* density,
                      const double* edgedx, const double* edgedy,
                      const double* celldx, const double* celldy) {

  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int jj = (gid % (nx + 1));
  const int ii = (gid / (nx + 1));

  if (ii < pad || jj < pad || jj >= (nx + 1) - pad || ii >= (ny + 1) - pad)
    return;

  // Update the momenta using the pressure gradients
  momentum_x[(ii * (nx + 1) + jj)] -=
      dt * (pressure[(ii * nx + jj)] - pressure[(ii * nx + jj) - 1]) /
      edgedx[jj];
  momentum_y[(ii * nx + jj)] -=
      dt * (pressure[(ii * nx + jj)] - pressure[(ii * nx + jj) - nx]) /
      edgedy[ii];

  // Calculate the zone edge centered density
  const double density_edge_x =
      (density[(ii * nx + jj)] * celldx[jj] * celldy[ii] +
       density[(ii * nx + jj) - 1] * celldx[jj - 1] * celldy[ii]) /
      (2.0 * edgedx[jj] * celldy[ii]);
  const double density_edge_y =
      (density[(ii * nx + jj)] * celldx[jj] * celldy[ii] +
       density[(ii * nx + jj) - nx] * celldx[jj] * celldy[ii - 1]) /
      (2.0 * celldx[jj] * edgedy[ii]);

  // Find the velocities from the momenta and edge centered mass densities
  velocity_x[(ii * (nx + 1) + jj)] =
      (density_edge_x == 0.0)
          ? 0.0
          : momentum_x[(ii * (nx + 1) + jj)] / density_edge_x;
  velocity_y[(ii * nx + jj)] =
      (density_edge_y == 0.0) ? 0.0
                              : momentum_y[(ii * nx + jj)] / density_edge_y;
}

// Calculate the stresses due to artificial viscosity
__global__ void calc_viscous_stresses(
    const int nx, const int ny, const int pad, const double dt, double* Qxx,
    double* Qyy, double* velocity_x, double* velocity_y, double* momentum_x,
    double* momentum_y, const double* density, const double* edgedx,
    const double* edgedy, const double* celldx, const double* celldy) {

  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int jj = (gid % (nx));
  const int ii = (gid / (nx));

  if (ii < pad || jj < pad || jj >= nx - pad || ii >= ny - pad)
    return;

  const double u_i = min(0.0, velocity_x[(ii * (nx + 1) + jj) + 1] -
                                  velocity_x[(ii * (nx + 1) + jj)]);
  const double u_ii =
      0.5 * (fabs(min(0.0, (velocity_x[(ii * (nx + 1) + jj) + 2] -
                            velocity_x[(ii * (nx + 1) + jj) + 1])) -
                  min(0.0, (velocity_x[(ii * (nx + 1) + jj) + 1] -
                            velocity_x[(ii * (nx + 1) + jj)]))) +
             fabs(min(0.0, (velocity_x[(ii * (nx + 1) + jj) + 1] -
                            velocity_x[(ii * (nx + 1) + jj)])) -
                  min(0.0, (velocity_x[(ii * (nx + 1) + jj)] -
                            velocity_x[(ii * (nx + 1) + jj) - 1]))));
  const double v_i =
      min(0.0, velocity_y[(ii * nx + jj) + nx] - velocity_y[(ii * nx + jj)]);
  const double v_ii =
      0.5 * (fabs(min(0.0, (velocity_y[(ii * nx + jj) + 2 * nx] -
                            velocity_y[(ii * nx + jj) + nx])) -
                  min(0.0, (velocity_y[(ii * nx + jj) + nx] -
                            velocity_y[(ii * nx + jj)]))) +
             fabs(min(0.0, (velocity_y[(ii * nx + jj) + nx] -
                            velocity_y[(ii * nx + jj)])) -
                  min(0.0, (velocity_y[(ii * nx + jj)] -
                            velocity_y[(ii * nx + jj) - nx]))));

  Qxx[(ii * nx + jj)] = -C_Q * density[(ii * nx + jj)] * u_i * u_ii;
  Qyy[(ii * nx + jj)] = -C_Q * density[(ii * nx + jj)] * v_i * v_ii;
}

// Calculate the acceleration due to artificial viscosity terms
__global__ void viscous_acceleration(
    const int nx, const int ny, const int pad, const double dt, double* Qxx,
    double* Qyy, double* velocity_x, double* velocity_y, double* momentum_x,
    double* momentum_y, const double* density, const double* edgedx,
    const double* edgedy, const double* celldx, const double* celldy) {

  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int jj = (gid % (nx + 1));
  const int ii = (gid / (nx + 1));

  if (ii < pad || jj < pad || jj >= (nx + 1) - pad || ii >= (ny + 1) - pad)
    return;

  momentum_x[(ii * (nx + 1) + jj)] -=
      dt * (Qxx[(ii * nx + jj)] - Qxx[(ii * nx + jj) - 1]) / celldx[jj];
  momentum_y[(ii * nx + jj)] -=
      dt * (Qyy[(ii * nx + jj)] - Qyy[(ii * nx + jj) - nx]) / celldy[ii];

  // Calculate the zone edge centered density
  const double density_edge_x =
      (density[(ii * nx + jj)] * celldx[jj] * celldy[ii] +
       density[(ii * nx + jj) - 1] * celldx[jj - 1] * celldy[ii]) /
      (2.0 * edgedx[jj] * celldy[ii]);
  const double density_edge_y =
      (density[(ii * nx + jj)] * celldx[jj] * celldy[ii] +
       density[(ii * nx + jj) - nx] * celldx[jj] * celldy[ii - 1]) /
      (2.0 * celldx[jj] * edgedy[ii]);

  // Find the velocities from the momenta and edge centered mass densities
  velocity_x[(ii * (nx + 1) + jj)] =
      (density_edge_x == 0.0)
          ? 0.0
          : momentum_x[(ii * (nx + 1) + jj)] / density_edge_x;
  velocity_y[(ii * nx + jj)] =
      (density_edge_y == 0.0) ? 0.0
                              : momentum_y[(ii * nx + jj)] / density_edge_y;
}

// Calculates the work done due to forces within the element
__global__ void shock_heating_and_work(
    const int nx, const int ny, const int pad, const double dt_h,
    double* energy, const double* pressure, const double* velocity_x,
    const double* velocity_y, const double* density, const double* Qxx,
    const double* Qyy, const double* celldx, const double* celldy) {

  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int jj = (gid % (nx));
  const int ii = (gid / (nx));

  if (ii < pad || jj < pad || jj >= nx - pad || ii >= ny - pad)
    return;

  const double div_vel_x = (velocity_x[(ii * (nx + 1) + jj) + 1] -
                            velocity_x[(ii * (nx + 1) + jj)]) /
                           celldx[jj];
  const double div_vel_y =
      (velocity_y[(ii * nx + jj) + nx] - velocity_y[(ii * nx + jj)]) /
      celldy[ii];
  const double div_vel_dt = (div_vel_x + div_vel_y) * dt_h;
  const double e_q = energy[(ii * nx + jj)] -
                     dt_h * (Qxx[(ii * nx + jj)] * div_vel_x +
                             Qyy[(ii * nx + jj)] * div_vel_y) /
                         density[(ii * nx + jj)];

  /// A working formulation that is second order in time for Pressure!?
  const double density_c = density[(ii * nx + jj)] / (1.0 + div_vel_dt);
  const double e_c =
      e_q - (pressure[(ii * nx + jj)] * div_vel_dt) / density[(ii * nx + jj)];
  const double work = 0.5 * div_vel_dt * (pressure[(ii * nx + jj)] +
                                          (GAM - 1.0) * e_c * density_c) /
                      density[(ii * nx + jj)];
  energy[(ii * nx + jj)] = (density[(ii * nx + jj)] == 0.0) ? 0.0 : e_q - work;
}

// Store the old density value
__global__ void store_old_density(const int nx, const int ny, const int pad,
                                  double* density, double* density_old) {

  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int jj = (gid % (nx));
  const int ii = (gid / (nx));

  if (ii < pad || jj < pad || jj >= nx - pad || ii >= ny - pad)
    return;

  density_old[(ii * nx + jj)] = density[(ii * nx + jj)];
}

// Calculate the flux in the x direction
__global__ void calc_x_mass_and_energy_flux(
    const int nx, const int ny, const int first, const int pad, const double dt,
    const double dt_h, double* density, double* density_old, double* energy,
    const double* velocity_x, double* mass_flux, double* energy_mass_flux_x,
    const double* celldx, const double* edgedx, const double* celldy,
    const double* edgedy) {

  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int jj = (gid % (nx + 1));
  const int ii = (gid / (nx + 1));

  if (ii < pad || jj < pad || jj >= (nx + 1) - pad || ii >= ny - pad)
    return;

  // Interpolate to make second order in time
  const double invdx = 1.0 / edgedx[jj];
  const double suc0 = 0.5 * invdx * (velocity_x[(ii * (nx + 1) + jj) + 1] -
                                     velocity_x[(ii * (nx + 1) + jj) - 1]);
  const double sur0 = 2.0 * invdx * (velocity_x[(ii * (nx + 1) + jj)] -
                                     velocity_x[(ii * (nx + 1) + jj) - 1]);
  const double sul0 = 2.0 * invdx * (velocity_x[(ii * (nx + 1) + jj) + 1] -
                                     velocity_x[(ii * (nx + 1) + jj)]);
  const double u_tc = velocity_x[(ii * (nx + 1) + jj)] -
                      0.5 * velocity_x[(ii * (nx + 1) + jj)] * dt *
                          minmod(suc0, minmod(sur0, sul0));

  // Van leer limiter
  double limiter = 0.0;
  const double density_diff =
      (density[(ii * nx + jj)] - density[(ii * nx + jj) - 1]);
  if (density_diff) {
    const double smoothness =
        (u_tc >= 0.0)
            ? (density[(ii * nx + jj) - 1] - density[(ii * nx + jj) - 2]) /
                  density_diff
            : (density[(ii * nx + jj) + 1] - density[(ii * nx + jj)]) /
                  density_diff;
    limiter = (smoothness + fabs(smoothness)) / (1.0 + fabs(smoothness));
  }

  // Calculate the flux
  const double density_upwind =
      (u_tc >= 0.0) ? density[(ii * nx + jj) - 1] : density[(ii * nx + jj)];
  mass_flux[(ii * (nx + 1) + jj)] =
      (u_tc * density_upwind +
       0.5 * fabs(u_tc) * (1.0 - fabs((u_tc * dt_h) / celldx[jj])) * limiter *
           density_diff);

  // Use MC limiter to get slope of energy
  const double a_x_0 =
      0.5 * invdx * (energy[(ii * nx + jj)] - energy[(ii * nx + jj) - 2]);
  const double b_x_0 =
      2.0 * invdx * (energy[(ii * nx + jj) - 1] - energy[(ii * nx + jj) - 2]);
  const double c_x_0 =
      2.0 * invdx * (energy[(ii * nx + jj)] - energy[(ii * nx + jj) - 1]);
  const double a_x_1 =
      0.5 * invdx * (energy[(ii * nx + jj) + 1] - energy[(ii * nx + jj) - 1]);
  const double b_x_1 =
      2.0 * invdx * (energy[(ii * nx + jj)] - energy[(ii * nx + jj) - 1]);
  const double c_x_1 =
      2.0 * invdx * (energy[(ii * nx + jj) + 1] - energy[(ii * nx + jj)]);

  // Calculate the interpolated densities
  const double edge_e_x = (u_tc > 0.0)
                              ? energy[(ii * nx + jj) - 1] +
                                    0.5 * minmod(minmod(a_x_0, b_x_0), c_x_0) *
                                        (celldx[jj - 1] - u_tc * dt_h)
                              : energy[(ii * nx + jj)] -
                                    0.5 * minmod(minmod(a_x_1, b_x_1), c_x_1) *
                                        (celldx[jj] + u_tc * dt_h);

  // Update the fluxes to now include the contribution from energy
  energy_mass_flux_x[(ii * (nx + 1) + jj)] =
      edgedy[ii] * edge_e_x * mass_flux[(ii * (nx + 1) + jj)];
}

// Advect mass and energy in the x direction
__global__ void advect_mass_and_energy_in_x(
    const int nx, const int ny, const int first, const int pad, const double dt,
    const double dt_h, double* density, double* density_old, double* energy,
    const double* velocity_x, double* mass_flux, double* energy_mass_flux_x,
    const double* celldx, const double* edgedx, const double* celldy,
    const double* edgedy) {

  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int jj = (gid % (nx));
  const int ii = (gid / (nx));

  if (ii < pad || jj < pad || jj >= nx - pad || ii >= ny - pad)
    return;

  density[(ii * nx + jj)] -=
      dt_h * (edgedy[ii + 1] * mass_flux[(ii * (nx + 1) + jj) + 1] -
              edgedy[ii] * mass_flux[(ii * (nx + 1) + jj)]) /
      (celldx[jj] * celldy[ii]);
  const double density_e =
      (density_old[(ii * nx + jj)] * energy[(ii * nx + jj)] -
       (dt_h * (energy_mass_flux_x[(ii * (nx + 1) + jj) + 1] -
                energy_mass_flux_x[(ii * (nx + 1) + jj)])) /
           (celldx[jj] * celldy[ii]));
  energy[(ii * nx + jj)] = (first)
                               ? (density_old[(ii * nx + jj)] == 0.0)
                                     ? 0.0
                                     : density_e / density_old[(ii * nx + jj)]
                               : (density[(ii * nx + jj)] == 0.0)
                                     ? 0.0
                                     : density_e / density[(ii * nx + jj)];
}

__global__ void calc_y_mass_and_energy_flux(
    const int nx, const int ny, const int first, const int pad, const double dt,
    const double dt_h, double* density, double* density_old, double* energy,
    const double* velocity_y, double* mass_flux_y, double* energy_mass_flux_y,
    const double* celldx, const double* edgedx, const double* celldy,
    const double* edgedy) {

  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int jj = (gid % (nx));
  const int ii = (gid / (nx));

  if (ii < pad || jj < pad || jj >= nx - pad || ii >= (ny + 1) - pad)
    return;

  // Compute the mass flux along the y edges
  // In the ghost cells flux is left as 0.0
  // Interpolate the velocity to make second order in time
  const double invdy = 1.0 / edgedy[ii];
  const double svc0 = 0.5 * invdy * (velocity_y[(ii * nx + jj) + nx] -
                                     velocity_y[(ii * nx + jj) - nx]);
  const double svr0 = 2.0 * invdy * (velocity_y[(ii * nx + jj)] -
                                     velocity_y[(ii * nx + jj) - nx]);
  const double svl0 = 2.0 * invdy * (velocity_y[(ii * nx + jj) + nx] -
                                     velocity_y[(ii * nx + jj)]);
  const double v_tc =
      velocity_y[(ii * nx + jj)] -
      0.5 * velocity_y[(ii * nx + jj)] * dt * minmod(svc0, minmod(svr0, svl0));

  // Van leer limiter
  const double density_diff =
      (density[(ii * nx + jj)] - density[(ii * nx + jj) - nx]);
  double limiter = 0.0;
  if (density_diff) {
    const double smoothness =
        (velocity_y[(ii * nx + jj)] >= 0.0)
            ? (density[(ii * nx + jj) - nx] -
               density[(ii * nx + jj) - 2 * nx]) /
                  density_diff
            : (density[(ii * nx + jj) + nx] - density[(ii * nx + jj)]) /
                  density_diff;
    limiter = (smoothness + fabs(smoothness)) / (1.0 + fabs(smoothness));
  }

  // Calculate the flux
  const double density_upwind =
      (v_tc >= 0.0) ? density[(ii * nx + jj) - nx] : density[(ii * nx + jj)];
  mass_flux_y[(ii * nx + jj)] =
      (v_tc * density_upwind +
       0.5 * fabs(v_tc) * (1.0 - fabs((v_tc * dt_h) / celldy[ii])) * limiter *
           density_diff);

  // Use MC limiter to get slope of energy
  const double a_y_0 =
      0.5 * invdy * (energy[(ii * nx + jj)] - energy[(ii * nx + jj) - 2 * nx]);
  const double b_y_0 = 2.0 * invdy * (energy[(ii * nx + jj) - nx] -
                                      energy[(ii * nx + jj) - 2 * nx]);
  const double c_y_0 =
      2.0 * invdy * (energy[(ii * nx + jj)] - energy[(ii * nx + jj) - nx]);
  const double a_y_1 =
      0.5 * invdy * (energy[(ii * nx + jj) + nx] - energy[(ii * nx + jj) - nx]);
  const double b_y_1 =
      2.0 * invdy * (energy[(ii * nx + jj)] - energy[(ii * nx + jj) - nx]);
  const double c_y_1 =
      2.0 * invdy * (energy[(ii * nx + jj) + nx] - energy[(ii * nx + jj)]);

  const double edge_e_y = (v_tc > 0.0)
                              ? energy[(ii * nx + jj) - nx] +
                                    0.5 * minmod(minmod(a_y_0, b_y_0), c_y_0) *
                                        (celldy[ii - 1] - v_tc * dt_h)
                              : energy[(ii * nx + jj)] -
                                    0.5 * minmod(minmod(a_y_1, b_y_1), c_y_1) *
                                        (celldy[ii] + v_tc * dt_h);

  // Update the fluxes to now include the contribution from energy
  energy_mass_flux_y[(ii * nx + jj)] =
      edgedx[jj] * edge_e_y * mass_flux_y[(ii * nx + jj)];
}

__global__ void advect_mass_and_energy_in_y(
    const int nx, const int ny, const int first, const int pad, const double dt,
    const double dt_h, double* density, double* density_old, double* energy,
    const double* velocity_y, double* mass_flux_y, double* energy_mass_flux_y,
    const double* celldx, const double* edgedx, const double* celldy,
    const double* edgedy) {

  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int jj = (gid % (nx));
  const int ii = (gid / (nx));

  if (ii < pad || jj < pad || jj >= nx - pad || ii >= ny - pad)
    return;

  density[(ii * nx + jj)] -=
      dt_h * (edgedx[jj + 1] * mass_flux_y[(ii * nx + jj) + nx] -
              edgedx[jj] * mass_flux_y[(ii * nx + jj)]) /
      (celldx[jj] * celldy[ii]);
  const double density_e =
      (density_old[(ii * nx + jj)] * energy[(ii * nx + jj)] -
       (dt_h * (energy_mass_flux_y[(ii * nx + jj) + nx] -
                energy_mass_flux_y[(ii * nx + jj)])) /
           (celldx[jj] * celldy[ii]));

  energy[(ii * nx + jj)] = (first)
                               ? (density_old[(ii * nx + jj)] == 0.0)
                                     ? 0.0
                                     : density_e / density_old[(ii * nx + jj)]
                               : (density[(ii * nx + jj)] == 0.0)
                                     ? 0.0
                                     : density_e / density[(ii * nx + jj)];
}

__global__ void momentum_x_flux_in_x(const int nx, const int ny, const int pad,
                                 const double dt_h, const double dt,
                                 double* velocity_x, double* velocity_y,
                                 double* momentum_x_flux_x, double* momentum_x,
                                 const double* density, const double* mass_flux,
                                 const double* edgedx, const double* edgedy,
                                 const double* celldx, const double* celldy) {

  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int jj = (gid % (nx));
  const int ii = (gid / (nx));

  if (ii < pad || jj < pad || jj >= nx - pad || ii >= ny - pad)
    return;

  // Calculate the cell centered x momentum fluxes in the x direction
  // Use MC limiter to get slope of velocity
  const double invdx = 1.0 / edgedx[jj];
  const double a_x_0 = 0.5 * invdx * (velocity_x[(ii * (nx + 1) + jj) + 1] -
                                      velocity_x[(ii * (nx + 1) + jj) - 1]);
  const double b_x_0 = 2.0 * invdx * (velocity_x[(ii * (nx + 1) + jj)] -
                                      velocity_x[(ii * (nx + 1) + jj) - 1]);
  const double c_x_0 = 2.0 * invdx * (velocity_x[(ii * (nx + 1) + jj) + 1] -
                                      velocity_x[(ii * (nx + 1) + jj)]);
  const double a_x_1 = 0.5 * invdx * (velocity_x[(ii * (nx + 1) + jj) + 2] -
                                      velocity_x[(ii * (nx + 1) + jj)]);
  const double b_x_1 = 2.0 * invdx * (velocity_x[(ii * (nx + 1) + jj) + 1] -
                                      velocity_x[(ii * (nx + 1) + jj)]);
  const double c_x_1 = 2.0 * invdx * (velocity_x[(ii * (nx + 1) + jj) + 2] -
                                      velocity_x[(ii * (nx + 1) + jj) + 1]);

  // Calculate the interpolated densities
  const double u_cell_x = 0.5 * (velocity_x[(ii * (nx + 1) + jj)] +
                                 velocity_x[(ii * (nx + 1) + jj) + 1]);
  const double f_x = edgedy[ii] * 0.5 * (mass_flux[(ii * (nx + 1) + jj)] +
                                         mass_flux[(ii * (nx + 1) + jj) + 1]);
  const double u_cell_x_interp =
      (u_cell_x > 0.0)
          ? velocity_x[(ii * (nx + 1) + jj)] +
                0.5 * minmod(minmod(a_x_0, b_x_0), c_x_0) *
                    (celldx[jj - 1] - u_cell_x * dt_h)
          : velocity_x[(ii * (nx + 1) + jj) + 1] -
                0.5 * minmod(minmod(a_x_1, b_x_1), c_x_1) *
                    (celldx[jj] + u_cell_x * dt_h);
  momentum_x_flux_x[(ii * nx + jj)] = f_x * u_cell_x_interp;
}

__global__ void advect_momentum_x_in_x(
    const int nx, const int ny, const int tt, const int pad, const double dt_h,
    const double dt, double* velocity_x, double* velocity_y,
    double* momentum_x_flux_x, double* momentum_x_flux_y,
    double* momentum_y_flux_x, double* momentum_y_flux_y, double* momentum_x,
    double* momentum_y, const double* density, const double* mass_flux,
    const double* mass_flux_y, const double* edgedx, const double* edgedy,
    const double* celldx, const double* celldy) {

  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int jj = (gid % (nx + 1));
  const int ii = (gid / (nx + 1));

  if (ii < pad || jj < pad || jj >= (nx + 1) - pad || ii >= ny - pad)
    return;

  momentum_x[(ii * (nx + 1) + jj)] -= dt_h *
                                      (momentum_x_flux_x[(ii * nx + jj)] -
                                       momentum_x_flux_x[(ii * nx + jj) - 1]) /
                                      (edgedx[jj] * celldy[ii]);
}

__global__ void advect_momentum_x_and_u_in_x(
    const int nx, const int ny, const int tt, const int pad, const double dt_h,
    const double dt, double* velocity_x, double* velocity_y,
    double* momentum_x_flux_x, double* momentum_x_flux_y,
    double* momentum_y_flux_x, double* momentum_y_flux_y, double* momentum_x,
    double* momentum_y, const double* density, const double* mass_flux,
    const double* mass_flux_y, const double* edgedx, const double* edgedy,
    const double* celldx, const double* celldy) {

  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int jj = (gid % (nx + 1));
  const int ii = (gid / (nx + 1));

  if (ii < pad || jj < pad || jj >= (nx + 1) - pad || ii >= ny - pad)
    return;

  momentum_x[(ii * (nx + 1) + jj)] -= dt_h *
                                      (momentum_x_flux_x[(ii * nx + jj)] -
                                       momentum_x_flux_x[(ii * nx + jj) - 1]) /
                                      (edgedx[jj] * celldy[ii]);
  const double density_edge_x =
      (density[(ii * nx + jj)] * celldx[jj] * celldy[ii] +
       density[(ii * nx + jj) - 1] * celldx[jj - 1] * celldy[ii]) /
      (2.0 * edgedx[jj] * celldy[ii]);
  velocity_x[(ii * (nx + 1) + jj)] =
      (density_edge_x == 0.0)
          ? 0.0
          : momentum_x[(ii * (nx + 1) + jj)] / density_edge_x;
}

__global__ void advect_momentum_x_in_y(
    const int nx, const int ny, const int tt, const int pad, const double dt_h,
    const double dt, double* velocity_x, double* velocity_y,
    double* momentum_x_flux_x, double* momentum_x_flux_y,
    double* momentum_y_flux_x, double* momentum_y_flux_y, double* momentum_x,
    double* momentum_y, const double* density, const double* mass_flux,
    const double* mass_flux_y, const double* edgedx, const double* edgedy,
    const double* celldx, const double* celldy) {

  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int jj = (gid % (nx + 1));
  const int ii = (gid / (nx + 1));

  if (ii < pad || jj < pad || jj >= (nx + 1) - pad || ii >= ny - pad)
    return;

  momentum_x[(ii * (nx + 1) + jj)] -=
      dt_h * (momentum_x_flux_y[(ii * (nx + 1) + jj) + (nx + 1)] -
              momentum_x_flux_y[(ii * (nx + 1) + jj)]) /
      (celldx[jj] * edgedy[ii]);
}

__global__ void advect_momentum_x_and_u_in_y(
    const int nx, const int ny, const int tt, const int pad, const double dt_h,
    const double dt, double* velocity_x, double* velocity_y,
    double* momentum_x_flux_x, double* momentum_x_flux_y,
    double* momentum_y_flux_x, double* momentum_y_flux_y, double* momentum_x,
    double* momentum_y, const double* density, const double* mass_flux,
    const double* mass_flux_y, const double* edgedx, const double* edgedy,
    const double* celldx, const double* celldy) {

  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int jj = (gid % (nx + 1));
  const int ii = (gid / (nx + 1));

  if (ii < pad || jj < pad || jj >= (nx + 1) - pad || ii >= ny - pad)
    return;

  momentum_x[(ii * (nx + 1) + jj)] -=
      dt_h * (momentum_x_flux_y[(ii * (nx + 1) + jj) + (nx + 1)] -
              momentum_x_flux_y[(ii * (nx + 1) + jj)]) /
      (celldx[jj] * edgedy[ii]);
  const double density_edge_x =
      (density[(ii * nx + jj)] * celldx[jj] * celldy[ii] +
       density[(ii * nx + jj) - 1] * celldx[jj - 1] * celldy[ii]) /
      (2.0 * edgedx[jj] * celldy[ii]);
  velocity_x[(ii * (nx + 1) + jj)] =
      (density_edge_x == 0.0)
          ? 0.0
          : momentum_x[(ii * (nx + 1) + jj)] / density_edge_x;
}

__global__ void momentum_x_flux_in_y(const int nx, const int ny, const int pad,
                                 const double dt_h, const double dt,
                                 double* velocity_x, double* velocity_y,
                                 double* momentum_x_flux_y, double* momentum_x,
                                 const double* density,
                                 const double* mass_flux_y,
                                 const double* edgedx, const double* edgedy,
                                 const double* celldx, const double* celldy) {

  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int jj = (gid % (nx + 1));
  const int ii = (gid / (nx + 1));

  if (ii < pad || jj < pad || jj >= (nx + 1) - pad || ii >= (ny + 1) - pad)
    return;

  // Use MC limiter to get slope of velocity
  const double invdy = 1.0 / edgedy[ii];
  const double a_y_0 =
      0.5 * invdy * (velocity_x[(ii * (nx + 1) + jj)] -
                     velocity_x[(ii * (nx + 1) + jj) - 2 * (nx + 1)]);
  const double b_y_0 =
      2.0 * invdy * (velocity_x[(ii * (nx + 1) + jj) - (nx + 1)] -
                     velocity_x[(ii * (nx + 1) + jj) - 2 * (nx + 1)]);
  const double c_y_0 =
      2.0 * invdy * (velocity_x[(ii * (nx + 1) + jj)] -
                     velocity_x[(ii * (nx + 1) + jj) - (nx + 1)]);
  const double a_y_1 =
      0.5 * invdy * (velocity_x[(ii * (nx + 1) + jj) + (nx + 1)] -
                     velocity_x[(ii * (nx + 1) + jj) - (nx + 1)]);
  const double b_y_1 =
      2.0 * invdy * (velocity_x[(ii * (nx + 1) + jj)] -
                     velocity_x[(ii * (nx + 1) + jj) - (nx + 1)]);
  const double c_y_1 =
      2.0 * invdy * (velocity_x[(ii * (nx + 1) + jj) + (nx + 1)] -
                     velocity_x[(ii * (nx + 1) + jj)]);
  const double v_cell_y =
      0.5 * (velocity_y[(ii * nx + jj) - 1] + velocity_y[(ii * nx + jj)]);

  const double f_y = edgedx[jj] * 0.5 * (mass_flux_y[(ii * nx + jj)] +
                                         mass_flux_y[(ii * nx + jj) - 1]);
  const double u_corner_y =
      (v_cell_y > 0.0)
          ? velocity_x[(ii * (nx + 1) + jj) - (nx + 1)] +
                0.5 * minmod(minmod(a_y_0, b_y_0), c_y_0) *
                    (celldy[ii - 1] - v_cell_y * dt_h)
          : velocity_x[(ii * (nx + 1) + jj)] -
                0.5 * minmod(minmod(a_y_1, b_y_1), c_y_1) *
                    (celldy[ii] + v_cell_y * dt_h);
  momentum_x_flux_y[(ii * (nx + 1) + jj)] = f_y * u_corner_y;
}

__global__ void momentum_y_flux_in_x(const int nx, const int ny, const int pad,
                                 const double dt_h, const double dt,
                                 const double* velocity_x, double* velocity_y,
                                 double* momentum_y_flux_x, double* momentum_y,
                                 const double* density, const double* mass_flux,
                                 const double* edgedx, const double* edgedy,
                                 const double* celldx, const double* celldy) {

  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int jj = (gid % (nx + 1));
  const int ii = (gid / (nx + 1));

  if (ii < pad || jj < pad || jj >= (nx + 1) - pad || ii >= (ny + 1) - pad) {
    return;
  }

  // Use MC limiter to get slope of velocity
  const double invdx = 1.0 / edgedx[jj];
  const double a_x_0 = 0.5 * invdx * (velocity_y[(ii * nx + jj)] -
                                      velocity_y[(ii * nx + jj) - 2]);
  const double b_x_0 = 2.0 * invdx * (velocity_y[(ii * nx + jj) - 1] -
                                      velocity_y[(ii * nx + jj) - 2]);
  const double c_x_0 = 2.0 * invdx * (velocity_y[(ii * nx + jj)] -
                                      velocity_y[(ii * nx + jj) - 1]);
  const double a_x_1 = 0.5 * invdx * (velocity_y[(ii * nx + jj) + 1] -
                                      velocity_y[(ii * nx + jj) - 1]);
  const double b_x_1 = 2.0 * invdx * (velocity_y[(ii * nx + jj)] -
                                      velocity_y[(ii * nx + jj) - 1]);
  const double c_x_1 = 2.0 * invdx * (velocity_y[(ii * nx + jj) + 1] -
                                      velocity_y[(ii * nx + jj)]);

  // Calculate the interpolated densities
  const double f_x =
      celldy[ii] * 0.5 * (mass_flux[(ii * (nx + 1) + jj)] +
                          mass_flux[(ii * (nx + 1) + jj) - (nx + 1)]);
  const double u_cell_x = 0.5 * (velocity_x[(ii * (nx + 1) + jj)] +
                                 velocity_x[(ii * (nx + 1) + jj) - (nx + 1)]);
  const double v_cell_x_interp =
      (u_cell_x > 0.0)
          ? velocity_y[(ii * nx + jj) - 1] +
                0.5 * minmod(minmod(a_x_0, b_x_0), c_x_0) *
                    (celldx[jj - 1] - u_cell_x * dt_h)
          : velocity_y[(ii * nx + jj)] -
                0.5 * minmod(minmod(a_x_1, b_x_1), c_x_1) *
                    (celldx[jj] + u_cell_x * dt_h);
  momentum_y_flux_x[(ii * (nx + 1) + jj)] = f_x * v_cell_x_interp;
}

__global__ void advect_momentum_y_and_v_in_y(
    const int nx, const int ny, const int pad, const double dt_h,
    const double dt, double* velocity_x, double* velocity_y,
    double* momentum_y_flux_y, double* momentum_y, const double* density,
    const double* mass_flux_y, const double* edgedx, const double* edgedy,
    const double* celldx, const double* celldy) {

  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int jj = (gid % (nx));
  const int ii = (gid / (nx));

  if (ii < pad || jj < pad || jj >= nx - pad || ii >= (ny + 1) - pad)
    return;

  momentum_y[(ii * nx + jj)] -= dt_h *
                                (momentum_y_flux_y[(ii * nx + jj)] -
                                 momentum_y_flux_y[(ii * nx + jj) - nx]) /
                                (celldx[jj] * edgedy[ii]);
  const double density_edge_y =
      (density[(ii * nx + jj)] * celldx[jj] * celldy[ii] +
       density[(ii * nx + jj) - nx] * celldx[jj] * celldy[ii - 1]) /
      (2.0 * celldx[jj] * edgedy[ii]);
  velocity_y[(ii * nx + jj)] =
      (density_edge_y == 0.0) ? 0.0
                              : momentum_y[(ii * nx + jj)] / density_edge_y;
}

__global__ void advect_momentum_y_and_v_in_x(
    const int nx, const int ny, const int pad, const double dt_h,
    const double dt, const double* velocity_x, double* velocity_y,
    double* momentum_y_flux_x, double* momentum_y, const double* density,
    const double* mass_flux, const double* edgedx, const double* edgedy,
    const double* celldx, const double* celldy) {

  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int jj = (gid % (nx));
  const int ii = (gid / (nx));

  if (ii < pad || jj < pad || jj >= nx - pad || ii >= (ny + 1) - pad)
    return;

  momentum_y[(ii * nx + jj)] -= dt_h *
                                (momentum_y_flux_x[(ii * (nx + 1) + jj) + 1] -
                                 momentum_y_flux_x[(ii * (nx + 1) + jj)]) /
                                (edgedx[jj] * celldy[ii]);
  const double density_edge_y =
      (density[(ii * nx + jj)] * celldx[jj] * celldy[ii] +
       density[(ii * nx + jj) - nx] * celldx[jj] * celldy[ii - 1]) /
      (2.0 * celldx[jj] * edgedy[ii]);
  velocity_y[(ii * nx + jj)] =
      (density_edge_y == 0.0) ? 0.0
                              : momentum_y[(ii * nx + jj)] / density_edge_y;
}

__global__ void advect_momentum_y_in_x(
    const int nx, const int ny, const int pad, const double dt_h,
    const double dt, const double* velocity_x, double* velocity_y,
    double* momentum_y_flux_x, double* momentum_y, const double* density,
    const double* mass_flux, const double* edgedx, const double* edgedy,
    const double* celldx, const double* celldy) {

  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int jj = (gid % (nx));
  const int ii = (gid / (nx));

  if (ii < pad || jj < pad || jj >= nx - pad || ii >= (ny + 1) - pad)
    return;

  momentum_y[(ii * nx + jj)] -= dt_h *
                                (momentum_y_flux_x[(ii * (nx + 1) + jj) + 1] -
                                 momentum_y_flux_x[(ii * (nx + 1) + jj)]) /
                                (edgedx[jj] * celldy[ii]);
}

__global__ void momentum_y_flux_in_y(const int nx, const int ny, const int pad,
                                 const double dt_h, const double dt,
                                 double* velocity_x, double* velocity_y,
                                 double* momentum_y_flux_y, double* momentum_y,
                                 const double* density,
                                 const double* mass_flux_y,
                                 const double* edgedx, const double* edgedy,
                                 const double* celldx, const double* celldy) {
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int jj = (gid % (nx));
  const int ii = (gid / (nx));

  if (ii < pad || jj < pad || jj >= nx - pad || ii >= ny - pad)
    return;

  // Use MC limiter to get slope of velocity
  const double invdy = 1.0 / edgedy[ii];
  const double a_y_0 = 0.5 * invdy * (velocity_y[(ii * nx + jj) + nx] -
                                      velocity_y[(ii * nx + jj) - nx]);
  const double b_y_0 = 2.0 * invdy * (velocity_y[(ii * nx + jj)] -
                                      velocity_y[(ii * nx + jj) - nx]);
  const double c_y_0 = 2.0 * invdy * (velocity_y[(ii * nx + jj) + nx] -
                                      velocity_y[(ii * nx + jj)]);
  const double a_y_1 = 0.5 * invdy * (velocity_y[(ii * nx + jj) + 2 * nx] -
                                      velocity_y[(ii * nx + jj)]);
  const double b_y_1 = 2.0 * invdy * (velocity_y[(ii * nx + jj) + nx] -
                                      velocity_y[(ii * nx + jj)]);
  const double c_y_1 = 2.0 * invdy * (velocity_y[(ii * nx + jj) + 2 * nx] -
                                      velocity_y[(ii * nx + jj) + nx]);

  const double f_y = celldx[jj] * 0.5 * (mass_flux_y[(ii * nx + jj)] +
                                         mass_flux_y[(ii * nx + jj) + nx]);
  const double v_cell_y =
      0.5 * (velocity_y[(ii * nx + jj)] + velocity_y[(ii * nx + jj) + nx]);
  const double v_cell_y_interp =
      (v_cell_y > 0.0)
          ? velocity_y[(ii * nx + jj)] +
                0.5 * minmod(minmod(a_y_0, b_y_0), c_y_0) *
                    (celldy[ii - 1] - v_cell_y * dt_h)
          : velocity_y[(ii * nx + jj) + nx] -
                0.5 * minmod(minmod(a_y_1, b_y_1), c_y_1) *
                    (celldy[ii] + v_cell_y * dt_h);
  momentum_y_flux_y[(ii * nx + jj)] = f_y * v_cell_y_interp;
}

__global__ void advect_momentum_y_in_y(
    const int nx, const int ny, const int pad, const double dt_h,
    const double dt, double* velocity_x, double* velocity_y,
    double* momentum_y_flux_y, double* momentum_y, const double* density,
    const double* mass_flux_y, const double* edgedx, const double* edgedy,
    const double* celldx, const double* celldy) {

  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int jj = (gid % (nx));
  const int ii = (gid / (nx));

  if (ii < pad || jj < pad || jj >= nx - pad || ii >= (ny + 1) - pad)
    return;

  momentum_y[(ii * nx + jj)] -= dt_h *
                                (momentum_y_flux_y[(ii * nx + jj)] -
                                 momentum_y_flux_y[(ii * nx + jj) - nx]) /
                                (celldx[jj] * edgedy[ii]);
}

__global__ void calc_min_timestep(const int nx, const int ny, const int pad,
                                  const double max_dt, double* Qxx, double* Qyy,
                                  const double* density, const double* energy,
                                  double* reduce_array, const int first_step,
                                  const double* celldx, const double* celldy) {

  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int jj = (gid % (nx));
  const int ii = (gid / (nx));

  const int tid = threadIdx.x;

  __shared__ double sdata[NTHREADS];
  sdata[tid] = max_dt;

  if (ii >= pad && ii < ny - pad && jj >= pad && jj < nx - pad) {
    // Constrain based on the sound speed within the system
    const double c_s = sqrt(GAM * (GAM - 1.0) * energy[(ii * nx + jj)]);
    const double thread_min_dt_x =
        celldx[jj] /
        sqrt(c_s * c_s + 2.0 * Qxx[(ii * nx + jj)] / density[(ii * nx + jj)]);
    const double thread_min_dt_y =
        celldy[ii] /
        sqrt(c_s * c_s + 2.0 * Qyy[(ii * nx + jj)] / density[(ii * nx + jj)]);
    sdata[tid] = min(thread_min_dt_x, thread_min_dt_y);
  }
  __syncthreads();

  min_reduce_in_shared<double, NTHREADS>(tid, sdata);
  if (tid == 0)
    reduce_array[blockIdx.x] = sdata[0];
}

__global__ void calc_mass_sum(const int nx, const int ny, const int pad,
                              const double* density, double* reduce_array) {
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int jj = (gid % (nx));
  const int ii = (gid / (nx));
  const int tid = threadIdx.x;

  __shared__ double sdata[NTHREADS];
  sdata[tid] = 0.0;

  if (ii >= pad && ii < ny - pad && jj >= pad && jj < nx - pad) {
    sdata[tid] = density[(ii * nx + jj)];
  }

  __syncthreads();

  sum_reduce_in_shared<double, NTHREADS>(tid, sdata);
  if (tid == 0)
    reduce_array[blockIdx.x] = sdata[0];
}
