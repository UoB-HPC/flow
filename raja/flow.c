#include "flow.h"
#include "../../comms.h"
#include "../../params.h"
#include "../flow_interface.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../../raja/shared.h"

#define min_l(a, b) (((a) < (b)) ? (a) : (b))

// Solve a single timestep on the given mesh
void solve_hydro_2d(Mesh* mesh, int tt, double* pressure, double* density,
                    double* density_old, double* energy, double* velocity_x,
                    double* velocity_y, double* momentum_x, double* momentum_y,
                    double* Qxx, double* Qyy, double* mass_flux_x,
                    double* mass_flux_y, double* momentum_x_flux_x,
                    double* momentum_x_flux_y, double* momentum_y_flux_x,
                    double* momentum_y_flux_y, double* reduce_array) {

  if (mesh->rank == MASTER) {
    printf("Timestep:        %.12e\n", mesh->dt);
  }

  equation_of_state(mesh->local_nx, mesh->local_ny, pressure, density, energy);

  pressure_acceleration(mesh->local_nx, mesh->local_ny, mesh, mesh->dt,
                        momentum_x, momentum_y, velocity_x, velocity_y,
                        pressure, density, mesh->edgedx, mesh->edgedy,
                        mesh->celldx, mesh->celldy);

  artificial_viscosity(mesh->local_nx, mesh->local_ny, mesh, mesh->dt, Qxx, Qyy,
                       velocity_x, velocity_y, momentum_x, momentum_y, density,
                       mesh->edgedx, mesh->edgedy, mesh->celldx, mesh->celldy);

  shock_heating_and_work(mesh->local_nx, mesh->local_ny, mesh, mesh->dt_h,
                         energy, pressure, velocity_x, velocity_y, density, Qxx,
                         Qyy, mesh->celldx, mesh->celldy);

  set_timestep(mesh->local_nx, mesh->local_ny, Qxx, Qyy, density, energy, mesh,
               reduce_array, tt == 0, mesh->celldx, mesh->celldy);

  // Perform advection
  advect_mass_and_energy(mesh->local_nx, mesh->local_ny, mesh, tt, mesh->dt,
                         mesh->dt_h, density, energy, density_old, mass_flux_x,
                         mass_flux_y, momentum_x_flux_x, momentum_x_flux_y,
                         velocity_x, velocity_y, mesh->edgedx, mesh->edgedy,
                         mesh->celldx, mesh->celldy);

  advect_momentum(mesh->local_nx, mesh->local_ny, tt, mesh, mesh->dt_h,
                  mesh->dt, velocity_x, velocity_y, momentum_x_flux_x,
                  momentum_x_flux_y, momentum_y_flux_x, momentum_y_flux_y,
                  momentum_x, momentum_y, density, mass_flux_x, mass_flux_y,
                  mesh->edgedx, mesh->edgedy, mesh->celldx, mesh->celldy);
}

// Calculate the pressure from GAMma law equation of state
void equation_of_state(const int nx, const int ny, double* pressure,
                       const double* density, const double* energy) {
  START_PROFILING(&compute_profile);

  RAJA::forall<exec_policy>(RAJA::RangeSegment(0, nx*ny), [=] RAJA_DEVICE (int ii) {
     // Only invoke simple GAMma law at the moment
     pressure[ii] = (GAM - 1.0) * density[ii] * energy[ii];
  });

  STOP_PROFILING(&compute_profile, __func__);
}

// Calculates the timestep from the current state
void set_timestep(const int nx, const int ny, double* Qxx, double* Qyy,
                  const double* density, const double* energy, Mesh* mesh,
                  double* reduce_array, const int first_step,
                  const double* celldx, const double* celldy) {

  const int pad = mesh->pad;

  START_PROFILING(&compute_profile);
  // Check the minimum timestep from the sound speed in the nx and ny directions
  RAJA::ReduceMin<reduce_policy, double> local_min_dt(mesh->max_dt);
  RAJA::forall<exec_policy>(RAJA::RangeSegment(0, nx*ny), [=] RAJA_DEVICE (int i) {
      const int ii = i / nx; 
      const int jj = i % nx;
      if(ii >= pad && ii < ny-pad && jj >= pad && jj < nx-pad) {
        // Constrain based on the sound speed within the system
        const double c_s = sqrt(GAM * (GAM - 1.0) * energy[(ii * nx + jj)]);
        const double thread_min_dt_x =
            celldx[jj] /
            sqrt(c_s * c_s + 2.0 * Qxx[(ii * nx + jj)] / density[(ii * nx + jj)]);
        const double thread_min_dt_y =
            celldy[ii] /
            sqrt(c_s * c_s + 2.0 * Qyy[(ii * nx + jj)] / density[(ii * nx + jj)]);
        const double thread_min_dt = min_l(thread_min_dt_x, thread_min_dt_y);
        local_min_dt.min(thread_min_dt);
      }
  });
  STOP_PROFILING(&compute_profile, __func__);

  double global_min_dt = reduce_all_min(local_min_dt.get());

  // Ensure that the timestep does not jump too far from one step to the next
  const double final_min_dt = min_l(global_min_dt, C_M * mesh->dt_h);
  mesh->dt = 0.5 * (C_T * final_min_dt + mesh->dt_h);
  mesh->dt_h = (first_step) ? mesh->dt : C_T * final_min_dt;
}

// Calculate change in momentum caused by pressure gradients, and then extract
// the velocities using edge centered density approximations
void pressure_acceleration(const int nx, const int ny, Mesh* mesh,
                           const double dt, double* momentum_x,
                           double* momentum_y, double* velocity_x,
                           double* velocity_y, const double* pressure,
                           const double* density, const double* edgedx,
                           const double* edgedy, const double* celldx,
                           const double* celldy) {
  START_PROFILING(&compute_profile);

  const int pad = mesh->pad;

  RAJA::forall<exec_policy>(RAJA::RangeSegment(0, (nx+1)*(ny+1)), [=] RAJA_DEVICE (int i) {
      const int ii = i / (nx+1); 
      const int jj = i % (nx+1);
      if(ii >= pad && ii < (ny+1)-pad && jj >= pad && jj < (nx+1)-pad) {
      // Update the momenta using the pressure gradients
      momentum_x[(ii * (nx + 1) + jj)] -=
          dt * (pressure[(ii * nx + jj)] - pressure[ii*nx + (jj - 1)]) /
          edgedx[jj];
      momentum_y[(ii * nx + jj)] -=
          dt * (pressure[(ii * nx + jj)] - pressure[(ii-1)*nx + jj]) /
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
  });

  STOP_PROFILING(&compute_profile, __func__);

  handle_boundary_2d(nx + 1, ny, mesh, velocity_x, INVERT_X, PACK);
  handle_boundary_2d(nx, ny + 1, mesh, velocity_y, INVERT_Y, PACK);
}

void artificial_viscosity(const int nx, const int ny, Mesh* mesh,
                          const double dt, double* Qxx, double* Qyy,
                          double* velocity_x, double* velocity_y,
                          double* momentum_x, double* momentum_y,
                          const double* density, const double* edgedx,
                          const double* edgedy, const double* celldx,
                          const double* celldy) {
  START_PROFILING(&compute_profile);

  const int pad = mesh->pad;

  // Calculate the artificial viscous stresses
  // PLPC Hydro Paper
  RAJA::forall<exec_policy>(RAJA::RangeSegment(0, nx*ny), [=] RAJA_DEVICE (int i) {
      const int ii = i / nx; 
      const int jj = i % nx;
      if(ii >= pad && ii < ny-pad && jj >= pad && jj < nx-pad) {
      const double u_i = min_l(0.0, velocity_x[(ii * (nx + 1) + jj) + 1] -
                                      velocity_x[(ii * (nx + 1) + jj)]);
      const double u_ii =
          0.5 * (fabs(min_l(0.0, (velocity_x[(ii * (nx + 1) + jj) + 2] -
                                velocity_x[(ii * (nx + 1) + jj) + 1])) -
                      min_l(0.0, (velocity_x[(ii * (nx + 1) + jj) + 1] -
                                velocity_x[(ii * (nx + 1) + jj)]))) +
                 fabs(min_l(0.0, (velocity_x[(ii * (nx + 1) + jj) + 1] -
                                velocity_x[(ii * (nx + 1) + jj)])) -
                      min_l(0.0, (velocity_x[(ii * (nx + 1) + jj)] -
                                velocity_x[(ii * (nx + 1) + jj) - 1]))));
      const double v_i = min_l(0.0, velocity_y[(ii * nx + jj) + nx] -
                                      velocity_y[(ii * nx + jj)]);
      const double v_ii =
          0.5 * (fabs(min_l(0.0, (velocity_y[(ii * nx + jj) + 2 * nx] -
                                velocity_y[(ii * nx + jj) + nx])) -
                      min_l(0.0, (velocity_y[(ii * nx + jj) + nx] -
                                velocity_y[(ii * nx + jj)]))) +
                 fabs(min_l(0.0, (velocity_y[(ii * nx + jj) + nx] -
                                velocity_y[(ii * nx + jj)])) -
                      min_l(0.0, (velocity_y[(ii * nx + jj)] -
                                velocity_y[(ii * nx + jj) - nx]))));
      Qxx[(ii * nx + jj)] = -C_Q * density[(ii * nx + jj)] * u_i * u_ii;
      Qyy[(ii * nx + jj)] = -C_Q * density[(ii * nx + jj)] * v_i * v_ii;
    }
  });

  STOP_PROFILING(&compute_profile, __func__);

  handle_boundary_2d(nx, ny, mesh, Qxx, NO_INVERT, PACK);
  handle_boundary_2d(nx, ny, mesh, Qyy, NO_INVERT, PACK);

  START_PROFILING(&compute_profile);

  // Update the momenta by the artificial viscous stresses
  RAJA::forall<exec_policy>(RAJA::RangeSegment(0, (nx+1)*(ny+1)), [=] RAJA_DEVICE (int i) {
      const int ii = i / (nx+1); 
      const int jj = i % (nx+1);
      if(ii >= pad && ii < (ny+1)-pad && jj >= pad && jj < (nx+1)-pad) {
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
  });
  STOP_PROFILING(&compute_profile, __func__);

  handle_boundary_2d(nx + 1, ny, mesh, velocity_x, INVERT_X, PACK);
  handle_boundary_2d(nx, ny + 1, mesh, velocity_y, INVERT_Y, PACK);
}

// Calculates the work done due to forces within the element
void shock_heating_and_work(const int nx, const int ny, Mesh* mesh,
                            const double dt_h, double* energy,
                            const double* pressure, const double* velocity_x,
                            const double* velocity_y, const double* density,
                            const double* Qxx, const double* Qyy,
                            const double* celldx, const double* celldy) {
  START_PROFILING(&compute_profile);

  const int pad = mesh->pad;

  RAJA::forall<exec_policy>(RAJA::RangeSegment(0, nx*ny), [=] RAJA_DEVICE (int i) {
      const int ii = i / nx; 
      const int jj = i % nx;
      if(ii >= pad && ii < ny-pad && jj >= pad && jj < nx-pad) {
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
          e_q -
          (pressure[(ii * nx + jj)] * div_vel_dt) / density[(ii * nx + jj)];
      const double work = 0.5 * div_vel_dt * (pressure[(ii * nx + jj)] +
                                              (GAM - 1.0) * e_c * density_c) /
                          density[(ii * nx + jj)];
      energy[(ii * nx + jj)] =
          (density[(ii * nx + jj)] == 0.0) ? 0.0 : e_q - work;
    }
  });

  STOP_PROFILING(&compute_profile, __func__);

  handle_boundary_2d(nx, ny, mesh, energy, NO_INVERT, PACK);
}

// Perform advection with monotonicity improvement
void advect_mass_and_energy(const int nx, const int ny, Mesh* mesh,
                            const int tt, const double dt, const double dt_h,
                            double* density, double* energy,
                            double* density_old, double* mass_flux_x,
                            double* mass_flux_y, double* eF_x, double* eF_y,
                            const double* velocity_x, const double* velocity_y,
                            const double* edgedx, const double* edgedy,
                            const double* celldx, const double* celldy) {
  START_PROFILING(&compute_profile);
  RAJA::forall<exec_policy>(RAJA::RangeSegment(0, nx*ny), [=] RAJA_DEVICE (int ii) {
    density_old[ii] = density[ii];
  });
  STOP_PROFILING(&compute_profile, "storing_old_density");

  if (tt % 2 == 0) {
    x_mass_and_energy_flux(nx, ny, 1, mesh, dt, dt_h, density, density_old,
                           energy, velocity_x, mass_flux_x, eF_x, celldx,
                           edgedx, celldy, edgedy);
    y_mass_and_energy_flux(nx, ny, 0, mesh, dt, dt_h, density, density_old,
                           energy, velocity_y, mass_flux_y, eF_y, celldx,
                           edgedx, celldy, edgedy);
  } else {
    y_mass_and_energy_flux(nx, ny, 1, mesh, dt, dt_h, density, density_old,
                           energy, velocity_y, mass_flux_y, eF_y, celldx,
                           edgedx, celldy, edgedy);
    x_mass_and_energy_flux(nx, ny, 0, mesh, dt, dt_h, density, density_old,
                           energy, velocity_x, mass_flux_x, eF_x, celldx,
                           edgedx, celldy, edgedy);
  }
}

// Calculate the flux in the x direction
void x_mass_and_energy_flux(const int nx, const int ny, const int first,
                            Mesh* mesh, const double dt, const double dt_h,
                            double* density, double* density_old,
                            double* energy, const double* velocity_x,
                            double* mass_flux_x, double* eF_x,
                            const double* celldx, const double* edgedx,
                            const double* celldy, const double* edgedy) {
  const int pad = mesh->pad;

  // Compute the mass fluxes along the x edges
  // In the ghost cells flux is left as 0.0
  START_PROFILING(&compute_profile);
  RAJA::forall<exec_policy>(RAJA::RangeSegment(0, (nx+1)*ny), [=] RAJA_DEVICE (int i) {
      const int ii = i / (nx+1); 
      const int jj = i % (nx+1);
      if(ii >= pad && ii < ny-pad && jj >= pad && jj < (nx+1)-pad) {
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
      mass_flux_x[(ii * (nx + 1) + jj)] =
          (u_tc * density_upwind +
           0.5 * fabs(u_tc) * (1.0 - fabs((u_tc * dt_h) / celldx[jj])) *
               limiter * density_diff);

      // Use MC limiter to get slope of energy
      const double a_x_0 =
          0.5 * invdx * (energy[(ii * nx + jj)] - energy[(ii * nx + jj) - 2]);
      const double b_x_0 = 2.0 * invdx * (energy[(ii * nx + jj) - 1] -
                                          energy[(ii * nx + jj) - 2]);
      const double c_x_0 =
          2.0 * invdx * (energy[(ii * nx + jj)] - energy[(ii * nx + jj) - 1]);
      const double a_x_1 = 0.5 * invdx * (energy[(ii * nx + jj) + 1] -
                                          energy[(ii * nx + jj) - 1]);
      const double b_x_1 =
          2.0 * invdx * (energy[(ii * nx + jj)] - energy[(ii * nx + jj) - 1]);
      const double c_x_1 =
          2.0 * invdx * (energy[(ii * nx + jj) + 1] - energy[(ii * nx + jj)]);

      // Calculate the interpolated densities
      const double edge_e_x =
          (u_tc > 0.0)
              ? energy[(ii * nx + jj) - 1] +
                    0.5 * minmod(minmod(a_x_0, b_x_0), c_x_0) *
                        (celldx[jj - 1] - u_tc * dt_h)
              : energy[(ii * nx + jj)] -
                    0.5 * minmod(minmod(a_x_1, b_x_1), c_x_1) *
                        (celldx[jj] + u_tc * dt_h);

      // Update the fluxes to now include the contribution from energy
      eF_x[(ii * (nx + 1) + jj)] =
          edgedy[ii] * edge_e_x * mass_flux_x[(ii * (nx + 1) + jj)];
    }
  });
  STOP_PROFILING(&compute_profile, "advect_mass_and_energy");

  handle_boundary_2d(nx + 1, ny, mesh, mass_flux_x, INVERT_X, PACK);

  // Calculate the new density values
  START_PROFILING(&compute_profile);
  RAJA::forall<exec_policy>(RAJA::RangeSegment(pad, nx*ny), [=] RAJA_DEVICE (int i) {
      const int ii = i / nx; 
      const int jj = i % nx;
      if(ii >= pad && ii < ny-pad && jj >= pad && jj < nx-pad) {
      density[(ii * nx + jj)] -=
          dt_h * (edgedy[ii + 1] * mass_flux_x[(ii * (nx + 1) + jj) + 1] -
                  edgedy[ii] * mass_flux_x[(ii * (nx + 1) + jj)]) /
          (celldx[jj] * celldy[ii]);
      const double density_e =
          (density_old[(ii * nx + jj)] * energy[(ii * nx + jj)] -
           (dt_h *
            (eF_x[(ii * (nx + 1) + jj) + 1] - eF_x[(ii * (nx + 1) + jj)])) /
               (celldx[jj] * celldy[ii]));
      energy[(ii * nx + jj)] =
          (first)
              ? (density_old[(ii * nx + jj)] == 0.0)
                    ? 0.0
                    : density_e / density_old[(ii * nx + jj)]
              : (density[(ii * nx + jj)] == 0.0)
                    ? 0.0
                    : density_e / density[(ii * nx + jj)];
    }
  });
  STOP_PROFILING(&compute_profile, "advect_mass_and_energy");

  handle_boundary_2d(nx, ny, mesh, density, NO_INVERT, PACK);
  handle_boundary_2d(nx, ny, mesh, energy, NO_INVERT, PACK);
}

// Calculate the flux in the y direction
void y_mass_and_energy_flux(const int nx, const int ny, const int first,
                            Mesh* mesh, const double dt, const double dt_h,
                            double* density, double* density_old,
                            double* energy, const double* velocity_y,
                            double* mass_flux_y, double* eF_y,
                            const double* celldx, const double* edgedx,
                            const double* celldy, const double* edgedy) {
  const int pad = mesh->pad;

  // Compute the mass flux along the y edges
  // In the ghost cells flux is left as 0.0
  START_PROFILING(&compute_profile);
  RAJA::forall<exec_policy>(RAJA::RangeSegment(0, nx*(ny+1)), [=] RAJA_DEVICE (int i) {
      const int ii = i / nx; 
      const int jj = i % nx;
      if(ii >= pad && ii < (ny+1)-pad && jj >= pad && jj < nx-pad) {

      // Interpolate the velocity to make second order in time
      const double invdy = 1.0 / edgedy[ii];
      const double svc0 = 0.5 * invdy * (velocity_y[(ii * nx + jj) + nx] -
                                         velocity_y[(ii * nx + jj) - nx]);
      const double svr0 = 2.0 * invdy * (velocity_y[(ii * nx + jj)] -
                                         velocity_y[(ii * nx + jj) - nx]);
      const double svl0 = 2.0 * invdy * (velocity_y[(ii * nx + jj) + nx] -
                                         velocity_y[(ii * nx + jj)]);
      const double v_tc = velocity_y[(ii * nx + jj)] -
                          0.5 * velocity_y[(ii * nx + jj)] * dt *
                              minmod(svc0, minmod(svr0, svl0));

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
      const double density_upwind = (v_tc >= 0.0) ? density[(ii * nx + jj) - nx]
                                                  : density[(ii * nx + jj)];
      mass_flux_y[(ii * nx + jj)] =
          (v_tc * density_upwind +
           0.5 * fabs(v_tc) * (1.0 - fabs((v_tc * dt_h) / celldy[ii])) *
               limiter * density_diff);

      // Use MC limiter to get slope of energy
      const double a_y_0 = 0.5 * invdy * (energy[(ii * nx + jj)] -
                                          energy[(ii * nx + jj) - 2 * nx]);
      const double b_y_0 = 2.0 * invdy * (energy[(ii * nx + jj) - nx] -
                                          energy[(ii * nx + jj) - 2 * nx]);
      const double c_y_0 =
          2.0 * invdy * (energy[(ii * nx + jj)] - energy[(ii * nx + jj) - nx]);
      const double a_y_1 = 0.5 * invdy * (energy[(ii * nx + jj) + nx] -
                                          energy[(ii * nx + jj) - nx]);
      const double b_y_1 =
          2.0 * invdy * (energy[(ii * nx + jj)] - energy[(ii * nx + jj) - nx]);
      const double c_y_1 =
          2.0 * invdy * (energy[(ii * nx + jj) + nx] - energy[(ii * nx + jj)]);

      const double edge_e_y =
          (v_tc > 0.0)
              ? energy[(ii * nx + jj) - nx] +
                    0.5 * minmod(minmod(a_y_0, b_y_0), c_y_0) *
                        (celldy[ii - 1] - v_tc * dt_h)
              : energy[(ii * nx + jj)] -
                    0.5 * minmod(minmod(a_y_1, b_y_1), c_y_1) *
                        (celldy[ii] + v_tc * dt_h);

      // Update the fluxes to now include the contribution from energy
      eF_y[(ii * nx + jj)] =
          edgedx[jj] * edge_e_y * mass_flux_y[(ii * nx + jj)];
    }
  });
  STOP_PROFILING(&compute_profile, "advect_mass_and_energy");

  handle_boundary_2d(nx, ny + 1, mesh, mass_flux_y, INVERT_Y, PACK);

  // Calculate the new density values
  START_PROFILING(&compute_profile);
  RAJA::forall<exec_policy>(RAJA::RangeSegment(0, nx*ny), [=] RAJA_DEVICE (int i) {
      const int ii = i / nx; 
      const int jj = i % nx;
      if(ii >= pad && ii < ny-pad && jj >= pad && jj < nx-pad) {
      density[(ii * nx + jj)] -=
          dt_h * (edgedx[jj + 1] * mass_flux_y[(ii * nx + jj) + nx] -
                  edgedx[jj] * mass_flux_y[(ii * nx + jj)]) /
          (celldx[jj] * celldy[ii]);
      const double density_e =
          (density_old[(ii * nx + jj)] * energy[(ii * nx + jj)] -
           (dt_h * (eF_y[(ii * nx + jj) + nx] - eF_y[(ii * nx + jj)])) /
               (celldx[jj] * celldy[ii]));
      energy[(ii * nx + jj)] =
          (first)
              ? (density_old[(ii * nx + jj)] == 0.0)
                    ? 0.0
                    : density_e / density_old[(ii * nx + jj)]
              : (density[(ii * nx + jj)] == 0.0)
                    ? 0.0
                    : density_e / density[(ii * nx + jj)];
    }
  });
  STOP_PROFILING(&compute_profile, "advect_mass_and_energy");

  handle_boundary_2d(nx, ny, mesh, density, NO_INVERT, PACK);
  handle_boundary_2d(nx, ny, mesh, energy, NO_INVERT, PACK);
}

// Advect momentum according to the velocity
void advect_momentum(const int nx, const int ny, const int tt, Mesh* mesh,
                     const double dt_h, const double dt, double* velocity_x,
                     double* velocity_y, double* momentum_x_flux_x,
                     double* momentum_x_flux_y, double* momentum_y_flux_x,
                     double* momentum_y_flux_y, double* momentum_x,
                     double* momentum_y, const double* density,
                     const double* mass_flux_x, const double* mass_flux_y,
                     const double* edgedx, const double* edgedy,
                     const double* celldx, const double* celldy) {
  const int pad = mesh->pad;

  if (tt % 2) {
    START_PROFILING(&compute_profile);
    momentum_x_flux_in_x(nx, ny, mesh, dt_h, velocity_x, momentum_x_flux_x,
                         mass_flux_x, edgedx, edgedy, celldx);
    STOP_PROFILING(&compute_profile, __func__);

    handle_boundary_2d(nx, ny, mesh, momentum_x_flux_x, NO_INVERT, PACK);

    START_PROFILING(&compute_profile);
    RAJA::forall<exec_policy>(RAJA::RangeSegment(0, (nx+1)*ny), [=] RAJA_DEVICE (int i) {
      const int ii = i / (nx+1); 
      const int jj = i % (nx+1);
      if(ii >= pad && ii < ny-pad && jj >= pad && jj < (nx+1)-pad) {
        momentum_x[(ii * (nx + 1) + jj)] -=
            dt_h * (momentum_x_flux_x[(ii * nx + jj)] -
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
    });
    STOP_PROFILING(&compute_profile, __func__);

    handle_boundary_2d(nx + 1, ny, mesh, velocity_x, INVERT_X, PACK);

    START_PROFILING(&compute_profile);
    momentum_x_flux_in_y(nx, ny, mesh, dt_h, velocity_x, velocity_y,
                         momentum_x_flux_y, mass_flux_y, edgedx, edgedy,
                         celldy);
    STOP_PROFILING(&compute_profile, __func__);

    handle_boundary_2d(nx + 1, ny + 1, mesh, momentum_x_flux_y, NO_INVERT,
                       PACK);

    START_PROFILING(&compute_profile);
    // Calculate the axial momentum
    RAJA::forall<exec_policy>(RAJA::RangeSegment(0, (nx+1)*ny), [=] RAJA_DEVICE (int i) {
      const int ii = i / (nx+1); 
      const int jj = i % (nx+1);
      if(ii >= pad && ii < ny-pad && jj >= pad && jj < (nx+1)-pad) {
        momentum_x[(ii * (nx + 1) + jj)] -=
            dt_h * (momentum_x_flux_y[(ii * (nx + 1) + jj) + (nx + 1)] -
                    momentum_x_flux_y[(ii * (nx + 1) + jj)]) /
            (celldx[jj] * edgedy[ii]);
      }
    });

    momentum_y_flux_in_x(nx, ny, mesh, dt_h, velocity_x, velocity_y,
                         momentum_y_flux_x, mass_flux_x, edgedx, celldy,
                         celldx);
    STOP_PROFILING(&compute_profile, __func__);

    handle_boundary_2d(nx + 1, ny + 1, mesh, momentum_y_flux_x, NO_INVERT,
                       PACK);

    START_PROFILING(&compute_profile);
    RAJA::forall<exec_policy>(RAJA::RangeSegment(0, nx*(ny+1)), [=] RAJA_DEVICE (int i) {
      const int ii = i / nx; 
      const int jj = i % nx;
      if(ii >= pad && ii < (ny+1)-pad && jj >= pad && jj < nx-pad) {
        momentum_y[(ii * nx + jj)] -=
            dt_h * (momentum_y_flux_x[(ii * (nx + 1) + jj) + 1] -
                    momentum_y_flux_x[(ii * (nx + 1) + jj)]) /
            (edgedx[jj] * celldy[ii]);
        const double density_edge_y =
            (density[(ii * nx + jj)] * celldx[jj] * celldy[ii] +
             density[(ii * nx + jj) - nx] * celldx[jj] * celldy[ii - 1]) /
            (2.0 * celldx[jj] * edgedy[ii]);
        velocity_y[(ii * nx + jj)] =
            (density_edge_y == 0.0)
                ? 0.0
                : momentum_y[(ii * nx + jj)] / density_edge_y;
      }
    });
    STOP_PROFILING(&compute_profile, __func__);

    handle_boundary_2d(nx, ny + 1, mesh, velocity_y, INVERT_Y, PACK);

    START_PROFILING(&compute_profile);
    momentum_y_flux_in_y(nx, ny, mesh, dt_h, velocity_y, momentum_y_flux_y,
                         mass_flux_y, edgedy, celldx, celldy);
    STOP_PROFILING(&compute_profile, __func__);

    handle_boundary_2d(nx, ny, mesh, momentum_y_flux_y, NO_INVERT, PACK);

    START_PROFILING(&compute_profile);
    RAJA::forall<exec_policy>(RAJA::RangeSegment(0, nx*(ny+1)), [=] RAJA_DEVICE (int i) {
      const int ii = i / nx; 
      const int jj = i % nx;
      if(ii >= pad && ii < (ny+1)-pad && jj >= pad && jj < nx-pad) {
        momentum_y[(ii * nx + jj)] -= dt_h *
                                      (momentum_y_flux_y[(ii * nx + jj)] -
                                       momentum_y_flux_y[(ii * nx + jj) - nx]) /
                                      (celldx[jj] * edgedy[ii]);
      }
    });
    STOP_PROFILING(&compute_profile, __func__);
  } else {
    START_PROFILING(&compute_profile);
    momentum_x_flux_in_y(nx, ny, mesh, dt_h, velocity_x, velocity_y,
                         momentum_x_flux_y, mass_flux_y, edgedx, edgedy,
                         celldy);
    STOP_PROFILING(&compute_profile, __func__);

    handle_boundary_2d(nx + 1, ny + 1, mesh, momentum_x_flux_y, NO_INVERT,
                       PACK);

    START_PROFILING(&compute_profile);
    // Calculate the axial momentum
    RAJA::forall<exec_policy>(RAJA::RangeSegment(0, (nx+1)*ny), [=] RAJA_DEVICE (int i) {
      const int ii = i / (nx+1); 
      const int jj = i % (nx+1);
      if(ii >= pad && ii < ny-pad && jj >= pad && jj < (nx+1)-pad) {
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
    });
    STOP_PROFILING(&compute_profile, __func__);

    handle_boundary_2d(nx + 1, ny, mesh, velocity_x, INVERT_X, PACK);

    START_PROFILING(&compute_profile);
    momentum_x_flux_in_x(nx, ny, mesh, dt_h, velocity_x, momentum_x_flux_x,
                         mass_flux_x, edgedx, edgedy, celldx);
    STOP_PROFILING(&compute_profile, __func__);

    handle_boundary_2d(nx, ny, mesh, momentum_x_flux_x, NO_INVERT, PACK);

    START_PROFILING(&compute_profile);
    RAJA::forall<exec_policy>(RAJA::RangeSegment(pad, ny-pad), [=] RAJA_DEVICE (int ii) {
#pragma omp simd
      for (int jj = pad; jj < (nx + 1) - pad; ++jj) {
        momentum_x[(ii * (nx + 1) + jj)] -=
            dt_h * (momentum_x_flux_x[(ii * nx + jj)] -
                    momentum_x_flux_x[(ii * nx + jj) - 1]) /
            (edgedx[jj] * celldy[ii]);
      }
    });

    momentum_y_flux_in_y(nx, ny, mesh, dt_h, velocity_y, momentum_y_flux_y,
                         mass_flux_y, edgedy, celldx, celldy);
    STOP_PROFILING(&compute_profile, __func__);

    handle_boundary_2d(nx, ny, mesh, momentum_y_flux_y, NO_INVERT, PACK);

    START_PROFILING(&compute_profile);
    RAJA::forall<exec_policy>(RAJA::RangeSegment(pad, (ny+1)-pad), [=] RAJA_DEVICE (int ii) {
#pragma omp simd
      for (int jj = pad; jj < nx - pad; ++jj) {
        momentum_y[(ii * nx + jj)] -= dt_h *
                                      (momentum_y_flux_y[(ii * nx + jj)] -
                                       momentum_y_flux_y[(ii * nx + jj) - nx]) /
                                      (celldx[jj] * edgedy[ii]);
        const double density_edge_y =
            (density[(ii * nx + jj)] * celldx[jj] * celldy[ii] +
             density[(ii * nx + jj) - nx] * celldx[jj] * celldy[ii - 1]) /
            (2.0 * celldx[jj] * edgedy[ii]);
        velocity_y[(ii * nx + jj)] =
            (density_edge_y == 0.0)
                ? 0.0
                : momentum_y[(ii * nx + jj)] / density_edge_y;
      }
    });
    STOP_PROFILING(&compute_profile, __func__);

    handle_boundary_2d(nx, ny + 1, mesh, velocity_y, INVERT_Y, PACK);

    START_PROFILING(&compute_profile);
    momentum_y_flux_in_x(nx, ny, mesh, dt_h, velocity_x, velocity_y,
                         momentum_y_flux_x, mass_flux_x, edgedx, celldy,
                         celldx);
    STOP_PROFILING(&compute_profile, __func__);

    handle_boundary_2d(nx + 1, ny + 1, mesh, momentum_y_flux_x, NO_INVERT,
                       PACK);

    START_PROFILING(&compute_profile);
    RAJA::forall<exec_policy>(RAJA::RangeSegment(pad, (ny+1)-pad), [=] RAJA_DEVICE (int ii) {
#pragma omp simd
      for (int jj = pad; jj < nx - pad; ++jj) {
        momentum_y[(ii * nx + jj)] -=
            dt_h * (momentum_y_flux_x[(ii * (nx + 1) + jj) + 1] -
                    momentum_y_flux_x[(ii * (nx + 1) + jj)]) /
            (edgedx[jj] * celldy[ii]);
      }
    });
    STOP_PROFILING(&compute_profile, __func__);
  }
}

// Calculates the x momentum flux along the x dimension
void momentum_x_flux_in_x(const int nx, const int ny, Mesh* mesh,
                          const double dt_h, double* velocity_x,
                          double* momentum_x_flux_x, const double* mass_flux_x,
                          const double* edgedx, const double* edgedy,
                          const double* celldx) {

  const int pad = mesh->pad;

// Calculate the cell centered x momentum fluxes in the x direction
  RAJA::forall<exec_policy>(RAJA::RangeSegment(pad, ny-pad), [=] RAJA_DEVICE (int ii) {
#pragma omp simd
    for (int jj = pad; jj < nx - pad; ++jj) {
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
      const double F_x =
          edgedy[ii] * 0.5 * (mass_flux_x[(ii * (nx + 1) + jj)] +
                              mass_flux_x[(ii * (nx + 1) + jj) + 1]);
      const double u_cell_x_interp =
          (u_cell_x > 0.0)
              ? velocity_x[(ii * (nx + 1) + jj)] +
                    0.5 * minmod(minmod(a_x_0, b_x_0), c_x_0) *
                        (celldx[jj - 1] - u_cell_x * dt_h)
              : velocity_x[(ii * (nx + 1) + jj) + 1] -
                    0.5 * minmod(minmod(a_x_1, b_x_1), c_x_1) *
                        (celldx[jj] + u_cell_x * dt_h);
      momentum_x_flux_x[(ii * nx + jj)] = F_x * u_cell_x_interp;
    }
  });
}

// Calculates the x momentum flux in the y dimension
void momentum_x_flux_in_y(const int nx, const int ny, Mesh* mesh,
                          const double dt_h, double* velocity_x,
                          double* velocity_y, double* momentum_x_flux_y,
                          const double* mass_flux_y, const double* edgedx,
                          const double* edgedy, const double* celldy) {

  const int pad = mesh->pad;

  RAJA::forall<exec_policy>(RAJA::RangeSegment(pad, (ny+1)-pad), [=] RAJA_DEVICE (int ii) {
#pragma omp simd
    for (int jj = pad; jj < (nx + 1) - pad; ++jj) {
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

      const double F_y = edgedx[jj] * 0.5 * (mass_flux_y[(ii * nx + jj)] +
                                             mass_flux_y[(ii * nx + jj) - 1]);
      const double u_corner_y =
          (v_cell_y > 0.0)
              ? velocity_x[(ii * (nx + 1) + jj) - (nx + 1)] +
                    0.5 * minmod(minmod(a_y_0, b_y_0), c_y_0) *
                        (celldy[ii - 1] - v_cell_y * dt_h)
              : velocity_x[(ii * (nx + 1) + jj)] -
                    0.5 * minmod(minmod(a_y_1, b_y_1), c_y_1) *
                        (celldy[ii] + v_cell_y * dt_h);
      momentum_x_flux_y[(ii * (nx + 1) + jj)] = F_y * u_corner_y;
    }
  });
}

// Calculates the y momentum flux in the x dimension
void momentum_y_flux_in_x(const int nx, const int ny, Mesh* mesh,
                          const double dt_h, const double* velocity_x,
                          double* velocity_y, double* momentum_y_flux_x,
                          const double* mass_flux_x, const double* edgedx,
                          const double* celldy, const double* celldx) {

  const int pad = mesh->pad;

// Calculate the corner centered y momentum fluxes in the x direction
// Calculate the cell centered y momentum fluxes in the y direction
  RAJA::forall<exec_policy>(RAJA::RangeSegment(pad, (ny+1)-pad), [=] RAJA_DEVICE (int ii) {
#pragma omp simd
    for (int jj = pad; jj < (nx + 1) - pad; ++jj) {

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
      const double F_x =
          celldy[ii] * 0.5 * (mass_flux_x[(ii * (nx + 1) + jj)] +
                              mass_flux_x[(ii * (nx + 1) + jj) - (nx + 1)]);
      const double u_cell_x =
          0.5 * (velocity_x[(ii * (nx + 1) + jj)] +
                 velocity_x[(ii * (nx + 1) + jj) - (nx + 1)]);
      const double v_cell_x_interp =
          (u_cell_x > 0.0)
              ? velocity_y[(ii * nx + jj) - 1] +
                    0.5 * minmod(minmod(a_x_0, b_x_0), c_x_0) *
                        (celldx[jj - 1] - u_cell_x * dt_h)
              : velocity_y[(ii * nx + jj)] -
                    0.5 * minmod(minmod(a_x_1, b_x_1), c_x_1) *
                        (celldx[jj] + u_cell_x * dt_h);
      momentum_y_flux_x[(ii * (nx + 1) + jj)] = F_x * v_cell_x_interp;
    }
  });
}

// Calculates the y momentum flux in the y dimension
void momentum_y_flux_in_y(const int nx, const int ny, Mesh* mesh,
                          const double dt_h, double* velocity_y,
                          double* momentum_y_flux_y, const double* mass_flux_y,
                          const double* edgedy, const double* celldx,
                          const double* celldy) {

  const int pad = mesh->pad;

  RAJA::forall<exec_policy>(RAJA::RangeSegment(pad, ny-pad), [=] RAJA_DEVICE (int ii) {
#pragma omp simd
    for (int jj = pad; jj < nx - pad; ++jj) {
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

      const double F_y = celldx[jj] * 0.5 * (mass_flux_y[(ii * nx + jj)] +
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
      momentum_y_flux_y[(ii * nx + jj)] = F_y * v_cell_y_interp;
    }
  });
}

// Prints some conservation values
void print_conservation(const int nx, const int ny, double* density,
                        double* energy, double* reduce_array, Mesh* mesh) {

  const int pad = mesh->pad;

  RAJA::ReduceSum<reduce_policy, double> mass_tot(0.0);
  RAJA::ReduceSum<reduce_policy, double> energy_tot(0.0);
  RAJA::forall<exec_policy>(RAJA::RangeSegment(pad, ny-pad), [=] RAJA_DEVICE (int ii) {
    for (int jj = pad; jj < nx - pad; ++jj) {
      mass_tot += density[(ii * nx + jj)];
      energy_tot += density[(ii * nx + jj)] * energy[(ii * nx + jj)];
    }
  });

  double global_mass_tot = reduce_to_master(mass_tot.get());
  double global_energy_tot = reduce_to_master(energy_tot.get());

  if (mesh->rank == MASTER) {
    printf("Total mass:      %.12e\n", global_mass_tot);
    printf("Total energy:    %.12e\n", global_energy_tot);
  }
}
