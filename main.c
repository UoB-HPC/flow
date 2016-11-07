#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "main.h"
#include "profiler.h"

#define ind0 (ii*nx + jj)
#define ind1 (ii*(nx+1) + jj)

int main(int argc, char** argv)
{
  if(argc != 4) {
    printf("usage: ./hydro.exe <nx> <ny> <niters>\n");
    exit(1);
  }

  const int nx = atoi(argv[1]) + 2*PAD;
  const int ny = atoi(argv[2]) + 2*PAD;
  const int niters = atoi(argv[3]);

  printf("Problem dimensions %dx%d for %d iterations.\n", nx-2*PAD, ny-2*PAD, niters);

  State state;
  Mesh mesh;
  initialise_mesh(nx, ny, &mesh);
  initialise_state(nx, ny,&state, &mesh);

  double wallclock = 0.0;
  double elapsed_sim_time = 0.0;

  set_timestep(
      nx, ny, state.Qxx, state.Qyy, state.rho, state.u, state.v, state.e, 
      &mesh, 0, mesh.edgedx, mesh.edgedy, mesh.celldx, mesh.celldy);

  struct Profile p;
  p.profiler_entry_count = 0;

  int tt;
  for(tt = 0; tt < niters; ++tt) {
    printf("Iteration %d\n", tt+1);

    const double s1 = omp_get_wtime();

    if(tt % VISIT_STEP == 0) {
      write_to_visit(nx, ny, state.rho, "density", tt, elapsed_sim_time);
    }

    START_PROFILING(&p);
    equation_of_state(
        nx, ny, state.P, state.rho, state.e);
    STOP_PROFILING(&p, "equation_of_state");

    START_PROFILING(&p);
    lagrangian_step(
        nx, ny, mesh.dt, state.rho_u, state.rho_v, state.u, state.v, state.P, state.rho,
        mesh.edgedx, mesh.edgedy, mesh.celldx, mesh.celldy);
    STOP_PROFILING(&p, "lagrangian_step");

    START_PROFILING(&p);
    artificial_viscosity(
        nx, ny, mesh.dt, state.Qxx, state.Qyy, state.u, state.v, state.rho_u, 
        state.rho_v, state.rho, mesh.edgedx, mesh.edgedy, mesh.celldx, mesh.celldy);
    STOP_PROFILING(&p, "artificial_viscosity");

    START_PROFILING(&p);
    set_timestep(
        nx, ny, state.Qxx, state.Qyy, state.rho, state.u, state.v, state.e, 
        &mesh, tt == 0, mesh.edgedx, mesh.edgedy, mesh.celldx, mesh.celldy);
    STOP_PROFILING(&p, "set_timestep");

    START_PROFILING(&p);
    shock_heating_and_work(
        nx, ny, mesh.dt, state.e, state.P, state.u, state.v, state.rho, state.Qxx,
        state.Qyy, mesh.edgedx, mesh.edgedy, mesh.celldx, mesh.celldy);
    STOP_PROFILING(&p, "shock_heating_and_work");

    // Perform advection
    START_PROFILING(&p);
    advect_mass(
        nx, ny, mesh.dt_h, state.rho, state.rho_old, state.slope_x0, state.slope_y0,
        state.F_x, state.F_y,
        state.u, state.v, mesh.edgedx, mesh.edgedy, mesh.celldx, mesh.celldy);
    STOP_PROFILING(&p, "advect_mass");

    START_PROFILING(&p);
    advect_momentum(
        nx, ny, mesh.dt_h, mesh.dt, state.u, state.v, state.slope_x1, 
        state.slope_y1, state.mF_x, state.mF_y, state.rho_u, state.rho_v, 
        state.rho, state.F_x, state.F_y,
        mesh.edgedx, mesh.edgedy, mesh.celldx, mesh.celldy);
    STOP_PROFILING(&p, "advect_momentum");

    START_PROFILING(&p);
    advect_energy(
        nx, ny, mesh.dt_h, mesh.dt, state.e, state.slope_x0, state.slope_y0, 
        state.F_x, state.F_y, state.u, state.v, state.rho_old, state.rho,
        mesh.edgedx, mesh.edgedy, mesh.celldx, mesh.celldy);
    STOP_PROFILING(&p, "advect_energy");

    const double s2 = omp_get_wtime();
    wallclock += s2-s1;

    elapsed_sim_time += mesh.dt;

    if(elapsed_sim_time >= SIM_END) {
      printf("reached end of simulation time\n");
      break;
    }

    print_conservation(nx, ny, &state);
  }

  PRINT_PROFILING_RESULTS(&p);

  printf("Wallclock %.2fs, Elapsed Simulation Time %.4fs\n", 
      wallclock, elapsed_sim_time);

  write_to_visit(nx+1, ny, state.u, "u", tt, elapsed_sim_time);
  write_to_visit(nx, ny, state.rho, "density", tt, elapsed_sim_time);

  finalise_state(&state);
  finalise_mesh(&mesh);

  return 0;
}

// Calculate the pressure from GAMma law equation of state
static inline void equation_of_state(
    const int nx, const int ny, double* P, const double* rho, const double* e)
{
#pragma omp parallel for
  for(int ii = 0; ii < ny; ++ii) {
    for(int jj = 0; jj < nx; ++jj) {
      // Only invoke simple GAMma law at the moment
      P[ind0] = (GAM - 1.0)*rho[ind0]*e[ind0];
    }
  }
}

// Calculates the timestep from the cuyyent state
static inline void set_timestep(
    const int nx, const int ny, double* Qxx, double* Qyy, const double* rho, 
    const double* u, const double* v, const double* e, Mesh* mesh, const int first_step,
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  double min_dt = MAX_DT;

  // Check the minimum timestep from the sound speed in the nx and ny directions
#pragma omp parallel for reduction(min: min_dt)
  for(int ii = PAD; ii < ny-PAD+1; ++ii) {
    for(int jj = PAD; jj < nx-PAD+1; ++jj) {
      // Constrain based on the artificial viscous stresses
      if(Qxx[ind0] != 0.0)
        min_dt = min(min_dt, 0.25*edgedx[jj]*sqrt(rho[ind0]/Qxx[ind0]));
      if(Qyy[ind0] != 0.0)
        min_dt = min(min_dt, 0.25*edgedy[ii]*sqrt(rho[ind0]/Qyy[ind0]));

      // Constrain based on the sound speed within the system
      const double c_s = sqrt(GAM*(GAM - 1.0)*e[ind0]);

      // TODO: possible DBZ
      min_dt = min(min_dt, (celldx[jj]/(fabs(u[ind1]) + c_s)));
      min_dt = min(min_dt, (celldy[ii]/(fabs(v[ind0]) + c_s)));
    }
  }

  mesh->dt = 0.5*(C_T*min_dt + mesh->dt_h);
  mesh->dt_h = (first_step) ? mesh->dt : C_T*min_dt;

  printf("dt %.12e dt_h %.12e\n", mesh->dt, mesh->dt_h);
}

// Calculate change in momentum caused by pressure gradients, and then extract
// the velocities using edge centered density approximations
static inline void lagrangian_step(
    const int nx, const int ny, const double dt, double* rho_u, double* rho_v, 
    double* u, double* v, const double* P, const double* rho, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
#pragma omp parallel for
  for(int ii = PAD; ii < ny-PAD+1; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD+1; ++jj) {
      // Update the momenta using the pressure gradients
      rho_u[ind1] -= dt*(P[ind0] - P[ind0-1])/(edgedx[jj]);
      rho_v[ind0] -= dt*(P[ind0] - P[ind0-nx])/(edgedy[ii]);

      // Calculate the zone edge centered density
      const double rho_edge_x = 
        (rho[ind0]*celldx[jj]*celldy[ii] + 
         rho[ind0-1]*celldx[jj - 1]*celldy[ii]) / 
        (2.0*edgedx[jj]*celldy[ii]);
      const double rho_edge_y = 
        (rho[ind0]*celldx[jj]*celldy[ii] + 
         rho[ind0-nx]*celldx[jj]*celldy[ii - 1]) / 
        (2.0*celldx[jj]*edgedy[ii]);

      // Find the velocities from the momenta and edge centered mass densities
      u[ind1] = (rho_edge_x == 0.0) ? 0.0 : rho_u[ind1] / rho_edge_x;
      v[ind0] = (rho_edge_y == 0.0) ? 0.0 : rho_v[ind0] / rho_edge_y;
    }
  }

  reflective_boundary(nx+1, ny, 1, u, INVERT_X);
  reflective_boundary(nx, ny+1, 1, v, INVERT_Y);
}

static inline void artificial_viscosity(
    const int nx, const int ny, const double dt, double* Qxx, double* Qyy, double* u, double* v, 
    double* rho_u, double* rho_v, const double* rho, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  // Calculate the artificial viscous stresses
#pragma omp parallel for 
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      const double u_diff = (u[ind1+1] - u[ind1]);
      const double v_diff = (v[ind0+nx] - v[ind0]);
      Qxx[ind0] = (u_diff >= 0.0) ? 0.0 : C_Q*rho[ind0]*u_diff*u_diff;
      Qyy[ind0] = (v_diff >= 0.0) ? 0.0 : C_Q*rho[ind0]*v_diff*v_diff;
    }
  }

  reflective_boundary(nx, ny, 1, Qxx, NO_INVERT);
  reflective_boundary(nx, ny, 1, Qyy, NO_INVERT);

  // Update the momenta by the artificial viscous stresses
#pragma omp parallel for
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      rho_u[ind1] -= (dt/(edgedx[jj]*celldy[ii]))*
        (celldx[jj]*Qxx[ind0] - celldx[jj-1]*Qxx[ind0-1]);
      rho_v[ind0] -= (dt/(edgedx[jj]*celldy[ii]))*
        (celldy[ii]*Qyy[ind0] - celldy[ii-1]*Qyy[ind0-nx]);

      // Calculate the zone edge centered density
      const double rho_edge_x = 
        (rho[ind0]*celldx[jj]*celldy[ii] + rho[ind0-1]*celldx[jj - 1]*celldy[ii]) / 
        (2.0*edgedx[jj]*celldy[ii]);
      const double rho_edge_y = 
        (rho[ind0]*celldx[jj]*celldy[ii] + rho[ind0-nx]*celldx[jj]*celldy[ii - 1]) / 
        (2.0*celldx[jj]*edgedy[ii]);

      // Find the velocities from the momenta and edge centered mass densities
      u[ind1] = (rho_edge_x == 0.0) ? 0.0 : rho_u[ind1] / rho_edge_x;
      v[ind0] = (rho_edge_y == 0.0) ? 0.0 : rho_v[ind0] / rho_edge_y;
    }
  }

  reflective_boundary(nx+1, ny, 1, u, INVERT_X);
  reflective_boundary(nx, ny+1, 1, v, INVERT_Y);
}

// Calculates the work done due to forces within the element
static inline void shock_heating_and_work(
    const int nx, const int ny, const double dt, double* e, const double* P, 
    const double* u, const double* v, const double* rho, const double* Qxx, const double* Qyy,
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  // Compute the change due to shock heating
#pragma omp parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      e[ind0] -= (rho[ind0] == 0.0) ? 0.0 : (dt*
          (Qxx[ind0]*(edgedx[jj+1]*u[ind1+1] - edgedx[jj]*u[ind1])+
           Qyy[ind0]*(edgedy[ii+1]*v[ind0+nx] - edgedy[ii]*v[ind0]))/
          (celldx[jj]*celldy[ii]*rho[ind0]));
    }
  }

#pragma omp parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      if(rho[ind0] == 0.0) {
        e[ind0] = 0.0;
        continue;
      }

      const double div_v = 
        (edgedx[jj + 1]*u[ind1+1] - edgedx[jj]*u[ind1]+
         edgedy[ii + 1]*v[ind0+nx] - edgedy[ii]*v[ind0])/ 
        (celldx[jj]*celldy[ii]);
      const double div_v_dt = div_v*dt;

      /// A working formulation that is second order in time for Pressure!?
      const double e_C = e[ind0] - (P[ind0]*div_v_dt)/rho[ind0];
      const double rho_C = rho[ind0]/(1.0 + div_v_dt);
      e[ind0] -= 0.5*div_v_dt*(P[ind0] + (GAM-1.0)*e_C*rho_C)/rho[ind0];
    }
  }

  reflective_boundary(nx, ny, 1, e, NO_INVERT);
}

// Perform advection with monotonicity improvement
static inline void advect_mass(
    const int nx, const int ny, const double dt_h, double* rho, double* rho_old, 
    double* slope_x, double* slope_y, double* F_x, double* F_y, const double* u, 
    const double* v, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  // Store the current value of rho
#pragma omp parallel for
  for(int ii = 0; ii < nx*ny; ++ii) {
    rho_old[ii] = rho[ii];
  }

#if 0
  // Compute the mass fluxes along the x edges
  // In the ghost cells flux is left as 0.0
#pragma omp parallel for
  for(int ii = PAD+1; ii < ny-(PAD+1); ++ii) {
#pragma omp simd
    for(int jj = PAD+1; jj < (nx+1)-(PAD+1); ++jj) {
      double rx_denom = (rho[ind0]-rho[ind0-1]);
      double rx = 0.0;
      if(rx_denom) {
        rx = (u[ind1] >= 0.0) 
          ? (rho[ind0-1] - rho[ind0-2])/rx_denom
          : (rho[ind0+1] - rho[ind0])/rx_denom;
      }

      //const double limiterx = max(0.0, max(min(1.0, 2.0*rx), min(2.0, rx)));
      const double limiterx = (rx + fabs(rx))/(1.0 + fabs(rx));
      const double hsx = (u[ind1] >= 0.0) ? 1.0 : -1.0;
      F_x[ind1] = 
        0.5*u[ind1]*((1.0+hsx)*rho[ind0-1]+(1.0-hsx)*rho[ind0])+
        0.5*fabs(u[ind1])*(1.0-fabs((u[ind1]*dt_h)/celldx[jj]))*
        limiterx*rx_denom;
    }
  }

  // Compute the mass flux along the y edges
  // In the ghost cells flux is left as 0.0
#pragma omp parallel for
  for(int ii = PAD+1; ii < (ny+1)-(PAD+1); ++ii) {
#pragma omp simd
    for(int jj = PAD+1; jj < nx-(PAD+1); ++jj) {
      double ry_denom = (rho[ind0]-rho[ind0-nx]);
      double ry = 0.0;
      if(ry_denom) {
        ry = (v[ind0] >= 0.0) 
          ? (rho[ind0-nx] - rho[ind0-2*nx])/ry_denom 
          : (rho[ind0+nx] - rho[ind0])/ry_denom;
      }

      //const double limitery = max(0.0, max(min(1.0, 2.0*ry), min(2.0, ry)));
      const double limitery = (ry + fabs(ry))/(1.0 + fabs(ry));
      const double hsy = (v[ind0] >= 0.0) ? 1.0 : -1.0;
      F_y[ind0] =
        0.5*v[ind0]*((1.0+hsy)*rho[ind0-nx]+(1.0-hsy)*rho[ind0])+
        0.5*fabs(v[ind0])*(1.0-fabs((v[ind0]*dt_h)/celldx[jj]))*
        limitery*ry_denom;
    }
  }
#endif // if 0


  /// TODO: What happens about the slopes defined on the halo regions, 
  // is it OK that they are simply set to 0???
#pragma omp parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      // Calculate the maximum and minimum neighbouring density
      const double rho_x_max = max(rho[ind0-1], max(rho[ind0], rho[ind0+1]));
      const double rho_x_min = min(rho[ind0-1], min(rho[ind0], rho[ind0+1]));
      const double rho_y_max = max(rho[ind0-nx], max(rho[ind0], rho[ind0+nx]));
      const double rho_y_min = min(rho[ind0-nx], min(rho[ind0], rho[ind0+nx]));

      // Construct absolute value of slope S_L or S_R
      const double s1_x = 
        min(rho_x_max - rho[ind0], rho[ind0] - rho_x_min) / (celldx[jj] / 2.0);
      const double s1_y = 
        min(rho_y_max - rho[ind0], rho[ind0] - rho_y_min) / (celldy[ii] / 2.0);

      // Calculate the density interpolated from the zone center to zone boundary
      const double rho_edge_l = 
        (celldx[jj]*rho[ind0-1] + celldx[jj - 1]*rho[ind0])/(celldx[jj] + celldx[jj - 1]);
      const double rho_edge_r = 
        (celldx[jj + 1]*rho[ind0] + celldx[jj]*rho[ind0+1])/(celldx[jj + 1] + celldx[jj]);
      const double rho_edge_d = 
        (celldy[ii]*rho[ind0-nx] + celldy[ii - 1]*rho[ind0])/(celldy[ii] + celldy[ii - 1]);
      const double rho_edge_u = 
        (celldy[ii + 1]*rho[ind0] + celldy[ii]*rho[ind0+nx])/(celldy[ii + 1] + celldy[ii]);

      // Construct the slope
      const double s2_x = (rho_edge_r - rho_edge_l) / celldx[jj];
      const double s2_y = (rho_edge_u - rho_edge_d) / celldy[ii];

      // Define the zone centered slope (culling 0's)
      slope_x[ind0] = (s2_x != 0.0) ? (s2_x / fabs(s2_x))*min(fabs(s2_x), s1_x) : 0.0;
      slope_y[ind0] = (s2_y != 0.0) ? (s2_y / fabs(s2_y))*min(fabs(s2_y), s1_y) : 0.0;
    }
  }

  // Calculate the zone edge centered energies, and flux
#pragma omp parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      // Calculate the interpolated densities
      const double edge_rho_x = (u[ind1] >= 0.0)
        ? rho[ind0-1] + 0.5*slope_x[ind0-1]*(celldx[jj-1] - u[ind1]*dt_h)
        : rho[ind0] - 0.5*slope_x[ind0]*(celldx[jj] + u[ind1]*dt_h);
      F_x[ind1] = edge_rho_x*u[ind1]; 
    }
  }

  // Calculate the zone edge centered energies, and flux
#pragma omp parallel for
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      // Calculate the interpolated densities
      const double edge_rho_y = (v[ind0] >= 0.0)
        ? rho[ind0-nx] + 0.5*slope_y[ind0-nx]*(celldy[ii-1] - v[ind0]*dt_h)
        : rho[ind0] - 0.5*slope_y[ind0]*(celldy[ii] + v[ind0]*dt_h);
      F_y[ind0] = edge_rho_y*v[ind0]; 
    }
  }

  reflective_boundary(nx+1, ny, 1, F_x, INVERT_X);
  reflective_boundary(nx, ny+1, 1, F_y, INVERT_Y);

  // Calculate the new density values
#pragma omp parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      rho[ind0] -= dt_h*
        (edgedx[jj+1]*F_x[ind1+1] - edgedx[jj]*F_x[ind1] +
         edgedy[ii+1]*F_y[ind0+nx] - edgedy[ii]*F_y[ind0])/
        (celldx[jj]*celldy[ii]);
    }
  }

  reflective_boundary(nx, ny, 1, rho, NO_INVERT);
}

// Advect momentum according to the velocity
static inline void advect_momentum(
    const int nx, const int ny, const double dt_h, const double dt, double* u, 
    double* v, double* slope_x, double* slope_y, double* mF_x, double* mF_y, 
    double* rho_u, double* rho_v, const double* rho, const double* F_x, const double* F_y,
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
#if 0
#pragma omp parallel for
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      // Calculate the zone edge centered density
      const double rho_edge_x = 
        (rho[ind0]*celldx[jj]*celldy[ii] + rho[ind0-1]*celldx[jj - 1]*celldy[ii]) / 
        (2.0*edgedx[jj]*celldy[ii]);
      const double rho_edge_y = 
        (rho[ind0]*celldy[ii]*celldx[jj] + rho[ind0-nx]*celldy[ii - 1]*celldx[jj]) / 
        (2.0*edgedx[jj]*celldy[ii]);
      u[ind1] = (rho_edge_x == 0.0) ? 0.0 : rho_u[ind1] / rho_edge_x;
      v[ind0] = (rho_edge_y == 0.0) ? 0.0 : rho_v[ind0] / rho_edge_y;
    }
  }

  reflective_boundary(nx+1, ny, 1, u, INVERT_X);
  reflective_boundary(nx, ny+1, 1, v, INVERT_Y);
#endif // if 0

  /// nx DIMENSION ADVECTION

#pragma omp parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      const double S_L = (u[ind1] - u[ind1-1])/edgedx[jj];
      const double S_R = (u[ind1+1] - u[ind1])/edgedx[jj+1];
      slope_x[ind1] = ((S_L >= 0.0) ^ (S_R < 0.0)) 
        ? (fabs(S_L) < fabs(S_R) ? S_L : S_R) : (0.0);
    }
  }

#pragma omp parallel for
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      const double S_D = (u[ind1] - u[ind1-(nx+1)])/edgedy[ii];
      const double S_U = (u[ind1+(nx+1)] - u[ind1])/edgedy[ii+1];
      slope_y[ind0] = ((S_D >= 0.0) ^ (S_U < 0.0)) 
        ? (fabs(S_D) < fabs(S_U) ? S_D : S_U) : (0.0);
    }
  }

  // Calculate the cell centered x momentum fluxes in the x direction
#pragma omp parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      const double f_x = edgedy[ii]*0.5*(F_x[ind1] + F_x[ind1+1]); 
      const double u_cell_x = 0.5*(u[ind1]+u[ind1+1]);
      mF_x[ind0] = f_x*((u_cell_x >= 0.0) 
          ? u[ind1] + slope_x[ind1]*(0.5*edgedx[jj]-u_cell_x*dt)
          : u[ind1+1] - slope_x[ind1+1]*(0.5*edgedx[jj]+u_cell_x*dt));
    }
  }

  // Calculate the corner center y momentum fluxes in the y direction
#pragma omp parallel for
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      const double f_y = edgedx[jj]*0.5*(F_y[ind0] + F_y[ind0-1]);
      const double v_cell_y = 0.5*(v[ind0]+v[ind0-1]);
      mF_y[ind1] = f_y*((v_cell_y >= 0.0)
          ? u[ind1-(nx+1)] + slope_y[ind0-nx]*(0.5*edgedx[jj]-v_cell_y*dt)
          : u[ind1] - slope_y[ind0]*(0.5*edgedx[jj]+v_cell_y*dt));
    }
  }

  reflective_boundary(nx, ny, 1, mF_x, NO_INVERT);
  reflective_boundary(nx+1, ny+1, 1, mF_y, NO_INVERT);

  // Calculate the axial momentum
#pragma omp parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      rho_u[ind1] -= dt_h*
        ((mF_x[ind0] - mF_x[ind0-1])/(edgedx[jj]*celldy[ii]) +
         (mF_y[ind1+(nx+1)] - mF_y[ind1])/(celldx[jj]*edgedy[ii]));
    }
  }

  // Zero the momentum fluxes for re-use
#pragma omp parallel for
  for(int ii = 0; ii < (nx+1)*(ny+1); ++ii) {
    mF_x[ii] = 0.0;
    mF_y[ii] = 0.0;
  }

  /// ny DIMENSION ADVECTION

  // Calculate the cell centered monotonic slopes for u and v
#pragma omp parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      const double S_L = (v[ind0] - v[ind0-1])/edgedx[jj];
      const double S_R = (v[ind0+1] - v[ind0])/edgedx[jj+1];
      slope_x[ind1] = ((S_L >= 0.0) ^ (S_R < 0.0)) 
        ? (fabs(S_L) < fabs(S_R) ? S_L : S_R) : (0.0);
    }
  }

  // Calculate the cell centered monotonic slopes for u and v
#pragma omp parallel for
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      const double S_D = (v[ind0] - v[ind0-nx])/edgedy[ii];
      const double S_U = (v[ind0+nx] - v[ind0])/edgedy[ii+1];
      slope_y[ind0] = ((S_D >= 0.0) ^ (S_U < 0.0)) 
        ? (fabs(S_D) < fabs(S_U) ? S_D : S_U) : (0.0);
    }
  }

  // Calculate the corner centered y momentum fluxes in the x direction
#pragma omp parallel for
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      const double f_x = celldy[ii]*0.5*(F_x[ind1] + F_x[ind1-(nx+1)]);
      const double u_cell_x = 0.5*(u[ind1]+u[ind1-(nx+1)]);
      mF_x[ind1] = f_x*((u_cell_x >= 0.0) 
          ? v[ind0-1] + slope_x[ind1-1]*(0.5*edgedx[jj]-u_cell_x*dt)
          : v[ind0] - slope_x[ind1]*(0.5*edgedx[jj]+u_cell_x*dt));
    }
  }

  // Calculate the cell centered y momentum fluxes in the y direction
#pragma omp parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      const double f_y = celldx[jj]*0.5*(F_y[ind0] + F_y[ind0+nx]);
      const double v_cell_y = 0.5*(v[ind0]+v[ind0+nx]);
      mF_y[ind0] = f_y*((v_cell_y >= 0.0)
          ? v[ind0] + slope_y[ind0]*(0.5*edgedx[jj]-v_cell_y*dt)
          : v[ind0+nx] - slope_y[ind0+nx]*(0.5*edgedx[jj]+v_cell_y*dt));
    }
  }

  reflective_boundary(nx+1, ny+1, 1, mF_x, NO_INVERT);
  reflective_boundary(nx, ny, 1, mF_y, NO_INVERT);

#pragma omp parallel for
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      rho_v[ind0] -= dt_h*(
          (mF_x[ind1+1] - mF_x[ind1])/(edgedx[jj]*celldy[ii]) +
          (mF_y[ind0] - mF_y[ind0-nx])/(celldx[jj]*edgedy[ii]));
    }
  }

#pragma omp parallel for
  for(int ii = 0; ii < (nx+1)*(ny+1); ++ii) {
    mF_x[ii] = 0.0;
    mF_y[ii] = 0.0;
  }
}

// Perform advection of internal energy
static inline void advect_energy(
    const int nx, const int ny, const double dt_h, const double dt, double* e, 
    double* slope_x, double* slope_y, double* F_x, double* F_y, const double* u, 
    const double* v, const double* rho_old, const double* rho, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy)
{
  /// TODO: What happens about the slopes defined on the halo regions, 
  // is it OK that they are simply set to 0???
#pragma omp parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      // Calculate the maximum and minimum neighbouring density
      const double ie_x_max = max(e[ind0-1], max(e[ind0], e[ind0+1]));
      const double ie_x_min = min(e[ind0-1], min(e[ind0], e[ind0+1]));
      const double ie_y_max = max(e[ind0-nx], max(e[ind0], e[ind0+nx]));
      const double ie_y_min = min(e[ind0-nx], min(e[ind0], e[ind0+nx]));

      // Construct absolute value of slope S_L or S_R
      const double s1_x = min(ie_x_max - e[ind0], e[ind0] - ie_x_min) / (celldx[jj] / 2.0);
      const double s1_y = min(ie_y_max - e[ind0], e[ind0] - ie_y_min) / (celldy[ii] / 2.0);

      // Calculate the density interpolated from the zone center to zone boundary
      const double ie_edge_l = 
        (celldx[jj]*e[ind0-1] + celldx[jj - 1]*e[ind0])/(celldx[jj] + celldx[jj - 1]);
      const double ie_edge_r = 
        (celldx[jj + 1]*e[ind0] + celldx[jj]*e[ind0+1])/(celldx[jj + 1] + celldx[jj]);
      const double ie_edge_d = 
        (celldy[ii]*e[ind0-nx] + celldy[ii - 1]*e[ind0])/(celldy[ii] + celldy[ii - 1]);
      const double ie_edge_u = 
        (celldy[ii + 1]*e[ind0] + celldy[ii]*e[ind0+nx])/(celldy[ii + 1] + celldy[ii]);

      // Construct the slope
      const double s2_x = (ie_edge_r - ie_edge_l) / celldx[jj];
      const double s2_y = (ie_edge_u - ie_edge_d) / celldy[ii];

      // Define the zone centered slope (culling 0's)
      slope_x[ind0] = (s2_x != 0.0) ? (s2_x / fabs(s2_x))*min(fabs(s2_x), s1_x) : 0.0;
      slope_y[ind0] = (s2_y != 0.0) ? (s2_y / fabs(s2_y))*min(fabs(s2_y), s1_y) : 0.0;
    }
  }

  // Calculate the zone edge centered energies, and flux
#pragma omp parallel for
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      // Calculate the interpolated densities
      const double edge_e_x = (u[ind1] > 0.0)
        ? e[ind0-1] + 0.5*slope_x[ind0-1]*(celldx[jj-1] - u[ind1]*dt)
        : e[ind0] - 0.5*slope_x[ind0]*(celldx[jj] + u[ind1]*dt);

      // Update the fluxes to now include the contribution from energy
      F_x[ind1] = edgedy[ii]*edge_e_x*F_x[ind1]; 
    }
  }

  // Calculate the zone edge centered energies, and flux
#pragma omp parallel for
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
#pragma omp simd
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      // Calculate the interpolated densities
      const double edge_e_y = (v[ind0] > 0.0)
        ? e[ind0-nx] + 0.5*slope_y[ind0-nx]*(celldy[ii-1] - v[ind0]*dt)
        : e[ind0] - 0.5*slope_y[ind0]*(celldy[ii] + v[ind0]*dt);

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
         (F_x[ind1+1] - F_x[ind1] + F_y[ind0+nx] - F_y[ind0]))/rho[ind0];
    }
  }

  reflective_boundary(nx, ny, 1, e, NO_INVERT);
}

// Enforce reflective boundary conditions on the problem state
static inline void reflective_boundary(
    const int nx, const int ny, const int depth, double* arr, int invert)
{
  double x_inversion_coeff = (invert == INVERT_X) ? -1.0 : 1.0;

  // reflect at the left
  //#pragma omp parallel for collapse(2)
  for(int ii = 0; ii < ny; ++ii) {
    for(int dd = 0; dd < depth; ++dd) {
      arr[ii*nx + (PAD - 1 - dd)] = x_inversion_coeff*arr[ii*nx + (PAD + dd)];
    }
  }

  // Reflect at the right
  //#pragma omp parallel for collapse(2)
  for(int ii = 0; ii < ny; ++ii) {
    for(int dd = 0; dd < depth; ++dd) {
      arr[ii*nx + (nx - PAD + dd)] = x_inversion_coeff*arr[ii*nx + (nx - 1 - PAD - dd)];
    }
  }

  double y_inversion_coeff = (invert == INVERT_Y) ? -1.0 : 1.0;

  // Reflect at the top
  //#pragma omp parallel for collapse(2)
  for(int dd = 0; dd < depth; ++dd) {
    for(int jj = 0; jj < nx; ++jj) {
      arr[(ny - PAD + dd)*nx + jj] = y_inversion_coeff*arr[(ny - 1 - PAD - dd)*nx + jj];
    }
  }

  // reflect at the bottom
  //#pragma omp parallel for collapse(2)
  for(int dd = 0; dd < depth; ++dd) {
    for(int jj = 0; jj < nx; ++jj) {
      arr[(PAD - 1 - dd)*nx + jj] = y_inversion_coeff*arr[(PAD + dd)*nx + jj];
    }
  }
}

// Initialise the state for the problem
static inline void initialise_state(
    const int nx, const int ny, State* state, Mesh* mesh)
{
  // Allocate memory for all state variables
  state->rho = (double*)malloc(sizeof(double)*nx*ny);
  state->rho_old = (double*)malloc(sizeof(double)*nx*ny);
  state->P = (double*)malloc(sizeof(double)*nx*ny);
  state->e = (double*)malloc(sizeof(double)*nx*ny);
  state->rho_u = (double*)malloc(sizeof(double)*(nx+1)*(ny+1));
  state->rho_v = (double*)malloc(sizeof(double)*(nx+1)*(ny+1));
  state->u = (double*)malloc(sizeof(double)*(nx+1)*(ny+1));
  state->v = (double*)malloc(sizeof(double)*(nx+1)*(ny+1));
  state->F_x = (double*)malloc(sizeof(double)*(nx+1)*(ny+1));
  state->F_y = (double*)malloc(sizeof(double)*(nx+1)*(ny+1));
  state->mF_x = (double*)malloc(sizeof(double)*(nx+1)*(ny+1));
  state->mF_y = (double*)malloc(sizeof(double)*(nx+1)*(ny+1));
  state->slope_x0 = (double*)malloc(sizeof(double)*nx*ny);
  state->slope_y0 = (double*)malloc(sizeof(double)*nx*ny);
  state->slope_x1 = (double*)malloc(sizeof(double)*(nx+1)*(ny+1));
  state->slope_y1 = (double*)malloc(sizeof(double)*(nx+1)*(ny+1));
  state->Qxx = (double*)malloc(sizeof(double)*(nx+1)*(ny+1));
  state->Qyy = (double*)malloc(sizeof(double)*(nx+1)*(ny+1));

  // Initialise all of the memory to default values
#pragma omp parallel for
  for(int ii = 0; ii < nx*ny; ++ii) {
    state->rho[ii] = 0.0;
    state->rho_old[ii] = 0.0;
    state->P[ii] = 0.0;
    state->e[ii] = 0.0;
    state->slope_x0[ii] = 0.0;
    state->slope_y0[ii] = 0.0;
  }

#pragma omp parallel for
  for(int ii = 0; ii < (nx+1)*(ny+1); ++ii) {
    state->F_x[ii] = 0.0;
    state->F_y[ii] = 0.0;
    state->mF_x[ii] = 0.0;
    state->mF_y[ii] = 0.0;
    state->Qxx[ii] = 0.0;
    state->Qyy[ii] = 0.0;
    state->slope_x1[ii] = 0.0;
    state->slope_y1[ii] = 0.0;
    state->rho_u[ii] = 0.0;
    state->u[ii] = 0.0;
    state->v[ii] = 0.0;
    state->rho_v[ii] = 0.0;
  }

  for(int ii = 0; ii < ny; ++ii) {
    for(int jj = 0; jj < nx; ++jj) {
#if 0
      if(jj < nx/2) { //LEFT SOD
        state->rho[ind0] = 1.0;
        state->e[ind0] = 2.5;
      }
#endif // if 0

#if 0
      if(jj >= nx/2) { //RIGHT SOD
        state->rho[ind0] = 1.0;
        state->e[ind0] = 2.5;
      }
#endif // if 0

#if 0
      if(ii < ny/2) { //UP SOD
        state->rho[ind0] = 1.0;
        state->e[ind0] = 2.5;
      }
#endif // if 0

#if 0
      if(ii >= ny/2) { //DOWN SOD
        state->rho[ind0] = 1.0;
        state->e[ind0] = 2.5;
      }
#endif // if 0

#if 0
      // BLUE HOLE TEST 
      const int m = 258;
      const int o = 10400;
      if((ii - m)*(ii - m) + (jj - m)*(jj - m) > o) {
        state->rho[ind0] = 1.0;
        state->e[ind0] = 2.5;
      }
#endif // if 0

      const int dist = 40;
      //if(jj >= nx/2 && jj < nx/2+2*dist && ii >= ny/2 && ii < ny/2+2*dist) {
      if(jj >= nx/2-dist && jj < nx/2+dist && 
          ii >= ny/2-dist && ii < ny/2+dist) {
        state->rho[ind0] = 1.0;
        state->e[ind0] = 2.5;
      }
      else {
        state->rho[ind0] = 0.125;
        state->e[ind0] = 2.0;
      }

#if 0
      // TODO: FIX THIS - CURRENTLY ASSUMING THAT THERE IS NO VELOCITY

      // Calculate the zone edge centered density
      const double rho_edge_x = 
        (state->rho[ind0]*mesh->celldx[jj]*mesh->celldy[ii] + 
         state->rho[ind0-1]*mesh->celldx[jj - 1]*mesh->celldy[ii]) / 
        (2.0*mesh->edgedx[jj]*mesh->celldy[ii]);
      const double rho_edge_y = 
        (state->rho[ind0]*mesh->celldx[jj]*mesh->celldy[ii] + 
         state->rho[ind0-nx]*mesh->celldx[jj]*mesh->celldy[ii - 1]) / 
        (2.0*mesh->celldx[jj]*mesh->edgedy[ii]);

      // Find the velocities from the momenta and edge centered mass densities
      state->rho_u[ind1] = state->u[ind1] * rho_edge_x;
      state->rho_v[ind0] = state->v[ind0] * rho_edge_y;
#endif // if 0
    }
    }
  }

  // Initialise the mesh describing variables
  static inline void initialise_mesh(
      const int nx, const int ny, Mesh* mesh)
  {
    mesh->edgedx = (double*)malloc(sizeof(double)*nx+1);
    mesh->celldx = (double*)malloc(sizeof(double)*nx);
    mesh->edgedy = (double*)malloc(sizeof(double)*ny+1);
    mesh->celldy = (double*)malloc(sizeof(double)*ny);
    mesh->dt = 0.5*C_T*MAX_DT;
    mesh->dt_h = 0.5*C_T*MAX_DT;

    // Simple uniform rectilinear initialisation
    for(int ii = 0; ii < ny+1; ++ii) {
      mesh->edgedy[ii] = 10.0 / (ny-2*PAD);
    }
    for(int ii = 0; ii < ny; ++ii) {
      mesh->celldy[ii] = 10.0 / (ny-2*PAD);
    }
    for(int ii = 0; ii < nx+1; ++ii) {
      mesh->edgedx[ii] = 10.0 / (nx-2*PAD);
    }
    for(int ii = 0; ii < nx; ++ii) {
      mesh->celldx[ii] = 10.0 / (nx-2*PAD);
    }
  }

  // Deallocate all of the state memory
  static inline void finalise_state(State* state)
  {
    free(state->F_x);
    free(state->F_y);
    free(state->rho);
    free(state->rho_old);
    free(state->slope_x0);
    free(state->slope_y0);
    free(state->slope_x1);
    free(state->slope_y1);
    free(state->u);
    free(state->v);
    free(state->P);
    free(state->e);
  }

  // Deallocate all of the mesh memory
  static inline void finalise_mesh(Mesh* mesh)
  {
    free(mesh->edgedy);
    free(mesh->celldy);
    free(mesh->edgedx);
    free(mesh->celldx);
  }

  // Write out data for visualisation in visit
  static inline void write_to_visit(
      const int nx, const int ny, const double* data, 
      const char* name, const int step, const double time)
  {
    char bovname[256];
    char datname[256];
    sprintf(bovname, "%s%d.bov", name, step);
    sprintf(datname, "%s%d.dat", name, step);

    FILE* bovfp = fopen(bovname, "w");

    if(!bovfp) {
      printf("Could not open file %s\n", bovname);
      exit(1);
    }

    fprintf(bovfp, "TIME: %.4f\n", time);
    fprintf(bovfp, "DATA_FILE: %s\n", datname);
    fprintf(bovfp, "DATA_SIZE: %d %d 1\n", nx, ny);
    fprintf(bovfp, "DATA_FORMAT: DOUBLE\n");
    fprintf(bovfp, "VARIABLE: %s\n", name);
    fprintf(bovfp, "DATA_ENDIAN: LITTLE\n");
    fprintf(bovfp, "CENTERING: zone\n");
    fprintf(bovfp, "BRICK_ORIGIN: 0. 0. 0.\n");
    fprintf(bovfp, "BRICK_SIZE: %d %d 1\n", nx, ny);
    fclose(bovfp);

    FILE* datfp = fopen(datname, "wb");
    if(!datfp) {
      printf("Could not open file %s\n", datname);
      exit(1);
    }

    fwrite(data, nx*ny, sizeof(data), datfp);
    fclose(datfp);
  }

  static inline void print_conservation(
      const int nx, const int ny, State* state) {
    double mass_tot = 0.0;
    double energy_tot = 0.0;
    double S_tot = 0.0;
    double T_tot = 0.0;
    for(int ii = PAD; ii < ny-PAD; ++ii) {
      for(int jj = PAD; jj < nx-PAD; ++jj) {
        mass_tot += state->rho[ind0];
        energy_tot += state->rho[ind0]*state->e[ind0];
        S_tot += state->rho_u[ind1];
        T_tot += state->rho_v[ind0];
      }
    }
    printf("total mass: %.12e\n", mass_tot);
    printf("total energy: %.12e\n", energy_tot);
#if 0
    printf("total S: %.12e\n", S_tot);
    printf("total T: %.12e\n", T_tot);
#endif // if 0
  }

