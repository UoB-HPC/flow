#include <stdlib.h>
#include "../../shared.h"
#include "../../mesh.h"

// Allocates some double precision data
void allocate_data(double** buf, size_t len)
{
#ifdef INTEL
  *buf = (double*)_mm_malloc(sizeof(double)*len, VEC_ALIGN);
#else
  *buf = (double*)malloc(sizeof(double)*len);
#endif

  double* b = *buf;
#pragma omp target enter data map(to: b[:len])
}

// Allocates a data array
void deallocate_data(double* buf)
{
#ifdef INTEL
  _mm_free(buf);
#else
  free(buf);
#endif
}

// Fetches data back from the device
void fetch_data(const int nx, const int ny, double* arr)
{
#pragma omp target update from(arr[:nx*ny])
}

// Initialises mesh data in device specific manner
void mesh_data_init(
    const int local_nx, const int local_ny, const int global_nx, const int global_ny,
    double* edgedx, double* edgedy, double* celldx, double* celldy)
{
  // Simple uniform rectilinear initialisation
#pragma omp target teams distribute parallel for
  for(int ii = 0; ii < local_ny+1; ++ii) {
    edgedy[ii] = 10.0 / (global_ny);
  }
#pragma omp target teams distribute parallel for
  for(int ii = 0; ii < local_ny; ++ii) {
    celldy[ii] = 10.0 / (global_ny);
  }
#pragma omp target teams distribute parallel for
  for(int ii = 0; ii < local_nx+1; ++ii) {
    edgedx[ii] = 10.0 / (global_nx);
  }
#pragma omp target teams distribute parallel for
  for(int ii = 0; ii < local_nx; ++ii) {
    celldx[ii] = 10.0 / (global_nx);
  }
}

// Initialise state data in device specific manner
void state_data_init(
    const int local_nx, const int local_ny, const int global_nx, const int global_ny,
    const int x_off, const int y_off,
    double* rho, double* e, double* rho_old, double* P, double* Qxx, double* Qyy,
    double* x, double* p, double* rho_u, double* rho_v, double* F_x, double* F_y,
    double* uF_x, double* uF_y, double* vF_x, double* vF_y)
{
  // Initialise all of the state to 0, but is this best for NUMA?
//#pragma omp parallel for
#pragma omp target teams distribute parallel for simd
  for(int ii = 0; ii < local_nx*local_ny; ++ii) {
    rho[ii] = 0.0;
    e[ii] = 0.0;
    rho_old[ii] = 0.0;
    P[ii] = 0.0;
  }

#pragma omp target teams distribute parallel for simd
  for(int ii = 0; ii < (local_nx+1)*(local_ny+1); ++ii) {
    Qxx[ii] = 0.0;
    Qyy[ii] = 0.0;
    x[ii] = 0.0;
    p[ii] = 0.0;
    rho_u[ii] = 0.0;
    rho_v[ii] = 0.0;
    F_x[ii] = 0.0;
    F_y[ii] = 0.0;
    uF_x[ii] = 0.0;
    uF_y[ii] = 0.0;
    vF_x[ii] = 0.0;
    vF_y[ii] = 0.0;
  }

  // TODO: Improve what follows, make it a somewhat more general problem 
  // selection mechanism for some important stock problems

  // WET STATE INITIALISATION
  // Initialise a default state for the energy and density on the mesh
#pragma omp target teams distribute parallel for simd
  for(int ii = 0; ii < local_ny; ++ii) {
    for(int jj = 0; jj < local_nx; ++jj) {
      rho[ii*local_nx+jj] = 0.125;
      e[ii*local_nx+jj] = 2.0;
      x[ii*local_nx+jj] = rho[ii*local_nx+jj]*0.1;
    }
  }

  // Introduce a problem
#pragma omp target teams distribute parallel for simd
  for(int ii = 0; ii < local_ny; ++ii) {
    for(int jj = 0; jj < local_nx; ++jj) {
      // CENTER SQUARE TEST
      if(jj+x_off >= (global_nx+2*PAD)/2-(global_nx/5) && 
          jj+x_off < (global_nx+2*PAD)/2+(global_nx/5) && 
          ii+y_off >= (global_ny+2*PAD)/2-(global_ny/5) && 
          ii+y_off < (global_ny+2*PAD)/2+(global_ny/5)) {
        rho[ii*local_nx+jj] = 1.0;
        e[ii*local_nx+jj] = 2.5;
        x[ii*local_nx+jj] = rho[ii*local_nx+jj]*0.1;
      }

#if 0
      // OFF CENTER SQUARE TEST
      const int dist = 100;
      if(jj+x_off-PAD >= global_nx/4-dist && 
          jj+x_off-PAD < global_nx/4+dist && 
          ii+y_off-PAD >= global_ny/2-dist && 
          ii+y_off-PAD < global_ny/2+dist) {
        rho[ii*local_nx+jj] = 1.0;
        e[ii*local_nx+jj] = 2.5;
        x[ii*local_nx+jj] = rho[ii*local_nx+jj]*e[ii*local_nx+jj];
      }
#endif // if 0

#if 0
      if(jj+x_off < ((global_nx+2*PAD)/2)) {
        rho[ii*local_nx+jj] = 1.0;
        e[ii*local_nx+jj] = 2.5;
        x[ii*local_nx+jj] = rho[ii*local_nx+jj]*0.1;
      }
#endif // if 0

#if 0
      if(ii+y_off < (global_ny+2*PAD)/2) {
        rho[ii*local_nx+jj] = 1.0;
        e[ii*local_nx+jj] = 2.5;
      }
#endif // if 0

#if 0
      if(ii+y_off > (global_ny+2*PAD)/2) {
        rho[ii*local_nx+jj] = 1.0;
        e[ii*local_nx+jj] = 2.5;
      }
#endif // if 0

#if 0
      if(jj+x_off > (global_nx+2*PAD)/2) {
        rho[ii*local_nx+jj] = 1.0;
        e[ii*local_nx+jj] = 2.5;
      }
#endif // if 0
    }
  }
}
