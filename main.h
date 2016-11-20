//#define MPI
#include <mpi.h>

// Controllable parameters for the application
#define GAM 1.4
#define PAD 2
#define MAX_DT 0.04
#define C_Q 1.0 // Suggested between 1.0 and 2.0
#define C_T 0.5
#define VISIT_STEP 10
#define SIM_END 100.0
#define LOAD_BALANCE 0
#define VEC_ALIGN 32

#define max(a, b) (((a) > (b)) ? (a) : (b))
#define min(a, b) (((a) < (b)) ? (a) : (b))

#define MASTER 0 // Master MPI rank
#define NVARS_TO_COMM 4 // rho, e, rho_u, rho_v
#define NNEIGHBOURS 4

enum { NO_INVERT, INVERT_X, INVERT_Y };
enum { NO_PACK, PACK };
enum { EDGE=-1, NORTH=0, EAST=1, SOUTH=2, WEST=3 };

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
  double* mF_x;
  double* mF_y;

  // Slopes for monotonic advection
  double* slope_x0;
  double* slope_y0;
  double* slope_x1;
  double* slope_y1;

  // Interpolated velocity values
  double* Qxx;
  double* Qyy;
} State;

/// Mesh structure
typedef struct 
{
  double* edgedx;
  double* edgedy;
  double* celldx;
  double* celldy;

  int niters;
  double dt_h;
  double dt;

  int rank;
  int nranks;
  int neighbours[4];
  int x_off;
  int y_off;

  int local_nx;
  int local_ny;
  int global_nx;
  int global_ny;

  double* north_buffer_out;
  double* east_buffer_out;
  double* south_buffer_out;
  double* west_buffer_out;
  double* north_buffer_in;
  double* east_buffer_in;
  double* south_buffer_in;
  double* west_buffer_in;

} Mesh;

static inline void initialise_comms(
    int argc, char** argv, Mesh* mesh);

#ifdef MPI

// Decomposes the ranks, potentially load balancing and minimising the
// ratio of perimeter to area
static inline void decompose_ranks(Mesh* mesh);

#endif

static inline void equation_of_state(
    const int nx, const int ny, double* P, const double* rho, const double* e);

static inline void set_timestep(
    const int nx, const int ny, double* Qxx, double* Qyy, const double* rho, 
    const double* u, const double* v, const double* e, Mesh* mesh, const int first_step,
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy);

static inline void lagrangian_step(
    const int nx, const int ny, Mesh* mesh, const double dt, double* rho_u, 
    double* rho_v, double* u, double* v, const double* P, const double* rho,
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy);

static inline void artificial_viscosity(
    const int nx, const int ny, Mesh* mesh, const double dt, double* Qxx, 
    double* Qyy, double* u, double* v, double* rho_u, double* rho_v, const double* rho, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy);

// Calculates the work done due to forces within the element
static inline void shock_heating_and_work(
    const int nx, const int ny, Mesh* mesh, const double dt, double* e, 
    const double* P, const double* u, const double* v, const double* rho, 
    const double* Qxx, const double* Qyy, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy);

// Perform advection with monotonicity improvement
static inline void advect_mass(
    const int nx, const int ny, Mesh* mesh, const int tt, const double dt_h, 
    double* rho, double* rho_old, double* F_x, double* F_y, const double* u, 
    const double* v, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy);

// Calculate the flux in the x direction
static inline void x_mass_flux(
    const int nx, const int ny, Mesh* mesh, const double dt_h, double* rho, 
    const double* u, double* F_x, const double* celldx, const double* edgedx, 
    const double* celldy, const double* edgedy);

// Calculate the flux in the y direction
static inline void y_mass_flux(
    const int nx, const int ny, Mesh* mesh, const double dt_h, double* rho, 
    const double* v, double* F_y, const double* celldx, const double* edgedx, 
    const double* celldy, const double* edgedy);

// Advect momentum according to the velocity
static inline void advect_momentum(
    const int nx, const int ny, Mesh* mesh, const double dt_h, const double dt, 
    double* u, double* v, double* slope_x, double* slope_y, double* mF_x, 
    double* mF_y, double* rho_u, double* rho_v, const double* rho, 
    const double* F_x, const double* F_y, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy);

// Perform advection of internal energy
static inline void advect_energy(
    const int nx, const int ny, Mesh* mesh, const double dt_h, const double dt, 
    double* e, double* slope_x, double* slope_y, double* F_x, double* F_y, 
    const double* u, const double* v, const double* rho_old, const double* rho,
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy);

// Enforce reflective boundary conditions on the problem state
static inline void handle_boundary(
    const int nx, const int ny, Mesh* mesh, 
    double* arr, const int invert, const int pack);

static inline void write_to_visit_over_mpi(
    Mesh* mesh, const int rank, const int nranks, double* local_arr, 
    const char* name, const int tt, const double elapsed_sim_time);

// Initialise the state for the problem
static inline void initialise_state(
    State* state, Mesh* mesh);

// Initialise the mesh describing variables
static inline void initialise_mesh(
    Mesh* mesh);

// Write out data for visualisation in visit
static inline void write_to_visit(
    const int nx, const int ny, const int x_off, const int y_off, 
    const double* data, const char* name, const int step, const double time);

// Prints some conservation values
static inline void print_conservation(
    const int nx, const int ny, State* state, Mesh* mesh);

// Deallocate all of the state memory
static inline void finalise_state(State* state);

// Deallocate all of the mesh memory
static inline void finalise_mesh(Mesh* mesh);
