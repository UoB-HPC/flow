#define MPI
#ifdef MPI
#include <mpi.h>
#endif

// Controllable parameters for the application
#define GAM 1.4
#define PAD 2
#define MAX_DT 0.004
#define C_Q 1.0 // Suggested between 1.0 and 2.0
#define C_T 0.5
#define VISIT_STEP 10
#define SIM_END 10.0
#define LOAD_BALANCE 0

#define max(a, b) (((a) > (b)) ? (a) : (b))
#define min(a, b) (((a) < (b)) ? (a) : (b))

#define MASTER 0 // Master MPI rank
#define NUM_VARS_TO_COMM 4 // rho, e, rho_u, rho_v

enum { NO_INVERT, INVERT_X, INVERT_Y };
enum { EDGE=-1, NORTH=0, EAST=1, SOUTH=2, WEST=3 };
enum { RHO_OFF, E_OFF, RHO_U_OFF, RHO_V_OFF };

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

  double dt_h;
  double dt;
} Mesh;

typedef struct
{
  int rank;
  int nranks;
  int neighbours[4];

  double* north_buffer_out;
  double* east_buffer_out;
  double* south_buffer_out;
  double* west_buffer_out;
  double* north_buffer_in;
  double* east_buffer_in;
  double* south_buffer_in;
  double* west_buffer_in;

} Comms;

static inline void initialise_comms(
    const int cells_x, const int cells_y, int argc, char** argv, 
    int *nx, int *ny, Comms* comms);

// Decomposes the ranks, potentially load balancing and minimising the
// ratio of perimeter to area
static inline void decompose_ranks(
    const int rank, const int nranks, const int mesh_x, const int mesh_y,
    int* nx, int* ny, int* neighbours);

static inline void communicate_halos(
    const int nx, const int ny, Comms* comms, double* rho, double* rho_u, 
    double* rho_v, double* e);

// Calculate the pressure from gamma law equation of state
static inline void equation_of_state(
    const int nx, const int ny, double* P, const double* rho, const double* e);

// Calculates the timestep from the current state
static inline void set_timestep(
    const int nx, const int ny, double* Qzz, double* Qrr, const double* rho, 
    const double* u, const double* v, const double* e, Mesh* mesh, const int first_step,
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy);

// Calculate change in momentum caused by pressure gradients, and then extract
// the velocities using edge centered density approximations
static inline void lagrangian_step(
    const int nx, const int ny, const double dt, double* rho_u, double* rho_v, 
    double* u, double* v, const double* P, const double* rho,
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy);

static inline void artificial_viscosity(
    const int nx, const int ny, const double dt, double* Qzz, double* Qrr, 
    double* u, double* v, double* rho_u, double* rho_v, const double* rho, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy);

// Calculates the work done due to forces within the element
static inline void shock_heating_and_work(
    const int nx, const int ny, const double dt, double* e, const double* P, 
    const double* u, const double* v, const double* rho, const double* Qzz, const double* Qrr,
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy);

// Perform advection with monotonicity improvement
static inline void advect_mass(
    const int nx, const int ny, const double dt_h, double* rho, double* rho_old, 
    double* slope_x, double* slope_y, double* F_x, double* F_y, const double* u, const double* v, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy);

// Advect momentum according to the velocity
static inline void advect_momentum(
    const int nx, const int ny, const double dt_h, const double dt, double* u, 
    double* v, double* slope_x, double* slope_y, double* gF_x, double* gF_y, 
    double* rho_u, double* rho_v, const double* rho, const double* F_x, const double* F_y,
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy);

// Perform advection of internal energy
static inline void advect_energy(
    const int nx, const int ny, const double dt_h, const double dt, double* e, 
    double* slope_x, double* slope_y, double* F_x, double* F_y, const double* u, 
    const double* v, const double* rho_old, const double* rho, 
    const double* edgedx, const double* edgedy, const double* celldx, const double* celldy);

// Enforce reflective boundary conditions on the problem state
static inline void reflective_boundary(
    const int nx, const int ny, const int depth, double* arr, const int invert);

// Initialise the state for the problem
static inline void initialise_state(
    const int nx, const int ny, State* state, Mesh* mesh);

// Initialise the mesh describing variables
static inline void initialise_mesh(
    const int nx, const int ny, Mesh* mesh);

// Deallocate all of the state memory
static inline void finalise_state(
    State* state);

// Deallocate all of the mesh memory
static inline void finalise_mesh(
    Mesh* mesh);

// Write out data for visualisation in visit
static inline void write_to_visit(
    const int nx, const int ny, const double* data, 
    const char* name, const int step, const double time);

static inline void print_conservation(
    const int nx, const int ny, State* state);

