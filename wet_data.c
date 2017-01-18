#include "wet_data.h"
#include "../shared.h"

// Initialises the shared_data variables for two dimensional applications
void initialise_wet_data_2d(
    const int local_nx, const int local_ny, WetData* wet_data)
{
  allocate_data(&wet_data->rho_u, (local_nx+1)*(local_ny+1));
  allocate_data(&wet_data->rho_v, (local_nx+1)*(local_ny+1));
  allocate_data(&wet_data->F_x, (local_nx+1)*(local_ny+1));
  allocate_data(&wet_data->F_y, (local_nx+1)*(local_ny+1));
  allocate_data(&wet_data->uF_x, (local_nx+1)*(local_ny+1));
  allocate_data(&wet_data->uF_y, (local_nx+1)*(local_ny+1));
  allocate_data(&wet_data->vF_x, (local_nx+1)*(local_ny+1));
  allocate_data(&wet_data->vF_y, (local_nx+1)*(local_ny+1));
}

void deallocate_wet_data(
    WetData* wet_data)
{
  deallocate_data(wet_data->rho_u);
  deallocate_data(wet_data->rho_v);
  deallocate_data(wet_data->F_x);
  deallocate_data(wet_data->F_y);
  deallocate_data(wet_data->uF_x);
  deallocate_data(wet_data->uF_y);
  deallocate_data(wet_data->vF_x);
  deallocate_data(wet_data->vF_y);
}

