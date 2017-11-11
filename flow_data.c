#include "flow_data.h"
#include "../shared.h"

// Initialises the shared_data variables for two dimensional applications
void initialise_flow_data_2d(const int local_nx, const int local_ny,
                             FlowData* flow_data) {
  allocate_data(&flow_data->momentum_x, (local_nx + 1) * (local_ny + 1));
  allocate_data(&flow_data->momentum_y, (local_nx + 1) * (local_ny + 1));
  allocate_data(&flow_data->mass_flux_x, (local_nx + 1) * (local_ny + 1));
  allocate_data(&flow_data->mass_flux_y, (local_nx + 1) * (local_ny + 1));
  allocate_data(&flow_data->uF_x, (local_nx + 1) * (local_ny + 1));
  allocate_data(&flow_data->uF_y, (local_nx + 1) * (local_ny + 1));
  allocate_data(&flow_data->vF_x, (local_nx + 1) * (local_ny + 1));
  allocate_data(&flow_data->vF_y, (local_nx + 1) * (local_ny + 1));
}

void deallocate_flow_data(FlowData* flow_data) {
  deallocate_data(flow_data->momentum_x);
  deallocate_data(flow_data->momentum_y);
  deallocate_data(flow_data->mass_flux_x);
  deallocate_data(flow_data->mass_flux_y);
  deallocate_data(flow_data->uF_x);
  deallocate_data(flow_data->uF_y);
  deallocate_data(flow_data->vF_x);
  deallocate_data(flow_data->vF_y);
}
