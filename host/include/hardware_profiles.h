#pragma once

#include "hardware_cost_model.h"

namespace hardware_cost {

// Frozen architectural profile.  These are hardware/primitive service
// constants, not values selected from a workload latency table.  Callers may
// override an intrinsic's rate in ResourceWork when its memory route or dtype
// has a more specific architectural throughput.
HardwareProfile Ascend910B3VectorProfile();

}  // namespace hardware_cost
