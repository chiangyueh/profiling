#pragma once

#include <vector>

#include "hardware_cost_model.h"

namespace hardware_cost {

// Shared lowering vocabulary.  Kernel frontends describe only hardware
// instructions and dependencies with these helpers; the evaluator never
// receives an operator identity.
PathNode Transfer(Resource resource, MemorySpace source, MemorySpace destination,
                  double bytes, double instructions = 1.0,
                  double dependencyWaves = 1.0);
PathNode Operations(Resource resource, double operations,
                    double instructions = 0.0);
PathNode EventHandshake(double instructions = 2.0);
PathNode PipeBarrier(double barriers = 1.0);
PathNode Concurrent(std::vector<PathNode> paths);

}  // namespace hardware_cost
