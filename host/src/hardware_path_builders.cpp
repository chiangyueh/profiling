#include "hardware_path_builders.h"

#include <utility>

namespace hardware_cost {

PathNode Transfer(Resource resource, MemorySpace source, MemorySpace destination,
                  double bytes, double instructions, double dependencyWaves)
{
    ResourceWork work;
    work.resource = resource;
    work.source = source;
    work.destination = destination;
    if (source == MemorySpace::GM || destination == MemorySpace::GM) {
        work.intermediate = MemorySpace::L2;
    }
    work.bytes = bytes;
    work.issues = instructions;
    work.latencyWaves = dependencyWaves;
    return PathNode::Work(work);
}

PathNode Operations(Resource resource, double operations, double instructions)
{
    ResourceWork work;
    work.resource = resource;
    work.operations = operations;
    work.issues = instructions;
    work.latencyWaves = 0.0;
    return PathNode::Work(work);
}

PathNode EventHandshake(double instructions)
{
    return Operations(Resource::SYNC, 0.0, instructions);
}

PathNode PipeBarrier(double barriers)
{
    return Operations(Resource::SYNC, barriers, 0.0);
}

PathNode Concurrent(std::vector<PathNode> paths)
{
    return PathNode::Parallel(std::move(paths));
}

}  // namespace hardware_cost
