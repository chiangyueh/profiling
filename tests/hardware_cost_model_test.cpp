#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

#include "hardware_cost_model.h"

namespace {

hardware_cost::ResourceWork Bytes(hardware_cost::Resource resource,
                                  hardware_cost::MemorySpace source,
                                  hardware_cost::MemorySpace destination,
                                  double bytes)
{
    hardware_cost::ResourceWork work;
    work.resource = resource;
    work.source = source;
    work.destination = destination;
    work.bytes = bytes;
    return work;
}

bool Near(double a, double b)
{
    return std::abs(a - b) <= 1e-12 * std::max({1.0, std::abs(a), std::abs(b)});
}

}  // namespace

int main()
{
    using namespace hardware_cost;
    HardwareProfile profile;
    profile.rates[static_cast<std::size_t>(Resource::MTE2)].bytesPerCycle = 16.0;
    profile.rates[static_cast<std::size_t>(Resource::MTE3)].bytesPerCycle = 16.0;
    profile.rates[static_cast<std::size_t>(Resource::VECTOR)].operationsPerCycle = 8.0;
    profile.aggregateHbmBytesPerCycle = 32.0;
    profile.kernelLaunchCycles = 10.0;
    profile.capacityBytes[static_cast<std::size_t>(MemorySpace::UB)] = 1024;
    HardwareCostModel model(profile);

    ResourceWork vector;
    vector.resource = Resource::VECTOR;
    vector.operations = 128.0;
    const PathNode load = PathNode::Work(Bytes(Resource::MTE2, MemorySpace::GM, MemorySpace::UB, 256.0));
    const PathNode compute = PathNode::Work(vector);
    const PathNode store = PathNode::Work(Bytes(Resource::MTE3, MemorySpace::UB, MemorySpace::GM, 256.0));

    const PathCost serial = model.EvaluatePath(PathNode::Sequence({load, compute, store}));
    assert(Near(serial.elapsedCycles, 48.0));

    ResourceWork hierarchical = Bytes(Resource::MTE2, MemorySpace::GM, MemorySpace::UB, 256.0);
    hierarchical.intermediate = MemorySpace::L2;
    const PathCost hierarchy = model.EvaluatePath(PathNode::Work(hierarchical));
    assert(Near(hierarchy.readBytes[static_cast<std::size_t>(MemorySpace::L2)], 256.0));
    assert(Near(hierarchy.writeBytes[static_cast<std::size_t>(MemorySpace::L2)], 256.0));

    HardwareProfile queuedProfile = profile;
    queuedProfile.rates[static_cast<std::size_t>(Resource::MTE2)].latencyCycles = 100.0;
    const PathCost queued = HardwareCostModel(queuedProfile).EvaluatePath(load);
    assert(Near(queued.elapsedCycles, 116.0));
    assert(Near(queued.resourceCycles[static_cast<std::size_t>(Resource::MTE2)], 16.0));

    ResourceWork dependentLoads = Bytes(Resource::MTE2, MemorySpace::GM,
                                        MemorySpace::UB, 256.0);
    dependentLoads.latencyWaves = 3.0;
    const PathCost dependent = HardwareCostModel(queuedProfile).EvaluatePath(
        PathNode::Work(dependentLoads));
    assert(Near(dependent.elapsedCycles, 316.0));
    // Completion latency is not shared-pipe occupancy.
    assert(Near(dependent.resourceCycles[
        static_cast<std::size_t>(Resource::MTE2)], 16.0));

    const PathCost pipelined = model.EvaluatePath(PathNode::Pipeline(8.0, {load, compute, store}));
    assert(Near(pipelined.elapsedCycles, 20.0));
    assert(pipelined.fillDrainCycles > 0.0);

    // Shared-resource children remain serialized even inside a parallel node.
    const PathCost shared = model.EvaluatePath(PathNode::Parallel({load, load}));
    assert(Near(shared.elapsedCycles, 32.0));

    KernelProgram program;
    program.availableCores = 2;
    program.corePaths = {PathNode::Sequence({load, compute, store}),
                         PathNode::Sequence({load, store})};
    const KernelCost kernel = model.Evaluate(program);
    assert(kernel.valid);
    assert(kernel.criticalCore == 0);
    assert(Near(kernel.criticalCoreCycles, 48.0));
    assert(Near(kernel.hbmCycles, 32.0));
    assert(Near(kernel.totalCycles, 58.0));
    assert(kernel.balanceCycles > 0.0);

    profile.parallelUnits[static_cast<std::size_t>(Resource::MTE2)] = 1.0;
    HardwareCostModel contendedModel(profile);
    const KernelCost contended = contendedModel.Evaluate(program);
    assert(contended.sharedResourceCycles > 0.0);
    assert(contended.totalCycles >= kernel.totalCycles);

    program.peakMemoryBytes[static_cast<std::size_t>(MemorySpace::UB)] = 2048;
    const KernelCost invalid = model.Evaluate(program);
    assert(!invalid.valid);
    assert(!std::isfinite(invalid.totalCycles));

    std::cout << "hardware_cost_model_test passed\n";
    return 0;
}
