#include "hardware_cost_model.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

namespace hardware_cost {
namespace {

constexpr std::size_t Index(Resource value)
{
    return static_cast<std::size_t>(value);
}

constexpr std::size_t Index(MemorySpace value)
{
    return static_cast<std::size_t>(value);
}

double Divide(double amount, double rate)
{
    if (amount <= 0.0) return 0.0;
    return rate > 0.0 ? amount / rate : std::numeric_limits<double>::infinity();
}

void Add(PathCost &target, const PathCost &value)
{
    target.valid = target.valid && value.valid;
    target.fillDrainCycles += value.fillDrainCycles;
    for (std::size_t i = 0; i < ResourceCount; ++i) {
        target.resourceCycles[i] += value.resourceCycles[i];
    }
    for (std::size_t i = 0; i < MemorySpaceCount; ++i) {
        target.readBytes[i] += value.readBytes[i];
        target.writeBytes[i] += value.writeBytes[i];
    }
}

double ResourceRoof(const PathCost &cost)
{
    return *std::max_element(cost.resourceCycles.begin(), cost.resourceCycles.end());
}

}  // namespace

PathNode PathNode::Work(ResourceWork value)
{
    PathNode node;
    node.kind = PathKind::WORK;
    node.work = std::move(value);
    return node;
}

PathNode PathNode::Sequence(std::vector<PathNode> values)
{
    PathNode node;
    node.kind = PathKind::SEQUENCE;
    node.children = std::move(values);
    return node;
}

PathNode PathNode::Parallel(std::vector<PathNode> values)
{
    PathNode node;
    node.kind = PathKind::PARALLEL;
    node.children = std::move(values);
    return node;
}

PathNode PathNode::Pipeline(double iterationCount, std::vector<PathNode> values)
{
    PathNode node;
    node.kind = PathKind::PIPELINE;
    node.children = std::move(values);
    node.iterations = iterationCount;
    return node;
}

HardwareCostModel::HardwareCostModel(HardwareProfile profile)
    : profile_(std::move(profile))
{
}

PathCost HardwareCostModel::EvaluatePath(const PathNode &path) const
{
    PathCost result;
    if (path.kind == PathKind::WORK) {
        const ResourceWork &work = path.work;
        const ResourceRate &fallback = profile_.rates[Index(work.resource)];
        const double byteRate = work.bytesPerCycle > 0.0 ? work.bytesPerCycle : fallback.bytesPerCycle;
        const double operationRate = work.operationsPerCycle > 0.0 ?
            work.operationsPerCycle : fallback.operationsPerCycle;
        const double issueCost = work.issueCycles > 0.0 ? work.issueCycles : fallback.issueCycles;
        const double latency = work.latencyCycles >= 0.0 ?
            work.latencyCycles : fallback.latencyCycles;
        const double occupancy = work.fixedCycles + Divide(work.bytes, byteRate) +
            Divide(work.operations, operationRate) + work.issues * issueCost;
        const double cycles = latency + occupancy;
        result.valid = std::isfinite(cycles) && cycles >= 0.0;
        result.elapsedCycles = result.valid ? cycles : std::numeric_limits<double>::infinity();
        // Latency can overlap with other outstanding requests and therefore
        // does not consume the shared issue resource for its full duration.
        result.resourceCycles[Index(work.resource)] = result.valid ?
            occupancy : std::numeric_limits<double>::infinity();
        const double sourceBytes = work.sourceBytes >= 0.0 ? work.sourceBytes : work.bytes;
        const double destinationBytes = work.destinationBytes >= 0.0 ? work.destinationBytes : work.bytes;
        if (work.source != MemorySpace::NONE) result.readBytes[Index(work.source)] += sourceBytes;
        if (work.destination != MemorySpace::NONE) result.writeBytes[Index(work.destination)] += destinationBytes;
        if (work.intermediate != MemorySpace::NONE) {
            result.readBytes[Index(work.intermediate)] += work.bytes;
            result.writeBytes[Index(work.intermediate)] += work.bytes;
        }
        return result;
    }

    if (path.children.empty()) return result;
    std::vector<PathCost> children;
    children.reserve(path.children.size());
    double serialCycles = 0.0;
    double longestChild = 0.0;
    for (const PathNode &child : path.children) {
        children.push_back(EvaluatePath(child));
        const PathCost &cost = children.back();
        Add(result, cost);
        serialCycles += cost.elapsedCycles;
        longestChild = std::max(longestChild, cost.elapsedCycles);
    }

    if (!result.valid || !std::isfinite(serialCycles)) {
        result.valid = false;
        result.elapsedCycles = std::numeric_limits<double>::infinity();
        return result;
    }
    if (path.kind == PathKind::SEQUENCE) {
        result.elapsedCycles = serialCycles;
        return result;
    }

    // Two paths cannot overlap beyond either their dependency duration or a
    // shared hardware resource's occupancy.  This makes the rule reusable for
    // arbitrary kernels without naming their operators.
    const double steadyState = std::max(longestChild, ResourceRoof(result));
    if (path.kind == PathKind::PARALLEL || path.iterations <= 1.0) {
        result.elapsedCycles = path.kind == PathKind::PARALLEL ? steadyState : serialCycles;
        return result;
    }

    const double fillDrain = std::max(0.0, serialCycles - steadyState) /
        std::max(1.0, path.iterations);
    result.fillDrainCycles += fillDrain;
    result.elapsedCycles = steadyState + fillDrain;
    return result;
}

KernelCost HardwareCostModel::Evaluate(const KernelProgram &program) const
{
    KernelCost result;
    if (program.corePaths.empty()) {
        result.valid = false;
        result.error = "kernel program has no active core path";
        result.totalCycles = std::numeric_limits<double>::infinity();
        return result;
    }
    for (std::size_t i = 0; i < MemorySpaceCount; ++i) {
        const uint64_t capacity = profile_.capacityBytes[i];
        if (capacity != 0 && program.peakMemoryBytes[i] > capacity) {
            result.valid = false;
            result.error = std::string("memory capacity exceeded: ") +
                MemorySpaceName(static_cast<MemorySpace>(i));
            result.totalCycles = std::numeric_limits<double>::infinity();
            return result;
        }
    }

    double sum = 0.0;
    std::array<double, ResourceCount> resourceOccupancy{};
    for (std::size_t core = 0; core < program.corePaths.size(); ++core) {
        const PathCost cost = EvaluatePath(program.corePaths[core]);
        if (!cost.valid) {
            result.valid = false;
            result.error = "one core path contains work without a hardware rate";
            result.totalCycles = std::numeric_limits<double>::infinity();
            return result;
        }
        sum += cost.elapsedCycles;
        for (std::size_t i = 0; i < MemorySpaceCount; ++i) {
            result.readBytes[i] += cost.readBytes[i];
            result.writeBytes[i] += cost.writeBytes[i];
        }
        for (std::size_t i = 0; i < ResourceCount; ++i) {
            resourceOccupancy[i] += cost.resourceCycles[i];
        }
        if (result.criticalCore < 0 || cost.elapsedCycles > result.criticalCoreCycles) {
            result.criticalCore = static_cast<int32_t>(core);
            result.criticalCoreCycles = cost.elapsedCycles;
            result.fillDrainCycles = cost.fillDrainCycles;
            result.criticalResourceCycles = cost.resourceCycles;
        }
    }

    const double activeCores = static_cast<double>(program.corePaths.size());
    result.averageCoreCycles = sum / activeCores;
    result.balanceCycles = std::max(0.0, result.criticalCoreCycles - result.averageCoreCycles);
    const int32_t available = std::max<int32_t>(1, program.availableCores);
    result.coreUtilization = std::min(1.0, activeCores / available);

    const std::size_t gm = Index(MemorySpace::GM);
    const std::size_t l2 = Index(MemorySpace::L2);
    const double gmTraffic = result.readBytes[gm] + result.writeBytes[gm];
    const double l2Traffic = result.readBytes[l2] + result.writeBytes[l2];
    result.hbmCycles = profile_.aggregateHbmBytesPerCycle > 0.0 ?
        gmTraffic / profile_.aggregateHbmBytesPerCycle : 0.0;
    result.l2Cycles = profile_.aggregateL2BytesPerCycle > 0.0 ?
        l2Traffic / profile_.aggregateL2BytesPerCycle : 0.0;
    for (std::size_t i = 0; i < ResourceCount; ++i) {
        const double units = profile_.parallelUnits[i];
        if (units > 0.0) {
            result.aggregateResourceCycles[i] = resourceOccupancy[i] / units;
            result.sharedResourceCycles = std::max(
                result.sharedResourceCycles, result.aggregateResourceCycles[i]);
        }
    }
    result.launchCycles = program.launchCycles > 0.0 ?
        program.launchCycles : profile_.kernelLaunchCycles +
            profile_.activeCoreLaunchCycles * activeCores;
    result.synchronizationCycles = program.synchronizationCycles;
    result.totalCycles = std::max({result.criticalCoreCycles, result.hbmCycles, result.l2Cycles,
                                   result.sharedResourceCycles}) +
        result.launchCycles + result.synchronizationCycles;
    return result;
}

const char *ResourceName(Resource resource)
{
    static constexpr std::array<const char *, ResourceCount> names = {
        "MTE2", "MTE1", "MTE3", "CUBE", "VECTOR", "SCALAR", "FIXPIPE", "SYNC"
    };
    const std::size_t index = Index(resource);
    return index < names.size() ? names[index] : "UNKNOWN";
}

const char *MemorySpaceName(MemorySpace space)
{
    static constexpr std::array<const char *, MemorySpaceCount> names = {
        "NONE", "GM", "L2", "L1", "L0A", "L0B", "L0C", "UB"
    };
    const std::size_t index = Index(space);
    return index < names.size() ? names[index] : "UNKNOWN";
}

}  // namespace hardware_cost
