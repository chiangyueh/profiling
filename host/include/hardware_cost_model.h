#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace hardware_cost {

// The evaluator deliberately has no operator enum or operator name.  Kernel
// frontends lower their shape/tiling/control flow into these hardware
// resources; the same evaluator then scores every resulting path.
enum class Resource : std::size_t {
    MTE2,
    MTE1,
    MTE3,
    CUBE,
    VECTOR,
    SCALAR,
    FIXPIPE,
    SYNC,
    COUNT,
};

enum class MemorySpace : std::size_t {
    NONE,
    GM,
    L2,
    L1,
    L0A,
    L0B,
    L0C,
    UB,
    COUNT,
};

enum class PathKind {
    WORK,
    SEQUENCE,
    PARALLEL,
    PIPELINE,
};

constexpr std::size_t ResourceCount = static_cast<std::size_t>(Resource::COUNT);
constexpr std::size_t MemorySpaceCount = static_cast<std::size_t>(MemorySpace::COUNT);

struct ResourceRate {
    double bytesPerCycle = 0.0;
    double operationsPerCycle = 0.0;
    double issueCycles = 0.0;
    // Completion latency that does not continuously occupy the issue pipe.
    // Keeping this separate is essential for queued MTE requests.
    double latencyCycles = 0.0;
};

struct HardwareProfile {
    std::array<ResourceRate, ResourceCount> rates{};
    std::array<uint64_t, MemorySpaceCount> capacityBytes{};

    // Number of independently issuing instances available chip-wide.  When
    // non-zero, total occupancy of that resource across all cores creates an
    // additional shared-resource roof (for example 40 Vector pipes or a
    // smaller number of concurrent GM issue streams).
    std::array<double, ResourceCount> parallelUnits{};

    // Aggregate chip roofs.  A zero value disables the corresponding roof.
    double aggregateHbmBytesPerCycle = 0.0;
    double aggregateL2BytesPerCycle = 0.0;
    double kernelLaunchCycles = 0.0;
    double activeCoreLaunchCycles = 0.0;
};

struct ResourceWork {
    Resource resource = Resource::SCALAR;
    MemorySpace source = MemorySpace::NONE;
    MemorySpace destination = MemorySpace::NONE;
    // Optional hierarchy hop traversed by this instruction, such as L2 on a
    // GM-to-UB transfer. It contributes traffic/roof pressure but not a
    // second instruction service time.
    MemorySpace intermediate = MemorySpace::NONE;
    double bytes = 0.0;
    // A hardware instruction may move a different number of bytes on its
    // internal port than on its source/destination link (FixPipe is the
    // common example).  Negative means use `bytes` for accounting.
    double sourceBytes = -1.0;
    double destinationBytes = -1.0;
    double operations = 0.0;
    double issues = 0.0;
    double fixedCycles = 0.0;

    // Intrinsics may override a platform-wide rate (for example Cube dtype
    // throughput).  Zero means use HardwareProfile::rates.
    double bytesPerCycle = 0.0;
    double operationsPerCycle = 0.0;
    double issueCycles = 0.0;
    // Negative means use ResourceRate::latencyCycles.
    double latencyCycles = -1.0;
    // Number of serialized dependency waves that pay completion latency.
    // Multiple requests inside one wave may remain outstanding together;
    // separate producer/consumer iterations cannot.  This is a hardware
    // property of the lowered path, not an operator-specific coefficient.
    double latencyWaves = 1.0;
};

struct PathNode {
    PathKind kind = PathKind::WORK;
    ResourceWork work{};
    std::vector<PathNode> children;

    // PIPELINE uses this count for fill/drain amortization.  SEQUENCE and
    // PARALLEL ignore it.
    double iterations = 1.0;

    static PathNode Work(ResourceWork value);
    static PathNode Sequence(std::vector<PathNode> values);
    static PathNode Parallel(std::vector<PathNode> values);
    static PathNode Pipeline(double iterationCount, std::vector<PathNode> values);
};

struct KernelProgram {
    std::vector<PathNode> corePaths;
    int32_t availableCores = 1;
    std::array<uint64_t, MemorySpaceCount> peakMemoryBytes{};
    double launchCycles = 0.0;
    double synchronizationCycles = 0.0;
};

struct PathCost {
    bool valid = true;
    double elapsedCycles = 0.0;
    double fillDrainCycles = 0.0;
    std::array<double, ResourceCount> resourceCycles{};
    std::array<double, MemorySpaceCount> readBytes{};
    std::array<double, MemorySpaceCount> writeBytes{};
};

struct KernelCost {
    bool valid = true;
    std::string error;
    double totalCycles = 0.0;
    double criticalCoreCycles = 0.0;
    double averageCoreCycles = 0.0;
    double balanceCycles = 0.0;
    double fillDrainCycles = 0.0;
    double hbmCycles = 0.0;
    double l2Cycles = 0.0;
    double sharedResourceCycles = 0.0;
    double launchCycles = 0.0;
    double synchronizationCycles = 0.0;
    double coreUtilization = 0.0;
    int32_t criticalCore = -1;
    std::array<double, ResourceCount> criticalResourceCycles{};
    std::array<double, ResourceCount> aggregateResourceCycles{};
    std::array<double, MemorySpaceCount> readBytes{};
    std::array<double, MemorySpaceCount> writeBytes{};
};

class HardwareCostModel {
public:
    explicit HardwareCostModel(HardwareProfile profile);

    PathCost EvaluatePath(const PathNode &path) const;
    KernelCost Evaluate(const KernelProgram &program) const;
    const HardwareProfile &Profile() const { return profile_; }

private:
    HardwareProfile profile_;
};

const char *ResourceName(Resource resource);
const char *MemorySpaceName(MemorySpace space);

}  // namespace hardware_cost
