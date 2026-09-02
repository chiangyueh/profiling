#include "proxy_model.h"

#include "hardware_cost_model.h"
#include "hardware_profiles.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace matmul_search {
namespace {

template <typename T>
T CeilDiv(T x, T y)
{
    return y <= 0 ? 0 : (x + y - 1) / y;
}

int64_t AlignUp(int64_t x, int64_t alignment)
{
    return alignment <= 0 ? x : CeilDiv(x, alignment) * alignment;
}

double SafeRatio(double numerator, double denominator)
{
    return denominator <= 0.0 ? 0.0 : numerator / denominator;
}

// Sum the bytes transferred for an axis split into tile-sized contiguous
// segments. Each DataCopy segment is conservatively rounded to a 32-byte
// block, matching the GM data-move granularity used by the Matmul path.
double SumAlignedSegments(int64_t length, int64_t tile, int64_t elementBytes)
{
    if (length <= 0 || tile <= 0 || elementBytes <= 0) return 0.0;
    const int64_t full = length / tile;
    const int64_t tail = length % tile;
    double bytes = static_cast<double>(full) * AlignUp(tile * elementBytes, 32);
    if (tail != 0) bytes += static_cast<double>(AlignUp(tail * elementBytes, 32));
    return bytes;
}

double L1CacheFraction(int32_t depth, int64_t majorTiles, int64_t kTiles)
{
    const int64_t required = std::max<int64_t>(1, majorTiles) * std::max<int64_t>(1, kTiles);
    if (depth <= 0) return 0.0;

    // This mirrors CubeInBuffer::Init for CFG_NORM. Two entries are reserved
    // for streaming ping-pong when K cannot be fully cached.
    int64_t cacheEntries = 0;
    if (depth > 2) {
        cacheEntries = depth < required ? depth - 2 : depth;
    } else if (depth >= required && !(kTiles == 1 && depth == 2)) {
        cacheEntries = depth;
    }
    return std::clamp(SafeRatio(static_cast<double>(cacheEntries),
                                static_cast<double>(required)), 0.0, 1.0);
}

bool HasL1DoubleBuffer(int32_t depth, int32_t majorStep, int32_t kStep)
{
    const int64_t oneBuffer = static_cast<int64_t>(majorStep) * kStep;
    return oneBuffer > 0 && depth / oneBuffer >= 2;
}

double BlendReuse(double fullCacheRepeat, double noCacheRepeat, double cacheFraction)
{
    return fullCacheRepeat + (noCacheRepeat - fullCacheRepeat) * (1.0 - cacheFraction);
}

struct TilePlan {
    hardware_cost::PathNode path;
    double aGmBytes = 0.0;
    double bGmBytes = 0.0;
    double cGmBytes = 0.0;
    double logicalInputBytes = 0.0;
    double mte1Bytes = 0.0;
    double actualMac = 0.0;
    double paddedMac = 0.0;
    double mmads = 0.0;
    double outputTiles = 0.0;
};

struct CorePlan {
    std::vector<hardware_cost::PathNode> paths;
    double aGmBytes = 0.0;
    double bGmBytes = 0.0;
    double cGmBytes = 0.0;
    double logicalInputBytes = 0.0;
    double mte1Bytes = 0.0;
    double actualMac = 0.0;
    double paddedMac = 0.0;
    double mmads = 0.0;
    double outputTiles = 0.0;

    void Add(TilePlan tile)
    {
        paths.push_back(std::move(tile.path));
        aGmBytes += tile.aGmBytes;
        bGmBytes += tile.bGmBytes;
        cGmBytes += tile.cGmBytes;
        logicalInputBytes += tile.logicalInputBytes;
        mte1Bytes += tile.mte1Bytes;
        actualMac += tile.actualMac;
        paddedMac += tile.paddedMac;
        mmads += tile.mmads;
        outputTiles += tile.outputTiles;
    }
};

hardware_cost::ResourceWork Transfer(
    hardware_cost::Resource resource, hardware_cost::MemorySpace source,
    hardware_cost::MemorySpace destination, double bytes, double issues)
{
    hardware_cost::ResourceWork work;
    work.resource = resource;
    work.source = source;
    work.destination = destination;
    if (source == hardware_cost::MemorySpace::GM ||
        destination == hardware_cost::MemorySpace::GM) {
        work.intermediate = hardware_cost::MemorySpace::L2;
    }
    work.bytes = bytes;
    work.issues = issues;
    return work;
}

TilePlan LowerTileToHardwarePath(const Workload &w, const TCubeTiling &t,
                                 const MatmulPathParameters &parameters, int64_t m,
                                 int64_t n, int64_t k, bool atomicOutput)
{
    using hardware_cost::MemorySpace;
    using hardware_cost::PathNode;
    using hardware_cost::Resource;
    using hardware_cost::ResourceWork;

    TilePlan out;
    const int64_t inputBytes = DTypeBytes(w.dtype);
    const int64_t outputBytes = OutputBytes(w.dtype);
    const int64_t accumulatorBytes = AccumulatorBytes(w.dtype);
    const int64_t k0 = (32 / DTypeBits(w.dtype)) * 8;
    const double cubeMacPerCycle = static_cast<double>((256 / DTypeBits(w.dtype)) * 256);

    const int64_t mTiles = std::max<int64_t>(1, CeilDiv<int64_t>(m, t.baseM));
    const int64_t nTiles = std::max<int64_t>(1, CeilDiv<int64_t>(n, t.baseN));
    const int64_t kTiles = std::max<int64_t>(1, CeilDiv<int64_t>(k, t.baseK));
    const int64_t mOuter = std::max<int64_t>(1, CeilDiv<int64_t>(mTiles, t.stepM));
    const int64_t nOuter = std::max<int64_t>(1, CeilDiv<int64_t>(nTiles, t.stepN));

    const double aOnePass = w.transA ?
        static_cast<double>(k) * SumAlignedSegments(m, t.baseM, inputBytes) :
        static_cast<double>(m) * SumAlignedSegments(k, t.baseK, inputBytes);
    const double bOnePass = w.transB ?
        static_cast<double>(n) * SumAlignedSegments(k, t.baseK, inputBytes) :
        static_cast<double>(k) * SumAlignedSegments(n, t.baseN, inputBytes);

    double aFullCacheRepeat = 1.0;
    double aNoCacheRepeat = 1.0;
    double bFullCacheRepeat = 1.0;
    double bNoCacheRepeat = 1.0;
    double aCacheFraction = 0.0;
    double bCacheFraction = 0.0;

    if (t.iterateOrder == 0) {
        // CFG_NORM ORDER_M: N is the inner output loop. A is retained only
        // within one stepN group; B is retained while M is traversed.
        aFullCacheRepeat = static_cast<double>(nOuter);
        aNoCacheRepeat = static_cast<double>(nTiles);
        bFullCacheRepeat = 1.0;
        bNoCacheRepeat = static_cast<double>(mTiles);
        aCacheFraction = L1CacheFraction(t.depthA1, 1, kTiles);
        bCacheFraction = L1CacheFraction(t.depthB1, std::min<int64_t>(t.stepN, nTiles), kTiles);
    } else {
        // CFG_NORM ORDER_N: M is the inner output loop. B is retained only
        // within one stepM group; A is retained while N is traversed.
        aFullCacheRepeat = 1.0;
        aNoCacheRepeat = static_cast<double>(nTiles);
        bFullCacheRepeat = static_cast<double>(mOuter);
        bNoCacheRepeat = static_cast<double>(mTiles);
        aCacheFraction = L1CacheFraction(t.depthA1, std::min<int64_t>(t.stepM, mTiles), kTiles);
        bCacheFraction = L1CacheFraction(t.depthB1, 1, kTiles);
    }

    const double aRepeat = BlendReuse(aFullCacheRepeat, aNoCacheRepeat, aCacheFraction);
    const double bRepeat = BlendReuse(bFullCacheRepeat, bNoCacheRepeat, bCacheFraction);
    out.aGmBytes = aOnePass * aRepeat;
    out.bGmBytes = bOnePass * bRepeat;
    out.logicalInputBytes = aOnePass * aNoCacheRepeat + bOnePass * bNoCacheRepeat;

    const double cOnePass = static_cast<double>(m) * SumAlignedSegments(n, t.baseN, outputBytes);
    out.cGmBytes = cOnePass * (atomicOutput ? 2.0 : 1.0);

    const int64_t alignedM = AlignUp(m, 16);
    const int64_t alignedN = AlignUp(n, 16);
    const int64_t alignedK = AlignUp(k, k0);
    out.actualMac = static_cast<double>(m) * n * k;
    out.paddedMac = static_cast<double>(alignedM) * alignedN * alignedK;
    out.mmads = static_cast<double>(mTiles) * nTiles * kTiles;
    out.outputTiles = static_cast<double>(mTiles) * nTiles;

    const double aMte1Bytes = static_cast<double>(alignedM) * alignedK * inputBytes * nTiles;
    const double bMte1Bytes = static_cast<double>(alignedN) * alignedK * inputBytes * mTiles;
    out.mte1Bytes = aMte1Bytes + bMte1Bytes;

    const double aLoadBlocks = static_cast<double>(mTiles) * kTiles * aRepeat;
    const double bLoadBlocks = static_cast<double>(nTiles) * kTiles * bRepeat;
    const double fixpipeBytes = static_cast<double>(alignedM) * alignedN *
        (accumulatorBytes + outputBytes);

    ResourceWork mte2AWork = Transfer(
        Resource::MTE2, MemorySpace::GM, MemorySpace::L1, out.aGmBytes, aLoadBlocks);
    ResourceWork mte2BWork = Transfer(
        Resource::MTE2, MemorySpace::GM, MemorySpace::L1, out.bGmBytes, bLoadBlocks);
    const PathNode mte2A = PathNode::Work(mte2AWork);
    const PathNode mte2B = PathNode::Work(mte2BWork);
    ResourceWork mte1AWork = Transfer(
        Resource::MTE1, MemorySpace::L1, MemorySpace::L0A, aMte1Bytes, out.mmads);
    ResourceWork mte1BWork = Transfer(
        Resource::MTE1, MemorySpace::L1, MemorySpace::L0B, bMte1Bytes, out.mmads);
    // Both L1-to-matrix-buffer routes use the measured MTE1 service rate.
    // The available matched CA/CB CCE observations do not establish a
    // route-specific throughput difference.
    mte1AWork.bytesPerCycle = 256.0;
    mte1BWork.bytesPerCycle = 256.0;
    const PathNode mte1A = PathNode::Work(mte1AWork);
    const PathNode mte1B = PathNode::Work(mte1BWork);

    ResourceWork cube;
    cube.resource = Resource::CUBE;
    cube.operations = out.paddedMac;
    cube.operationsPerCycle = cubeMacPerCycle;
    cube.issues = out.mmads;
    const PathNode cubePath = PathNode::Work(cube);

    ResourceWork fixpipe = Transfer(Resource::FIXPIPE, MemorySpace::L0C,
                                    MemorySpace::GM, fixpipeBytes, out.outputTiles);
    // The internal L0C/FixPipe traffic includes accumulator conversion; only
    // the final output bytes reach GM.
    fixpipe.destinationBytes = out.cGmBytes;
    const PathNode fixpipePath = PathNode::Work(fixpipe);

    ResourceWork scalar;
    scalar.resource = Resource::SCALAR;
    scalar.fixedCycles = parameters.scalarCoreSetupCycles;
    scalar.operations = out.mmads * parameters.scalarPerMmadCycles +
        out.outputTiles * parameters.scalarPerOutputTileCycles;
    scalar.operationsPerCycle = 1.0;
    const PathNode scalarPath = PathNode::Work(scalar);

    const bool l0InputDb = t.dbL0A == 2 && t.dbL0B == 2;
    PathNode downstream = l0InputDb ?
        PathNode::Pipeline(out.mmads,
            {PathNode::Sequence({mte1A, mte1B}), cubePath}) :
        PathNode::Sequence({mte1A, mte1B, cubePath});

    if (t.dbL0C == 2) {
        downstream = PathNode::Pipeline(out.outputTiles,
                                        {std::move(downstream), fixpipePath});
    } else {
        downstream = PathNode::Sequence({std::move(downstream), fixpipePath});
    }

    const bool aL1Db = HasL1DoubleBuffer(t.depthA1, t.stepM, t.stepKa);
    const bool bL1Db = HasL1DoubleBuffer(t.depthB1, t.stepN, t.stepKb);
    if (aL1Db && bL1Db) {
        downstream = PathNode::Pipeline(std::max(1.0, aLoadBlocks + bLoadBlocks),
            {PathNode::Sequence({mte2A, mte2B}), std::move(downstream)});
    } else if (aL1Db || bL1Db) {
        const PathNode &overlappedMte2 = aL1Db ? mte2A : mte2B;
        const PathNode &serializedMte2 = aL1Db ? mte2B : mte2A;
        downstream = PathNode::Sequence({serializedMte2,
            PathNode::Pipeline(std::max(1.0, aL1Db ? aLoadBlocks : bLoadBlocks),
                               {overlappedMte2, std::move(downstream)})});
    } else {
        downstream = PathNode::Sequence({mte2A, mte2B, std::move(downstream)});
    }

    out.path = PathNode::Parallel({std::move(downstream), scalarPath});
    return out;
}

hardware_cost::HardwareProfile MakeHardwareProfile(const PlatformCaps &caps,
                                                    const MatmulPathParameters &parameters,
                                                    int32_t usedCores)
{
    using hardware_cost::MemorySpace;
    using hardware_cost::Resource;
    hardware_cost::HardwareProfile profile = hardware_cost::Ascend910B3VectorProfile();
    const auto resource = [](Resource value) { return static_cast<std::size_t>(value); };
    const auto memory = [](MemorySpace value) { return static_cast<std::size_t>(value); };
    profile.rates[resource(Resource::CUBE)].issueCycles = parameters.cubeIssueCycles;
    profile.rates[resource(Resource::SCALAR)].operationsPerCycle = 1.0;

    profile.capacityBytes[memory(MemorySpace::L0A)] = caps.l0aBytes;
    profile.capacityBytes[memory(MemorySpace::L0B)] = caps.l0bBytes;
    profile.capacityBytes[memory(MemorySpace::L0C)] = caps.l0cBytes;
    profile.capacityBytes[memory(MemorySpace::L1)] = caps.l1Bytes;
    profile.capacityBytes[memory(MemorySpace::UB)] = caps.ubBytes;
    profile.capacityBytes[memory(MemorySpace::L2)] = caps.l2Bytes;
    if (caps.hbmBytesPerCycle > 0) {
        profile.aggregateHbmBytesPerCycle =
            static_cast<double>(caps.hbmBytesPerCycle) * std::max(1, usedCores);
    }
    if (caps.l2BytesPerCycle > 0) {
        profile.aggregateL2BytesPerCycle =
            static_cast<double>(caps.l2BytesPerCycle) * std::max(1, usedCores);
    }
    profile.kernelLaunchCycles = parameters.kernelFixedCycles;
    return profile;
}

}  // namespace

ProxyModel::ProxyModel(PlatformCaps caps, MatmulPathParameters parameters)
    : caps_(caps), parameters_(parameters)
{
}

ProxyBreakdown ProxyModel::Score(
    const Workload &w, const Candidate &, const TCubeTiling &t) const
{
    ProxyBreakdown out;
    if (t.usedCoreNum <= 0 || t.singleCoreM <= 0 || t.singleCoreN <= 0 ||
        t.singleCoreK <= 0 || t.baseM <= 0 || t.baseN <= 0 || t.baseK <= 0 ||
        t.stepM <= 0 || t.stepN <= 0 || t.stepKa <= 0 || t.stepKb <= 0) {
        out.total = std::numeric_limits<double>::infinity();
        return out;
    }

    const int64_t mParts = std::max<int64_t>(1, CeilDiv<int64_t>(w.m, t.singleCoreM));
    const int64_t nParts = std::max<int64_t>(1, CeilDiv<int64_t>(w.n, t.singleCoreN));
    const int64_t kParts = std::max<int64_t>(1, CeilDiv<int64_t>(w.k, t.singleCoreK));
    const bool splitK = kParts > 1;
    const int64_t totalTiles = mParts * nParts * kParts;
    const int32_t usedCores = std::max(1, t.usedCoreNum);
    std::vector<CorePlan> cores(static_cast<size_t>(usedCores));

    if (splitK) {
        for (int32_t core = 0; core < usedCores && core < totalTiles; ++core) {
            const int64_t mIndex = core % mParts;
            const int64_t nIndex = (core / mParts) % nParts;
            const int64_t kIndex = core / (mParts * nParts);
            const int64_t mUse = std::min<int64_t>(t.singleCoreM, w.m - mIndex * t.singleCoreM);
            const int64_t nUse = std::min<int64_t>(t.singleCoreN, w.n - nIndex * t.singleCoreN);
            const int64_t kUse = std::min<int64_t>(t.singleCoreK, w.k - kIndex * t.singleCoreK);
            cores[core].Add(LowerTileToHardwarePath(w, t, parameters_, mUse, nUse, kUse, true));
        }
    } else {
        const int64_t mnTiles = mParts * nParts;
        const int64_t rounds = CeilDiv<int64_t>(mnTiles, usedCores);
        const int64_t fullRoundCores = mnTiles % usedCores == 0 ? usedCores : mnTiles % usedCores;
        for (int32_t core = 0; core < usedCores; ++core) {
            const int64_t coreRounds = core < fullRoundCores ? rounds : rounds - 1;
            const int64_t firstTile = core < fullRoundCores ?
                static_cast<int64_t>(core) * rounds :
                fullRoundCores * rounds + (core - fullRoundCores) * (rounds - 1);
            for (int64_t round = 0; round < coreRounds; ++round) {
                const int64_t tile = firstTile + round;
                const int64_t mIndex = tile / nParts;
                const int64_t nIndex = tile % nParts;
                const int64_t mUse = std::min<int64_t>(t.singleCoreM, w.m - mIndex * t.singleCoreM);
                const int64_t nUse = std::min<int64_t>(t.singleCoreN, w.n - nIndex * t.singleCoreN);
                cores[core].Add(LowerTileToHardwarePath(w, t, parameters_, mUse, nUse, w.k, false));
            }
        }
    }

    double actualMac = 0.0;
    double paddedMac = 0.0;
    double logicalInputBytes = 0.0;
    hardware_cost::KernelProgram program;
    program.availableCores = std::max(1, std::min(w.maxCores, caps_.coreNum));
    program.launchCycles = parameters_.kernelFixedCycles;
    if (splitK) {
        program.synchronizationCycles = parameters_.splitKBaseCycles +
            parameters_.splitKPerCoreCycles * usedCores;
    }
    program.corePaths.reserve(cores.size());
    for (int32_t core = 0; core < usedCores; ++core) {
        const CorePlan &cost = cores[core];
        program.corePaths.push_back(hardware_cost::PathNode::Sequence(cost.paths));
        actualMac += cost.actualMac;
        paddedMac += cost.paddedMac;
        logicalInputBytes += cost.logicalInputBytes;
        out.estimatedAGmBytes += cost.aGmBytes;
        out.estimatedBGmBytes += cost.bGmBytes;
        out.estimatedCGmBytes += cost.cGmBytes;
        out.estimatedMte1Bytes += cost.mte1Bytes;
        out.estimatedMmadCount += cost.mmads;
        out.estimatedOutputTileCount += cost.outputTiles;
    }

    const hardware_cost::HardwareCostModel hardwareModel(
        MakeHardwareProfile(caps_, parameters_, usedCores));
    const hardware_cost::KernelCost hardware = hardwareModel.Evaluate(program);
    if (!hardware.valid) {
        out.total = std::numeric_limits<double>::infinity();
        return out;
    }

    const auto resource = [](hardware_cost::Resource value) {
        return static_cast<std::size_t>(value);
    };
    out.total = hardware.totalCycles;
    out.criticalCoreId = hardware.criticalCore;
    out.criticalCoreCycles = hardware.criticalCoreCycles;
    out.averageCoreCycles = hardware.averageCoreCycles;
    out.pipelineCycles = hardware.criticalCoreCycles;
    out.cubeCycles = hardware.criticalResourceCycles[resource(hardware_cost::Resource::CUBE)];
    out.mte2Cycles = hardware.criticalResourceCycles[resource(hardware_cost::Resource::MTE2)];
    out.mte1Cycles = hardware.criticalResourceCycles[resource(hardware_cost::Resource::MTE1)];
    out.fixpipeCycles = hardware.criticalResourceCycles[resource(hardware_cost::Resource::FIXPIPE)];
    out.scalarCycles = hardware.criticalResourceCycles[resource(hardware_cost::Resource::SCALAR)];
    out.fillDrainCycles = hardware.fillDrainCycles;
    out.balancePenalty = hardware.balanceCycles;
    out.launchCycles = hardware.launchCycles;
    out.splitKPenalty = hardware.synchronizationCycles;
    out.gmCycles = hardware.hbmCycles;
    out.coreUtilization = hardware.coreUtilization;

    if (splitK) {
        const int64_t c0 = 32 / OutputBytes(w.dtype);
        out.estimatedCGmBytes += static_cast<double>(AlignUp(static_cast<int64_t>(w.m) * w.n, c0)) *
            OutputBytes(w.dtype);
    }

    out.l1Cycles = out.mte1Cycles;
    out.estimatedGmBytes = out.estimatedAGmBytes + out.estimatedBGmBytes + out.estimatedCGmBytes;

    out.tailEfficiency = std::min(1.0, SafeRatio(actualMac, paddedMac));
    const double cubeMacPerCycle = static_cast<double>((256 / DTypeBits(w.dtype)) * 256);
    out.tailPenalty = std::max(0.0, paddedMac - actualMac) /
        (cubeMacPerCycle * usedCores);
    out.l1CacheHitRate = std::clamp(1.0 - SafeRatio(
        out.estimatedAGmBytes + out.estimatedBGmBytes, logicalInputBytes), 0.0, 1.0);
    out.arithmeticIntensity = SafeRatio(2.0 * actualMac, out.estimatedGmBytes);

    return out;
}

}  // namespace matmul_search
