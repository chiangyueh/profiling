#include "proxy_model.h"

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

double OverlapStages(double first, double second, double workItems, double &fillDrain)
{
    if (first <= 0.0) return second;
    if (second <= 0.0) return first;
    if (workItems <= 1.0) return first + second;

    // For a double-buffered two-stage pipeline, the bottleneck total is the
    // steady state. One average non-bottleneck item accounts for fill/drain.
    const double fill = std::min(first, second) / workItems;
    fillDrain += fill;
    return std::max(first, second) + fill;
}

struct TileCost {
    double cycles = 0.0;
    double pipelineCycles = 0.0;
    double cubeCycles = 0.0;
    double mte2Cycles = 0.0;
    double mte1Cycles = 0.0;
    double fixpipeCycles = 0.0;
    double scalarCycles = 0.0;
    double fillDrainCycles = 0.0;
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

struct CoreCost : TileCost {
    void Add(const TileCost &tile)
    {
        cycles += tile.cycles;
        pipelineCycles += tile.pipelineCycles;
        cubeCycles += tile.cubeCycles;
        mte2Cycles += tile.mte2Cycles;
        mte1Cycles += tile.mte1Cycles;
        fixpipeCycles += tile.fixpipeCycles;
        scalarCycles += tile.scalarCycles;
        fillDrainCycles += tile.fillDrainCycles;
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

TileCost PredictTile(const Workload &w, const TCubeTiling &t, const ProxyWeights &weights,
                     int64_t m, int64_t n, int64_t k, bool atomicOutput)
{
    TileCost out;
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

    out.cubeCycles = out.paddedMac / cubeMacPerCycle + out.mmads * weights.cubeIssueCycles;

    const double aMte1Bytes = static_cast<double>(alignedM) * alignedK * inputBytes * nTiles;
    const double bMte1Bytes = static_cast<double>(alignedN) * alignedK * inputBytes * mTiles;
    out.mte1Bytes = aMte1Bytes + bMte1Bytes;
    out.mte1Cycles = out.mte1Bytes / std::max(1.0, weights.mte1BytesPerCyclePerCore) +
        2.0 * out.mmads * weights.mte1IssueCycles;

    const double aLoadBlocks = static_cast<double>(mTiles) * kTiles * aRepeat;
    const double bLoadBlocks = static_cast<double>(nTiles) * kTiles * bRepeat;
    const double mte2A = out.aGmBytes / std::max(1.0, weights.mte2BytesPerCyclePerCore) +
        aLoadBlocks * weights.mte2IssueCycles;
    const double mte2B = out.bGmBytes / std::max(1.0, weights.mte2BytesPerCyclePerCore) +
        bLoadBlocks * weights.mte2IssueCycles;
    out.mte2Cycles = mte2A + mte2B;

    const double fixpipeBytes = static_cast<double>(alignedM) * alignedN *
        (accumulatorBytes + outputBytes);
    out.fixpipeCycles = fixpipeBytes / std::max(1.0, weights.fixpipeBytesPerCyclePerCore) +
        out.outputTiles * weights.fixpipeIssueCycles;

    out.scalarCycles = weights.scalarCoreSetupCycles +
        out.mmads * weights.scalarPerMmadCycles +
        out.outputTiles * weights.scalarPerOutputTileCycles;

    const bool l0InputDb = t.dbL0A == 2 && t.dbL0B == 2;
    double downstream = l0InputDb ?
        OverlapStages(out.mte1Cycles, out.cubeCycles, out.mmads, out.fillDrainCycles) :
        out.mte1Cycles + out.cubeCycles;

    if (t.dbL0C == 2) {
        downstream = OverlapStages(downstream, out.fixpipeCycles,
                                   out.outputTiles, out.fillDrainCycles);
    } else {
        downstream += out.fixpipeCycles;
    }

    const bool aL1Db = HasL1DoubleBuffer(t.depthA1, t.stepM, t.stepKa);
    const bool bL1Db = HasL1DoubleBuffer(t.depthB1, t.stepN, t.stepKb);
    if (aL1Db && bL1Db) {
        downstream = OverlapStages(out.mte2Cycles, downstream,
                                   std::max(1.0, aLoadBlocks + bLoadBlocks), out.fillDrainCycles);
    } else if (aL1Db || bL1Db) {
        const double overlappedMte2 = aL1Db ? mte2A : mte2B;
        const double serializedMte2 = aL1Db ? mte2B : mte2A;
        downstream = serializedMte2 + OverlapStages(overlappedMte2, downstream,
            std::max(1.0, aL1Db ? aLoadBlocks : bLoadBlocks), out.fillDrainCycles);
    } else {
        downstream += out.mte2Cycles;
    }

    out.pipelineCycles = downstream;
    out.cycles = std::max(out.pipelineCycles, out.scalarCycles);
    return out;
}

}  // namespace

ProxyModel::ProxyModel(PlatformCaps caps, ProxyWeights weights)
    : caps_(caps), weights_(weights)
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
    std::vector<CoreCost> cores(static_cast<size_t>(usedCores));
    ProxyWeights effectiveWeights = weights_;
    const double hbmBytesPerCyclePerCore = caps_.hbmBytesPerCycle > 0 ?
        static_cast<double>(caps_.hbmBytesPerCycle) : weights_.fallbackHbmBytesPerCycle;
    effectiveWeights.mte2BytesPerCyclePerCore = std::min(
        effectiveWeights.mte2BytesPerCyclePerCore, hbmBytesPerCyclePerCore);

    if (splitK) {
        for (int32_t core = 0; core < usedCores && core < totalTiles; ++core) {
            const int64_t mIndex = core % mParts;
            const int64_t nIndex = (core / mParts) % nParts;
            const int64_t kIndex = core / (mParts * nParts);
            const int64_t mUse = std::min<int64_t>(t.singleCoreM, w.m - mIndex * t.singleCoreM);
            const int64_t nUse = std::min<int64_t>(t.singleCoreN, w.n - nIndex * t.singleCoreN);
            const int64_t kUse = std::min<int64_t>(t.singleCoreK, w.k - kIndex * t.singleCoreK);
            cores[core].Add(PredictTile(w, t, effectiveWeights, mUse, nUse, kUse, true));
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
                cores[core].Add(PredictTile(w, t, effectiveWeights, mUse, nUse, w.k, false));
            }
        }
    }

    double coreCycleSum = 0.0;
    double actualMac = 0.0;
    double paddedMac = 0.0;
    double logicalInputBytes = 0.0;
    for (int32_t core = 0; core < usedCores; ++core) {
        const CoreCost &cost = cores[core];
        coreCycleSum += cost.cycles;
        actualMac += cost.actualMac;
        paddedMac += cost.paddedMac;
        logicalInputBytes += cost.logicalInputBytes;
        out.estimatedAGmBytes += cost.aGmBytes;
        out.estimatedBGmBytes += cost.bGmBytes;
        out.estimatedCGmBytes += cost.cGmBytes;
        out.estimatedMte1Bytes += cost.mte1Bytes;
        out.estimatedMmadCount += cost.mmads;
        out.estimatedOutputTileCount += cost.outputTiles;
        if (out.criticalCoreId < 0 || cost.cycles > out.criticalCoreCycles) {
            out.criticalCoreId = core;
            out.criticalCoreCycles = cost.cycles;
            out.pipelineCycles = cost.pipelineCycles;
            out.cubeCycles = cost.cubeCycles;
            out.mte2Cycles = cost.mte2Cycles;
            out.mte1Cycles = cost.mte1Cycles;
            out.fixpipeCycles = cost.fixpipeCycles;
            out.scalarCycles = cost.scalarCycles;
            out.fillDrainCycles = cost.fillDrainCycles;
        }
    }

    if (splitK) {
        const int64_t c0 = 32 / OutputBytes(w.dtype);
        out.estimatedCGmBytes += static_cast<double>(AlignUp(static_cast<int64_t>(w.m) * w.n, c0)) *
            OutputBytes(w.dtype);
    }

    out.averageCoreCycles = coreCycleSum / usedCores;
    out.balancePenalty = std::max(0.0, out.criticalCoreCycles - out.averageCoreCycles);
    out.l1Cycles = out.mte1Cycles;
    out.estimatedGmBytes = out.estimatedAGmBytes + out.estimatedBGmBytes + out.estimatedCGmBytes;
    out.gmCycles = out.estimatedGmBytes /
        std::max(1.0, hbmBytesPerCyclePerCore * usedCores);

    const int32_t availableCores = std::max(1, std::min(w.maxCores, caps_.coreNum));
    out.coreUtilization = std::min(1.0, SafeRatio(usedCores, availableCores));
    out.tailEfficiency = std::min(1.0, SafeRatio(actualMac, paddedMac));
    const double cubeMacPerCycle = static_cast<double>((256 / DTypeBits(w.dtype)) * 256);
    out.tailPenalty = std::max(0.0, paddedMac - actualMac) /
        (cubeMacPerCycle * usedCores);
    out.l1CacheHitRate = std::clamp(1.0 - SafeRatio(
        out.estimatedAGmBytes + out.estimatedBGmBytes, logicalInputBytes), 0.0, 1.0);
    out.arithmeticIntensity = SafeRatio(2.0 * actualMac, out.estimatedGmBytes);

    out.launchCycles = weights_.kernelFixedCycles;
    if (splitK) {
        out.splitKPenalty = weights_.splitKBaseCycles +
            weights_.splitKPerCoreCycles * usedCores;
    }
    out.total = std::max(out.criticalCoreCycles, out.gmCycles) +
        out.launchCycles + out.splitKPenalty;
    return out;
}

}  // namespace matmul_search
