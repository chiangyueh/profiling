#include "indexed_read_path.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

#include "hardware_path_builders.h"

namespace hardware_cost {
namespace {

constexpr uint64_t kBlockBytes = 32;
constexpr uint64_t kLastDimReservedUb = 10 * 1024;
constexpr uint64_t kTransposeReservedUb = 2048;
constexpr uint64_t kTransposeSlice = 128;
constexpr uint64_t kCacheLineBytes = 512;

constexpr std::size_t MemoryIndex(MemorySpace value)
{
    return static_cast<std::size_t>(value);
}

uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return alignment == 0 ? value : CeilDiv(value, alignment) * alignment;
}

uint64_t Product(const std::vector<uint64_t> &shape, std::size_t begin,
                 std::size_t end)
{
    uint64_t product = 1;
    for (std::size_t i = begin; i < end; ++i) {
        if (shape[i] != 0 && product > std::numeric_limits<uint64_t>::max() / shape[i]) {
            throw std::overflow_error("indexed-read tensor geometry overflows uint64");
        }
        product *= shape[i];
    }
    return product;
}

PathNode RandomReadTransactions(double usefulBytes, double elements)
{
    // Scalar GlobalTensor::GetValue accesses cannot be represented as one
    // contiguous byte stream.  Each address consumes a GM transaction while
    // a bounded number of requests can remain outstanding.  The accounting
    // remains generic and can be reused by any irregular-load frontend.
    ResourceWork work;
    work.resource = Resource::MTE2;
    work.source = MemorySpace::GM;
    work.destination = MemorySpace::UB;
    work.intermediate = MemorySpace::L2;
    work.bytes = elements * static_cast<double>(kBlockBytes);
    work.sourceBytes = work.bytes;
    work.destinationBytes = usefulBytes;
    work.issues = elements;
    work.latencyWaves = CeilDiv(static_cast<uint64_t>(elements), 16);
    return PathNode::Work(work);
}

PathNode LowerLastDimensionCore(const IndexedReadProblem &problem,
                                const IndexedReadSchedule &schedule,
                                uint64_t rows, uint64_t virtualTasks)
{
    const double valueBytes = problem.valueBytes;
    const double indexBytes = problem.indexBytes;
    std::vector<PathNode> stages;

    uint64_t chunks = 0;
    double outputElements = 0.0;
    double sourceElements = 0.0;
    double indexElements = 0.0;
    double mte2SourceCalls = 0.0;
    double mte2IndexCalls = 0.0;
    double mte3Calls = 0.0;

    if (schedule.scalarModeLength != 0) {
        chunks = virtualTasks;
        const uint64_t elementsPerTask = CeilDiv(
            schedule.indexAxisElements, std::max<uint64_t>(1, schedule.indexSliceCount));
        outputElements = static_cast<double>(virtualTasks * elementsPerTask);
        indexElements = outputElements;
        mte2IndexCalls = static_cast<double>(chunks);
        mte3Calls = static_cast<double>(chunks);
    } else {
        outputElements = static_cast<double>(rows * schedule.indexAxisElements);
        indexElements = outputElements;
        if (schedule.scalarMode || schedule.batchProcess) {
            chunks = CeilDiv(rows, std::max<uint64_t>(1, schedule.eachCalculationLines));
        } else {
            chunks = rows * std::max<uint64_t>(1, schedule.indexSliceCount);
        }
        mte2IndexCalls = static_cast<double>(chunks);
        mte3Calls = static_cast<double>(chunks);
    }

    if (schedule.scalarMode) {
        stages.push_back(Transfer(Resource::MTE2, MemorySpace::GM, MemorySpace::UB,
            indexElements * indexBytes, mte2IndexCalls, mte2IndexCalls));
        stages.push_back(Operations(Resource::VECTOR, outputElements * 3.0,
                                    static_cast<double>(chunks) * 3.0));
        stages.push_back(PipeBarrier(static_cast<double>(chunks)));
        stages.push_back(Operations(Resource::SCALAR,
            outputElements * (6.0 + static_cast<double>(problem.sourceShape.size()))));
        stages.push_back(RandomReadTransactions(outputElements * valueBytes,
                                                outputElements));
    } else {
        uint64_t sourceReplays = 1;
        if (!schedule.batchProcess && schedule.sourceSliceCount > 1) {
            sourceReplays = std::max<uint64_t>(1, schedule.indexSliceCount);
        }
        sourceElements = static_cast<double>(rows * schedule.sourceAxisElements * sourceReplays);
        mte2SourceCalls = schedule.batchProcess ? static_cast<double>(chunks) :
            static_cast<double>(rows * sourceReplays *
                std::max<uint64_t>(1, schedule.sourceSliceCount));
        stages.push_back(Transfer(Resource::MTE2, MemorySpace::GM, MemorySpace::UB,
            indexElements * indexBytes, mte2IndexCalls, mte2IndexCalls));
        stages.push_back(Transfer(Resource::MTE2, MemorySpace::GM, MemorySpace::UB,
            sourceElements * valueBytes, mte2SourceCalls, mte2SourceCalls));

        double vectorOperations = outputElements *
            (schedule.sourceSliceCount > 1 ?
                5.0 + 7.0 * schedule.sourceSliceCount : 5.0);
        if (schedule.indexAxisElements == 1) vectorOperations += outputElements * 3.0;
        stages.push_back(Operations(Resource::VECTOR, vectorOperations,
                                    static_cast<double>(chunks) * 5.0));
        stages.push_back(PipeBarrier(static_cast<double>(chunks)));
    }

    stages.push_back(Transfer(Resource::MTE3, MemorySpace::UB, MemorySpace::GM,
        outputElements * valueBytes, mte3Calls, mte3Calls));
    // Each DataCopy path has a producer/consumer set+wait pair on both sides.
    stages.push_back(EventHandshake(4.0 *
        (mte2SourceCalls + mte2IndexCalls + mte3Calls)));
    return PathNode::Sequence(std::move(stages));
}

std::vector<std::pair<uint64_t, uint64_t>> TransposeCoreGeometry(
    uint64_t pre, uint64_t post, uint64_t cores)
{
    std::vector<std::pair<uint64_t, uint64_t>> geometry;
    geometry.reserve(static_cast<std::size_t>(cores));
    if (pre > cores) {
        const uint64_t base = pre / cores;
        const uint64_t extra = pre % cores;
        for (uint64_t core = 0; core < cores; ++core) {
            geometry.emplace_back(base + (core < extra ? 1 : 0), post);
        }
        return geometry;
    }

    const uint64_t baseCores = cores / pre;
    const uint64_t extraGroups = cores % pre;
    for (uint64_t group = 0; group < pre; ++group) {
        const uint64_t groupCores = baseCores + (group < extraGroups ? 1 : 0);
        const uint64_t basePost = post / groupCores;
        const uint64_t extraPost = post % groupCores;
        for (uint64_t local = 0; local < groupCores; ++local) {
            geometry.emplace_back(1, basePost + (local < extraPost ? 1 : 0));
        }
    }
    return geometry;
}

PathNode LowerTransposeCore(const IndexedReadProblem &problem,
                            const IndexedReadSchedule &schedule,
                            uint64_t pre, uint64_t post)
{
    const uint64_t lanes = pre * post;
    const uint64_t carryPartsPerPre = CeilDiv(post, schedule.carryElements);
    const uint64_t carryParts = pre * carryPartsPerPre;
    const uint64_t sourceSlices = CeilDiv(
        schedule.sourceAxisElements, schedule.transposeAxisSlice);
    const uint64_t indexTransposeSlices = CeilDiv(
        schedule.indexAxisElements, schedule.transposeAxisSlice);
    const uint64_t gatherSlices = CeilDiv(
        schedule.indexAxisElements, schedule.gatherIndexSlice);
    const uint64_t outputSlices = indexTransposeSlices;

    const double sourceBytes = static_cast<double>(lanes * schedule.sourceAxisElements) *
        problem.valueBytes;
    const double indexBytes = static_cast<double>(lanes * schedule.indexAxisElements) *
        problem.indexBytes;
    const double outputBytes = static_cast<double>(lanes * schedule.indexAxisElements) *
        problem.valueBytes;

    // Original GM -> UB, workspace round trip, gather workspace, and final
    // workspace -> output are all explicit in the source kernel.
    const double mte2Bytes = 2.0 * sourceBytes + 2.0 * indexBytes + outputBytes;
    const double mte3Bytes = sourceBytes + indexBytes + 2.0 * outputBytes;
    const double mte2Calls = static_cast<double>(carryParts) *
        (sourceSlices + indexTransposeSlices + outputSlices) +
        static_cast<double>(lanes) * (1 + gatherSlices);
    const double mte3Calls = static_cast<double>(carryParts) *
        (sourceSlices + indexTransposeSlices + outputSlices) +
        static_cast<double>(lanes * gatherSlices);

    std::vector<PathNode> stages;
    stages.push_back(Transfer(Resource::MTE2, MemorySpace::GM, MemorySpace::UB,
                              mte2Bytes, mte2Calls, mte2Calls));
    const double vectorOperations =
        2.0 * (sourceBytes + indexBytes + outputBytes) / kBlockBytes +
        static_cast<double>(lanes * schedule.indexAxisElements) * 5.0;
    stages.push_back(Operations(Resource::VECTOR, vectorOperations,
                                mte2Calls + mte3Calls));
    stages.push_back(PipeBarrier(static_cast<double>(carryParts) * 3.0));
    stages.push_back(Transfer(Resource::MTE3, MemorySpace::UB, MemorySpace::GM,
                              mte3Bytes, mte3Calls, mte3Calls));
    stages.push_back(EventHandshake(4.0 * (mte2Calls + mte3Calls)));
    return PathNode::Sequence(std::move(stages));
}

}  // namespace

IndexedReadSchedule PlanIndexedReadSchedule(const IndexedReadProblem &problem,
                                            uint64_t coreCap,
                                            uint64_t ubDivisor)
{
    if (problem.sourceShape.empty() || problem.sourceShape.size() != problem.indexShape.size() ||
        problem.axis >= problem.sourceShape.size() || problem.valueBytes == 0 ||
        problem.indexBytes == 0 || coreCap == 0 || ubDivisor == 0) {
        throw std::invalid_argument("invalid indexed-read problem or schedule bound");
    }
    if (std::any_of(problem.sourceShape.begin(), problem.sourceShape.end(),
                    [](uint64_t value) { return value == 0; }) ||
        std::any_of(problem.indexShape.begin(), problem.indexShape.end(),
                    [](uint64_t value) { return value == 0; })) {
        throw std::invalid_argument("indexed-read shapes must be nonzero");
    }
    const uint64_t boundedCoreCap = std::min<uint64_t>(
        coreCap, std::max<int32_t>(1, problem.availableVectorCores));

    IndexedReadSchedule schedule;
    schedule.visibleUbBytes = problem.physicalUbBytes / ubDivisor;
    schedule.sourceAxisElements = problem.sourceShape[problem.axis];
    schedule.indexAxisElements = problem.indexShape[problem.axis];
    schedule.preElements = Product(problem.indexShape, 0, problem.axis);
    schedule.postElements = Product(problem.indexShape, problem.axis + 1,
                                    problem.indexShape.size());

    if (problem.axis + 1 != problem.sourceShape.size()) {
        schedule.route = IndexedReadRoute::TRANSPOSE;
        schedule.activeCores = std::max<uint64_t>(1, std::min(
            schedule.preElements * schedule.postElements, boundedCoreCap));
        schedule.carryElements = std::max<uint64_t>(1, kCacheLineBytes / problem.valueBytes);
        schedule.transposeAxisSlice = kTransposeSlice;
        const uint64_t available = schedule.visibleUbBytes > kTransposeReservedUb ?
            schedule.visibleUbBytes - kTransposeReservedUb : 0;
        const uint64_t xAlignedBytes = AlignUp(
            schedule.sourceAxisElements, kBlockBytes / problem.valueBytes) * problem.valueBytes;
        const uint64_t transBuffer = kTransposeSlice * kCacheLineBytes;
        const uint64_t fixedBuffers = std::max(transBuffer, xAlignedBytes) + transBuffer;
        const uint64_t remaining = available > fixedBuffers ? available - fixedBuffers : 0;
        const uint64_t maxIndexSlice = remaining / (kBlockBytes * 2) *
            (kBlockBytes / problem.indexBytes);
        schedule.gatherIndexSlice = std::max<uint64_t>(1, std::min(
            AlignUp(schedule.indexAxisElements, kBlockBytes / problem.indexBytes),
            maxIndexSlice));
        const uint64_t usedWorkspaceLanes = std::min(
            schedule.carryElements, schedule.postElements);
        schedule.workspacePerCoreBytes = usedWorkspaceLanes *
            (AlignUp(schedule.sourceAxisElements * problem.valueBytes,
                     problem.indexBytes) +
             schedule.indexAxisElements * problem.indexBytes);
        schedule.sourceBufferBytes = transBuffer;
        schedule.outputBufferBytes = transBuffer;
        return schedule;
    }

    schedule.route = IndexedReadRoute::LAST_DIMENSION;
    schedule.nonCollectingRows = Product(problem.sourceShape, 0,
                                         problem.sourceShape.size() - 1);
    const uint64_t internalValueBytes = problem.valueBytes == 1 ? 2 : problem.valueBytes;
    const uint64_t internalIndexBytes = problem.indexBytes == 4 ? 8 : problem.indexBytes;
    const uint64_t available = schedule.visibleUbBytes > kLastDimReservedUb ?
        schedule.visibleUbBytes - kLastDimReservedUb : 0;
    const uint64_t sourceAlign = AlignUp(
        schedule.sourceAxisElements * internalValueBytes,
        kBlockBytes * (problem.valueBytes == 1 ? 2 : 1));
    const uint64_t outputAlign = AlignUp(
        schedule.indexAxisElements * internalValueBytes,
        kBlockBytes * (problem.valueBytes == 1 ? 2 : 1));
    const uint64_t indexAlign = AlignUp(
        schedule.indexAxisElements * internalIndexBytes, kBlockBytes * 2);
    schedule.batchProcess = sourceAlign + outputAlign + indexAlign <= available / 2;

    if (schedule.batchProcess) {
        const uint64_t perLine = sourceAlign + outputAlign + indexAlign;
        schedule.eachCalculationLines = std::max<uint64_t>(1, available / perLine);
        schedule.sourceBufferBytes = sourceAlign * schedule.eachCalculationLines;
        schedule.indexBufferBytes = indexAlign * schedule.eachCalculationLines;
        schedule.outputBufferBytes = outputAlign * schedule.eachCalculationLines;
    } else if (sourceAlign <= available / 2) {
        schedule.sourceSliceCount = 1;
        schedule.eachCalculationLines = 1;
        schedule.sourceBufferBytes = sourceAlign;
        const uint64_t denominator = internalIndexBytes + internalValueBytes;
        schedule.indexBufferBytes = AlignUp(
            (available - sourceAlign) / denominator * internalIndexBytes,
            kBlockBytes * 2);
        schedule.outputBufferBytes = AlignUp(
            schedule.indexBufferBytes / internalIndexBytes * internalValueBytes,
            kBlockBytes);
        schedule.indexSliceCount = CeilDiv(indexAlign,
            std::max<uint64_t>(1, schedule.indexBufferBytes));
    } else {
        schedule.eachCalculationLines = 1;
        schedule.sourceBufferBytes = available / 2;
        schedule.sourceSliceCount = CeilDiv(sourceAlign,
            std::max<uint64_t>(1, schedule.sourceBufferBytes));
        const double denominator = static_cast<double>(internalIndexBytes) +
            2.0 * internalValueBytes + 0.25;
        schedule.indexBufferBytes = AlignUp(
            static_cast<uint64_t>(schedule.sourceBufferBytes / denominator * internalIndexBytes),
            kBlockBytes * 8);
        schedule.outputBufferBytes = AlignUp(
            schedule.indexBufferBytes / internalIndexBytes * internalValueBytes,
            kBlockBytes);
        schedule.indexSliceCount = CeilDiv(indexAlign,
            std::max<uint64_t>(1, schedule.indexBufferBytes));
        schedule.maskBufferBytes = AlignUp(
            schedule.indexBufferBytes / internalIndexBytes / 8, kBlockBytes);
    }

    schedule.scalarMode = schedule.sourceSliceCount > 5 ||
        schedule.sourceAxisElements / schedule.indexAxisElements > 256;
    if (schedule.scalarMode) {
        const uint64_t indexBlocks = CeilDiv(indexAlign, kBlockBytes * 2) * 2;
        const uint64_t outputBlocks = CeilDiv(
            schedule.indexAxisElements * problem.valueBytes, kBlockBytes);
        schedule.eachCalculationLines = std::max<uint64_t>(1,
            available / kBlockBytes / std::max<uint64_t>(1, indexBlocks + outputBlocks));
        const bool multiRow = schedule.eachCalculationLines > 1 &&
            (schedule.nonCollectingRows > boundedCoreCap ||
             (schedule.nonCollectingRows <= boundedCoreCap &&
              schedule.indexAxisElements < boundedCoreCap * kBlockBytes));
        if (multiRow) {
            schedule.indexBufferBytes = schedule.eachCalculationLines * indexBlocks * kBlockBytes;
            schedule.outputBufferBytes = schedule.eachCalculationLines * outputBlocks * kBlockBytes;
        } else {
            schedule.indexSliceCount = CeilDiv(indexAlign,
                std::max<uint64_t>(1, schedule.indexBufferBytes));
            schedule.scalarModeLength = schedule.indexSliceCount * schedule.nonCollectingRows;
            if (schedule.scalarModeLength < boundedCoreCap) {
                schedule.indexSliceCount = std::max<uint64_t>(1,
                    boundedCoreCap / schedule.nonCollectingRows);
                schedule.scalarModeLength = schedule.indexSliceCount *
                    schedule.nonCollectingRows;
            }
            schedule.indexBufferBytes = AlignUp(
                indexAlign / schedule.indexSliceCount + internalIndexBytes,
                kBlockBytes * 2);
            schedule.outputBufferBytes = AlignUp(
                schedule.indexBufferBytes / internalIndexBytes * problem.valueBytes,
                kBlockBytes);
        }
    }

    const uint64_t logicalWork = schedule.scalarModeLength != 0 ?
        schedule.scalarModeLength : schedule.nonCollectingRows;
    schedule.activeCores = std::max<uint64_t>(1, std::min(logicalWork, boundedCoreCap));
    return schedule;
}

KernelProgram LowerIndexedReadPath(const IndexedReadProblem &problem,
                                   const IndexedReadSchedule &schedule)
{
    KernelProgram program;
    program.availableCores = std::max<int32_t>(1, problem.availableVectorCores);
    program.corePaths.reserve(static_cast<std::size_t>(schedule.activeCores));

    if (schedule.route == IndexedReadRoute::TRANSPOSE) {
        const auto geometry = TransposeCoreGeometry(
            schedule.preElements, schedule.postElements, schedule.activeCores);
        for (const auto &core : geometry) {
            program.corePaths.push_back(LowerTransposeCore(
                problem, schedule, core.first, core.second));
        }
        program.peakMemoryBytes[MemoryIndex(MemorySpace::UB)] =
            schedule.sourceBufferBytes + schedule.outputBufferBytes;
        return program;
    }

    if (schedule.scalarModeLength != 0) {
        const uint64_t base = schedule.scalarModeLength / schedule.activeCores;
        const uint64_t extra = schedule.scalarModeLength % schedule.activeCores;
        for (uint64_t core = 0; core < schedule.activeCores; ++core) {
            const uint64_t tasks = base + (core < extra ? 1 : 0);
            program.corePaths.push_back(LowerLastDimensionCore(
                problem, schedule, 0, tasks));
        }
    } else {
        const uint64_t base = schedule.nonCollectingRows / schedule.activeCores;
        const uint64_t extra = schedule.nonCollectingRows % schedule.activeCores;
        for (uint64_t core = 0; core < schedule.activeCores; ++core) {
            const uint64_t rows = base + (core < extra ? 1 : 0);
            program.corePaths.push_back(LowerLastDimensionCore(
                problem, schedule, rows, 0));
        }
    }
    // Scalar mode never initializes the source tensor queue; source values
    // are fetched by indexed GlobalTensor loads.  Do not count the dormant
    // vector-path source buffer against live UB capacity.
    program.peakMemoryBytes[MemoryIndex(MemorySpace::UB)] =
        (schedule.scalarMode ? 0 : schedule.sourceBufferBytes) +
        schedule.indexBufferBytes + schedule.outputBufferBytes +
        schedule.maskBufferBytes * 2;
    return program;
}

}  // namespace hardware_cost
