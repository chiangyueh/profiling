#include "indexed_update_path.h"

#include "hardware_path_builders.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace hardware_cost {
namespace {

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

PathNode ConcurrentTransfers(std::vector<PathNode> transfers)
{
    // Consecutive DataCopyPad requests overlap their completion latency, but
    // still share MTE issue and byte service.
    return Concurrent(std::move(transfers));
}

double CastLaneWork(uint64_t elements, uint32_t sourceBytes, uint32_t destinationBytes)
{
    // VECTOR's profile rate is in 32-bit lanes/cycle.
    return static_cast<double>(elements) *
        std::max<uint32_t>(sourceBytes, destinationBytes) / 4.0;
}

PathNode LowerSmallIteration(const IndexedUpdateProblem &problem,
                             uint64_t currentRows, uint64_t inputAlign,
                             uint64_t indicesAlign, uint64_t updatesAlign,
                             uint64_t indicesOneTime, uint64_t inputOneTime,
                             uint64_t updatesOneTime)
{
    const double indexElements = static_cast<double>(currentRows) * indicesOneTime;
    const double updateElements = static_cast<double>(currentRows) * updatesOneTime;
    const double inputElements = static_cast<double>(currentRows) * inputOneTime;
    std::vector<PathNode> stages;

    stages.push_back(EventHandshake());
    stages.push_back(ConcurrentTransfers({
        Transfer(Resource::MTE2, MemorySpace::GM, MemorySpace::UB,
                 indexElements * problem.indexBytes),
        Transfer(Resource::MTE2, MemorySpace::GM, MemorySpace::UB,
                 updateElements * problem.valueBytes),
        Transfer(Resource::MTE2, MemorySpace::GM, MemorySpace::UB,
                 inputElements * problem.valueBytes),
    }));

    if (problem.promoteAccumulatorToFp32) {
        stages.push_back(EventHandshake());
        // This is the executed CANN-8.1 extent: its ProcessSmall update cast
        // uses indicesAlign, while the allocation still uses updatesAlign.
        stages.push_back(Operations(Resource::VECTOR,
            CastLaneWork(inputAlign, problem.valueBytes, 4) +
            CastLaneWork(indicesAlign, problem.valueBytes, 4), 2.0));
    }
    if (problem.indexBytes == 8) {
        stages.push_back(EventHandshake());
        stages.push_back(Operations(Resource::VECTOR,
            CastLaneWork(indicesAlign, problem.indexBytes, 4), 1.0));
    }
    stages.push_back(PipeBarrier());

    // The nested j/k loop always pays address and index work; the update path
    // then performs either a set or a read/add/write.
    const double scalarOperations = indexElements *
        (6.0 + (problem.accumulate ? 4.0 : 2.0));
    stages.push_back(Operations(Resource::SCALAR, scalarOperations));
    stages.push_back(PipeBarrier());

    if (problem.promoteAccumulatorToFp32) {
        stages.push_back(Operations(Resource::VECTOR,
            CastLaneWork(inputAlign, 4, problem.valueBytes), 1.0));
        stages.push_back(EventHandshake());
    }
    stages.push_back(Transfer(Resource::MTE3, MemorySpace::UB, MemorySpace::GM,
                              inputElements * problem.valueBytes));
    (void)updatesAlign;
    return PathNode::Sequence(std::move(stages));
}

PathNode LowerPartitionedInputLoop(const IndexedUpdateProblem &problem,
                                   const IndexedUpdateSchedule &schedule,
                                   uint64_t currentInput)
{
    std::vector<PathNode> stages;
    stages.push_back(EventHandshake());
    stages.push_back(Transfer(Resource::MTE2, MemorySpace::GM, MemorySpace::UB,
                              static_cast<double>(currentInput) * problem.valueBytes));
    if (problem.promoteAccumulatorToFp32) {
        stages.push_back(EventHandshake());
        stages.push_back(Operations(Resource::VECTOR,
            CastLaneWork(schedule.inputAlign, problem.valueBytes, 4), 1.0));
    }

    for (uint64_t loop = 0; loop < schedule.indicesLoop; ++loop) {
        const uint64_t currentIndices = loop + 1 == schedule.indicesLoop ?
            schedule.indicesLast : schedule.indicesEach;
        stages.push_back(ConcurrentTransfers({
            Transfer(Resource::MTE2, MemorySpace::GM, MemorySpace::UB,
                     static_cast<double>(currentIndices) * problem.indexBytes),
            Transfer(Resource::MTE2, MemorySpace::GM, MemorySpace::UB,
                     static_cast<double>(currentIndices) * problem.valueBytes),
        }));
        stages.push_back(EventHandshake());

        double vectorWork = static_cast<double>(schedule.indicesAlign);
        double vectorInstructions = 1.0;
        if (problem.indexBytes == 8) {
            vectorWork += CastLaneWork(schedule.indicesAlign, problem.indexBytes, 4);
            vectorInstructions += 1.0;
        }
        if (problem.promoteAccumulatorToFp32) {
            vectorWork += CastLaneWork(schedule.updatesAlign, problem.valueBytes, 4);
            vectorInstructions += 1.0;
        }
        stages.push_back(Operations(Resource::VECTOR, vectorWork, vectorInstructions));
        if (problem.indexBytes == 8) stages.push_back(PipeBarrier());
        stages.push_back(EventHandshake());

        const double inspected = static_cast<double>(currentIndices);
        const double updateFraction = schedule.inputOneTime == 0 ? 1.0 :
            std::min(1.0, static_cast<double>(currentInput) / schedule.inputOneTime);
        const double committed = inspected * updateFraction;
        stages.push_back(Operations(Resource::SCALAR,
            inspected * 4.0 + committed * (problem.accumulate ? 4.0 : 2.0)));
    }

    stages.push_back(PipeBarrier());
    if (problem.promoteAccumulatorToFp32) {
        stages.push_back(Operations(Resource::VECTOR,
            CastLaneWork(schedule.inputAlign, 4, problem.valueBytes), 1.0));
        stages.push_back(EventHandshake());
    }
    stages.push_back(Transfer(Resource::MTE3, MemorySpace::UB, MemorySpace::GM,
                              static_cast<double>(currentInput) * problem.valueBytes));
    return PathNode::Sequence(std::move(stages));
}

uint64_t LiveUbBytes(const IndexedUpdateProblem &problem, uint64_t inputAlign,
                     uint64_t indicesAlign, uint64_t updatesAlign)
{
    uint64_t bytes = inputAlign * problem.valueBytes;
    bytes += updatesAlign * problem.valueBytes;
    bytes += indicesAlign * 4;
    if (problem.indexBytes == 8) bytes += indicesAlign * problem.indexBytes;
    if (problem.promoteAccumulatorToFp32) {
        bytes += (inputAlign + updatesAlign) * 4;
    }
    return bytes;
}

}  // namespace

KernelProgram LowerIndexedUpdatePath(const IndexedUpdateProblem &problem,
                                     const IndexedUpdateSchedule &schedule)
{
    KernelProgram program;
    program.availableCores = std::max<int32_t>(1, problem.availableVectorCores);
    program.launchCycles = 0.0;
    const uint64_t cores = std::max<uint64_t>(1, schedule.usedCoreNum);
    program.corePaths.reserve(static_cast<std::size_t>(cores));

    uint64_t peakUb = 0;
    for (uint64_t core = 0; core < cores; ++core) {
        std::vector<PathNode> coreWork;
        if (schedule.mode == 1) {
            const bool lastCore = core + 1 == cores;
            const uint64_t loopCount = lastCore ? schedule.lastIndicesLoop : schedule.indicesLoop;
            const uint64_t each = lastCore ? schedule.lastIndicesEach : schedule.indicesEach;
            const uint64_t last = lastCore ? schedule.lastIndicesLast : schedule.indicesLast;
            const uint64_t inputAlign = AlignUp(each * schedule.inputOneTime, 32);
            const uint64_t indicesAlign = AlignUp(each * schedule.indicesOneTime, 32);
            const uint64_t updatesAlign = AlignUp(each * schedule.updatesOneTime, 32);
            peakUb = std::max(peakUb, LiveUbBytes(
                problem, inputAlign, indicesAlign, updatesAlign));

            for (uint64_t loop = 0; loop < loopCount; ++loop) {
                const uint64_t currentRows = loop + 1 == loopCount ? last : each;
                coreWork.push_back(LowerSmallIteration(
                    problem, currentRows, inputAlign, indicesAlign, updatesAlign,
                    schedule.indicesOneTime, schedule.inputOneTime,
                    schedule.updatesOneTime));
            }
        } else {
            uint64_t currentNum = 0;
            uint64_t pieceEach = schedule.inputEach;
            uint64_t pieceLast = schedule.inputLast;
            if (schedule.eachNum == 0) {
                const uint64_t eachPiece = std::max<uint64_t>(1, schedule.eachPiece);
                const uint64_t currentPiece = core % eachPiece;
                currentNum = 1;
                if (currentPiece + 1 == eachPiece) {
                    const uint64_t consumed = schedule.inputOnePiece * (eachPiece - 1);
                    const uint64_t remaining = schedule.inputOneTime > consumed ?
                        schedule.inputOneTime - consumed : 0;
                    pieceEach = CeilDiv(remaining, std::max<uint64_t>(1, schedule.inputLoop));
                    pieceLast = remaining > pieceEach * (schedule.inputLoop - 1) ?
                        remaining - pieceEach * (schedule.inputLoop - 1) : 0;
                }
            } else {
                currentNum = schedule.eachNum + (core < schedule.extraTaskCore ? 1 : 0);
            }

            peakUb = std::max(peakUb, LiveUbBytes(
                problem, schedule.inputAlign, schedule.indicesAlign,
                schedule.updatesAlign));
            for (uint64_t item = 0; item < currentNum; ++item) {
                for (uint64_t loop = 0; loop < schedule.inputLoop; ++loop) {
                    const uint64_t currentInput = loop + 1 == schedule.inputLoop ?
                        pieceLast : pieceEach;
                    coreWork.push_back(LowerPartitionedInputLoop(
                        problem, schedule, currentInput));
                }
            }
        }
        program.corePaths.push_back(PathNode::Sequence(std::move(coreWork)));
    }
    program.peakMemoryBytes[MemoryIndex(MemorySpace::UB)] = peakUb;
    return program;
}

}  // namespace hardware_cost
