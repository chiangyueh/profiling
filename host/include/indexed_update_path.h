#pragma once

#include <cstdint>

#include "hardware_cost_model.h"

namespace hardware_cost {

// Generic semantics for an indexed in-place update path.  There is no
// operator identity here: any frontend with the same GM/UB/vector/scalar
// behavior lowers to this structure and is evaluated by HardwareCostModel.
struct IndexedUpdateProblem {
    uint32_t valueBytes = 2;
    uint32_t indexBytes = 4;
    bool accumulate = false;
    bool promoteAccumulatorToFp32 = false;
    int32_t availableVectorCores = 40;
};

// Raw schedule inputs emitted by the source tiler.  Branches in the lowerer
// follow these fields and the kernel control flow; they never inspect an
// operator name, workload id, measured rank, or measured latency.
struct IndexedUpdateSchedule {
    uint64_t usedCoreNum = 1;
    uint64_t eachNum = 0;
    uint64_t extraTaskCore = 0;
    uint64_t eachPiece = 1;
    uint64_t inputOnePiece = 0;
    uint64_t inputOneTime = 1;
    uint64_t indicesOneTime = 1;
    uint64_t updatesOneTime = 1;
    uint64_t inputEach = 0;
    uint64_t indicesEach = 0;
    uint64_t inputLast = 0;
    uint64_t indicesLast = 0;
    uint64_t inputLoop = 0;
    uint64_t indicesLoop = 0;
    uint64_t inputAlign = 0;
    uint64_t indicesAlign = 0;
    uint64_t updatesAlign = 0;
    uint64_t lastIndicesLoop = 0;
    uint64_t lastIndicesEach = 0;
    uint64_t lastIndicesLast = 0;
    uint64_t oneTime = 0;
    uint64_t mode = 0;
};

KernelProgram LowerIndexedUpdatePath(const IndexedUpdateProblem &problem,
                                     const IndexedUpdateSchedule &schedule);

}  // namespace hardware_cost
