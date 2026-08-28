#pragma once

#include <cstdint>
#include <vector>

#include "hardware_cost_model.h"

namespace hardware_cost {

// Semantic input to an indexed-read kernel.  This describes tensor geometry
// and element representation only; it intentionally carries no operator
// name, measured latency, or empirical rank.
struct IndexedReadProblem {
    std::vector<uint64_t> sourceShape;
    std::vector<uint64_t> indexShape;
    uint32_t axis = 0;
    uint32_t valueBytes = 2;
    uint32_t indexBytes = 4;
    int32_t availableVectorCores = 20;
    uint64_t physicalUbBytes = 196352;
};

enum class IndexedReadRoute : uint32_t {
    TRANSPOSE = 1,
    LAST_DIMENSION = 2,
};

// Hardware-facing schedule reconstructed from the source tiler's bounded
// core/UB inputs.  Fields are execution geometry, not learned features.
struct IndexedReadSchedule {
    IndexedReadRoute route = IndexedReadRoute::LAST_DIMENSION;
    uint64_t activeCores = 1;
    uint64_t visibleUbBytes = 0;
    uint64_t nonCollectingRows = 1;
    uint64_t sourceAxisElements = 1;
    uint64_t indexAxisElements = 1;
    uint64_t preElements = 1;
    uint64_t postElements = 1;

    bool batchProcess = false;
    bool scalarMode = false;
    bool specialDataMove = false;
    uint64_t eachCalculationLines = 1;
    uint64_t sourceSliceCount = 1;
    uint64_t indexSliceCount = 1;
    uint64_t scalarModeLength = 0;
    uint64_t sourceBufferBytes = 0;
    uint64_t indexBufferBytes = 0;
    uint64_t outputBufferBytes = 0;
    uint64_t maskBufferBytes = 0;

    uint64_t carryElements = 1;
    uint64_t transposeAxisSlice = 128;
    uint64_t gatherIndexSlice = 1;
    uint64_t workspacePerCoreBytes = 0;
};

IndexedReadSchedule PlanIndexedReadSchedule(const IndexedReadProblem &problem,
                                            uint64_t coreCap,
                                            uint64_t ubDivisor);
KernelProgram LowerIndexedReadPath(const IndexedReadProblem &problem,
                                   const IndexedReadSchedule &schedule);

}  // namespace hardware_cost
