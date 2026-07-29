#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "kernel_tiling/kernel_tiling.h"

namespace matmul_search {

enum class DType {
    FP16,
    BF16,
    FP32,
    INT8,
};

enum class ExecutionMode {
    NONE,
    BASE_ITERATE_ALL,
};

struct Workload {
    std::string id;
    int32_t m = 1;
    int32_t n = 1;
    int32_t k = 1;
    DType dtype = DType::FP16;
    bool transA = false;
    bool transB = false;
    int32_t maxCores = 24;
};

struct Candidate {
    int32_t singleM = -1;
    int32_t singleN = -1;
    int32_t singleK = -1;
    int32_t baseM = -1;
    int32_t baseN = -1;
    int32_t baseK = -1;
    int32_t traverse = 0;
    bool dbA = true;
    bool dbB = true;
    bool splitK = false;

    std::string Key() const;
    bool operator==(const Candidate &other) const;
};

struct ProxyBreakdown {
    double total = 0.0;
    int32_t criticalCoreId = -1;
    double criticalCoreCycles = 0.0;
    double averageCoreCycles = 0.0;
    double pipelineCycles = 0.0;
    double cubeCycles = 0.0;
    double gmCycles = 0.0;
    double l1Cycles = 0.0;
    double mte2Cycles = 0.0;
    double mte1Cycles = 0.0;
    double fixpipeCycles = 0.0;
    double scalarCycles = 0.0;
    double fillDrainCycles = 0.0;
    double launchCycles = 0.0;
    double tailPenalty = 0.0;
    double balancePenalty = 0.0;
    double splitKPenalty = 0.0;
    double tailEfficiency = 0.0;
    double coreUtilization = 0.0;
    double arithmeticIntensity = 0.0;
    double estimatedGmBytes = 0.0;
    double estimatedAGmBytes = 0.0;
    double estimatedBGmBytes = 0.0;
    double estimatedCGmBytes = 0.0;
    double estimatedMte1Bytes = 0.0;
    double l1CacheHitRate = 0.0;
    double estimatedMmadCount = 0.0;
    double estimatedOutputTileCount = 0.0;
};

struct Evaluation {
    Workload workload;
    Candidate candidate;
    bool valid = false;
    std::string error;
    std::string source;
    int32_t sourceIteration = -1;
    TCubeTiling tiling{};
    ProxyBreakdown proxy;
    std::string tilingSignature;
    ExecutionMode executionMode = ExecutionMode::NONE;
    int64_t officialReturn = -1;
    int32_t officialCoreNum = 0;
    int32_t officialMDim = 0;
    int32_t officialNDim = 0;
};

struct SearchOptions {
    int32_t beamWidth = 64;
    int32_t tabuIterations = 48;
    int32_t lnsRounds = 8;
    int32_t topK = 20;
    int32_t seedCount = 6;
    // Zero means unlimited. This is a search-budget knob, not a legality rule.
    int32_t maxCoreRounds = 0;
    int32_t maxBaseM = 512;
    int32_t maxBaseN = 512;
    int32_t maxBaseK = 1024;
};

struct PlatformCaps {
    int32_t coreNum = 24;
    uint64_t l0aBytes = 64 * 1024;
    uint64_t l0bBytes = 64 * 1024;
    uint64_t l0cBytes = 128 * 1024;
    uint64_t l1Bytes = 1024 * 1024;
    uint64_t ubBytes = 192 * 1024;
    uint64_t l2Bytes = 192 * 1024 * 1024;
    uint64_t l2BytesPerCycle = 0;
    uint64_t hbmBytesPerCycle = 0;
};

std::string DTypeToString(DType dtype);
std::string ExecutionModeToString(ExecutionMode mode);
bool ParseDType(const std::string &text, DType &dtype);
int32_t DTypeBits(DType dtype);
int32_t DTypeBytes(DType dtype);
int32_t OutputBytes(DType dtype);
int32_t AccumulatorBytes(DType dtype);
std::string TilingSignature(const TCubeTiling &tiling);

}  // namespace matmul_search
