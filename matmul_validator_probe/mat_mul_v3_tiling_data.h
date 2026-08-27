#ifndef MATMUL_VALIDATOR_PROBE_TILING_DATA_H
#define MATMUL_VALIDATOR_PROBE_TILING_DATA_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

#pragma pack(push, 8)
struct alignas(8) L2cacheUseInfo {
    uint32_t l2CacheFlag;
};

struct alignas(8) L2cacheTilePara {
    uint32_t mTileCntL2;
    uint32_t nTileCntL2;
    uint32_t mTileBlock;
    uint32_t nTileBlock;
    uint32_t calOrder;
};

struct alignas(8) MatMulRunInfo {
    uint32_t transA;
    uint32_t transB;
    uint32_t nd2nzA;
    uint32_t nd2nzB;
    uint32_t isNzA;
    uint32_t isNzB;
    uint32_t isHf32;
};

struct alignas(8) MatmulTilingData {
    TCubeTiling matmulTiling;
    L2cacheTilePara tileL2cacheTiling;
    MatMulRunInfo matmulRunInfo;
    L2cacheUseInfo l2cacheUseInfo;
    uint32_t baseAN;
    uint32_t baseAD;
    uint32_t baseBN;
    uint32_t baseBD;
};
#pragma pack(pop)

#endif
