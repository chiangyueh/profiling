#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "mat_mul_v3_tiling_data.h"

using namespace AscendC;
using namespace matmul;

// The installed CANN 8.1 BASE kernel includes mat_mul_v3_common.h.  Its
// definitions are supplied here so that the original kernel and block headers
// themselves remain completely unchanged.
#define __OP_KERNEL_MATMUL_V3_COMMON_H__

constexpr uint64_t ALIGNED_H = 16;
constexpr uint32_t ROW_FIRST = 1;
constexpr uint32_t COL_FIRST = 2;
constexpr uint32_t ALL_L2_CACHE_ENABLE = 1;
constexpr uint32_t C_L2_DISABLE = 16;
constexpr MatmulConfig MM_CFG_NO_PRELOAD =
    GetMDLConfig(false, false, 0, false, false, false, true);

template <class T>
__aicore__ inline void GetSizeC0(uint64_t &c0Size)
{
    c0Size = sizeof(T) == sizeof(float) ? 8 : 16;
}

__aicore__ inline uint64_t MMV3DivCeil(uint64_t a, uint64_t b)
{
    return b == 0 ? a : (a + b - 1) / b;
}

__aicore__ inline uint64_t GetCurrentBlockIdx()
{
    if ASCEND_IS_AIV {
        return GetBlockIdx() / GetTaskRation();
    }
    return GetBlockIdx();
}

__aicore__ inline uint64_t MMLcm(uint64_t m, uint64_t n)
{
    if (m == 0 || n == 0) {
        return 0;
    }
    uint64_t total = m * n;
    while (n != 0) {
        uint64_t remainder = m % n;
        m = n;
        n = remainder;
    }
    return total / m;
}

template <class A, class B, class C, class Bias>
__aicore__ inline void SetL2CacheEnable(
    const L2cacheUseInfo &info,
    GlobalTensor<A> &a,
    GlobalTensor<B> &b,
    GlobalTensor<C> &c,
    GlobalTensor<Bias> &bias)
{
    (void)a;
    (void)b;
    (void)bias;
    if ((info.l2CacheFlag & ALL_L2_CACHE_ENABLE) == 0 &&
        (info.l2CacheFlag & C_L2_DISABLE) != 0) {
        c.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }
}

// This is the unmodified installed CANN 8.1 MatMulV3 BASE implementation.
#include "mat_mul_base_kernel.h"

__aicore__ inline void ReadTiling(GM_ADDR tilingGM, MatmulTilingData &tiling)
{
    uint32_t *local = reinterpret_cast<uint32_t *>(&tiling);
    const __gm__ uint32_t *global = reinterpret_cast<const __gm__ uint32_t *>(tilingGM);
    for (uint32_t i = 0; i < sizeof(MatmulTilingData) / sizeof(uint32_t); ++i) {
        local[i] = global[i];
    }
}

extern "C" __global__ __aicore__ void matmul_validator_probe_base_fp16(
    GM_ADDR aGM,
    GM_ADDR bGM,
    GM_ADDR biasGM,
    GM_ADDR offsetWGM,
    GM_ADDR cGM,
    GM_ADDR workspaceGM,
    GM_ADDR tilingGM)
{
    MatmulTilingData tiling;
    ReadTiling(tilingGM, tiling);
    using A = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
    using B = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
    using C = MatmulType<TPosition::GM, CubeFormat::ND, half>;
    using Bias = MatmulType<TPosition::GM, CubeFormat::ND, half>;
    TPipe pipe;
    MatmulBaseKernel<A, B, C, Bias, MatmulBaseBlock, MM_CFG_NO_PRELOAD> op;
    op.Init(
        aGM,
        bGM,
        cGM,
        biasGM,
        offsetWGM,
        GetUserWorkspace(workspaceGM),
        &tiling,
        &pipe);
    op.Process();
}
