#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "mat_mul_v3_tiling_data.h"

using namespace AscendC;
using namespace matmul;

// CANN 8.5's shared header declares multi-batch configurations that are not
// present in the installed CANN 8.1 compiler. The BASE kernel only needs this
// subset; the official block and kernel implementations remain unmodified.
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
    if (sizeof(T) == sizeof(float)) {
        c0Size = 8;
    } else if (sizeof(T) == sizeof(int8_t)) {
        c0Size = 32;
    } else {
        c0Size = 16;
    }
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

template <class A_T, class B_T, class C_T, class BiasT>
__aicore__ inline void SetL2CacheEnable(
    const L2cacheUseInfo &l2EnableInfo,
    GlobalTensor<A_T> &aGlobal,
    GlobalTensor<B_T> &bGlobal,
    GlobalTensor<C_T> &cGlobal,
    GlobalTensor<BiasT> &biasGlobal)
{
    (void)aGlobal;
    (void)bGlobal;
    (void)biasGlobal;
    if ((l2EnableInfo.l2CacheFlag & ALL_L2_CACHE_ENABLE) == 0 &&
        (l2EnableInfo.l2CacheFlag & C_L2_DISABLE) != 0) {
        cGlobal.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }
}

#include "mat_mul_base_kernel.h"

extern "C" __global__ __aicore__ void mat_mul_v3_base_fixed(
    GM_ADDR aGM,
    GM_ADDR bGM,
    GM_ADDR biasGM,
    GM_ADDR offsetWGM,
    GM_ADDR cGM,
    GM_ADDR workspaceGM,
    GM_ADDR tilingGM)
{
    __gm__ uint8_t* user = GetUserWorkspace(workspaceGM);
    MatmulTilingData tilingData;
    uint32_t *localTiling = reinterpret_cast<uint32_t *>(&tilingData);
    const __gm__ uint32_t *globalTiling =
        reinterpret_cast<const __gm__ uint32_t *>(tilingGM);
    for (uint32_t i = 0; i < sizeof(MatmulTilingData) / sizeof(uint32_t); ++i) {
        localTiling[i] = globalTiling[i];
    }

    using aType =
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, false>;
    using bType =
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, false>;
    using cType =
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_Y>;
    using biasType =
        MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;

    TPipe pipe;
    MatmulBaseKernel<
        aType,
        bType,
        cType,
        biasType,
        MatmulBaseBlock,
        MM_CFG_NO_PRELOAD>
        op;
    op.Init(
        aGM,
        bGM,
        cGM,
        biasGM,
        offsetWGM,
        user,
        &tilingData,
        &pipe);
    op.Process();
}
