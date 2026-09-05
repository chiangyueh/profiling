#include "kernel_operator.h"
#include "mat_mul_v3_tiling_data.h"

__aicore__ inline void DirectReadMatmulTiling(
    GM_ADDR tilingGM, MatmulTilingData &tiling)
{
    uint32_t *local = reinterpret_cast<uint32_t *>(&tiling);
    const __gm__ uint32_t *global =
        reinterpret_cast<const __gm__ uint32_t *>(tilingGM);
    for (uint32_t index = 0;
         index < sizeof(MatmulTilingData) / sizeof(uint32_t); ++index) {
        local[index] = global[index];
    }
}

// The stock macro depends on framework-generated tiling classes.  The direct
// runner deliberately supplies the complete CANN 8.1 POD itself.
#undef GET_TILING_DATA
#define GET_TILING_DATA(variable, address) \
    MatmulTilingData variable; DirectReadMatmulTiling(address, variable)

#undef TILING_KEY_VAR
#define TILING_KEY_VAR MATMUL_DIRECT_TILING_KEY
#define mat_mul_v3 MATMUL_DIRECT_KERNEL

// ascendc_library's generated static wrapper has already called
// AscendC::GetUserWorkspace before entering MATMUL_DIRECT_KERNEL.  The
// original dynamic MatMulV3 entry expects the untouched framework workspace
// and performs that conversion itself.  Keeping both conversions skips the
// reserved system workspace twice; workspace-backed kernels (notably suffix
// 20/21/30/31) then access beyond the allocation and report 507053.  At this
// direct ABI boundary the incoming pointer is already the user pointer.
#define GetUserWorkspace(address) (address)
#include "mat_mul_v3.cpp"
#undef GetUserWorkspace
