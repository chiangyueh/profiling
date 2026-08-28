// --- BEGIN: colleague standalone path retained verbatim for the 4.log comparison.
// This path hand-fills MatmulTilingData and launches MM_KERNEL=0 directly.
#if !defined(MM_USE_OFFICIAL_HOST_TILER) || MM_USE_OFFICIAL_HOST_TILER == 0
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include "data_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mat_mul_v3_tiling_data.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_matmul_v3_custom.h"
#else
#include "tikicpulib.h"
extern "C" void matmul_v3_custom(uint8_t *a, uint8_t *b, uint8_t *bias,
                                 uint8_t *offsetW, uint8_t *c,
                                 uint8_t *workspace, uint8_t *tiling);
#endif

#ifndef MM_KERNEL
#define MM_KERNEL 0
#endif

static int32_t EnvI(const char *name, int32_t def)
{
    const char *v = getenv(name);
    return (v && *v) ? atoi(v) : def;
}
static uint32_t CeilDiv(uint32_t a, uint32_t b) { return b ? (a + b - 1) / b : a; }
static uint32_t CeilAlign(uint32_t a, uint32_t b) { return b ? (a + b - 1) / b * b : a; }
static uint32_t Min(uint32_t a, uint32_t b) { return a < b ? a : b; }

int32_t main(int32_t argc, char *argv[])
{
    const int32_t M = EnvI("MM_M", 512);
    const int32_t N = EnvI("MM_N", 512);
    const int32_t K = EnvI("MM_K", 512);
    const int32_t baseM = EnvI("MM_BASE_M", 128);
    const int32_t baseN = EnvI("MM_BASE_N", 256);
    const int32_t baseK = EnvI("MM_BASE_K", 64);
    const int32_t singleCoreM = EnvI("MM_SINGLE_M", 128);
    const int32_t singleCoreN = EnvI("MM_SINGLE_N", 256);
    const int32_t stepM  = EnvI("MM_STEP_M", 1);
    const int32_t stepN  = EnvI("MM_STEP_N", 1);
    const int32_t stepKa = EnvI("MM_STEP_Ka", 4);
    const int32_t stepKb = EnvI("MM_STEP_Kb", 4);
    const int32_t dbL0A = EnvI("MM_DB_L0A", 2);
    const int32_t dbL0B = EnvI("MM_DB_L0B", 2);
    const int32_t dbL0C = EnvI("MM_DB_L0C", 2);
    const int32_t iterOrder = EnvI("MM_ITER_ORDER", 0);

    const int32_t mmKernel = MM_KERNEL;
    std::cout << "mmKernel: " << mmKernel << std::endl;
    const bool isSplitK   = (mmKernel == 2 || mmKernel == 21 || mmKernel == 3 || mmKernel == 4);
    const bool isMultiK   = (mmKernel == 3 || mmKernel == 4);
    const int32_t singleCoreK = isSplitK ? (stepKa * baseK) : K;

    const uint32_t mCnt = CeilDiv(M, singleCoreM);
    const uint32_t nCnt = CeilDiv(N, singleCoreN);
    const uint32_t kCnt = CeilDiv(K, singleCoreK);

    uint32_t usedCoreNum = isMultiK ? (mCnt * nCnt * kCnt) : (mCnt * nCnt);
    usedCoreNum = Min(usedCoreNum, 24u);

    size_t aSize = static_cast<size_t>(M) * K * sizeof(uint16_t);
    size_t bSize = static_cast<size_t>(K) * N * sizeof(uint16_t);
    size_t cSize = static_cast<size_t>(M) * N * sizeof(float);
    size_t tilingSize = sizeof(MatmulTilingData);

    // --- workspace: rpc(20MB) + буфер частичных сумм для split-K ---
    const size_t RPC = 20ull * 1024 * 1024;
    const uint32_t align256 = 256 / 2; // 256B / dtype(fp16=2) = 128
    size_t sysWorkspaceSize;
    if (mmKernel == 2 || mmKernel == 21) {
        sysWorkspaceSize = (size_t)M * CeilAlign(N, align256) * 4 + RPC;
    } else if (isMultiK) {
        sysWorkspaceSize = (size_t)usedCoreNum * singleCoreM * singleCoreN * 2 * 4 + RPC;
    } else {
        sysWorkspaceSize = RPC;
    }

    uint8_t *tilingHost = (uint8_t *)malloc(tilingSize);
    memset(tilingHost, 0, tilingSize);
    MatmulTilingData *t = reinterpret_cast<MatmulTilingData *>(tilingHost);
    t->matmulTiling.usedCoreNum = usedCoreNum;
    t->matmulTiling.M  = M;
    t->matmulTiling.N  = N;
    t->matmulTiling.Ka = K;
    t->matmulTiling.Kb = K;
    t->matmulTiling.singleCoreM = singleCoreM;
    t->matmulTiling.singleCoreN = singleCoreN;
    t->matmulTiling.singleCoreK = singleCoreK;
    t->matmulTiling.baseM = baseM;
    t->matmulTiling.baseN = baseN;
    t->matmulTiling.baseK = baseK;
    t->matmulTiling.depthA1 = stepM * stepKa * 2;
    t->matmulTiling.depthB1 = stepN * stepKb * 2;
    t->matmulTiling.stepM  = stepM;
    t->matmulTiling.stepN  = stepN;
    t->matmulTiling.stepKa = stepKa;
    t->matmulTiling.stepKb = stepKb;
    t->matmulTiling.isBias = 0;
    t->matmulTiling.transLength  = 0;
    t->matmulTiling.iterateOrder = iterOrder;
    t->matmulTiling.depthAL1CacheUB = 0;
    t->matmulTiling.depthBL1CacheUB = 0;
    t->matmulTiling.dbL0A = dbL0A;
    t->matmulTiling.dbL0B = dbL0B;
    t->matmulTiling.dbL0C = dbL0C;

    t->matmulRunInfo.transA = 0;
    t->matmulRunInfo.transB = 0;
    t->matmulRunInfo.nd2nzA = 0;
    t->matmulRunInfo.nd2nzB = 0;
    t->matmulRunInfo.isHf32 = 0;
    t->matmulRunInfo.isNzA  = 0;
    t->matmulRunInfo.isNzB  = 0;

    t->l2cacheUseInfo.l2CacheFlag   = 0;
    t->tileL2cacheTiling.mTileCntL2 = 1;
    t->tileL2cacheTiling.nTileCntL2 = 1;
    t->tileL2cacheTiling.mTileBlock = 1;
    t->tileL2cacheTiling.nTileBlock = 1;
    t->tileL2cacheTiling.calOrder   = 0;

    uint32_t blockDim = t->matmulTiling.usedCoreNum;

#ifdef ASCENDC_CPU_DEBUG
    uint8_t *a = (uint8_t *)AscendC::GmAlloc(aSize);
    uint8_t *b = (uint8_t *)AscendC::GmAlloc(bSize);
    uint8_t *c = (uint8_t *)AscendC::GmAlloc(cSize);
    uint8_t *ws = (uint8_t *)AscendC::GmAlloc(sysWorkspaceSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    uint8_t *offsetW = (uint8_t *)AscendC::GmAlloc(1024);

    memset(a, 0, aSize);
    memset(b, 0, bSize);
    ReadFile("./input/x1_gm.bin", aSize, a, aSize);
    ReadFile("./input/x2_gm.bin", bSize, b, bSize);
    memcpy(tiling, tilingHost, tilingSize);

    ICPU_RUN_KF(matmul_v3_custom, blockDim, a, b, nullptr, offsetW, c, ws, tiling);

    WriteFile("./output/output.bin", c, cSize);
#else
CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *aHost = nullptr, *bHost = nullptr, *cHost = nullptr;
    uint8_t *aDev = nullptr, *bDev = nullptr, *cDev = nullptr;
    uint8_t *wsDev = nullptr, *tilingDev = nullptr, *offsetWDev = nullptr;

    CHECK_ACL(aclrtMallocHost((void **)&aHost, aSize));
    CHECK_ACL(aclrtMallocHost((void **)&bHost, bSize));
    CHECK_ACL(aclrtMallocHost((void **)&cHost, cSize));
    CHECK_ACL(aclrtMalloc((void **)&aDev, aSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&bDev, bSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&cDev, cSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&wsDev, sysWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&tilingDev, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&offsetWDev, 1024, ACL_MEM_MALLOC_HUGE_FIRST));

    memset(aHost, 0, aSize);
    memset(bHost, 0, bSize);
    ReadFile("./input/x1_gm.bin", aSize, aHost, aSize);
    ReadFile("./input/x2_gm.bin", bSize, bHost, bSize);

    CHECK_ACL(aclrtMemcpy(aDev, aSize, aHost, aSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(bDev, bSize, bHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(tilingDev, tilingSize, tilingHost, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));

    ACLRT_LAUNCH_KERNEL(matmul_v3_custom)
        (blockDim, stream, aDev, bDev, nullptr, offsetWDev, cDev, wsDev, tilingDev);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(cHost, cSize, cDev, cSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output.bin", cHost, cSize);
#endif

    std::cout << "shape M=" << M << " N=" << N << " K=" << K << " MM_KERNEL=" << mmKernel << std::endl;
    std::cout << "usedCoreNum=" << t->matmulTiling.usedCoreNum
              << " workspace_mb=" << (sysWorkspaceSize >> 20) << std::endl;
    std::cout << "baseM=" << t->matmulTiling.baseM
              << " baseN=" << t->matmulTiling.baseN
              << " baseK=" << t->matmulTiling.baseK << std::endl;
    std::cout << "singleM=" << t->matmulTiling.singleCoreM
              << " singleN=" << t->matmulTiling.singleCoreN
              << " singleK=" << t->matmulTiling.singleCoreK << std::endl;
    std::cout << "stepM=" << t->matmulTiling.stepM
              << " stepN=" << t->matmulTiling.stepN
              << " stepKa=" << t->matmulTiling.stepKa
              << " stepKb=" << t->matmulTiling.stepKb << std::endl;
#ifdef ASCENDC_CPU_DEBUG
    AscendC::GmFree((void *)a);
    AscendC::GmFree((void *)b);
    AscendC::GmFree((void *)c);
    AscendC::GmFree((void *)ws);
    AscendC::GmFree((void *)tiling);
    AscendC::GmFree((void *)offsetW);
#else

    CHECK_ACL(aclrtFree(aDev));       CHECK_ACL(aclrtFree(bDev));
    CHECK_ACL(aclrtFree(cDev));       CHECK_ACL(aclrtFree(wsDev));
    CHECK_ACL(aclrtFree(tilingDev));  CHECK_ACL(aclrtFree(offsetWDev));
    CHECK_ACL(aclrtFreeHost(aHost));  CHECK_ACL(aclrtFreeHost(bHost));
    CHECK_ACL(aclrtFreeHost(cHost));
    CHECK_ACL(aclrtDestroyStream(stream));
#ifndef MM_SHARED_SERVER_SAFE
    CHECK_ACL(aclrtResetDevice(deviceId));
#endif
    CHECK_ACL(aclFinalize());
#endif

    free(tilingHost);
    return 0;
}
// --- END: colleague standalone path retained verbatim.

#else

// +++ BEGIN: official MatMulV3 operator path used for 3.log.
// aclnnMatmulGetWorkspaceSize enters the operator framework, which creates the
// gert::TilingContext and invokes the registered MatmulV3TilingFunc.  No tiling
// field, blockDim, tiling key, or kernel family is selected in this runner.
#include <cstdlib>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "aclnn_matmul.h"
#include "data_utils.h"

#define OFFICIAL_ACL_CALL(expr)                                                                    \
    do {                                                                                            \
        const auto officialRet = (expr);                                                           \
        if (officialRet != ACL_SUCCESS) {                                                          \
            std::cerr << #expr << " failed, rc=" << officialRet << std::endl;                     \
            return 1;                                                                              \
        }                                                                                           \
    } while (0)

static int32_t OfficialEnvI(const char *name, int32_t defaultValue)
{
    const char *value = std::getenv(name);
    return value != nullptr && *value != '\0' ? std::atoi(value) : defaultValue;
}

static aclTensor *CreateOfficialNdTensor(const std::vector<int64_t> &shape, aclDataType dtype, void *deviceAddress)
{
    return aclCreateTensor(shape.data(), shape.size(), dtype, nullptr, 0, ACL_FORMAT_ND, shape.data(), shape.size(),
                           deviceAddress);
}

int main()
{
    const int32_t m = OfficialEnvI("MM_M", 512);
    const int32_t n = OfficialEnvI("MM_N", 512);
    const int32_t k = OfficialEnvI("MM_K", 512);

    const size_t aSize = static_cast<size_t>(m) * k * sizeof(uint16_t);
    const size_t bSize = static_cast<size_t>(k) * n * sizeof(uint16_t);
    const size_t cSize = static_cast<size_t>(m) * n * sizeof(uint16_t);

    std::vector<uint8_t> aHost(aSize);
    std::vector<uint8_t> bHost(bSize);
    std::vector<uint8_t> cHost(cSize);
    size_t fileSize = 0;
    if (!ReadFile("./input/x1_gm.bin", fileSize, aHost.data(), aSize) ||
        !ReadFile("./input/x2_gm.bin", fileSize, bHost.data(), bSize)) {
        return 1;
    }

    OFFICIAL_ACL_CALL(aclInit(nullptr));
    OFFICIAL_ACL_CALL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    OFFICIAL_ACL_CALL(aclrtCreateStream(&stream));

    void *aDevice = nullptr;
    void *bDevice = nullptr;
    void *cDevice = nullptr;
    OFFICIAL_ACL_CALL(aclrtMalloc(&aDevice, aSize, ACL_MEM_MALLOC_HUGE_FIRST));
    OFFICIAL_ACL_CALL(aclrtMalloc(&bDevice, bSize, ACL_MEM_MALLOC_HUGE_FIRST));
    OFFICIAL_ACL_CALL(aclrtMalloc(&cDevice, cSize, ACL_MEM_MALLOC_HUGE_FIRST));
    OFFICIAL_ACL_CALL(aclrtMemcpy(aDevice, aSize, aHost.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE));
    OFFICIAL_ACL_CALL(aclrtMemcpy(bDevice, bSize, bHost.data(), bSize, ACL_MEMCPY_HOST_TO_DEVICE));

    const std::vector<int64_t> aShape = {m, k};
    const std::vector<int64_t> bShape = {k, n};
    const std::vector<int64_t> cShape = {m, n};
    aclTensor *aTensor = CreateOfficialNdTensor(aShape, ACL_FLOAT16, aDevice);
    aclTensor *bTensor = CreateOfficialNdTensor(bShape, ACL_FLOAT16, bDevice);
    aclTensor *cTensor = CreateOfficialNdTensor(cShape, ACL_FLOAT16, cDevice);
    if (aTensor == nullptr || bTensor == nullptr || cTensor == nullptr) {
        std::cerr << "aclCreateTensor failed" << std::endl;
        return 1;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    constexpr int8_t cubeMathType = 1;
    std::cout << "OFFICIAL_HOST_TILER_BEGIN shape=" << m << "x" << n << "x" << k << std::endl;
    OFFICIAL_ACL_CALL(
        aclnnMatmulGetWorkspaceSize(aTensor, bTensor, cTensor, cubeMathType, &workspaceSize, &executor));

    void *workspace = nullptr;
    if (workspaceSize > 0) {
        OFFICIAL_ACL_CALL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    OFFICIAL_ACL_CALL(aclnnMatmul(workspace, workspaceSize, executor, stream));
    OFFICIAL_ACL_CALL(aclrtSynchronizeStream(stream));
    OFFICIAL_ACL_CALL(aclrtMemcpy(cHost.data(), cSize, cDevice, cSize, ACL_MEMCPY_DEVICE_TO_HOST));
    if (!WriteFile("./output/output.bin", cHost.data(), cSize)) {
        return 1;
    }
    std::cout << "OFFICIAL_HOST_TILER_END workspace_bytes=" << workspaceSize << std::endl;

    aclDestroyTensor(aTensor);
    aclDestroyTensor(bTensor);
    aclDestroyTensor(cTensor);
    if (workspace != nullptr) {
        OFFICIAL_ACL_CALL(aclrtFree(workspace));
    }
    OFFICIAL_ACL_CALL(aclrtFree(aDevice));
    OFFICIAL_ACL_CALL(aclrtFree(bDevice));
    OFFICIAL_ACL_CALL(aclrtFree(cDevice));
    OFFICIAL_ACL_CALL(aclrtDestroyStream(stream));
    // Shared-server safety: deliberately do not call aclrtResetDevice.
    OFFICIAL_ACL_CALL(aclFinalize());
    return 0;
}
// +++ END: official MatMulV3 operator path used for 3.log.

#endif
