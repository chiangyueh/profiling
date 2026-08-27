#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclrtlaunch_matmul_validator_probe_base_fp16.h"
#include "mat_mul_v3_tiling_data.h"

namespace {

constexpr int32_t kM = 128;
constexpr int32_t kN = 128;
constexpr int32_t kK = 128;
constexpr int32_t kInvalidSingleCoreK = 64;
constexpr size_t kWorkspaceBytes = 20U * 1024U * 1024U;

void Check(aclError error, const char *operation)
{
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed, rc=" + std::to_string(error));
    }
}

struct DeviceState {
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    void *a = nullptr;
    void *b = nullptr;
    void *c = nullptr;
    void *workspace = nullptr;
    void *tiling = nullptr;
    bool initialized = false;

    ~DeviceState()
    {
        if (tiling != nullptr) {
            (void)aclrtFree(tiling);
        }
        if (workspace != nullptr) {
            (void)aclrtFree(workspace);
        }
        if (c != nullptr) {
            (void)aclrtFree(c);
        }
        if (b != nullptr) {
            (void)aclrtFree(b);
        }
        if (a != nullptr) {
            (void)aclrtFree(a);
        }
        if (stream != nullptr) {
            (void)aclrtDestroyStream(stream);
        }
        if (context != nullptr) {
            (void)aclrtDestroyContext(context);
        }
        if (initialized) {
            (void)aclFinalize();
        }
    }
};

MatmulTilingData MakeDeliberatelyInvalidTiling()
{
    MatmulTilingData data{};
    auto &t = data.matmulTiling;
    t.usedCoreNum = 1;
    t.M = kM;
    t.N = kN;
    t.Ka = kK;
    t.Kb = kK;
    t.singleCoreM = kM;
    t.singleCoreN = kN;
    t.singleCoreK = kInvalidSingleCoreK;
    t.baseM = kM;
    t.baseN = kN;
    t.baseK = 64;
    t.depthA1 = 2;
    t.depthB1 = 2;
    t.stepM = 1;
    t.stepN = 1;
    t.stepKa = 1;
    t.stepKb = 1;
    t.dbL0A = 2;
    t.dbL0B = 2;
    t.dbL0C = 1;

    data.tileL2cacheTiling.mTileCntL2 = 1;
    data.tileL2cacheTiling.nTileCntL2 = 1;
    data.tileL2cacheTiling.mTileBlock = 1;
    data.tileL2cacheTiling.nTileBlock = 1;
    data.tileL2cacheTiling.calOrder = 0;
    return data;
}

bool ValidateImmediatelyBeforeLaunch(
    const MatmulTilingData &data, std::string &reason)
{
    const auto &t = data.matmulTiling;
    if (t.singleCoreK != t.Ka || t.singleCoreK != t.Kb) {
        reason = "BASE_SINGLE_CORE_K_MUST_EQUAL_K";
        return false;
    }
    return true;
}

}  // namespace

int main()
{
    try {
        const MatmulTilingData tiling = MakeDeliberatelyInvalidTiling();
        const size_t aBytes = static_cast<size_t>(kM) * kK * sizeof(aclFloat16);
        const size_t bBytes = static_cast<size_t>(kK) * kN * sizeof(aclFloat16);
        const size_t cElements = static_cast<size_t>(kM) * kN;
        const size_t cBytes = cElements * sizeof(aclFloat16);
        const aclFloat16 one = aclFloatToFloat16(1.0F);
        std::vector<aclFloat16> a(static_cast<size_t>(kM) * kK, one);
        std::vector<aclFloat16> b(static_cast<size_t>(kK) * kN, one);
        std::vector<aclFloat16> c(cElements, aclFloatToFloat16(0.0F));

        DeviceState state;
        Check(aclInit(nullptr), "aclInit");
        state.initialized = true;
        Check(aclrtSetDevice(0), "aclrtSetDevice");
        Check(aclrtCreateContext(&state.context, 0), "aclrtCreateContext");
        Check(aclrtCreateStream(&state.stream), "aclrtCreateStream");
        Check(aclrtMalloc(&state.a, aBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc A");
        Check(aclrtMalloc(&state.b, bBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc B");
        Check(aclrtMalloc(&state.c, cBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc C");
        Check(aclrtMalloc(&state.workspace, kWorkspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc workspace");
        Check(aclrtMalloc(&state.tiling, sizeof(tiling), ACL_MEM_MALLOC_HUGE_FIRST), "malloc tiling");
        Check(aclrtMemcpy(state.a, aBytes, a.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy A");
        Check(aclrtMemcpy(state.b, bBytes, b.data(), bBytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy B");
        Check(aclrtMemcpy(
            state.tiling,
            sizeof(tiling),
            &tiling,
            sizeof(tiling),
            ACL_MEMCPY_HOST_TO_DEVICE),
            "copy tiling");

        // First run: no validator.  The original BASE kernel receives K=128
        // together with singleCoreK=64 and therefore computes only half of K.
        Check(
            ACLRT_LAUNCH_KERNEL(matmul_validator_probe_base_fp16)(
                1,
                state.stream,
                state.a,
                state.b,
                nullptr,
                nullptr,
                state.c,
                state.workspace,
                state.tiling),
            "unchecked kernel launch");
        Check(aclrtSynchronizeStream(state.stream), "unchecked kernel synchronize");
        Check(aclrtMemcpy(c.data(), cBytes, state.c, cBytes, ACL_MEMCPY_DEVICE_TO_HOST), "copy C");

        const float expected = static_cast<float>(kK);
        size_t mismatchCount = 0;
        float first = aclFloat16ToFloat(c.front());
        float minimum = first;
        float maximum = first;
        for (aclFloat16 value : c) {
            const float actual = aclFloat16ToFloat(value);
            minimum = std::min(minimum, actual);
            maximum = std::max(maximum, actual);
            if (std::fabs(actual - expected) > 0.5F) {
                ++mismatchCount;
            }
        }
        if (mismatchCount == 0) {
            throw std::runtime_error("the deliberately invalid tiling did not reproduce a wrong output");
        }
        std::cout << "MATMUL_VALIDATOR_PROBE_UNCHECKED status=wrong_output"
                  << " shape=" << kM << "x" << kN << "x" << kK
                  << " K=" << kK
                  << " singleCoreK=" << kInvalidSingleCoreK
                  << " expected=" << expected
                  << " observed_first=" << first
                  << " observed_min=" << minimum
                  << " observed_max=" << maximum
                  << " mismatches=" << mismatchCount << "/" << cElements << '\n';

        // Second run: the exact same tiling reaches one check immediately
        // before launch.  Rejection means no second kernel is submitted.
        std::string rejection;
        if (ValidateImmediatelyBeforeLaunch(tiling, rejection)) {
            throw std::runtime_error("validator unexpectedly accepted the deliberately invalid tiling");
        }
        std::cout << "MATMUL_VALIDATOR_PROBE_CHECKED status=rejected"
                  << " reason=" << rejection
                  << " K=" << kK
                  << " singleCoreK=" << kInvalidSingleCoreK
                  << " kernel_launched=0\n";
        std::cout << "MATMUL_VALIDATOR_PROBE_RESULT status=passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "MATMUL_VALIDATOR_PROBE_RESULT status=failed error="
                  << error.what() << '\n';
        return 1;
    }
}
