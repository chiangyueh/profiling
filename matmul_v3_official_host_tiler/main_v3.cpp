#include <cstdlib>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_matmul.h"
#include "data_utils.h"

#define CHECK_ACL_RETURN(expr)                                                                     \
    do {                                                                                            \
        const aclError ret = (expr);                                                               \
        if (ret != ACL_SUCCESS) {                                                                  \
            std::cerr << #expr << " failed, rc=" << ret << std::endl;                            \
            return 1;                                                                              \
        }                                                                                           \
    } while (0)

static int32_t EnvInt(const char *name, int32_t defaultValue)
{
    const char *value = std::getenv(name);
    return value != nullptr && *value != '\0' ? std::atoi(value) : defaultValue;
}

static aclTensor *CreateNdTensor(const std::vector<int64_t> &shape, aclDataType dtype, void *deviceAddress)
{
    return aclCreateTensor(shape.data(), shape.size(), dtype, nullptr, 0, ACL_FORMAT_ND,
                           shape.data(), shape.size(), deviceAddress);
}

int main()
{
    const int32_t m = EnvInt("MM_M", 512);
    const int32_t n = EnvInt("MM_N", 512);
    const int32_t k = EnvInt("MM_K", 512);

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

    CHECK_ACL_RETURN(aclInit(nullptr));
    CHECK_ACL_RETURN(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL_RETURN(aclrtCreateStream(&stream));

    void *aDevice = nullptr;
    void *bDevice = nullptr;
    void *cDevice = nullptr;
    CHECK_ACL_RETURN(aclrtMalloc(&aDevice, aSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL_RETURN(aclrtMalloc(&bDevice, bSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL_RETURN(aclrtMalloc(&cDevice, cSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL_RETURN(aclrtMemcpy(aDevice, aSize, aHost.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL_RETURN(aclrtMemcpy(bDevice, bSize, bHost.data(), bSize, ACL_MEMCPY_HOST_TO_DEVICE));

    const std::vector<int64_t> aShape = {m, k};
    const std::vector<int64_t> bShape = {k, n};
    const std::vector<int64_t> cShape = {m, n};
    aclTensor *aTensor = CreateNdTensor(aShape, ACL_FLOAT16, aDevice);
    aclTensor *bTensor = CreateNdTensor(bShape, ACL_FLOAT16, bDevice);
    aclTensor *cTensor = CreateNdTensor(cShape, ACL_FLOAT16, cDevice);
    if (aTensor == nullptr || bTensor == nullptr || cTensor == nullptr) {
        std::cerr << "aclCreateTensor failed" << std::endl;
        return 1;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    constexpr int8_t cubeMathType = 1;
    CHECK_ACL_RETURN(
        aclnnMatmulGetWorkspaceSize(aTensor, bTensor, cTensor, cubeMathType, &workspaceSize, &executor));

    void *workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL_RETURN(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    CHECK_ACL_RETURN(aclnnMatmul(workspace, workspaceSize, executor, stream));
    CHECK_ACL_RETURN(aclrtSynchronizeStream(stream));
    CHECK_ACL_RETURN(aclrtMemcpy(cHost.data(), cSize, cDevice, cSize, ACL_MEMCPY_DEVICE_TO_HOST));
    if (!WriteFile("./output/output.bin", cHost.data(), cSize)) {
        return 1;
    }

    aclDestroyTensor(aTensor);
    aclDestroyTensor(bTensor);
    aclDestroyTensor(cTensor);
    if (workspace != nullptr) {
        CHECK_ACL_RETURN(aclrtFree(workspace));
    }
    CHECK_ACL_RETURN(aclrtFree(aDevice));
    CHECK_ACL_RETURN(aclrtFree(bDevice));
    CHECK_ACL_RETURN(aclrtFree(cDevice));
    CHECK_ACL_RETURN(aclrtDestroyStream(stream));
    CHECK_ACL_RETURN(aclFinalize());
    return 0;
}
