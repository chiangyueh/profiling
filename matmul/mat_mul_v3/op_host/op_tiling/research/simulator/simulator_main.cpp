#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclrtlaunch_mat_mul_v3_base_fixed.h"

namespace {

constexpr uint32_t kBlockDim = 1;
constexpr size_t kM = 32;
constexpr size_t kN = 32;
constexpr size_t kK = 128;
constexpr size_t kWorkspaceBytes = 20U * 1024U * 1024U;

void Check(aclError error, const char* operation)
{
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed, rc=" + std::to_string(error));
    }
}

std::vector<uint8_t> ReadBinary(const std::string& path)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("cannot open tiling blob: " + path);
    }
    const auto size = input.tellg();
    if (size <= 0) {
        throw std::runtime_error("tiling blob is empty: " + path);
    }
    std::vector<uint8_t> data(static_cast<size_t>(size));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(data.data()), size);
    if (!input) {
        throw std::runtime_error("cannot read tiling blob: " + path);
    }
    return data;
}

void WriteBinary(const std::string& path, const void* data, size_t size)
{
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("cannot create output: " + path);
    }
    output.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    if (!output) {
        throw std::runtime_error("cannot write output: " + path);
    }
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "usage: " << argv[0] << " TILING_BIN OUTPUT_BIN\n";
        return 2;
    }

    try {
        const std::vector<uint8_t> tiling = ReadBinary(argv[1]);
        const size_t aBytes = kM * kK * sizeof(uint16_t);
        const size_t bBytes = kK * kN * sizeof(uint16_t);
        const size_t cBytes = kM * kN * sizeof(uint16_t);
        std::vector<uint16_t> a(kM * kK, 0x3c00);
        std::vector<uint16_t> b(kK * kN, 0x3c00);
        std::vector<uint16_t> c(kM * kN, 0x5a5a);

        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(0), "aclrtSetDevice");
        aclrtContext context = nullptr;
        aclrtStream stream = nullptr;
        Check(aclrtCreateContext(&context, 0), "aclrtCreateContext");
        Check(aclrtCreateStream(&stream), "aclrtCreateStream");

        void* aDevice = nullptr;
        void* bDevice = nullptr;
        void* cDevice = nullptr;
        void* workspaceDevice = nullptr;
        void* tilingDevice = nullptr;
        Check(aclrtMalloc(&aDevice, aBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(a)");
        Check(aclrtMalloc(&bDevice, bBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(b)");
        Check(aclrtMalloc(&cDevice, cBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(c)");
        Check(
            aclrtMalloc(
                &workspaceDevice,
                kWorkspaceBytes,
                ACL_MEM_MALLOC_HUGE_FIRST),
            "aclrtMalloc(workspace)");
        Check(
            aclrtMalloc(
                &tilingDevice,
                tiling.size(),
                ACL_MEM_MALLOC_HUGE_FIRST),
            "aclrtMalloc(tiling)");

        Check(
            aclrtMemcpy(
                aDevice,
                aBytes,
                a.data(),
                aBytes,
                ACL_MEMCPY_HOST_TO_DEVICE),
            "aclrtMemcpy(a)");
        Check(
            aclrtMemcpy(
                bDevice,
                bBytes,
                b.data(),
                bBytes,
                ACL_MEMCPY_HOST_TO_DEVICE),
            "aclrtMemcpy(b)");
        Check(
            aclrtMemcpy(
                cDevice,
                cBytes,
                c.data(),
                cBytes,
                ACL_MEMCPY_HOST_TO_DEVICE),
            "aclrtMemcpy(c)");
        Check(
            aclrtMemcpy(
                tilingDevice,
                tiling.size(),
                tiling.data(),
                tiling.size(),
                ACL_MEMCPY_HOST_TO_DEVICE),
            "aclrtMemcpy(tiling)");

        ACLRT_LAUNCH_KERNEL(mat_mul_v3_base_fixed)(
            kBlockDim,
            stream,
            aDevice,
            bDevice,
            nullptr,
            nullptr,
            cDevice,
            workspaceDevice,
            tilingDevice);
        Check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
        Check(
            aclrtMemcpy(
                c.data(),
                cBytes,
                cDevice,
                cBytes,
                ACL_MEMCPY_DEVICE_TO_HOST),
            "aclrtMemcpy(output)");
        WriteBinary(argv[2], c.data(), cBytes);
        std::cout << "SIMULATOR_KERNEL_RESULT\n";
        std::cout << "  shape=32x32x128 dtype=fp16 block_dim=1\n";
        std::cout << "  tiling_bytes=" << tiling.size() << "\n";
        std::cout << "  output_bytes=" << cBytes << "\n";
        std::cout << "  output=" << argv[2] << "\n";

        Check(aclrtFree(tilingDevice), "aclrtFree(tiling)");
        Check(aclrtFree(workspaceDevice), "aclrtFree(workspace)");
        Check(aclrtFree(cDevice), "aclrtFree(c)");
        Check(aclrtFree(bDevice), "aclrtFree(b)");
        Check(aclrtFree(aDevice), "aclrtFree(a)");
        Check(aclrtDestroyStream(stream), "aclrtDestroyStream");
        Check(aclrtDestroyContext(context), "aclrtDestroyContext");
        Check(aclrtResetDevice(0), "aclrtResetDevice");
        Check(aclFinalize(), "aclFinalize");
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "fatal: " << exception.what() << "\n";
        return 1;
    }
}
