#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "lib/matmul/bmm_tiling.h"
#include "platform/platform_info.h"

namespace tuningtiling {
class TuningTilingDef;
}

namespace RuntimeKb {
uint32_t CommonHash(const void *input, uint32_t size, uint32_t seed);
bool InitBank(fe::PlatFormInfos *platform);
uint32_t QueryBank(
    const void *input, size_t size, const std::string &opType,
    std::shared_ptr<tuningtiling::TuningTilingDef> &tiling);
}

namespace {

constexpr uint32_t kRuntimeBankHashSeed = 271828;
constexpr size_t kMatMulV3InputBytes = 183;

std::vector<uint8_t> ReadBinary(const std::string &path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open input key: " + path);
    }
    return std::vector<uint8_t>(
        std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

void PrintUsage()
{
    std::cerr
        << "Usage:\n"
        << "  tiling_bank_probe --platform SOC\n"
        << "  tiling_bank_probe --hash INPUT_BIN\n"
        << "  tiling_bank_probe --query INPUT_BIN SOC AIC_CORES\n";
}

uint64_t ReadSize(
    platform_ascendc::PlatformAscendC *platform,
    platform_ascendc::CoreMemType type)
{
    uint64_t value = 0;
    platform->GetCoreMemSize(type, value);
    return value;
}

}  // namespace

int main(int argc, char **argv)
{
    try {
        if (argc == 3 && std::string(argv[1]) == "--platform") {
            auto *platform =
                platform_ascendc::PlatformAscendCManager::GetInstance(argv[2]);
            if (platform == nullptr) {
                throw std::runtime_error(
                    "PlatformAscendCManager returned null for " +
                    std::string(argv[2]));
            }
            uint64_t l2Bw = 0;
            uint64_t hbmBw = 0;
            platform->GetCoreMemBw(
                platform_ascendc::CoreMemType::L2, l2Bw);
            platform->GetCoreMemBw(
                platform_ascendc::CoreMemType::HBM, hbmBw);
            std::cout
                << "soc=" << argv[2]
                << " aic=" << platform->GetCoreNumAic()
                << " L0A=" << ReadSize(
                    platform, platform_ascendc::CoreMemType::L0_A)
                << " L0B=" << ReadSize(
                    platform, platform_ascendc::CoreMemType::L0_B)
                << " L0C=" << ReadSize(
                    platform, platform_ascendc::CoreMemType::L0_C)
                << " L1=" << ReadSize(
                    platform, platform_ascendc::CoreMemType::L1)
                << " L2=" << ReadSize(
                    platform, platform_ascendc::CoreMemType::L2)
                << " L2_Bpc_per_core=" << l2Bw
                << " HBM_Bpc_per_core=" << hbmBw
                << '\n';
            return 0;
        }
        if (argc < 3) {
            PrintUsage();
            return 2;
        }
        const std::string mode = argv[1];
        const auto key = ReadBinary(argv[2]);
        if (key.size() != kMatMulV3InputBytes) {
            throw std::runtime_error(
                "MatMulV3 input key must be 183 bytes, got " +
                std::to_string(key.size()));
        }
        const uint32_t hash = RuntimeKb::CommonHash(
            key.data(), static_cast<uint32_t>(key.size()), kRuntimeBankHashSeed);
        if (mode == "--hash") {
            if (argc != 3) {
                PrintUsage();
                return 2;
            }
            std::cout << hash << '\n';
            return 0;
        }
        if (mode != "--query" || argc != 5) {
            PrintUsage();
            return 2;
        }

        const std::string soc = argv[3];
        const uint32_t cores = static_cast<uint32_t>(std::stoul(argv[4]));
        std::shared_ptr<tuningtiling::TuningTilingDef> tiling;
        fe::PlatFormInfos platform;
        fe::OptionalInfos optional;
        auto &platformManager = fe::PlatformInfoManager::Instance();
        const uint32_t initializeRc =
            platformManager.InitializePlatformInfo();
        if (initializeRc != 0) {
            std::cerr << "fatal: InitializePlatformInfo failed"
                      << " rc=" << initializeRc << '\n';
            return 4;
        }
        const uint32_t platformRc =
            platformManager.GetPlatformInfos(soc, platform, optional);
        if (platformRc != 0) {
            std::cerr << "fatal: GetPlatformInfos failed"
                      << " soc=" << soc
                      << " rc=" << platformRc << '\n';
            return 4;
        }
        platform.SetCoreNum(cores);
        if (!RuntimeKb::InitBank(&platform)) {
            std::cerr << "fatal: RuntimeKb::InitBank failed"
                      << " soc=" << soc
                      << " cores=" << cores << '\n';
            return 4;
        }
        const uint32_t rc = RuntimeKb::QueryBank(
            key.data(), key.size(), "MatMulV3", tiling);

        const bool found = rc == 0 && tiling != nullptr;
        std::cout << "hash=" << hash << " query_api=platform_initialized"
                  << " query_rc=" << rc
                  << " found=" << (found ? 1 : 0) << '\n';
        return found ? 0 : 3;
    } catch (const std::exception &exception) {
        std::cerr << "fatal: " << exception.what() << '\n';
        return 1;
    }
}
