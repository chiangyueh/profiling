#include "official_tiling.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <sstream>

namespace matmul_search {
namespace {

matmul_tiling::DataType ToCannInputType(DType dtype)
{
    switch (dtype) {
        case DType::FP16: return matmul_tiling::DataType::DT_FLOAT16;
        case DType::BF16: return matmul_tiling::DataType::DT_BF16;
        case DType::FP32: return matmul_tiling::DataType::DT_FLOAT;
        case DType::INT8: return matmul_tiling::DataType::DT_INT8;
    }
    return matmul_tiling::DataType::DT_FLOAT16;
}

matmul_tiling::DataType ToCannOutputType(DType dtype)
{
    if (dtype == DType::INT8) {
        return matmul_tiling::DataType::DT_INT32;
    }
    return ToCannInputType(dtype);
}

void AppendSetterError(std::ostringstream &os, const char *name, int32_t rc)
{
    if (rc != 0) {
        if (os.tellp() > 0) {
            os << '|';
        }
        os << name << '=' << rc;
    }
}

template <typename T>
T CeilDiv(T x, T y)
{
    return y <= 0 ? 0 : (x + y - 1) / y;
}

std::string ContractError(const std::string &reason)
{
    return "tiling_contract:" + reason;
}

int32_t BaseKAlignment(const Workload &workload)
{
    return workload.dtype == DType::FP32 &&
        !workload.transA && workload.transB ? 8 : 16;
}

}  // namespace

std::string ValidateTilingContract(const Workload &workload, const TCubeTiling &tiling,
                                   const PlatformCaps &caps, ExecutionMode &mode)
{
    mode = ExecutionMode::NONE;
    if (tiling.M != workload.m || tiling.N != workload.n ||
        tiling.Ka != workload.k || tiling.Kb != workload.k) {
        return ContractError("official shape does not match workload");
    }
    if (tiling.usedCoreNum <= 0 || tiling.singleCoreM <= 0 || tiling.singleCoreN <= 0 ||
        tiling.singleCoreK <= 0 || tiling.baseM <= 0 || tiling.baseN <= 0 || tiling.baseK <= 0) {
        return ContractError("non-positive core or tile dimension");
    }

    const int32_t coreLimit = std::max(1, std::min(workload.maxCores, caps.coreNum));
    if (tiling.usedCoreNum > coreLimit) {
        return ContractError("usedCoreNum exceeds workload/platform limit");
    }
    const int32_t k0 = BaseKAlignment(workload);
    if (tiling.baseM % 16 != 0 || tiling.baseN % 16 != 0 ||
        k0 <= 0 || tiling.baseK % k0 != 0) {
        return ContractError("baseMNK violates Cube M0/N0/K0 alignment");
    }
    const int64_t alignedSingleM = CeilDiv<int64_t>(tiling.singleCoreM, 16) * 16;
    const int64_t alignedSingleN = CeilDiv<int64_t>(tiling.singleCoreN, 16) * 16;
    const int64_t alignedSingleK = CeilDiv<int64_t>(tiling.singleCoreK, k0) * k0;
    if (tiling.baseM > alignedSingleM || tiling.baseN > alignedSingleN ||
        tiling.baseK > alignedSingleK) {
        return ContractError("base tile exceeds the aligned official single-core shape");
    }
    if (tiling.stepM <= 0 || tiling.stepN <= 0 || tiling.stepKa <= 0 || tiling.stepKb <= 0 ||
        tiling.depthA1 <= 0 || tiling.depthB1 <= 0) {
        return ContractError("non-positive L1 depth or step");
    }
    if (tiling.iterateOrder < 0 || tiling.iterateOrder > 1) {
        return ContractError("iterateOrder is not ORDER_M/ORDER_N");
    }
    if (tiling.isBias != 0) {
        return ContractError("kernel template does not provide bias");
    }

    const int64_t oneA1 = static_cast<int64_t>(tiling.stepM) * tiling.stepKa;
    const int64_t oneB1 = static_cast<int64_t>(tiling.stepN) * tiling.stepKb;
    if (oneA1 <= 0 || oneB1 <= 0 || tiling.depthA1 % oneA1 != 0 || tiling.depthB1 % oneB1 != 0) {
        return ContractError("L1 depth is not stepM/N * stepKa/b times a buffer count");
    }
    const int64_t dbA1 = tiling.depthA1 / oneA1;
    const int64_t dbB1 = tiling.depthB1 / oneB1;
    if (dbA1 < 1 || dbA1 > 2 || dbB1 < 1 || dbB1 > 2) {
        return ContractError("L1 buffer count is outside [1,2]");
    }
    if (tiling.dbL0A < 1 || tiling.dbL0A > 2 || tiling.dbL0B < 1 || tiling.dbL0B > 2 ||
        tiling.dbL0C < 1 || tiling.dbL0C > 2) {
        return ContractError("invalid final L0 buffer count");
    }

    const uint64_t inputBytes = static_cast<uint64_t>(DTypeBytes(workload.dtype));
    const uint64_t l0a = static_cast<uint64_t>(tiling.baseM) * tiling.baseK *
        inputBytes * tiling.dbL0A;
    const uint64_t l0b = static_cast<uint64_t>(tiling.baseN) * tiling.baseK *
        inputBytes * tiling.dbL0B;
    const uint64_t l0c = static_cast<uint64_t>(tiling.baseM) * tiling.baseN *
        AccumulatorBytes(workload.dtype) * tiling.dbL0C;
    if (l0a > caps.l0aBytes || l0b > caps.l0bBytes || l0c > caps.l0cBytes) {
        return ContractError("final DB-adjusted L0 allocation exceeds platform capacity");
    }
    const uint64_t a1 = static_cast<uint64_t>(tiling.depthA1) * tiling.baseM *
        tiling.baseK * inputBytes;
    const uint64_t b1 = static_cast<uint64_t>(tiling.depthB1) * tiling.baseN *
        tiling.baseK * inputBytes;
    const uint64_t effectiveL1 = CeilDiv<uint64_t>(caps.l1Bytes, 1024) * 1024;
    if (a1 > effectiveL1 || b1 > effectiveL1 || a1 + b1 > effectiveL1) {
        return ContractError("final A1/B1 allocation exceeds L1 capacity");
    }
    if (tiling.shareL1Size < 0 || tiling.shareL0CSize < 0 || tiling.shareUbSize < 0 ||
        static_cast<uint64_t>(tiling.shareL1Size) > effectiveL1 ||
        static_cast<uint64_t>(tiling.shareL0CSize) > caps.l0cBytes ||
        static_cast<uint64_t>(tiling.shareUbSize) > caps.ubBytes) {
        return ContractError("official shared workspace exceeds platform capacity");
    }

    const uint64_t mParts = CeilDiv<uint64_t>(workload.m, tiling.singleCoreM);
    const uint64_t nParts = CeilDiv<uint64_t>(workload.n, tiling.singleCoreN);
    const uint64_t kParts = CeilDiv<uint64_t>(workload.k, tiling.singleCoreK);
    const uint64_t logicalTiles = mParts * nParts * kParts;

    if (kParts == 1) {
        if (tiling.singleCoreK < workload.k) {
            return ContractError("base mode does not cover the full K axis");
        }
        mode = ExecutionMode::BASE_ITERATE_ALL;
        return {};
    }
    (void)logicalTiles;
    return ContractError(
        "generic seed stage accepts BASE only; MatMulV3 Split-K templates "
        "are generated by the exact callback-contract search");
}

OfficialTilingEngine::OfficialTilingEngine(const std::string &socVersion)
{
    platform_ = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion.c_str());
    if (platform_ == nullptr) {
        error_ = "PlatformAscendCManager returned null";
        return;
    }

    caps_.coreNum = static_cast<int32_t>(platform_->GetCoreNumAic());
    if (caps_.coreNum <= 0) {
        caps_.coreNum = static_cast<int32_t>(platform_->GetCoreNum());
    }
    if (caps_.coreNum <= 0) {
        caps_.coreNum = 24;
    }

    auto readSize = [this](platform_ascendc::CoreMemType type, uint64_t fallback) {
        uint64_t size = 0;
        platform_->GetCoreMemSize(type, size);
        return size == 0 ? fallback : size;
    };
    caps_.l0aBytes = readSize(platform_ascendc::CoreMemType::L0_A, caps_.l0aBytes);
    caps_.l0bBytes = readSize(platform_ascendc::CoreMemType::L0_B, caps_.l0bBytes);
    caps_.l0cBytes = readSize(platform_ascendc::CoreMemType::L0_C, caps_.l0cBytes);
    caps_.l1Bytes = readSize(platform_ascendc::CoreMemType::L1, caps_.l1Bytes);
    caps_.ubBytes = readSize(platform_ascendc::CoreMemType::UB, caps_.ubBytes);
    caps_.l2Bytes = readSize(platform_ascendc::CoreMemType::L2, caps_.l2Bytes);

    platform_->GetCoreMemBw(platform_ascendc::CoreMemType::L2, caps_.l2BytesPerCycle);
    platform_->GetCoreMemBw(platform_ascendc::CoreMemType::HBM, caps_.hbmBytesPerCycle);
}

bool OfficialTilingEngine::IsReady() const
{
    return platform_ != nullptr;
}

const std::string &OfficialTilingEngine::Error() const
{
    return error_;
}

const PlatformCaps &OfficialTilingEngine::Caps() const
{
    return caps_;
}

Evaluation OfficialTilingEngine::Evaluate(const Workload &workload, const Candidate &candidate) const
{
    Evaluation result;
    result.workload = workload;
    result.candidate = candidate;

    if (platform_ == nullptr) {
        result.error = error_;
        return result;
    }
    if (workload.m <= 0 || workload.n <= 0 || workload.k <= 0) {
        result.error = "shape must be positive";
        return result;
    }
    if (workload.dtype == DType::INT8) {
        result.error = "MatMulV3 supports fp16, bf16, and fp32 inputs; int8 belongs to a different operator path";
        return result;
    }

    try {
        matmul_tiling::MultiCoreMatmulTiling tiler(*platform_);
        std::ostringstream setterErrors;
        const auto inputType = ToCannInputType(workload.dtype);
        const auto outputType = ToCannOutputType(workload.dtype);

        AppendSetterError(setterErrors, "SetDim", tiler.SetDim(std::min(workload.maxCores, caps_.coreNum)));
        AppendSetterError(setterErrors, "SetAType", tiler.SetAType(
            matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputType, workload.transA));
        AppendSetterError(setterErrors, "SetBType", tiler.SetBType(
            matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputType, workload.transB));
        AppendSetterError(setterErrors, "SetCType", tiler.SetCType(
            matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, outputType));
        AppendSetterError(setterErrors, "SetShape", tiler.SetShape(workload.m, workload.n, workload.k));
        AppendSetterError(setterErrors, "SetOrgShape", tiler.SetOrgShape(workload.m, workload.n, workload.k));
        AppendSetterError(setterErrors, "SetBufferSpace", tiler.SetBufferSpace(
            static_cast<int32_t>(caps_.l1Bytes), static_cast<int32_t>(caps_.l0cBytes),
            static_cast<int32_t>(caps_.ubBytes)));
        AppendSetterError(setterErrors, "SetBias", tiler.SetBias(false));
        if (!candidate.splitK && candidate.baseM > 0 && candidate.baseN > 0) {
            AppendSetterError(setterErrors, "SetSingleShapeBaseTemplate", tiler.SetSingleShape(
                candidate.baseM, candidate.baseN, -1));
        } else if (candidate.singleM > 0 || candidate.singleN > 0 || candidate.singleK > 0) {
            AppendSetterError(setterErrors, "SetSingleShape", tiler.SetSingleShape(
                candidate.singleM, candidate.singleN, candidate.singleK));
        }
        AppendSetterError(setterErrors, "SetFixSplit", tiler.SetFixSplit(
            candidate.baseM, candidate.baseN, candidate.baseK));
        AppendSetterError(setterErrors, "SetDoubleBuffer", tiler.SetDoubleBuffer(
            candidate.dbA, candidate.dbB, false, false, true, true));

        if (candidate.traverse == 1) {
            AppendSetterError(setterErrors, "SetTraverseM", tiler.SetTraverse(matmul_tiling::MatrixTraverse::FIRSTM));
        } else if (candidate.traverse == 2) {
            AppendSetterError(setterErrors, "SetTraverseN", tiler.SetTraverse(matmul_tiling::MatrixTraverse::FIRSTN));
        }
        tiler.EnableMultiCoreSplitK(candidate.splitK);

        if (setterErrors.tellp() > 0) {
            result.error = setterErrors.str();
            return result;
        }

        result.officialReturn = tiler.GetTiling(result.tiling);
        if (result.officialReturn == -1) {
            result.error = "MultiCoreMatmulTiling::GetTiling returned -1";
            return result;
        }
        if (tiler.GetCoreNum(result.officialCoreNum, result.officialMDim, result.officialNDim) != 0) {
            result.error = "MultiCoreMatmulTiling::GetCoreNum returned nonzero";
            return result;
        }
        const int64_t expectedMDim = CeilDiv<int64_t>(workload.m, result.tiling.singleCoreM);
        const int64_t expectedNDim = CeilDiv<int64_t>(workload.n, result.tiling.singleCoreN);
        if (result.officialCoreNum != result.tiling.usedCoreNum ||
            result.officialMDim != expectedMDim || result.officialNDim != expectedNDim) {
            result.error = "tiling_contract:GetCoreNum disagrees with final TCubeTiling M/N partition";
            return result;
        }
        result.tilingSignature = TilingSignature(result.tiling);
        result.error = ValidateTilingContract(workload, result.tiling, caps_, result.executionMode);
        if (!result.error.empty()) {
            return result;
        }

        result.valid = true;
        return result;
    } catch (const std::exception &ex) {
        result.error = std::string("exception:") + ex.what();
        return result;
    } catch (...) {
        result.error = "unknown exception";
        return result;
    }
}

}  // namespace matmul_search
