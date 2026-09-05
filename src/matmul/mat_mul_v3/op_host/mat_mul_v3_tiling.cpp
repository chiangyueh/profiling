/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file mat_mul_v3.cpp
 */
#include "mat_mul_v3_tiling.h"
#include "mat_mul_v3_compile_info.h"
#include "mat_mul_v3_base_tiling.h"

#include "tiling/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "cube_tiling_runtime.h"
#include "cache_tiling.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

#define OP_LOGI(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)

#define OP_LOGD(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)

namespace optiling {
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr) {                                                                        \
    std::printf("nullptr error!");                                                               \
    return ge::GRAPH_FAILED;                                                                     \
  }
}  // namespace optiling

namespace optiling {
#define CUBE_INNER_ERR_REPORT(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_TILING_CHECK(cond, log_func, expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      expr;                                   \
    }                                         \
  } while (0)
}  // namespace optiling

using namespace optiling::matmul_v3;

namespace {
static const size_t DEST_MAX = 100;
static const size_t MAX_LEN_SIMPLIFIED_KEY = 256;
static const int32_t INPUT0_INDEX = 0;
static const int32_t INPUT1_INDEX = 1;
static const int32_t BIAS_INDEX = 2;
}

namespace optiling {

REGISTER_TILING_TEMPLATE("MatMulV3", MatmulV3BaseTiling, 0);

namespace {
constexpr uint64_t MATMUL_TILING_KEY_OFFSET = 10000000000000000000UL;

struct SourceRoute {
    const char *name;
    TilingCalcSelect selector;
};

const std::array<SourceRoute, 7> SOURCE_ROUTES{{
    {"ALL", TilingCalcSelect::ALL},
    {"BASE", TilingCalcSelect::BASE},
    {"SINGLE_CORE_SPLIT_K", TilingCalcSelect::SINGLE_CORE_SPLIT_K},
    {"DETERMINISTIC_SPLIT_K", TilingCalcSelect::DETERMINISTIC_SPLIT_K},
    {"AL1_FULL_LOAD", TilingCalcSelect::AL1_FULL_LOAD},
    {"BL1_FULL_LOAD", TilingCalcSelect::BL1_FULL_LOAD},
    {"BL1_FULL_LOAD_FIXPIPE", TilingCalcSelect::BL1_FULL_LOAD_FIXPIPE},
}};

class MatmulV3SourceRouteTiling final : public MatmulV3BaseTiling {
public:
    MatmulV3SourceRouteTiling(gert::TilingContext *context, MatmulTilingData *data,
                             TilingCalcSelect selector, uint64_t coreCap)
        : MatmulV3BaseTiling(context, data, selector), coreCap_(coreCap) {}

protected:
    ge::graphStatus GetPlatformInfo() override
    {
        const ge::graphStatus status = MatmulV3BaseTiling::GetPlatformInfo();
        if (status == ge::GRAPH_SUCCESS) {
            compileInfo_.aicNum = std::max<uint64_t>(1, std::min(compileInfo_.aicNum, coreCap_));
        }
        return status;
    }

private:
    uint64_t coreCap_;
};

std::string JsonEscape(const char *value)
{
    std::ostringstream output;
    for (const unsigned char ch : std::string(value == nullptr ? "" : value)) {
        switch (ch) {
            case '\\': output << "\\\\"; break;
            case '"': output << "\\\""; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (ch < 0x20) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<uint32_t>(ch) << std::dec;
                } else {
                    output << ch;
                }
        }
    }
    return output.str();
}

bool RouteMatches(const TilingCalcSelect selector, const uint64_t tilingKey)
{
    if (tilingKey < MATMUL_TILING_KEY_OFFSET) {
        return false;
    }
    if (selector == TilingCalcSelect::ALL) {
        return true;
    }
    const uint64_t suffix = tilingKey - MATMUL_TILING_KEY_OFFSET;
    const uint64_t split = (suffix / 10) % 10;
    const uint64_t fullLoad = (suffix / 100) % 10;
    const uint64_t fix = (suffix / 10000) % 10;
    switch (selector) {
        case TilingCalcSelect::BASE:
            return split == 0 && fullLoad == 0 && fix == 0;
        case TilingCalcSelect::SINGLE_CORE_SPLIT_K:
            return split == 2;
        case TilingCalcSelect::DETERMINISTIC_SPLIT_K:
            return split == 3;
        case TilingCalcSelect::AL1_FULL_LOAD:
            return split == 0 && fullLoad == 1;
        case TilingCalcSelect::BL1_FULL_LOAD:
            return split == 0 && fullLoad == 2 && fix == 0;
        case TilingCalcSelect::BL1_FULL_LOAD_FIXPIPE:
            return split == 0 && fullLoad == 2 && (fix == 1 || fix == 2);
        default:
            return false;
    }
}

void AppendSourceRouteAudit(gert::TilingContext *context)
{
    const char *path = std::getenv("MATMUL_SOURCE_ROUTE_AUDIT_PATH");
    if (path == nullptr || *path == '\0') {
        return;
    }
    const auto *compileInfo = reinterpret_cast<const MatmulV3CompileInfo *>(context->GetCompileInfo());
    if (compileInfo == nullptr || compileInfo->aicNum == 0) {
        return;
    }
    uint64_t maxCoreCap = compileInfo->aicNum;
    if (const char *requested = std::getenv("MATMUL_SOURCE_ROUTE_MAX_CORES")) {
        char *end = nullptr;
        const unsigned long parsed = std::strtoul(requested, &end, 10);
        if (end != requested && *end == '\0' && parsed > 0) {
            maxCoreCap = std::min<uint64_t>(maxCoreCap, parsed);
        }
    }
    const char *workloadId = std::getenv("MATMUL_SOURCE_ROUTE_WORKLOAD_ID");
    std::ofstream audit(path, std::ios::app);
    if (!audit) {
        return;
    }
    for (uint64_t coreCap = 1; coreCap <= maxCoreCap; ++coreCap) {
        for (const SourceRoute &route : SOURCE_ROUTES) {
            MatmulTilingData routeData;
            MatmulV3SourceRouteTiling tiler(context, &routeData, route.selector, coreCap);
            const ge::graphStatus status = tiler.DoTiling();
            const uint64_t tilingKey = context->GetTilingKey();
            const bool matched = status == ge::GRAPH_SUCCESS && RouteMatches(route.selector, tilingKey);
            audit << "{\"schema\":\"matmul_v3_source_route_observation_v1\""
                  << ",\"workload_id\":\"" << JsonEscape(workloadId) << "\""
                  << ",\"route\":\"" << route.name << "\""
                  << ",\"core_cap\":" << coreCap
                  << ",\"status\":\"" << (status == ge::GRAPH_SUCCESS ? "success" : "failed") << "\""
                  << ",\"route_matched\":" << (matched ? "true" : "false");
            if (status == ge::GRAPH_SUCCESS) {
                std::vector<uint8_t> bytes(routeData.GetDataSize());
                routeData.SaveToBuffer(bytes.data(), bytes.size());
                audit << ",\"tiling_key\":" << tilingKey
                      << ",\"block_dim\":" << context->GetBlockDim()
                      << ",\"workspace_bytes\":" << context->GetWorkspaceSizes(1)[0]
                      << ",\"raw_tiling_hex\":\"";
                for (const uint8_t byte : bytes) {
                    audit << std::hex << std::setw(2) << std::setfill('0')
                          << static_cast<uint32_t>(byte);
                }
                audit << std::dec << "\"";
            }
            audit << "}\n";
            audit.flush();
        }
    }
}
} // namespace

static ge::graphStatus MatmulV3TilingFunc(gert::TilingContext* context)
{
    OP_TILING_CHECK(context == nullptr,
            CUBE_INNER_ERR_REPORT("MatMulV3", "context is null"),
            return ge::GRAPH_FAILED);
    // Calibration only: enumerate each original route and core budget, then
    // run the untouched production ALL dispatcher last so the real executor
    // receives exactly the normal official result.
    AppendSourceRouteAudit(context);
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForMatmulV3(gert::TilingParseContext *context) {
    OP_TILING_CHECK(context == nullptr,
                CUBE_INNER_ERR_REPORT("MatMulV3", "context is null"),
                return ge::GRAPH_FAILED);
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "platformInfoPtr is null"),
                return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<MatmulV3CompileInfo>();
    OP_TILING_CHECK(compileInfoPtr == nullptr,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "compileInfoPtr is null"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    platformInfo->GetPlatformRes("version", "SoC_version", compileInfoPtr->socVersionStr);
    std::string val;
    std::string dataMoveL12Bt;
    platformInfo->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_fix_pipe_l0c2out", val);
    platformInfo->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_data_move_l12bt", dataMoveL12Bt);
    compileInfoPtr->supportL0c2out = !val.empty();
    compileInfoPtr->supportL12BtBf16 = (dataMoveL12Bt.find("bf16") != std::string::npos);
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
    compileInfoPtr->btSize = compileInfoPtr->supportL0c2out ? 1024UL : 0UL; // 1024 is btSize
    compileInfoPtr->btSize = compileInfoPtr->supportL12BtBf16 ? 4096UL : compileInfoPtr->btSize; // 4096 is btSize
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Size);

    gert::GemmCompileInfo tbeCompileInfo;
    tbeCompileInfo.ParseRuntimePlatformInfo(context->GetNodeName(), *platformInfo);
    tbeCompileInfo.core_num = compileInfoPtr->aicNum;
    OP_TILING_CHECK(tbeCompileInfo.core_num <= 0L,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "aicNum value is [%d]", tbeCompileInfo.core_num),
                return ge::GRAPH_FAILED);
    optiling::PlatformInfo::GetInstance().SetInstance(tbeCompileInfo);
    OP_LOGI(context->GetNodeName(),
            "parse compile info success soc:%d, l1Size:%lu, l2Size:%lu, coreNum:%lu, supportL0c2out:%d, supportL12BtBf16:%d",
            static_cast<int>(compileInfoPtr->socVersion),
            compileInfoPtr->l1Size,
            compileInfoPtr->l2Size,
            compileInfoPtr->aicNum,
            compileInfoPtr->supportL0c2out,
            compileInfoPtr->supportL12BtBf16);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GenSimplifiedKeyForMatMulV3(gert::TilingContext *context, ge::char_t *simplifiedKey) {
    OP_LOGI(context->GetNodeName(), "Enter GenSimplifiedKeyForMatMulV3.");
    OP_TILING_CHECK(simplifiedKey == nullptr, CUBE_INNER_ERR_REPORT("MatMulV3", "simplifiedKey is null"),
                    return ge::GRAPH_FAILED);
    std::string simpKeyTemp = "";
    strcat_s(simplifiedKey, DEST_MAX, "diy,");
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(INPUT0_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(INPUT1_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetOutputDesc(0));
    if (context->GetInputDesc(BIAS_INDEX) != nullptr) {
        simpKeyTemp = std::to_string(context->GetInputDesc(INPUT0_INDEX)->GetStorageFormat()) + "/" +
                      std::to_string(context->GetInputDesc(INPUT1_INDEX)->GetStorageFormat()) + "/" +
                      std::to_string(ge::FORMAT_ND) + "/" + // bias的format均为FormatND，因此约束为仅通过FORMAT_ND参与匹配
                      std::to_string(context->GetOutputDesc(0)->GetStorageFormat()) + "/" +
                      std::to_string(context->GetInputDesc(INPUT0_INDEX)->GetDataType()) + "/" +
                      std::to_string(context->GetInputDesc(INPUT1_INDEX)->GetDataType()) + "/" +
                      std::to_string(context->GetInputDesc(BIAS_INDEX)->GetDataType()) + "/" +
                      std::to_string(context->GetOutputDesc(0)->GetDataType());
        strcat_s(simplifiedKey, DEST_MAX, simpKeyTemp.c_str());
    } else {
        // 二进制发布json有无bias时合并为同一个json发布，当无法获取bias信息时，当前约定使用input0的信息代替
        simpKeyTemp = std::to_string(context->GetInputDesc(INPUT0_INDEX)->GetStorageFormat()) + "/" +
                      std::to_string(context->GetInputDesc(INPUT1_INDEX)->GetStorageFormat()) + "/" +
                      std::to_string(ge::FORMAT_ND) + "/" +
                      std::to_string(context->GetOutputDesc(0)->GetStorageFormat()) + "/" +
                      std::to_string(context->GetInputDesc(INPUT0_INDEX)->GetDataType()) + "/" +
                      std::to_string(context->GetInputDesc(INPUT1_INDEX)->GetDataType()) + "/" +
                      std::to_string(context->GetInputDesc(INPUT0_INDEX)->GetDataType()) + "/" +
                      std::to_string(context->GetOutputDesc(0)->GetDataType());
        strcat_s(simplifiedKey, DEST_MAX, simpKeyTemp.c_str());
    }
    OP_TILING_CHECK(strlen(simplifiedKey) > MAX_LEN_SIMPLIFIED_KEY,
                           CUBE_INNER_ERR_REPORT("MatMulV3", "len of simplifiedKey exceeds max length."),
                           return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MatMulV3)
    .Tiling(MatmulV3TilingFunc)
    .TilingParse<MatmulV3CompileInfo>(TilingPrepareForMatmulV3)
    .GenSimplifiedKey(GenSimplifiedKeyForMatMulV3);
}
