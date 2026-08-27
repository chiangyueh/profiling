#!/usr/bin/env python3
"""Prepare FASG or FIAS overlays from only the dedicated CANN-8.1 lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["CANN81_SOURCE_LOCK"] = str(ROOT / "remaining_operators_cann81_lock.json")
import prepare_fasg_strategy_overlays as fasg
import prepare_fias_source_overlay as fias


LOCK = json.loads((ROOT / "remaining_operators_cann81_lock.json").read_text(encoding="utf-8"))


DELEGATE_SOURCE = r'''#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <map>
#include <string>

#include "exe_graph/runtime/tiling_context.h"
#include "experiment/platform/platform/platform_infos_def.h"
#include "register/op_impl_registry.h"

namespace optiling {
namespace {

constexpr const char *kAuditEnvironment = "@AUDIT_ENV@";
constexpr const char *kCoreEnvironment = "@CORE_ENV@";
constexpr const char *kCapacityEnvironment = "@CAPACITY_ENV@";
constexpr const char *kOfficialLibraryEnvironment = "@OFFICIAL_LIBRARY_ENV@";
constexpr const char *kAuditSchema = "@AUDIT_SCHEMA@";
constexpr const char *kCapacityAuditField = "@CAPACITY_AUDIT_FIELD@";
constexpr const char *kOfficialTilingSymbol = "@OFFICIAL_TILING_SYMBOL@";

using OfficialTiling = ge::graphStatus (*)(gert::TilingContext *);

#if defined(ATTENTION_DELEGATE_FASG)
struct ExactFasgCompileInfo {
    uint32_t aivNum;
    uint32_t aicNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0aSize;
    uint64_t l0bSize;
    uint64_t l0cSize;
    uint64_t l2CacheSize;
    int64_t coreNum;
};
#else
struct ExactFiasCompileInfo {
    uint32_t aivNum;
    uint32_t aicNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0CSize;
    uint64_t l0ASize;
    uint64_t l0BSize;
    size_t defaultSysWorkspaceSize;
    uint32_t socShortName;
};
#endif

bool ParseUnsigned(const char *text, uint64_t &value)
{
    if (text == nullptr || *text == '\0') return false;
    errno = 0;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0ULL) return false;
    value = static_cast<uint64_t>(parsed);
    return true;
}

uint64_t Fnv1a64(const uint8_t *data, size_t size)
{
    uint64_t value = 1469598103934665603ULL;
    for (size_t index = 0; index < size; ++index) {
        value ^= static_cast<uint64_t>(data[index]);
        value *= 1099511628211ULL;
    }
    return value;
}

class ScopedHardwareView {
public:
    explicit ScopedHardwareView(fe::PlatFormInfos *platform) : platform_(platform) {}

    bool Apply()
    {
        if (platform_ == nullptr || !platform_->GetPlatformRes("SoCInfo", originalSoc_) ||
            !platform_->GetPlatformRes("AICoreSpec", originalAicore_)) return false;
        std::map<std::string, std::string> soc = originalSoc_;
        std::map<std::string, std::string> aicore = originalAicore_;
        uint64_t requested = 0;
        uint64_t divisor = 0;
        if (!ParseUnsigned(std::getenv(kCoreEnvironment), requested) ||
            !ParseUnsigned(std::getenv(kCapacityEnvironment), divisor) ||
            (divisor != 1ULL && divisor != 2ULL && divisor != 4ULL && divisor != 8ULL)) return false;
        uint64_t runtimeAic = 0;
        uint64_t runtimeAiv = 0;
        if (!ParseUnsigned(soc["cube_core_cnt"].c_str(), runtimeAic) ||
            !ParseUnsigned(soc["vector_core_cnt"].c_str(), runtimeAiv) ||
            requested > runtimeAic || runtimeAiv % runtimeAic != 0ULL) return false;
        const uint64_t aivPerAic = runtimeAiv / runtimeAic;
        soc["ai_core_cnt"] = std::to_string(requested);
        soc["cube_core_cnt"] = std::to_string(requested);
        soc["vector_core_cnt"] = std::to_string(requested * aivPerAic);
#if defined(ATTENTION_DELEGATE_FASG)
        uint64_t capacity = 0;
        if (!ParseUnsigned(soc["l2_size"].c_str(), capacity) || capacity / divisor == 0ULL) return false;
        soc["l2_size"] = std::to_string(capacity / divisor);
#else
        uint64_t capacity = 0;
        if (!ParseUnsigned(aicore["ub_size"].c_str(), capacity) || capacity / divisor == 0ULL) return false;
        aicore["ub_size"] = std::to_string(capacity / divisor);
#endif
        platform_->SetPlatformRes("SoCInfo", soc);
        platform_->SetPlatformRes("AICoreSpec", aicore);
        applied_ = true;
        return true;
    }

    ~ScopedHardwareView()
    {
        if (!applied_) return;
        platform_->SetPlatformRes("SoCInfo", originalSoc_);
        platform_->SetPlatformRes("AICoreSpec", originalAicore_);
    }

private:
    fe::PlatFormInfos *platform_ = nullptr;
    bool applied_ = false;
    std::map<std::string, std::string> originalSoc_;
    std::map<std::string, std::string> originalAicore_;
};

#if defined(ATTENTION_DELEGATE_FASG)
class ScopedCompileInfoView {
public:
    explicit ScopedCompileInfoView(gert::TilingContext *context)
        : compile_(context == nullptr ? nullptr
                                      : const_cast<ExactFasgCompileInfo *>(
                                            context->GetCompileInfo<ExactFasgCompileInfo>())) {}

    bool Apply()
    {
        if (compile_ == nullptr) return false;
        uint64_t requested = 0;
        uint64_t divisor = 0;
        if (!ParseUnsigned(std::getenv(kCoreEnvironment), requested) ||
            !ParseUnsigned(std::getenv(kCapacityEnvironment), divisor) ||
            (divisor != 1ULL && divisor != 2ULL && divisor != 4ULL && divisor != 8ULL) ||
            compile_->aicNum == 0U || compile_->aivNum == 0U ||
            compile_->aivNum % compile_->aicNum != 0U || requested > compile_->aicNum ||
            compile_->l2CacheSize == 0ULL || compile_->l2CacheSize / divisor == 0ULL) return false;
        original_ = *compile_;
        const uint64_t aivPerAic = compile_->aivNum / compile_->aicNum;
        compile_->aicNum = static_cast<uint32_t>(requested);
        compile_->aivNum = static_cast<uint32_t>(requested * aivPerAic);
        compile_->l2CacheSize /= divisor;
        applied_ = true;
        return true;
    }

    ~ScopedCompileInfoView()
    {
        if (applied_) *compile_ = original_;
    }

private:
    ExactFasgCompileInfo *compile_ = nullptr;
    ExactFasgCompileInfo original_{};
    bool applied_ = false;
};
#endif

OfficialTiling ResolveOfficialTiling()
{
    const char *path = std::getenv(kOfficialLibraryEnvironment);
    if (path == nullptr || *path == '\0') return nullptr;
    void *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) return nullptr;
    dlerror();
    auto function = reinterpret_cast<OfficialTiling>(dlsym(handle, kOfficialTilingSymbol));
    return dlerror() == nullptr ? function : nullptr;
}

ge::graphStatus Audit(gert::TilingContext *context, ge::graphStatus status)
{
    const char *path = std::getenv(kAuditEnvironment);
    if (path == nullptr || context == nullptr) return status;
    auto *raw = context->GetRawTilingData();
    const size_t size = raw == nullptr ? 0U : raw->GetDataSize();
    const auto *bytes = raw == nullptr ? nullptr : static_cast<const uint8_t *>(raw->GetData());
    const uint64_t digest = bytes == nullptr ? 0ULL : Fnv1a64(bytes, size);
    std::FILE *output = std::fopen(path, "a");
    if (output != nullptr) {
        const char *core = std::getenv(kCoreEnvironment);
        const char *capacity = std::getenv(kCapacityEnvironment);
        std::fprintf(output,
            "{\"schema\":\"%s\",\"status\":%d,\"aiv_core_cap\":\"%s\",\"%s\":\"%s\","
            "\"tiling_key\":%llu,\"block_dim\":%u,\"raw_bytes\":%llu,\"raw_fnv1a64\":%llu}\n",
            kAuditSchema, static_cast<int>(status), core == nullptr ? "missing" : core, kCapacityAuditField,
            capacity == nullptr ? "missing" : capacity, static_cast<unsigned long long>(context->GetTilingKey()),
            context->GetBlockDim(), static_cast<unsigned long long>(size),
            static_cast<unsigned long long>(digest));
        std::fclose(output);
    }
    return status;
}

ge::graphStatus ExactCANN81Tiling(gert::TilingContext *context)
{
    if (context == nullptr) return ge::GRAPH_FAILED;
    ScopedHardwareView hardware(context->GetPlatformInfo());
    if (!hardware.Apply()) return Audit(context, ge::GRAPH_FAILED);
#if defined(ATTENTION_DELEGATE_FASG)
    ScopedCompileInfoView compileInfo(context);
    if (!compileInfo.Apply()) return Audit(context, ge::GRAPH_FAILED);
#endif
    OfficialTiling official = ResolveOfficialTiling();
    if (official == nullptr) return Audit(context, ge::GRAPH_FAILED);
    return Audit(context, official(context));
}

}  // namespace

#if defined(ATTENTION_DELEGATE_FASG)
using OfficialParse = ge::graphStatus (*)(gert::TilingParseContext *);
ge::graphStatus ExactCANN81Parse(gert::TilingParseContext *context)
{
    const char *path = std::getenv(kOfficialLibraryEnvironment);
    if (path == nullptr || *path == '\0') return ge::GRAPH_FAILED;
    void *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) return ge::GRAPH_FAILED;
    dlerror();
    auto function = reinterpret_cast<OfficialParse>(dlsym(handle,
        "_ZN8optiling39TilingPrepareForFlashAttentionScoreGradEPN4gert18TilingParseContextE"));
    if (dlerror() != nullptr || function == nullptr) return ge::GRAPH_FAILED;
    return function(context);
}

IMPL_OP(FlashAttentionScoreGrad)
    .Tiling(ExactCANN81Tiling)
    .TilingInputsDataDependency({12, 13, 14, 15, 16})
    .TilingParse<ExactFasgCompileInfo>(ExactCANN81Parse);
#else
ge::graphStatus ExactCANN81Parse(gert::TilingParseContext *) { return ge::GRAPH_SUCCESS; }

IMPL_OP_OPTILING(FusedInferAttentionScore)
    .TilingInputsDataDependency({5, 6, 15, 16, 23},
                                {gert::TilingPlacement::TILING_ON_HOST,
                                 gert::TilingPlacement::TILING_ON_AICPU})
    .Tiling(ExactCANN81Tiling)
    .TilingParse<ExactFiasCompileInfo>(ExactCANN81Parse);
#endif

}  // namespace optiling
'''


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def delegate_source(operator: str) -> tuple[str, str]:
    if operator == "flash_attention_score_grad":
        values = {
            "@AUDIT_ENV@": "FASG_TILING_AUDIT_PATH",
            "@CORE_ENV@": "FASG_SOURCE_AIV_CAP",
            "@CAPACITY_ENV@": "FASG_SOURCE_L2_DIVISOR",
            "@OFFICIAL_LIBRARY_ENV@": "FASG_OFFICIAL_TILING_LIBRARY",
            "@AUDIT_SCHEMA@": "fasg_raw_tiling_observation_v2",
            "@CAPACITY_AUDIT_FIELD@": "l2_cap_divisor",
            "@OFFICIAL_TILING_SYMBOL@": "_ZN8optiling29TilingFlashAttentionGradScoreEPN4gert13TilingContextE",
        }
        define = "ATTENTION_DELEGATE_FASG=1"
    else:
        values = {
            "@AUDIT_ENV@": "FIAS_TILING_AUDIT_PATH",
            "@CORE_ENV@": "FIAS_SOURCE_AIV_CAP",
            "@CAPACITY_ENV@": "FIAS_SOURCE_UB_DIVISOR",
            "@OFFICIAL_LIBRARY_ENV@": "FIAS_OFFICIAL_TILING_LIBRARY",
            "@AUDIT_SCHEMA@": "fias_raw_tiling_observation_v2",
            "@CAPACITY_AUDIT_FIELD@": "ub_cap_divisor",
            "@OFFICIAL_TILING_SYMBOL@": "_ZN8optiling34DoOpTilingFusedInferAttentionScoreEPN4gert13TilingContextE",
        }
        define = "ATTENTION_DELEGATE_FIAS=1"
    source = DELEGATE_SOURCE
    for old, new in values.items():
        source = source.replace(old, new)
    if "@" in source:
        raise RuntimeError("unexpanded exact CANN-8.1 delegate token")
    return source, define


def prepare_exact_delegate(operator: str, project: Path, metadata: dict[str, object]) -> dict[str, object]:
    source, define = delegate_source(operator)
    op_root = project / str(LOCK["operators"][operator]["relative_root"])
    host = op_root / "ophost"
    target = host / "exact_cann81_tiling_delegate.cpp"
    cmake = host / "CMakeLists.txt"
    original_cmake = cmake.read_text(encoding="utf-8")
    source_pattern = r"target_sources\(optiling PRIVATE\n.*?\n\)"
    replacement = (
        "target_sources(optiling PRIVATE\n"
        "        exact_cann81_tiling_delegate.cpp\n"
        ")\n"
        "target_compile_definitions(optiling PRIVATE {})\n"
        "target_link_libraries(optiling PRIVATE dl platform)".format(define)
    )
    cmake_text, count = re.subn(source_pattern, replacement, original_cmake, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError("cannot isolate exact CANN-8.1 attention delegate source")
    if operator == "fused_infer_attention_score":
        cmake_text, depends_count = re.subn(
            r"set\(fused_infer_attention_score_depends .*?\n\s*PARENT_SCOPE\)\n", "",
            cmake_text, count=1, flags=re.DOTALL)
        if depends_count != 1:
            raise RuntimeError("cannot remove alpha001 FIAS host dependencies")
    dependency_cmake: list[dict[str, str]] = []
    for relative in metadata["build_harness"]["build_scope"]["op_host_directories"]:
        dependency_host = project / str(relative)
        if dependency_host == host:
            continue
        dependency = dependency_host / "CMakeLists.txt"
        before = dependency.read_text(encoding="utf-8")
        after, dependency_count = re.subn(source_pattern, "", before, count=1, flags=re.DOTALL)
        if dependency_count != 1:
            raise RuntimeError("cannot exclude alpha001 dependency tiler: {}".format(dependency))
        dependency.write_text(after, encoding="utf-8")
        dependency_cmake.append({"path": str(dependency.relative_to(project)),
                                 "sha256": _digest_text(after)})
    target.write_text(source, encoding="utf-8")
    cmake.write_text(cmake_text, encoding="utf-8")
    metadata = dict(metadata)
    metadata.update({
        "schema": "remaining_attention_exact_cann81_delegate_v1",
        "operator": ("FlashAttentionScoreGrad" if operator == "flash_attention_score_grad"
                     else "FusedInferAttentionScore"),
        "runtime_op": operator,
        "delegate_source": str(target.relative_to(project)),
        "delegate_source_sha256": _digest_text(source),
        "delegate_cmake_sha256": _digest_text(cmake_text),
        "excluded_alpha001_dependency_cmake": dependency_cmake,
        "host_tiling_origin": "installed_cann81_rc1_liboptiling",
        "alpha001_host_tiling_compiled": False,
        "official_dispatch_delegated_without_strategy_forcing": True,
        "strategy_algorithm_changes": False,
        "kernel_algorithm_changes": False,
        "instrumentation": {
            "enabled": True,
            "audit_schema": ("fasg_raw_tiling_observation_v2" if operator == "flash_attention_score_grad"
                             else "fias_raw_tiling_observation_v2"),
            "audit_environment": ("FASG_TILING_AUDIT_PATH" if operator == "flash_attention_score_grad"
                                  else "FIAS_TILING_AUDIT_PATH"),
            "source_budget_environment": ("FASG_SOURCE_AIV_CAP" if operator == "flash_attention_score_grad"
                                          else "FIAS_SOURCE_AIV_CAP"),
            "mutates_tiling_output_fields": False,
            "temporarily_mutates_compile_info_before_official_tiler":
                operator == "flash_attention_score_grad",
            "mutates_process_local_platform_view_before_official_tiler": True,
        },
    })
    (project / "source_candidate_overlay.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def existing_exact_delegate(operator: str, project: Path) -> dict[str, object] | None:
    manifest = project / "source_candidate_overlay.json"
    if not manifest.is_file():
        return None
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    if metadata.get("schema") != "remaining_attention_exact_cann81_delegate_v1":
        return None
    source, _ = delegate_source(operator)
    target = project / str(metadata.get("delegate_source", ""))
    cmake = target.parent / "CMakeLists.txt"
    if (not target.is_file() or target.read_text(encoding="utf-8") != source or
            not cmake.is_file() or hashlib.sha256(cmake.read_bytes()).hexdigest() !=
            metadata.get("delegate_cmake_sha256") or
            metadata.get("runtime_op") != operator or
            metadata.get("alpha001_host_tiling_compiled") is not False):
        raise RuntimeError("existing exact CANN-8.1 attention delegate is incomplete")
    for item in metadata.get("excluded_alpha001_dependency_cmake", []):
        dependency = project / str(item.get("path", ""))
        if (not dependency.is_file() or hashlib.sha256(dependency.read_bytes()).hexdigest() !=
                item.get("sha256")):
            raise RuntimeError("existing exact CANN-8.1 dependency exclusion is incomplete")
    return metadata


def source_compatibility(name: str) -> dict[str, object]:
    value = dict(LOCK["sources"][name])
    value["commit"] = value["official_commit"]
    if name == "cann_ops":
        value["build_harness_cmake_sha256"] = value["root_cmake_sha256"]
    return value


def fasg_operator() -> dict[str, object]:
    item = dict(LOCK["operators"]["flash_attention_score_grad"])
    item["pinned_files"] = {
        "ophost/flash_attention_score_grad_tiling.cpp": item["entry_sha256"],
        **item["strategy_files"],
    }
    return item


def fias_operator() -> dict[str, object]:
    item = dict(LOCK["operators"]["fused_infer_attention_score"])
    item["pinned_files"] = {
        item["tiler"]: item["tiler_sha256"],
        "../incre_flash_attention/ophost/incre_flash_attention_tiling.cc": item["decode_tiler_sha256"],
    }
    return item


def configure_modules() -> None:
    advanced = source_compatibility("cann_ops_adv")
    base = source_compatibility("cann_ops")
    fasg.LOCK = LOCK
    fasg.OP = fasg_operator()
    fasg.SOURCE = advanced
    fasg.HARNESS_SOURCE = base
    fasg.MACRO = str(fasg.OP["registration_macro"])
    fasg.OPERATOR = str(fasg.OP["registration_operator"])
    fasg.ENTRY_RELATIVE = Path(str(fasg.OP["relative_root"])) / "ophost/flash_attention_score_grad_tiling.cpp"
    fias.LOCK = LOCK
    fias.SOURCE = advanced
    fias.OP = fias_operator()
    fias.OP_ROOT = Path(str(fias.OP["relative_root"]))
    fias.FIAS_RELATIVE = fias.OP_ROOT / "ophost/fused_infer_attention_score_tiling.cpp"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", required=True,
                        choices=("flash_attention_score_grad", "fused_infer_attention_score"))
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--harness-root", required=True, type=Path)
    parser.add_argument("--output-parent", required=True, type=Path)
    args = parser.parse_args()
    configure_modules()
    source_root = args.source_root.resolve()
    harness_root = args.harness_root.resolve()
    output_parent = args.output_parent.resolve()
    if not output_parent.is_dir():
        raise RuntimeError("output parent is absent")
    project_name = ("fasg_official_semantic_dispatch" if args.operator == "flash_attention_score_grad"
                    else "fias_source_dispatch")
    existing = existing_exact_delegate(args.operator, output_parent / project_name)
    if existing is not None:
        print(json.dumps({"schema": "remaining_attention_exact_cann81_delegate_batch_v1",
                          "operator": args.operator, "result": existing,
                          "matmul_included": False}, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.operator == "flash_attention_score_grad":
        rows = fasg.require_pinned_source(source_root)
        harness_text, provenance = fasg.source_build_harness(harness_root, "flash_attention_score_grad")
        result = fasg.write_dispatcher_overlay(
            source_root, output_parent, rows, harness_root, harness_text, provenance)
    else:
        fias.require_pinned_source(source_root)
        harness_text, provenance = fasg.source_build_harness(harness_root, "fused_infer_attention_score")
        result = fias.write_overlay(source_root, output_parent, harness_root, harness_text, provenance)
    result = prepare_exact_delegate(args.operator, output_parent / project_name, result)
    print(json.dumps({"schema": "remaining_attention_cann81_overlay_v1", "operator": args.operator,
                      "result": result, "legacy_source_lock_read_for_execution": False,
                      "matmul_included": False}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, json.JSONDecodeError) as error:
        print("remaining attention overlay error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
