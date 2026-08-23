#!/usr/bin/env python3
"""Prepare a bounded CANN-8.1 build compatibility overlay for GatherElementsV2.

The source implementation was extracted from CANN 8.3.RC2 because public
CANN 8.1 does not expose this Ascend910B route.  The parent build framework is
an exact public CANN-8.1 checkout.  Only CMake target wiring, an observational
audit, and original-tiler resource inputs are changed; no tiling or kernel
algorithm is rewritten.  Real-NPU output equality remains mandatory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
LOCK = json.loads((ROOT / "non_matmul_source_lock.json").read_text(encoding="utf-8"))
PARENT = LOCK["sources"]["cann_ops"]
EXTRACTED = LOCK["sources"]["extracted_installed_source"]
OP = LOCK["operators"]["gather_elements_v2"]
# The extracted source is an operator directory, while the public CANN-8.1
# framework discovers operators beneath category directories.  Preserve the
# extracted directory verbatim, but place it under the original framework's
# ``src/index`` discovery root.
EXTRACTED_OP_ROOT = Path(OP["relative_root"])
OVERLAY_OP_ROOT = Path("src/index") / EXTRACTED_OP_ROOT
ENTRY = OVERLAY_OP_ROOT / "op_host/gather_elements_v2_tiling.cpp"
AUDIT = "GATHER_ELEMENTS_SOURCE_TILING_AUDIT_V1"
CORE = "GATHER_ELEMENTS_SOURCE_AIV_CAP_V1"
UB = "GATHER_ELEMENTS_SOURCE_UB_CAP_V1"
BUILD_SCOPE = "SOURCE_CANDIDATE_BUILD_SCOPE_V1"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def run(argv: list[str]) -> str:
    done = subprocess.run(argv, text=True, capture_output=True, check=False)
    if done.returncode:
        raise RuntimeError("command failed rc={} argv={} stderr={}".format(done.returncode, " ".join(argv), done.stderr.strip()))
    return done.stdout.strip()


def require_parent(path: Path) -> None:
    if run(["git", "-C", str(path), "rev-parse", "HEAD"]) != PARENT["commit"]:
        raise RuntimeError("CANN-8.1 parent revision mismatch")
    if run(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"]):
        raise RuntimeError("CANN-8.1 parent worktree is modified")
    if not (path / "CMakeLists.txt").is_file():
        raise RuntimeError("CANN-8.1 parent lacks its root CMakeLists.txt")


def require_extracted(path: Path) -> None:
    if EXTRACTED.get("source_version") != "CANN 8.3.RC2 reported by extracted run.sh":
        raise RuntimeError("GatherElements extracted-source version evidence is missing")
    for relative, expected in OP["pinned_files"].items():
        item = path / relative
        if not item.is_file() or digest(item) != expected:
            raise RuntimeError("GatherElements extracted-source hash mismatch: {}".format(relative))
    for relative in ("op_host/gather_elements_v2_last_dim_tiling.h", "op_host/gather_elements_v2_tiling.h",
                     "op_host/gather_elements_v2_def.cpp", "op_host/op_api/aclnn_gather.cpp",
                     "op_host/op_api/gather_elements_v2.cpp", "op_kernel/gather_elements_v2.cpp"):
        if not (path / relative).is_file():
            raise RuntimeError("GatherElements extracted source is incomplete: {}".format(relative))


def cmake_adapter() -> str:
    return r'''# CANN-8.1 target-wiring compatibility adapter. No tiling algorithm lives here.
add_ops_compile_options(OP_NAME GatherElementsV2 OPTIONS --cce-auto-sync=on -Wno-deprecated-declarations -Werror)

target_sources(optiling PRIVATE op_host/gather_elements_v2_tiling.cpp)
target_sources(opsproto PRIVATE op_host/gather_elements_v2_def.cpp)
target_sources(op_host_aclnnInner PRIVATE op_host/gather_elements_v2_def.cpp)
target_sources(opapi PRIVATE op_host/op_api/aclnn_gather.cpp op_host/op_api/gather_elements_v2.cpp)

foreach(TARGET_NAME optiling opsproto op_host_aclnnInner)
  target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${CMAKE_SOURCE_DIR}/src/common/inc
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/external
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof)
endforeach()

install(DIRECTORY op_kernel/ DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic FILES_MATCHING PATTERN "*.cpp" PATTERN "*.h")
install(FILES op_host/op_api/aclnn_gather.h DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)
'''


def scoped_root_cmake(source: str) -> str:
    """Pin the official framework to the one requested operator.

    ``cmake/scripts/prepare.sh`` in the public 8.1 framework starts a second
    CMake configure but drops the outer ``ASCEND_OP_NAME`` argument.  Without
    this structural pin that *official* helper enumerates every public
    operator.  The replacement is deliberately before any operator CMake is
    evaluated and changes only source-tree selection; it does not touch the
    GatherElements tiler or a kernel source.
    """
    original = 'set(ASCEND_OP_NAME                "ALL"                           CACHE   STRING   "operators that need to be compiled")'
    replacement = '''# SOURCE_CANDIDATE_BUILD_SCOPE_V1: the public prepare helper omits
# ASCEND_OP_NAME on its nested configure.  Pin this detached build tree to
# GatherElementsV2 so it cannot compile unrelated operators.
set(ASCEND_OP_NAME "gather_elements_v2" CACHE STRING "operators that need to be compiled" FORCE)'''
    if source.count(original) != 1:
        raise RuntimeError("cannot locate public ASCEND_OP_NAME build-selection anchor")
    return source.replace(original, replacement)


def instrument(text: str) -> str:
    if AUDIT in text or CORE in text or UB in text:
        return text
    include = '#include "platform/platform_infos_def.h"\n'
    prelude = r'''#include "platform/platform_infos_def.h"
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// GATHER_ELEMENTS_SOURCE_TILING_AUDIT_V1: post-return observation only.
static uint64_t GatherElementsSourceAuditHash(const uint8_t *data, size_t size) {
    uint64_t value = 1469598103934665603ULL;
    for (size_t i = 0; i < size; ++i) { value ^= data[i]; value *= 1099511628211ULL; }
    return value;
}
static ge::graphStatus GatherElementsSourceAudit(gert::TilingContext *context, ge::graphStatus status) {
    const char *path = std::getenv("GATHER_ELEMENTS_TILING_AUDIT_PATH");
    if (path == nullptr || context == nullptr) { return status; }
    auto *raw = context->GetRawTilingData();
    const size_t size = raw == nullptr ? 0U : raw->GetDataSize();
    const auto *data = raw == nullptr ? nullptr : static_cast<const uint8_t *>(raw->GetData());
    const char *aiv = std::getenv("GATHER_ELEMENTS_SOURCE_AIV_CAP");
    const char *ub = std::getenv("GATHER_ELEMENTS_SOURCE_UB_DIVISOR");
    std::FILE *out = std::fopen(path, "a");
    if (out != nullptr) {
        std::fprintf(out, "{\"schema\":\"gather_elements_raw_tiling_observation_v1\",\"status\":%d,\"aiv_core_cap\":\"%s\",\"ub_cap_divisor\":\"%s\",\"tiling_key\":%llu,\"block_dim\":%u,\"raw_bytes\":%llu,\"raw_fnv1a64\":%llu}\n",
            static_cast<int>(status), aiv == nullptr ? "runtime" : aiv, ub == nullptr ? "1" : ub,
            static_cast<unsigned long long>(context->GetTilingKey()), context->GetBlockDim(),
            static_cast<unsigned long long>(size), static_cast<unsigned long long>(GatherElementsSourceAuditHash(data, size)));
        std::fclose(out);
    }
    return status;
}
'''
    if text.count(include) != 1:
        raise RuntimeError("cannot find GatherElements include anchor")
    text = text.replace(include, prelude)
    pattern = r"ge::graphStatus TilingGatherElementsV2\(gert::TilingContext\* context\)\n\{.*?\n\}\n\nge::graphStatus TilingPrepareForGatherElementsV2"
    match = re.search(pattern, text, flags=re.DOTALL)
    if match is None:
        raise RuntimeError("cannot find GatherElements source dispatcher")
    entry = match.group(0)[:-len("\n\nge::graphStatus TilingPrepareForGatherElementsV2")]
    audited = entry.replace("return ge::GRAPH_FAILED;", "return GatherElementsSourceAudit(context, ge::GRAPH_FAILED);")
    audited = audited.replace("return tilingObject.SetKernelTiling();", "return GatherElementsSourceAudit(context, tilingObject.SetKernelTiling());")
    audited = audited.replace("return ge::GRAPH_SUCCESS;", "return GatherElementsSourceAudit(context, ge::GRAPH_SUCCESS);")
    text = text[:match.start()] + audited + "\n\nge::graphStatus TilingPrepareForGatherElementsV2" + text[match.end():]
    core = "    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();\n    uint64_t ubSizePlatForm;"
    core_replacement = r'''    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    // GATHER_ELEMENTS_SOURCE_AIV_CAP_V1: finite input before original tiling.
    const char *aivText = std::getenv("GATHER_ELEMENTS_SOURCE_AIV_CAP");
    if (aivText != nullptr) {
        errno = 0; char *end = nullptr;
        const unsigned long long cap = std::strtoull(aivText, &end, 10);
        if (errno != 0 || end == aivText || *end != '\0' || cap == 0ULL || cap > compileInfo->totalCoreNum) {
            OP_LOGE(context->GetNodeName(), "invalid GatherElements source AIV cap: %s", aivText);
            return ge::GRAPH_FAILED;
        }
        compileInfo->totalCoreNum = static_cast<uint32_t>(cap);
    }
    uint64_t ubSizePlatForm;'''
    if text.count(core) != 1:
        raise RuntimeError("cannot find GatherElements core-cap anchor")
    text = text.replace(core, core_replacement)
    ub = '''    OP_CHECK_IF((compileInfo->ubSizePlatForm <= 0), OP_LOGE(context->GetNodeName(), "Failed to get ub size"),
                return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "ub_size_platform is %lu", compileInfo->ubSizePlatForm);'''
    ub_replacement = r'''    OP_CHECK_IF((compileInfo->ubSizePlatForm <= 0), OP_LOGE(context->GetNodeName(), "Failed to get ub size"),
                return ge::GRAPH_FAILED);
    // GATHER_ELEMENTS_SOURCE_UB_CAP_V1: only lowers source-visible UB.
    const char *ubText = std::getenv("GATHER_ELEMENTS_SOURCE_UB_DIVISOR");
    if (ubText != nullptr) {
        errno = 0; char *end = nullptr;
        const unsigned long long divisor = std::strtoull(ubText, &end, 10);
        if (errno != 0 || end == ubText || *end != '\0' ||
            (divisor != 1ULL && divisor != 2ULL && divisor != 4ULL && divisor != 8ULL) ||
            static_cast<uint64_t>(compileInfo->ubSizePlatForm) / divisor == 0ULL) {
            OP_LOGE(context->GetNodeName(), "invalid GatherElements source UB divisor: %s", ubText);
            return ge::GRAPH_FAILED;
        }
        compileInfo->ubSizePlatForm /= static_cast<int64_t>(divisor);
    }
    OP_LOGD(context->GetNodeName(), "ub_size_platform is %lu", compileInfo->ubSizePlatForm);'''
    if text.count(ub) != 1:
        raise RuntimeError("cannot find GatherElements UB-cap anchor")
    return text.replace(ub, ub_replacement)


def copied_hashes(extracted: Path) -> dict[str, str]:
    return {str(path.relative_to(extracted)): digest(path) for path in sorted(extracted.rglob("*"))
            if path.is_file() and "tests" not in path.parts and path.name != "run.sh"}


def scope_target_build_config(target_root: Path) -> list[str]:
    """Keep only the extracted Ascend910B build metadata.

    The CANN-8.3 extracted operator carries configuration records for Kirin and
    other SoCs.  The public 8.1 package generator validates every record it
    sees and rejects those unrelated SoC strings before it can build the 910B
    target.  This limits *build metadata* to the selected target; it neither
    edits the tiler nor changes any device kernel source.
    """
    config_root = target_root / "op_host/config"
    keep = config_root / "ascend910b"
    if not keep.is_dir():
        raise RuntimeError("extracted GatherElements source lacks its Ascend910B config")
    removed: list[str] = []
    for child in sorted(config_root.iterdir()):
        if child == keep:
            continue
        if not child.is_dir():
            raise RuntimeError("unexpected non-directory GatherElements config entry: {}".format(child))
        shutil.rmtree(child)
        removed.append(child.name)
    return removed


def narrow_op_definition_to_910b(source: str) -> str:
    """Keep the original Ascend910B OpDef declaration and discard others.

    The 8.1 registration generator rejects the newer Kirin SoC names before
    it can build the matching 910B record.  This affects only registration
    metadata: the tiler, its search, and every device-kernel source are left
    byte-for-byte untouched.
    """
    original = '''        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");

        OpAICoreConfig config_kirin = GetKirinCoreConfig();
        this->AICore().AddConfig("kirinx90", config_kirin);
        this->AICore().AddConfig("kirin9030", config_kirin);
    }

private:
    OpAICoreConfig GetKirinCoreConfig() const
    {
        OpAICoreConfig config_kirin;
        config_kirin.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        config_kirin.Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        config_kirin.Input("index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        config_kirin.Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        return config_kirin;
    }
'''
    replacement = '''        // CANN-8.1 compatibility: preserve only the source's original
        // Ascend910B registration. Tiling and device code are unchanged.
        this->AICore().AddConfig("ascend910b");
    }
'''
    if source.count(original) != 1:
        raise RuntimeError("cannot locate GatherElements non-910B OpDef configuration block")
    return source.replace(original, replacement)


def logging_header_shim() -> str:
    """Compatibility declaration for the CANN-8.3 tiler's removed log header.

    CANN-8.1 has no ``log/log.h`` public header, while this extracted tiler
    uses it only for diagnostics and ``OP_CHECK_IF``'s existing early-return
    control macro.  This local header deliberately preserves that control
    macro and makes logging a no-op; it supplies no tiling data or decision.
    """
    return '''#pragma once
// CANN-8.1 compatibility shim for the CANN-8.3 diagnostic header.
// It changes logging only; it does not change a tiling calculation.
#ifndef OP_LOGD
#define OP_LOGD(...) do { } while (0)
#endif
#ifndef OP_LOGE
#define OP_LOGE(...) do { } while (0)
#endif
#ifndef OP_CHECK_IF
#define OP_CHECK_IF(condition, log_statement, return_statement) \\
    do { if (condition) { log_statement; return_statement; } } while (0)
#endif
#ifndef OP_CHECK_NULL_WITH_CONTEXT
#define OP_CHECK_NULL_WITH_CONTEXT(context, pointer) \\
    do { if ((pointer) == nullptr) { OP_LOGE("GatherElementsV2", "null pointer: %s", #pointer); return ge::GRAPH_FAILED; } } while (0)
#endif
'''


def arithmetic_header_shim() -> str:
    """CANN-8.1 header compatibility for the three arithmetic helpers used.

    The extracted CANN-8.3 header names these helpers ``Ops::Base`` while the
    8.1 implementation exposes the same ceil/floor arithmetic under another
    internal header that drags an unrelated JSON dependency.  This header is
    limited to the three functions actually called by the untouched Gather
    tiler, with the same zero-divisor semantics as public CANN-8.1 ``op_util``.
    """
    return '''#pragma once
// CANN-8.1 compatibility shim: exact integer ceil/floor helpers only.
#include <type_traits>
namespace Ops { namespace Base {
template <typename T>
inline typename std::enable_if<std::is_signed<T>::value, T>::type CeilDiv(T x, T y) {
    if (y != 0 && x != 0) { const T q = x / y; return (x % y != 0 && ((x ^ y) >= 0)) ? q + 1 : q; }
    return x;
}
template <typename T>
inline typename std::enable_if<std::is_unsigned<T>::value, T>::type CeilDiv(T x, T y) {
    if (y != 0 && x != 0) { const T q = x / y; return x % y != 0 ? q + 1 : q; }
    return x;
}
template <typename T>
inline typename std::enable_if<std::is_integral<T>::value, T>::type FloorDiv(T x, T y) {
    return y == 0 ? x : x / y;
}
template <typename T>
inline typename std::enable_if<std::is_integral<T>::value, T>::type CeilAlign(T x, T align) {
    return CeilDiv(x, align) * align;
}
} }  // namespace Ops::Base
'''


def expected(extracted: Path) -> str:
    return instrument((extracted / "op_host/gather_elements_v2_tiling.cpp").read_text(encoding="utf-8"))


def existing(extracted: Path, output: Path) -> dict[str, Any] | None:
    if not output.exists():
        return None
    manifest_path = output / "source_candidate_overlay.json"
    if not manifest_path.is_file():
        raise RuntimeError("existing GatherElements overlay lacks manifest")
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("operator") != "GatherElementsV2" or manifest.get("official_commit") != PARENT["commit"] or
            manifest.get("strategy_algorithm_changes") is not False or
            manifest.get("source_hardware_envelope_heuristic_enumeration") is not True):
        raise RuntimeError("existing GatherElements overlay provenance differs")
    if (output / ENTRY).read_text(encoding="utf-8") != expected(extracted):
        raise RuntimeError("existing GatherElements tiling changes exceed contract")
    definition = output / OVERLAY_OP_ROOT / "op_host/gather_elements_v2_def.cpp"
    if definition.read_text(encoding="utf-8") != narrow_op_definition_to_910b(
            (extracted / "op_host/gather_elements_v2_def.cpp").read_text(encoding="utf-8")):
        raise RuntimeError("existing GatherElements non-910B registration scope differs")
    if (output / OVERLAY_OP_ROOT / "CMakeLists.txt").read_text(encoding="utf-8") != cmake_adapter():
        raise RuntimeError("existing GatherElements CMake adapter differs")
    config_root = output / OVERLAY_OP_ROOT / "op_host/config"
    if sorted(path.name for path in config_root.iterdir()) != ["ascend910b"]:
        raise RuntimeError("existing GatherElements build-config scope differs")
    shim = output / OVERLAY_OP_ROOT / "op_host/log/log.h"
    if not shim.is_file() or shim.read_text(encoding="utf-8") != logging_header_shim():
        raise RuntimeError("existing GatherElements logging-header compatibility differs")
    arithmetic = output / OVERLAY_OP_ROOT / "op_host/util/math_util.h"
    if not arithmetic.is_file() or arithmetic.read_text(encoding="utf-8") != arithmetic_header_shim():
        raise RuntimeError("existing GatherElements arithmetic-header compatibility differs")
    parent_root = Path(PARENT["local_cache_hint"])
    if (output / "CMakeLists.txt").read_text(encoding="utf-8") != scoped_root_cmake(
            (parent_root / "CMakeLists.txt").read_text(encoding="utf-8")):
        raise RuntimeError("existing GatherElements root build scope differs")
    manifest["resumed_existing_overlay"] = True
    return manifest


def prepare(parent: Path, extracted: Path, output_parent: Path) -> dict[str, Any]:
    output = output_parent / "gather_elements_v2_compat_source"
    resume = existing(extracted, output)
    if resume is not None:
        return resume
    run(["git", "-C", str(parent), "worktree", "add", "--detach", str(output), PARENT["commit"]])
    root_cmake = output / "CMakeLists.txt"
    root_cmake.write_text(scoped_root_cmake(root_cmake.read_text(encoding="utf-8")), encoding="utf-8")
    target_root = output / OVERLAY_OP_ROOT
    if target_root.exists():
        raise RuntimeError("CANN-8.1 parent unexpectedly already contains GatherElementsV2")
    shutil.copytree(extracted, target_root, ignore=shutil.ignore_patterns("tests", "run.sh", "*.pyc", "__pycache__"))
    removed_configs = scope_target_build_config(target_root)
    definition = target_root / "op_host/gather_elements_v2_def.cpp"
    definition_before = definition.read_text(encoding="utf-8")
    definition.write_text(narrow_op_definition_to_910b(definition_before), encoding="utf-8")
    shim = target_root / "op_host/log/log.h"
    shim.parent.mkdir(parents=True, exist_ok=False)
    shim.write_text(logging_header_shim(), encoding="utf-8")
    arithmetic = target_root / "op_host/util/math_util.h"
    arithmetic.parent.mkdir(parents=True, exist_ok=False)
    arithmetic.write_text(arithmetic_header_shim(), encoding="utf-8")
    for path in (target_root / "op_host/CMakeLists.txt", target_root / "op_graph/CMakeLists.txt"):
        if path.exists():
            path.unlink()
    (target_root / "CMakeLists.txt").write_text(cmake_adapter(), encoding="utf-8")
    target = output / ENTRY
    before = target.read_text(encoding="utf-8")
    target.write_text(instrument(before), encoding="utf-8")
    manifest: dict[str, Any] = {
        "schema": "gather_elements_compat_original_tiler_overlay_v1", "operator": "GatherElementsV2",
        "source_family": "cann_ops", "cmake_op_name": "gather_elements_v2", "official_url": PARENT["url"],
        "official_commit": PARENT["commit"], "overlay": str(output), "audit_entry_relative": str(ENTRY), "audit_sentinel": AUDIT,
        "modified_source_files": [
            {"path": str(ENTRY), "sha256_before": hashlib.sha256(before.encode()).hexdigest(), "sha256_after": digest(target), "kind": "observational_audit_and_source_resource_inputs"},
            {"path": str(OVERLAY_OP_ROOT / "op_host/gather_elements_v2_def.cpp"), "sha256_before": hashlib.sha256(definition_before.encode()).hexdigest(), "sha256_after": digest(definition), "kind": "Ascend910B_registration_scope_only"},
            {"path": str(OVERLAY_OP_ROOT / "op_host/log/log.h"), "sha256_before": None, "sha256_after": digest(shim), "kind": "CANN_8_1_missing_logging_header_compatibility"},
            {"path": str(OVERLAY_OP_ROOT / "op_host/util/math_util.h"), "sha256_before": None, "sha256_after": digest(arithmetic), "kind": "CANN_8_1_missing_arithmetic_header_compatibility"},
        ],
        "compatibility_port": {"source_version": EXTRACTED["source_version"], "extracted_root": str(extracted),
            "extracted_files_sha256": copied_hashes(extracted), "allowed_changes": ["CANN-8.1 CMake target wiring", "Ascend910B-only build-config scope", "CANN-8.1 missing logging-header shim", "CANN-8.1 missing arithmetic-header shim", "observational source audit", "source AIV input", "source UB capacity input"],
            "tiling_algorithm_changes": False, "kernel_algorithm_changes": False,
            "non_tiling_registration_change": "remove source declarations for ascend910_93/kirinx90/kirin9030 while preserving its original ascend910b OpDef declaration",
            "formal_data_gate": "package build plus exact real-NPU output equality against installed operator",
            "target_build_config_scope": {"kept": ["ascend910b"], "removed_unrelated": removed_configs}},
        "build_scope": {"sentinel": BUILD_SCOPE, "operator": "gather_elements_v2",
                        "reason": "official nested prepare helper otherwise drops ASCEND_OP_NAME and compiles unrelated operators",
                        "structural_only": True},
        "instrumentation": {"enabled": True, "audit_schema": "gather_elements_raw_tiling_observation_v1",
            "audit_environment": "GATHER_ELEMENTS_TILING_AUDIT_PATH", "source_budget_environment": "GATHER_ELEMENTS_SOURCE_AIV_CAP", "mutates_tiling_context": False},
        "strategy_algorithm_changes": False, "source_compile_info_core_budget_enumeration": True,
        "source_hardware_envelope_heuristic_enumeration": True,
        "hardware_envelope_heuristic": {"enabled": True, "environment": "GATHER_ELEMENTS_SOURCE_UB_DIVISOR", "audit_field": "ub_cap_divisor", "resource": "source_visible_ub_capacity", "divisors": [2, 4, 8], "max_anchors": 16},
        "candidate_rule": "complete original source AIV-cap contexts first; only below twenty identities, selected original contexts with lower source-visible UB capacities; exact real output validation required",
        "forbidden": LOCK["collection_contract"]["forbidden"],
    }
    (output / "source_candidate_overlay.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-source-root", required=True, type=Path)
    parser.add_argument("--extracted-source-root", required=True, type=Path)
    parser.add_argument("--output-parent", required=True, type=Path)
    args = parser.parse_args()
    if not args.output_parent.is_dir():
        raise RuntimeError("output parent does not exist: {}".format(args.output_parent))
    require_parent(args.parent_source_root)
    require_extracted(args.extracted_source_root)
    print(json.dumps({"schema": "gather_elements_compat_overlay_batch_v1", "matmul_included": False,
                      "overlay": prepare(args.parent_source_root, args.extracted_source_root, args.output_parent)}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print("GatherElements source-candidate error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
