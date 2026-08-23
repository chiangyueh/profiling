#!/usr/bin/env python3
"""Prepare a source-rule FusedInferAttentionScore candidate overlay.

The overlay preserves the operator's original semantic dispatcher: decode
continues through IncreFlashAttention and prefill continues through
PromptFlashAttention.  It does not force either branch onto shapes that belong
to the other branch.  The only candidate input is a finite, runtime-bounded
AIV/AIC budget supplied before each branch's original tiling calculation.  No
raw tiling field, tiling key, workspace size, or kernel code is edited.

The public advanced source checkout has no root CMakeLists.txt.  We therefore
attach the same attested public 8.1 cann-ops build harness used by the FASG
overlay; its only structural adjustment is ``src/common`` -> ``src/utils``.
This script creates a worktree and provenance manifest only.  It does not
compile, launch, reset, or otherwise call an NPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from prepare_fasg_strategy_overlays import (digest, materialize_build_support_files,
                                            require_repo_source_bundle, run, source_build_harness)


ROOT = Path(__file__).resolve().parent
LOCK = json.loads((ROOT / "non_matmul_source_lock.json").read_text(encoding="utf-8"))
SOURCE = LOCK["sources"]["cann_ops_adv"]
OP = LOCK["operators"]["fused_infer_attention_score"]
OP_ROOT = Path(OP["relative_root"])
FIAS_RELATIVE = OP_ROOT / "ophost/fused_infer_attention_score_tiling.cpp"
IFA_RELATIVE = Path("src/transformer/incre_flash_attention/ophost/incre_flash_attention_tiling.cc")
AUDIT_SENTINEL = "FIAS_SOURCE_TILING_AUDIT_V1"
CORE_SENTINEL = "FIAS_SOURCE_AIV_CAP_V1"
UB_SENTINEL = "FIAS_SOURCE_UB_CAP_V1"


def require_pinned_source(source_root: Path) -> None:
    require_repo_source_bundle(source_root, "cann_ops_adv", SOURCE["commit"])
    for relative, expected in OP["pinned_files"].items():
        actual_path = source_root / OP_ROOT / relative
        if not actual_path.is_file() or digest(actual_path) != expected:
            raise RuntimeError("pinned source hash mismatch: {}".format(relative))


def instrument_fias_tiler(source: str) -> str:
    """Attach audit plus PFA compile-info core budget before original tiling."""
    if AUDIT_SENTINEL in source or CORE_SENTINEL in source or UB_SENTINEL in source:
        return source
    include_anchor = '#include "platform/platform_info.h"\n'
    audit = r'''#include "platform/platform_info.h"
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
// FIAS_SOURCE_TILING_AUDIT_V1: observational only; no tiling state is modified.
static uint64_t FIASSourceAuditHash(const uint8_t *data, size_t size)
{
    uint64_t value = 1469598103934665603ULL;
    for (size_t index = 0; index < size; ++index) { value ^= data[index]; value *= 1099511628211ULL; }
    return value;
}

static ge::graphStatus FIASSourceAuditResult(gert::TilingContext *context, ge::graphStatus status)
{
    const char *path = std::getenv("FIAS_TILING_AUDIT_PATH");
    if (path == nullptr || context == nullptr) { return status; }
    auto *raw = context->GetRawTilingData();
    const size_t rawSize = raw == nullptr ? 0U : raw->GetDataSize();
    const auto *rawBytes = raw == nullptr ? nullptr : static_cast<const uint8_t *>(raw->GetData());
    const uint64_t digest = rawBytes == nullptr ? 0ULL : FIASSourceAuditHash(rawBytes, rawSize);
    const char *coreCap = std::getenv("FIAS_SOURCE_AIV_CAP");
    std::FILE *output = std::fopen(path, "a");
    if (output != nullptr) {
        const char *ubDivisor = std::getenv("FIAS_SOURCE_UB_DIVISOR");
        std::fprintf(output, "{\"schema\":\"fias_raw_tiling_observation_v1\",\"status\":%d,\"aiv_core_cap\":\"%s\",\"ub_cap_divisor\":\"%s\",\"tiling_key\":%llu,\"block_dim\":%u,\"raw_bytes\":%llu,\"raw_fnv1a64\":%llu}\n",
                     static_cast<int>(status), coreCap == nullptr ? "runtime" : coreCap, ubDivisor == nullptr ? "1" : ubDivisor,
                     static_cast<unsigned long long>(context->GetTilingKey()), context->GetBlockDim(),
                     static_cast<unsigned long long>(rawSize), static_cast<unsigned long long>(digest));
        std::fclose(output);
    }
    return status;
}
'''
    if source.count(include_anchor) != 1:
        raise RuntimeError("cannot locate FIAS include anchor")
    source = source.replace(include_anchor, audit)
    assignment = '''        tempCompileInfoPtr.aivNum = ascendcPlatform.GetCoreNumAiv();
        tempCompileInfoPtr.aicNum = ascendcPlatform.GetCoreNumAic();'''
    replacement = r'''        tempCompileInfoPtr.aivNum = ascendcPlatform.GetCoreNumAiv();
        tempCompileInfoPtr.aicNum = ascendcPlatform.GetCoreNumAic();
        // FIAS_SOURCE_AIV_CAP_V1: a finite source-input candidate axis. The
        // original PromptFlashAttention calculation owns all tiling fields.
        const char *requestedAivText = std::getenv("FIAS_SOURCE_AIV_CAP");
        if (requestedAivText != nullptr) {
            errno = 0;
            char *parseEnd = nullptr;
            const unsigned long long requestedAiv = std::strtoull(requestedAivText, &parseEnd, 10);
            const uint64_t runtimeAiv = tempCompileInfoPtr.aivNum;
            const uint64_t runtimeAic = tempCompileInfoPtr.aicNum;
            if (errno != 0 || parseEnd == requestedAivText || *parseEnd != '\0' || requestedAiv == 0ULL ||
                requestedAiv > runtimeAiv || runtimeAiv == 0ULL || runtimeAic == 0ULL || runtimeAic % runtimeAiv != 0ULL) {
                OPS_LOG_E(context->GetNodeName(), "invalid FIAS source AIV cap: %s", requestedAivText);
                return ge::GRAPH_FAILED;
            }
            const uint64_t aicPerAiv = runtimeAic / runtimeAiv;
            tempCompileInfoPtr.aivNum = static_cast<uint32_t>(requestedAiv);
            tempCompileInfoPtr.aicNum = static_cast<uint32_t>(requestedAiv * aicPerAiv);
        }'''
    if source.count(assignment) != 1:
        raise RuntimeError("cannot locate FIAS PFA compile-info core assignment")
    source = source.replace(assignment, replacement)
    pfa_ub_anchor = '''        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, tempCompileInfoPtr.l0BSize);
        tempCompileInfoPtr.socShortName = ascendcPlatform.GetSocVersion();'''
    pfa_ub_replacement = r'''        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, tempCompileInfoPtr.l0BSize);
        // FIAS_SOURCE_UB_CAP_V1: optional bounded hardware-envelope input.
        // It only lowers the capacity shown to the original PFA tiler; all
        // resulting allocations therefore fit the physical UB.
        const char *ubDivisorText = std::getenv("FIAS_SOURCE_UB_DIVISOR");
        if (ubDivisorText != nullptr) {
            errno = 0;
            char *parseEnd = nullptr;
            const unsigned long long divisor = std::strtoull(ubDivisorText, &parseEnd, 10);
            if (errno != 0 || parseEnd == ubDivisorText || *parseEnd != '\0' ||
                (divisor != 1ULL && divisor != 2ULL && divisor != 4ULL && divisor != 8ULL) ||
                tempCompileInfoPtr.ubSize / divisor == 0ULL) {
                OPS_LOG_E(context->GetNodeName(), "invalid FIAS source UB divisor: %s", ubDivisorText);
                return ge::GRAPH_FAILED;
            }
            tempCompileInfoPtr.ubSize /= divisor;
        }
        tempCompileInfoPtr.socShortName = ascendcPlatform.GetSocVersion();'''
    if source.count(pfa_ub_anchor) != 1:
        raise RuntimeError("cannot locate FIAS PFA UB capacity assignment")
    source = source.replace(pfa_ub_anchor, pfa_ub_replacement)
    original_wrapper = '''__attribute__((visibility("default"))) ge::graphStatus
DeviceDoOpTilingFusedInferAttentionScore(gert::TilingContext *context)
{
    return DoOpTilingFusedInferAttentionScore(context);
}'''
    replacement_wrapper = '''__attribute__((visibility("default"))) ge::graphStatus
DeviceDoOpTilingFusedInferAttentionScore(gert::TilingContext *context)
{
    return FIASSourceAuditResult(context, DoOpTilingFusedInferAttentionScore(context));
}'''
    if source.count(original_wrapper) != 1:
        raise RuntimeError("cannot locate FIAS source tiling entry for audit")
    original_incre_wrapper = '''__attribute__((visibility("default"))) ge::graphStatus DeviceDoOpTilingIncreFlashAttention(gert::TilingContext *context)
{
    return TilingIncreFlashAttention(context);
}'''
    replacement_incre_wrapper = '''__attribute__((visibility("default"))) ge::graphStatus DeviceDoOpTilingIncreFlashAttention(gert::TilingContext *context)
{
    return FIASSourceAuditResult(context, TilingIncreFlashAttention(context));
}'''
    if source.count(original_incre_wrapper) != 1:
        raise RuntimeError("cannot locate FIAS decode source tiling entry for audit")
    return source.replace(original_wrapper, replacement_wrapper).replace(original_incre_wrapper, replacement_incre_wrapper)


def instrument_ifa_tiler(source: str) -> str:
    """Apply the same bounded source core input to the original decode path."""
    if CORE_SENTINEL in source or UB_SENTINEL in source:
        return source
    include_anchor = '#include "incre_flash_attention_tiling.h"\n'
    if source.count(include_anchor) != 1:
        raise RuntimeError("cannot locate IFA include anchor")
    source = source.replace(include_anchor, include_anchor + "#include <cerrno>\n#include <cstdlib>\n")
    assignment = '''    aicNum_ = ascendcPlatform.GetCoreNumAic();
    aivNum_ = ascendcPlatform.GetCoreNumAiv();'''
    replacement = r'''    aicNum_ = ascendcPlatform.GetCoreNumAic();
    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    // FIAS_SOURCE_AIV_CAP_V1: the original IncreFlashAttention calculation
    // receives a smaller, runtime-bounded core budget before it computes its
    // own split and block dimension. No output field is rewritten afterwards.
    const char *requestedAivText = std::getenv("FIAS_SOURCE_AIV_CAP");
    if (requestedAivText != nullptr) {
        errno = 0;
        char *parseEnd = nullptr;
        const unsigned long long requestedAiv = std::strtoull(requestedAivText, &parseEnd, 10);
        const uint64_t runtimeAiv = aivNum_;
        const uint64_t runtimeAic = aicNum_;
        if (errno != 0 || parseEnd == requestedAivText || *parseEnd != '\0' || requestedAiv == 0ULL ||
            requestedAiv > runtimeAiv || runtimeAiv == 0ULL || runtimeAic == 0ULL || runtimeAic % runtimeAiv != 0ULL) {
            OPS_LOG_E(context_->opName, "invalid FIAS source AIV cap: %s", requestedAivText);
            return ge::GRAPH_FAILED;
        }
        const uint64_t aicPerAiv = runtimeAic / runtimeAiv;
        aivNum_ = static_cast<uint32_t>(requestedAiv);
        aicNum_ = static_cast<uint32_t>(requestedAiv * aicPerAiv);
    }
    // FIAS_SOURCE_UB_CAP_V1: same bounded source-visible UB rule as the
    // original prefill route. The decode tiler owns every resulting field.
    const char *ubDivisorText = std::getenv("FIAS_SOURCE_UB_DIVISOR");
    if (ubDivisorText != nullptr) {
        errno = 0;
        char *parseEnd = nullptr;
        const unsigned long long divisor = std::strtoull(ubDivisorText, &parseEnd, 10);
        if (errno != 0 || parseEnd == ubDivisorText || *parseEnd != '\0' ||
            (divisor != 1ULL && divisor != 2ULL && divisor != 4ULL && divisor != 8ULL) || ubSize_ / divisor == 0ULL) {
            OPS_LOG_E(context_->opName, "invalid FIAS source UB divisor: %s", ubDivisorText);
            return ge::GRAPH_FAILED;
        }
        ubSize_ /= divisor;
    }'''
    if source.count(assignment) != 1:
        raise RuntimeError("cannot locate IFA core assignment")
    return source.replace(assignment, replacement)


def existing_overlay(source_root: Path, output: Path, harness_text: str,
                     harness_provenance: dict[str, str]) -> dict[str, Any] | None:
    if not output.exists():
        return None
    manifest_path = output / "source_candidate_overlay.json"
    if not manifest_path.is_file():
        raise RuntimeError("existing FIAS overlay lacks provenance manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "operator": "FusedInferAttentionScore", "official_commit": SOURCE["commit"],
        "source_family": "cann_ops_adv", "cmake_op_name": "fused_infer_attention_score",
        "audit_entry_relative": str(FIAS_RELATIVE), "audit_sentinel": AUDIT_SENTINEL,
        "strategy_algorithm_changes": False, "source_compile_info_core_budget_enumeration": True,
        "source_hardware_envelope_heuristic_enumeration": True,
        "hardware_envelope_heuristic": {"enabled": True, "environment": "FIAS_SOURCE_UB_DIVISOR",
                                         "audit_field": "ub_cap_divisor", "resource": "source_visible_ub_capacity",
                                         "divisors": [2, 4, 8], "max_anchors": 16}, "build_harness": harness_provenance,
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise RuntimeError("existing FIAS overlay provenance does not match the requested source contract")
    expected_files = {
        FIAS_RELATIVE: instrument_fias_tiler((source_root / FIAS_RELATIVE).read_text(encoding="utf-8")),
        IFA_RELATIVE: instrument_ifa_tiler((source_root / IFA_RELATIVE).read_text(encoding="utf-8")),
    }
    for relative, expected in expected_files.items():
        actual = (output / relative).read_text(encoding="utf-8")
        if actual != expected:
            raise RuntimeError("existing FIAS overlay changes source outside the allowed contract: {}".format(relative))
    if (output / "CMakeLists.txt").read_text(encoding="utf-8") != harness_text:
        raise RuntimeError("existing FIAS build harness differs from the pinned allowed transformation")
    materialize_build_support_files(output, None, harness_provenance)
    manifest["resumed_existing_overlay"] = True
    return manifest


def write_overlay(source_root: Path, output_parent: Path, harness_root: Path, harness_text: str,
                  harness_provenance: dict[str, str]) -> dict[str, Any]:
    output = output_parent / "fias_source_dispatch"
    existing = existing_overlay(source_root, output, harness_text, harness_provenance)
    if existing is not None:
        return existing
    run(["git", "-C", str(source_root), "worktree", "add", "--detach", str(output), "HEAD"])
    changed: list[dict[str, str]] = []
    transforms = {
        FIAS_RELATIVE: instrument_fias_tiler,
        IFA_RELATIVE: instrument_ifa_tiler,
    }
    for relative, transform in transforms.items():
        target = output / relative
        before = target.read_text(encoding="utf-8")
        after = transform(before)
        target.write_text(after, encoding="utf-8")
        changed.append({
            "path": str(relative), "sha256_before": hashlib.sha256(before.encode("utf-8")).hexdigest(),
            "sha256_after": digest(target),
        })
    (output / "CMakeLists.txt").write_text(harness_text, encoding="utf-8")
    materialize_build_support_files(output, harness_root, harness_provenance)
    manifest: dict[str, Any] = {
        "schema": "fias_original_dispatch_overlay_v1",
        "operator": "FusedInferAttentionScore",
        "source_family": "cann_ops_adv",
        "cmake_op_name": "fused_infer_attention_score",
        "audit_entry_relative": str(FIAS_RELATIVE),
        "audit_sentinel": AUDIT_SENTINEL,
        "official_url": SOURCE["url"], "official_tag": SOURCE["tag"], "official_commit": SOURCE["commit"],
        "overlay": str(output), "modified_source_files": changed,
        "source_dispatch": {
            "decode": "original IncreFlashAttention predicate/path",
            "prefill": "original PromptFlashAttention predicate/path",
            "forced_branch": False,
        },
        "instrumentation": {
            "enabled": True, "audit_schema": "fias_raw_tiling_observation_v1",
            "audit_environment": "FIAS_TILING_AUDIT_PATH",
            "source_budget_environment": "FIAS_SOURCE_AIV_CAP",
            "mutates_tiling_context": False,
        },
        "strategy_algorithm_changes": False,
        "source_compile_info_core_budget_enumeration": True,
        "source_hardware_envelope_heuristic_enumeration": True,
        "hardware_envelope_heuristic": {"enabled": True, "environment": "FIAS_SOURCE_UB_DIVISOR",
                                         "audit_field": "ub_cap_divisor", "resource": "source_visible_ub_capacity",
                                         "divisors": [2, 4, 8], "max_anchors": 16},
        "build_scope": harness_provenance["build_scope"],
        "candidate_rule": "run the original semantic dispatcher for all finite runtime-derived AIV/AIC source budgets; only if those complete original contexts expose fewer than twenty raw identities, rerun selected original contexts with source-visible UB divisors 2/4/8; retain only exact raw identities that later pass real output validation",
        "build_harness": harness_provenance,
        "forbidden": LOCK["collection_contract"]["forbidden"],
    }
    (output / "source_candidate_overlay.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--harness-root", required=True, type=Path)
    parser.add_argument("--output-parent", required=True, type=Path)
    args = parser.parse_args()
    if not args.output_parent.is_dir():
        raise RuntimeError("output parent does not exist: {}".format(args.output_parent))
    require_pinned_source(args.source_root)
    harness_text, harness_provenance = source_build_harness(args.harness_root, "fused_infer_attention_score")
    output = write_overlay(args.source_root, args.output_parent, args.harness_root, harness_text, harness_provenance)
    print(json.dumps({"schema": "fias_original_dispatch_overlay_batch_v1", "matmul_included": False,
                      "overlay": output}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print("FIAS source-candidate error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
