#!/usr/bin/env python3
"""Prepare a source-rule ScatterElementsV2 candidate overlay.

The pinned public source has a single real 910B tiler and a documented
last-axis predicate.  This overlay never tries to make other axes valid.  It
only supplies a finite, runtime-bounded AIV core budget *before* the original
``Init`` calculation decides its own work split, then audits the raw output
after the original tiler returns.  No tile field, block dimension, tiling key,
workspace, or kernel source is rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
LOCK = json.loads((ROOT / "non_matmul_source_lock.json").read_text(encoding="utf-8"))
SOURCE = LOCK["sources"]["cann_ops"]
OP = LOCK["operators"]["scatter_elements_v2"]
RELATIVE = Path(OP["relative_root"]) / "op_host/scatter_elements_v2_tiling.cc"
AUDIT_SENTINEL = "SCATTER_ELEMENTS_SOURCE_TILING_AUDIT_V1"
CORE_SENTINEL = "SCATTER_ELEMENTS_SOURCE_AIV_CAP_V1"
BUILD_SCOPE = "SOURCE_CANDIDATE_BUILD_SCOPE_V1"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def run(argv: list[str]) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError("command failed rc={} argv={} stderr={}".format(
            result.returncode, " ".join(argv), result.stderr.strip()))
    return result.stdout.strip()


def require_pinned_source(source_root: Path) -> None:
    if run(["git", "-C", str(source_root), "rev-parse", "HEAD"]) != SOURCE["commit"]:
        raise RuntimeError("source revision mismatch")
    if run(["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=no"]):
        raise RuntimeError("source worktree is modified")
    for relative, expected in OP["pinned_files"].items():
        path = source_root / OP["relative_root"] / relative
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError("pinned source hash mismatch: {}".format(relative))
    if not (source_root / "CMakeLists.txt").is_file():
        raise RuntimeError("pinned public cann-ops source lacks its root build file")


def instrument(source: str) -> str:
    if AUDIT_SENTINEL in source or CORE_SENTINEL in source:
        return source
    include_anchor = '#include "scatter_elements_v2_tiling.h"\n'
    includes = '''#include "scatter_elements_v2_tiling.h"
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
'''
    if source.count(include_anchor) != 1:
        raise RuntimeError("cannot locate ScatterElements include anchor")
    source = source.replace(include_anchor, includes)
    core_assignment = '''    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) {'''
    core_replacement = r'''    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    // SCATTER_ELEMENTS_SOURCE_AIV_CAP_V1: finite source input only. The
    // original tiler calculates every resulting split and output field.
    const char *requestedAivText = std::getenv("SCATTER_ELEMENTS_SOURCE_AIV_CAP");
    if (requestedAivText != nullptr) {
        errno = 0;
        char *parseEnd = nullptr;
        const unsigned long long requestedAiv = std::strtoull(requestedAivText, &parseEnd, 10);
        if (errno != 0 || parseEnd == requestedAivText || *parseEnd != '\0' || requestedAiv == 0ULL ||
            requestedAiv > coreNum) {
            OP_LOGE(tilingContext->GetNodeName(), "invalid ScatterElements source AIV cap: %s.", requestedAivText);
            return ge::GRAPH_FAILED;
        }
        coreNum = static_cast<uint32_t>(requestedAiv);
    }
    if (coreNum == 0) {'''
    if source.count(core_assignment) != 1:
        raise RuntimeError("cannot locate ScatterElements original core assignment")
    source = source.replace(core_assignment, core_replacement)
    ub_assignment = '''    max_ub = ubSizePlatForm / max_ub * max_ub / BUFFER_NUM;
    OP_LOGD(tilingContext->GetNodeName(), "ubSizePlatForm: %lu.", ubSizePlatForm);'''
    ub_replacement = r'''    max_ub = ubSizePlatForm / max_ub * max_ub / BUFFER_NUM;
    // SCATTER_ELEMENTS_SOURCE_UB_CAP_V1: optional local hardware-envelope
    // expansion. It only lowers source-visible UB, so every original formula
    // result remains within the actual physical UB allocation.
    const char *ubDivisorText = std::getenv("SCATTER_ELEMENTS_SOURCE_UB_DIVISOR");
    if (ubDivisorText != nullptr) {
        errno = 0;
        char *parseEnd = nullptr;
        const unsigned long long divisor = std::strtoull(ubDivisorText, &parseEnd, 10);
        if (errno != 0 || parseEnd == ubDivisorText || *parseEnd != '\0' ||
            (divisor != 1ULL && divisor != 2ULL && divisor != 4ULL && divisor != 8ULL) || max_ub / divisor == 0ULL) {
            OP_LOGE(tilingContext->GetNodeName(), "invalid ScatterElements source UB divisor: %s.", ubDivisorText);
            return ge::GRAPH_FAILED;
        }
        max_ub /= divisor;
    }
    OP_LOGD(tilingContext->GetNodeName(), "ubSizePlatForm: %lu.", ubSizePlatForm);'''
    if source.count(ub_assignment) != 1:
        raise RuntimeError("cannot locate ScatterElements original UB capacity assignment")
    source = source.replace(ub_assignment, ub_replacement)
    entry = '''  ge::graphStatus TilingScatterElementsV2(gert::TilingContext* context) {
    ScatterElementsV2Tiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunKernelTiling();
  }'''
    audit_entry = r'''  // SCATTER_ELEMENTS_SOURCE_TILING_AUDIT_V1: observational post-return audit.
  static uint64_t ScatterElementsSourceAuditHash(const uint8_t *data, size_t size) {
    uint64_t value = 1469598103934665603ULL;
    for (size_t index = 0; index < size; ++index) { value ^= data[index]; value *= 1099511628211ULL; }
    return value;
  }
  static ge::graphStatus ScatterElementsSourceAudit(gert::TilingContext *context, ge::graphStatus status) {
    const char *path = std::getenv("SCATTER_ELEMENTS_TILING_AUDIT_PATH");
    if (path == nullptr || context == nullptr) { return status; }
    auto *raw = context->GetRawTilingData();
    const size_t rawSize = raw == nullptr ? 0U : raw->GetDataSize();
    const auto *rawBytes = raw == nullptr ? nullptr : static_cast<const uint8_t *>(raw->GetData());
    const char *coreCap = std::getenv("SCATTER_ELEMENTS_SOURCE_AIV_CAP");
    std::FILE *output = std::fopen(path, "a");
    if (output != nullptr) {
      const char *ubDivisor = std::getenv("SCATTER_ELEMENTS_SOURCE_UB_DIVISOR");
      std::fprintf(output, "{\"schema\":\"scatter_elements_raw_tiling_observation_v1\",\"status\":%d,\"aiv_core_cap\":\"%s\",\"ub_cap_divisor\":\"%s\",\"tiling_key\":%llu,\"block_dim\":%u,\"raw_bytes\":%llu,\"raw_fnv1a64\":%llu}\n",
          static_cast<int>(status), coreCap == nullptr ? "runtime" : coreCap, ubDivisor == nullptr ? "1" : ubDivisor,
          static_cast<unsigned long long>(context->GetTilingKey()), context->GetBlockDim(),
          static_cast<unsigned long long>(rawSize), static_cast<unsigned long long>(ScatterElementsSourceAuditHash(rawBytes, rawSize)));
      std::fclose(output);
    }
    return status;
  }
  ge::graphStatus TilingScatterElementsV2(gert::TilingContext* context) {
    ScatterElementsV2Tiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        return ScatterElementsSourceAudit(context, ge::GRAPH_FAILED);
    }
    return ScatterElementsSourceAudit(context, tilingObject.RunKernelTiling());
  }'''
    if source.count(entry) != 1:
        raise RuntimeError("cannot locate ScatterElements source entry")
    return source.replace(entry, audit_entry)


def expected_overlay_source(source_root: Path) -> str:
    return instrument((source_root / RELATIVE).read_text(encoding="utf-8"))


def scoped_root_cmake(source: str) -> str:
    """Keep the public nested prepare build scoped to ScatterElementsV2."""
    original = 'set(ASCEND_OP_NAME                "ALL"                           CACHE   STRING   "operators that need to be compiled")'
    replacement = '''# SOURCE_CANDIDATE_BUILD_SCOPE_V1: the public prepare helper omits
# ASCEND_OP_NAME on its nested configure.  This detached build tree is limited
# to ScatterElementsV2; no operator algorithm is changed here.
set(ASCEND_OP_NAME "scatter_elements_v2" CACHE STRING "operators that need to be compiled" FORCE)'''
    if source.count(original) != 1:
        raise RuntimeError("cannot locate public ASCEND_OP_NAME build-selection anchor")
    return source.replace(original, replacement)


def existing_overlay(source_root: Path, output: Path) -> dict[str, Any] | None:
    if not output.exists():
        return None
    manifest_path = output / "source_candidate_overlay.json"
    if not manifest_path.is_file():
        raise RuntimeError("existing ScatterElements overlay lacks its manifest")
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "operator": "ScatterElementsV2", "source_family": "cann_ops", "cmake_op_name": "scatter_elements_v2",
        "audit_entry_relative": str(RELATIVE), "audit_sentinel": AUDIT_SENTINEL,
        "official_commit": SOURCE["commit"], "strategy_algorithm_changes": False,
        "source_compile_info_core_budget_enumeration": True, "source_hardware_envelope_heuristic_enumeration": True,
        "hardware_envelope_heuristic": {"enabled": True, "environment": "SCATTER_ELEMENTS_SOURCE_UB_DIVISOR",
                                         "audit_field": "ub_cap_divisor", "resource": "source_visible_ub_capacity",
                                         "divisors": [2, 4, 8], "max_anchors": 16},
        "build_scope": {"sentinel": BUILD_SCOPE, "operator": "scatter_elements_v2",
                        "reason": "official nested prepare helper otherwise drops ASCEND_OP_NAME and compiles unrelated operators",
                        "structural_only": True},
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise RuntimeError("existing ScatterElements overlay provenance differs")
    if (output / RELATIVE).read_text(encoding="utf-8") != expected_overlay_source(source_root):
        raise RuntimeError("existing ScatterElements overlay changes source outside its contract")
    if (output / "CMakeLists.txt").read_text(encoding="utf-8") != scoped_root_cmake(
            (source_root / "CMakeLists.txt").read_text(encoding="utf-8")):
        raise RuntimeError("existing ScatterElements root build scope differs")
    manifest["resumed_existing_overlay"] = True
    return manifest


def write_overlay(source_root: Path, output_parent: Path) -> dict[str, Any]:
    output = output_parent / "scatter_elements_v2_source"
    existing = existing_overlay(source_root, output)
    if existing is not None:
        return existing
    run(["git", "-C", str(source_root), "worktree", "add", "--detach", str(output), SOURCE["commit"]])
    root_cmake = output / "CMakeLists.txt"
    root_cmake.write_text(scoped_root_cmake(root_cmake.read_text(encoding="utf-8")), encoding="utf-8")
    target = output / RELATIVE
    before = target.read_text(encoding="utf-8")
    after = instrument(before)
    target.write_text(after, encoding="utf-8")
    manifest: dict[str, Any] = {
        "schema": "scatter_elements_original_tiler_overlay_v1", "operator": "ScatterElementsV2",
        "source_family": "cann_ops", "cmake_op_name": "scatter_elements_v2",
        "audit_entry_relative": str(RELATIVE), "audit_sentinel": AUDIT_SENTINEL,
        "official_url": SOURCE["url"], "official_commit": SOURCE["commit"], "overlay": str(output),
        "modified_source_files": [{"path": str(RELATIVE), "sha256_before": hashlib.sha256(before.encode()).hexdigest(), "sha256_after": digest(target)}],
        "instrumentation": {"enabled": True, "audit_schema": "scatter_elements_raw_tiling_observation_v1",
                            "audit_environment": "SCATTER_ELEMENTS_TILING_AUDIT_PATH",
                            "source_budget_environment": "SCATTER_ELEMENTS_SOURCE_AIV_CAP", "mutates_tiling_context": False},
        "strategy_algorithm_changes": False, "source_compile_info_core_budget_enumeration": True,
        "source_hardware_envelope_heuristic_enumeration": True,
        "build_scope": {"sentinel": BUILD_SCOPE, "operator": "scatter_elements_v2",
                        "reason": "official nested prepare helper otherwise drops ASCEND_OP_NAME and compiles unrelated operators",
                        "structural_only": True},
        "hardware_envelope_heuristic": {"enabled": True, "environment": "SCATTER_ELEMENTS_SOURCE_UB_DIVISOR",
                                         "audit_field": "ub_cap_divisor", "resource": "source_visible_ub_capacity",
                                         "divisors": [2, 4, 8], "max_anchors": 16},
        "candidate_rule": "run the original documented last-axis tiler for every finite runtime-bounded AIV core budget; if that complete original set is below 20, rerun selected contexts with smaller source-visible UB capacities; accept only raw identities that later pass exact output validation",
        "forbidden": LOCK["collection_contract"]["forbidden"],
    }
    (output / "source_candidate_overlay.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-parent", required=True, type=Path)
    args = parser.parse_args()
    if not args.output_parent.is_dir():
        raise RuntimeError("output parent does not exist: {}".format(args.output_parent))
    require_pinned_source(args.source_root)
    print(json.dumps({"schema": "scatter_elements_original_tiler_overlay_batch_v1", "matmul_included": False,
                      "overlay": write_overlay(args.source_root, args.output_parent)}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print("ScatterElements source-candidate error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
