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
LOCK = json.loads((ROOT / "scatter_elements_v2_cann81_lock.json").read_text(encoding="utf-8"))
SOURCE = LOCK["source"]
OP = LOCK["operator"]
RELATIVE = Path(OP["relative_root"]) / OP["tiler"]
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
    attestation = source_root / ".scatter_elements_v2_cann81_attestation.json"
    if not attestation.is_file():
        raise RuntimeError("private source lacks its dedicated CANN-8.1 attestation")
    value = json.loads(attestation.read_text(encoding="utf-8"))
    if (value.get("schema") != "scatter_elements_v2_cann81_source_bundle_v1" or
            value.get("archive_sha256") != SOURCE["archive_sha256"] or
            value.get("official_commit") != SOURCE["official_commit"] or
            value.get("network_calls") != 0 or value.get("installed_cann_writes") != 0):
        raise RuntimeError("private source attestation does not match the dedicated CANN-8.1 lock")
    path = source_root / RELATIVE
    if not path.is_file() or digest(path) != OP["tiler_sha256"]:
        raise RuntimeError("pinned ScatterElementsV2 tiler hash mismatch")
    cmake = source_root / "CMakeLists.txt"
    if not cmake.is_file() or digest(cmake) != SOURCE["root_cmake_sha256"]:
        raise RuntimeError("pinned CANN-8.1 root build file hash mismatch")
    status = run(["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=no"])
    if status:
        raise RuntimeError("private official source snapshot is modified")


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
    class_method = '''        void TilingDataPrint() const;'''
    class_method_replacement = '''        void TilingDataPrint() const;
        void SourceAudit(ge::graphStatus status) const;'''
    if source.count(class_method) != 1:
        raise RuntimeError("cannot locate ScatterElements tiling class audit anchor")
    source = source.replace(class_method, class_method_replacement)
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
  void ScatterElementsV2Tiling::SourceAudit(ge::graphStatus status) const {
    const char *path = std::getenv("SCATTER_ELEMENTS_TILING_AUDIT_PATH");
    if (path == nullptr || tilingContext == nullptr) { return; }
    auto *raw = tilingContext->GetRawTilingData();
    const size_t rawSize = raw == nullptr ? 0U : raw->GetDataSize();
    const auto *rawBytes = raw == nullptr ? nullptr : static_cast<const uint8_t *>(raw->GetData());
    const char *coreCap = std::getenv("SCATTER_ELEMENTS_SOURCE_AIV_CAP");
    std::FILE *output = std::fopen(path, "a");
    if (output != nullptr) {
      const char *ubDivisor = std::getenv("SCATTER_ELEMENTS_SOURCE_UB_DIVISOR");
      std::fprintf(output, "{\"schema\":\"scatter_elements_raw_tiling_observation_v1\",\"event\":\"tiling_generated\",\"operator_type\":\"ScatterElementsV2\",\"status\":%d,\"source_compile_context_sha256\":\"af4fa87f4760e73b93a31a301827e9e2c286c58f3ff32a0be7344adf2c5543f7\",\"aiv_core_cap\":\"%s\",\"ub_cap_divisor\":\"%s\",\"compile_info_vars\":{\"tiling_key\":%llu,\"block_dim\":%u,\"raw_tiling_bytes\":%llu,\"raw_tiling_fnv1a64\":\"%llu\"},\"tiling\":{\"used_core_num\":%llu,\"each_num\":%llu,\"extra_task_core\":%llu,\"each_piece\":%llu,\"input_one_piece\":%llu,\"input_count\":%llu,\"indices_count\":%llu,\"updates_count\":%llu,\"input_one_time\":%llu,\"indices_one_time\":%llu,\"updates_one_time\":%llu,\"input_each\":%llu,\"indices_each\":%llu,\"input_last\":%llu,\"indices_last\":%llu,\"input_loop\":%llu,\"indices_loop\":%llu,\"input_align\":%llu,\"indices_align\":%llu,\"updates_align\":%llu,\"last_indices_loop\":%llu,\"last_indices_each\":%llu,\"last_indices_last\":%llu,\"one_time\":%llu,\"last_one_time\":%llu,\"mode_flag\":%llu,\"source_visible_max_ub\":%llu,\"workspace_bytes\":%llu}}\n",
          static_cast<int>(status), coreCap == nullptr ? "runtime" : coreCap, ubDivisor == nullptr ? "1" : ubDivisor,
          static_cast<unsigned long long>(tilingContext->GetTilingKey()), tilingContext->GetBlockDim(),
          static_cast<unsigned long long>(rawSize), static_cast<unsigned long long>(ScatterElementsSourceAuditHash(rawBytes, rawSize)),
          static_cast<unsigned long long>(usedCoreNum), static_cast<unsigned long long>(eachNum),
          static_cast<unsigned long long>(extraTaskCore), static_cast<unsigned long long>(eachPiece),
          static_cast<unsigned long long>(inputOnePiece), static_cast<unsigned long long>(inputCount),
          static_cast<unsigned long long>(indicesCount), static_cast<unsigned long long>(updatesCount),
          static_cast<unsigned long long>(inputOneTime), static_cast<unsigned long long>(indicesOneTime),
          static_cast<unsigned long long>(updatesOneTime), static_cast<unsigned long long>(inputEach),
          static_cast<unsigned long long>(indicesEach), static_cast<unsigned long long>(inputLast),
          static_cast<unsigned long long>(indicesLast), static_cast<unsigned long long>(inputLoop),
          static_cast<unsigned long long>(indicesLoop), static_cast<unsigned long long>(inputAlign),
          static_cast<unsigned long long>(indicesAlign), static_cast<unsigned long long>(updatesAlign),
          static_cast<unsigned long long>(lastIndicesLoop), static_cast<unsigned long long>(lastIndicesEach),
          static_cast<unsigned long long>(lastIndicesLast), static_cast<unsigned long long>(oneTime),
          static_cast<unsigned long long>(lastOneTime), static_cast<unsigned long long>(modeFlag),
          static_cast<unsigned long long>(max_ub), static_cast<unsigned long long>(workspaceSize));
      std::fclose(output);
    }
  }
  ge::graphStatus TilingScatterElementsV2(gert::TilingContext* context) {
    ScatterElementsV2Tiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        tilingObject.SourceAudit(ge::GRAPH_FAILED);
        return ge::GRAPH_FAILED;
    }
    const ge::graphStatus status = tilingObject.RunKernelTiling();
    tilingObject.SourceAudit(status);
    return status;
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
    source = source.replace(original, replacement)
    source_install = '''foreach (_op_name ${OP_LIST})
    install(FILES ${ASCEND_IMPL_OUT_DIR}/dynamic/${_op_name}.py
            DESTINATION ${IMPL_DYNAMIC_INSTALL_DIR}
            OPTIONAL
    )
endforeach ()

foreach (_op_name ${OP_LIST})
    install(FILES ${ASCEND_IMPL_OUT_DIR}/dynamic/${_op_name}.cpp
            DESTINATION ${IMPL_DYNAMIC_INSTALL_DIR}
            OPTIONAL
    )
endforeach ()

install(DIRECTORY ${OPS_ADV_UTILS_KERNEL_INC}/
        DESTINATION ${IMPL_INSTALL_DIR}/ascendc/common
)

foreach (op_dir ${OP_DIR_LIST})
    get_filename_component(_op_name "${op_dir}" NAME)

    file(GLOB KERNEL_FILES
            ${op_dir}/op_kernel/*.cpp
            ${op_dir}/op_kernel/*.h
    )

    install(FILES ${KERNEL_FILES}
            DESTINATION ${IMPL_DYNAMIC_INSTALL_DIR}
            OPTIONAL
    )
endforeach ()'''
    binary_only = '''# SCATTER_ELEMENTS_CANN81_PRECOMPILED_PACKAGE_V1: the official
# sources remain build inputs, but the private runtime package installs only
# CANN-8.1 host libraries, metadata and precompiled 910B device objects.  A
# missing binary key therefore fails instead of invoking Python/TBE runtime
# compilation. No source algorithm or generated binary is changed.'''
    if source.count(source_install) != 1:
        raise RuntimeError("cannot locate official dynamic-source install block")
    return source.replace(source_install, binary_only)


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
        "official_commit": SOURCE["official_commit"], "strategy_algorithm_changes": False,
        "source_compile_info_core_budget_enumeration": True, "source_hardware_envelope_heuristic_enumeration": True,
        "hardware_envelope_heuristic": {"enabled": True, "environment": "SCATTER_ELEMENTS_SOURCE_UB_DIVISOR",
                                         "audit_field": "ub_cap_divisor", "resource": "source_visible_ub_capacity",
                                         "divisors": [2, 4, 8], "max_anchors": 16},
        "build_scope": {"sentinel": BUILD_SCOPE, "operator": "scatter_elements_v2",
                        "reason": "official nested prepare helper otherwise drops ASCEND_OP_NAME and compiles unrelated operators",
                        "structural_only": True, "runtime_package": "precompiled_910b_binaries_only"},
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise RuntimeError("existing ScatterElements overlay provenance differs")
    if (output / RELATIVE).read_text(encoding="utf-8") != expected_overlay_source(source_root):
        raise RuntimeError("existing ScatterElements overlay changes source outside its contract")
    if (output / "CMakeLists.txt").read_text(encoding="utf-8") != scoped_root_cmake(
            (source_root / "CMakeLists.txt").read_text(encoding="utf-8")):
        raise RuntimeError("existing ScatterElements root build scope differs")
    helper = output / "cmake/util/gen_ops_filter.sh"
    if not helper.is_file():
        raise RuntimeError("existing ScatterElements overlay lacks the official package helper")
    helper.chmod(helper.stat().st_mode | 0o111)
    manifest["resumed_existing_overlay"] = True
    return manifest


def write_overlay(source_root: Path, output_parent: Path) -> dict[str, Any]:
    output = output_parent / "scatter_elements_v2_source"
    existing = existing_overlay(source_root, output)
    if existing is not None:
        return existing
    run(["git", "-C", str(source_root), "worktree", "add", "--detach", str(output), "HEAD"])
    # The repository bundle intentionally contains source bytes only, so its
    # archive does not preserve executable mode bits.  CANN's package target
    # invokes this official helper directly; restore that one build metadata
    # bit inside the private worktree.
    helper = output / "cmake/util/gen_ops_filter.sh"
    if not helper.is_file():
        raise RuntimeError("official CANN package helper is absent")
    helper.chmod(helper.stat().st_mode | 0o111)
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
        "official_url": SOURCE["official_url"], "official_commit": SOURCE["official_commit"], "overlay": str(output),
        "modified_source_files": [{"path": str(RELATIVE), "sha256_before": hashlib.sha256(before.encode()).hexdigest(), "sha256_after": digest(target)}],
        "instrumentation": {"enabled": True, "audit_schema": "scatter_elements_raw_tiling_observation_v1",
                            "audit_environment": "SCATTER_ELEMENTS_TILING_AUDIT_PATH",
                            "source_budget_environment": "SCATTER_ELEMENTS_SOURCE_AIV_CAP", "mutates_tiling_context": False},
        "strategy_algorithm_changes": False, "source_compile_info_core_budget_enumeration": True,
        "source_hardware_envelope_heuristic_enumeration": True,
        "build_scope": {"sentinel": BUILD_SCOPE, "operator": "scatter_elements_v2",
                        "reason": "official nested prepare helper otherwise drops ASCEND_OP_NAME and compiles unrelated operators",
                        "structural_only": True, "runtime_package": "precompiled_910b_binaries_only"},
        "hardware_envelope_heuristic": {"enabled": True, "environment": "SCATTER_ELEMENTS_SOURCE_UB_DIVISOR",
                                         "audit_field": "ub_cap_divisor", "resource": "source_visible_ub_capacity",
                                         "divisors": [2, 4, 8], "max_anchors": 16},
        "candidate_rule": "run the original documented last-axis tiler for every finite runtime-bounded AIV core budget; if that complete original set is below 20, rerun selected contexts with smaller source-visible UB capacities; accept only raw identities that later pass exact output validation",
        "forbidden": LOCK["forbidden"],
    }
    (output / "source_candidate_overlay.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-parent", required=True, type=Path)
    args = parser.parse_args()
    args.source_root = args.source_root.resolve()
    args.output_parent = args.output_parent.resolve()
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
