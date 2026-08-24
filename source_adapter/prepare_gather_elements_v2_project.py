#!/usr/bin/env python3
"""Prepare one private CANN-8.1 build project for GatherElementsV2.

The installed CANN 8.1 package exposes ``GatherElements`` as a built-in
dynamic Python operator.  Replacing a Python file under a private
``ASCEND_OPP_PATH`` does *not* make the generic compiler select it.  CANN's
supported custom-operator route instead requires an op-proto library and an
op-tiling library in one vendor package.  This helper constructs exactly that
private project from the repository-pinned sources.

Only the project under ``--output`` is written.  The CANN installation and
the source bundle are read-only inputs.  The two small source edits are
strictly before the original tiler: a bounded AIV/UB input envelope and an
audit of the raw tiling emitted by the original formulas.  No generated raw
tiling field is changed after ``SaveToBuffer``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parent
LOCK = json.loads((ROOT / "non_matmul_source_lock.json").read_text(encoding="utf-8"))
SOURCE_BUNDLE = ROOT / "vendor_source" / "gather_elements_v2_source.zip"
AUDIT_SCHEMA = "gather_elements_v2_source_observation_v1"
SOURCE_OPERATOR = "GatherElementsV2"
VENDOR = "gather_elements_source"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def member_path(name: str) -> Path:
    item = PurePosixPath(name)
    if item.is_absolute() or ".." in item.parts:
        raise RuntimeError("unsafe GatherElements source member: {}".format(name))
    return Path(*item.parts)


def source_prefix(archive: zipfile.ZipFile) -> PurePosixPath:
    marker = "op_host/gather_elements_v2_tiling.cpp"
    roots = [PurePosixPath(info.filename).parent.parent for info in archive.infolist()
             if info.filename.endswith(marker)]
    require(len(roots) == 1, "GatherElements source archive lacks a unique root")
    return roots[0]


def extract_operator(destination: Path) -> None:
    require(SOURCE_BUNDLE.is_file(), "missing pinned GatherElements source archive")
    with zipfile.ZipFile(SOURCE_BUNDLE) as archive:
        prefix = source_prefix(archive)
        for info in archive.infolist():
            if info.is_dir():
                continue
            original = member_path(info.filename)
            try:
                relative = original.relative_to(Path(*prefix.parts))
            except ValueError:
                continue
            if not relative.parts:
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def compat_cmake() -> str:
    """Use CANN-8.1's project targets, not the 8.3-only source macros."""
    return r'''# Private CANN-8.1 compatibility harness for the pinned GatherElementsV2 source.
# It only binds the official host/kernel sources to the existing CANN project
# targets; it does not change the source tiling formulas or kernel algorithm.
add_ops_compile_options(
        OP_NAME GatherElementsV2
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
)

target_sources(optiling PRIVATE
        op_host/gather_elements_v2_tiling.cpp
)
target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
        ${CMAKE_SOURCE_DIR}/src/common/inc
        ${CMAKE_SOURCE_DIR}/src/common/op_host/op_tiling
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/external
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
)
target_include_directories(opsproto PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
        ${CMAKE_SOURCE_DIR}/src/common/inc
        ${CMAKE_SOURCE_DIR}/src/common/op_host/op_tiling
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/external
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
)

# The prototype is required for CANN to recognize GatherElementsV2 as a real
# custom operator.  Do not use a synthetic type without this library.
target_sources(op_host_aclnnInner PRIVATE
        op_host/gather_elements_v2_def.cpp
)
target_include_directories(op_host_aclnnInner PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
        ${CMAKE_SOURCE_DIR}/src/common/inc
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/external
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
)

# Keep the official dynamic kernel sources in the vendor package.  CANN can
# build only a launched key from these sources; the harness never fabricates a
# kernel or a tiling field.
install(FILES
        op_kernel/gather_elements_v2.cpp
        op_kernel/gather_elements_v2_common.h
        op_kernel/gather_elements_v2_last_dim.h
        op_kernel/gather_elements_v2_scalar.h
        op_kernel/gather_elements_v2_transpose.h
        DESTINATION ${IMPL_DYNAMIC_INSTALL_DIR}
)
'''


def audit_header(source_sha: str) -> str:
    return r'''#ifndef GATHER_ELEMENTS_V2_SOURCE_AUDIT_H
#define GATHER_ELEMENTS_V2_SOURCE_AUDIT_H

// GATHER_ELEMENTS_V2_SOURCE_AUDIT_V1
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <fstream>

#include "register/op_impl_registry.h"

// The extracted 8.3 source includes log/log.h, which CANN 8.1 does not
// publish.  Its logging is observational only; retain the source's call
// sites as no-ops on the older framework rather than replacing tiling logic.
#ifndef OP_LOGD
#define OP_LOGD(...) ((void)0)
#endif
#ifndef OP_LOGE
#define OP_LOGE(...) ((void)0)
#endif
#ifndef OP_CHECK_IF
#define OP_CHECK_IF(condition, log_expression, return_expression) \
    do { if (condition) { log_expression; return_expression; } } while (0)
#endif
#ifndef OP_CHECK_NULL_WITH_CONTEXT
#define OP_CHECK_NULL_WITH_CONTEXT(context, pointer) \
    do { if ((pointer) == nullptr) { OP_LOGE(context, "null pointer"); return ge::GRAPH_FAILED; } } while (0)
#endif

namespace gather_elements_source_audit {
constexpr const char *kSchema = "''' + AUDIT_SCHEMA + r'''";
constexpr const char *kSourceSha256 = "''' + source_sha + r'''";

inline uint64_t ParseBounded(const char *name, uint64_t current, const uint64_t *allowed, size_t count)
{
    const char *text = std::getenv(name);
    if (text == nullptr || *text == '\0') return current;
    errno = 0;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0 || parsed > current) return 0;
    for (size_t index = 0; index < count; ++index) {
        if (parsed == allowed[index]) return static_cast<uint64_t>(parsed);
    }
    return 0;
}

inline uint64_t CoreCap(uint64_t physical)
{
    const uint64_t caps[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    // No environment override means the untouched platform value.  An
    // explicit invalid override returns zero so the caller rejects the
    // request before entering the original formulas.
    if (std::getenv("GATHER_ELEMENTS_SOURCE_AIV_CAP") == nullptr) return physical;
    const uint64_t value = ParseBounded("GATHER_ELEMENTS_SOURCE_AIV_CAP", physical, caps, sizeof(caps) / sizeof(caps[0]));
    return value;
}

inline uint64_t VisibleUb(uint64_t physical)
{
    const uint64_t divisors[] = {1, 2, 4, 8};
    if (std::getenv("GATHER_ELEMENTS_SOURCE_UB_DIVISOR") == nullptr) return physical;
    const uint64_t divisor = ParseBounded("GATHER_ELEMENTS_SOURCE_UB_DIVISOR", 8, divisors, sizeof(divisors) / sizeof(divisors[0]));
    if (divisor == 0 || physical / divisor == 0) return 0;
    return physical / divisor;
}

inline uint64_t RequestedCoreCap()
{
    const char *text = std::getenv("GATHER_ELEMENTS_SOURCE_AIV_CAP");
    return text == nullptr ? 0 : std::strtoull(text, nullptr, 10);
}

inline uint64_t RequestedUbDivisor()
{
    const char *text = std::getenv("GATHER_ELEMENTS_SOURCE_UB_DIVISOR");
    return text == nullptr ? 1 : std::strtoull(text, nullptr, 10);
}

inline uint64_t Fnv1a64(const uint8_t *data, size_t bytes)
{
    uint64_t value = 1469598103934665603ULL;
    for (size_t index = 0; index < bytes; ++index) {
        value ^= static_cast<uint64_t>(data[index]);
        value *= 1099511628211ULL;
    }
    return value;
}

inline void Emit(gert::TilingContext *context, uint64_t block_dim, uint64_t tiling_key)
{
    const char *path = std::getenv("GATHER_ELEMENTS_TILING_AUDIT_PATH");
    if (path == nullptr || *path == '\0' || context == nullptr) return;
    const auto *raw = context->GetRawTilingData();
    if (raw == nullptr || raw->GetData() == nullptr || raw->GetDataSize() == 0) return;
    const auto *bytes = reinterpret_cast<const uint8_t *>(raw->GetData());
    const size_t size = raw->GetDataSize();
    std::ofstream out(path, std::ios::out | std::ios::app);
    if (!out.good()) return;
    out << "{\"schema\":\"" << kSchema
        << "\",\"event\":\"tiling_generated\",\"operator_type\":\"GatherElementsV2"
        << "\",\"status\":0"
        << "\",\"source_compile_context_sha256\":\"" << kSourceSha256
        << "\",\"aiv_core_cap\":" << RequestedCoreCap()
        << ",\"ub_cap_divisor\":" << RequestedUbDivisor()
        << ",\"compile_info_vars\":{\"block_dim\":" << block_dim
        << ",\"tiling_key\":" << tiling_key
        << ",\"raw_tiling_bytes\":" << size
        << ",\"raw_tiling_fnv1a64\":\"" << Fnv1a64(bytes, size) << "\"}}\n";
}
}  // namespace gather_elements_source_audit
#endif
'''


def math_util_compat_header() -> str:
    """Bridge the 8.3 ``Ops::Base`` spelling onto CANN-8.1's op_util API."""
    return r'''#ifndef GATHER_ELEMENTS_V2_CANN81_MATH_UTIL_COMPAT_H
#define GATHER_ELEMENTS_V2_CANN81_MATH_UTIL_COMPAT_H

// The extracted source calls the public 8.3 Ops::Base helpers.  CANN 8.1
// ships the same integer operations under ops in op_util.h.  This header is a
// spelling-only compatibility bridge; it preserves every input and result of
// the source formulas.
#include "op_util.h"

namespace Ops {
namespace Base {
template <typename T>
inline T CeilDiv(T value, T divisor) { return ops::CeilDiv(value, divisor); }

template <typename T>
inline T FloorDiv(T value, T divisor) { return ops::FloorDiv(value, divisor); }

template <typename T>
inline T CeilAlign(T value, T align) { return ops::CeilAlign(value, align); }
}  // namespace Base
}  // namespace Ops
#endif
'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    require(text.count(old) == 1, "cannot locate unique {} compatibility anchor".format(label))
    return text.replace(old, new)


def instrument(operator: Path) -> dict[str, str]:
    tiling = operator / "op_host" / "gather_elements_v2_tiling.cpp"
    last = operator / "op_host" / "gather_elements_v2_last_dim_tiling.h"
    require(tiling.is_file() and last.is_file(), "GatherElementsV2 host tiler sources are incomplete")
    source_sha = digest(tiling)
    header = operator / "op_host" / "gather_elements_v2_source_audit.h"
    header.write_text(audit_header(source_sha), encoding="utf-8")
    math_compat = operator / "op_host" / "util" / "math_util.h"
    math_compat.parent.mkdir(parents=True, exist_ok=True)
    math_compat.write_text(math_util_compat_header(), encoding="utf-8")

    body = tiling.read_text(encoding="utf-8")
    body = replace_once(body, '#include "platform/platform_infos_def.h"\n',
                        '#include "platform/platform_infos_def.h"\n#include "gather_elements_v2_source_audit.h"\n',
                        "tiling audit include")
    body = replace_once(body,
                        '    coreNum_ = compileInfo->totalCoreNum;\n    ubSize_ = compileInfo->ubSizePlatForm;',
                        '    coreNum_ = gather_elements_source_audit::CoreCap(compileInfo->totalCoreNum);\n'
                        '    ubSize_ = gather_elements_source_audit::VisibleUb(compileInfo->ubSizePlatForm);\n'
                        '    OP_CHECK_IF(coreNum_ == 0 || ubSize_ == 0, OP_LOGE("GatherElementsV2", "invalid bounded source budget"), return ge::GRAPH_FAILED);',
                        "non-last source budget")
    body = replace_once(body,
                        '    tilingContext_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());\n    TilingDataPrint();',
                        '    tilingContext_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());\n'
                        '    gather_elements_source_audit::Emit(tilingContext_, usedCoreNum_, tilingKey_);\n'
                        '    TilingDataPrint();',
                        "non-last raw tiling audit")
    tiling.write_text(body, encoding="utf-8")

    last_body = last.read_text(encoding="utf-8")
    last_body = replace_once(last_body, '#include "log/log.h"\n#include "tiling/platform/platform_ascendc.h"\n',
                             '#include "tiling/platform/platform_ascendc.h"\n#include "gather_elements_v2_source_audit.h"\n',
                             "last-dimension audit include")
    last_body = replace_once(last_body,
                             '    totalCoreNum_ = static_cast<int64_t>(compileInfo->totalCoreNum);\n'
                             '    ubSizePlatForm_ = static_cast<int64_t>(compileInfo->ubSizePlatForm - RESERVED_UB);',
                             '    totalCoreNum_ = static_cast<int64_t>(gather_elements_source_audit::CoreCap(compileInfo->totalCoreNum));\n'
                             '    const uint64_t boundedUb = gather_elements_source_audit::VisibleUb(compileInfo->ubSizePlatForm);\n'
                             '    OP_CHECK_IF(totalCoreNum_ <= 0 || boundedUb <= RESERVED_UB, OP_LOGE(context_, "invalid bounded source budget"), return ge::GRAPH_FAILED);\n'
                             '    ubSizePlatForm_ = static_cast<int64_t>(boundedUb - RESERVED_UB);',
                             "last-dimension source budget")
    last_body = replace_once(last_body,
                             '    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());\n}',
                             '    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());\n'
                             '    gather_elements_source_audit::Emit(context_, needUsedCore_, TILING_KEY_2);\n}',
                             "last-dimension raw tiling audit")
    last.write_text(last_body, encoding="utf-8")
    return {
        "original_tiling_cpp_sha256": source_sha,
        "instrumented_tiling_cpp_sha256": digest(tiling),
        "instrumented_last_dim_header_sha256": digest(last),
        "audit_header_sha256": digest(header),
        "math_util_compat_header_sha256": digest(math_compat),
    }


def retain_ascend910b_config(operator: Path) -> list[str]:
    """Keep only the source's Ascend910B binary declaration.

    CANN 8.1's package generator rejects the newer Kirin target names before
    it reaches the selected platform.  The campaign is explicitly 910B-only;
    omitting unrelated target declarations is a packaging compatibility step,
    not a tiling or kernel-algorithm change.
    """
    config = operator / "op_host" / "config"
    expected = config / "ascend910b" / "gather_elements_v2_binary.json"
    require(expected.is_file(), "GatherElementsV2 lacks its Ascend910B binary declaration")
    removed: list[str] = []
    for child in config.iterdir():
        if child.name == "ascend910b":
            continue
        if child.is_dir():
            removed.append(child.name)
            shutil.rmtree(child)
        else:
            raise RuntimeError("unexpected GatherElementsV2 config entry: {}".format(child))
    return removed


def retain_ascend910b_op_proto(operator: Path) -> str:
    """Remove newer-platform registrations rejected by CANN-8.1's generator."""
    definition = operator / "op_host" / "gather_elements_v2_def.cpp"
    body = definition.read_text(encoding="utf-8")
    body = replace_once(
        body,
        '        this->AICore().AddConfig("ascend910b");\n'
        '        this->AICore().AddConfig("ascend910_93");\n\n'
        '        OpAICoreConfig config_kirin = GetKirinCoreConfig();\n'
        '        this->AICore().AddConfig("kirinx90", config_kirin);\n'
        '        this->AICore().AddConfig("kirin9030", config_kirin);',
        '        this->AICore().AddConfig("ascend910b");',
        "non-910B op-proto registrations",
    )
    definition.write_text(body, encoding="utf-8")
    return digest(definition)


def make_build_helpers_executable(project: Path) -> list[str]:
    """Restore executable bits required by CANN's copied CMake helpers.

    Repository source bundles are stored as content archives and do not retain
    POSIX mode bits.  These helpers execute only from the private build
    project; neither the source archive nor the installed toolkit is changed.
    """
    helpers = [project / "cmake" / "util" / "gen_ops_filter.sh"]
    restored: list[str] = []
    for helper in helpers:
        require(helper.is_file(), "CANN build helper is absent: {}".format(helper))
        helper.chmod(helper.stat().st_mode | 0o111)
        restored.append(str(helper.relative_to(project)))
    return restored


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cann-ops-source", required=True, type=Path)
    parser.add_argument("--cann-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(args.cann_ops_source.is_dir(), "missing private pinned cann-ops source")
    require(args.cann_root.is_dir(), "missing CANN root")
    require(not args.output.exists(), "refuse to overwrite GatherElementsV2 source project")
    require((args.cann_ops_source / "CMakeLists.txt").is_file(), "cann-ops source lacks its root CMake project")
    require(digest(SOURCE_BUNDLE) == "9f1d9753d47ec8e0f0fb3dde1de5b047e048900ff2e9a014c1ed614d7ed57470",
            "pinned GatherElementsV2 source archive hash mismatches")

    shutil.copytree(args.cann_ops_source, args.output, ignore=shutil.ignore_patterns(".git", "build", "output"))
    operator = args.output / "src" / "index" / "gather_elements_v2"
    require(not operator.exists(), "pinned CANN 8.1 source unexpectedly already contains GatherElementsV2")
    extract_operator(operator)
    (operator / "CMakeLists.txt").write_text(compat_cmake(), encoding="utf-8")
    provenance = instrument(operator)
    pruned_platform_configs = retain_ascend910b_config(operator)
    provenance["instrumented_op_proto_sha256"] = retain_ascend910b_op_proto(operator)
    build_helper_modes = make_build_helpers_executable(args.output)
    pinned = LOCK["operators"]["gather_elements_v2"]["pinned_files"]
    for relative, expected in pinned.items():
        actual = operator / relative
        require(actual.is_file(), "pinned GatherElementsV2 file is absent: {}".format(relative))
        if relative not in ("op_host/gather_elements_v2_tiling.cpp", "op_host/gather_elements_v2_last_dim_tiling.h",
                            "op_host/gather_elements_v2_def.cpp"):
            require(digest(actual) == expected, "pinned GatherElementsV2 file hash mismatches: {}".format(relative))

    manifest: dict[str, Any] = {
        "schema": "gather_elements_v2_custom_project_v1",
        "operator": SOURCE_OPERATOR,
        "runtime_op": "gather_elements",
        "vendor": VENDOR,
        "source_kind": "pinned_extracted_gather_elements_v2_source_with_cann81_cmake_compatibility",
        "source_bundle": str(SOURCE_BUNDLE),
        "source_bundle_sha256": digest(SOURCE_BUNDLE),
        "cann_root": str(args.cann_root.resolve()),
        "cann_ops_source": str(args.cann_ops_source.resolve()),
        "project_root": str(args.output.resolve()),
        "operator_root": str(operator.resolve()),
        "source_file": str((operator / "op_host" / "gather_elements_v2_tiling.cpp").resolve()),
        "source_file_sha256": digest(operator / "op_host" / "gather_elements_v2_tiling.cpp"),
        "instrumentation": {
            "enabled": True,
            "mutates_tiling_context": False,
            "audit_schema": AUDIT_SCHEMA,
            "audit_environment": "GATHER_ELEMENTS_TILING_AUDIT_PATH",
            "source_budget_environment": "GATHER_ELEMENTS_SOURCE_AIV_CAP",
            "dispatch_environment": "GATHER_ELEMENTS_SOURCE_DISPATCH",
            "dispatch_value": "aclop_compile_and_execute",
        },
        "hardware_envelope_heuristic": {
            "enabled": True,
            "environment": "GATHER_ELEMENTS_SOURCE_UB_DIVISOR",
            "audit_field": "ub_cap_divisor",
            "resource": "source_visible_ub_capacity",
            "divisors": [2, 4, 8],
            "max_anchors": 16,
        },
        "compatibility_port": {
            "source_version": "extracted CANN-8.3 GatherElementsV2 source",
            "target_version": "CANN-8.1.RC1 build framework",
            "allowed_changes": ["CMake target binding", "pre-source bounded hardware budget", "post-generation raw-tiling audit"],
            "tiling_algorithm_changes": False,
            "kernel_algorithm_changes": False,
            "pruned_unrelated_platform_configs": pruned_platform_configs,
            "private_build_helper_execute_bits_restored": build_helper_modes,
        },
        "provenance": provenance,
        "toolkit_install_modified": False,
        "matmul_included": False,
    }
    (args.output / "gather_elements_v2_project.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "prepared", "operator": SOURCE_OPERATOR, "project": str(args.output),
                      "toolkit_install_modified": False, "npu_calls": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, zipfile.BadZipFile) as error:
        print("GatherElementsV2 project preparation error: {}".format(error), file=os.sys.stderr)
        raise SystemExit(2)
