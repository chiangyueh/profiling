#!/usr/bin/env python3
"""Prepare a private CANN-8.1 custom-op project for GatherElementsV2.

The extracted GatherElementsV2 implementation is a CANN 8.3 source tree.
It cannot be paired with the installed CANN 8.1 ``aclnnGather`` API: that API
constructs the built-in GatherElements operator, not GatherElementsV2.  This
preparer instead uses CANN 8.1's shipped custom-operator project template to
build one complete private package: OpProto, host tiler, generated CANN 8.1
custom API, and the source kernel are delivered together.

Only build wiring, CANN-8.1 compatibility headers, Ascend910B registration
scope, observational audit, and bounded source resource inputs are added.
Neither the extracted tiling algorithm nor the device kernel algorithm is
rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
LOCK = json.loads((ROOT / "non_matmul_source_lock.json").read_text(encoding="utf-8"))
EXTRACTED = LOCK["sources"]["extracted_installed_source"]
OP = LOCK["operators"]["gather_elements_v2"]

AUDIT = "GATHER_ELEMENTS_SOURCE_TILING_AUDIT_V2"
CORE = "GATHER_ELEMENTS_SOURCE_AIV_CAP_V2"
UB = "GATHER_ELEMENTS_SOURCE_UB_CAP_V2"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def require_extracted(path: Path) -> None:
    if EXTRACTED.get("source_version") != "CANN 8.3.RC2 reported by extracted run.sh":
        raise RuntimeError("GatherElements extracted-source version evidence is missing")
    for relative, expected in OP["pinned_files"].items():
        item = path / relative
        if not item.is_file() or digest(item) != expected:
            raise RuntimeError("GatherElements extracted-source hash mismatch: {}".format(relative))
    required = (
        "op_host/gather_elements_v2_def.cpp",
        "op_host/gather_elements_v2_tiling.cpp",
        "op_host/gather_elements_v2_tiling.h",
        "op_host/gather_elements_v2_last_dim_tiling.h",
        "op_kernel/gather_elements_v2.cpp",
    )
    if any(not (path / item).is_file() for item in required):
        raise RuntimeError("GatherElements extracted source is incomplete")


def logging_header_shim() -> str:
    return '''#pragma once
// CANN-8.1 compatibility shim for CANN-8.3 diagnostic declarations only.
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
    return '''#pragma once
// CANN-8.1 compatibility shim for the original tiler's integer helpers.
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
} }
'''


def instrument_tiler(text: str) -> str:
    if AUDIT in text:
        return text
    include = '#include "platform/platform_infos_def.h"\n'
    prelude = r'''#include "platform/platform_infos_def.h"
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// GATHER_ELEMENTS_SOURCE_TILING_AUDIT_V2: observation after the original
// source decision. It never mutates raw tiling data.
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
        std::fprintf(out, "{\\\"schema\\\":\\\"gather_elements_raw_tiling_observation_v2\\\",\\\"status\\\":%d,\\\"aiv_core_cap\\\":\\\"%s\\\",\\\"ub_cap_divisor\\\":\\\"%s\\\",\\\"tiling_key\\\":%llu,\\\"block_dim\\\":%u,\\\"raw_bytes\\\":%llu,\\\"raw_fnv1a64\\\":%llu}\\n",
            static_cast<int>(status), aiv == nullptr ? "runtime" : aiv, ub == nullptr ? "1" : ub,
            static_cast<unsigned long long>(context->GetTilingKey()), context->GetBlockDim(),
            static_cast<unsigned long long>(size), static_cast<unsigned long long>(GatherElementsSourceAuditHash(data, size)));
        std::fclose(out);
    }
    return status;
}
'''
    if text.count(include) != 1:
        raise RuntimeError("cannot find GatherElements platform include anchor")
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
    // GATHER_ELEMENTS_SOURCE_AIV_CAP_V2: bounded legal source input.
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
    // GATHER_ELEMENTS_SOURCE_UB_CAP_V2: only lowers visible UB capacity.
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


def narrow_definition(source: str) -> str:
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
    replacement = '''        // CANN-8.1 compatibility: retain only source's Ascend910B declaration.
        this->AICore().AddConfig("ascend910b");
    }
'''
    if source.count(original) != 1:
        raise RuntimeError("cannot locate non-910B GatherElements OpDef block")
    return source.replace(original, replacement)


def patch_template_function(source: str) -> str:
    old = "-shared -std=c++11 ${OPBUILD_OPS_SRC}"
    new = "-shared -std=c++14 ${OPBUILD_OPS_SRC}"
    if source.count(old) != 1:
        raise RuntimeError("cannot locate CANN-8.1 custom-template host standard anchor")
    source = source.replace(old, new)
    old_include = "-I ${ASCEND_CANN_PACKAGE_PATH}/include -I ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel"
    new_include = "-I ${ASCEND_CANN_PACKAGE_PATH}/include -I ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include -I ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/experiment/platform -I ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/experiment/metadef -I ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/experiment/runtime -I ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel"
    if source.count(old_include) != 1:
        raise RuntimeError("cannot locate CANN-8.1 custom-template host include anchor")
    return source.replace(old_include, new_include)


def cann81_host_include_directories() -> str:
    """Return the public and architecture-specific host headers CANN 8.1 ships.

    The stock custom-op template only adds ``<cann>/include``.  In the
    installed 8.1 package, the public platform, metadef, runtime and opdev
    declarations are instead below ``<cann>/<host>-linux/include``.  Source
    GatherElementsV2 includes those public declarations directly, so every
    host-side target needs those *headers*; this does not change any tiling
    or kernel behavior.
    """
    return """${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/experiment/platform
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/experiment/metadef
        ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/experiment/runtime"""


def patch_template_interface(source: str) -> str:
    old_standard = "$<$<COMPILE_LANGUAGE:CXX>:-std=c++11>"
    if source.count(old_standard) != 1:
        raise RuntimeError("cannot locate CANN-8.1 custom-template C++ standard anchor")
    source = source.replace(old_standard, "$<$<COMPILE_LANGUAGE:CXX>:-std=c++14>")
    old_include = """target_include_directories(intf_pub INTERFACE ${ASCEND_CANN_PACKAGE_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
)"""
    new_include = """target_include_directories(intf_pub INTERFACE
    %s
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
)""" % cann81_host_include_directories()
    if source.count(old_include) != 1:
        raise RuntimeError("cannot locate CANN-8.1 custom-template interface include anchor")
    return source.replace(old_include, new_include)


def patch_template_op_host(source: str) -> str:
    """Add CANN-8.1's arch-specific public headers to all host targets."""
    include_block = """target_include_directories(%s PRIVATE
        %s
        ${CMAKE_CURRENT_SOURCE_DIR}
    )\n"""
    anchors = {
        "cust_op_proto": "target_compile_definitions(cust_op_proto PRIVATE OP_PROTO_LIB)",
        "cust_optiling": "target_compile_definitions(cust_optiling PRIVATE OP_TILING_LIB)",
        "cust_opapi": "if(ENABLE_CROSS_COMPILE)\n    target_link_directories(cust_opapi PRIVATE",
    }
    for target, anchor in anchors.items():
        if source.count(anchor) != 1:
            raise RuntimeError("cannot locate CANN-8.1 custom-template %s include anchor" % target)
        block = include_block % (target, cann81_host_include_directories())
        source = source.replace(anchor, block + anchor)
    return source


def expected_manifest(output: Path, extracted: Path, template: Path) -> dict[str, Any]:
    return {
        "schema": "gather_elements_complete_custom_package_overlay_v1",
        "operator": "GatherElementsV2",
        "cmake_op_name": "gather_elements_v2",
        "source_family": "extracted_installed_source",
        "source_version": EXTRACTED["source_version"],
        "source_revision": "CANN-8.3.RC2-extracted-installed-source",
        "overlay": str(output),
        "template_root": str(template),
        "source_root": str(extracted),
        "audit_entry_relative": "op_host/gather_elements_v2_tiling.cpp",
        "audit_sentinel": AUDIT,
        "instrumentation": {
            "enabled": True, "audit_schema": "gather_elements_raw_tiling_observation_v2",
            "audit_environment": "GATHER_ELEMENTS_TILING_AUDIT_PATH",
            "source_budget_environment": "GATHER_ELEMENTS_SOURCE_AIV_CAP",
            "mutates_tiling_context": False,
        },
        "hardware_envelope_heuristic": {
            "enabled": True, "environment": "GATHER_ELEMENTS_SOURCE_UB_DIVISOR",
            "audit_field": "ub_cap_divisor", "resource": "source_visible_ub_capacity",
            "divisors": [2, 4, 8], "max_anchors": 16,
        },
        "strategy_algorithm_changes": False,
        "kernel_algorithm_changes": False,
        "build_scope": "one complete private GatherElementsV2 custom package",
        "formal_data_gate": "generated CANN-8.1 custom API + source tiler + source kernel must execute and match installed aclnnGather output",
    }


def existing(output: Path, extracted: Path, template: Path) -> dict[str, Any] | None:
    manifest_path = output / "source_candidate_overlay.json"
    if not output.exists():
        return None
    if not manifest_path.is_file():
        raise RuntimeError("existing GatherElements custom overlay lacks manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = expected_manifest(output, extracted, template)
    for key in ("schema", "operator", "cmake_op_name", "source_family", "source_version", "source_revision", "audit_sentinel",
                "strategy_algorithm_changes", "kernel_algorithm_changes", "build_scope", "formal_data_gate"):
        if manifest.get(key) != expected.get(key):
            raise RuntimeError("existing GatherElements custom overlay provenance differs: {}".format(key))
    tiler = output / expected["audit_entry_relative"]
    source_tiler = extracted / "op_host/gather_elements_v2_tiling.cpp"
    if not tiler.is_file() or tiler.read_text(encoding="utf-8") != instrument_tiler(source_tiler.read_text(encoding="utf-8")):
        raise RuntimeError("existing GatherElements custom overlay changes its tiler unexpectedly")
    definition = output / "op_host/gather_elements_v2_def.cpp"
    if definition.read_text(encoding="utf-8") != narrow_definition((extracted / "op_host/gather_elements_v2_def.cpp").read_text(encoding="utf-8")):
        raise RuntimeError("existing GatherElements custom overlay changes its OpDef unexpectedly")
    template_files = (
        ("cmake/func.cmake", patch_template_function),
        ("cmake/intf.cmake", patch_template_interface),
        ("op_host/CMakeLists.txt", patch_template_op_host),
    )
    for relative, patcher in template_files:
        actual = output / relative
        original = template / relative
        if not actual.is_file() or not original.is_file() or actual.read_text(encoding="utf-8") != patcher(original.read_text(encoding="utf-8")):
            raise RuntimeError("existing GatherElements custom overlay changes CANN-8.1 build wiring unexpectedly: {}".format(relative))
    manifest["resumed_existing_overlay"] = True
    return manifest


def prepare(extracted: Path, template: Path, output_parent: Path) -> dict[str, Any]:
    output = output_parent / "gather_elements_v2_complete_custom"
    resumed = existing(output, extracted, template)
    if resumed is not None:
        return resumed
    if output.exists():
        raise RuntimeError("refuse to overwrite incomplete custom overlay: {}".format(output))
    shutil.copytree(template, output)
    for directory in (output / "op_host", output / "op_kernel"):
        for child in directory.iterdir():
            if child.name != "CMakeLists.txt":
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
    source_host = extracted / "op_host"
    for name in ("gather_elements_v2_def.cpp", "gather_elements_v2_tiling.cpp", "gather_elements_v2_tiling.h", "gather_elements_v2_last_dim_tiling.h"):
        target = output / "op_host" / name
        text = (source_host / name).read_text(encoding="utf-8")
        if name == "gather_elements_v2_tiling.cpp":
            text = instrument_tiler(text)
        elif name == "gather_elements_v2_def.cpp":
            text = narrow_definition(text)
        target.write_text(text, encoding="utf-8")
    shutil.copytree(extracted / "op_kernel", output / "op_kernel", dirs_exist_ok=True, ignore=shutil.ignore_patterns("CMakeLists.txt"))
    log_header = output / "op_host/log/log.h"
    log_header.parent.mkdir(parents=True, exist_ok=True)
    log_header.write_text(logging_header_shim(), encoding="utf-8")
    math_header = output / "op_host/util/math_util.h"
    math_header.parent.mkdir(parents=True, exist_ok=True)
    math_header.write_text(arithmetic_header_shim(), encoding="utf-8")
    func = output / "cmake/func.cmake"
    func.write_text(patch_template_function(func.read_text(encoding="utf-8")), encoding="utf-8")
    interface = output / "cmake/intf.cmake"
    interface.write_text(patch_template_interface(interface.read_text(encoding="utf-8")), encoding="utf-8")
    op_host_cmake = output / "op_host/CMakeLists.txt"
    op_host_cmake.write_text(patch_template_op_host(op_host_cmake.read_text(encoding="utf-8")), encoding="utf-8")
    manifest = expected_manifest(output, extracted, template)
    manifest["modified_source_files"] = [
        {"path": "op_host/gather_elements_v2_tiling.cpp", "kind": "observational_audit_and_bounded_source_resource_inputs"},
        {"path": "op_host/gather_elements_v2_def.cpp", "kind": "Ascend910B_registration_scope_only"},
        {"path": "op_host/log/log.h", "kind": "CANN-8.1_logging_header_compatibility"},
        {"path": "op_host/util/math_util.h", "kind": "CANN-8.1_integer_helper_compatibility"},
        {"path": "cmake/func.cmake", "kind": "CANN-8.1_custom_template_C++14_host_build_wiring"},
        {"path": "cmake/intf.cmake", "kind": "CANN-8.1_architecture_specific_host_headers_and_C++14"},
        {"path": "op_host/CMakeLists.txt", "kind": "CANN-8.1_architecture_specific_host_headers"},
    ]
    manifest["extracted_source_hashes"] = {
        str(path.relative_to(extracted)): digest(path)
        for path in sorted(extracted.rglob("*")) if path.is_file() and "tests" not in path.parts and ".git" not in path.parts and path.name != "run.sh"
    }
    (output / "source_candidate_overlay.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted-source-root", required=True, type=Path)
    parser.add_argument("--template-root", required=True, type=Path)
    parser.add_argument("--output-parent", required=True, type=Path)
    args = parser.parse_args()
    if not args.template_root.is_dir() or not (args.template_root / "CMakeLists.txt").is_file():
        raise RuntimeError("CANN custom-op template is absent or incomplete")
    if not args.output_parent.is_dir():
        raise RuntimeError("overlay output parent does not exist")
    require_extracted(args.extracted_source_root)
    print(json.dumps({"schema": "gather_elements_complete_custom_package_overlay_batch_v1",
                      "overlay": prepare(args.extracted_source_root, args.template_root, args.output_parent)},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print("GatherElements complete custom-package prepare error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
