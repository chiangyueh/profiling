#!/usr/bin/env python3
"""Validate the private GatherElementsV2 CANN package and write its manifest.

The package is built below this checkout.  This helper only reads that build
output and writes the requested manifest; it never changes the CANN install,
an OPP directory, device state, or an environment outside its own process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path


SCHEMA = "gather_elements_v2_cann81_prebuilt_package_v2"
OPERATOR = "GatherElementsV2"
VENDOR = "gather_elements_source"
AUDIT_SCHEMA = "gather_elements_v2_source_observation_v1"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def architecture_library(root: Path, relative: str) -> Path:
    candidates = [platform.machine(), "aarch64", "x86_64"]
    for architecture in candidates:
        path = root / relative.format(architecture=architecture)
        if path.is_file():
            return path
    raise RuntimeError("private package lacks {} for any supported architecture".format(relative))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--package-root", required=True, type=Path)
    parser.add_argument("--cann-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    project = args.project.resolve()
    package = args.package_root.resolve()
    cann = args.cann_root.resolve()
    require(project.is_dir(), "private GatherElementsV2 project is absent: {}".format(project))
    require(package.is_dir(), "private GatherElementsV2 package is absent: {}".format(package))
    require((cann / "opp" / "version.info").is_file(), "CANN OPP version file is absent")
    version_text = (cann / "opp" / "version.info").read_text(encoding="utf-8")
    require("version_dir=8.1.RC1" in version_text,
            "GatherElementsV2 must be fully built by CANN 8.1.RC1")
    project_manifest = project / "gather_elements_v2_project.json"
    require(project_manifest.is_file(), "private project provenance is absent: {}".format(project_manifest))
    provenance = json.loads(project_manifest.read_text(encoding="utf-8"))
    require(provenance.get("operator") == OPERATOR and provenance.get("vendor") == VENDOR,
            "private project has the wrong operator or vendor")
    require(provenance.get("toolkit_install_modified") is False,
            "private project does not attest a read-only CANN installation")
    compatibility = provenance.get("compatibility_port")
    require(isinstance(compatibility, dict) and compatibility.get("tiling_algorithm_changes") is False and
            compatibility.get("kernel_algorithm_changes") is False and
            compatibility.get("runtime_python_compilation") is False and
            compatibility.get("precompiled_device_kernels_required") is True and
            compatibility.get("target_version") == "CANN-8.1.RC1 host ABI and Ascend C device compiler",
            "private project does not attest source-preserving tiling/kernel behavior")

    source_file = project / "src/index/gather_elements_v2/op_host/gather_elements_v2_tiling.cpp"
    audit_header = project / "src/index/gather_elements_v2/op_host/gather_elements_v2_source_audit.h"
    private_opapi_source = project / "src/index/gather_elements_v2/op_host/op_api/aclnn_gather_elements_v2_private.cpp"
    config = package / "op_impl/ai_core/tbe/config/ascend910b/aic-ascend910b-ops-info.json"
    kernel_root = package / "op_impl/ai_core/tbe/kernel/ascend910b/gather_elements_v2"
    kernel_config_root = package / "op_impl/ai_core/tbe/kernel/config/ascend910b"
    binary_info_config = kernel_config_root / "binary_info_config.json"
    operator_binary_config = kernel_config_root / "gather_elements_v2.json"
    op_api = package / "op_api/lib/libcust_opapi.so"
    proto = architecture_library(package, "op_proto/lib/linux/{architecture}/libcust_opsproto_rt2.0.so")
    tiling = architecture_library(package, "op_impl/ai_core/tbe/op_tiling/lib/linux/{architecture}/libcust_opmaster_rt2.0.so")
    for path in (source_file, audit_header, private_opapi_source, config, binary_info_config,
                 operator_binary_config, op_api):
        require(path.is_file(), "private GatherElementsV2 package file is absent: {}".format(path))
    op_api_bytes = op_api.read_bytes()
    require(b"aclnnPrivateGatherElementsV2GetWorkspaceSize" in op_api_bytes and
            b"aclnnPrivateGatherElementsV2" in op_api_bytes,
            "private GatherElementsV2 package lacks its CANN-8.1 level-2 OpAPI exports")
    source_provenance = provenance.get("provenance", {})
    require(source_provenance.get("cann81_private_opapi_sha256") == digest(private_opapi_source),
            "private GatherElementsV2 CANN-8.1 OpAPI source attestation mismatches")
    kernel_objects = sorted(kernel_root.glob("*.o"))
    require(len(kernel_objects) == 4,
            "CANN 8.1 package must contain four precompiled GatherElementsV2 dtype kernels; found={}".format(
                len(kernel_objects)))
    kernel_metadata = [path.with_suffix(".json") for path in kernel_objects]
    require(all(path.is_file() for path in kernel_metadata),
            "a precompiled GatherElementsV2 kernel lacks matching CANN metadata")
    dynamic_root = package / "op_impl/ai_core/tbe/gather_elements_source_impl/dynamic"
    require(not dynamic_root.exists() or not any(dynamic_root.glob("*.py")),
            "runtime package unexpectedly contains a Python/TBE operator adapter")
    require(not dynamic_root.exists() or not any(dynamic_root.glob("*.cpp")),
            "runtime package unexpectedly contains Ascend C source instead of precompiled kernels")
    config_data = json.loads(config.read_text(encoding="utf-8"))
    op_data = config_data.get(OPERATOR)
    require(isinstance(op_data, dict) and op_data.get("opFile", {}).get("value") == "gather_elements_v2" and
            op_data.get("opInterface", {}).get("value") == "gather_elements_v2",
            "private GatherElementsV2 config does not register its exact generic-dispatch interface")
    audit_text = audit_header.read_text(encoding="utf-8")
    require("GATHER_ELEMENTS_V2_SOURCE_AUDIT_V1" in audit_text and AUDIT_SCHEMA in audit_text and
            "GATHER_ELEMENTS_SOURCE_AIV_CAP" in audit_text and "GATHER_ELEMENTS_SOURCE_UB_DIVISOR" in audit_text,
            "private tiler lacks the bounded-input/raw-tiling audit contract")

    manifest = {
        "schema": SCHEMA,
        "operator": OPERATOR,
        "source_operator_type": OPERATOR,
        "runtime_op": "gather_elements",
        "vendor": VENDOR,
        "source_kind": provenance["source_kind"],
        "cann_root": str(cann),
        "cann_version_file_sha256": digest(cann / "opp" / "version.info"),
        "project_root": str(project),
        "package_root": str(package),
        "source_file": str(source_file),
        "source_file_sha256": digest(source_file),
        "op_api_library": str(op_api),
        "op_api_library_sha256": digest(op_api),
        "op_proto_library": str(proto),
        "op_proto_library_sha256": digest(proto),
        "op_tiling_library": str(tiling),
        "op_tiling_library_sha256": digest(tiling),
        "ops_config": str(config),
        "ops_config_sha256": digest(config),
        "kernel_binary_root": str(kernel_root),
        "kernel_binary_info_config": str(binary_info_config),
        "kernel_binary_info_config_sha256": digest(binary_info_config),
        "kernel_operator_config": str(operator_binary_config),
        "kernel_operator_config_sha256": digest(operator_binary_config),
        "precompiled_device_kernels": [
            {"object": str(obj), "object_sha256": digest(obj),
             "metadata": str(meta), "metadata_sha256": digest(meta)}
            for obj, meta in zip(kernel_objects, kernel_metadata)
        ],
        "runtime_python_compilation": False,
        "build_cann_version": "8.1.RC1",
        "instrumentation": {
            "enabled": True,
            "mutates_tiling_context": False,
            "audit_schema": AUDIT_SCHEMA,
            "audit_environment": "GATHER_ELEMENTS_TILING_AUDIT_PATH",
            "source_budget_environment": "GATHER_ELEMENTS_SOURCE_AIV_CAP",
            "dispatch_environment": "GATHER_ELEMENTS_SOURCE_DISPATCH",
            "dispatch_value": "cann81_prebuilt_aclnn",
            "operator_type_environment": "GATHER_ELEMENTS_SOURCE_OPERATOR_TYPE",
            "opapi_library_environment": "GATHER_ELEMENTS_SOURCE_OPAPI_LIBRARY",
        },
        "hardware_envelope_heuristic": {
            "enabled": True,
            "environment": "GATHER_ELEMENTS_SOURCE_UB_DIVISOR",
            "audit_field": "ub_cap_divisor",
            "resource": "source_visible_ub_capacity",
            "divisors": [2, 4, 8],
            "max_anchors": 16,
        },
        "compatibility_port": compatibility,
        "strategy_algorithm_changes": False,
        "kernel_algorithm_changes": False,
        "toolkit_install_modified": False,
        "matmul_included": False,
        "formal_data_gate": "the CANN 8.1 level-2 OpAPI and L0 launcher must load from the private package, emit one C++ host-tiler raw identity, launch a precompiled Ascend C kernel, and exactly match the deterministic GatherElements coordinate reference; rank-one preflight also matches installed aclnnGather",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "operator": OPERATOR, "package_root": str(package),
                      "toolkit_install_modified": False, "npu_calls": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, json.JSONDecodeError) as error:
        print("GatherElementsV2 package validation error: {}".format(error), file=__import__("sys").stderr)
        raise SystemExit(2)
