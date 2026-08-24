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


SCHEMA = "gather_elements_v2_private_cann_package_v1"
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
    # CANN's generated install.sh copies packages/vendors/<vendor> below the
    # directory named by ASCEND_CUSTOM_OPP_PATH.  The runtime variable must
    # therefore name this private packages root, not the vendor directory.
    custom_opp_root = package.parent.parent
    require(custom_opp_root / "vendors" / VENDOR == package,
            "private GatherElementsV2 package is not below packages/vendors/<vendor>")
    require((cann / "opp" / "version.info").is_file(), "CANN OPP version file is absent")
    project_manifest = project / "gather_elements_v2_project.json"
    require(project_manifest.is_file(), "private project provenance is absent: {}".format(project_manifest))
    provenance = json.loads(project_manifest.read_text(encoding="utf-8"))
    require(provenance.get("operator") == OPERATOR and provenance.get("vendor") == VENDOR,
            "private project has the wrong operator or vendor")
    require(provenance.get("toolkit_install_modified") is False,
            "private project does not attest a read-only CANN installation")
    compatibility = provenance.get("compatibility_port")
    require(isinstance(compatibility, dict) and compatibility.get("tiling_algorithm_changes") is False and
            compatibility.get("kernel_algorithm_changes") is False,
            "private project does not attest source-preserving tiling/kernel behavior")

    source_file = project / "src/index/gather_elements_v2/op_host/gather_elements_v2_tiling.cpp"
    audit_header = project / "src/index/gather_elements_v2/op_host/gather_elements_v2_source_audit.h"
    config = package / "op_impl/ai_core/tbe/config/ascend910b/aic-ascend910b-ops-info.json"
    dynamic_root = package / "op_impl/ai_core/tbe/gather_elements_source_impl/dynamic"
    adapter = dynamic_root / "gather_elements_v2.py"
    kernel = dynamic_root / "gather_elements_v2.cpp"
    proto = architecture_library(package, "op_proto/lib/linux/{architecture}/libcust_opsproto_rt2.0.so")
    tiling = architecture_library(package, "op_impl/ai_core/tbe/op_tiling/lib/linux/{architecture}/libcust_opmaster_rt2.0.so")
    for path in (source_file, audit_header, config, adapter, kernel):
        require(path.is_file(), "private GatherElementsV2 package file is absent: {}".format(path))
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
        "custom_opp_root": str(custom_opp_root),
        "source_file": str(source_file),
        "source_file_sha256": digest(source_file),
        "op_proto_library": str(proto),
        "op_proto_library_sha256": digest(proto),
        "op_tiling_library": str(tiling),
        "op_tiling_library_sha256": digest(tiling),
        "ops_config": str(config),
        "ops_config_sha256": digest(config),
        "dynamic_adapter": str(adapter),
        "dynamic_adapter_sha256": digest(adapter),
        "dynamic_kernel": str(kernel),
        "dynamic_kernel_sha256": digest(kernel),
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
        "compatibility_port": compatibility,
        "strategy_algorithm_changes": False,
        "kernel_algorithm_changes": False,
        "toolkit_install_modified": False,
        "matmul_included": False,
        "formal_data_gate": "the private CANN package must be selected by aclopCompileAndExecute, emit one host-tiler raw identity, launch, and exactly match installed aclnnGather",
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
