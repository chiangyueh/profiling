#!/usr/bin/env python3
"""Validate one checkout-local, official CANN-8.1 ScatterElementsV2 package."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path


SCHEMA = "scatter_elements_v2_cann81_package_v1"
OPERATOR = "ScatterElementsV2"
VENDOR = "scatter_elements_source"
AUDIT_SCHEMA = "scatter_elements_raw_tiling_observation_v1"
OFFICIAL_TILING_SHA256 = "af4fa87f4760e73b93a31a301827e9e2c286c58f3ff32a0be7344adf2c5543f7"


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
    for architecture in (platform.machine(), "aarch64", "x86_64"):
        path = root / relative.format(architecture=architecture)
        if path.is_file():
            return path
    raise RuntimeError("package library is absent: {}".format(relative))


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
    version = cann / "opp/version.info"
    require(project.is_dir() and package.is_dir(), "private project or package is absent")
    require(version.is_file() and "version_dir=8.1.RC1" in version.read_text(encoding="utf-8"),
            "ScatterElementsV2 requires CANN 8.1.RC1")

    overlay_manifest = project / "source_candidate_overlay.json"
    source = project / "src/index/scatter_elements_v2/op_host/scatter_elements_v2_tiling.cc"
    config = package / "op_impl/ai_core/tbe/config/ascend910b/aic-ascend910b-ops-info.json"
    op_api = package / "op_api/lib/libcust_opapi.so"
    proto = architecture_library(package, "op_proto/lib/linux/{architecture}/libcust_opsproto_rt2.0.so")
    tiling = architecture_library(package, "op_impl/ai_core/tbe/op_tiling/lib/linux/{architecture}/libcust_opmaster_rt2.0.so")
    for path in (overlay_manifest, source, config, op_api, proto, tiling):
        require(path.is_file(), "package input is absent: {}".format(path))

    overlay = json.loads(overlay_manifest.read_text(encoding="utf-8"))
    require(overlay.get("operator") == OPERATOR and overlay.get("official_commit") ==
            "c214b710edbe24017dc7dc92170a50bd8ff38171", "wrong official 8.1 source provenance")
    modified = overlay.get("modified_source_files")
    require(isinstance(modified, list) and len(modified) == 1 and
            modified[0].get("sha256_before") == OFFICIAL_TILING_SHA256,
            "wrong official ScatterElementsV2 host-tiler source")
    require(overlay.get("strategy_algorithm_changes") is False,
            "source overlay must retain the official tiling algorithm")
    source_text = source.read_text(encoding="utf-8")
    require("SCATTER_ELEMENTS_SOURCE_TILING_AUDIT_V1" in source_text and
            "SCATTER_ELEMENTS_SOURCE_AIV_CAP_V1" in source_text and
            "source_visible_max_ub" in source_text,
            "bounded input/audit instrumentation is absent")

    config_data = json.loads(config.read_text(encoding="utf-8"))
    require(OPERATOR in config_data, "package does not register ScatterElementsV2 for ascend910b")
    kernel_root = package / "op_impl/ai_core/tbe/kernel/ascend910b/scatter_elements_v2"
    kernel_objects = sorted(kernel_root.glob("*.o")) if kernel_root.is_dir() else []
    require(kernel_objects, "package has no precompiled Ascend910B ScatterElementsV2 kernel")
    metadata = [path.with_suffix(".json") for path in kernel_objects]
    require(all(path.is_file() for path in metadata), "kernel metadata is incomplete")
    dynamic_root = package / "op_impl/ai_core/tbe/scatter_elements_source_impl/dynamic"
    require(not dynamic_root.exists() or not any(dynamic_root.iterdir()),
            "runtime package contains source and could invoke Python/TBE compilation")

    manifest = {
        "schema": SCHEMA,
        "operator": OPERATOR,
        "runtime_op": "scatter_elements",
        "vendor": VENDOR,
        "official_source_commit": overlay["official_commit"],
        "source_kind": "official_cann_ops_cann81_native",
        "build_cann_version": "8.1.RC1",
        "cann_root": str(cann),
        "cann_version_file_sha256": digest(version),
        "project_root": str(project),
        "package_root": str(package),
        "source_file": str(source),
        "source_file_sha256": digest(source),
        "official_tiling_source_sha256": OFFICIAL_TILING_SHA256,
        "op_api_library": str(op_api),
        "op_api_library_sha256": digest(op_api),
        "op_proto_library": str(proto),
        "op_proto_library_sha256": digest(proto),
        "op_tiling_library": str(tiling),
        "op_tiling_library_sha256": digest(tiling),
        "ops_config": str(config),
        "ops_config_sha256": digest(config),
        "precompiled_device_kernels": [
            {"object": str(obj), "object_sha256": digest(obj),
             "metadata": str(meta), "metadata_sha256": digest(meta)}
            for obj, meta in zip(kernel_objects, metadata)
        ],
        "instrumentation": {
            "enabled": True,
            "mutates_tiling_context": False,
            "audit_schema": AUDIT_SCHEMA,
            "audit_environment": "SCATTER_ELEMENTS_TILING_AUDIT_PATH",
            "source_budget_environment": "SCATTER_ELEMENTS_SOURCE_AIV_CAP",
            "dispatch_environment": "SCATTER_ELEMENTS_SOURCE_DISPATCH",
            "dispatch_value": "cann81_native_aclnn",
            "opapi_library_environment": "SCATTER_ELEMENTS_SOURCE_OPAPI_LIBRARY",
        },
        "hardware_envelope_heuristic": overlay["hardware_envelope_heuristic"],
        "strategy_algorithm_changes": False,
        "kernel_algorithm_changes": False,
        "runtime_python_compilation": False,
        "toolkit_install_modified": False,
        "matmul_included": False,
        "formal_data_gate": "private CANN 8.1 host tiler, precompiled kernel launch, and exact installed aclnnScatter output equality",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "operator": OPERATOR, "kernel_count": len(kernel_objects)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, json.JSONDecodeError) as error:
        print("ScatterElementsV2 package validation error: {}".format(error), file=__import__("sys").stderr)
        raise SystemExit(2)
