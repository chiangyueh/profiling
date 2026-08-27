#!/usr/bin/env python3
"""Attest one complete source-instrumented CANN-8.1 attention package."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOCK = json.loads((ROOT / "remaining_operators_cann81_lock.json").read_text(encoding="utf-8"))
SPECS = {
    "flash_attention_score_grad": {
        "operator": "FlashAttentionScoreGrad",
        "source_family": "cann_ops_adv",
        "audit_schema": "fasg_raw_tiling_observation_v1",
        "audit_env": "FASG_TILING_AUDIT_PATH",
        "budget_env": "FASG_SOURCE_AIV_CAP",
        "envelope_env": "FASG_SOURCE_L2_DIVISOR",
        "dispatch_env": "FASG_SOURCE_DISPATCH",
        "dispatch_value": "cann81_prebuilt_aclnn",
        "opapi_env": "FASG_SOURCE_OPAPI_LIBRARY",
        "cmake_op": "flash_attention_score_grad",
    },
    "fused_infer_attention_score": {
        "operator": "FusedInferAttentionScore",
        "source_family": "cann_ops_adv",
        "audit_schema": "fias_raw_tiling_observation_v1",
        "audit_env": "FIAS_TILING_AUDIT_PATH",
        "budget_env": "FIAS_SOURCE_AIV_CAP",
        "envelope_env": "FIAS_SOURCE_UB_DIVISOR",
        "dispatch_env": "FIAS_SOURCE_DISPATCH",
        "dispatch_value": "cann81_prebuilt_aclnn",
        "opapi_env": "FIAS_SOURCE_OPAPI_LIBRARY",
        "cmake_op": "fused_infer_attention_score",
    },
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def unique_file(root: Path, pattern: str, label: str) -> Path:
    matches = sorted(path.resolve() for path in root.glob(pattern) if path.is_file())
    require(len(matches) == 1, "expected one {} under {}, found {}".format(label, root, len(matches)))
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", required=True, choices=tuple(SPECS))
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--build-root", required=True, type=Path)
    parser.add_argument("--package-root", required=True, type=Path)
    parser.add_argument("--kernel-copy-manifest", required=True, type=Path)
    parser.add_argument("--cann-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    spec = SPECS[args.operator]
    locked = LOCK["operators"][args.operator]
    project = args.project.resolve()
    build = args.build_root.resolve()
    package = args.package_root.resolve()
    cann = args.cann_root.resolve()
    version = cann / "opp/version.info"

    require(project.is_dir() and build.is_dir() and package.is_dir(),
            "private project, build, or package root is absent")
    require(version.is_file() and "version_dir=8.1.RC1" in version.read_text(encoding="utf-8"),
            "remaining operators require CANN 8.1.RC1")
    require(args.kernel_copy_manifest.is_file(), "installed-kernel copy manifest is absent")
    kernel_copy = json.loads(args.kernel_copy_manifest.read_text(encoding="utf-8"))
    require(kernel_copy.get("schema") == "installed_cann81_attention_kernel_copy_v1" and
            kernel_copy.get("operator") == args.operator and
            Path(kernel_copy.get("cann_root", "")).resolve() == cann and
            kernel_copy.get("opp_version_sha256") == digest(version) and
            kernel_copy.get("source_read_only") is True and
            kernel_copy.get("toolkit_install_modified") is False,
            "installed CANN-8.1 kernel provenance is invalid")
    overlay_path = project / "source_candidate_overlay.json"
    require(overlay_path.is_file(), "source overlay manifest is absent")
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    require(overlay.get("operator") == spec["operator"] and
            overlay.get("strategy_algorithm_changes") is False and
            overlay.get("kernel_algorithm_changes") is False,
            "wrong or source-changing overlay")
    if args.operator == "flash_attention_score_grad":
        enabled = overlay.get("enabled_original_registrations")
        require(overlay.get("schema") == "fasg_original_dispatcher_overlay_v1" and
                overlay.get("strategy_class") == "official_semantic_dispatch" and
                overlay.get("original_strategy_registry_preserved") is True and
                isinstance(enabled, list) and len(enabled) == 8 and
                len({(row.get("class"), row.get("priority")) for row in enabled
                     if isinstance(row, dict)}) == 8 and
                overlay.get("disabled_original_registrations") == [],
                "FASG overlay does not preserve all eight official dispatcher registrations")
    else:
        require(overlay.get("schema") == "fias_original_dispatch_overlay_v1" and
                overlay.get("source_dispatch", {}).get("forced_branch") is False,
                "FIAS overlay is not the original semantic dispatcher")
    expected_commit = LOCK["sources"][spec["source_family"]]["official_commit"]
    require(overlay.get("official_commit") == expected_commit, "wrong official CANN-8.1 commit")

    modified = overlay.get("modified_source_files", overlay.get("modified_registration_files"))
    require(isinstance(modified, list) and modified, "overlay has no attested host-tiler modification")
    for row in modified:
        require(isinstance(row, dict) and isinstance(row.get("path"), str) and
                isinstance(row.get("sha256_after"), str), "invalid modified-source attestation")
        changed_file = project / row["path"]
        require(changed_file.is_file() and digest(changed_file) == row["sha256_after"],
                "instrumented source differs from attestation: {}".format(changed_file))

    harness = overlay.get("build_harness")
    expected_cmake_hash = (harness.get("generated_cmake_sha256") if isinstance(harness, dict)
                           else overlay.get("build_harness_sha256"))
    require(isinstance(expected_cmake_hash, str) and digest(project / "CMakeLists.txt") == expected_cmake_hash,
            "CANN-8.1 build harness differs from attestation")

    if args.operator == "fused_infer_attention_score":
        source_file = project / locked["relative_root"] / locked["tiler"]
        official_hash = locked["tiler_sha256"]
    else:
        source_file = project / locked["relative_root"] / "ophost/flash_attention_score_grad_tiling.cpp"
        official_hash = locked["entry_sha256"]
    require(source_file.is_file(), "instrumented host tiler is absent")

    tiling = unique_file(package, "op_impl/ai_core/tbe/op_tiling/lib/linux/*/libcust_opmaster_rt2.0.so",
                         "packaged host-tiler library")
    built_tiling = build / "libcust_opmaster_rt2.0.so"
    require(built_tiling.is_file() and digest(built_tiling) == digest(tiling),
            "packaged host tiler is not the instrumented library from this strategy build")
    opapi = package / "op_api/lib/libcust_opapi.so"
    require(opapi.is_file(), "private CANN-8.1 OpAPI library is absent")
    ops_config = unique_file(package, "op_impl/ai_core/tbe/config/ascend910b/*-ops-info.json",
                             "Ascend910B op-info config")
    proto = unique_file(package, "op_proto/lib/linux/*/libcust_opsproto_rt2.0.so", "op-proto library")
    kernel_root = package / "op_impl/ai_core/tbe/kernel/ascend910b" / spec["cmake_op"]
    require(Path(kernel_copy.get("private_kernel_root", "")).resolve() == kernel_root.resolve(),
            "private kernel directory differs from its copy manifest")
    installed_kernel_root = (cann / "opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b" /
                             spec["cmake_op"]).resolve()
    require(Path(kernel_copy.get("installed_kernel_root", "")).resolve() == installed_kernel_root,
            "kernel copy did not originate from the installed CANN-8.1 binary package")
    kernel_objects = sorted(kernel_root.rglob("*.o")) if kernel_root.is_dir() else []
    require(kernel_objects, "private package has no precompiled Ascend910B kernel objects for {}".format(
        spec["operator"]))
    precompiled_kernels = []
    for kernel_object in kernel_objects:
        metadata = kernel_object.with_suffix(".json")
        require(metadata.is_file(), "kernel metadata is absent: {}".format(metadata))
        relative = kernel_object.relative_to(kernel_root)
        installed_object = installed_kernel_root / relative
        installed_metadata = installed_object.with_suffix(".json")
        require(installed_object.is_file() and installed_metadata.is_file() and
                digest(installed_object) == digest(kernel_object) and
                digest(installed_metadata) == digest(metadata),
                "private kernel differs from its installed CANN-8.1 binary")
        precompiled_kernels.append({
            "object": str(kernel_object.resolve()), "object_sha256": digest(kernel_object),
            "metadata": str(metadata.resolve()), "metadata_sha256": digest(metadata),
            "installed_object": str(installed_object), "installed_metadata": str(installed_metadata),
        })
    runtime_sources = [path for path in package.glob("op_impl/ai_core/tbe/*/dynamic/**/*")
                       if path.is_file() and path.suffix in (".py", ".cpp", ".cc")]
    require(not runtime_sources, "private attention package unexpectedly installs runtime source")
    envelope = overlay.get("hardware_envelope_heuristic")
    require(isinstance(envelope, dict) and envelope.get("environment") == spec["envelope_env"] and
            tuple(envelope.get("divisors", ())) == (2, 4, 8), "hardware-envelope contract is invalid")

    manifest = {
        "schema": "remaining_operator_cann81_prebuilt_package_v3",
        "operator": spec["operator"],
        "runtime_op": args.operator,
        "strategy_class": (overlay.get("strategy_class") if args.operator == "flash_attention_score_grad"
                           else "original_semantic_dispatch"),
        "original_strategy_registry_preserved": (
            overlay.get("original_strategy_registry_preserved") is True),
        "enabled_original_registrations": overlay.get("enabled_original_registrations", []),
        "disabled_original_registrations": overlay.get("disabled_original_registrations", []),
        "source_kind": "official_{}_cann81_private_prebuilt_package".format(spec["source_family"]),
        "official_source_commit": expected_commit,
        "build_cann_version": "8.1.RC1",
        "cann_root": str(cann),
        "cann_version_file_sha256": digest(version),
        "project_root": str(project),
        "package_root": str(package),
        "source_file": str(source_file),
        "source_file_sha256": digest(source_file),
        "official_tiling_source_sha256": official_hash,
        "op_tiling_library": str(tiling),
        "op_tiling_library_sha256": digest(tiling),
        "op_api_library": str(opapi.resolve()),
        "op_api_library_sha256": digest(opapi),
        "op_proto_library": str(proto),
        "op_proto_library_sha256": digest(proto),
        "ops_config": str(ops_config),
        "ops_config_sha256": digest(ops_config),
        "precompiled_device_kernels": precompiled_kernels,
        "device_kernel_origin": "installed_cann81_binary_package_private_copy",
        "installed_kernel_copy_manifest": str(args.kernel_copy_manifest.resolve()),
        "installed_kernel_copy_manifest_sha256": digest(args.kernel_copy_manifest),
        "instrumentation": {
            "enabled": True,
            "mutates_tiling_context": False,
            "audit_schema": spec["audit_schema"],
            "audit_environment": spec["audit_env"],
            "source_budget_environment": spec["budget_env"],
            "dispatch_environment": spec["dispatch_env"],
            "dispatch_value": spec["dispatch_value"],
            "opapi_library_environment": spec["opapi_env"],
        },
        "hardware_envelope_heuristic": envelope,
        "strategy_algorithm_changes": False,
        "kernel_algorithm_changes": False,
        "runtime_python_compilation": False,
        "toolkit_install_modified": False,
        "matmul_included": False,
        "scatter_elements_included": False,
        "formal_data_gate": (
            "load the complete private CANN 8.1 package and its private OpAPI, preserve the original semantic "
            "dispatcher, observe the instrumented raw "
            "tiling identity, launch the read-only copy of the installed CANN-8.1 Ascend910B kernel, and exactly match the "
            "separately generated installed-reference output"
        ),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    print(json.dumps({"status": "passed", "operator": spec["operator"],
                      "device_kernel_origin": manifest["device_kernel_origin"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, json.JSONDecodeError) as error:
        print("remaining CANN-8.1 host-tiler validation error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
