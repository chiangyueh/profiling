#!/usr/bin/env python3
"""Create a private attention OPP from one bridge and exact installed CANN 8.1 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import tempfile
from pathlib import Path


SPECS = {
    "flash_attention_score_grad": {
        "operator": "FlashAttentionScoreGrad",
        "vendor": "fasg_source",
        "audit_schema": "fasg_raw_tiling_observation_v2",
        "audit_env": "FASG_TILING_AUDIT_PATH",
        "budget_env": "FASG_SOURCE_AIV_CAP",
        "capacity_env": "FASG_SOURCE_L2_DIVISOR",
        "capacity_field": "l2_cap_divisor",
        "official_library_env": "FASG_OFFICIAL_TILING_LIBRARY",
        "custom_library_env": "FASG_SOURCE_TILING_LIBRARY",
    },
    "fused_infer_attention_score": {
        "operator": "FusedInferAttentionScore",
        "vendor": "fias_source",
        "audit_schema": "fias_raw_tiling_observation_v2",
        "audit_env": "FIAS_TILING_AUDIT_PATH",
        "budget_env": "FIAS_SOURCE_AIV_CAP",
        "capacity_env": "FIAS_SOURCE_UB_DIVISOR",
        "capacity_field": "ub_cap_divisor",
        "official_library_env": "FIAS_OFFICIAL_TILING_LIBRARY",
        "custom_library_env": "FIAS_SOURCE_TILING_LIBRARY",
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


def installed_library(cann: Path, architecture: str, relative: Path) -> tuple[Path, Path]:
    candidates = (cann / (architecture + "-linux") / relative, cann / relative)
    for candidate in candidates:
        if candidate.is_file():
            return candidate, candidate.resolve()
    raise RuntimeError("installed CANN 8.1 library is absent: {}".format(relative))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", required=True, choices=tuple(SPECS))
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--build-root", required=True, type=Path)
    parser.add_argument("--cann-root", required=True, type=Path)
    parser.add_argument("--package-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    spec = SPECS[args.operator]
    project = args.project.resolve()
    build = args.build_root.resolve()
    cann = args.cann_root.resolve()
    package = args.package_root.resolve()
    version = cann / "opp/version.info"
    require(version.is_file() and "version_dir=8.1.RC1" in version.read_text(encoding="utf-8"),
            "attention campaign requires installed CANN 8.1.RC1")
    overlay_path = project / "source_candidate_overlay.json"
    require(overlay_path.is_file(), "exact CANN 8.1 delegate manifest is absent")
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    require(overlay.get("schema") == "remaining_attention_exact_cann81_delegate_v1" and
            overlay.get("runtime_op") == args.operator and
            overlay.get("alpha001_host_tiling_compiled") is False and
            overlay.get("official_dispatch_delegated_without_strategy_forcing") is True,
            "attention delegate provenance is invalid")
    delegate = project / str(overlay["delegate_source"])
    require(delegate.is_file() and digest(delegate) == overlay["delegate_source_sha256"],
            "attention delegate source is missing or changed")
    built_tiling = build / "libcust_opmaster_rt2.0.so"
    require(built_tiling.is_file(), "exact CANN 8.1 bridge library was not built")
    compiled = sorted(path for path in (build / "CMakeFiles/optiling.dir").rglob("*.o") if path.is_file())
    require(compiled and all(path.name in ("fallback_comm.cpp.o", "exact_cann81_tiling_delegate.cpp.o")
                             for path in compiled),
            "alpha001 attention tiler object entered the exact CANN 8.1 bridge")
    architecture = platform.machine()
    official_tiling_entrypoint = (cann / "opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux" /
                                  architecture / "liboptiling.so")
    official_tiling = official_tiling_entrypoint.resolve()
    require(official_tiling.is_file(), "installed exact CANN 8.1 host tiler is absent")
    opapi_entrypoint, opapi = installed_library(cann, architecture, Path("lib64/libopapi.so"))
    installed_config = (cann / "opp/built-in/op_impl/ai_core/tbe/config/ascend910b" /
                        "aic-ascend910b-ops-info.json").resolve()
    require(installed_config.is_file(), "installed exact CANN 8.1 op config is absent")
    config = json.loads(installed_config.read_text(encoding="utf-8"))
    require(spec["operator"] in config, "installed op config does not contain {}".format(spec["operator"]))
    installed_kernel = (cann / "opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b" /
                        args.operator).resolve()
    require(installed_kernel.is_dir(), "installed exact CANN 8.1 kernel directory is absent")
    kernel_objects = sorted(installed_kernel.rglob("*.o"))
    require(kernel_objects, "installed exact CANN 8.1 kernel objects are absent")
    require(not package.exists(), "refuse to overwrite an existing private attention package")
    package.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".exact_cann81_attention.", dir=package.parent))
    try:
        custom_tiling = (temporary / "op_impl/ai_core/tbe/op_tiling/lib/linux" /
                         architecture / "libcust_opmaster_rt2.0.so")
        custom_tiling.parent.mkdir(parents=True)
        shutil.copy2(built_tiling, custom_tiling)
        private_config = (temporary / "op_impl/ai_core/tbe/config/ascend910b" /
                          "aic-ascend910b-ops-info.json")
        private_config.parent.mkdir(parents=True)
        private_config.write_text(json.dumps({spec["operator"]: config[spec["operator"]]},
                                             ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        private_kernel = temporary / "op_impl/ai_core/tbe/kernel/ascend910b" / args.operator
        copied = []
        for source_object in kernel_objects:
            source_metadata = source_object.with_suffix(".json")
            require(source_metadata.is_file(), "kernel metadata is absent: {}".format(source_metadata))
            relative = source_object.relative_to(installed_kernel)
            target_object = private_kernel / relative
            target_metadata = target_object.with_suffix(".json")
            target_object.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_object, target_object)
            shutil.copy2(source_metadata, target_metadata)
            require(digest(source_object) == digest(target_object) and
                    digest(source_metadata) == digest(target_metadata), "private kernel copy mismatch")
            copied.append({
                "object": str((package / target_object.relative_to(temporary)).resolve()),
                "object_sha256": digest(target_object),
                "metadata": str((package / target_metadata.relative_to(temporary)).resolve()),
                "metadata_sha256": digest(target_metadata),
                "installed_object": str(source_object),
                "installed_metadata": str(source_metadata),
            })
        (temporary / "version.info").write_text(
            "version=1.0\nmaster_version=1.0\ncompatible_version=[1.0]\n",
            encoding="utf-8")
        os.replace(temporary, package)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    custom_tiling = (package / "op_impl/ai_core/tbe/op_tiling/lib/linux" /
                     architecture / "libcust_opmaster_rt2.0.so")
    private_config = (package / "op_impl/ai_core/tbe/config/ascend910b" /
                      "aic-ascend910b-ops-info.json")
    manifest = {
        "schema": "remaining_operator_exact_cann81_package_v5",
        "operator": spec["operator"],
        "runtime_op": args.operator,
        "strategy_class": "official_semantic_dispatch",
        "official_dispatch_delegated_without_strategy_forcing": True,
        "source_kind": "installed_cann81_rc1_binary_tiler_with_process_local_hardware_view",
        "build_cann_version": "8.1.RC1",
        "cann_root": str(cann),
        "cann_version_file_sha256": digest(version),
        "project_root": str(project),
        "package_root": str(package),
        "source_file": str(delegate),
        "source_file_sha256": digest(delegate),
        "official_tiling_source_sha256": digest(official_tiling),
        "official_tiling_entrypoint": str(official_tiling_entrypoint),
        "official_tiling_library": str(official_tiling),
        "official_tiling_library_sha256": digest(official_tiling),
        "op_tiling_library": str(custom_tiling),
        "op_tiling_library_sha256": digest(custom_tiling),
        "op_api_entrypoint": str(opapi_entrypoint),
        "op_api_library": str(opapi),
        "op_api_library_sha256": digest(opapi),
        "ops_config": str(private_config),
        "ops_config_sha256": digest(private_config),
        "installed_ops_config": str(installed_config),
        "installed_operator_config_sha256": hashlib.sha256(
            json.dumps(config[spec["operator"]], ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
        "precompiled_device_kernels": copied,
        "device_kernel_origin": "installed_cann81_binary_package_private_copy",
        "instrumentation": {
            "enabled": True,
            "mutates_tiling_output_fields": False,
            "temporarily_mutates_compile_info_before_official_tiler":
                args.operator == "flash_attention_score_grad",
            "mutates_process_local_platform_view_before_official_tiler": True,
            "audit_schema": spec["audit_schema"],
            "audit_environment": spec["audit_env"],
            "source_budget_environment": spec["budget_env"],
            "dispatch_environment": ("FASG_SOURCE_DISPATCH" if args.operator == "flash_attention_score_grad"
                                     else "FIAS_SOURCE_DISPATCH"),
            "dispatch_value": "exact_cann81_installed_delegate",
            "official_tiling_library_environment": spec["official_library_env"],
            "custom_tiling_library_environment": spec["custom_library_env"],
        },
        "hardware_envelope_heuristic": {
            "enabled": True,
            "environment": spec["capacity_env"],
            "audit_field": spec["capacity_field"],
            "resource": ("process_local_l2_capacity" if args.operator == "flash_attention_score_grad"
                         else "process_local_ub_capacity"),
            "divisors": [2, 4, 8],
            "max_anchors": 16,
        },
        "strategy_algorithm_changes": False,
        "kernel_algorithm_changes": False,
        "runtime_python_compilation": False,
        "alpha001_host_tiling_compiled": False,
        "installed_cann_writes": 0,
        "toolkit_install_modified": False,
        "matmul_included": False,
        "scatter_elements_included": False,
        "formal_data_gate": (
            "load the exact installed CANN 8.1 RC1 host tiler before the one-function private bridge; "
            "run the official semantic dispatcher under a process-local bounded hardware view; launch exact "
            "installed CANN 8.1 OpAPI and copied kernel artifacts; retain only exact-output validated rows"
        ),
    }
    if args.operator == "flash_attention_score_grad":
        manifest.update({
            "original_strategy_registry_preserved": True,
            "original_strategy_registry_origin": "installed_cann81_rc1_liboptiling",
            "disabled_original_registrations": [],
        })
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    print(json.dumps({"status": "passed", "operator": args.operator,
                      "host_tiler": "installed_exact_cann81_rc1", "kernel_pairs": len(copied)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, json.JSONDecodeError) as error:
        print("exact CANN-8.1 attention package error: {}".format(error), file=os.sys.stderr)
        raise SystemExit(2)
