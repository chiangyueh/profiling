#!/usr/bin/env python3
"""Create an isolated custom OPP root with a source-built host tiler.

The installed CANN 8.1 operators are dynamic: their official device source is
already delivered under the built-in OPP tree and the runtime compiles only the
tiling key needed by an actual workload.  FlashAttentionScoreGrad alone has
884 device-key branches, so eagerly rebuilding all of them merely to replace
the host tiler is not a viable collection precondition.

This helper creates a minimal private OPP root: its ``built-in`` entry is a
read-only symlink to the installed OPP tree and ``vendors/config.ini`` selects
the source-built tiler. It copies the exact installed dynamic device
implementation/configuration for the operator and installs only the two host
tiling libraries built from the pinned source overlay. No kernel source, raw
tiling field, or installed device binary is modified. The controller requires
a source audit, a real execution, and exact output equality before recording
any latency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any


RUNTIME_LAYOUT = {
    "FlashAttentionScoreGrad": {
        "config_key": "FlashAttentionScoreGrad", "dynamic_file": "flash_attention_score_grad.py",
        "ascendc_dirs": ("flash_attention_score_grad",),
    },
    "FusedInferAttentionScore": {
        "config_key": "FusedInferAttentionScore", "dynamic_file": "fused_infer_attention_score.py",
        "ascendc_dirs": ("fused_infer_attention_score", "incre_flash_attention", "prompt_flash_attention"),
    },
    "GatherElementsV2": {
        # ACLNN GatherElements uses the installed GatherElements dynamic
        # device implementation; the source overlay replaces only its tiler.
        "config_key": "GatherElements", "dynamic_file": "gather_elements.py", "ascendc_dirs": (),
    },
    "ScatterElementsV2": {
        "config_key": "ScatterElementsV2", "dynamic_file": "scatter_elements_v2.py",
        "ascendc_dirs": ("scatter_elements_v2",),
    },
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("missing manifest: {}".format(path))
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("manifest is not an object: {}".format(path))
    return value


def copied_tree_hashes(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): digest(path) for path in sorted(root.rglob("*")) if path.is_file()}


def copy_file(source: Path, destination: Path) -> dict[str, str]:
    if not source.is_file():
        raise RuntimeError("installed dynamic OPP asset is missing: {}".format(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    actual = digest(destination)
    if actual != digest(source):
        raise RuntimeError("copied asset hash mismatch: {}".format(destination))
    return {"source": str(source), "destination": str(destination), "sha256": actual}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-manifest", required=True, type=Path)
    parser.add_argument("--installed-op-impl", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()

    build = load_json(args.build_manifest)
    if (build.get("schema") != "source_candidate_build_v1" or build.get("target") != "optiling" or
            build.get("strategy_algorithm_changes") is not False):
        raise RuntimeError("build manifest does not describe a compatible source host tiler")
    operator = str(build.get("operator"))
    layout = RUNTIME_LAYOUT.get(operator)
    if layout is None:
        raise RuntimeError("operator is not allowed in the dynamic OPP materializer: {}".format(operator))
    if not args.installed_op_impl.is_dir():
        raise RuntimeError("installed built-in OPP root is missing: {}".format(args.installed_op_impl))
    if not args.destination.is_dir() or any(args.destination.iterdir()):
        raise RuntimeError("destination must be an existing empty directory: {}".format(args.destination))
    host_arch = platform.machine()
    if host_arch not in ("aarch64", "x86_64"):
        raise RuntimeError("unsupported CANN host architecture: {}".format(host_arch))
    vendor = "source_" + str(build["cmake_op_name"]).lower().replace("-", "_")
    # ``libopapi`` resolves custom tilers from a complete OPP root selected
    # through ASCEND_OPP_PATH.  A vendor directory by itself (or the newer
    # ASCEND_CUSTOM_OPP_PATH convenience variable) is not a CANN-8.1 loader
    # contract.  Keep the installed built-in tree read-only and expose it via
    # a private symlink; only this checkout's vendor payload is written.
    root = args.destination
    installed_opp_root = args.installed_op_impl.parent
    builtin_link = root / "built-in"
    os.symlink(installed_opp_root, builtin_link, target_is_directory=True)
    priority_file = root / "vendors" / "config.ini"
    priority_file.parent.mkdir(parents=True, exist_ok=True)
    priority_file.write_text("load_priority={}\n".format(vendor), encoding="utf-8")
    vendor_root = args.destination / "vendors" / vendor
    destination = vendor_root / "op_impl" / "ai_core" / "tbe"

    host_master = Path(str(build.get("host_tiling_artifact", "")))
    host_compat = Path(str(build.get("host_tiling_compat_artifact", "")))
    if not host_master.is_file() or not host_compat.is_file():
        raise RuntimeError("host tiling build omitted an artifact")
    if digest(host_master) != build.get("host_tiling_artifact_sha256") or digest(host_compat) != build.get("host_tiling_compat_artifact_sha256"):
        raise RuntimeError("host tiling artifact hash mismatches its build manifest")
    copied: list[dict[str, str]] = []
    # CANN resolves the host tiler under the architecture of the process that
    # calls ACLNN.  The NPU host here is aarch64; a hard-coded x86_64 path
    # silently falls back to the installed tiler and cannot emit our audit.
    copied.append(copy_file(host_master, destination / "op_tiling" / "lib" / "linux" / host_arch / "libcust_opmaster_rt2.0.so"))
    copied.append(copy_file(host_compat, destination / "op_tiling/liboptiling.so"))

    installed_tbe = args.installed_op_impl / "ai_core" / "tbe"
    config = installed_tbe / "config/ascend910b/aic-ascend910b-ops-info.json"
    config_data = load_json(config)
    key = str(layout["config_key"])
    if key not in config_data:
        raise RuntimeError("installed op-info lacks runtime operator {}".format(key))
    selected = {key: config_data[key]}
    config_destination = destination / "config/ascend910b/aic-ascend910b-ops-info.json"
    config_destination.parent.mkdir(parents=True, exist_ok=True)
    config_destination.write_text(json.dumps(selected, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    copied.append({"source": str(config) + "#" + key, "destination": str(config_destination), "sha256": digest(config_destination)})

    dynamic = installed_tbe / "impl/dynamic" / str(layout["dynamic_file"])
    copied.append(copy_file(dynamic, destination / "impl/dynamic" / dynamic.name))
    for name in layout["ascendc_dirs"]:
        source_dir = installed_tbe / "impl/ascendc" / str(name)
        if not source_dir.is_dir():
            raise RuntimeError("installed AscendC source directory is missing: {}".format(source_dir))
        target_dir = destination / "impl/ascendc" / str(name)
        shutil.copytree(source_dir, target_dir)
        for relative, value in copied_tree_hashes(target_dir).items():
            copied.append({"source": str(source_dir / relative), "destination": str(target_dir / relative), "sha256": value})

    delivery = root / "installed_dynamic_device_assets.json"
    delivery.write_text(json.dumps({
        "schema": "installed_dynamic_device_assets_v1", "operator": operator,
        "installed_op_impl": str(args.installed_op_impl), "runtime_config_key": key,
        "dynamic_device_compilation": "installed CANN dynamic implementation; actual keys are selected at runtime",
        "assets": copied,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output = {
        "schema": "source_candidate_custom_opp_v1", "operator": operator,
        "cmake_op_name": build["cmake_op_name"], "source_family": build["source_family"],
        "strategy_class": build.get("strategy_class"), "strategy_priority": build.get("strategy_priority"),
        "official_commit": build["official_commit"], "vendor": vendor, "custom_opp_root": str(root),
        # This is a full private OPP view: built-in is a read-only symlink to
        # the installed package and vendors/config.ini selects this one host
        # tiler.  No installed CANN path is changed.
        "runtime_opp_root": str(root),
        "runtime_opp_layout": {
            "built_in_symlink": str(builtin_link),
            "built_in_target": str(installed_opp_root),
            "vendor_priority_file": str(priority_file),
            "vendor_priority": vendor,
        },
        "custom_opp_vendor_root": str(vendor_root),
        "host_tiling_arch": host_arch,
        "source_package": str(delivery), "source_package_sha256": digest(delivery),
        "source_tiling_observation_enabled": True, "source_compile_info_core_budget_enumeration": True,
        "source_hardware_envelope_heuristic_enumeration": build["source_hardware_envelope_heuristic_enumeration"],
        "hardware_envelope_heuristic": build["hardware_envelope_heuristic"],
        "compatibility_port": build.get("compatibility_port"), "instrumentation": build["instrumentation"],
        "build_harness": build.get("build_harness"), "overlay_manifest_sha256": build["overlay_manifest_sha256"],
        "strategy_algorithm_changes": False, "toolkit_install_modified": False, "matmul_included": False,
        "device_kernel_delivery": {
            "mode": "installed_dynamic_source_passthrough", "precompiled_all_tiling_keys": False,
            "formal_data_gate": "real source audit + operator execution + exact installed-reference output equality",
        },
    }
    manifest = root / "source_candidate_package.json"
    manifest.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print("dynamic OPP materialization error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
