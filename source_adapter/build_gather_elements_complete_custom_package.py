#!/usr/bin/env python3
"""Build and attest one private, complete CANN-8.1 GatherElementsV2 package.

This is deliberately not an ``optiling``-only build.  The package contains
the source OpDef, source host tiler, source device kernel, and the CANN-8.1
API generated from that OpDef.  A caller can therefore prove that a source
tiling observation belongs to the kernel that was actually executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def run(command: list[str], *, cwd: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, text=True, check=False)
    if completed.returncode:
        raise RuntimeError("command failed rc={}: {}".format(completed.returncode, " ".join(command)))


def host_arch() -> str:
    value = os.uname().machine
    if value not in ("x86_64", "aarch64"):
        raise RuntimeError("unsupported CANN host architecture: {}".format(value))
    return value


def expected(overlay: Path, build: Path, cann: Path, vendor: str) -> dict[str, Any]:
    overlay_manifest = overlay / "source_candidate_overlay.json"
    if not overlay_manifest.is_file():
        raise RuntimeError("GatherElements complete custom overlay is missing its manifest")
    source = json.loads(overlay_manifest.read_text(encoding="utf-8"))
    if source.get("schema") != "gather_elements_complete_custom_package_overlay_v1":
        raise RuntimeError("unexpected GatherElements complete custom overlay schema")
    architecture = host_arch()
    vendor_root = build / "package" / "packages" / "vendors" / vendor
    opapi = vendor_root / "op_api" / "lib" / "libcust_opapi.so"
    tiler = vendor_root / "op_impl" / "ai_core" / "tbe" / "op_tiling" / "lib" / "linux" / architecture / "libcust_opmaster_rt2.0.so"
    proto = vendor_root / "op_proto" / "lib" / "linux" / architecture / "libcust_opsproto_rt2.0.so"
    return {
        "schema": "gather_elements_complete_custom_package_v1",
        "operator": "GatherElementsV2",
        "runtime_op": "gather_elements",
        "source_overlay": str(overlay),
        "source_overlay_sha256": digest(overlay_manifest),
        "source_version": source["source_version"],
        "source_revision": source["source_revision"],
        "cann_root": str(cann.resolve()),
        "cann_version_file_sha256": digest(cann / "opp" / "version.info"),
        "vendor": vendor,
        # CANN 8.1's generated ``set_env.bash`` exports this exact vendor
        # directory, not the parent ``packages`` directory.  The runtime
        # loader searches this colon-separated vendor-root path for the
        # custom OpProto/tiler/kernel which belongs to the generated API.
        "custom_opp_root": str(vendor_root),
        "custom_opp_vendor_root": str(vendor_root),
        "custom_opapi": str(opapi),
        "custom_tiling_library": str(tiler),
        "custom_proto_library": str(proto),
        "custom_api": {
            "get_workspace_symbol": "aclnnGatherElementsV2GetWorkspaceSize",
            "launch_symbol": "aclnnGatherElementsV2",
            "input_order": ["x", "index", "dim", "out"],
        },
        "instrumentation": source["instrumentation"],
        "hardware_envelope_heuristic": source["hardware_envelope_heuristic"],
        "strategy_algorithm_changes": False,
        "kernel_algorithm_changes": False,
        "generated_api_required": True,
        "formal_data_gate": "the generated GatherElementsV2 custom API must emit the source tiler audit, launch the source kernel, and exactly match the installed aclnnGather reference",
    }


def valid_existing(manifest_path: Path, expected_manifest: dict[str, Any]) -> dict[str, Any] | None:
    if not manifest_path.is_file():
        return None
    item = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("schema", "operator", "runtime_op", "source_overlay_sha256", "cann_root", "cann_version_file_sha256",
                "vendor", "custom_opp_root", "custom_opp_vendor_root", "custom_opapi", "custom_tiling_library", "custom_proto_library",
                "custom_api", "strategy_algorithm_changes", "kernel_algorithm_changes", "generated_api_required"):
        if item.get(key) != expected_manifest.get(key):
            raise RuntimeError("existing complete GatherElements package provenance differs: {}".format(key))
    for key in ("custom_opapi", "custom_tiling_library", "custom_proto_library"):
        path = Path(str(item[key]))
        if not path.is_file():
            raise RuntimeError("existing complete GatherElements package is incomplete: {}".format(path))
        item[key + "_sha256"] = digest(path)
    return item


def build(overlay: Path, build_dir: Path, cann: Path, vendor: str, jobs: int) -> dict[str, Any]:
    if not cann.is_dir() or not (cann / "opp" / "version.info").is_file():
        raise RuntimeError("CANN root or OPP version information is absent: {}".format(cann))
    build_dir.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = build_dir / "complete_custom_package.json"
    planned = expected(overlay, build_dir, cann, vendor)
    prior = valid_existing(manifest_path, planned)
    if prior is not None:
        prior["resumed_existing_complete_package"] = True
        return prior
    if build_dir.exists():
        raise RuntimeError("incomplete private GatherElements package build exists and is intentionally not overwritten: {}".format(build_dir))
    build_dir.mkdir()
    install = build_dir / "package"
    configure = [
        "cmake", "-S", str(overlay), "-B", str(build_dir / "cmake"), "-G", "Unix Makefiles",
        "-DCMAKE_BUILD_TYPE=Release", "-DASCEND_CANN_PACKAGE_PATH=" + str(cann),
        "-DASCEND_COMPUTE_UNIT=ascend910b", "-Dvendor_name=" + vendor,
        "-DENABLE_TEST=False", "-DENABLE_SOURCE_PACKAGE=False", "-DENABLE_BINARY_PACKAGE=True",
        "-DASCEND_PACK_SHARED_LIBRARY=False", "-DCMAKE_INSTALL_PREFIX=" + str(install),
    ]
    run(configure, cwd=build_dir)
    # ``binary`` produces the source device kernel.  The two libraries below
    # are separately named because the template's binary target intentionally
    # does not depend on generated host API/proto libraries.
    run(["cmake", "--build", str(build_dir / "cmake"), "--target", "binary", "npu_supported_ops", "cust_op_proto", "cust_opapi", "optiling_compat", "modify_vendor", "gen_version_info",
         "--parallel", str(jobs)], cwd=build_dir)
    run(["cmake", "--install", str(build_dir / "cmake")], cwd=build_dir)
    for key in ("custom_opapi", "custom_tiling_library", "custom_proto_library"):
        path = Path(str(planned[key]))
        if not path.is_file():
            raise RuntimeError("complete GatherElements package did not install {}".format(path))
    symbols = subprocess.run(["nm", "-D", "--defined-only", str(planned["custom_opapi"])], text=True,
                             capture_output=True, check=False)
    if symbols.returncode:
        raise RuntimeError("cannot inspect generated GatherElements custom API symbols")
    for symbol in planned["custom_api"].values():
        if isinstance(symbol, str) and symbol not in symbols.stdout:
            raise RuntimeError("generated GatherElements custom API does not export {}".format(symbol))
    for key in ("custom_opapi", "custom_tiling_library", "custom_proto_library"):
        planned[key + "_sha256"] = digest(Path(str(planned[key])))
    planned["kernel_payload_root"] = str(Path(str(planned["custom_opp_vendor_root"])) / "op_impl" / "ai_core" / "tbe" / "kernel")
    if not Path(planned["kernel_payload_root"]).is_dir():
        raise RuntimeError("complete GatherElements package lacks installed source kernel payload")
    manifest_path.write_text(json.dumps(planned, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return planned


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", required=True, type=Path)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--cann-root", required=True, type=Path)
    parser.add_argument("--vendor", required=True)
    parser.add_argument("--jobs", required=True, type=int)
    args = parser.parse_args()
    if args.jobs < 1 or not args.vendor or any(value in args.vendor for value in "/\\"):
        raise RuntimeError("invalid private package build arguments")
    print(json.dumps(build(args.overlay, args.build_dir, args.cann_root, args.vendor, args.jobs), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print("GatherElements complete custom-package build error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
