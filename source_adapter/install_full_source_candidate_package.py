#!/usr/bin/env python3
"""Install one complete source-built custom operator package privately.

The package is built from the pinned source overlay and installed only below
the campaign state directory.  Unlike the earlier host-tiler-only overlay,
this records the package's own opapi, opsproto, and optiling libraries so the
runner can call the exact source package by absolute path.  Nothing is copied
from, installed into, or otherwise changes the system CANN installation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
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


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise RuntimeError("installed source package is missing {}: {}".format(description, path))
    return path.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-manifest", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()

    if not args.build_manifest.is_file():
        raise RuntimeError("missing build manifest: {}".format(args.build_manifest))
    build: dict[str, Any] = json.loads(args.build_manifest.read_text(encoding="utf-8"))
    package_text = build.get("source_package")
    if (build.get("schema") != "source_candidate_build_v1" or build.get("target") != "package" or
            build.get("source_tiling_observation_enabled") is not True or
            build.get("source_compile_info_core_budget_enumeration") is not True or not package_text):
        raise RuntimeError("build manifest does not describe a completed source package")
    package = Path(str(package_text))
    if not package.is_file() or digest(package) != build.get("source_package_sha256"):
        raise RuntimeError("source package is absent or hash-mismatched")
    if not args.destination.is_dir() or any(args.destination.iterdir()):
        raise RuntimeError("destination must be an existing empty directory: {}".format(args.destination))

    result = subprocess.run([str(package), "--quiet", "--install-path=" + str(args.destination)],
                            capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError("official package installer failed rc={} output={}".format(
            result.returncode, (result.stdout + result.stderr).strip()))

    host_arch = platform.machine()
    if host_arch not in ("aarch64", "x86_64"):
        raise RuntimeError("unsupported CANN host architecture: {}".format(host_arch))
    vendor_root = args.destination / "vendors" / str(build["vendor"])
    if not vendor_root.is_dir():
        raise RuntimeError("installed package is missing its isolated vendor directory: {}".format(vendor_root))
    libraries = {
        "opapi": require_file(vendor_root / "op_api/lib/libcust_opapi.so", "source opapi"),
        "opsproto": require_file(vendor_root / "op_proto/lib/linux" / host_arch / "libcust_opsproto_rt2.0.so", "source opsproto"),
        "optiling": require_file(vendor_root / "op_impl/ai_core/tbe/op_tiling/lib/linux" / host_arch / "libcust_opmaster_rt2.0.so", "source optiling"),
    }
    output = {
        "schema": "source_candidate_full_package_v2",
        "operator": build["operator"], "cmake_op_name": build["cmake_op_name"],
        "source_family": build["source_family"], "strategy_class": build.get("strategy_class"),
        "strategy_priority": build.get("strategy_priority"), "official_commit": build["official_commit"],
        "vendor": build["vendor"], "custom_opp_root": str(args.destination.resolve()),
        "custom_opp_vendor_root": str(vendor_root.resolve()), "host_tiling_arch": host_arch,
        "source_package": str(package.resolve()), "source_package_sha256": digest(package),
        "source_tiling_observation_enabled": True, "source_compile_info_core_budget_enumeration": True,
        "source_hardware_envelope_heuristic_enumeration": build["source_hardware_envelope_heuristic_enumeration"],
        "hardware_envelope_heuristic": build["hardware_envelope_heuristic"],
        "compatibility_port": build.get("compatibility_port"), "instrumentation": build["instrumentation"],
        "build_harness": build.get("build_harness"), "overlay_manifest_sha256": build["overlay_manifest_sha256"],
        "strategy_algorithm_changes": False, "toolkit_install_modified": False, "matmul_included": False,
        "execution_route": "explicit_private_source_package_api",
        "package_libraries": {name: {"path": str(path), "sha256": digest(path)} for name, path in libraries.items()},
    }
    manifest = args.destination / "source_candidate_package.json"
    manifest.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print("full source-candidate install error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
