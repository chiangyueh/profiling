#!/usr/bin/env python3
"""Read-only contract check for the GatherElements custom-source dispatch.

This is intentionally a static check.  It verifies the exact CANN API,
library symbol, OPP config, source registration, and runner linkage required
before a physical-NPU smoke is allowed to begin.  It neither calls ACL nor
creates, changes, or deletes any CANN/NPU resource.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def symbol_present(library: Path, symbol: str) -> bool:
    completed = subprocess.run(["nm", "-D", str(library)], text=True, capture_output=True, check=False)
    return completed.returncode == 0 and re.search(r"\b" + re.escape(symbol) + r"\b", completed.stdout) is not None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cann-root", required=True, type=Path)
    parser.add_argument("--runner-source", required=True, type=Path)
    parser.add_argument("--runner-cmake", required=True, type=Path)
    parser.add_argument("--overlay-manifest", type=Path)
    args = parser.parse_args()
    cann = args.cann_root.resolve()
    header = cann / "include" / "acl" / "acl_op_compiler.h"
    source = cann / "opp" / "built-in" / "op_impl" / "ai_core" / "tbe" / "impl" / "dynamic" / "gather_elements.py"
    config = cann / "opp" / "built-in" / "op_impl" / "ai_core" / "tbe" / "config" / "ascend910b" / "aic-ascend910b-ops-info.json"
    library_candidates = [
        cann / "lib64" / "libacl_op_compiler.so",
        cann / (platform.machine() + "-linux") / "lib64" / "libacl_op_compiler.so",
        cann / "aarch64-linux" / "lib64" / "libacl_op_compiler.so",
        cann / "x86_64-linux" / "lib64" / "libacl_op_compiler.so",
    ]
    library = next((path for path in library_candidates if path.is_file()), None)
    require(header.is_file(), "missing CANN generic compiler header: {}".format(header))
    require(source.is_file(), "missing installed GatherElements dynamic source: {}".format(source))
    require(config.is_file(), "missing Ascend910B GatherElements OPP config: {}".format(config))
    require(library is not None, "missing libacl_op_compiler.so under {}".format(cann))

    header_text = header.read_text(encoding="utf-8")
    source_text = source.read_text(encoding="utf-8")
    runner_text = args.runner_source.read_text(encoding="utf-8")
    cmake_text = args.runner_cmake.read_text(encoding="utf-8")
    op_config = json.loads(config.read_text(encoding="utf-8")).get("GatherElements")
    require("aclopCompileAndExecute" in header_text,
            "CANN header does not publish aclopCompileAndExecute")
    require(symbol_present(library, "aclopCompileAndExecute"),
            "CANN library does not export aclopCompileAndExecute: {}".format(library))
    require(isinstance(op_config, dict), "Ascend910B OPP config lacks GatherElements")
    require(op_config.get("opFile", {}).get("value") == "gather_elements",
            "GatherElements OPP config has unexpected opFile")
    require(op_config.get("opInterface", {}).get("value") == "gather_elements",
            "GatherElements OPP config has unexpected opInterface")
    require('@register_operator("GatherElements")' in source_text,
            "installed dynamic source does not register GatherElements")
    require('aclopCompileAndExecute("GatherElements"' in runner_text,
            "runner does not invoke GatherElements through generic OPP dispatch")
    require("acl_op_compiler" in cmake_text,
            "runner is not linked against libacl_op_compiler")
    overlay = None
    if args.overlay_manifest is not None:
        require(args.overlay_manifest.is_file(),
                "missing private GatherElements overlay manifest: {}".format(args.overlay_manifest))
        overlay = json.loads(args.overlay_manifest.read_text(encoding="utf-8"))
        require(overlay.get("schema") == "gather_elements_native_dynamic_overlay_v4",
                "private GatherElements overlay has an unexpected schema")
        custom_root = Path(str(overlay.get("custom_opp_root", "")))
        vendor = str(overlay.get("vendor", ""))
        vendor_root = Path(str(overlay.get("vendor_root", "")))
        impl = str(overlay.get("vendor_impl_directory", ""))
        source_file = Path(str(overlay.get("source_file", "")))
        custom_config = vendor_root / "op_impl" / "ai_core" / "tbe" / "config" / "ascend910b" / config.name
        require(vendor_root == custom_root / "vendors" / vendor,
                "private vendor root is not exactly under custom_opp/vendors")
        require(impl == vendor + "_impl",
                "private source implementation directory is not <vendor>_impl")
        require(source_file == vendor_root / "op_impl" / "ai_core" / "tbe" / impl / "dynamic" / "gather_elements.py",
                "private source file does not match the CANN custom-package layout")
        require(source_file.is_file(), "private instrumented GatherElements source is absent")
        require(custom_config.is_file(), "private GatherElements OPP config is absent")
        private_config = json.loads(custom_config.read_text(encoding="utf-8")).get("GatherElements")
        require(private_config == op_config,
                "private GatherElements config is not an exact installed-source declaration copy")
        private_source = source_file.read_text(encoding="utf-8")
        require("GATHER_ELEMENTS_NATIVE_DYNAMIC_SOURCE_AUDIT_V1" in private_source,
                "private GatherElements source lacks its dispatch audit marker")
        instrumentation = overlay.get("instrumentation")
        require(isinstance(instrumentation, dict) and
                instrumentation.get("dispatch_environment") == "GATHER_ELEMENTS_SOURCE_DISPATCH" and
                instrumentation.get("dispatch_value") == "aclop_compile_and_execute",
                "private source manifest lacks the explicit generic-dispatch contract")
    print("GATHER_ELEMENTS_DISPATCH_CONTRACT " + json.dumps({
        "status": "passed",
        "dispatch_api": "aclopCompileAndExecute",
        "reference_api": "aclnnGather",
        "compiler_header": str(header),
        "compiler_library": str(library),
        "source": str(source),
        "config": str(config),
        "op_type": "GatherElements",
        "op_file": op_config["opFile"]["value"],
        "op_interface": op_config["opInterface"]["value"],
        "overlay_manifest": None if args.overlay_manifest is None else str(args.overlay_manifest),
        "static_only": True,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print("GATHER_ELEMENTS_DISPATCH_CONTRACT " + json.dumps({"status": "failed", "error": str(error)}, sort_keys=True))
        raise SystemExit(2)
