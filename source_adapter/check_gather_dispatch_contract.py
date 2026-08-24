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
import os
import platform
import re
import subprocess
import sys
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def symbol_present(library: Path, symbol: str) -> bool:
    completed = subprocess.run(["nm", "-D", str(library)], text=True, capture_output=True, check=False)
    return completed.returncode == 0 and re.search(r"\b" + re.escape(symbol) + r"\b", completed.stdout) is not None


def check_private_python_resolution(runtime_root: Path, private_tbe_root: Path, source_file: Path) -> dict[str, str]:
    """Exercise CANN's failing import branch without ACL or an NPU call.

    ``tbe.common.utils.op_tiling`` reads ``ASCEND_OPP_PATH/scene.info`` while
    importing.  Running it in a fresh child catches a partial OPP view before
    generic compilation can reach a physical device.  The child environment
    is discarded when this function returns.
    """
    environment = dict(os.environ)
    old_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(private_tbe_root) + (os.pathsep + old_pythonpath if old_pythonpath else "")
    environment["ASCEND_OPP_PATH"] = str(runtime_root)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    code = r'''
import json
import os
import tbe.common.utils.op_tiling as tiling
import impl.dynamic.gather_elements as gather
print(json.dumps({
    "scene_info_path": tiling.scene_info_path,
    "sys_version": tiling.sys_version,
    "source_file": gather.__file__,
    "ascend_opp_path": os.environ.get("ASCEND_OPP_PATH"),
}, sort_keys=True))
'''
    completed = subprocess.run([sys.executable, "-c", code], env=environment, text=True,
                               capture_output=True, check=False)
    require(completed.returncode == 0,
            "private OPP Python import gate failed: {}".format((completed.stderr or completed.stdout)[-1200:]))
    value = None
    for line in reversed(completed.stdout.splitlines()):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            value = parsed
            break
    if value is None:
        raise RuntimeError("private OPP Python import gate emitted no JSON")
    require(value.get("ascend_opp_path") == str(runtime_root),
            "private OPP Python import gate did not use its runtime root")
    require(value.get("scene_info_path") == str(runtime_root / "scene.info"),
            "private OPP Python import gate did not read private-root scene.info")
    require(bool(value.get("sys_version")),
            "private OPP Python import gate did not initialize sys_version")
    require(Path(str(value.get("source_file", ""))).resolve() == source_file.resolve(),
            "private OPP Python import gate selected installed GatherElements instead of the private source")
    return {key: str(value[key]) for key in ("scene_info_path", "sys_version", "source_file")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cann-root", required=True, type=Path)
    parser.add_argument("--runner-source", required=True, type=Path)
    parser.add_argument("--runner-cmake", required=True, type=Path)
    parser.add_argument("--campaign-source", required=True, type=Path)
    parser.add_argument("--launch-script", required=True, type=Path)
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
    campaign_text = args.campaign_source.read_text(encoding="utf-8")
    launch_text = args.launch_script.read_text(encoding="utf-8")
    source_operator_type = "GatherElements"
    source_module = "gather_elements"
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
    require("kGatherElementsSourceOperatorType" in runner_text and source_operator_type in runner_text,
            "runner does not invoke the registered GatherElements operator type")
    require("acl_op_compiler" in cmake_text,
            "runner is not linked against libacl_op_compiler")
    require('environment["PYTHONPATH"]' in campaign_text and "private_tbe_import_root" in campaign_text,
            "source worker must prepend only the private canonical TBE import root")
    overlay = None
    if args.overlay_manifest is not None:
        require(args.overlay_manifest.is_file(),
                "missing private GatherElements overlay manifest: {}".format(args.overlay_manifest))
        overlay = json.loads(args.overlay_manifest.read_text(encoding="utf-8"))
        require(overlay.get("schema") == "gather_elements_native_dynamic_overlay_v7",
                "private GatherElements overlay has an unexpected schema")
        require(overlay.get("source_operator_type") == source_operator_type and
                overlay.get("source_module") == source_module,
                "private overlay lacks the registered GatherElements identity")
        runtime_root = Path(str(overlay.get("runtime_opp_root", "")))
        source_file = Path(str(overlay.get("source_file", "")))
        builtin = runtime_root / "built-in"
        layout = overlay.get("runtime_opp_layout")
        require(isinstance(layout, dict) and
                layout.get("mode") == "complete_private_opp_view_with_canonical_dynamic_module_override",
                "private overlay does not use the complete canonical-module OPP view")
        private_tbe_root = Path(str(layout.get("private_tbe_import_root", "")))
        require(source_file == private_tbe_root / "impl" / "dynamic" / (source_module + ".py"),
                "private source file is not the canonical CANN dynamic module path")
        require(source_file.is_file(), "private instrumented GatherElements source is absent")
        require(runtime_root.is_dir() and builtin.is_dir() and not builtin.is_symlink(),
                "private OPP root does not contain a canonical built-in override tree")
        linked_entries = layout.get("linked_root_entries")
        require(isinstance(linked_entries, list) and
                sorted(linked_entries) == sorted(path.name for path in (cann / "opp").iterdir() if path.name != "built-in"),
                "private OPP root does not mirror every required installed root entry")
        for name in linked_entries:
            private_entry, installed_entry = runtime_root / str(name), cann / "opp" / str(name)
            require(private_entry.is_symlink() and private_entry.resolve() == installed_entry.resolve(),
                    "private OPP root link is invalid: {}".format(name))
        private_source = source_file.read_text(encoding="utf-8")
        require("GATHER_ELEMENTS_NATIVE_DYNAMIC_SOURCE_AUDIT_V1" in private_source and
                '"event": "module_imported"' in private_source and
                "_ge_emit_import_audit()" in private_source and
                '"compile_info_vars"' in private_source and
                '"source_compile_context_sha256"' in private_source and
                '@register_operator("{}")'.format(source_operator_type) in private_source,
                "private GatherElements source lacks its required source-context audit markers")
        instrumentation = overlay.get("instrumentation")
        require(isinstance(instrumentation, dict) and
                instrumentation.get("dispatch_environment") == "GATHER_ELEMENTS_SOURCE_DISPATCH" and
                instrumentation.get("dispatch_value") == "aclop_compile_and_execute" and
                instrumentation.get("source_budget_environment") == "GATHER_ELEMENTS_SOURCE_AIV_CAP",
                "private source manifest lacks the explicit generic-dispatch contract")
        envelope = overlay.get("hardware_envelope_heuristic")
        require(isinstance(envelope, dict) and
                envelope.get("environment") == "GATHER_ELEMENTS_SOURCE_UB_DIVISOR" and
                envelope.get("audit_field") == "ub_cap_divisor",
                "private source manifest has an invalid UB-envelope environment")
        resolution = check_private_python_resolution(runtime_root, private_tbe_root, source_file)
    print("GATHER_ELEMENTS_DISPATCH_CONTRACT " + json.dumps({
        "status": "passed",
        "dispatch_api": "aclopCompileAndExecute",
        "reference_api": "aclnnGather",
        "compiler_header": str(header),
        "compiler_library": str(library),
        "source": str(source),
        "config": str(config),
        "op_type": "GatherElements",
        "source_operator_type": source_operator_type,
        "op_file": op_config["opFile"]["value"],
        "op_interface": op_config["opInterface"]["value"],
        "source_selector": "private_complete_opp_canonical_dynamic_module",
        "overlay_manifest": None if args.overlay_manifest is None else str(args.overlay_manifest),
        "private_python_import_gate": None if overlay is None else resolution,
        "static_only": True,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print("GATHER_ELEMENTS_DISPATCH_CONTRACT " + json.dumps({"status": "failed", "error": str(error)}, sort_keys=True))
        raise SystemExit(2)
