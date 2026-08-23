#!/usr/bin/env python3
"""Build one attested original-source candidate overlay.

This is deliberately generic: the manifest produced by an operator-specific
preparer declares the real CMake operator name, audit entry, source family and
the finite source-input axes.  The builder never reconstructs a tiling,
changes an opaque raw buffer, or infers a candidate from a result.  It only
packages the already-audited source overlay into an isolated custom OPP
package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
LOCK = json.loads((ROOT / "non_matmul_source_lock.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def run(argv: list[str]) -> None:
    result = subprocess.run(argv, check=False)
    if result.returncode:
        raise RuntimeError("command failed rc={} argv={}".format(result.returncode, " ".join(argv)))


def default_vendor(manifest: dict[str, Any]) -> str:
    strategy = str(manifest.get("strategy_class", "source_dispatch"))
    operator = str(manifest["cmake_op_name"])
    token = re.sub(r"[^a-z0-9]+", "_", operator.lower() + "_" + strategy.lower()).strip("_")
    return ("src_" + token)[:63].rstrip("_")


def validate_manifest(overlay: Path) -> dict[str, Any]:
    path = overlay / "source_candidate_overlay.json"
    if not path.is_file():
        raise RuntimeError("overlay lacks source_candidate_overlay.json")
    manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    required = ("operator", "cmake_op_name", "source_family", "official_commit",
                "audit_entry_relative", "audit_sentinel", "instrumentation")
    missing = [field for field in required if not manifest.get(field)]
    if missing:
        raise RuntimeError("overlay provenance lacks: {}".format(", ".join(missing)))
    if manifest.get("strategy_algorithm_changes") is not False:
        raise RuntimeError("overlay must attest no source strategy algorithm changes")
    if manifest.get("source_compile_info_core_budget_enumeration") is not True:
        raise RuntimeError("overlay must use a finite original-source compile-info axis")
    if manifest.get("source_hardware_envelope_heuristic_enumeration") not in (True, False):
        raise RuntimeError("overlay hardware-envelope provenance is ambiguous")
    envelope = manifest.get("hardware_envelope_heuristic")
    if not isinstance(envelope, dict) or envelope.get("enabled") != bool(manifest["source_hardware_envelope_heuristic_enumeration"]):
        raise RuntimeError("overlay hardware-envelope provenance is ambiguous")
    if envelope.get("enabled") and (not envelope.get("environment") or not envelope.get("audit_field") or
                                    tuple(envelope.get("divisors", ())) != (2, 4, 8) or int(envelope.get("max_anchors", 0)) < 1):
        raise RuntimeError("overlay hardware-envelope rule is incomplete")
    instrumentation = manifest["instrumentation"]
    if instrumentation.get("enabled") is not True or instrumentation.get("mutates_tiling_context") is not False:
        raise RuntimeError("source tiling audit must be enabled and observational")
    scope = manifest.get("build_scope")
    if (not isinstance(scope, dict) or scope.get("sentinel") != "SOURCE_CANDIDATE_BUILD_SCOPE_V1" or
            scope.get("operator") != manifest["cmake_op_name"] or scope.get("structural_only") is not True):
        raise RuntimeError("overlay lacks an attested one-operator build scope")
    entry = overlay / str(manifest["audit_entry_relative"])
    if not entry.is_file() or str(manifest["audit_sentinel"]) not in entry.read_text(encoding="utf-8"):
        raise RuntimeError("audit entry does not match its source provenance")
    if manifest["source_family"] == "cann_ops_adv":
        source = LOCK["sources"]["cann_ops_adv"]
        harness = manifest.get("build_harness", {})
        if manifest["official_commit"] != source["commit"] or harness.get("official_commit") != LOCK["sources"]["cann_ops"]["commit"]:
            raise RuntimeError("advanced overlay source/build-harness provenance mismatch")
        if not (overlay / "CMakeLists.txt").is_file():
            raise RuntimeError("advanced overlay lacks its attested build harness")
    elif manifest["source_family"] == "cann_ops":
        source = LOCK["sources"]["cann_ops"]
        if manifest["official_commit"] != source["commit"]:
            raise RuntimeError("cann-ops overlay source provenance mismatch")
    else:
        raise RuntimeError("unsupported source family: {}".format(manifest["source_family"]))
    compatibility = manifest.get("compatibility_port")
    if compatibility is not None:
        if not isinstance(compatibility, dict):
            raise RuntimeError("compatibility-port provenance must be an object")
        if (compatibility.get("tiling_algorithm_changes") is not False or
                compatibility.get("kernel_algorithm_changes") is not False or
                not compatibility.get("source_version") or not compatibility.get("formal_data_gate") or
                not isinstance(compatibility.get("extracted_files_sha256"), dict)):
            raise RuntimeError("compatibility port does not prove source-preserving limits")
        allowed = set(compatibility.get("allowed_changes", ()))
        if not allowed or any("tiling algorithm" in item.lower() or "kernel algorithm" in item.lower() for item in allowed):
            raise RuntimeError("compatibility port authorizes an algorithm change")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", required=True, type=Path)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--cann-root", required=True, type=Path)
    parser.add_argument("--target", choices=("optiling", "package"), default="optiling")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--vendor", default=None)
    args = parser.parse_args()
    if args.jobs < 1:
        raise RuntimeError("--jobs must be positive")
    if args.build_dir.exists():
        raise RuntimeError("refuse to overwrite existing build directory: {}".format(args.build_dir))
    if not args.cann_root.is_dir():
        raise RuntimeError("CANN root does not exist: {}".format(args.cann_root))
    manifest = validate_manifest(args.overlay)
    vendor = args.vendor or default_vendor(manifest)
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,62}", vendor):
        raise RuntimeError("invalid vendor name: {}".format(vendor))
    cmake_args = [
        "cmake", "-S", str(args.overlay), "-B", str(args.build_dir), "-G", "Unix Makefiles",
        "-DBUILD_OPEN_PROJECT=ON", "-DASCEND_COMPUTE_UNIT=ascend910b",
        "-DASCEND_OP_NAME=" + str(manifest["cmake_op_name"]), "-DVENDOR_NAME=" + vendor,
        "-DCUSTOM_ASCEND_CANN_PACKAGE_PATH=" + str(args.cann_root),
    ]
    # Neither pinned public 8.1 checkout carries the release-only
    # ``version.info`` file that this CMake check requires. Do not invent or
    # edit that metadata: the manifest instead attests the pinned commits,
    # source hashes, build harness and installed CANN path, then exact NPU
    # output comparison remains the admission gate.
    cmake_args.append("-DCHECK_COMPATIBLE=OFF")
    run(cmake_args)
    run(["cmake", "--build", str(args.build_dir), "--target", args.target, "--parallel", str(args.jobs)])
    artifact = args.build_dir / "libcust_opmaster_rt2.0.so"
    packages = sorted(args.build_dir.glob("CANN-custom_ops-*.run"))
    # The unmodified public 8.1 root's ``optiling`` post-build action creates
    # this compatibility loader entry.  Do not invoke its separate
    # ``optiling_compat`` target: in a detached build that target constructs a
    # relative symlink against an installation-only directory and is broken.
    # The post-build entry below is the source tree's actual host-tiler
    # delivery path and resolves to the same just-built library.
    compatibility_artifact = args.build_dir / "custom/op_impl/ai_core/tbe/op_tiling/liboptiling.so"
    if args.target == "optiling" and not artifact.is_file():
        raise RuntimeError("host tiling artifact is missing after build")
    if args.target == "optiling" and not compatibility_artifact.is_file():
        raise RuntimeError("official host-tiler compatibility entry is missing after build")
    if args.target == "package" and len(packages) != 1:
        raise RuntimeError("expected one source package, found {}".format(len(packages)))
    output = {
        "schema": "source_candidate_build_v1", "operator": manifest["operator"],
        "cmake_op_name": manifest["cmake_op_name"], "source_family": manifest["source_family"],
        "strategy_class": manifest.get("strategy_class"), "strategy_priority": manifest.get("strategy_priority"),
        "official_commit": manifest["official_commit"], "matmul_included": False,
        "source_tiling_observation_enabled": True, "overlay_manifest_sha256": digest(args.overlay / "source_candidate_overlay.json"),
        "target": args.target, "vendor": vendor, "build_dir": str(args.build_dir),
        "host_tiling_artifact": str(artifact) if artifact.is_file() else None,
        "host_tiling_artifact_sha256": digest(artifact) if artifact.is_file() else None,
        "host_tiling_compat_artifact": str(compatibility_artifact) if compatibility_artifact.is_file() else None,
        "host_tiling_compat_artifact_sha256": digest(compatibility_artifact) if compatibility_artifact.is_file() else None,
        "source_package": str(packages[0]) if packages else None,
        "source_package_sha256": digest(packages[0]) if packages else None,
        "instrumentation": manifest["instrumentation"], "build_harness": manifest.get("build_harness"),
        "strategy_algorithm_changes": False,
        "source_compile_info_core_budget_enumeration": True,
        "source_hardware_envelope_heuristic_enumeration": manifest["source_hardware_envelope_heuristic_enumeration"],
        "hardware_envelope_heuristic": manifest["hardware_envelope_heuristic"],
        "compatibility_port": manifest.get("compatibility_port"),
        "toolkit_install_modified": False, "npu_calls": 0,
    }
    (args.build_dir / "source_candidate_build.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print("source-candidate build error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
