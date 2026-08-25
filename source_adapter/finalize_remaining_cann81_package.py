#!/usr/bin/env python3
"""Attest one source-instrumented CANN-8.1 attention host tiler."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
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
        "dispatch_value": "cann81_native_aclnn",
        "tiling_env": "FASG_SOURCE_TILING_LIBRARY",
    },
    "fused_infer_attention_score": {
        "operator": "FusedInferAttentionScore",
        "source_family": "cann_ops_adv",
        "audit_schema": "fias_raw_tiling_observation_v1",
        "audit_env": "FIAS_TILING_AUDIT_PATH",
        "budget_env": "FIAS_SOURCE_AIV_CAP",
        "envelope_env": "FIAS_SOURCE_UB_DIVISOR",
        "dispatch_env": "FIAS_SOURCE_DISPATCH",
        "dispatch_value": "cann81_native_aclnn",
        "tiling_env": "FIAS_SOURCE_TILING_LIBRARY",
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


def installed_library(cann: Path, name: str) -> Path:
    candidates = [
        cann / "lib64" / name,
        cann / platform.machine() / "lib64" / name,
        cann / "aarch64-linux" / "lib64" / name,
        cann / "x86_64-linux" / "lib64" / name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError("installed CANN library is absent: {}".format(name))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", required=True, choices=tuple(SPECS))
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--build-root", required=True, type=Path)
    parser.add_argument("--cann-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    spec = SPECS[args.operator]
    locked = LOCK["operators"][args.operator]
    project = args.project.resolve()
    build = args.build_root.resolve()
    cann = args.cann_root.resolve()
    version = cann / "opp/version.info"

    require(project.is_dir() and build.is_dir(), "private project or host-tiler build is absent")
    require(version.is_file() and "version_dir=8.1.RC1" in version.read_text(encoding="utf-8"),
            "remaining operators require CANN 8.1.RC1")
    overlay_path = project / "source_candidate_overlay.json"
    require(overlay_path.is_file(), "source overlay manifest is absent")
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    require(overlay.get("operator") == spec["operator"] and
            overlay.get("strategy_algorithm_changes") is False and
            overlay.get("kernel_algorithm_changes") is False,
            "wrong or source-changing overlay")
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

    tiling = build / "libcust_opmaster_rt2.0.so"
    require(tiling.is_file(), "CANN-8.1 host-tiler library was not built")
    installed_opapi = installed_library(cann, "libopapi.so")
    installed_acl = installed_library(cann, "libascendcl.so")
    envelope = overlay.get("hardware_envelope_heuristic")
    require(isinstance(envelope, dict) and envelope.get("environment") == spec["envelope_env"] and
            tuple(envelope.get("divisors", ())) == (2, 4, 8), "hardware-envelope contract is invalid")

    manifest = {
        "schema": "remaining_operator_cann81_host_tiler_v1",
        "operator": spec["operator"],
        "runtime_op": args.operator,
        "strategy_class": overlay.get("strategy_class"),
        "source_kind": "official_{}_cann81_native_host_tiler".format(spec["source_family"]),
        "official_source_commit": expected_commit,
        "build_cann_version": "8.1.RC1",
        "cann_root": str(cann),
        "cann_version_file_sha256": digest(version),
        "project_root": str(project),
        "package_root": str(build),
        "source_file": str(source_file),
        "source_file_sha256": digest(source_file),
        "official_tiling_source_sha256": official_hash,
        "op_tiling_library": str(tiling),
        "op_tiling_library_sha256": digest(tiling),
        "installed_opapi_library": str(installed_opapi),
        "installed_opapi_library_sha256": digest(installed_opapi),
        "installed_ascendcl_library": str(installed_acl),
        "installed_ascendcl_library_sha256": digest(installed_acl),
        "device_kernel_origin": "installed_cann81_same_official_source_release",
        "instrumentation": {
            "enabled": True,
            "mutates_tiling_context": False,
            "audit_schema": spec["audit_schema"],
            "audit_environment": spec["audit_env"],
            "source_budget_environment": spec["budget_env"],
            "dispatch_environment": spec["dispatch_env"],
            "dispatch_value": spec["dispatch_value"],
            "tiling_library_environment": spec["tiling_env"],
        },
        "hardware_envelope_heuristic": envelope,
        "strategy_algorithm_changes": False,
        "kernel_algorithm_changes": False,
        "runtime_python_compilation": False,
        "toolkit_install_modified": False,
        "matmul_included": False,
        "scatter_elements_included": False,
        "formal_data_gate": (
            "direct-load the instrumented official CANN 8.1 host tiler, observe its raw tiling identity, "
            "launch the installed kernel from the same CANN 8.1 official source release, and exactly match "
            "the installed-reference output"
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
