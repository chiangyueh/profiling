#!/usr/bin/env python3
"""Build one isolated FlashAttentionScoreGrad original-strategy overlay.

The overlay must first be made by prepare_fasg_strategy_overlays.py.  The only
additional edit performed here is to the source release metadata in version.info
so the public 8.1.RC1 source tag's pre-release marker passes its own CMake
compatibility check against the installed 8.1.RC1 package.  No tiling or kernel
source is edited by this script.

`--target optiling` is a host-only proof that the selected original strategy
links. `--target package` is the explicit source-package build used before a
candidate can be dispatched through ASCEND_CUSTOM_OPP_PATH. This script does
not launch an NPU or install anything into the toolkit.
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
SOURCE = LOCK["sources"]["cann_ops_adv"]


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


def package_subversion(cann_root: Path) -> str:
    version_file = cann_root / "opp/version.info"
    if not version_file.is_file():
        raise RuntimeError("missing installed OPP version file: {}".format(version_file))
    match = re.search(r"^Version=([^\r\n]+)", version_file.read_text(encoding="utf-8"), re.MULTILINE)
    if not match:
        raise RuntimeError("cannot parse installed OPP version")
    package_version = match.group(1)
    prefix, separator, _ = package_version.rpartition(".")
    if not separator:
        raise RuntimeError("installed OPP version has no package-build suffix: {}".format(package_version))
    # The source checker strips one suffix from each side. Append a neutral
    # source-build suffix so its compared subversion exactly matches the package.
    return prefix + ".0"


def vendor_name(strategy: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", strategy.lower())
    return ("fasg_" + token)[:63].rstrip("_")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", required=True, type=Path)
    parser.add_argument("--build-dir", required=True, type=Path,
                        help="new build directory outside this repository")
    parser.add_argument("--cann-root", required=True, type=Path)
    parser.add_argument("--target", choices=("optiling", "package"), default="optiling")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--vendor", default=None,
                        help="isolated custom-OPP vendor name (default: derived from strategy class)")
    args = parser.parse_args()
    if args.jobs < 1:
        raise RuntimeError("--jobs must be positive")
    if args.build_dir.exists():
        raise RuntimeError("refuse to overwrite existing build directory: {}".format(args.build_dir))
    manifest_path = args.overlay / "source_candidate_overlay.json"
    if not manifest_path.is_file():
        raise RuntimeError("overlay lacks source_candidate_overlay.json")
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    instrumentation = manifest.get("instrumentation", {})
    audit_entry = args.overlay / "src/transformer/flash_attention_score_grad/ophost/flash_attention_score_grad_tiling.cpp"
    if (
        manifest.get("official_commit") != SOURCE["commit"]
        or manifest.get("algorithm_source_changes") is not False
        or instrumentation.get("enabled") is not True
        or instrumentation.get("mutates_tiling_context") is not False
        or "FASG_SOURCE_TILING_AUDIT_V1" not in audit_entry.read_text(encoding="utf-8")
    ):
        raise RuntimeError("overlay provenance/invariant check failed")
    if not args.cann_root.is_dir():
        raise RuntimeError("CANN root does not exist: {}".format(args.cann_root))
    selected_vendor = args.vendor or vendor_name(str(manifest["strategy_class"]))
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,62}", selected_vendor):
        raise RuntimeError("invalid vendor name: {}".format(selected_vendor))

    version_info = args.overlay / "version.info"
    original_version_digest = digest(version_info)
    original_version = version_info.read_text(encoding="utf-8")
    target_version = package_subversion(args.cann_root)
    updated_version, count = re.subn(r"^Version=[^\r\n]*", "Version=" + target_version,
                                     original_version, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError("cannot replace overlay version metadata")
    version_info.write_text(updated_version, encoding="utf-8")

    run([
        "cmake", "-S", str(args.overlay), "-B", str(args.build_dir), "-G", "Unix Makefiles",
        "-DBUILD_OPEN_PROJECT=ON",
        "-DASCEND_COMPUTE_UNIT=ascend910b",
        "-DASCEND_OP_NAME=flash_attention_score_grad",
        "-DVENDOR_NAME=" + selected_vendor,
        "-DCUSTOM_ASCEND_CANN_PACKAGE_PATH=" + str(args.cann_root),
    ])
    run(["cmake", "--build", str(args.build_dir), "--target", args.target,
         "--parallel", str(args.jobs)])
    artifact = args.build_dir / "libcust_opmaster_rt2.0.so"
    if args.target == "optiling" and not artifact.is_file():
        raise RuntimeError("host tiling artifact is missing after successful build")
    package_files = sorted(args.build_dir.glob("CANN-custom_ops-*.run"))
    if args.target == "package" and len(package_files) != 1:
        raise RuntimeError("expected exactly one source package, found {}".format(len(package_files)))
    output = {
        "schema": "fasg_original_strategy_build_v1",
        "operator": manifest["operator"],
        "strategy_class": manifest["strategy_class"],
        "strategy_priority": manifest["strategy_priority"],
        "matmul_included": False,
        "source_tiling_observation_enabled": True,
        "overlay_manifest_sha256": digest(manifest_path),
        "target": args.target,
        "vendor": selected_vendor,
        "build_dir": str(args.build_dir),
        "host_tiling_artifact": str(artifact) if artifact.is_file() else None,
        "host_tiling_artifact_sha256": digest(artifact) if artifact.is_file() else None,
        "source_package": str(package_files[0]) if package_files else None,
        "source_package_sha256": digest(package_files[0]) if package_files else None,
        "metadata_compatibility_edit": {
            "path": str(version_info),
            "sha256_before": original_version_digest,
            "sha256_after": digest(version_info),
            "source_version_before": re.search(r"^Version=([^\r\n]+)", original_version, re.MULTILINE).group(1),
            "source_version_after": target_version,
            "reason": "make the public 8.1.RC1 source tag pass its own exact package-version check; no tiling/kernel source changed",
        },
        "algorithm_source_changes": False,
        "npu_calls": 0,
        "toolkit_install_modified": False,
    }
    (args.build_dir / "source_candidate_build.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as error:
        print("source-candidate build error: {}".format(error), file=sys.stderr)
        sys.exit(2)
