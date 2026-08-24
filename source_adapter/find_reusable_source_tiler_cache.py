#!/usr/bin/env python3
"""Find an exact, private host-tiler/custom-OPP cache that can be reused.

This is intentionally read-only.  A previous campaign may already have built
the same original source host tiler, while a later controller-only fix has a
different state id.  Reusing it is safe only when the overlay digest, both
host-library digests, custom-OPP manifest, host architecture, and installed
dynamic-device source root all match the current execution environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def object_from(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def overlay_semantic_digest(path: Path) -> str | None:
    """Digest source semantics, excluding the private overlay location.

    Each campaign state has a different checkout-local overlay path.  That
    path is provenance for a particular build directory, but it is not part
    of the original source, audit, or resource-input semantics.  Normalizing
    it permits reuse of an attested existing host tiler without recompilation.
    """
    value = object_from(path)
    if value is None:
        return None
    value.pop("overlay", None)
    value.pop("resumed_existing_overlay", None)
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def matching_build(path: Path, desired_overlay_semantics: str) -> dict[str, Any] | None:
    build = object_from(path)
    if build is None:
        return None
    if (build.get("schema") != "source_candidate_build_v1" or
            build.get("target") != "optiling" or
            build.get("strategy_algorithm_changes") is not False):
        return None
    for artifact_key, digest_key in (
        ("host_tiling_artifact", "host_tiling_artifact_sha256"),
        ("host_tiling_compat_artifact", "host_tiling_compat_artifact_sha256"),
    ):
        artifact = Path(str(build.get(artifact_key, "")))
        if not artifact.is_file() or digest(artifact) != build.get(digest_key):
            return None
    try:
        state = path.parents[2]
    except IndexError:
        return None
    raw_digest = str(build.get("overlay_manifest_sha256", ""))
    for prior_overlay in state.rglob("source_candidate_overlay.json"):
        if digest(prior_overlay) == raw_digest and overlay_semantic_digest(prior_overlay) == desired_overlay_semantics:
            return build
    return None


def matching_package(path: Path, build: dict[str, Any], overlay_digest: str,
                     installed_op_impl: Path) -> bool:
    package = object_from(path)
    if package is None:
        return False
    if (package.get("schema") != "source_candidate_custom_opp_v1" or
            package.get("overlay_manifest_sha256") != overlay_digest or
            package.get("operator") != build.get("operator") or
            package.get("cmake_op_name") != build.get("cmake_op_name") or
            package.get("strategy_class") != build.get("strategy_class") or
            package.get("official_commit") != build.get("official_commit") or
            package.get("strategy_algorithm_changes") is not False or
            package.get("toolkit_install_modified") is not False or
            package.get("host_tiling_arch") != platform.machine()):
        return False
    vendor_root = Path(str(package.get("custom_opp_vendor_root", "")))
    runtime_opp_root = Path(str(package.get("runtime_opp_root", "")))
    source_package = Path(str(package.get("source_package", "")))
    layout = package.get("runtime_opp_layout")
    if (not vendor_root.is_dir() or not source_package.is_file() or not runtime_opp_root.is_dir() or
            not isinstance(layout, dict)):
        return False
    builtin = runtime_opp_root / "built-in"
    priority = runtime_opp_root / "vendors" / "config.ini"
    try:
        builtin_target = builtin.resolve(strict=True)
        expected_builtin = installed_op_impl.parent.resolve(strict=True)
    except OSError:
        return False
    if (builtin_target != expected_builtin or not priority.is_file() or
            priority.read_text(encoding="utf-8").strip() != "load_priority={}".format(package.get("vendor")) or
            Path(str(layout.get("built_in_symlink", ""))) != builtin or
            Path(str(layout.get("vendor_priority_file", ""))) != priority):
        return False
    if digest(source_package) != package.get("source_package_sha256"):
        return False
    delivery = object_from(source_package)
    if delivery is None or delivery.get("schema") != "installed_dynamic_device_assets_v1":
        return False
    assets = delivery.get("assets")
    if not isinstance(assets, list):
        return False
    for asset in assets:
        if not isinstance(asset, dict):
            return False
        destination = Path(str(asset.get("destination", "")))
        if not destination.is_file() or digest(destination) != asset.get("sha256"):
            return False
    tbe = vendor_root / "op_impl" / "ai_core" / "tbe" / "op_tiling"
    master = tbe / "lib" / "linux" / platform.machine() / "libcust_opmaster_rt2.0.so"
    compat = tbe / "liboptiling.so"
    if (not master.is_file() or not compat.is_file() or
            digest(master) != build.get("host_tiling_artifact_sha256") or
            digest(compat) != build.get("host_tiling_compat_artifact_sha256")):
        return False
    try:
        source_root = Path(str(delivery.get("installed_op_impl", ""))).resolve(strict=True)
        expected_root = installed_op_impl.resolve(strict=True)
    except OSError:
        return False
    if source_root != expected_root:
        return False
    mode = package.get("device_kernel_delivery")
    return isinstance(mode, dict) and mode.get("mode") == "installed_dynamic_source_passthrough"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-parent", required=True, type=Path)
    parser.add_argument("--overlay-manifest", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--installed-op-impl", required=True, type=Path)
    args = parser.parse_args()

    if not args.overlay_manifest.is_file():
        raise RuntimeError("overlay manifest is missing: {}".format(args.overlay_manifest))
    if not args.installed_op_impl.is_dir():
        raise RuntimeError("installed OPP root is missing: {}".format(args.installed_op_impl))
    desired_overlay_semantics = overlay_semantic_digest(args.overlay_manifest)
    if desired_overlay_semantics is None:
        raise RuntimeError("overlay manifest is not valid JSON: {}".format(args.overlay_manifest))
    # The prior clean route used ``_host``.  Keep the second name to make the
    # check forward-compatible, while its manifest validation remains exact.
    patterns = (
        "*/package_builds/{}_host/source_candidate_build.json".format(args.label),
        "*/package_builds/{}_host_tiler/source_candidate_build.json".format(args.label),
    )
    candidates = sorted({path for pattern in patterns for path in args.state_parent.glob(pattern)},
                        key=lambda item: str(item))
    reusable_build: Path | None = None
    for build_manifest in candidates:
        build = matching_build(build_manifest, desired_overlay_semantics)
        if build is None:
            continue
        reusable_build = build_manifest
        try:
            prior_state = build_manifest.parents[2]
        except IndexError:
            continue
        package_manifest = prior_state / "custom_opp" / args.label / "source_candidate_package.json"
        if matching_package(package_manifest, build, str(build.get("overlay_manifest_sha256")), args.installed_op_impl):
            print(json.dumps({"status": "reused", "build_manifest": str(build_manifest),
                              "custom_opp_manifest": str(package_manifest)}, sort_keys=True))
            return 0
    if reusable_build is not None:
        print(json.dumps({"status": "repackage", "build_manifest": str(reusable_build)}, sort_keys=True))
        return 0
    print(json.dumps({"status": "absent"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        raise SystemExit("source host-tiler cache check error: {}".format(error))
