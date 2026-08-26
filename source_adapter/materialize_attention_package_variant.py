#!/usr/bin/env python3
"""Create one FASG package variant without recompiling identical kernels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def copy_or_link(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def unique_tiler(root: Path) -> Path:
    matches = sorted(root.glob(
        "op_impl/ai_core/tbe/op_tiling/lib/linux/*/libcust_opmaster_rt2.0.so"))
    if len(matches) != 1 or not matches[0].is_file():
        raise RuntimeError("base package must contain exactly one packaged host tiler")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-package-root", required=True, type=Path)
    parser.add_argument("--variant-build-root", required=True, type=Path)
    parser.add_argument("--variant-project", required=True, type=Path)
    parser.add_argument("--base-kernel-copy-manifest", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    base = args.base_package_root.resolve()
    build = args.variant_build_root.resolve()
    project = args.variant_project.resolve()
    destination = args.destination.resolve()
    overlay_path = project / "source_candidate_overlay.json"
    built_tiler = build / "libcust_opmaster_rt2.0.so"
    if not base.is_dir() or not overlay_path.is_file() or not built_tiler.is_file():
        raise RuntimeError("base package, variant overlay, or variant host tiler is absent")
    if not args.base_kernel_copy_manifest.is_file():
        raise RuntimeError("base installed-kernel copy manifest is absent")
    kernel_copy = json.loads(args.base_kernel_copy_manifest.read_text(encoding="utf-8"))
    if (kernel_copy.get("schema") != "installed_cann81_attention_kernel_copy_v1" or
            Path(kernel_copy.get("private_kernel_root", "")).resolve() !=
            (base / "op_impl/ai_core/tbe/kernel/ascend910b/flash_attention_score_grad").resolve()):
        raise RuntimeError("base installed-kernel copy manifest does not describe the FASG package")
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    if (overlay.get("operator") != "FlashAttentionScoreGrad" or
            overlay.get("strategy_algorithm_changes") is not False or
            overlay.get("kernel_algorithm_changes") is not False):
        raise RuntimeError("variant is not an attested source-preserving FASG strategy")
    if destination.exists():
        raise RuntimeError("refuse to merge an existing FASG package variant: {}".format(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(base, destination, symlinks=True, copy_function=copy_or_link)
        packaged_tiler = unique_tiler(destination)
        # copytree may have hard-linked this file to the base package. Unlink
        # before replacement so the shared, already-attested kernel package is
        # never modified.
        packaged_tiler.unlink()
        shutil.copy2(built_tiler, packaged_tiler)
        if digest(packaged_tiler) != digest(built_tiler):
            raise RuntimeError("variant host-tiler replacement hash mismatch")
        variant_kernel_root = (destination /
            "op_impl/ai_core/tbe/kernel/ascend910b/flash_attention_score_grad").resolve()
        variant_pairs = []
        for row in kernel_copy.get("kernel_pairs", []):
            base_object = Path(row["object"]).resolve()
            relative = base_object.relative_to(Path(kernel_copy["private_kernel_root"]).resolve())
            variant_object = variant_kernel_root / relative
            variant_metadata = variant_object.with_suffix(".json")
            installed_object = Path(row["installed_object"]).resolve()
            installed_metadata = Path(row["installed_metadata"]).resolve()
            if (not variant_object.is_file() or not variant_metadata.is_file() or
                    digest(variant_object) != digest(installed_object) or
                    digest(variant_metadata) != digest(installed_metadata)):
                raise RuntimeError("variant kernel differs from the installed CANN-8.1 binary")
            variant_pairs.append({**row, "object": str(variant_object),
                                  "metadata": str(variant_metadata)})
        if not variant_pairs:
            raise RuntimeError("base installed-kernel copy manifest has no kernel pairs")
        variant_kernel_copy = {**kernel_copy, "private_kernel_root": str(variant_kernel_root),
                               "kernel_pairs": variant_pairs}
        variant_kernel_manifest = destination / "installed_kernel_copy_manifest.json"
        if variant_kernel_manifest.exists():
            variant_kernel_manifest.unlink()
        variant_kernel_manifest.write_text(
            json.dumps(variant_kernel_copy, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        manifest = {
            "schema": "fasg_prebuilt_package_variant_v1",
            "operator": "FlashAttentionScoreGrad",
            "strategy_class": overlay.get("strategy_class"),
            "strategy_priority": overlay.get("strategy_priority"),
            "base_package_root": str(base),
            "variant_package_root": str(destination),
            "variant_host_tiler": str(packaged_tiler),
            "variant_host_tiler_sha256": digest(packaged_tiler),
            "installed_kernel_copy_manifest": str(variant_kernel_manifest),
            "shared_precompiled_kernel_files": len(list(destination.glob(
                "op_impl/ai_core/tbe/kernel/ascend910b/flash_attention_score_grad/**/*.o"))),
            "kernel_algorithm_changes": False,
            "toolkit_install_modified": False,
        }
        if manifest["shared_precompiled_kernel_files"] < 1:
            raise RuntimeError("base package has no FASG Ascend910B kernel objects")
        (destination / "fasg_prebuilt_package_variant.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except BaseException:
        if destination.exists():
            shutil.rmtree(destination)
        raise
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, json.JSONDecodeError) as error:
        print("FASG package-variant error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
