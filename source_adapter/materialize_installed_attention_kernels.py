#!/usr/bin/env python3
"""Copy matching CANN-8.1 binary-package kernels into one private OPP."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


OPS = {
    "flash_attention_score_grad": "flash_attention_score_grad",
    "fused_infer_attention_score": "fused_infer_attention_score",
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", required=True, choices=tuple(OPS))
    parser.add_argument("--cann-root", required=True, type=Path)
    parser.add_argument("--package-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    cann = args.cann_root.resolve()
    package = args.package_root.resolve()
    op = OPS[args.operator]
    version = cann / "opp/version.info"
    require(version.is_file() and "version_dir=8.1.RC1" in version.read_text(encoding="utf-8"),
            "installed OPP is not CANN 8.1.RC1")
    source = cann / "opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b" / op
    require(source.is_dir(),
            "matching installed CANN-8.1 binary kernel directory is absent: {}".format(source))
    objects = sorted(path for path in source.rglob("*.o") if path.is_file())
    require(objects, "installed CANN-8.1 binary package has no {} kernels".format(op))
    pairs = []
    for obj in objects:
        metadata = obj.with_suffix(".json")
        require(metadata.is_file(), "installed kernel metadata is absent: {}".format(metadata))
        pairs.append((obj, metadata))

    destination = package / "op_impl/ai_core/tbe/kernel/ascend910b" / op
    require(package.is_dir(), "private package root is absent")
    if destination.exists():
        require(destination.is_dir() and not any(destination.iterdir()),
                "refuse to merge an existing private attention kernel directory: {}".format(destination))
    else:
        destination.mkdir(parents=True)
    copied = []
    try:
        for obj, metadata in pairs:
            relative = obj.relative_to(source)
            target_object = destination / relative
            target_metadata = target_object.with_suffix(".json")
            target_object.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(obj, target_object)
            shutil.copy2(metadata, target_metadata)
            require(digest(obj) == digest(target_object) and digest(metadata) == digest(target_metadata),
                    "copied attention kernel hash mismatch")
            copied.append({
                "installed_object": str(obj), "object": str(target_object),
                "object_sha256": digest(target_object),
                "installed_metadata": str(metadata), "metadata": str(target_metadata),
                "metadata_sha256": digest(target_metadata),
            })
        manifest = {
            "schema": "installed_cann81_attention_kernel_copy_v1",
            "operator": args.operator,
            "cann_root": str(cann),
            "opp_version_sha256": digest(version),
            "installed_kernel_root": str(source),
            "private_kernel_root": str(destination),
            "kernel_pairs": copied,
            "source_read_only": True,
            "toolkit_install_modified": False,
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except BaseException:
        if destination.exists():
            shutil.rmtree(destination)
        raise
    print(json.dumps({"status": "passed", "operator": args.operator,
                      "kernel_pairs": len(copied)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, json.JSONDecodeError) as error:
        print("attention kernel materialization error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
