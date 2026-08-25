#!/usr/bin/env python3
"""Materialize one repository-pinned official CANN-8.1 source tree privately."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent
LOCK = json.loads((ROOT / "remaining_operators_cann81_lock.json").read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def safe_relative(name: str) -> Path:
    item = PurePosixPath(name)
    if item.is_absolute() or ".." in item.parts:
        raise RuntimeError("unsafe archive member: {}".format(name))
    return Path(*item.parts)


def run(argv: list[str]) -> None:
    done = subprocess.run(argv, text=True, capture_output=True, check=False)
    if done.returncode:
        raise RuntimeError("local snapshot setup failed: {}: {}".format(" ".join(argv), done.stderr.strip()))


def initialize_snapshot(root: Path, kind: str) -> None:
    for argv in (
        ["git", "init", "--quiet", str(root)],
        ["git", "-C", str(root), "config", "user.name", "remaining-cann81-source-bundle"],
        ["git", "-C", str(root), "config", "user.email", "remaining-cann81-source@local.invalid"],
        ["git", "-C", str(root), "add", "-A"],
        ["git", "-C", str(root), "commit", "--quiet", "-m", "private official {} source snapshot".format(kind)],
    ):
        run(argv)


def validate_files(root: Path, kind: str) -> None:
    if kind == "cann_ops":
        source = LOCK["sources"][kind]
        cmake = root / "CMakeLists.txt"
        if not cmake.is_file() or digest(cmake) != source["root_cmake_sha256"]:
            raise RuntimeError("materialized CANN-8.1 build harness does not match its lock")
        return
    fias = LOCK["operators"]["fused_infer_attention_score"]
    fasg = LOCK["operators"]["flash_attention_score_grad"]
    checks = {
        Path(fias["relative_root"]) / fias["tiler"]: fias["tiler_sha256"],
        Path(fias["decode_tiler"]): fias["decode_tiler_sha256"],
        Path(fasg["relative_root"]) / "ophost/flash_attention_score_grad_tiling.cpp": fasg["entry_sha256"],
    }
    checks.update({Path(fasg["relative_root"]) / path: value for path, value in fasg["strategy_files"].items()})
    for relative, expected in checks.items():
        path = root / relative
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError("materialized advanced source mismatch: {}".format(relative))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("cann_ops", "cann_ops_adv"))
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists():
        raise RuntimeError("refuse to overwrite private source cache: {}".format(destination))
    if not destination.parent.is_dir():
        raise RuntimeError("source-cache parent is missing: {}".format(destination.parent))
    source = LOCK["sources"][args.kind]
    archive = ROOT / source["archive"]
    if not archive.is_file() or digest(archive) != source["archive_sha256"]:
        raise RuntimeError("pinned official source archive is absent or hash-mismatched")
    temporary = Path(tempfile.mkdtemp(prefix=".remaining_cann81.", dir=destination.parent))
    try:
        with tarfile.open(archive, "r:gz") as tar:
            for info in tar.getmembers():
                relative = safe_relative(info.name)
                target = temporary / relative
                if info.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not info.isfile():
                    raise RuntimeError("unsupported archive member: {}".format(info.name))
                target.parent.mkdir(parents=True, exist_ok=True)
                source_file = tar.extractfile(info)
                if source_file is None:
                    raise RuntimeError("cannot read archive member: {}".format(info.name))
                with source_file, target.open("wb") as output:
                    shutil.copyfileobj(source_file, output)
        validate_files(temporary, args.kind)
        attestation = {
            "schema": "remaining_operators_cann81_source_bundle_v1",
            "bundle_kind": args.kind,
            "archive": source["archive"],
            "archive_sha256": source["archive_sha256"],
            "official_commit": source["official_commit"],
            "network_calls": 0,
            "installed_cann_writes": 0,
        }
        (temporary / ".remaining_operators_cann81_attestation.json").write_text(
            json.dumps(attestation, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        # Compatibility metadata for the existing, source-only overlay
        # transformers.  It attests the same two local archives; no network or
        # legacy GatherElements source is introduced.
        (temporary / ".source_bundle_attestation.json").write_text(json.dumps({
            "schema": "repo_source_bundle_v1",
            "bundle_kind": args.kind,
            "bundle_file": Path(source["archive"]).name,
            "bundle_sha256": source["archive_sha256"],
            "official_commit": source["official_commit"],
            "network_calls": 0,
            "shared_paths_modified": False,
        }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        initialize_snapshot(temporary, args.kind)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print("REMAINING_CANN81_SOURCE_CACHE passed kind={}".format(args.kind))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, tarfile.TarError) as error:
        print("fatal: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
