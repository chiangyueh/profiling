#!/usr/bin/env python3
"""Materialize the repository-pinned official CANN-8.1 source privately."""

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
LOCK = json.loads((ROOT / "scatter_elements_v2_cann81_lock.json").read_text(encoding="utf-8"))
SOURCE = LOCK["source"]


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


def initialize_snapshot(root: Path) -> None:
    for argv in (
        ["git", "init", "--quiet", str(root)],
        ["git", "-C", str(root), "config", "user.name", "scatter-source-bundle"],
        ["git", "-C", str(root), "config", "user.email", "scatter-source-bundle@local.invalid"],
        ["git", "-C", str(root), "add", "-A"],
        ["git", "-C", str(root), "commit", "--quiet", "-m", "private official source snapshot"],
    ):
        run(argv)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists():
        raise RuntimeError("refuse to overwrite private source cache: {}".format(destination))
    if not destination.parent.is_dir():
        raise RuntimeError("source-cache parent is missing: {}".format(destination.parent))
    archive = ROOT / str(SOURCE["archive"])
    if not archive.is_file() or digest(archive) != SOURCE["archive_sha256"]:
        raise RuntimeError("pinned official source archive is absent or hash-mismatched")
    temporary = Path(tempfile.mkdtemp(prefix=".scatter_cann81.", dir=destination.parent))
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
                source = tar.extractfile(info)
                if source is None:
                    raise RuntimeError("cannot read archive member: {}".format(info.name))
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
        cmake = temporary / "CMakeLists.txt"
        tiler = temporary / LOCK["operator"]["relative_root"] / LOCK["operator"]["tiler"]
        if (not cmake.is_file() or digest(cmake) != SOURCE["root_cmake_sha256"] or
                not tiler.is_file() or digest(tiler) != LOCK["operator"]["tiler_sha256"]):
            raise RuntimeError("materialized source does not match the pinned official files")
        (temporary / ".scatter_elements_v2_cann81_attestation.json").write_text(json.dumps({
            "schema": "scatter_elements_v2_cann81_source_bundle_v1",
            "archive": SOURCE["archive"],
            "archive_sha256": SOURCE["archive_sha256"],
            "official_commit": SOURCE["official_commit"],
            "network_calls": 0,
            "installed_cann_writes": 0
        }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        initialize_snapshot(temporary)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print("SCATTER_CANN81_SOURCE_CACHE passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, tarfile.TarError) as error:
        print("fatal: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
