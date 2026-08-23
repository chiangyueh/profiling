#!/usr/bin/env python3
"""Safely unpack a pinned source bundle shipped with this repository.

The campaign must build the modified host tilers on the target NPU host, but
must not depend on network access or shared source directories there.  This
helper unpacks one read-only source archive committed in ``vendor_source``
into the current checkout's private ``.source_cache``.  It never downloads,
compiles, invokes ACL, or touches a system CANN installation.
"""

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
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor_source"
BUNDLES = {
    "cann_ops": {
        "file": "cann_ops_8_1rc1.tar.gz",
        "sha256": "c4a2f6bf4ea6784dbfb2ef554858221a4e8f603bad9e4f5597a5083efb73c18e",
        "commit": "c214b710edbe24017dc7dc92170a50bd8ff38171",
        "format": "tar",
    },
    "cann_ops_adv": {
        "file": "cann_ops_adv_8_1rc1.tar.gz",
        "sha256": "753d7f581955c088f28f478f6aa7c26abf0c66187d5f82f653658c0ce45b45d0",
        "commit": "d9b54c8395cfa31ab5c35cfa4225e0fb35ee5553",
        "format": "tar",
    },
    "gather_elements_v2": {
        "file": "gather_elements_v2_source.zip",
        "sha256": "9f1d9753d47ec8e0f0fb3dde1de5b047e048900ff2e9a014c1ed614d7ed57470",
        "format": "gather_zip",
    },
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def member_path(name: str) -> Path:
    item = PurePosixPath(name)
    if item.is_absolute() or ".." in item.parts:
        raise RuntimeError("unsafe source-bundle member: {}".format(name))
    return Path(*item.parts)


def gather_prefix(names: list[str]) -> PurePosixPath:
    marker = "op_host/gather_elements_v2_tiling.cpp"
    matches = [PurePosixPath(name).parent.parent for name in names if name.endswith(marker)]
    if len(matches) != 1:
        raise RuntimeError("GatherElements source bundle lacks a unique source root")
    return matches[0]


def local_git_snapshot(root: Path, kind: str) -> None:
    """Provide worktree semantics without requiring a network clone.

    Overlay preparers create independent edited worktrees.  Initialising this
    private repository records only the already-unpacked, hash-attested bundle;
    it does not contact a remote and its commit id is deliberately *not* used
    as provenance (the attestation carries the pinned upstream commit).
    """
    commands = (
        ["git", "init", "--quiet", str(root)],
        ["git", "-C", str(root), "config", "user.name", "profiling-source-bundle"],
        ["git", "-C", str(root), "config", "user.email", "profiling-source-bundle@local.invalid"],
        ["git", "-C", str(root), "add", "-A"],
        ["git", "-C", str(root), "commit", "--quiet", "-m", "private source bundle: " + kind],
    )
    for command in commands:
        done = subprocess.run(command, text=True, capture_output=True, check=False)
        if done.returncode:
            raise RuntimeError("private source snapshot setup failed: {} {}".format(" ".join(command), done.stderr.strip()))


def extract(kind: str, bundle: Path, destination: Path) -> None:
    temporary = Path(tempfile.mkdtemp(prefix=".source_bundle.", dir=destination.parent))
    try:
        if BUNDLES[kind]["format"] == "tar":
            with tarfile.open(bundle, "r:gz") as archive:
                for info in archive.getmembers():
                    relative = member_path(info.name)
                    if not info.isfile():
                        if info.isdir():
                            (temporary / relative).mkdir(parents=True, exist_ok=True)
                            continue
                        raise RuntimeError("unsupported source-bundle entry: {}".format(info.name))
                    target = temporary / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(info) as source, target.open("wb") as output:
                        if source is None:
                            raise RuntimeError("cannot read source-bundle member: {}".format(info.name))
                        shutil.copyfileobj(source, output)
        else:
            with zipfile.ZipFile(bundle) as archive:
                prefix = gather_prefix(archive.namelist())
                for info in archive.infolist():
                    item = PurePosixPath(info.filename)
                    relative_all = member_path(info.filename)
                    try:
                        relative = relative_all.relative_to(Path(*prefix.parts))
                    except ValueError:
                        continue
                    if not relative.parts or info.is_dir():
                        continue
                    target = temporary / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info, "r") as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
        (temporary / ".source_bundle_attestation.json").write_text(json.dumps({
            "schema": "repo_source_bundle_v1", "bundle_kind": kind,
            "bundle_file": BUNDLES[kind]["file"], "bundle_sha256": digest(bundle),
            "official_commit": BUNDLES[kind].get("commit"), "network_calls": 0,
            "shared_paths_modified": False,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        local_git_snapshot(temporary, kind)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=tuple(BUNDLES))
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    if args.destination.exists():
        raise RuntimeError("refuse to overwrite existing private source cache: {}".format(args.destination))
    if not args.destination.parent.is_dir():
        raise RuntimeError("private source-cache parent is missing: {}".format(args.destination.parent))
    item = BUNDLES[args.kind]
    bundle = VENDOR / str(item["file"])
    if not bundle.is_file() or digest(bundle) != item["sha256"]:
        raise RuntimeError("repository source bundle is absent or hash-mismatched: {}".format(bundle))
    extract(args.kind, bundle, args.destination)
    print(json.dumps({"schema": "repo_source_bundle_v1", "status": "materialized", "kind": args.kind,
                      "destination": str(args.destination), "network_calls": 0, "npu_calls": 0,
                      "compilations": 0, "shared_paths_modified": False}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, tarfile.TarError, zipfile.BadZipFile) as error:
        print("repository source-bundle error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
