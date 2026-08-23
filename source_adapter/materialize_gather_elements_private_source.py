#!/usr/bin/env python3
"""Materialize the attested GatherElements source into one private checkout.

Public CANN 8.1 does not carry the Ascend910B GatherElementsV2 host source
needed by this compatibility build. A previously archived, hash-pinned source
package may therefore be read as an input, but is never modified. This helper
extracts it only into the calling profiling checkout's private cache. It
neither compiles nor calls ACL/NPU APIs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent
LOCK = json.loads((ROOT / "non_matmul_source_lock.json").read_text(encoding="utf-8"))
PINNED = LOCK["operators"]["gather_elements_v2"]["pinned_files"]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def fail(message: str) -> None:
    raise RuntimeError(message)


def archive_prefix(names: list[str]) -> PurePosixPath:
    marker = "op_host/gather_elements_v2_tiling.cpp"
    matches = [PurePosixPath(name).parent.parent for name in names if name.endswith(marker)]
    if len(matches) != 1:
        fail("archive must contain exactly one GatherElementsV2 source root")
    return matches[0]


def safe_relative(member: str, prefix: PurePosixPath) -> Path | None:
    item = PurePosixPath(member)
    if item.is_absolute() or ".." in item.parts:
        fail("unsafe archive member: {}".format(member))
    try:
        relative = item.relative_to(prefix)
    except ValueError:
        return None
    return Path(*relative.parts) if relative.parts else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    if not args.archive.is_file():
        fail("GatherElements source archive is missing: {}".format(args.archive))
    if args.destination.exists():
        fail("refuse to overwrite GatherElements private source: {}".format(args.destination))
    if not args.destination.parent.is_dir():
        fail("destination parent is missing: {}".format(args.destination.parent))

    with zipfile.ZipFile(args.archive) as archive:
        prefix = archive_prefix(archive.namelist())
        temporary = Path(tempfile.mkdtemp(prefix=".gather_elements_extract.", dir=args.destination.parent))
        try:
            for info in archive.infolist():
                relative = safe_relative(info.filename, prefix)
                if relative is None or info.is_dir():
                    continue
                target = temporary / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info, "r") as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
            for relative, expected in PINNED.items():
                actual_path = temporary / relative
                if not actual_path.is_file() or digest(actual_path) != expected:
                    fail("archive does not match pinned GatherElements source: {}".format(relative))
            os.replace(temporary, args.destination)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    print(json.dumps({
        "schema": "gather_elements_private_source_v1",
        "status": "materialized",
        "archive": str(args.archive),
        "archive_sha256": digest(args.archive),
        "destination": str(args.destination),
        "pinned_file_count": len(PINNED),
        "shared_source_modified": False,
        "npu_calls": 0,
        "compilations": 0,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, zipfile.BadZipFile) as error:
        print("GatherElements private-source error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
