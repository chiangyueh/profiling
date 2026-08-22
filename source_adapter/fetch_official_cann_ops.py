#!/usr/bin/env python3
"""Fetch the pinned public CANN source as a small, sparse, external worktree.

This is deliberately explicit: the main campaign never downloads source.  The
destination must be new, so this helper cannot overwrite a local source tree.
It performs no compilation and makes no ACL/NPU call.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOCK = json.loads((ROOT / "source_lock.json").read_text(encoding="utf-8"))
SPARSE_PATHS = (
    "CMakeLists.txt",
    "build.sh",
    "cmake",
    "src/common",
    "src/matmul/mat_mul_v3",
)


def run(argv: list[str]) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(
            "command failed rc={} argv={} stderr={}".format(
                result.returncode, " ".join(argv), result.stderr.strip()
            )
        )
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path,
                        help="new directory outside the profiling Git repository")
    parser.add_argument("--depth", type=int, default=4096,
                        help="metadata depth; blobs remain filtered (default: 4096)")
    args = parser.parse_args()
    if args.depth < 1:
        raise RuntimeError("--depth must be positive")
    if args.destination.exists():
        raise RuntimeError("refuse to overwrite existing destination: {}".format(args.destination))
    if not args.destination.parent.is_dir():
        raise RuntimeError("destination parent does not exist: {}".format(args.destination.parent))

    run([
        "git", "clone", "--no-checkout", "--filter=blob:none", "--depth", str(args.depth),
        LOCK["official_source"]["url"], str(args.destination),
    ])
    run(["git", "-C", str(args.destination), "sparse-checkout", "init", "--no-cone"])
    run(["git", "-C", str(args.destination), "sparse-checkout", "set", *SPARSE_PATHS])
    run(["git", "-C", str(args.destination), "checkout", "--detach", LOCK["official_source"]["commit"]])

    actual = run(["git", "-C", str(args.destination), "rev-parse", "HEAD"])
    if actual != LOCK["official_source"]["commit"]:
        raise RuntimeError("checkout mismatch: expected={} actual={}".format(
            LOCK["official_source"]["commit"], actual))
    required = args.destination / "src/matmul/mat_mul_v3/op_host/mat_mul_v3_base_tiling.cpp"
    if not required.is_file():
        raise RuntimeError("sparse checkout is missing the pinned MatMulV3 source")
    print(json.dumps({
        "status": "fetched",
        "url": LOCK["official_source"]["url"],
        "commit": actual,
        "destination": str(args.destination),
        "sparse_paths": SPARSE_PATHS,
        "downloads_are_runtime_campaign_input": False,
        "npu_calls": 0,
        "compilations": 0,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as error:
        print("official-source error: {}".format(error), file=sys.stderr)
        sys.exit(2)
