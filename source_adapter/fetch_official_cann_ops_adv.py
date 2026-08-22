#!/usr/bin/env python3
"""Explicitly fetch the pinned public CANN advanced-operator source.

This helper is intentionally separate from any measurement command: it never
downloads during a profiling run, and it never compiles or calls an NPU.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOCK = json.loads((ROOT / "non_matmul_source_lock.json").read_text(encoding="utf-8"))
SOURCE = LOCK["sources"]["cann_ops_adv"]


def run(argv: list[str]) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError("command failed rc={} argv={} stderr={}".format(
            result.returncode, " ".join(argv), result.stderr.strip()))
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path,
                        help="new directory outside the profiling Git repository")
    parser.add_argument("--depth", default=4096, type=int,
                        help="metadata history depth; blobs are filtered (default: 4096)")
    args = parser.parse_args()
    if args.depth < 1:
        raise RuntimeError("--depth must be positive")
    if args.destination.exists():
        raise RuntimeError("refuse to overwrite existing destination: {}".format(args.destination))
    if not args.destination.parent.is_dir():
        raise RuntimeError("destination parent does not exist: {}".format(args.destination.parent))

    run(["git", "clone", "--no-checkout", "--filter=blob:none", "--depth", str(args.depth),
         SOURCE["url"], str(args.destination)])
    run(["git", "-C", str(args.destination), "checkout", "--detach", SOURCE["commit"]])
    actual = run(["git", "-C", str(args.destination), "rev-parse", "HEAD"])
    if actual != SOURCE["commit"]:
        raise RuntimeError("checkout mismatch: expected={} actual={}".format(SOURCE["commit"], actual))
    required = args.destination / "src/transformer/flash_attention_score_grad/ophost/flash_attention_score_grad_tiling.cpp"
    if not required.is_file():
        raise RuntimeError("pinned source is missing FlashAttentionScoreGrad tiling source")
    print(json.dumps({
        "status": "fetched",
        "url": SOURCE["url"],
        "tag": SOURCE["tag"],
        "commit": actual,
        "destination": str(args.destination),
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
