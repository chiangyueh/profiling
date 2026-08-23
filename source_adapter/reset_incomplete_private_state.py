#!/usr/bin/env python3
"""Remove one incomplete, private campaign cache directory.

This is deliberately narrower than a cache clean.  It will only remove a
non-symlink direct child of the supplied private parent and only while the
corresponding completion manifest is absent.  It cannot target CANN, the
source bundles, results, or any directory outside the active campaign state.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def existing_parent(path: Path) -> Path:
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError("private parent must be a real directory: {}".format(path))
    return path.resolve(strict=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--required-absent", required=True, type=Path)
    parser.add_argument("--kind", required=True, choices=("host_tiler_build", "dynamic_opp_root"))
    args = parser.parse_args()

    parent = existing_parent(args.parent)
    target = args.target
    required_absent = args.required_absent
    if required_absent.exists() or required_absent.is_symlink():
        raise RuntimeError("completion manifest exists; refusing to remove: {}".format(required_absent))
    if not target.exists() and not target.is_symlink():
        print(json.dumps({"event": "SOURCE_PRIVATE_STATE_ABSENT", "kind": args.kind,
                          "target": str(target)}, sort_keys=True))
        return 0
    if target.is_symlink():
        raise RuntimeError("refusing to remove a symlink target: {}".format(target))
    if target.parent.resolve(strict=True) != parent:
        raise RuntimeError("target is not a direct child of the private parent: {}".format(target))
    if not target.is_dir():
        raise RuntimeError("target is not a directory: {}".format(target))

    shutil.rmtree(target)
    print(json.dumps({"event": "SOURCE_PRIVATE_STATE_CLEARED", "kind": args.kind,
                      "reason": "completion_manifest_absent", "target": str(target)},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print("private campaign state reset error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
