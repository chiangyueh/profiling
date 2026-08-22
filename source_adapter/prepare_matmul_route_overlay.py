#!/usr/bin/env python3
"""Create one isolated MatMulV3 route adapter from a pinned official worktree.

The original worktree is never modified.  The only source change in the new
worktree replaces the tiling-template registration with a subclass that invokes
one existing TilingCalcSelect value.  MatmulV3BaseTiling and its search code are
checked byte-for-byte and are not edited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOCK = json.loads((ROOT / "source_lock.json").read_text(encoding="utf-8"))
MATMUL = LOCK["operators"]["mat_mul_v3"]
ROUTES = tuple(route for route in MATMUL["original_routes"] if route != "ALL")
REGISTRATION = 'REGISTER_TILING_TEMPLATE("MatMulV3", MatmulV3BaseTiling, 0);'


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def run(argv: list[str]) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(
            "command failed rc={} argv={} stderr={}".format(
                result.returncode, " ".join(argv), result.stderr.strip()
            )
        )
    return result.stdout.strip()


def require_clean_pinned_worktree(source: Path) -> None:
    if not source.is_dir():
        raise RuntimeError("official source root does not exist: {}".format(source))
    head = run(["git", "-C", str(source), "rev-parse", "HEAD"])
    if head != LOCK["official_source"]["commit"]:
        raise RuntimeError(
            "official source commit mismatch: expected={} actual={}".format(
                LOCK["official_source"]["commit"], head
            )
        )
    status = run(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"])
    if status:
        raise RuntimeError("official source worktree is modified; refuse to build an overlay from it")


def adapter_source(route: str) -> str:
    return """namespace {{
class MatmulV3RouteTiling final : public MatmulV3BaseTiling {{
public:
    explicit MatmulV3RouteTiling(gert::TilingContext* context)
        : MatmulV3BaseTiling(context, &routeTilingData_, TilingCalcSelect::{route}) {{}}

private:
    MatmulTilingData routeTilingData_;
}};
}} // namespace

REGISTER_TILING_TEMPLATE(\"MatMulV3\", MatmulV3RouteTiling, 0);""".format(route=route)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="new, non-existent git worktree directory")
    parser.add_argument("--route", required=True, choices=ROUTES)
    args = parser.parse_args()

    if args.output.exists():
        raise RuntimeError("refuse to overwrite existing output: {}".format(args.output))
    if args.output.parent.exists() is False:
        raise RuntimeError("output parent does not exist: {}".format(args.output.parent))
    require_clean_pinned_worktree(args.official_root)

    original_base = args.official_root / MATMUL["relative_root"] / "op_host/mat_mul_v3_base_tiling.cpp"
    expected_base_hash = MATMUL["pinned_files"]["op_host/mat_mul_v3_base_tiling.cpp"]
    if digest(original_base) != expected_base_hash:
        raise RuntimeError("original MatmulV3BaseTiling source does not match the pinned official file")

    run([
        "git", "-C", str(args.official_root), "worktree", "add", "--detach", str(args.output),
        LOCK["official_source"]["commit"],
    ])
    target = args.output / MATMUL["relative_root"] / "op_host/mat_mul_v3_tiling.cpp"
    before = target.read_text(encoding="utf-8")
    if before.count(REGISTRATION) != 1:
        raise RuntimeError("expected exactly one original MatMulV3 template registration in overlay")
    target.write_text(before.replace(REGISTRATION, adapter_source(args.route)), encoding="utf-8")

    overlay_base = args.output / MATMUL["relative_root"] / "op_host/mat_mul_v3_base_tiling.cpp"
    if digest(overlay_base) != expected_base_hash:
        raise RuntimeError("safety failure: the overlay changed original MatmulV3 tiling search source")

    metadata = {
        "schema": "matmul_v3_original_route_overlay_v1",
        "official_url": LOCK["official_source"]["url"],
        "official_commit": LOCK["official_source"]["commit"],
        "route": args.route,
        "search_source": str(overlay_base),
        "search_source_sha256": digest(overlay_base),
        "search_source_modified": False,
        "registration_source": str(target),
        "registration_source_sha256_before": hashlib.sha256(before.encode("utf-8")).hexdigest(),
        "registration_source_sha256_after": digest(target),
        "candidate_rule": "one existing original TilingCalcSelect route; raw tiling is not edited",
        "not_used": ["RuntimeKb", "callback selection", "CCE data", "cost-model tile synthesis"],
    }
    (args.output / "route_adapter_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as error:
        print("source-adapter error: {}".format(error), file=sys.stderr)
        sys.exit(2)
