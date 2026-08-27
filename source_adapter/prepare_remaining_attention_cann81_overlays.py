#!/usr/bin/env python3
"""Prepare FASG or FIAS overlays from only the dedicated CANN-8.1 lock."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["CANN81_SOURCE_LOCK"] = str(ROOT / "remaining_operators_cann81_lock.json")
import prepare_fasg_strategy_overlays as fasg
import prepare_fias_source_overlay as fias


LOCK = json.loads((ROOT / "remaining_operators_cann81_lock.json").read_text(encoding="utf-8"))


def source_compatibility(name: str) -> dict[str, object]:
    value = dict(LOCK["sources"][name])
    value["commit"] = value["official_commit"]
    if name == "cann_ops":
        value["build_harness_cmake_sha256"] = value["root_cmake_sha256"]
    return value


def fasg_operator() -> dict[str, object]:
    item = dict(LOCK["operators"]["flash_attention_score_grad"])
    item["pinned_files"] = {
        "ophost/flash_attention_score_grad_tiling.cpp": item["entry_sha256"],
        **item["strategy_files"],
    }
    return item


def fias_operator() -> dict[str, object]:
    item = dict(LOCK["operators"]["fused_infer_attention_score"])
    item["pinned_files"] = {
        item["tiler"]: item["tiler_sha256"],
        "../incre_flash_attention/ophost/incre_flash_attention_tiling.cc": item["decode_tiler_sha256"],
    }
    return item


def configure_modules() -> None:
    advanced = source_compatibility("cann_ops_adv")
    base = source_compatibility("cann_ops")
    fasg.LOCK = LOCK
    fasg.OP = fasg_operator()
    fasg.SOURCE = advanced
    fasg.HARNESS_SOURCE = base
    fasg.MACRO = str(fasg.OP["registration_macro"])
    fasg.OPERATOR = str(fasg.OP["registration_operator"])
    fasg.ENTRY_RELATIVE = Path(str(fasg.OP["relative_root"])) / "ophost/flash_attention_score_grad_tiling.cpp"
    fias.LOCK = LOCK
    fias.SOURCE = advanced
    fias.OP = fias_operator()
    fias.OP_ROOT = Path(str(fias.OP["relative_root"]))
    fias.FIAS_RELATIVE = fias.OP_ROOT / "ophost/fused_infer_attention_score_tiling.cpp"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", required=True,
                        choices=("flash_attention_score_grad", "fused_infer_attention_score"))
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--harness-root", required=True, type=Path)
    parser.add_argument("--output-parent", required=True, type=Path)
    args = parser.parse_args()
    configure_modules()
    source_root = args.source_root.resolve()
    harness_root = args.harness_root.resolve()
    output_parent = args.output_parent.resolve()
    if not output_parent.is_dir():
        raise RuntimeError("output parent is absent")
    if args.operator == "flash_attention_score_grad":
        rows = fasg.require_pinned_source(source_root)
        harness_text, provenance = fasg.source_build_harness(harness_root, "flash_attention_score_grad")
        result = fasg.write_dispatcher_overlay(
            source_root, output_parent, rows, harness_root, harness_text, provenance)
    else:
        fias.require_pinned_source(source_root)
        harness_text, provenance = fasg.source_build_harness(harness_root, "fused_infer_attention_score")
        result = fias.write_overlay(source_root, output_parent, harness_root, harness_text, provenance)
    print(json.dumps({"schema": "remaining_attention_cann81_overlay_v1", "operator": args.operator,
                      "result": result, "legacy_source_lock_read_for_execution": False,
                      "matmul_included": False}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, json.JSONDecodeError) as error:
        print("remaining attention overlay error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
