#!/usr/bin/env python3
"""Deterministic legal shapes for the three non-Scatter CANN-8.1 campaigns."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import non_matmul_candidate_catalog as attention_catalog


SOURCE_AIV_CAPS = tuple(range(1, 21))
MIN_SUCCESSFUL_TILINGS_PER_SHAPE = 20
FORMAL_RECORD_TARGET = 5000


def catalog(operator: str) -> list[dict[str, Any]]:
    if operator == "gather_elements":
        result = attention_catalog.index_workloads("gather_elements")
    elif operator == "flash_attention_score_grad":
        result = [item for item in
                  (attention_catalog.attention_grad_workloads() +
                   attention_catalog.attention_grad_multi_tiling_reserve(
                       len(attention_catalog.attention_grad_workloads())))
                  if item["dtype"] in ("fp16", "bf16")]
    elif operator == "fused_infer_attention_score":
        result = (attention_catalog.fused_attention_workloads() +
                  attention_catalog.fused_attention_multi_tiling_reserve(
                      len(attention_catalog.fused_attention_workloads())))
    else:
        raise ValueError("unsupported operator: {}".format(operator))
    validate(operator, result)
    return result


def validate(operator: str, workloads: list[dict[str, Any]]) -> None:
    if len(workloads) < 250:
        raise ValueError("{} needs at least 250 reviewed shapes".format(operator))
    ids: set[str] = set()
    for item in workloads:
        if item.get("op") != operator or item.get("workload_id") in ids:
            raise ValueError("invalid or duplicate workload")
        ids.add(str(item["workload_id"]))
        if operator == "gather_elements":
            shape, index_shape = item["shape"], item["index_shape"]
            rank = len(shape)
            axis = item["axis"] + rank if item["axis"] < 0 else item["axis"]
            if (not shape or len(index_shape) != rank or axis < 0 or axis >= rank or
                    any(not isinstance(value, int) or value <= 0 for value in shape) or
                    any(not isinstance(value, int) or value <= 0 for value in index_shape) or
                    any(index_shape[i] > shape[i] for i in range(rank) if i != axis)):
                raise ValueError("illegal GatherElements geometry")
        else:
            if (item["dtype"] not in ("fp16", "bf16") or item["head_dim"] % 16 or
                    item["q_heads"] % item["kv_heads"]):
                raise ValueError("illegal attention geometry")


def audit(operator: str, workloads: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "remaining_operator_candidate_catalog_v1", "operator": operator,
        "reviewed_shapes": len(workloads), "source_aiv_caps": list(SOURCE_AIV_CAPS),
        "minimum_successful_tilings_per_shape": MIN_SUCCESSFUL_TILINGS_PER_SHAPE,
        "formal_record_target": FORMAL_RECORD_TARGET, "matmul_included": False,
        "scatter_elements_included": False,
        "random_generation": False, "historical_data_read": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", required=True,
                        choices=("gather_elements", "flash_attention_score_grad", "fused_infer_attention_score"))
    args = parser.parse_args()
    values = catalog(args.operator)
    print(json.dumps(audit(args.operator, values), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
