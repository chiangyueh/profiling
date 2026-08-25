#!/usr/bin/env python3
"""Reviewed ScatterElementsV2 shapes; no candidate tiling fields are generated."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any


FORMAL_LATENCY_TARGET = 5000
MIN_SUCCESSFUL_TILINGS_PER_SHAPE = 20
SOURCE_AIV_CAPS = tuple(range(1, 21))
SOURCE_HARDWARE_ENVELOPE_HEURISTICS = {
    "scatter_elements": {"resource": "source_visible_ub_capacity", "divisors": (2, 4, 8), "max_anchors": 16}
}


def row(index: int, tags: list[str], shape: list[int], index_shape: list[int],
        dtype: str, index_dtype: str, reduce: int) -> dict[str, Any]:
    return {
        "workload_id": "scatter_elements_{:03d}".format(index),
        "op": "scatter_elements", "coverage": tags,
        "shape": shape, "index_shape": index_shape, "axis": len(shape) - 1,
        "dtype": dtype, "index_dtype": index_dtype, "reduce": reduce,
    }


def catalog() -> list[dict[str, Any]]:
    prefixes = ((1024,), (1536,), (32, 32), (16, 64), (32, 64),
                (4, 16, 16), (8, 16, 16), (2, 4, 16, 16))
    axis_extents = (17, 31, 63, 65, 127, 129, 257, 513)
    index_extents = (1, 3, 15, 17, 31, 63, 65, 127)
    modes = (("fp16", "int32"), ("bf16", "int64"), ("fp32", "int32"), ("int32", "int64"))
    output: list[dict[str, Any]] = []
    for ordinal in range(len(prefixes) * len(axis_extents)):
        code = (ordinal * 37) % (len(prefixes) * len(axis_extents))
        prefix = prefixes[code % len(prefixes)]
        axis_extent = axis_extents[code // len(prefixes)]
        for mode_index, (dtype, index_dtype) in enumerate(modes):
            index_extent = min(axis_extent, index_extents[(ordinal * 5 + mode_index * 3) % len(index_extents)])
            reduction = (ordinal + mode_index) % 2
            rank = len(prefix) + 1
            output.append(row(len(output), ["reviewed_lattice", "last_axis", "rank{}".format(rank),
                            dtype, index_dtype, "reduce_assign" if reduction == 0 else "reduce_add"],
                            list(prefix) + [axis_extent], list(prefix) + [index_extent],
                            dtype, index_dtype, reduction))
    # Four explicit boundary rows leave ten spare groups beyond the 250 groups
    # needed for exactly 5,000 accepted measurements.
    for shape, extent, dtype, index_dtype, reduction in (
        ([32, 33, 65], 19, "fp32", "int32", 0),
        ([2, 64, 128, 256], 63, "int32", "int64", 1),
        ([16, 64, 64], 1, "fp16", "int32", 0),
        ([32, 33, 129], 31, "bf16", "int64", 1),
    ):
        output.append(row(len(output), ["explicit_boundary", "last_axis"], shape, shape[:-1] + [extent],
                          dtype, index_dtype, reduction))
    validate(output)
    return output


def validate(workloads: list[dict[str, Any]]) -> None:
    if len(workloads) != 260 or len({item["workload_id"] for item in workloads}) != len(workloads):
        raise ValueError("ScatterElementsV2 catalog must contain 260 unique shapes")
    for item in workloads:
        if item["op"] != "scatter_elements" or item["axis"] != len(item["shape"]) - 1:
            raise ValueError("catalog contains a non-Scatter or non-last-axis row")
        if len(item["shape"]) != len(item["index_shape"]):
            raise ValueError("shape rank mismatch")
        if any(item["index_shape"][i] != item["shape"][i] for i in range(item["axis"])):
            raise ValueError("illegal non-axis index shape")
        if item["index_shape"][-1] > item["shape"][-1]:
            raise ValueError("last-axis index extent is out of range")


def audit(workloads: list[dict[str, Any]]) -> dict[str, Any]:
    tags = Counter(tag for item in workloads for tag in item["coverage"])
    return {
        "schema": "scatter_elements_v2_candidate_catalog_v1",
        "operator": "scatter_elements", "matmul_included": False,
        "semantic_workloads": len(workloads), "workloads_per_op": {"scatter_elements": len(workloads)},
        "source_aiv_caps": list(SOURCE_AIV_CAPS),
        "source_discovery_upper_bound": len(workloads) * len(SOURCE_AIV_CAPS),
        "minimum_successful_tilings_per_shape": MIN_SUCCESSFUL_TILINGS_PER_SHAPE,
        "formal_latency_target": FORMAL_LATENCY_TARGET,
        "formal_record_budget_per_op": {"scatter_elements": FORMAL_LATENCY_TARGET},
        "source_hardware_envelope_heuristics": {
            "scatter_elements": {"resource": "source_visible_ub_capacity", "divisors": [2, 4, 8], "max_anchors": 16}
        },
        "coverage_tags": dict(sorted(tags.items())),
        "generation": "reviewed_finite_legal_shape_lattice_no_random_no_tiling_field_enumeration",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    rows = catalog()
    if args.audit:
        print(json.dumps(audit(rows), sort_keys=True))
    if args.json:
        for item in rows:
            print(json.dumps(item, sort_keys=True))
    if not args.audit and not args.json:
        parser.error("choose --audit and/or --json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
