#!/usr/bin/env python3
"""Write the three preregistered MatMul frontier workloads."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SHAPES = (
    # Victor reports M x K x N; the runner and model use M, N, K.
    ("victor_m2048_k7168_n1536", 2048, 1536, 7168),
    ("victor_m2048_k2048_n7168", 2048, 7168, 2048),
    ("victor_m4096_k7168_n512", 4096, 512, 7168),
)
FORMAL_PER_SHAPE = 720


def build_workloads() -> list[dict[str, str]]:
    rows = []
    for workload_id, m, n, k in SHAPES:
        rows.append({
            "workload_id": workload_id,
            "m": str(m), "n": str(n), "k": str(k),
            "dtype": "fp16", "trans_a": "0", "trans_b": "0",
            "max_cores": "20",
            "search_family": "source_route_frontier",
            "coverage_intent": "all_applicable_execution_graphs",
            "calibration_partition": "frontier_discovery",
            "required_successful_tilings": str(FORMAL_PER_SHAPE),
            "coverage": (
                f"preregistered_victor_shape;fp16;nn;m{m};n{n};k{k};"
                "original_source_routes;all_core_caps;hardware_local_frontier"
            ),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build_workloads()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("MATMUL_VICTOR_FRONTIER_CATALOG shapes=3 candidates=2160 baselines=3 records=2163")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
