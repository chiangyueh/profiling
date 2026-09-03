#!/usr/bin/env python3
"""Freeze the significant regressions from MatMul model validation v3.

The workload choice is prior measurement metadata only.  Neither these
labels nor the earlier latencies are visible to candidate generation or the
hardware simulator.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from generate_matmul_model_validation_workloads import build_catalog


REGRESSED_RANKS = (
    0, 1, 6, 22, 31, 36, 37, 42, 45, 67, 79, 86, 90, 104, 112,
    139, 147, 159, 167, 171, 176, 180, 181, 182, 183, 184, 186,
    188, 190, 191, 193, 194, 197, 198, 199,
)

# This split is fixed before the diagnostic measurements.  Model equations
# may be inspected against analysis rows; holdout rows are reserved for the
# subsequent unchanged-model check.
HOLDOUT_RANKS = frozenset((1, 22, 37, 67, 90, 139, 159, 181, 186, 191, 198))

EXPECTED_WORKLOADS = len(REGRESSED_RANKS)
ANALYSIS_WORKLOADS = EXPECTED_WORKLOADS - len(HOLDOUT_RANKS)
HOLDOUT_WORKLOADS = len(HOLDOUT_RANKS)


def build_diagnostic_workloads() -> list[dict[str, str]]:
    catalog = {row["workload_id"]: row for row in build_catalog()}
    rows: list[dict[str, str]] = []
    for rank in REGRESSED_RANKS:
        workload_id = f"matmul_rank_{rank:03d}"
        if workload_id not in catalog:
            raise RuntimeError(f"missing frozen workload {workload_id}")
        row = dict(catalog[workload_id])
        row["diagnostic_partition"] = (
            "holdout" if rank in HOLDOUT_RANKS else "analysis"
        )
        row["selection_basis"] = "v3_regressed_above_measured_noise"
        rows.append(row)
    if len(rows) != EXPECTED_WORKLOADS:
        raise RuntimeError(
            f"generated {len(rows)} workloads, expected {EXPECTED_WORKLOADS}"
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build_diagnostic_workloads()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        "MATMUL_REGRESSION_DIAGNOSTIC_WORKLOADS "
        f"shapes={len(rows)} analysis={ANALYSIS_WORKLOADS} "
        f"holdout={HOLDOUT_WORKLOADS}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
