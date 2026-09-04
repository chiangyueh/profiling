#!/usr/bin/env python3
"""Frozen MatMul hardware-factor calibration and holdout workloads."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Spec:
    m: int
    n: int
    k: int
    dtype: str = "fp16"
    trans_a: int = 0
    trans_b: int = 0


# Candidate counts are intentionally different by execution graph.  They are
# the union of bounded one-factor sweeps, not a Cartesian product.
CALIBRATION = {
    "base": (42, (
        Spec(2048, 1536, 7168), Spec(2048, 7168, 2048),
        Spec(4096, 512, 7168), Spec(192, 512, 16384),
        Spec(448, 128, 16384), Spec(192, 320, 24576),
        Spec(512, 320, 40960), Spec(384, 448, 49152),
        Spec(1024, 1024, 4096), Spec(4096, 4096, 4096, "bf16"),
        Spec(2048, 4096, 1024, "bf16", 0, 1),
        Spec(176, 2048, 16384, "fp16", 1, 0),
        Spec(320, 512, 2048, "fp32", 1, 1),
        Spec(448, 3072, 1024, "fp32", 0, 1),
        Spec(4096, 160, 1024, "fp32", 1, 0),
        Spec(257, 1009, 4097),
    )),
    "single_core_split_k": (42, (
        Spec(192, 512, 16384), Spec(448, 128, 16384),
        Spec(192, 320, 24576), Spec(512, 320, 40960),
        Spec(384, 448, 49152), Spec(256, 640, 32768),
        Spec(320, 512, 2048, "fp32", 1, 1),
        Spec(4096, 160, 1024, "fp32", 1, 0),
        Spec(448, 3072, 1024, "fp32", 0, 1),
        Spec(160, 5120, 2048, "bf16", 1, 1),
    )),
    "deterministic_split_k": (31, (
        Spec(128, 128, 8192), Spec(128, 128, 16384),
        Spec(128, 128, 32768), Spec(128, 128, 49152, "bf16"),
        Spec(192, 320, 24576), Spec(448, 128, 16384),
        Spec(384, 448, 49152), Spec(512, 320, 40960, "bf16"),
    )),
    # AL1 must fit the complete A resident set in C220 L1.  Keeping K=4096
    # and varying M/N crosses core-wave and tail boundaries without violating
    # that source constraint.
    "al1_full_load": (27, (
        Spec(16, 320, 4096, "fp32", 0, 1),
        Spec(12, 304, 4096, "fp32", 0, 1),
        Spec(8, 288, 4096, "fp32", 0, 1),
    )),
    "bl1_full_load": (38, (
        Spec(65536, 128, 128), Spec(65536, 192, 64),
        Spec(65536, 96, 128, "bf16"), Spec(65536, 64, 32, "fp32"),
    )),
    "bl1_full_load_fixpipe": (24, (
        Spec(8192, 7, 16), Spec(8192, 17, 128),
        Spec(8192, 31, 128, "bf16"), Spec(4096, 7, 8, "fp32"),
    )),
    "bl1_full_load_vec_nz2nd": (24, (
        Spec(4096, 7, 8, "fp32"), Spec(4096, 17, 16, "fp32"),
        Spec(8192, 31, 32, "fp32"), Spec(16384, 63, 64, "fp32"),
    )),
}


# These shapes are measured in the same run but carry a frozen holdout label.
# Cost-model changes must be derived from CALIBRATION only and then checked on
# these rows without changing their candidate set.
HOLDOUT = {
    "base": (
        Spec(1536, 2560, 6144), Spec(768, 1792, 3072, "bf16", 1, 0),
        Spec(320, 1536, 7168, "fp32", 0, 1),
    ),
    "single_core_split_k": (
        Spec(224, 384, 32768), Spec(640, 192, 24576, "bf16"),
        Spec(384, 768, 16384, "fp32", 1, 0),
    ),
    "deterministic_split_k": (
        Spec(160, 160, 24576), Spec(256, 192, 40960, "bf16"),
        Spec(320, 256, 32768),
    ),
    "al1_full_load": (
        Spec(4, 272, 4096, "fp32", 0, 1),
        Spec(16, 256, 4096, "fp32", 0, 1),
        Spec(8, 240, 4096, "fp32", 0, 1),
    ),
    "bl1_full_load": (
        Spec(32768, 128, 128), Spec(49152, 96, 64, "bf16"),
        Spec(32768, 64, 32, "fp32"),
    ),
    "bl1_full_load_fixpipe": (
        Spec(10240, 9, 16), Spec(12288, 17, 32, "bf16"),
        Spec(16384, 31, 64),
    ),
    "bl1_full_load_vec_nz2nd": (
        Spec(10240, 9, 8, "fp32"), Spec(12288, 17, 16, "fp32"),
        Spec(16384, 31, 24, "fp32"),
    ),
}


def build_workloads() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for partition, groups in (("calibration", CALIBRATION), ("holdout", HOLDOUT)):
        for family, value in groups.items():
            if partition == "calibration":
                required, specs = value
            else:
                required, specs = 20, value
            short = {
                "base": "base", "single_core_split_k": "sc",
                "deterministic_split_k": "det", "al1_full_load": "al1",
                "bl1_full_load": "bl1",
                "bl1_full_load_fixpipe": "fix",
                "bl1_full_load_vec_nz2nd": "vec",
            }[family]
            for index, spec in enumerate(specs):
                rows.append({
                    "workload_id": f"matmul_hw_{partition}_{short}_{index:02d}",
                    "m": str(spec.m), "n": str(spec.n), "k": str(spec.k),
                    "dtype": spec.dtype, "trans_a": str(spec.trans_a),
                    "trans_b": str(spec.trans_b), "max_cores": "20",
                    "search_family": "hardware_factor_region",
                    "target_kernel_family": family,
                    "calibration_partition": partition,
                    "required_successful_tilings": str(required),
                    "coverage": (
                        f"{partition};{family};{spec.dtype};"
                        f"t{spec.trans_a}{spec.trans_b};m{spec.m};n{spec.n};k{spec.k}"
                    ),
                })
    candidate_records = sum(
        int(row["required_successful_tilings"]) for row in rows
    )
    if len(rows) != 70 or candidate_records != 2185:
        raise RuntimeError(
            f"internal campaign size error: shapes={len(rows)} candidates={candidate_records}"
        )
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
    print("MATMUL_HARDWARE_CALIBRATION_CATALOG shapes=70 candidates=2185 baselines=70 records=2255")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
