#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from tiling_search.calibration_workloads import (
    encode_template_quotas,
    generate_template_calibration_workloads,
)
from tiling_search.domain import Hardware


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aic-cores", type=int, required=True)
    parser.add_argument("--l0a-bytes", type=int, required=True)
    parser.add_argument("--l0b-bytes", type=int, required=True)
    parser.add_argument("--l0c-bytes", type=int, required=True)
    parser.add_argument("--l1-bytes", type=int, required=True)
    parser.add_argument("--ub-bytes", type=int, required=True)
    parser.add_argument("--l2-bytes", type=int, required=True)
    parser.add_argument("--l2-bpc", type=float, default=1.0)
    parser.add_argument("--hbm-bpc", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hardware = Hardware(
        aic_cores=args.aic_cores,
        l0a_bytes=args.l0a_bytes,
        l0b_bytes=args.l0b_bytes,
        l0c_bytes=args.l0c_bytes,
        l1_bytes=args.l1_bytes,
        l2_bytes=args.l2_bytes,
        l2_bytes_per_cycle_per_core=args.l2_bpc,
        hbm_bytes_per_cycle_per_core=args.hbm_bpc,
        ub_bytes=args.ub_bytes,
    )
    specs = generate_template_calibration_workloads(hardware)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "id",
        "m",
        "n",
        "k",
        "dtype",
        "trans_a",
        "trans_b",
        "max_cores",
        "template_quotas",
        "design_axis",
        "design_value",
    )
    with args.output.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=columns)
        writer.writeheader()
        for spec in specs:
            workload = spec.workload
            writer.writerow(
                {
                    "id": workload.workload_id,
                    "m": workload.m,
                    "n": workload.n,
                    "k": workload.k,
                    "dtype": workload.dtype,
                    "trans_a": int(workload.trans_a),
                    "trans_b": int(workload.trans_b),
                    "max_cores": workload.max_cores,
                    "template_quotas": encode_template_quotas(
                        spec.template_quotas
                    ),
                    "design_axis": spec.design_axis,
                    "design_value": f"{spec.design_value:.8f}",
                }
            )
    totals: dict[str, int] = {}
    for spec in specs:
        for template, quota in spec.template_quotas.items():
            totals[template.value] = totals.get(template.value, 0) + quota
    print(
        "TEMPLATE_CALIBRATION_DESIGN "
        f"workloads={len(specs)} "
        f"l1_bytes={hardware.effective_l1_bytes} "
        "target_measurements="
        + ",".join(
            f"{name}:{totals[name]}" for name in sorted(totals)
        )
    )


if __name__ == "__main__":
    main()
