#!/usr/bin/env python3
"""Analyze controlled MatMul factor sweeps without mixing holdout feedback."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from analyze_matmul_model_validation import paired_result, valid_profile
from analyze_matmul_regression_diagnostic import (
    aggregate,
    number,
    pairwise_order_accuracy,
    spearman,
)
from profile_official_tilings import IncrementalJsonl


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--official-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-directory", type=Path, required=True)
    parser.add_argument("--expected-shapes", type=int, default=70)
    parser.add_argument("--require-runtime-kb-attested", action="store_true")
    args = parser.parse_args()

    workloads = read_rows(args.workloads)
    candidates = read_rows(args.candidates)
    profiles = read_rows(args.profile)
    official_profiles = read_rows(args.official_profile)
    if args.require_runtime_kb_attested:
        unattested = [
            row for row in profiles
            if row.get("candidate_role") == "searched"
            and valid_profile(row)
            and (
                row.get("runtime_kb_attested") != "1"
                or row.get("workspace_bytes", "") == ""
            )
        ]
        if unattested:
            raise RuntimeError(
                "successful candidate profiles lack RuntimeKb execution "
                f"attestation: {len(unattested)} rows"
            )
    if len(workloads) != args.expected_shapes:
        raise RuntimeError(
            f"workload count {len(workloads)}, expected {args.expected_shapes}"
        )

    candidates_by_id: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in candidates:
        if row.get("candidate_role") == "searched":
            candidates_by_id[row["workload_id"]][row["rank"]] = row
    profiles_by_id: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in profiles:
        if row.get("candidate_role") == "searched" and valid_profile(row):
            profiles_by_id[row["workload_id"]][row["rank"]] = row
    official_by_id = {
        row["workload_id"]: row
        for row in official_profiles if valid_profile(row)
    }

    per_shape: list[dict] = []
    for workload in workloads:
        workload_id = workload["workload_id"]
        required = int(workload["required_successful_tilings"])
        candidate_map = candidates_by_id.get(workload_id, {})
        profile_map = profiles_by_id.get(workload_id, {})
        official = official_by_id.get(workload_id)
        common_ranks = sorted(set(candidate_map) & set(profile_map), key=int)
        if official is None:
            raise RuntimeError(f"{workload_id}: valid official baseline is missing")
        if len(common_ranks) != required:
            raise RuntimeError(
                f"{workload_id}: successful={len(common_ranks)}, required={required}"
            )

        joined = []
        for output_rank in common_ranks:
            candidate = candidate_map[output_rank]
            measured = profile_map[output_rank]
            joined.append({
                "output_rank": int(output_rank),
                "global_model_rank": int(candidate["global_model_rank"]),
                "predicted_cycles": number(candidate, "new_model_cycles"),
                "measured_ms": number(measured, "median_ms"),
                "stddev_ms": number(measured, "stddev_ms"),
                "kernel_family": candidate["model_kernel_family"],
                "kernel_suffix": candidate["model_kernel_suffix"],
                "used_core_num": int(candidate["used_core_num"]),
                "controlled_factor": candidate["controlled_factor"],
                "factor_signature": candidate["factor_signature"],
                "is_reserve": candidate["is_reserve"] == "1",
                "single_core": {
                    axis: int(candidate[f"single_core_{axis}"])
                    for axis in ("m", "n", "k")
                },
                "base": {
                    axis: int(candidate[f"base_{axis}"])
                    for axis in ("m", "n", "k")
                },
                "bottleneck": candidate["new_model_bottleneck"],
                "predicted_breakdown": json.loads(
                    candidate.get("new_model_breakdown") or "{}"
                ),
                "versus_official": paired_result(official, measured),
            })

        predicted_order = sorted(
            joined, key=lambda row: (
                row["predicted_cycles"], row["global_model_rank"],
                row["output_rank"],
            )
        )
        measured_order = sorted(
            joined, key=lambda row: (row["measured_ms"], row["output_rank"])
        )
        measured_rank = {
            row["output_rank"]: rank
            for rank, row in enumerate(measured_order, 1)
        }
        model_rank = {
            row["output_rank"]: rank
            for rank, row in enumerate(predicted_order, 1)
        }
        for row in joined:
            row["measured_rank"] = measured_rank[row["output_rank"]]
            row["model_rank_within_measured"] = model_rank[row["output_rank"]]
        model_top = predicted_order[0]
        measured_best = measured_order[0]
        model_top_profile = profile_map[str(model_top["output_rank"])]
        measured_best_profile = profile_map[str(measured_best["output_rank"])]
        noise_pct = max(
            1.0,
            200.0 * math.hypot(
                number(model_top_profile, "stddev_ms"),
                number(measured_best_profile, "stddev_ms"),
            ) / measured_best["measured_ms"],
        )
        regret = 100.0 * (
            model_top["measured_ms"] / measured_best["measured_ms"] - 1.0
        )
        per_shape.append({
            "workload": workload,
            "partition": workload["calibration_partition"],
            "target_kernel_family": workload["target_kernel_family"],
            "successful_model_tilings": len(joined),
            "official": {
                "median_ms": number(official, "median_ms"),
                "stddev_ms": number(official, "stddev_ms"),
            },
            "spearman_predicted_vs_measured": spearman(
                [row["predicted_cycles"] for row in joined],
                [row["measured_ms"] for row in joined],
            ),
            "pairwise_order_accuracy": pairwise_order_accuracy(
                [row["predicted_cycles"] for row in joined],
                [row["measured_ms"] for row in joined],
            ),
            "model_top1_output_rank": model_top["output_rank"],
            "model_top1_global_rank": model_top["global_model_rank"],
            "measured_best_output_rank": measured_best["output_rank"],
            "measured_best_model_rank": model_rank[measured_best["output_rank"]],
            "measured_best_global_model_rank": measured_best["global_model_rank"],
            "model_top1_regret_pct": regret,
            "model_top1_within_measured_best_noise": regret <= noise_pct,
            "model_top1_vs_official": model_top["versus_official"],
            "measured_best_vs_official": measured_best["versus_official"],
            "candidates": sorted(joined, key=lambda row: row["output_rank"]),
        })

    partitions = ("calibration", "holdout")
    families = sorted({row["target_kernel_family"] for row in per_shape})
    result = {
        "schema": "matmul_hardware_factor_calibration_v1",
        "status": "complete",
        "method": {
            "candidate_selection": "bounded one-factor hardware sweeps plus legal reserves",
            "measurement": "one warmup, three device-event launches, full validation of final timed output",
            "candidate_latency_records": 2185,
            "official_baselines": 70,
            "latency_records": 2255,
            "latency_history_or_cce_table_used_by_model": False,
            "holdout_feedback_permitted_during_calibration": False,
            "runtime_kb_execution_attested": args.require_runtime_kb_attested,
        },
        "aggregate": aggregate(per_shape),
        "by_partition": {
            partition: aggregate([
                row for row in per_shape if row["partition"] == partition
            ]) for partition in partitions
        },
        "by_family": {
            family: aggregate([
                row for row in per_shape
                if row["target_kernel_family"] == family
            ]) for family in families
        },
        "per_shape": per_shape,
    }
    if result["aggregate"]["latency_records_including_official"] != 2255:
        raise RuntimeError("formal latency record count is not 2255")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    log = IncrementalJsonl(args.log_directory, 50 * 1024 * 1024)
    for row in per_shape:
        log.write(
            f"hardware-ranking:{row['workload']['workload_id']}",
            {"record_type": "hardware_factor_ranking", **row},
        )
    log.write(
        "campaign:hardware_calibration_analysis_complete",
        {
            "schema": result["schema"],
            "record_type": "hardware_factor_analysis_summary",
            "status": "complete",
            "aggregate": result["aggregate"],
            "by_partition": result["by_partition"],
            "by_family": result["by_family"],
        },
    )
    log.close()
    summary = result["aggregate"]
    print(
        "MATMUL_HARDWARE_CALIBRATION_COMPLETE "
        f"shapes={summary['shape_count']} records=2255 "
        f"median_spearman={summary['median_spearman']:.6f} "
        f"median_top1_regret_pct={summary['median_top1_regret_pct']:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
