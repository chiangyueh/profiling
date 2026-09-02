#!/usr/bin/env python3
"""Analyze one hardware-simulator choice against official MatMulV3."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from profile_official_tilings import IncrementalJsonl


EXPECTED_SHAPES = 200
COLLEAGUE_US = {
    "matmul_rank_000": {"official": 0.30954, "best": 0.29842},
    "matmul_rank_001": {"official": 0.40432, "best": 0.39330},
    "matmul_rank_002": {"official": 0.21746, "best": 0.21046},
}
SELECTION_TIMING_FIELDS = (
    "candidate_generation_ms",
    "static_legality_ms",
    "tiling_solver_select_ms",
    "official_callback_ms",
    "final_callback_ms",
    "tiling_solver_total_ms",
)
EXECUTION_TIMING_FIELDS = (
    "bank_prepare_ms",
    "host_runner_wall_ms",
    "device_prepare_ms",
    "executor_setup_ms",
    "numeric_preflight_ms",
    "warmup_wall_ms",
    "measurement_wall_ms",
    "runner_total_ms",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def truthy(value: object) -> bool:
    return str(value or "").lower() in {"1", "true", "yes", "on"}


def number(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    return float(value) if value not in ("", None) else 0.0


def valid_profile(row: dict[str, str]) -> bool:
    return (
        truthy(row.get("success"))
        and truthy(row.get("preflight_passed"))
        and row.get("preflight_mode") == "numeric_signed_axes_full_v3"
        and number(row, "median_ms") > 0
    )


def paired_result(
    official: dict[str, str], selected: dict[str, str]
) -> dict[str, float | str]:
    official_ms = number(official, "median_ms")
    selected_ms = number(selected, "median_ms")
    official_std = number(official, "stddev_ms")
    selected_std = number(selected, "stddev_ms")
    delta_pct = 100.0 * (selected_ms - official_ms) / official_ms
    noise_pct = max(
        1.0,
        200.0 * math.hypot(official_std, selected_std) / official_ms,
    )
    verdict = (
        "improved" if delta_pct < -noise_pct
        else "regressed" if delta_pct > noise_pct
        else "within_noise"
    )
    return {
        "official_ms": official_ms,
        "selected_ms": selected_ms,
        "speedup_vs_official": official_ms / selected_ms,
        "delta_pct": delta_pct,
        "noise_pct": noise_pct,
        "verdict": verdict,
    }


def timing_values(row: dict[str, str], fields: tuple[str, ...]) -> dict[str, float]:
    return {field: number(row, field) for field in fields}


def aggregate_timing(rows: list[dict[str, float]]) -> dict:
    if not rows:
        return {}
    result = {}
    totals = {}
    for field in rows[0]:
        values = [row[field] for row in rows]
        totals[field] = sum(values)
        result[field] = {
            "total_ms": totals[field],
            "mean_ms": statistics.fmean(values),
            "median_ms": statistics.median(values),
            "max_ms": max(values),
        }
    envelope_fields = {
        "tiling_solver_total_ms",
        "host_runner_wall_ms",
        "runner_total_ms",
    }
    positive = {
        field: value
        for field, value in totals.items()
        if value > 0 and field not in envelope_fields
    }
    result["largest_total_stage"] = (
        max(positive, key=positive.get) if positive else None
    )
    return result


def summarize(rows: list[dict]) -> dict:
    speedups = [row["comparison"]["speedup_vs_official"] for row in rows]
    return {
        "shape_count": len(rows),
        "verdict_counts": dict(Counter(
            row["comparison"]["verdict"] for row in rows
        )),
        "geomean_speedup_vs_official": math.exp(
            statistics.fmean(math.log(value) for value in speedups)
        ),
        "median_speedup_vs_official": statistics.median(speedups),
        "total_latency_speedup_vs_official": (
            sum(row["comparison"]["official_ms"] for row in rows)
            / sum(row["comparison"]["selected_ms"] for row in rows)
        ),
        "median_delta_pct": statistics.median(
            row["comparison"]["delta_pct"] for row in rows
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--official-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-directory", type=Path, required=True)
    parser.add_argument("--expected-shapes", type=int, default=EXPECTED_SHAPES)
    args = parser.parse_args()

    workloads = read_rows(args.workloads)
    candidates = read_rows(args.candidates)
    profiles = read_rows(args.profile)
    official_profiles = read_rows(args.official_profile)
    if len(workloads) != args.expected_shapes:
        raise RuntimeError(
            f"workload count {len(workloads)}, expected {args.expected_shapes}"
        )
    if len(candidates) != args.expected_shapes:
        raise RuntimeError(
            f"selected candidate count {len(candidates)}, expected {args.expected_shapes}"
        )

    candidate_by_id = {
        row["workload_id"]: row
        for row in candidates
        if row.get("candidate_role") == "searched" and row.get("rank") == "1"
    }
    profile_by_id = {
        row["workload_id"]: row
        for row in profiles
        if row.get("candidate_role") == "searched"
        and row.get("rank") == "1"
        and valid_profile(row)
    }
    official_by_id = {
        row["workload_id"]: row
        for row in official_profiles
        if valid_profile(row)
    }

    per_shape: list[dict] = []
    for workload in workloads:
        workload_id = workload["workload_id"]
        candidate = candidate_by_id.get(workload_id)
        selected = profile_by_id.get(workload_id)
        official = official_by_id.get(workload_id)
        if candidate is None or selected is None or official is None:
            raise RuntimeError(
                f"{workload_id}: complete official/selected pair is missing"
            )
        record = {
            "workload": workload,
            "selected_tiling": {
                "used_core_num": candidate.get("used_core_num", ""),
                "base_m": candidate.get("base_m", ""),
                "base_n": candidate.get("base_n", ""),
                "base_k": candidate.get("base_k", ""),
                "single_core_m": candidate.get("single_core_m", ""),
                "single_core_n": candidate.get("single_core_n", ""),
                "single_core_k": candidate.get("single_core_k", ""),
                "kernel_family": candidate.get("callback_kernel_family", ""),
            },
            "predicted_ratio_vs_official": number(
                candidate, "new_model_ratio_vs_official"
            ),
            "predicted_cycles": number(candidate, "new_model_cycles"),
            "model_bottleneck": candidate.get("new_model_bottleneck", ""),
            "ideal_region": {
                "anchor_count": int(candidate.get("ideal_anchor_count", "0") or 0),
                "plan_count": int(candidate.get("ideal_region_count", "0") or 0),
                "evaluations": int(
                    candidate.get("ideal_discovery_evaluations", "0") or 0
                ),
                "execution_graphs_represented": candidate.get(
                    "execution_graphs_represented", ""
                ).split(";"),
            },
            "comparison": paired_result(official, selected),
            "selection_timing_ms": timing_values(
                candidate, SELECTION_TIMING_FIELDS
            ),
            "official_execution_timing_ms": timing_values(
                official, EXECUTION_TIMING_FIELDS
            ),
            "selected_execution_timing_ms": timing_values(
                selected, EXECUTION_TIMING_FIELDS
            ),
        }
        if workload_id in COLLEAGUE_US:
            record["colleague_reported"] = COLLEAGUE_US[workload_id]
        per_shape.append(record)

    aggregate = summarize(per_shape)
    aggregate["latency_record_count"] = 2 * len(per_shape)
    aggregate["selection_timing"] = aggregate_timing([
        row["selection_timing_ms"] for row in per_shape
    ])
    aggregate["official_execution_timing"] = aggregate_timing([
        row["official_execution_timing_ms"] for row in per_shape
    ])
    aggregate["selected_execution_timing"] = aggregate_timing([
        row["selected_execution_timing_ms"] for row in per_shape
    ])
    aggregate["by_selected_execution_graph"] = {
        family: summarize([
            row for row in per_shape
            if row["selected_tiling"].get("kernel_family") == family
        ])
        for family in sorted({
            row["selected_tiling"].get("kernel_family", "unknown")
            for row in per_shape
        })
    }

    result = {
        "schema": "matmul_hardware_simulator_validation_v3",
        "status": "complete",
        "method": {
            "selection": (
                "one final tiling chosen from hardware-derived local optima "
                "and adjacent schedule transitions"
            ),
            "comparison": "selected tiling versus installed official MatMulV3 on the same NPU",
            "measurement": "device-event latency after every-C numeric validation",
            "noise_filter": "max(1% of official median, 2*hypot(stddevs))",
            "old_cost_model_used": False,
            "latency_history_or_cce_table_used": False,
        },
        "aggregate": aggregate,
        "per_shape": per_shape,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger = IncrementalJsonl(args.log_directory, 50 * 1024 * 1024)
    for row in per_shape:
        workload_id = row["workload"]["workload_id"]
        logger.write(
            f"analysis:{workload_id}",
            {
                "schema": result["schema"],
                "record_type": "workload_analysis",
                **row,
            },
        )
    logger.write(
        "campaign:analysis_complete",
        {
            "schema": result["schema"],
            "record_type": "campaign_analysis",
            "status": "complete",
            "aggregate": aggregate,
        },
    )
    logger.close()
    verdicts = aggregate["verdict_counts"]
    print(
        "MATMUL_MODEL_VALIDATION_COMPLETE "
        f"shapes={len(per_shape)} records={aggregate['latency_record_count']} "
        f"improved={verdicts.get('improved', 0)} "
        f"within_noise={verdicts.get('within_noise', 0)} "
        f"regressed={verdicts.get('regressed', 0)} "
        f"geomean_speedup={aggregate['geomean_speedup_vs_official']:.6f} "
        f"logs={args.log_directory}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
