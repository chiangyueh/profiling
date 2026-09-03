#!/usr/bin/env python3
"""Analyze within-shape MatMul tiling order for frozen regressions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from analyze_matmul_model_validation import paired_result, valid_profile
from profile_official_tilings import IncrementalJsonl


EXPECTED_SHAPES = 35
MINIMUM_MODEL_TILINGS = 20


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def number(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    return float(value) if value not in ("", None) else 0.0


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    begin = 0
    while begin < len(order):
        end = begin + 1
        while end < len(order) and values[order[end]] == values[order[begin]]:
            end += 1
        rank = (begin + 1 + end) / 2.0
        for position in range(begin, end):
            ranks[order[position]] = rank
        begin = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float:
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    left_norm = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_norm = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def spearman(left: list[float], right: list[float]) -> float:
    return pearson(average_ranks(left), average_ranks(right))


def pairwise_order_accuracy(predicted: list[float], measured: list[float]) -> float:
    correct = 0.0
    comparable = 0
    for first in range(len(predicted)):
        for second in range(first + 1, len(predicted)):
            predicted_delta = predicted[first] - predicted[second]
            measured_delta = measured[first] - measured[second]
            if predicted_delta == 0 or measured_delta == 0:
                continue
            comparable += 1
            if predicted_delta * measured_delta > 0:
                correct += 1.0
    return correct / comparable if comparable else 0.0


def verdict_counts(comparisons: list[dict]) -> dict[str, int]:
    return dict(Counter(item["verdict"] for item in comparisons))


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"shape_count": 0}
    top = [row["model_top1_vs_official"] for row in rows]
    best = [row["measured_best_vs_official"] for row in rows]
    return {
        "shape_count": len(rows),
        "model_candidate_measurements": sum(
            row["successful_model_tilings"] for row in rows
        ),
        "latency_records_including_official": sum(
            row["successful_model_tilings"] + 1 for row in rows
        ),
        "model_top1_verdicts": verdict_counts(top),
        "measured_best_verdicts": verdict_counts(best),
        "median_spearman": statistics.median(
            row["spearman_predicted_vs_measured"] for row in rows
        ),
        "median_pairwise_order_accuracy": statistics.median(
            row["pairwise_order_accuracy"] for row in rows
        ),
        "median_top1_regret_pct": statistics.median(
            row["model_top1_regret_pct"] for row in rows
        ),
        "maximum_top1_regret_pct": max(
            row["model_top1_regret_pct"] for row in rows
        ),
        "model_top1_within_measured_noise_count": sum(
            row["model_top1_within_measured_best_noise"] for row in rows
        ),
        "measured_best_model_rank_capture": {
            "top1": sum(row["measured_best_model_rank"] <= 1 for row in rows),
            "top5": sum(row["measured_best_model_rank"] <= 5 for row in rows),
            "top10": sum(row["measured_best_model_rank"] <= 10 for row in rows),
            "top20": sum(row["measured_best_model_rank"] <= 20 for row in rows),
        },
        "measured_families": dict(Counter(
            candidate["kernel_family"]
            for row in rows for candidate in row["candidates"]
        )),
        "measured_core_counts": dict(sorted(Counter(
            candidate["used_core_num"]
            for row in rows for candidate in row["candidates"]
        ).items())),
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
    parser.add_argument(
        "--minimum-model-tilings",
        type=int,
        default=MINIMUM_MODEL_TILINGS,
    )
    args = parser.parse_args()

    workloads = read_rows(args.workloads)
    candidates = read_rows(args.candidates)
    profiles = read_rows(args.profile)
    official_profiles = read_rows(args.official_profile)
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
        for row in official_profiles
        if valid_profile(row)
    }

    per_shape: list[dict] = []
    for workload in workloads:
        workload_id = workload["workload_id"]
        candidate_map = candidates_by_id.get(workload_id, {})
        profile_map = profiles_by_id.get(workload_id, {})
        official = official_by_id.get(workload_id)
        common_ranks = sorted(
            set(candidate_map) & set(profile_map), key=int
        )
        if official is None:
            raise RuntimeError(f"{workload_id}: valid official baseline is missing")
        if len(common_ranks) < args.minimum_model_tilings:
            raise RuntimeError(
                f"{workload_id}: {len(common_ranks)} successful model tilings, "
                f"required {args.minimum_model_tilings}"
            )

        joined = []
        for output_rank in common_ranks:
            candidate = candidate_map[output_rank]
            measured = profile_map[output_rank]
            joined.append({
                "output_rank": int(output_rank),
                "global_model_rank": int(
                    candidate.get("global_model_rank")
                    or candidate.get("new_model_rank")
                    or output_rank
                ),
                "predicted_cycles": number(candidate, "new_model_cycles"),
                "measured_ms": number(measured, "median_ms"),
                "stddev_ms": number(measured, "stddev_ms"),
                "kernel_family": candidate.get("model_kernel_family", ""),
                "kernel_suffix": candidate.get("model_kernel_suffix", ""),
                "used_core_num": int(candidate.get("used_core_num", "0") or 0),
                "single_core": {
                    "m": int(candidate.get("single_core_m", "0") or 0),
                    "n": int(candidate.get("single_core_n", "0") or 0),
                    "k": int(candidate.get("single_core_k", "0") or 0),
                },
                "base": {
                    "m": int(candidate.get("base_m", "0") or 0),
                    "n": int(candidate.get("base_n", "0") or 0),
                    "k": int(candidate.get("base_k", "0") or 0),
                },
                "selection_reason": candidate.get(
                    "diagnostic_selection_reason", ""
                ),
                "bottleneck": candidate.get("new_model_bottleneck", ""),
                "predicted_breakdown": json.loads(
                    candidate.get("new_model_breakdown") or "{}"
                ),
                "versus_official": paired_result(official, measured),
            })

        by_prediction = sorted(
            joined,
            key=lambda row: (
                row["predicted_cycles"], row["global_model_rank"],
                row["output_rank"],
            ),
        )
        by_measurement = sorted(
            joined,
            key=lambda row: (row["measured_ms"], row["output_rank"]),
        )
        measured_rank = {
            row["output_rank"]: rank
            for rank, row in enumerate(by_measurement, 1)
        }
        model_rank = {
            row["output_rank"]: rank
            for rank, row in enumerate(by_prediction, 1)
        }
        for row in joined:
            row["measured_rank"] = measured_rank[row["output_rank"]]
            row["model_rank_within_measured"] = model_rank[row["output_rank"]]

        model_top = by_prediction[0]
        measured_best = by_measurement[0]
        model_top_profile = profile_map[str(model_top["output_rank"])]
        measured_best_profile = profile_map[str(measured_best["output_rank"])]
        top_best_noise_pct = max(
            1.0,
            200.0 * math.hypot(
                number(model_top_profile, "stddev_ms"),
                number(measured_best_profile, "stddev_ms"),
            ) / measured_best["measured_ms"],
        )
        top1_regret_pct = 100.0 * (
            model_top["measured_ms"] / measured_best["measured_ms"] - 1.0
        )
        predicted = [row["predicted_cycles"] for row in joined]
        measured = [row["measured_ms"] for row in joined]
        per_shape.append({
            "workload": workload,
            "partition": workload.get("diagnostic_partition", "analysis"),
            "successful_model_tilings": len(joined),
            "official": {
                "median_ms": number(official, "median_ms"),
                "stddev_ms": number(official, "stddev_ms"),
            },
            "spearman_predicted_vs_measured": spearman(predicted, measured),
            "pairwise_order_accuracy": pairwise_order_accuracy(
                predicted, measured
            ),
            "model_top1_output_rank": model_top["output_rank"],
            "model_top1_global_rank": model_top["global_model_rank"],
            "measured_best_output_rank": measured_best["output_rank"],
            "measured_best_model_rank": model_rank[
                measured_best["output_rank"]
            ],
            "measured_best_global_model_rank": measured_best[
                "global_model_rank"
            ],
            "model_top1_regret_pct": top1_regret_pct,
            "model_top1_within_measured_best_noise": (
                top1_regret_pct <= top_best_noise_pct
            ),
            "model_top1_vs_official": model_top["versus_official"],
            "measured_best_vs_official": measured_best["versus_official"],
            "candidates": sorted(joined, key=lambda row: row["output_rank"]),
        })

    partitions = sorted({row["partition"] for row in per_shape})
    result = {
        "schema": "matmul_regression_ranking_diagnostic_v1",
        "status": "complete",
        "method": {
            "workloads": "35 frozen v3 regressions above measured noise",
            "candidate_selection": (
                "model top1 plus legal family, core occupancy, and "
                "family/core intersections; model-rank fill last"
            ),
            "measurement": (
                "device-event latency after full numeric signed-axis preflight"
            ),
            "minimum_model_tilings_per_shape": args.minimum_model_tilings,
            "official_baselines_per_shape": 1,
            "latency_history_or_cce_table_used_by_model": False,
        },
        "aggregate": aggregate(per_shape),
        "by_partition": {
            partition: aggregate([
                row for row in per_shape if row["partition"] == partition
            ])
            for partition in partitions
        },
        "per_shape": per_shape,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger = IncrementalJsonl(args.log_directory, 50 * 1024 * 1024)
    for row in per_shape:
        logger.write(
            f"ranking:{row['workload']['workload_id']}",
            {
                "schema": result["schema"],
                "record_type": "within_shape_ranking_analysis",
                **row,
            },
        )
    logger.write(
        "campaign:regression_ranking_analysis_complete",
        {
            "schema": result["schema"],
            "record_type": "regression_ranking_analysis_summary",
            "status": "complete",
            "aggregate": result["aggregate"],
            "by_partition": result["by_partition"],
        },
    )
    logger.close()
    summary = result["aggregate"]
    print(
        "MATMUL_REGRESSION_RANKING_COMPLETE "
        f"shapes={summary['shape_count']} "
        f"model_tilings={summary['model_candidate_measurements']} "
        f"records={summary['latency_records_including_official']} "
        f"median_spearman={summary['median_spearman']:.6f} "
        f"median_top1_regret_pct={summary['median_top1_regret_pct']:.6f} "
        f"logs={args.log_directory}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
