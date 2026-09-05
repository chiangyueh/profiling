#!/usr/bin/env python3
"""Analyze controlled MatMul factor sweeps without mixing holdout feedback."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from profile_direct_matmul import JsonlLog


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
    delta_pct = 100.0 * (selected_ms - official_ms) / official_ms
    noise_pct = max(
        1.0,
        200.0 * math.hypot(
            number(official, "stddev_ms"), number(selected, "stddev_ms")
        ) / official_ms,
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
    correct = 0
    comparable = 0
    for first in range(len(predicted)):
        for second in range(first + 1, len(predicted)):
            predicted_delta = predicted[first] - predicted[second]
            measured_delta = measured[first] - measured[second]
            if predicted_delta == 0 or measured_delta == 0:
                continue
            comparable += 1
            correct += predicted_delta * measured_delta > 0
    return correct / comparable if comparable else 0.0


def paired_hardware_effects(rows: list[dict]) -> dict[str, dict]:
    """Summarize one-factor changes against their same-run anchor."""

    anchors = {
        row["pair_id"]: row for row in rows
        if row.get("pair_id") and row.get("design_role") == "paired_anchor"
    }
    effects: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        pair_id = row.get("pair_id", "")
        if row.get("design_role") != "paired_factor" or pair_id not in anchors:
            continue
        anchor = anchors[pair_id]
        delta_pct = 100.0 * (
            row["measured_ms"] / anchor["measured_ms"] - 1.0
        )
        effects[row["controlled_factor"]].append({
            "pair_id": pair_id,
            "output_rank": row["output_rank"],
            "anchor_output_rank": anchor["output_rank"],
            "factor_signature": row["factor_signature"],
            "candidate_ms": row["measured_ms"],
            "anchor_ms": anchor["measured_ms"],
            "delta_pct": delta_pct,
        })
    return {
        factor: {
            "pair_count": len(values),
            "median_delta_pct": statistics.median(
                value["delta_pct"] for value in values
            ),
            "minimum_delta_pct": min(value["delta_pct"] for value in values),
            "maximum_delta_pct": max(value["delta_pct"] for value in values),
            "measurements": sorted(
                values, key=lambda value: (value["pair_id"], value["output_rank"])
            ),
        }
        for factor, values in sorted(effects.items())
    }


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"shape_count": 0}
    top = [row["model_top1_vs_official"] for row in rows]
    best = [row["measured_best_vs_official"] for row in rows]
    verdicts = lambda values: dict(Counter(row["verdict"] for row in values))
    return {
        "shape_count": len(rows),
        "model_candidate_measurements": sum(
            row["successful_model_tilings"] for row in rows
        ),
        "latency_records_including_official": sum(
            row["successful_model_tilings"] + 1 for row in rows
        ),
        "model_top1_verdicts": verdicts(top),
        "measured_best_verdicts": verdicts(best),
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
        "model_family_choice_correct_count": sum(
            row["model_top1_kernel_family"]
            == row["measured_best_kernel_family"]
            for row in rows
        ),
        "model_family_choice_accuracy": sum(
            row["model_top1_kernel_family"]
            == row["measured_best_kernel_family"]
            for row in rows
        ) / len(rows),
        "production_source_anchor_delta_pct": {
            "median": statistics.median(
                row["production_source_anchor_vs_official"]["delta_pct"]
                for row in rows
            ),
            "maximum_absolute": max(
                abs(row["production_source_anchor_vs_official"]["delta_pct"])
                for row in rows
            ),
        },
        "source_anchor_validator_disagreement_count": sum(
            bool(candidate.get("source_validator_violations"))
            for row in rows for candidate in row["candidates"]
            if candidate.get("source_anchor")
        ),
        "measured_best_model_rank_capture": {
            f"top{limit}": sum(
                row["measured_best_model_rank"] <= limit for row in rows
            ) for limit in (1, 5, 10, 20)
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
    parser.add_argument("--require-direct-tiling-applied", action="store_true")
    args = parser.parse_args()

    workloads = read_rows(args.workloads)
    candidates = read_rows(args.candidates)
    profiles = read_rows(args.profile)
    official_profiles = read_rows(args.official_profile)
    if len(workloads) != args.expected_shapes:
        raise RuntimeError(
            f"workload count {len(workloads)}, expected {args.expected_shapes}"
        )
    formal_total = sum(
        int(row["required_successful_tilings"]) for row in workloads
    )
    record_total = formal_total + len(workloads)

    candidates_by_id: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in candidates:
        if row.get("candidate_role") == "searched":
            candidates_by_id[row["workload_id"]][row["rank"]] = row
    if args.require_direct_tiling_applied:
        unattested = []
        for row in profiles:
            if row.get("candidate_role") != "searched" or not valid_profile(row):
                continue
            candidate = candidates_by_id.get(row["workload_id"], {}).get(row["rank"])
            if candidate is None or not (
                row.get("measurement_source") == "direct_tiling_buffer"
                and row.get("tiling_applied") == "1"
                and row.get("full_output_validated") == "1"
                and row.get("actual_tiling_sha256", "")
                and row.get("actual_tiling_fnv1a64", "")
                and row.get("model_schedule_sha256")
                    == candidate.get("model_schedule_sha256")
                and row.get("actual_kernel_suffix")
                    == candidate.get("model_kernel_suffix")
                and row.get("actual_block_dim")
                    == candidate.get("used_core_num")
            ):
                unattested.append(row)
        if unattested:
            raise RuntimeError(
                "successful candidates lack exact direct-kernel execution "
                f"attestation: {len(unattested)} rows"
            )
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
                "pair_id": candidate.get("pair_id", ""),
                "design_role": candidate.get("design_role", ""),
                "hardware_stratum": candidate.get("hardware_stratum", ""),
                "is_reserve": candidate["is_reserve"] == "1",
                "source_anchor": candidate.get("source_anchor") == "1",
                "source_route": candidate.get("source_route", ""),
                "source_core_cap": candidate.get("source_core_cap", ""),
                "source_validator_violations": candidate.get(
                    "source_validator_violations", ""
                ),
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
        production_source_anchors = [
            row for row in joined
            if row["source_anchor"]
            and "ALL" in row["source_route"].split("+")
            and "20" in row["source_core_cap"].split(",")
        ]
        if not production_source_anchors:
            raise RuntimeError(
                f"{workload_id}: measured candidates lack the original ALL@20 source anchor"
            )
        production_source_anchor = min(
            production_source_anchors, key=lambda row: row["measured_ms"]
        )
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
            "coverage_intent": workload["coverage_intent"],
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
            "model_top1_kernel_family": model_top["kernel_family"],
            "measured_best_output_rank": measured_best["output_rank"],
            "measured_best_model_rank": model_rank[measured_best["output_rank"]],
            "measured_best_global_model_rank": measured_best["global_model_rank"],
            "measured_best_kernel_family": measured_best["kernel_family"],
            "model_top1_regret_pct": regret,
            "model_top1_within_measured_best_noise": regret <= noise_pct,
            "model_top1_vs_official": model_top["versus_official"],
            "measured_best_vs_official": measured_best["versus_official"],
            "production_source_anchor_output_rank": production_source_anchor["output_rank"],
            "production_source_anchor_vs_official": production_source_anchor["versus_official"],
            "paired_hardware_effects": paired_hardware_effects(joined),
            "candidates": sorted(joined, key=lambda row: row["output_rank"]),
        })

    partitions = sorted({row["partition"] for row in per_shape})
    coverage_intents = sorted({row["coverage_intent"] for row in per_shape})
    result = {
        "schema": "matmul_hardware_frontier_measurement_v2",
        "status": "complete",
        "method": {
            "candidate_selection": (
                "every applicable original source route/core-cap anchor plus "
                "hardware-local factor strata; model scored only after freeze"
            ),
            "measurement": "one warmup, three device-event launches, full validation of final timed output",
            "candidate_latency_records": formal_total,
            "official_baselines": len(workloads),
            "latency_records": record_total,
            "latency_history_or_cce_table_used_by_model": False,
            "candidate_set_frozen_before_simulator_scoring": True,
            "holdout_feedback_permitted_during_calibration": False,
            "direct_tiling_buffer_execution_attested": (
                args.require_direct_tiling_applied
            ),
        },
        "aggregate": aggregate(per_shape),
        "by_partition": {
            partition: aggregate([
                row for row in per_shape if row["partition"] == partition
            ]) for partition in partitions
        },
        "by_coverage_intent": {
            intent: aggregate([
                row for row in per_shape
                if row["coverage_intent"] == intent
            ]) for intent in coverage_intents
        },
        "per_shape": per_shape,
    }
    if result["aggregate"]["latency_records_including_official"] != record_total:
        raise RuntimeError(
            f"formal latency record count is not {record_total}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    log = JsonlLog(args.log_directory, 50 * 1024 * 1024)
    for row in per_shape:
        log.append_once(
            f"hardware-ranking:{row['workload']['workload_id']}",
            {"record_type": "hardware_factor_ranking", **row},
        )
    log.append_once(
        "campaign:victor_frontier_analysis_complete",
        {
            "schema": result["schema"],
            "record_type": "hardware_factor_analysis_summary",
            "status": "complete",
            "aggregate": result["aggregate"],
            "by_partition": result["by_partition"],
            "by_coverage_intent": result["by_coverage_intent"],
        },
    )
    log.close()
    summary = result["aggregate"]
    print(
        "MATMUL_VICTOR_FRONTIER_COMPLETE "
        f"shapes={summary['shape_count']} records={record_total} "
        f"median_spearman={summary['median_spearman']:.6f} "
        f"median_top1_regret_pct={summary['median_top1_regret_pct']:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
