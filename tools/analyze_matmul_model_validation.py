#!/usr/bin/env python3
"""Analyze paired old/new cost-model rankings against NPU measurements."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from profile_official_tilings import IncrementalJsonl


EXPECTED_SHAPES = 200
EXPECTED_TILINGS = 24
COLLEAGUE_US = {
    "matmul_rank_000": {"official": 0.30954, "best": 0.29842},
    "matmul_rank_001": {"official": 0.40432, "best": 0.39330},
    "matmul_rank_002": {"official": 0.21746, "best": 0.21046},
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def truthy(value: object) -> bool:
    return str(value or "").lower() in {"1", "true", "yes", "on"}


def valid_profile(row: dict[str, str]) -> bool:
    return (
        truthy(row.get("success"))
        and truthy(row.get("preflight_passed"))
        and row.get("preflight_mode") == "numeric_signed_axes_full_v3"
        and float(row.get("median_ms") or 0) > 0
    )


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def spearman(left: list[float], right: list[float]) -> float:
    if len(left) < 2:
        return float("nan")
    a = rankdata(left)
    b = rankdata(right)
    mean_a = statistics.fmean(a)
    mean_b = statistics.fmean(b)
    numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    denominator = math.sqrt(
        sum((x - mean_a) ** 2 for x in a)
        * sum((y - mean_b) ** 2 for y in b)
    )
    return numerator / denominator if denominator else float("nan")


def pair_accuracy(
    scores: list[float], latencies: list[float], deviations: list[float]
) -> tuple[int, int]:
    correct = 0
    comparable = 0
    for first in range(len(scores)):
        for second in range(first + 1, len(scores)):
            noise_ms = max(
                0.01 * min(latencies[first], latencies[second]),
                2.0 * math.hypot(deviations[first], deviations[second]),
            )
            if abs(latencies[first] - latencies[second]) <= noise_ms:
                continue
            comparable += 1
            if (scores[first] < scores[second]) == (
                latencies[first] < latencies[second]
            ):
                correct += 1
    return correct, comparable


def model_metrics(
    name: str,
    records: list[dict],
    official_ms: float,
    official_std: float,
) -> dict:
    score_name = f"{name}_model_cycles"
    scores = [float(record["candidate"][score_name]) for record in records]
    latencies = [float(record["profile"]["median_ms"]) for record in records]
    deviations = [float(record["profile"]["stddev_ms"] or 0) for record in records]
    selected = min(range(len(records)), key=scores.__getitem__)
    oracle = min(range(len(records)), key=latencies.__getitem__)
    top3 = set(sorted(range(len(records)), key=scores.__getitem__)[:3])
    correct, comparable = pair_accuracy(scores, latencies, deviations)
    selected_delta = 100.0 * (latencies[selected] - official_ms) / official_ms
    selected_noise = max(
        1.0,
        200.0 * math.hypot(official_std, deviations[selected]) / official_ms,
    )
    if selected_delta < -selected_noise:
        selected_verdict = "improved"
    elif selected_delta > selected_noise:
        selected_verdict = "regressed"
    else:
        selected_verdict = "within_noise"
    return {
        "spearman": spearman(scores, latencies),
        "pair_correct": correct,
        "pair_comparable": comparable,
        "pair_accuracy": correct / comparable if comparable else None,
        "selected_rank": int(records[selected]["candidate"]["rank"]),
        "selected_role": records[selected]["candidate"]["candidate_role"],
        "selected_ms": latencies[selected],
        "selected_speedup_vs_official": official_ms / latencies[selected],
        "selected_delta_pct": selected_delta,
        "selected_noise_pct": selected_noise,
        "selected_verdict_vs_official": selected_verdict,
        "top1_regret_pct": 100.0 * (latencies[selected] / latencies[oracle] - 1.0),
        "top3_contains_measured_fastest": oracle in top3,
    }


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--official-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-directory", type=Path, required=True)
    parser.add_argument("--expected-shapes", type=int, default=EXPECTED_SHAPES)
    parser.add_argument("--expected-tilings", type=int, default=EXPECTED_TILINGS)
    args = parser.parse_args()

    workloads = read_rows(args.workloads)
    candidates = read_rows(args.candidates)
    profiles = read_rows(args.profile)
    official_profiles = read_rows(args.official_profile)
    if len(workloads) != args.expected_shapes:
        raise RuntimeError(
            f"workload count {len(workloads)}, expected {args.expected_shapes}"
        )

    candidate_by_key = {
        (row["workload_id"], row["candidate_role"], row["rank"]): row
        for row in candidates
    }
    profiles_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in profiles:
        if row.get("candidate_role") in {"bank_seed_control", "searched"}:
            profiles_by_id[row["workload_id"]].append(row)
    official_by_id = {
        row["workload_id"]: row for row in official_profiles if valid_profile(row)
    }

    per_shape: list[dict] = []
    aggregate_pairs = {
        "old": [0, 0],
        "new": [0, 0],
    }
    control_mismatch = 0
    for workload in workloads:
        workload_id = workload["workload_id"]
        official = official_by_id.get(workload_id)
        if official is None:
            raise RuntimeError(f"{workload_id}: valid official baseline missing")
        records: list[dict] = []
        seen_signatures: set[str] = set()
        for profile in sorted(
            profiles_by_id[workload_id],
            key=lambda row: (
                0 if row["candidate_role"] == "bank_seed_control" else 1,
                int(row["rank"]),
            ),
        ):
            key = (workload_id, profile["candidate_role"], profile["rank"])
            candidate = candidate_by_key.get(key)
            signature = profile.get("tiling_signature", "")
            if (
                candidate is None
                or not valid_profile(profile)
                or not signature
                or signature in seen_signatures
            ):
                continue
            seen_signatures.add(signature)
            records.append({"candidate": candidate, "profile": profile})
        if len(records) != args.expected_tilings:
            raise RuntimeError(
                f"{workload_id}: {len(records)} valid distinct tilings, "
                f"expected {args.expected_tilings}"
            )

        official_ms = float(official["median_ms"])
        oracle_ms = min(float(row["profile"]["median_ms"]) for row in records)
        control = next(
            row for row in records
            if row["candidate"]["candidate_role"] == "bank_seed_control"
        )
        control_ms = float(control["profile"]["median_ms"])
        control_std = float(control["profile"]["stddev_ms"] or 0)
        official_std = float(official["stddev_ms"] or 0)
        control_noise = max(
            0.01 * official_ms,
            2.0 * math.hypot(control_std, official_std),
        )
        control_attested = abs(control_ms - official_ms) <= control_noise
        control_mismatch += not control_attested
        old_metrics = model_metrics("old", records, official_ms, official_std)
        new_metrics = model_metrics("new", records, official_ms, official_std)
        aggregate_pairs["old"][0] += old_metrics["pair_correct"]
        aggregate_pairs["old"][1] += old_metrics["pair_comparable"]
        aggregate_pairs["new"][0] += new_metrics["pair_correct"]
        aggregate_pairs["new"][1] += new_metrics["pair_comparable"]
        record = {
            "workload": workload,
            "official_ms": official_ms,
            "official_seed_control_ms": control_ms,
            "official_seed_control_attested": control_attested,
            "measured_fastest_ms": oracle_ms,
            "oracle_speedup_vs_official": official_ms / oracle_ms,
            "old_model": old_metrics,
            "new_model": new_metrics,
        }
        if workload_id in COLLEAGUE_US:
            record["colleague_reported"] = COLLEAGUE_US[workload_id]
        per_shape.append(record)

    aggregate: dict[str, object] = {
        "shape_count": len(per_shape),
        "tilings_per_shape": args.expected_tilings,
        "latency_record_count": len(per_shape) * (args.expected_tilings + 1),
        "official_seed_control_mismatch_count": control_mismatch,
        "oracle_median_speedup_vs_official": median(
            [row["oracle_speedup_vs_official"] for row in per_shape]
        ),
    }
    for name in ("old", "new"):
        correct, comparable = aggregate_pairs[name]
        metrics = [row[f"{name}_model"] for row in per_shape]
        aggregate[f"{name}_model"] = {
            "noise_filtered_pair_correct": correct,
            "noise_filtered_pair_comparable": comparable,
            "noise_filtered_pair_accuracy": (
                correct / comparable if comparable else None
            ),
            "median_spearman": median(
                [row["spearman"] for row in metrics if math.isfinite(row["spearman"])]
            ),
            "median_top1_regret_pct": median(
                [row["top1_regret_pct"] for row in metrics]
            ),
            "top3_fastest_recall": statistics.fmean(
                row["top3_contains_measured_fastest"] for row in metrics
            ),
            "median_selected_speedup_vs_official": median(
                [row["selected_speedup_vs_official"] for row in metrics]
            ),
            "selected_verdict_counts": {
                verdict: sum(
                    row["selected_verdict_vs_official"] == verdict
                    for row in metrics
                )
                for verdict in ("improved", "within_noise", "regressed")
            },
        }

    old_aggregate = aggregate["old_model"]
    new_aggregate = aggregate["new_model"]
    old_pair = old_aggregate["noise_filtered_pair_accuracy"]
    new_pair = new_aggregate["noise_filtered_pair_accuracy"]
    if old_pair is None or new_pair is None:
        aggregate["new_vs_old_verdict"] = "insufficient_noise_separated_pairs"
        pair_delta = None
    else:
        pair_delta = new_pair - old_pair
    regret_delta = (
        old_aggregate["median_top1_regret_pct"]
        - new_aggregate["median_top1_regret_pct"]
    )
    if pair_delta is None:
        pass
    elif pair_delta > 0 and regret_delta > 0:
        aggregate["new_vs_old_verdict"] = "better_on_both_ranking_metrics"
    elif pair_delta < 0 and regret_delta < 0:
        aggregate["new_vs_old_verdict"] = "worse_on_both_ranking_metrics"
    else:
        aggregate["new_vs_old_verdict"] = "mixed"

    by_family: dict[str, dict] = {}
    families = sorted({
        row["workload"].get("search_family", "unknown")
        for row in per_shape
    })
    for family in families:
        family_rows = [
            row for row in per_shape
            if row["workload"].get("search_family", "unknown") == family
        ]
        family_result: dict[str, object] = {"shape_count": len(family_rows)}
        for name in ("old", "new"):
            metrics = [row[f"{name}_model"] for row in family_rows]
            correct = sum(row["pair_correct"] for row in metrics)
            comparable = sum(row["pair_comparable"] for row in metrics)
            family_result[f"{name}_model"] = {
                "noise_filtered_pair_accuracy": (
                    correct / comparable if comparable else None
                ),
                "median_spearman": median([
                    row["spearman"] for row in metrics
                    if math.isfinite(row["spearman"])
                ]),
                "median_top1_regret_pct": median([
                    row["top1_regret_pct"] for row in metrics
                ]),
                "top3_fastest_recall": statistics.fmean(
                    row["top3_contains_measured_fastest"] for row in metrics
                ),
                "selected_verdict_counts": {
                    verdict: sum(
                        row["selected_verdict_vs_official"] == verdict
                        for row in metrics
                    )
                    for verdict in ("improved", "within_noise", "regressed")
                },
            }
        by_family[family] = family_result
    aggregate["by_search_family"] = by_family

    result = {
        "schema": "matmul_model_validation_analysis_v1",
        "status": "complete",
        "method": {
            "candidate_pool": "same callback-fixed parameter pool for both models",
            "measurement": "NPU device events after every-C numeric validation",
            "noise_filter": "max(1% of faster median, 2*hypot(stddevs))",
            "no_latency_training_or_operator_label_dispatch": True,
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
    old_result = aggregate["old_model"]
    new_result = aggregate["new_model"]
    old_pair_text = (
        f"{old_result['noise_filtered_pair_accuracy']:.4f}"
        if old_result["noise_filtered_pair_accuracy"] is not None else "NA"
    )
    new_pair_text = (
        f"{new_result['noise_filtered_pair_accuracy']:.4f}"
        if new_result["noise_filtered_pair_accuracy"] is not None else "NA"
    )
    print(
        "MATMUL_MODEL_VALIDATION_COMPLETE "
        f"shapes={len(per_shape)} records={aggregate['latency_record_count']} "
        f"old_pair={old_pair_text} new_pair={new_pair_text} "
        f"old_regret={old_result['median_top1_regret_pct']:.4f}% "
        f"new_regret={new_result['median_top1_regret_pct']:.4f}% "
        f"logs={args.log_directory}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
