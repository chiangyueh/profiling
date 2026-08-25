#!/usr/bin/env python3
"""Validate hardware-path ranking against held-out device-event evidence.

The scorer receives only dtype semantics and raw source-tiling fields.  Device
latency is kept in a separate map and joined only after every prediction has
returned; this script performs no fitting or parameter update.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


VALUE_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4, "int32": 4, "int8": 1, "uint8": 1}
INDEX_BYTES = {"int32": 4, "int64": 8}
SCHEDULE_FIELDS = (
    "used_core_num", "each_num", "extra_task_core", "each_piece", "input_one_piece",
    "input_one_time", "indices_one_time", "updates_one_time", "input_each",
    "indices_each", "input_last", "indices_last", "input_loop", "indices_loop",
    "input_align", "indices_align", "updates_align", "last_indices_loop",
    "last_indices_each", "last_indices_last", "one_time", "mode_flag",
)


def decode_execution_semantics(tiling_key: int) -> tuple[int, int, int, int]:
    """Decode the instantiated CANN-8.1 kernel path into generic semantics."""
    value_kind = tiling_key // 100
    index_kind = (tiling_key // 10) % 10
    update_kind = tiling_key % 10
    value_bytes = {1: 4, 2: 2, 3: 4, 4: 1, 5: 1, 6: 2}[value_kind]
    index_bytes = {1: 4, 2: 8}[index_kind]
    accumulate = int(update_kind == 2)
    promote = int(accumulate and value_kind in {2, 6})
    return value_bytes, index_bytes, accumulate, promote


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + end - 1) / 2.0
        for index in order[start:end]:
            result[index] = average
        start = end
    return result


def spearman(left: list[float], right: list[float]) -> float:
    if len(left) < 2:
        return 0.0
    x = rank(left)
    y = rank(right)
    mx = statistics.fmean(x)
    my = statistics.fmean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    return numerator / (dx * dy) if dx and dy else 0.0


def load_inputs(path: Path) -> tuple[list[str], dict[str, dict], dict[str, dict]]:
    scoring_rows: list[str] = []
    metadata: dict[str, dict] = {}
    measured_evidence: dict[str, dict] = {}
    next_id = 0
    with path.open(encoding="utf-8") as source:
        for line in source:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("record_type") != "formal_latency_candidate":
                continue
            workload = record["workload"]
            observation = record["candidate"]["latency_tiling_observation"]
            tiling = observation["tiling"]
            dtype = workload["dtype"]
            index_dtype = workload["index_dtype"]
            reduce_mode = int(workload["reduce"])
            tiling_key = int(observation["compile_info_vars"]["tiling_key"])
            value_bytes, execution_index_bytes, accumulate, promote = \
                decode_execution_semantics(tiling_key)
            if value_bytes != VALUE_BYTES[dtype] or accumulate != int(reduce_mode != 0):
                raise RuntimeError(
                    f"compiled path disagrees with value/reduce semantics for {workload['workload_id']}"
                )
            opaque_id = f"c{next_id:06d}"
            next_id += 1

            # This is the complete scorer input.  It deliberately contains no
            # operator name, measured latency, measured winner, or record rank.
            fields = [
                opaque_id,
                str(value_bytes),
                str(execution_index_bytes),
                str(accumulate),
                str(promote),
                "20",
            ]
            fields.extend(str(int(tiling[name])) for name in SCHEDULE_FIELDS)
            scoring_rows.append(" ".join(fields))

            metadata[opaque_id] = {
                "workload_id": workload["workload_id"],
                "source_context": observation["source_compile_context_identity"],
                "used_core_num": int(tiling["used_core_num"]),
                "mode": int(tiling["mode_flag"]),
                "semantic_family": "/".join((dtype, index_dtype, str(reduce_mode))),
                "execution_tiling_key": tiling_key,
                "index_normalized_before_kernel": execution_index_bytes != INDEX_BYTES[index_dtype],
            }
            # Latency is isolated from scoring_rows and is read only by the
            # validation join below.
            latency = record["candidate"]["latency_result"]
            measured_evidence[opaque_id] = {
                "median_ms": float(latency["median_ms"]),
                "samples_ms": [float(value) for value in latency["samples_ms"]],
            }
    return scoring_rows, metadata, measured_evidence


def score(binary: Path, rows: list[str]) -> dict[str, dict[str, float]]:
    completed = subprocess.run(
        [str(binary)], input="\n".join(rows) + "\n", text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"hardware scorer failed rc={completed.returncode}: {completed.stderr.strip()}")
    predictions: dict[str, dict[str, float]] = {}
    names = ("cycles", "critical", "hbm", "l2", "shared", "mte2", "vector", "scalar", "mte3", "utilization")
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) < 2 or fields[1] != "valid":
            raise RuntimeError(f"invalid hardware score row: {line}")
        predictions[fields[0]] = {name: float(value) for name, value in zip(names, fields[2:])}
    if len(predictions) != len(rows):
        raise RuntimeError(f"scorer returned {len(predictions)} rows for {len(rows)} inputs")
    return predictions


def validate(metadata: dict[str, dict], evidence: dict[str, dict], predicted: dict[str, dict]) -> dict:
    measured = {key: value["median_ms"] for key, value in evidence.items()}

    def relative_standard_error(item: str) -> float:
        samples = evidence[item]["samples_ms"]
        if len(samples) < 2 or measured[item] <= 0.0:
            return 0.0
        return statistics.stdev(samples) / math.sqrt(len(samples)) / measured[item]

    groups: dict[str, list[str]] = defaultdict(list)
    for opaque_id, item in metadata.items():
        groups[item["workload_id"]].append(opaque_id)

    regrets: list[float] = []
    correlations: list[float] = []
    exact = 0
    within = Counter()
    chosen_cores = Counter()
    measured_best_cores = Counter()
    uncertainty_consistent = 0
    uncertainty_pct: list[float] = []
    separated_groups = 0
    separated_exact = 0
    top3 = 0
    separated_pairs = 0
    separated_pairwise_correct = 0
    family_groups: dict[str, list[dict]] = defaultdict(list)
    for ids in groups.values():
        chosen = min(ids, key=lambda item: predicted[item]["cycles"])
        measured_best = min(ids, key=measured.__getitem__)
        best_latency = measured[measured_best]
        regret = 100.0 * (measured[chosen] / best_latency - 1.0)
        combined_uncertainty = 200.0 * math.sqrt(
            relative_standard_error(chosen) ** 2 +
            relative_standard_error(measured_best) ** 2)
        uncertainty_pct.append(combined_uncertainty)
        uncertainty_consistent += regret <= max(1.0, combined_uncertainty)
        regrets.append(regret)
        exact += chosen == measured_best
        top3 += chosen in sorted(ids, key=measured.__getitem__)[:3]
        for threshold in (1, 3, 5, 10):
            within[threshold] += regret <= threshold
        chosen_cores[metadata[chosen]["used_core_num"]] += 1
        measured_best_cores[metadata[measured_best]["used_core_num"]] += 1
        correlations.append(spearman(
            [predicted[item]["cycles"] for item in ids],
            [measured[item] for item in ids],
        ))
        ordered = sorted(ids, key=measured.__getitem__)
        if len(ordered) > 1:
            best, second = ordered[:2]
            best_gap = 100.0 * (measured[second] / measured[best] - 1.0)
            best_gap_uncertainty = 200.0 * math.sqrt(
                relative_standard_error(best) ** 2 +
                relative_standard_error(second) ** 2)
            if best_gap > max(1.0, best_gap_uncertainty):
                separated_groups += 1
                separated_exact += chosen == best

        # Pairwise ranking is much less brittle than naming one minimum from
        # twenty five-sample medians. Count only pairs whose measured gap is
        # larger than two standard errors of both candidates.
        for left_index, left in enumerate(ids):
            for right in ids[left_index + 1:]:
                absolute_uncertainty = 2.0 * math.sqrt(
                    (relative_standard_error(left) * measured[left]) ** 2 +
                    (relative_standard_error(right) * measured[right]) ** 2)
                measured_delta = measured[left] - measured[right]
                if abs(measured_delta) <= absolute_uncertainty:
                    continue
                separated_pairs += 1
                predicted_delta = predicted[left]["cycles"] - predicted[right]["cycles"]
                separated_pairwise_correct += (predicted_delta < 0.0) == (measured_delta < 0.0)

        family_groups[metadata[ids[0]]["semantic_family"]].append({
            "exact": chosen == measured_best,
            "consistent": regret <= max(1.0, combined_uncertainty),
            "regret": regret,
            "spearman": correlations[-1],
        })

    family_summary = {}
    for family, rows in sorted(family_groups.items()):
        family_summary[family] = {
            "groups": len(rows),
            "top1_exact": sum(row["exact"] for row in rows),
            "top1_uncertainty_consistent": sum(row["consistent"] for row in rows),
            "median_regret_pct": statistics.median(row["regret"] for row in rows),
            "median_spearman": statistics.median(row["spearman"] for row in rows),
        }

    return {
        "schema": "hardware_path_ranking_validation_v1",
        "method": "frozen_hardware_path_equations_no_training",
        "scorer_input_contains_operator_name": False,
        "scorer_input_contains_measured_latency": False,
        "candidate_records": len(metadata),
        "complete_groups": len(groups),
        "candidates_per_group": sorted({len(value) for value in groups.values()}),
        "top1_exact_groups": exact,
        "top3_contains_predicted_groups": top3,
        "measurement_uncertainty": {
            "rule": "two_standard_errors_of_five_device_event_samples",
            "median_pct": statistics.median(uncertainty_pct),
            "predicted_top1_consistent_groups": uncertainty_consistent,
            "statistically_separated_winner_groups": separated_groups,
            "exact_among_separated_groups": separated_exact,
        },
        "top1_within_pct": {str(key): within[key] for key in (1, 3, 5, 10)},
        "top1_regret_pct": {
            "median": statistics.median(regrets),
            "p90": sorted(regrets)[min(len(regrets) - 1, math.ceil(0.9 * len(regrets)) - 1)],
            "maximum": max(regrets),
        },
        "spearman": {
            "mean": statistics.fmean(correlations),
            "median": statistics.median(correlations),
        },
        "statistically_separated_pairwise_order": {
            "pairs": separated_pairs,
            "correct": separated_pairwise_correct,
            "accuracy": separated_pairwise_correct / separated_pairs if separated_pairs else 0.0,
        },
        "semantic_family_breakdown": family_summary,
        "source_mode_candidate_counts": dict(sorted(
            Counter(item["mode"] for item in metadata.values()).items())),
        "index_normalized_before_kernel_candidates": sum(
            item["index_normalized_before_kernel"] for item in metadata.values()),
        "predicted_best_core_counts": dict(sorted(chosen_cores.items())),
        "measured_best_core_counts": dict(sorted(measured_best_cores.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--scorer", type=Path, default=Path("build/indexed_update_cost"))
    args = parser.parse_args()
    rows, metadata, measured = load_inputs(args.evidence)
    if not rows:
        raise SystemExit("no formal_latency_candidate records found")
    predictions = score(args.scorer.resolve(), rows)
    print(json.dumps(validate(metadata, measured, predictions), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
