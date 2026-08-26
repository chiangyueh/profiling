#!/usr/bin/env python3
"""Validate a generic indexed-read hardware path against Gather evidence.

The scorer sees tensor geometry and bounded hardware inputs only.  Device
latency is kept in a separate map and joined after all predictions return.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from validate_indexed_update_ranking import score, validate


VALUE_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4, "int32": 4, "int8": 1, "uint8": 1}
INDEX_BYTES = {"int32": 4, "int64": 8}


def route_breakdown(metadata: dict[str, dict], evidence: dict[str, dict],
                    predictions: dict[str, dict]) -> dict[str, dict]:
    groups: dict[str, list[str]] = defaultdict(list)
    for key, item in metadata.items():
        groups[item["workload_id"]].append(key)

    def standard_error(key: str) -> float:
        samples = evidence[key]["samples_ms"]
        return statistics.stdev(samples) / math.sqrt(len(samples))

    totals: dict[str, dict[str, int]] = defaultdict(lambda: {
        "groups": 0,
        "separated_pairs": 0,
        "correct_pairs": 0,
        "separated_winners": 0,
        "exact_winners": 0,
    })
    for ids in groups.values():
        route = "transpose" if metadata[ids[0]]["mode"] == 1 else "last_dimension"
        row = totals[route]
        row["groups"] += 1
        measured = lambda key: evidence[key]["median_ms"]
        ordered = sorted(ids, key=measured)
        best, second = ordered[:2]
        winner_uncertainty = 2.0 * math.sqrt(
            standard_error(best) ** 2 + standard_error(second) ** 2)
        if measured(second) - measured(best) > winner_uncertainty:
            row["separated_winners"] += 1
            chosen = min(ids, key=lambda key: predictions[key]["cycles"])
            row["exact_winners"] += chosen == best
        for offset, left in enumerate(ids):
            for right in ids[offset + 1:]:
                uncertainty = 2.0 * math.sqrt(
                    standard_error(left) ** 2 + standard_error(right) ** 2)
                measured_delta = measured(left) - measured(right)
                if abs(measured_delta) <= uncertainty:
                    continue
                row["separated_pairs"] += 1
                predicted_delta = predictions[left]["cycles"] - predictions[right]["cycles"]
                row["correct_pairs"] += (predicted_delta < 0) == (measured_delta < 0)
    for row in totals.values():
        row["pairwise_accuracy"] = (
            row["correct_pairs"] / row["separated_pairs"]
            if row["separated_pairs"] else 0.0
        )
    return dict(sorted(totals.items()))


def load_inputs(path: Path) -> tuple[list[str], dict[str, dict], dict[str, dict]]:
    scoring_rows: list[str] = []
    metadata: dict[str, dict] = {}
    evidence: dict[str, dict] = {}
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
            candidate = record["candidate"]
            observation = candidate["latency_tiling_observation"]
            compile_vars = observation["compile_info_vars"]
            source_shape = [int(value) for value in workload["shape"]]
            index_shape = [int(value) for value in workload["index_shape"]]
            if not source_shape or len(source_shape) != len(index_shape):
                raise RuntimeError(f"invalid indexed-read geometry: {workload['workload_id']}")
            axis = int(workload["axis"]) % len(source_shape)
            opaque_id = f"r{next_id:06d}"
            next_id += 1

            # Complete scorer input: no operator name, workload id, measured
            # latency, measured rank, or raw identity hash is present.
            fields = [
                opaque_id,
                str(VALUE_BYTES[workload["dtype"]]),
                str(INDEX_BYTES[workload["index_dtype"]]),
                "20",
                "196352",
                str(axis),
                str(int(observation["aiv_core_cap"])),
                str(int(observation["ub_cap_divisor"])),
                str(len(source_shape)),
            ]
            fields.extend(str(value) for value in source_shape)
            fields.extend(str(value) for value in index_shape)
            scoring_rows.append(" ".join(fields))

            tiling_key = int(compile_vars["tiling_key"])
            metadata[opaque_id] = {
                "workload_id": workload["workload_id"],
                "source_context": observation["source_compile_context_identity"],
                "used_core_num": int(compile_vars["block_dim"]),
                "mode": tiling_key,
                "semantic_family": "/".join((
                    workload["dtype"], workload["index_dtype"],
                    "last_dimension" if tiling_key == 2 else "transpose",
                )),
                "execution_tiling_key": tiling_key,
                "index_normalized_before_kernel": False,
            }
            latency = candidate["latency_result"]
            evidence[opaque_id] = {
                "median_ms": float(latency["median_ms"]),
                "samples_ms": [float(value) for value in latency["samples_ms"]],
            }
    return scoring_rows, metadata, evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--scorer", type=Path, default=Path("build/indexed_read_cost"))
    args = parser.parse_args()
    rows, metadata, evidence = load_inputs(args.evidence)
    if not rows:
        raise SystemExit("no formal_latency_candidate records found")
    predictions = score(args.scorer.resolve(), rows)
    block_dim_matches = sum(
        round(predictions[key]["utilization"] * 20) == item["used_core_num"]
        for key, item in metadata.items()
    )
    if block_dim_matches != len(metadata):
        raise RuntimeError(
            f"source schedule reconstruction mismatched block_dim for "
            f"{len(metadata) - block_dim_matches} candidates"
        )
    result = validate(metadata, evidence, predictions)
    result["evidence_schema"] = "gather_elements_cann81_native_measurement_v1"
    result["lowering"] = "tensor_geometry_to_shared_hardware_path_ir"
    result["source_block_dim_reconstructed_candidates"] = block_dim_matches
    result["route_breakdown"] = route_breakdown(metadata, evidence, predictions)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
