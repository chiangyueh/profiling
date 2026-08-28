#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import profile_official_tilings as profiler
import refine_matmul_v3_candidates as contract


SCHEMA = "matmul_controlled_frontier_measurement_v1"
EXPECTED_BLOCKS = {
    "l2": 1600,
    "concurrency": 1200,
    "buffer": 1200,
    "splitk": 1000,
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def numeric(value: str) -> Any:
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


class RotatingJsonl:
    def __init__(self, directory: Path, max_bytes: int) -> None:
        self.directory = directory
        self.max_bytes = max_bytes
        directory.mkdir(parents=True, exist_ok=True)
        if any(directory.iterdir()):
            raise RuntimeError(f"formal log directory is not empty: {directory}")
        self.index = 0
        self.size = 0
        self.stream = None
        self._rotate()

    def _rotate(self) -> None:
        if self.stream is not None:
            self.stream.close()
        self.index += 1
        self.size = 0
        self.stream = (self.directory / f"{self.index}.log").open(
            "wb"
        )

    def write(self, record: dict[str, Any]) -> None:
        data = (
            json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        if self.size and self.size + len(data) > self.max_bytes:
            self._rotate()
        assert self.stream is not None
        self.stream.write(data)
        self.size += len(data)

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()
            self.stream = None


def candidate_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["workload_id"], row["candidate_role"], row["rank"]


def profile_success(row: dict[str, str]) -> bool:
    return (
        truthy(row.get("success"))
        and truthy(row.get("preflight_passed"))
        and row.get("preflight_mode") == "numeric_signed_axes_full_v3"
        and row.get("median_ms", "") != ""
    )


def model_breakdown(row: dict[str, str]) -> dict[str, Any]:
    value = row.get("search_model_breakdown", "")
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {"raw": value}
    return parsed if isinstance(parsed, dict) else {"raw": value}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--official-profile", type=Path, required=True)
    parser.add_argument("--official-samples", type=Path, required=True)
    parser.add_argument("--log-directory", type=Path, required=True)
    parser.add_argument("--soc", required=True)
    parser.add_argument("--aic-cores", type=int, required=True)
    parser.add_argument("--max-log-bytes", type=int, default=50 * 1024 * 1024)
    args = parser.parse_args()

    workloads = read_rows(args.workloads)
    candidates = read_rows(args.candidates)
    profiles = read_rows(args.profile)
    samples = read_rows(args.samples)
    official_profiles = read_rows(args.official_profile)
    official_samples = read_rows(args.official_samples)
    workload_by_id = {row["workload_id"]: row for row in workloads}
    candidate_by_key = {candidate_key(row): row for row in candidates}
    profile_by_workload: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in profiles:
        if row.get("candidate_role") != "api_auto_baseline":
            profile_by_workload[row["workload_id"]].append(row)
    sample_by_key: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in samples:
        sample_by_key[candidate_key(row)].append(float(row["latency_ms"]))
    official_sample_by_id: dict[str, list[float]] = defaultdict(list)
    for row in official_samples:
        official_sample_by_id[row["workload_id"]].append(float(row["latency_ms"]))
    official_by_id = {row["workload_id"]: row for row in official_profiles}

    writer = RotatingJsonl(args.log_directory, args.max_log_bytes)
    block_counts: Counter[str] = Counter()
    formal_count = 0
    try:
        writer.write(
            {
                "schema": SCHEMA,
                "record_type": "campaign_begin",
                "soc": args.soc,
                "aic_cores": args.aic_cores,
                "semantic_shapes": len(workloads),
                "formal_latency_target": sum(EXPECTED_BLOCKS.values()),
                "minimum_successful_distinct_tilings_per_shape": 20,
                "numeric_validation": "every_C_element_signed_axis_oracle",
                "timing": "NPU_device_event",
                "historical_latency_or_cce_read": False,
            }
        )
        for workload_index, workload in enumerate(workloads, 1):
            workload_id = workload["workload_id"]
            official = official_by_id.get(workload_id)
            if official is None or not profile_success(official):
                raise RuntimeError(f"{workload_id}: valid official baseline missing")
            baseline_samples = official_sample_by_id[workload_id]
            if not baseline_samples:
                raise RuntimeError(f"{workload_id}: official raw samples missing")
            writer.write(
                {
                    "schema": SCHEMA,
                    "record_type": "official_baseline",
                    "workload_index": workload_index,
                    "workload": workload,
                    "median_ms": float(official["median_ms"]),
                    "stddev_ms": float(official["stddev_ms"]),
                    "samples_ms": baseline_samples,
                    "preflight_mode": official["preflight_mode"],
                }
            )

            for rejected in profile_by_workload[workload_id]:
                if (
                    rejected.get("candidate_role") not in
                    {"bank_seed_control", "searched"}
                    or profile_success(rejected)
                ):
                    continue
                key = candidate_key(rejected)
                candidate = candidate_by_key.get(key, {})
                writer.write(
                    {
                        "schema": SCHEMA,
                        "record_type": "candidate_rejected",
                        "workload_index": workload_index,
                        "workload": workload,
                        "candidate_role": rejected.get("candidate_role", ""),
                        "candidate_rank": numeric(rejected.get("rank", "")),
                        "pair_id": candidate.get("pair_id", ""),
                        "changed_factor": candidate.get("changed_factor", ""),
                        "callback_tiling_sha256": candidate.get(
                            "callback_tiling_sha256", ""
                        ),
                        "failure": rejected.get("error", ""),
                        "preflight_mode": rejected.get("preflight_mode", ""),
                        "counted_as_formal_latency": False,
                    }
                )

            rows = sorted(
                (row for row in profile_by_workload[workload_id] if profile_success(row)),
                key=lambda row: (
                    0 if row["candidate_role"] == "bank_seed_control" else 1,
                    int(row["rank"]),
                ),
            )
            distinct: list[dict[str, str]] = []
            seen: set[str] = set()
            for row in rows:
                signature = row.get("tiling_signature", "")
                if not signature or signature in seen:
                    continue
                key = candidate_key(row)
                if key not in candidate_by_key:
                    continue
                if not sample_by_key[key]:
                    raise RuntimeError(f"{workload_id} {key[1]} rank={key[2]}: raw samples missing")
                seen.add(signature)
                distinct.append(row)
                if len(distinct) == 20:
                    break
            if len(distinct) != 20:
                raise RuntimeError(
                    f"{workload_id}: {len(distinct)} distinct validated tilings, expected 20"
                )

            block = workload["experiment_block"]
            for group_rank, profile in enumerate(distinct, 1):
                key = candidate_key(profile)
                candidate = candidate_by_key[key]
                knowledge = profiler.make_knowledge(candidate)
                record = {
                    "schema": SCHEMA,
                    "record_type": "formal_latency_candidate",
                    "global_index": formal_count + 1,
                    "workload_index": workload_index,
                    "group_rank": group_rank,
                    "workload": workload,
                    "candidate_role": profile["candidate_role"],
                    "candidate_rank": int(profile["rank"]),
                    "experiment_block": block,
                    "pair_id": candidate.get("pair_id", ""),
                    "changed_factor": candidate.get("changed_factor", ""),
                    "pair_variant": candidate.get("pair_variant", ""),
                    "controlled_sequence": numeric(candidate.get("controlled_sequence", "")),
                    "kernel_family": candidate.get("callback_kernel_family", ""),
                    "kernel_variant": candidate.get("callback_kernel_variant", ""),
                    "callback_tiling_key": numeric(candidate.get("callback_tiling_key", "")),
                    "callback_tiling_sha256": candidate.get("callback_tiling_sha256", ""),
                    "callback_fixed_point": True,
                    "tiling": {field: knowledge[field] for field in contract.KNOWLEDGE_FIELDS},
                    "model": {
                        "cycles": numeric(candidate.get("search_model_cycles", "")),
                        "ratio_vs_bank_seed": numeric(
                            candidate.get("search_model_ratio_vs_bank_seed", "")
                        ),
                        "breakdown": model_breakdown(candidate),
                    },
                    "measurement": {
                        "kind": "NPU_device_event",
                        "median_ms": float(profile["median_ms"]),
                        "stddev_ms": float(profile["stddev_ms"]),
                        "min_ms": float(profile["min_ms"]),
                        "max_ms": float(profile["max_ms"]),
                        "samples_ms": sample_by_key[key],
                    },
                    "correctness": {
                        "passed": True,
                        "preflight_mode": profile["preflight_mode"],
                        "compared_elements": int(workload["m"]) * int(workload["n"]),
                    },
                }
                writer.write(record)
                formal_count += 1
                block_counts[block] += 1
            writer.write(
                {
                    "schema": SCHEMA,
                    "record_type": "workload_group_result",
                    "workload_index": workload_index,
                    "workload_id": workload_id,
                    "experiment_block": block,
                    "status": "admitted",
                    "valid_latency_count": 20,
                    "distinct_tiling_count": 20,
                }
            )

        if formal_count != sum(EXPECTED_BLOCKS.values()):
            raise RuntimeError(f"formal count {formal_count}, expected 5000")
        if dict(block_counts) != EXPECTED_BLOCKS:
            raise RuntimeError(
                f"formal block counts {dict(block_counts)}, expected {EXPECTED_BLOCKS}"
            )
        writer.write(
            {
                "schema": SCHEMA,
                "record_type": "campaign_summary",
                "status": "complete",
                "semantic_shapes": len(workloads),
                "formal_latency_count": formal_count,
                "formal_latency_count_by_block": dict(block_counts),
                "all_groups_have_20_distinct_correct_tilings": True,
                "log_file_count": writer.index,
                "log_rotation_max_bytes": args.max_log_bytes,
            }
        )
    finally:
        writer.close()

    for path in sorted(args.log_directory.glob("*.log")):
        if path.stat().st_size > args.max_log_bytes:
            raise RuntimeError(f"log exceeds rotation bound: {path}")
    print(
        f"MATMUL_CONTROLLED_FRONTIER_COMPLETE records={formal_count} "
        f"shapes={len(workloads)} logs={args.log_directory}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
