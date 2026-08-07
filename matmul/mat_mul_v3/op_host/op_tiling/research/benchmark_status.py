#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from profile import measurement_completed, measurement_key, truthy


def rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--soc", required=True)
    parser.add_argument("--aic", type=int, required=True)
    parser.add_argument("--toolkit", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--emit-records", action="store_true")
    parser.add_argument("--workloads", type=Path)
    parser.add_argument("--expected-per-workload", type=int, default=0)
    parser.add_argument("--frontier-only", action="store_true")
    parser.add_argument("--exclude-run-id", default="")
    args = parser.parse_args()

    candidates = [
        row for row in rows(args.candidates) if row.get("candidate_role") == "searched"
    ]
    workload_ids = (
        {row.get("workload_id") or row.get("id", "") for row in rows(args.workloads)}
        if args.workloads is not None
        else {row.get("workload_id", "") for row in candidates}
    )
    candidate_counts = Counter(row.get("workload_id", "") for row in candidates)
    missing_workloads = sorted(workload_ids - set(candidate_counts))
    short_workloads = {
        workload_id: candidate_counts[workload_id]
        for workload_id in sorted(workload_ids)
        if (
            args.expected_per_workload > 0
            and candidate_counts[workload_id] < args.expected_per_workload
        )
    }
    frontier_complete = not missing_workloads and not short_workloads
    if args.frontier_only:
        payload = {
            "stage": args.stage,
            "workloads": len(workload_ids),
            "candidates": len(candidates),
            "expected_per_workload": args.expected_per_workload,
            "missing_workloads": missing_workloads,
            "short_workloads": short_workloads,
            "complete": frontier_complete,
        }
        print(
            "BENCHMARK_FRONTIER_STATUS "
            + json.dumps(payload, separators=(",", ":"), sort_keys=True)
        )
        raise SystemExit(0 if frontier_complete else 4)
    resume = {
        row.get("record_id", ""): row
        for row in rows(args.resume)
        if row.get("record_id")
    }
    expected = [
        measurement_key(args.soc, args.aic, args.toolkit, candidate)
        for candidate in candidates
    ]
    final = [
        resume[key]
        for key in expected
        if key in resume and measurement_completed(resume[key])
    ]
    outcomes = Counter()
    for row in final:
        if not truthy(row.get("success")):
            outcomes["runtime_rejected"] += 1
        elif truthy(row.get("pair_validated")):
            outcomes["paired_success"] += 1
        else:
            outcomes["unpaired"] += 1
    complete = frontier_complete and bool(expected) and len(expected) == len(final)
    payload = {
        "stage": args.stage,
        "expected": len(expected),
        "terminal": len(final),
        "pending": len(expected) - len(final),
        "outcomes": dict(sorted(outcomes.items())),
        "frontier_complete": frontier_complete,
        "missing_workloads": missing_workloads,
        "short_workloads": short_workloads,
        "complete": complete,
    }
    print(
        "BENCHMARK_STAGE_STATUS "
        + json.dumps(payload, separators=(",", ":"), sort_keys=True)
    )
    if args.emit_records:
        print("BENCHMARK_RECORDS_BEGIN")
        emitted_records = [
            row for row in resume.values() if row.get("run_id") != args.exclude_run_id
        ]
        for row in sorted(
            emitted_records,
            key=lambda item: (
                item.get("workload_id", ""),
                item.get("benchmark_stage", ""),
                int(item.get("rank") or 0),
                item.get("record_id", ""),
            ),
        ):
            print(
                "BENCHMARK_RECORD "
                + json.dumps(
                    row,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
        print(
            "BENCHMARK_RECORDS_END "
            + json.dumps(
                {
                    "records": len(emitted_records),
                    "current_run_records_already_streamed": (
                        len(resume) - len(emitted_records)
                    ),
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    raise SystemExit(0 if payload["complete"] else 3)


if __name__ == "__main__":
    main()
