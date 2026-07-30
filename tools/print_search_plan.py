#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def compact_counts(counts: Counter[str]) -> str:
    return ",".join(
        f"{name or 'unknown'}:{count}" for name, count in sorted(counts.items())
    ) or "none"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the bounded search frontier without dumping candidates."
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    args = parser.parse_args()

    workloads = read_csv(args.workloads)
    candidates = [
        row
        for row in read_csv(args.candidates)
        if row.get("candidate_role") == "searched"
    ]
    by_workload: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        by_workload[row.get("workload_id", "")].append(row)

    source_counts = Counter(
        row.get("search_candidate_source", "") for row in candidates
    )
    template_counts = Counter(row.get("search_template", "") for row in candidates)
    per_workload = [
        len(by_workload.get(row.get("id", ""), [])) for row in workloads
    ]
    print("SEARCH_FRONTIER_BEGIN")
    print(
        "SEARCH_FRONTIER "
        f"workloads={len(workloads)} candidates={len(candidates)} "
        f"per_workload_min={min(per_workload, default=0)} "
        f"per_workload_median={statistics.median(per_workload) if per_workload else 0:g} "
        f"per_workload_max={max(per_workload, default=0)} "
        f"sources={compact_counts(source_counts)} "
        f"templates={compact_counts(template_counts)}"
    )
    for workload in workloads:
        workload_id = workload.get("id", "")
        rows = by_workload.get(workload_id, [])
        sources = Counter(row.get("search_candidate_source", "") for row in rows)
        templates = Counter(row.get("search_template", "") for row in rows)
        print(
            f"SEARCH_WORKLOAD {workload_id} "
            f"shape={workload.get('m')}x{workload.get('n')}x{workload.get('k')} "
            f"dtype={workload.get('dtype')} "
            f"trans={workload.get('trans_a')}{workload.get('trans_b')} "
            f"candidates={len(rows)} "
            f"sources={compact_counts(sources)} "
            f"templates={compact_counts(templates)}"
        )
    print("SEARCH_FRONTIER_END")


if __name__ == "__main__":
    main()
