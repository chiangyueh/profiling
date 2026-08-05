from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


GEOMETRY_FIELDS = ("base_m", "base_n", "base_k")
PIPELINE_FIELDS = (
    "depth_a1",
    "depth_b1",
    "step_m",
    "step_n",
    "step_ka",
    "step_kb",
    "db_l0a",
    "db_l0b",
    "db_l0c",
    "iterate_order",
)
EXECUTION_FIELDS = (
    "used_core_num",
    "single_core_m",
    "single_core_n",
    "single_core_k",
    "l2_m_tile_count",
    "l2_n_tile_count",
    "l2_m_tile_block",
    "l2_n_tile_block",
    "l2_iterate_order",
)


@dataclass(frozen=True)
class Coverage:
    workload_id: str
    template: str
    selected: int
    target_quota: int
    behavior_bins: int
    max_bin_share: float
    geometries: int
    pipelines: int
    executions: int
    active_core_values: int
    bank_distance_max: float
    bank_changed_median: float


def _metrics(row: dict[str, str]) -> dict[str, float]:
    raw = row.get("search_behavior_metrics", "")
    return json.loads(raw) if raw else {}


def _unique_tuples(
    rows: Iterable[dict[str, str]], fields: tuple[str, ...]
) -> int:
    return len({tuple(row[field] for field in fields) for row in rows})


def group_coverage(rows: list[dict[str, str]]) -> Coverage:
    if not rows:
        raise ValueError("candidate distribution group is empty")
    metrics = [_metrics(row) for row in rows]
    bins = Counter(row.get("search_behavior_key", "") for row in rows)
    quota = max(
        int(float(item.get("calibration_target_quota", 0.0)))
        for item in metrics
    )
    return Coverage(
        workload_id=rows[0]["workload_id"],
        template=rows[0]["search_template"],
        selected=len(rows),
        target_quota=quota,
        behavior_bins=len(bins),
        max_bin_share=max(bins.values()) / len(rows),
        geometries=_unique_tuples(rows, GEOMETRY_FIELDS),
        pipelines=_unique_tuples(rows, PIPELINE_FIELDS),
        executions=_unique_tuples(rows, EXECUTION_FIELDS),
        active_core_values=len(
            {item.get("active_cores", 0.0) for item in metrics}
        ),
        bank_distance_max=max(
            item.get("bank_behavior_distance", 0.0) for item in metrics
        ),
        bank_changed_median=statistics.median(
            item.get("bank_changed_fields", 0.0) for item in metrics
        ),
    )


def coverage_violations(coverage: Coverage) -> list[str]:
    violations: list[str] = []
    if coverage.target_quota and coverage.selected != coverage.target_quota:
        violations.append(
            f"quota={coverage.selected}/{coverage.target_quota}"
        )
    if coverage.selected < 8:
        return violations
    minimum_bins = max(4, min(8, coverage.selected // 4))
    if coverage.behavior_bins < minimum_bins:
        violations.append(
            f"behavior_bins={coverage.behavior_bins}<{minimum_bins}"
        )
    if coverage.max_bin_share > 0.55:
        violations.append(
            f"max_bin_share={coverage.max_bin_share:.3f}>0.55"
        )
    if coverage.bank_distance_max < 0.25:
        violations.append(
            f"bank_distance_max={coverage.bank_distance_max:.3f}<0.25"
        )

    if coverage.template == "AL1_FULL_LOAD":
        minimum_geometry = min(12, max(4, coverage.selected // 3))
        minimum_pipeline = min(12, max(4, coverage.selected // 3))
        minimum_execution = min(8, max(4, coverage.selected // 4))
        if coverage.geometries < minimum_geometry:
            violations.append(
                f"geometries={coverage.geometries}<{minimum_geometry}"
            )
        if coverage.pipelines < minimum_pipeline:
            violations.append(
                f"pipelines={coverage.pipelines}<{minimum_pipeline}"
            )
        if coverage.executions < minimum_execution:
            violations.append(
                f"executions={coverage.executions}<{minimum_execution}"
            )
    elif coverage.template == "BL1_FULL_LOAD":
        minimum_structure = min(
            16, max(4, coverage.selected // 3)
        )
        if coverage.geometries < minimum_structure:
            violations.append(
                f"geometries={coverage.geometries}<{minimum_structure}"
            )
        if coverage.pipelines < minimum_structure:
            violations.append(
                f"pipelines={coverage.pipelines}<{minimum_structure}"
            )
        if coverage.executions < 3:
            violations.append(f"executions={coverage.executions}<3")
    elif coverage.template in {
        "BL1_FULL_LOAD_FIXPIPE",
        "BL1_FULL_LOAD_VEC_NZ2ND",
    }:
        minimum_structure = min(
            8, max(4, coverage.selected // 2)
        )
        if coverage.geometries < minimum_structure:
            violations.append(
                f"geometries={coverage.geometries}<{minimum_structure}"
            )
        if coverage.pipelines < minimum_structure:
            violations.append(
                f"pipelines={coverage.pipelines}<{minimum_structure}"
            )
        if coverage.executions < 3:
            violations.append(f"executions={coverage.executions}<3")
    elif coverage.template == "DETERMINISTIC_SPLIT_K":
        if coverage.pipelines < 2:
            violations.append(f"pipelines={coverage.pipelines}<2")
        if coverage.executions < min(8, coverage.selected):
            violations.append(
                f"executions={coverage.executions}<"
                f"{min(8, coverage.selected)}"
            )
        if coverage.active_core_values < 2:
            violations.append(
                f"active_core_values={coverage.active_core_values}<2"
            )
    elif coverage.template == "SINGLE_CORE_SPLIT_K":
        if coverage.pipelines < 4:
            violations.append(f"pipelines={coverage.pipelines}<4")
        if coverage.executions < min(24, coverage.selected // 2):
            violations.append(
                f"executions={coverage.executions}<"
                f"{min(24, coverage.selected // 2)}"
            )
        if coverage.active_core_values < 2:
            violations.append(
                f"active_core_values={coverage.active_core_values}<2"
            )
    return violations


def audit_rows(
    rows: list[dict[str, str]],
) -> tuple[list[Coverage], list[str]]:
    searched = [
        row for row in rows if row.get("candidate_role") == "searched"
    ]
    violations: list[str] = []
    identities = [
        (row["workload_id"], row["tiling_signature"]) for row in searched
    ]
    if len(set(identities)) != len(identities):
        violations.append(
            f"duplicate_signatures={len(identities)-len(set(identities))}"
        )
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in searched:
        grouped[(row["workload_id"], row["search_template"])].append(row)
    coverage = [
        group_coverage(group)
        for _, group in sorted(grouped.items())
    ]
    for item in coverage:
        for violation in coverage_violations(item):
            violations.append(
                f"{item.workload_id}/{item.template}:{violation}"
            )
    return coverage, violations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    args = parser.parse_args()
    with args.candidates.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    coverage, violations = audit_rows(rows)
    by_template: dict[str, list[Coverage]] = defaultdict(list)
    for item in coverage:
        by_template[item.template].append(item)
    for template, groups in sorted(by_template.items()):
        print(
            "DISTRIBUTION_AUDIT "
            f"template={template} "
            f"workloads={len(groups)} "
            f"candidates={sum(item.selected for item in groups)} "
            f"behavior_bins={sum(item.behavior_bins for item in groups)} "
            f"geometry_min={min(item.geometries for item in groups)} "
            f"pipeline_min={min(item.pipelines for item in groups)} "
            f"execution_min={min(item.executions for item in groups)} "
            "max_bin_share="
            f"{max(item.max_bin_share for item in groups):.6g} "
            "bank_changed_median="
            f"{statistics.median(item.bank_changed_median for item in groups):.6g}"
        )
    if violations:
        for violation in violations:
            print(f"DISTRIBUTION_AUDIT_FAILURE {violation}")
        raise SystemExit(1)
    print(
        "DISTRIBUTION_AUDIT_OK "
        f"workload_templates={len(coverage)} "
        "candidate_distribution_is_hardware_behavior_diverse"
    )


if __name__ == "__main__":
    main()
