#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from profile import (
    KNOWLEDGE_COLUMNS,
    measurement_completed,
    measurement_key,
    truthy,
)


GEOMETRY_COLUMNS = (
    "single_core_m",
    "single_core_n",
    "single_core_k",
    "base_m",
    "base_n",
    "base_k",
)
PIPELINE_COLUMNS = (
    "depth_a1",
    "depth_b1",
    "step_m",
    "step_n",
    "iterate_order",
    "step_ka",
    "step_kb",
    "db_l0a",
    "db_l0b",
    "db_l0c",
)
EXECUTION_COLUMNS = (
    "used_core_num",
    "single_core_m",
    "single_core_n",
    "single_core_k",
    "l2_m_tile_count",
    "l2_n_tile_count",
    "l2_m_tile_block",
    "l2_n_tile_block",
    "l2_iterate_order",
    "tiling_enable",
)
FEEDBACK_SOURCES = {
    "feedback_winner_transfer",
    "feedback_winner_mutation",
    "feedback_promising_mutation",
    "feedback_regression_counterfactual",
}


def rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _tuple(row: dict[str, str], columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(row.get(column, "") for column in columns)


def frontier_audit(
    candidates: list[dict[str, str]],
    workload_ids: set[str],
    expected_per_workload: int,
    *,
    prior_candidates: list[dict[str, str]] = (),
    resume_rows: list[dict[str, str]] = (),
) -> tuple[dict[str, object], bool]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        grouped[row.get("workload_id", "")].append(row)
    prior_grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in prior_candidates:
        prior_grouped[row.get("workload_id", "")].append(row)

    prior_fingerprints = {
        (
            row.get("workload_id", ""),
            row.get("tiling_signature", ""),
        )
        for row in prior_candidates
    }
    feedback_evidence = {
        row.get("workload_id", "")
        for row in resume_rows
        if (
            (
                row.get("workload_id", ""),
                row.get("tiling_signature", ""),
            )
            in prior_fingerprints
            and truthy(row.get("success"))
            and truthy(row.get("pair_validated"))
            and (
                (
                    row.get("status_vs_official") == "improved"
                    and row.get("status_vs_bank") == "improved"
                )
                or row.get("status_vs_official") == "regressed"
                or row.get("status_vs_bank") == "regressed"
            )
        )
    }
    failures: dict[str, list[str]] = {}
    workload_stats: dict[str, dict[str, object]] = {}
    all_templates: Counter[str] = Counter()
    all_sources: Counter[str] = Counter()
    knowledge_columns = tuple(KNOWLEDGE_COLUMNS.values())

    for workload_id in sorted(workload_ids):
        family = grouped.get(workload_id, [])
        count = len(family)
        signatures = {row.get("tiling_signature", "") for row in family}
        behavior_bins = {
            row.get("search_behavior_key", "")
            for row in family
            if row.get("search_behavior_key")
        }
        geometries = {_tuple(row, GEOMETRY_COLUMNS) for row in family}
        pipelines = {_tuple(row, PIPELINE_COLUMNS) for row in family}
        executions = {_tuple(row, EXECUTION_COLUMNS) for row in family}
        templates = Counter(row.get("search_template", "") for row in family)
        sources = Counter(row.get("candidate_source", "") for row in family)
        try:
            available_template_counts = Counter(
                json.loads(
                    family[0].get("host_callback_template_counts", "{}")
                )
            ) if family else Counter()
        except (TypeError, ValueError):
            available_template_counts = Counter()
        available_templates = {
            template
            for template, available in available_template_counts.items()
            if int(available) > 0
        }
        varied_fields = sum(
            len({row.get(column, "") for row in family}) > 1
            for column in knowledge_columns
        )
        all_templates.update(templates)
        all_sources.update(sources)

        issues: list[str] = []
        if expected_per_workload > 0 and count != expected_per_workload:
            issues.append(
                f"candidate_count={count},expected={expected_per_workload}"
            )
        if len(signatures) != count:
            issues.append(
                f"unique_signatures={len(signatures)},candidates={count}"
            )
        behavior_floor = min(count, max(2, count // 8))
        structure_floor = min(count, max(2, count // 16))
        if len(behavior_bins) < behavior_floor:
            issues.append(
                f"behavior_bins={len(behavior_bins)},minimum={behavior_floor}"
            )
        for label, values in (
            ("geometries", geometries),
            ("pipelines", pipelines),
            ("executions", executions),
        ):
            if len(values) < structure_floor:
                issues.append(
                    f"{label}={len(values)},minimum={structure_floor}"
                )
        varied_floor = min(len(knowledge_columns), 8 if count >= 8 else 4)
        if varied_fields < varied_floor:
            issues.append(
                f"varied_23_fields={varied_fields},minimum={varied_floor}"
            )
        template_floor = min(len(available_templates), count)
        if len(templates) < template_floor:
            issues.append(
                f"templates={len(templates)},available={template_floor}"
            )
        if (
            len(available_templates) <= count
            and not available_templates.issubset(templates)
        ):
            missing_templates = sorted(
                available_templates.difference(templates)
            )
            issues.append(
                "missing_callback_templates=" + ",".join(missing_templates)
            )

        prior = prior_grouped.get(workload_id, [])
        prior_signatures = {
            row.get("tiling_signature", "") for row in prior
        }
        prior_bins = {
            row.get("search_behavior_key", "")
            for row in prior
            if row.get("search_behavior_key")
        }
        overlap = len(signatures & prior_signatures)
        novel_bins = len(behavior_bins - prior_bins)
        feedback_selected = sum(
            count
            for source, count in sources.items()
            if source in FEEDBACK_SOURCES
        )
        if prior:
            if overlap:
                issues.append(f"prior_signature_overlap={overlap},maximum=0")
            novel_floor = min(count, max(2, count // 16))
            if novel_bins < novel_floor:
                issues.append(
                    f"new_behavior_bins={novel_bins},minimum={novel_floor}"
                )
            if workload_id in feedback_evidence and feedback_selected == 0:
                issues.append(
                    "feedback_candidates=0,measured_winner_or_regression=1"
                )

        workload_stats[workload_id] = {
            "candidates": count,
            "unique_signatures": len(signatures),
            "behavior_bins": len(behavior_bins),
            "geometries": len(geometries),
            "pipelines": len(pipelines),
            "executions": len(executions),
            "varied_23_fields": varied_fields,
            "templates": dict(sorted(templates.items())),
            "callback_templates": dict(
                sorted(available_template_counts.items())
            ),
            "sources": dict(sorted(sources.items())),
            "prior_signature_overlap": overlap,
            "new_behavior_bins": novel_bins,
            "feedback_candidates": feedback_selected,
        }
        if issues:
            failures[workload_id] = issues

    payload: dict[str, object] = {
        "workloads": len(workload_ids),
        "candidates": len(candidates),
        "templates": dict(sorted(all_templates.items())),
        "sources": dict(sorted(all_sources.items())),
        "failed_workloads": len(failures),
        "failures": failures,
        "workload_stats": workload_stats,
    }
    return payload, not failures


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
    parser.add_argument("--prior-candidates", type=Path)
    args = parser.parse_args()

    candidates = [
        row for row in rows(args.candidates) if row.get("candidate_role") == "searched"
    ]
    prior_candidates = [
        row
        for row in rows(args.prior_candidates)
        if row.get("candidate_role") == "searched"
    ] if args.prior_candidates is not None else []
    resume_rows = rows(args.resume)
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
    audit, audit_complete = frontier_audit(
        candidates,
        workload_ids,
        args.expected_per_workload,
        prior_candidates=prior_candidates,
        resume_rows=resume_rows,
    )
    frontier_complete = (
        not missing_workloads
        and not short_workloads
        and audit_complete
    )
    if args.frontier_only:
        payload = {
            "stage": args.stage,
            "workloads": len(workload_ids),
            "candidates": len(candidates),
            "expected_per_workload": args.expected_per_workload,
            "missing_workloads": missing_workloads,
            "short_workloads": short_workloads,
            "structural_audit": audit,
            "complete": frontier_complete,
        }
        print(
            "BENCHMARK_FRONTIER_STATUS "
            + json.dumps(payload, separators=(",", ":"), sort_keys=True)
        )
        raise SystemExit(0 if frontier_complete else 4)
    resume = {
        row.get("record_id", ""): row
        for row in resume_rows
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
        "structural_audit_complete": audit_complete,
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
