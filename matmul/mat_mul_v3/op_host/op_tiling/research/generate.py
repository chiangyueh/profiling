#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from callback import CallbackError, exact_roundtrip, invoke
from tiling_search import (
    CandidateEngine,
    GenerationBudget,
    Hardware,
    MeasuredObservation,
    SearchConfig,
    Workload,
)
from tiling_search.domain import KNOWLEDGE_FIELDS, Candidate, Schedule
from tiling_search.behavior import select_behavior_coverage
from tiling_search.feedback import fingerprint, load_feedback


FIELD_COLUMNS = {
    "usedCoreNum": "used_core_num",
    "singleCoreM": "single_core_m",
    "singleCoreN": "single_core_n",
    "singleCoreK": "single_core_k",
    "baseM": "base_m",
    "baseN": "base_n",
    "baseK": "base_k",
    "depthA1": "depth_a1",
    "depthB1": "depth_b1",
    "stepM": "step_m",
    "stepN": "step_n",
    "iterateOrder": "iterate_order",
    "stepKa": "step_ka",
    "stepKb": "step_kb",
    "dbL0A": "db_l0a",
    "dbL0B": "db_l0b",
    "dbL0C": "db_l0c",
    "l2MTileCnt": "l2_m_tile_count",
    "l2NTileCnt": "l2_n_tile_count",
    "l2MTileBlock": "l2_m_tile_block",
    "l2NTileBlock": "l2_n_tile_block",
    "l2IterateOrder": "l2_iterate_order",
    "tilingEnable": "tiling_enable",
}

COLUMNS = [
    "workload_id",
    "m",
    "n",
    "k",
    "dtype",
    "trans_a",
    "trans_b",
    "max_cores",
    "rank",
    "candidate_role",
    "candidate_source",
    "search_template",
    "search_rationale",
    "search_acquisition",
    "search_behavior_key",
    "search_behavior_metrics",
    "tiling_signature",
    *FIELD_COLUMNS.values(),
    "callback_tiling_key",
    "callback_block_dim",
    "callback_workspace_bytes",
    "callback_tiling_sha256",
]


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def load_workloads(path: Path, aic_cores: int) -> list[Workload]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    workloads = []
    for row in rows:
        workloads.append(
            Workload(
                workload_id=row.get("workload_id") or row["id"],
                m=int(row["m"]),
                n=int(row["n"]),
                k=int(row["k"]),
                dtype=row["dtype"].lower(),
                trans_a=truthy(row.get("trans_a")),
                trans_b=truthy(row.get("trans_b")),
                max_cores=int(row.get("max_cores") or aic_cores),
            )
        )
    return workloads


def load_resume_feedback(
    path: Path,
    soc: str,
    aic_cores: int,
) -> tuple[list[MeasuredObservation], set[tuple]]:
    observations: list[MeasuredObservation] = []
    exclusions: set[tuple] = set()
    if not path.is_file():
        return observations, exclusions
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        if (
            row.get("candidate_role") != "searched"
            or row.get("soc") != soc
            or int(row.get("aic") or 0) != aic_cores
        ):
            continue
        try:
            workload = Workload(
                workload_id=row["workload_id"],
                m=int(row["m"]),
                n=int(row["n"]),
                k=int(row["k"]),
                dtype=row["dtype"],
                trans_a=truthy(row.get("trans_a")),
                trans_b=truthy(row.get("trans_b")),
                max_cores=aic_cores,
            )
            schedule = Schedule.from_signature(row["tiling_signature"])
        except (KeyError, ValueError):
            continue
        exclusions.add(fingerprint(workload, schedule))
        if (
            not truthy(row.get("success"))
            or not row.get("official_ms")
            or not row.get("bank_ms")
        ):
            continue
        try:
            candidate_ms = float(row["median_ms"])
            official_ms = float(row["official_ms"])
            bank_ms = float(row["bank_ms"])
        except (KeyError, ValueError):
            continue
        observations.append(
            MeasuredObservation(
                workload=workload,
                schedule=schedule,
                ratio_vs_official=candidate_ms / official_ms,
                ratio_vs_bank=candidate_ms / bank_ms,
                source=row.get("candidate_source", ""),
                record_id=row.get("record_id", path.stem),
                status_vs_official=row.get("status_vs_official", ""),
                status_vs_bank=row.get("status_vs_bank", ""),
            )
        )
    return observations, exclusions


def verify_upstream_contract(source_root: Path) -> None:
    tuning_header = source_root / "op_host/op_tiling/matmul_v3_tuning.h"
    key_header = source_root / "op_kernel/mat_mul_v3_tiling_key.h"
    tuning_text = tuning_header.read_text(encoding="utf-8")
    key_text = key_header.read_text(encoding="utf-8")
    observed = tuple(
        re.findall(
            r"TUNING_TILING_DATA_FIELD_DEF\(uint32_t,\s*([A-Za-z0-9_]+)\)",
            tuning_text,
        )
    )
    if observed != KNOWLEDGE_FIELDS:
        raise RuntimeError(
            "research schema differs from official 8.5 MatMulV3TunnerTiling: "
            f"observed={observed}"
        )
    required_modes = {
        "MAT_MUL_V3_BASE_SPLIT_K": "0",
        "MAT_MUL_V3_SINGLE_CORE_SPLIT_K": "2",
        "MAT_MUL_V3_DETERMINISTIC_SPLIT_K": "3",
        "MAT_MUL_V3_AL1_FULLLOAD": "1",
        "MAT_MUL_V3_BL1_FULLLOAD": "2",
    }
    for name, value in required_modes.items():
        if not re.search(rf"#define\s+{name}\s+{value}\b", key_text):
            raise RuntimeError(
                f"official 8.5 tiling key no longer defines {name}={value}"
            )


def candidate_row(
    workload: Workload,
    rank: int,
    role: str,
    source: str,
    schedule: Schedule,
    callback,
    candidate: Candidate | None = None,
) -> dict[str, str]:
    row = {column: "" for column in COLUMNS}
    row.update(
        {
            "workload_id": workload.workload_id,
            "m": str(workload.m),
            "n": str(workload.n),
            "k": str(workload.k),
            "dtype": workload.dtype,
            "trans_a": str(int(workload.trans_a)),
            "trans_b": str(int(workload.trans_b)),
            "max_cores": str(workload.max_cores),
            "rank": str(rank),
            "candidate_role": role,
            "candidate_source": source,
            "search_template": (
                candidate.template.value if candidate is not None else "OFFICIAL"
            ),
            "search_rationale": (
                candidate.rationale if candidate is not None else "callback control"
            ),
            "search_acquisition": (
                f"{candidate.acquisition:.12g}" if candidate is not None else "0"
            ),
            "search_behavior_key": (
                json.dumps(candidate.behavior_key, separators=(",", ":"))
                if candidate is not None
                else ""
            ),
            "search_behavior_metrics": (
                json.dumps(
                    dict(candidate.metrics),
                    separators=(",", ":"),
                    sort_keys=True,
                )
                if candidate is not None
                else ""
            ),
            "tiling_signature": schedule.signature_text(),
            "callback_tiling_key": str(callback.key),
            "callback_block_dim": str(callback.block_dim),
            "callback_workspace_bytes": str(sum(callback.workspaces)),
            "callback_tiling_sha256": callback.sha256,
        }
    )
    for field, column in FIELD_COLUMNS.items():
        row[column] = str(schedule[field])
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--soc", required=True)
    parser.add_argument("--aic-cores", type=int, required=True)
    parser.add_argument("--l0a-bytes", type=int, required=True)
    parser.add_argument("--l0b-bytes", type=int, required=True)
    parser.add_argument("--l0c-bytes", type=int, required=True)
    parser.add_argument("--l1-bytes", type=int, required=True)
    parser.add_argument("--l2-bytes", type=int, required=True)
    parser.add_argument("--l2-bpc", type=float, default=1.0)
    parser.add_argument("--hbm-bpc", type=float, default=1.0)
    parser.add_argument("--observations", type=Path, action="append", default=[])
    parser.add_argument("--exclusions", type=Path, action="append", default=[])
    parser.add_argument("--resume-feedback", type=Path)
    parser.add_argument("--npu-candidates", type=int, default=40)
    parser.add_argument("--callback-candidates", type=int, default=48)
    parser.add_argument("--behavior-candidates", type=int, default=320)
    parser.add_argument("--workload-limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_upstream_contract(args.source_root)
    from tbe.common.platform import set_current_compile_soc_info

    warnings.filterwarnings("ignore", category=Warning, module="requests")
    set_current_compile_soc_info(args.soc)
    hardware = Hardware(
        aic_cores=args.aic_cores,
        l0a_bytes=args.l0a_bytes,
        l0b_bytes=args.l0b_bytes,
        l0c_bytes=args.l0c_bytes,
        l1_bytes=args.l1_bytes,
        l2_bytes=args.l2_bytes,
        l2_bytes_per_cycle_per_core=args.l2_bpc,
        hbm_bytes_per_cycle_per_core=args.hbm_bpc,
    )
    workloads = load_workloads(args.workloads, args.aic_cores)
    if args.workload_limit > 0:
        workloads = workloads[: args.workload_limit]
    observations, exclusions = load_feedback(
        soc=args.soc,
        aic_cores=args.aic_cores,
        observation_paths=args.observations,
        exclusion_paths=args.exclusions,
    )
    if args.resume_feedback is not None:
        resume_observations, resume_exclusions = load_resume_feedback(
            args.resume_feedback, args.soc, args.aic_cores
        )
        observations.extend(resume_observations)
        exclusions.update(resume_exclusions)
    engine = CandidateEngine(
        config=SearchConfig(
            budget=GenerationBudget(
                raw_attempts=16000,
                legal_candidates=7000,
                behavior_candidates=args.behavior_candidates,
                callback_candidates=args.callback_candidates,
                npu_candidates=args.npu_candidates,
            )
        ),
        observations=observations,
        exclusions=exclusions,
    )

    output_rows: list[dict[str, str]] = []
    all_rows: list[dict[str, str]] = []
    source_counts: Counter[str] = Counter()
    template_counts: Counter[str] = Counter()
    print("SEARCH_FRONTIER_BEGIN")
    for workload in workloads:
        official = invoke(workload)
        bank = exact_roundtrip(workload, official.schedule)
        output_rows.append(
            candidate_row(
                workload,
                0,
                "bank_seed_control",
                "official_callback_control",
                bank.schedule,
                bank,
            )
        )
        result = engine.generate(workload, hardware, local_anchor=None)
        ordered: list[Candidate] = [
            *result.callback_candidates,
            *(
                candidate
                for candidate in result.candidates
                if candidate.schedule.signature()
                not in {
                    selected.schedule.signature()
                    for selected in result.callback_candidates
                }
            ),
        ]
        callback_accepted: list[tuple[Candidate, object]] = []
        callback_rejected = 0
        seen: set[tuple[int, ...]] = set()
        for candidate in ordered:
            signature = candidate.schedule.signature()
            if signature in seen:
                continue
            seen.add(signature)
            try:
                callback = exact_roundtrip(workload, candidate.schedule)
            except Exception as exception:
                callback_rejected += 1
                rejected = candidate_row(
                    workload,
                    0,
                    "callback_rejected",
                    candidate.source,
                    candidate.schedule,
                    official,
                    candidate,
                )
                rejected["search_rationale"] = (
                    f"{candidate.rationale}; callback={str(exception)[:240]}"
                )
                all_rows.append(rejected)
                continue
            callback_accepted.append((candidate, callback))
            if len(callback_accepted) >= args.callback_candidates:
                break
        callback_by_signature = {
            candidate.schedule.signature(): callback
            for candidate, callback in callback_accepted
        }
        selected_candidates = select_behavior_coverage(
            workload,
            (candidate for candidate, _ in callback_accepted),
            observations,
            hardware,
            args.npu_candidates,
        )
        accepted = [
            (
                candidate,
                callback_by_signature[candidate.schedule.signature()],
            )
            for candidate in selected_candidates
        ]
        for rank, (candidate, callback) in enumerate(accepted, 1):
            row = candidate_row(
                workload,
                rank,
                "searched",
                candidate.source,
                candidate.schedule,
                callback,
                candidate,
            )
            output_rows.append(row)
            all_rows.append(row)
            source_counts[candidate.source] += 1
            template_counts[candidate.template.value] += 1
        reports = ";".join(
            (
                f"{report.template.value}:raw={report.raw_generated},"
                f"common={report.common_legal},"
                f"template={report.template_legal},"
                f"emitted={report.emitted}"
            )
            for report in result.reports
        )
        print(
            "SEARCH_WORKLOAD "
            f"{workload.workload_id} shape={workload.m}x{workload.n}x{workload.k} "
            f"dtype={workload.dtype} trans={int(workload.trans_a)}"
            f"{int(workload.trans_b)} selected={len(accepted)} "
            f"callback_rejected={callback_rejected} "
            f"behavior_bins={result.behavior_bins} "
            f"excluded={result.excluded_fingerprints} solvers={reports}"
        )
        if len(accepted) < args.npu_candidates:
            print(
                f"SEARCH_CAPABILITY_GAP {workload.workload_id} "
                f"requested={args.npu_candidates} accepted={len(accepted)}"
            )

    for path, rows in ((args.output, output_rows), (args.all_output, all_rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    searched = sum(row["candidate_role"] == "searched" for row in output_rows)
    print(
        "SEARCH_FRONTIER "
        f"workloads={len(workloads)} candidates={searched} "
        f"sources={dict(sorted(source_counts.items()))} "
        f"templates={dict(sorted(template_counts.items()))} "
        f"paired_controls={len(workloads)} observations={len(observations)}"
    )
    print("SEARCH_FRONTIER_END")


if __name__ == "__main__":
    main()
