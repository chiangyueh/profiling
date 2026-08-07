#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import warnings
from collections import Counter
from dataclasses import replace
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from callback import CallbackError, exact_roundtrip, invoke
from tiling_search import (
    adaptive_calibration_evidence,
    BankRelativeEffectModel,
    BankRelativeSafetyModel,
    CandidateEngine,
    DIRECT_BASE_AUDIT_BANK_WORKLOADS,
    DIRECT_BASE_AUDIT_PAIRED_RECORDS,
    DIRECT_BASE_AUDIT_UNIQUE_RECORDS,
    DIRECT_BASE_AUDIT_WINNER_WORKLOADS,
    DIRECT_BASE_AUDIT_WORKLOADS,
    DIRECT_BASE_L2_RESIDENT_RATIO,
    DIRECT_BASE_RULE_VERSION,
    DIRECT_RULE_AUDIT_RECORDS,
    DIRECT_RULE_AUDIT_UNIQUE_RECORDS,
    DIRECT_RULE_AUDIT_WORKLOADS,
    DIRECT_RULE_TRUSTED_WINNER_EXECUTION_EQUIVALENT,
    DIRECT_RULE_TRUSTED_WINNER_STRUCTURAL_NEAR,
    DIRECT_RULE_TRUSTED_WINNER_TEMPLATE_MATCHES,
    DIRECT_RULE_TRUSTED_WINNER_WORKLOADS,
    GenerationBudget,
    Hardware,
    MeasuredObservation,
    OneShotDecision,
    SearchConfig,
    Workload,
    direct_rule_candidate,
    plan_template_race,
    select_adaptive_calibration_candidates,
    select_calibration_candidates,
    select_one_shot_candidate,
    validate_bank_relative_selector,
)
from tiling_search.contracts import template_of
from tiling_search.domain import (
    KNOWLEDGE_FIELDS,
    Candidate,
    Schedule,
    Template,
)
from tiling_search.calibration_workloads import decode_template_quotas
from tiling_search.behavior import (
    FeedbackCostModel,
    behavior_distance,
    behavior_key,
    behavior_vector,
    select_behavior_coverage,
    validate_feedback_model,
)
from tiling_search.feedback import (
    fingerprint,
    load_feedback,
    measurement_status_consistent,
)


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
    "bank_tiling_signature",
    "deployment_strategy",
    "host_history_load_ms",
    "host_model_setup_ms",
    "host_generation_ms",
    "host_callback_ms",
    "host_selection_ms",
    "host_tiling_total_ms",
    "host_generated_candidates",
    "host_callback_candidates",
    *FIELD_COLUMNS.values(),
    "callback_tiling_key",
    "callback_block_dim",
    "callback_workspace_bytes",
    "callback_tiling_sha256",
]

STRICT_NUMERIC_PREFLIGHT_MODES = {
    "numeric_ones_full_v2",
    "numeric_signed_axes_full_v3",
}


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def optional_schedule(value: str | None) -> Schedule | None:
    if not value:
        return None
    try:
        return Schedule.from_signature(value)
    except ValueError:
        return None


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
                max_cores=min(
                    int(row.get("max_cores") or aic_cores),
                    aic_cores,
                ),
            )
        )
    return workloads


def load_calibration_quotas(
    path: Path,
) -> dict[str, dict[Template, int]]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    return {
        row.get("workload_id") or row["id"]: decode_template_quotas(
            row.get("template_quotas", "")
        )
        for row in rows
        if row.get("template_quotas")
    }


def load_strict_paired_template_counts(
    paths: list[Path],
    soc: str,
    aic_cores: int,
    toolkit: str | None,
    workloads: dict[str, Workload],
) -> Counter[tuple[str, Template]]:
    counts: Counter[tuple[str, Template]] = Counter()
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for path in paths:
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as source:
            rows = csv.DictReader(source)
            for row in rows:
                if (
                    row.get("candidate_role") != "searched"
                    or row.get("soc") != soc
                    or int(row.get("aic") or 0) != aic_cores
                    or (
                        toolkit is not None
                        and row.get("toolkit") != toolkit
                    )
                    or not truthy(row.get("success"))
                    or not truthy(row.get("pair_validated"))
                    or row.get("preflight_mode")
                    not in STRICT_NUMERIC_PREFLIGHT_MODES
                    or not row.get("official_ms")
                    or not row.get("bank_ms")
                ):
                    continue
                schedule = optional_schedule(
                    row.get("tiling_signature")
                )
                workload_id = row.get("workload_id", "")
                workload = workloads.get(workload_id)
                try:
                    row_shape = (
                        int(row.get("m") or -1),
                        int(row.get("n") or -1),
                        int(row.get("k") or -1),
                    )
                except ValueError:
                    continue
                if (
                    schedule is None
                    or workload is None
                    or row_shape != (workload.m, workload.n, workload.k)
                    or row.get("dtype") != workload.dtype
                    or truthy(row.get("trans_a")) != workload.trans_a
                    or truthy(row.get("trans_b")) != workload.trans_b
                ):
                    continue
                identity = (workload_id, schedule.signature())
                if identity in seen:
                    continue
                seen.add(identity)
                counts[(workload_id, template_of(schedule))] += 1
    return counts


def load_retryable_unpaired_fingerprints(
    paths: list[Path],
    workload_ids: set[str],
    soc: str,
    aic_cores: int,
    toolkit: str | None,
) -> set[tuple]:
    """Return executable calibration fingerprints that still need pairing."""

    unpaired: set[tuple] = set()
    completed: set[tuple] = set()
    for path in paths:
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as source:
            rows = csv.DictReader(source)
            for row in rows:
                if (
                    row.get("candidate_role") != "searched"
                    or row.get("workload_id") not in workload_ids
                    or row.get("soc") != soc
                    or int(row.get("aic") or 0) != aic_cores
                    or (
                        toolkit is not None
                        and row.get("toolkit") != toolkit
                    )
                    or not truthy(row.get("success"))
                    or row.get("preflight_mode")
                    not in STRICT_NUMERIC_PREFLIGHT_MODES
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
                    schedule = Schedule.from_signature(
                        row["tiling_signature"]
                    )
                except (KeyError, ValueError):
                    continue
                item = fingerprint(workload, schedule)
                if (
                    truthy(row.get("pair_validated"))
                    and row.get("official_ms")
                    and row.get("bank_ms")
                ):
                    completed.add(item)
                else:
                    unpaired.add(item)
    return unpaired - completed


def merge_candidate_rows(
    previous: list[dict[str, str]],
    current: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Append newly generated schedules without changing exact identities."""
    workload_order = [
        row["workload_id"]
        for row in current
        if row.get("candidate_role") == "bank_seed_control"
    ]
    controls = {
        row["workload_id"]: row
        for row in current
        if row.get("candidate_role") == "bank_seed_control"
    }
    merged: list[dict[str, str]] = []
    for workload_id in workload_order:
        merged.append(dict(controls[workload_id]))
        seen: set[str] = set()
        rank = 1
        for row in [*previous, *current]:
            if (
                row.get("workload_id") != workload_id
                or row.get("candidate_role") != "searched"
            ):
                continue
            signature = row.get("tiling_signature", "")
            if not signature or signature in seen:
                continue
            seen.add(signature)
            candidate = {column: row.get(column, "") for column in COLUMNS}
            candidate["rank"] = str(rank)
            merged.append(candidate)
            rank += 1
    return merged


def load_resume_feedback(
    path: Path,
    soc: str,
    aic_cores: int,
    toolkit: str | None = None,
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
            or (
                toolkit is not None
                and row.get("toolkit") != toolkit
            )
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
        if not truthy(row.get("success")):
            if row.get("preflight_mode") in {
                "",
                "baseline_drift",
                "provisional",
                "runner_failed",
            }:
                continue
            exclusions.add(fingerprint(workload, schedule))
            # Every failed searched schedule is negative NPU executability
            # evidence. Runners preserve a more specific preflight_mode such
            # as output_not_written or timeout, but restricting feedback to
            # the literal "runtime_rejected" label silently discards those
            # failures and makes the next search repeat the same bad region.
            observations.append(
                MeasuredObservation(
                    workload=workload,
                    schedule=schedule,
                    ratio_vs_official=1.0,
                    ratio_vs_bank=1.0,
                    source="runtime_rejected",
                    record_id=row.get("record_id", path.stem),
                    status_vs_official="runtime_rejected",
                    status_vs_bank="runtime_rejected",
                    bank_schedule=optional_schedule(
                        row.get("bank_tiling_signature")
                    ),
                )
            )
            continue
        preflight_mode = row.get("preflight_mode")
        if preflight_mode in STRICT_NUMERIC_PREFLIGHT_MODES:
            # A completed numeric preflight is exact executability evidence
            # even if surrounding baseline drift makes its latency unusable.
            # Exclude the fingerprint without feeding the unpaired timing to
            # the cost model.
            exclusions.add(fingerprint(workload, schedule))
        verified = (
            truthy(row.get("pair_validated"))
            and preflight_mode in STRICT_NUMERIC_PREFLIGHT_MODES
        )
        if not verified:
            if preflight_mode in STRICT_NUMERIC_PREFLIGHT_MODES:
                observations.append(
                    MeasuredObservation(
                        workload=workload,
                        schedule=schedule,
                        ratio_vs_official=1.0,
                        ratio_vs_bank=1.0,
                        source="runtime_verified",
                        record_id=row.get("record_id", path.stem),
                        bank_schedule=optional_schedule(
                            row.get("bank_tiling_signature")
                        ),
                    )
                )
            continue
        if not row.get("official_ms") or not row.get("bank_ms"):
            continue
        try:
            candidate_ms = float(row["median_ms"])
            official_ms = float(row["official_ms"])
            bank_ms = float(row["bank_ms"])
        except (KeyError, ValueError):
            continue
        exclusions.add(fingerprint(workload, schedule))
        ratio_vs_official = candidate_ms / official_ms
        ratio_vs_bank = candidate_ms / bank_ms
        if not measurement_status_consistent(
            ratio_vs_official,
            ratio_vs_bank,
            row.get("status_vs_official", ""),
            row.get("status_vs_bank", ""),
        ):
            observations.append(
                MeasuredObservation(
                    workload=workload,
                    schedule=schedule,
                    ratio_vs_official=1.0,
                    ratio_vs_bank=1.0,
                    source="runtime_verified",
                    record_id=row.get("record_id", path.stem),
                    bank_schedule=optional_schedule(
                        row.get("bank_tiling_signature")
                    ),
                )
            )
            continue
        observations.append(
            MeasuredObservation(
                workload=workload,
                schedule=schedule,
                ratio_vs_official=ratio_vs_official,
                ratio_vs_bank=ratio_vs_bank,
                source=row.get("candidate_source", ""),
                record_id=(
                    f"{row.get('run_id') or path.stem}:"
                    f"{workload.workload_id}"
                ),
                status_vs_official=row.get("status_vs_official", ""),
                status_vs_bank=row.get("status_vs_bank", ""),
                verified=True,
                structured_verified=(
                    preflight_mode == "numeric_signed_axes_full_v3"
                ),
                bank_schedule=optional_schedule(
                    row.get("bank_tiling_signature")
                ),
            )
        )
    return observations, exclusions


def load_unpaired_one_shot_candidates(
    path: Path,
    soc: str,
    aic_cores: int,
    toolkit: str | None = None,
) -> dict[tuple[int, int, int, str, bool, bool, int], Schedule]:
    """Recover executable one-shot customs whose paired latency was invalid."""

    if not path.is_file():
        return {}
    pending = {}
    with path.open(newline="", encoding="utf-8") as source:
        rows = csv.DictReader(source)
        for row in rows:
            if (
                row.get("candidate_role") != "searched"
                or row.get("candidate_source")
                not in {
                    "one_shot_bank_relative",
                    "one_shot_custom_policy",
                    "one_shot_research_candidate",
                }
                or row.get("soc") != soc
                or int(row.get("aic") or 0) != aic_cores
                or (
                    toolkit is not None
                    and row.get("toolkit") != toolkit
                )
                or not truthy(row.get("success"))
                or truthy(row.get("pair_validated"))
                or row.get("preflight_mode")
                not in STRICT_NUMERIC_PREFLIGHT_MODES
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
                schedule = Schedule.from_signature(
                    row["tiling_signature"]
                )
            except (KeyError, ValueError):
                continue
            pending[workload.identity()] = schedule
    return pending


def hydrate_bank_context(
    observations: list[MeasuredObservation],
) -> tuple[list[MeasuredObservation], int, int]:
    """Attach current RuntimeKb bank schedules to trusted legacy records."""

    missing: dict[
        tuple[int, int, int, str, bool, bool, int],
        Workload,
    ] = {}
    for observation in observations:
        if (
            observation.bank_schedule is None
            and (
                observation.verified
                or observation.source
                in {"runtime_rejected", "runtime_verified"}
            )
        ):
            missing.setdefault(
                observation.workload.identity(),
                observation.workload,
            )
    resolved: dict[
        tuple[int, int, int, str, bool, bool, int],
        Schedule,
    ] = {}
    failures = 0
    for identity, workload in sorted(missing.items()):
        try:
            official = invoke(workload)
            resolved[identity] = exact_roundtrip(
                workload, official.schedule
            ).schedule
        except Exception:
            failures += 1
    hydrated = [
        (
            replace(
                observation,
                bank_schedule=resolved.get(
                    observation.workload.identity()
                ),
            )
            if (
                observation.bank_schedule is None
                and observation.workload.identity() in resolved
            )
            else observation
        )
        for observation in observations
    ]
    return hydrated, len(resolved), failures


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
    parser.add_argument("--toolkit")
    parser.add_argument("--aic-cores", type=int, required=True)
    parser.add_argument("--l0a-bytes", type=int, required=True)
    parser.add_argument("--l0b-bytes", type=int, required=True)
    parser.add_argument("--l0c-bytes", type=int, required=True)
    parser.add_argument("--l1-bytes", type=int, required=True)
    parser.add_argument("--ub-bytes", type=int, required=True)
    parser.add_argument("--l2-bytes", type=int, required=True)
    parser.add_argument("--l2-bpc", type=float, default=1.0)
    parser.add_argument("--hbm-bpc", type=float, default=1.0)
    parser.add_argument("--observations", type=Path, action="append", default=[])
    parser.add_argument("--exclusions", type=Path, action="append", default=[])
    parser.add_argument(
        "--resume-feedback",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--append-candidates", type=Path)
    parser.add_argument("--npu-candidates", type=int, default=40)
    parser.add_argument("--callback-candidates", type=int, default=48)
    parser.add_argument("--behavior-candidates", type=int, default=320)
    parser.add_argument("--workload-limit", type=int, default=0)
    parser.add_argument("--skip-model-validation", action="store_true")
    parser.add_argument(
        "--search-stage",
        choices=("stage1", "stage2"),
        default="stage1",
    )
    parser.add_argument(
        "--selection-mode",
        choices=(
            "adaptive-calibration",
            "calibration",
            "campaign",
            "compact-deployment",
            "direct-rule",
            "one-shot",
        ),
        default="campaign",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    learned_deployment = args.selection_mode in {
        "compact-deployment",
        "one-shot",
    }
    direct_deployment = args.selection_mode == "direct-rule"
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
        ub_bytes=args.ub_bytes,
    )
    workloads = load_workloads(args.workloads, args.aic_cores)
    calibration_quotas = (
        load_calibration_quotas(args.workloads)
        if args.selection_mode
        in {"adaptive-calibration", "calibration"}
        else {}
    )
    if args.workload_limit > 0:
        workloads = workloads[: args.workload_limit]
    if args.selection_mode == "calibration":
        missing_quotas = [
            workload.workload_id
            for workload in workloads
            if workload.workload_id not in calibration_quotas
        ]
        if missing_quotas:
            raise ValueError(
                "template calibration requires an explicit non-legacy "
                "template quota for every workload: "
                + ",".join(missing_quotas)
            )
    history_load_start = time.perf_counter_ns()
    if direct_deployment:
        observations = []
        exclusions = set()
        strict_paired_counts = Counter()
    else:
        observations, exclusions = load_feedback(
            soc=args.soc,
            aic_cores=args.aic_cores,
            observation_paths=args.observations,
            exclusion_paths=args.exclusions,
        )
        strict_paired_counts = load_strict_paired_template_counts(
            args.resume_feedback,
            args.soc,
            args.aic_cores,
            args.toolkit,
            {workload.workload_id: workload for workload in workloads},
        )
    unpaired_one_shot: dict[
        tuple[int, int, int, str, bool, bool, int],
        Schedule,
    ] = {}
    for resume_feedback in (() if direct_deployment else args.resume_feedback):
        resume_observations, resume_exclusions = load_resume_feedback(
            resume_feedback,
            args.soc,
            args.aic_cores,
            args.toolkit,
        )
        unpaired_one_shot.update(
            load_unpaired_one_shot_candidates(
                resume_feedback,
                args.soc,
                args.aic_cores,
                args.toolkit,
            )
        )
        observations.extend(resume_observations)
        exclusions.update(resume_exclusions)
    if args.selection_mode == "calibration":
        retryable_unpaired = load_retryable_unpaired_fingerprints(
            args.resume_feedback,
            set(calibration_quotas),
            args.soc,
            args.aic_cores,
            args.toolkit,
        )
        exclusions.difference_update(retryable_unpaired)
        print(
            "CALIBRATION_RETRYABLE_UNPAIRED "
            f"fingerprints={len(retryable_unpaired)}"
        )
    if args.selection_mode in {
        "adaptive-calibration",
        "one-shot",
    }:
        observations, hydrated, hydration_failures = hydrate_bank_context(
            observations
        )
        print(
            "BANK_CONTEXT "
            f"hydrated_workloads={hydrated} "
            f"failed_workloads={hydration_failures} "
            "source=current_RuntimeKb_callback"
        )
    if learned_deployment:
        # A deployment decision must not learn from an earlier measurement of
        # the target shape. This also makes a resumed run regenerate the same
        # fingerprint so profile.py can reuse its exact completed measurement.
        target_identities = {
            workload.identity() for workload in workloads
        }
        target_prefixes = {
            identity[:6] for identity in target_identities
        }
        observations = [
            observation
            for observation in observations
            if observation.workload.identity() not in target_identities
        ]
        exclusions = {
            item
            for item in exclusions
            if item[:6] not in target_prefixes
        }
    history_load_ms = (
        time.perf_counter_ns() - history_load_start
    ) / 1_000_000.0
    print(
        "TILING_HISTORY_LOAD "
        f"strategy={args.selection_mode} "
        f"history_records={len(observations)} "
        f"ms={history_load_ms:.6f}"
    )
    if direct_deployment:
        print(
            "DIRECT_TEMPLATE_RULE "
            f"version={DIRECT_BASE_RULE_VERSION} "
            "runtime_history=0 model=0 candidate_pool=1 "
            f"audit_paired={DIRECT_BASE_AUDIT_PAIRED_RECORDS} "
            f"audit_unique={DIRECT_BASE_AUDIT_UNIQUE_RECORDS} "
            f"audit_workloads={DIRECT_BASE_AUDIT_WORKLOADS} "
            "audit_winner_workloads="
            f"{DIRECT_BASE_AUDIT_WINNER_WORKLOADS} "
            f"audit_bank_workloads={DIRECT_BASE_AUDIT_BANK_WORKLOADS} "
            f"audit_all_records={DIRECT_RULE_AUDIT_RECORDS} "
            f"audit_all_unique={DIRECT_RULE_AUDIT_UNIQUE_RECORDS} "
            f"audit_all_workloads={DIRECT_RULE_AUDIT_WORKLOADS} "
            "trusted_winner_workloads="
            f"{DIRECT_RULE_TRUSTED_WINNER_WORKLOADS} "
            "trusted_winner_template_matches="
            f"{DIRECT_RULE_TRUSTED_WINNER_TEMPLATE_MATCHES} "
            "trusted_winner_execution_equivalent="
            f"{DIRECT_RULE_TRUSTED_WINNER_EXECUTION_EQUIVALENT} "
            "trusted_winner_structural_near="
            f"{DIRECT_RULE_TRUSTED_WINNER_STRUCTURAL_NEAR} "
            f"l2_resident_ratio={DIRECT_BASE_L2_RESIDENT_RATIO:.12f} "
            "rules=template,geometry,l1,core,l2"
        )
    latency_observations = sum(
        observation.source
        not in {"runtime_rejected", "runtime_verified"}
        for observation in observations
    )
    runtime_rejections = sum(
        observation.source == "runtime_rejected"
        for observation in observations
    )
    runtime_verified_only = sum(
        observation.source == "runtime_verified"
        for observation in observations
    )
    if not direct_deployment:
        print(
            "COST_MODEL_EVIDENCE "
            f"records={len(observations)} "
            f"latency={latency_observations} "
            f"runtime_rejected={runtime_rejections} "
            f"runtime_verified_only={runtime_verified_only}"
        )
    if (
        not args.skip_model_validation
        and args.selection_mode not in {"calibration", "direct-rule"}
    ):
        for leave_workload_out in (False, True):
            validation = validate_feedback_model(
                observations,
                hardware,
                leave_workload_out=leave_workload_out,
            )
            if learned_deployment:
                print(
                    "RUNTIME_RISK_VALIDATION "
                    f"mode={validation.mode} "
                    f"runtime_samples={validation.runtime_samples} "
                    f"risk_auc={validation.runtime_risk_auc:.6g} "
                    "latency_model_used_for_deployment=0"
                )
                continue
            print(
                "COST_MODEL_VALIDATION "
                f"mode={validation.mode} "
                f"latency_samples={validation.latency_samples} "
                f"spearman={validation.latency_spearman:.6g} "
                f"mae={validation.latency_mae:.6g} "
                f"log_mae={validation.latency_log_mae:.6g} "
                f"median_factor={validation.latency_median_factor:.6g} "
                f"p90_factor={validation.latency_p90_factor:.6g} "
                f"pairwise={validation.pairwise_accuracy:.6g} "
                f"pairwise_n={validation.pairwise_comparisons} "
                f"top_quartile_recall={validation.top_quartile_recall:.6g} "
                "best_candidate_percentile="
                f"{validation.best_candidate_percentile:.6g} "
                f"runtime_samples={validation.runtime_samples} "
                f"risk_auc={validation.runtime_risk_auc:.6g} "
                f"analytical_spearman={validation.analytical_spearman:.6g} "
                "analytical_pairwise="
                f"{validation.analytical_pairwise_accuracy:.6g} "
                "validation_only=1"
            )
    engine_observations = (
        ()
        if args.selection_mode == "compact-deployment"
        else (
            observations
            if args.selection_mode
            in {"adaptive-calibration", "campaign", "one-shot"}
            else ()
        )
    )
    if args.selection_mode == "compact-deployment":
        generation_budget = GenerationBudget(
            raw_attempts=384,
            legal_candidates=96,
            behavior_candidates=min(args.behavior_candidates, 48),
            callback_candidates=min(args.callback_candidates, 24),
            npu_candidates=1,
        )
    else:
        generation_budget = GenerationBudget(
            raw_attempts=16000,
            legal_candidates=7000,
            behavior_candidates=args.behavior_candidates,
            callback_candidates=args.callback_candidates,
            npu_candidates=args.npu_candidates,
        )
    engine = (
        None
        if direct_deployment
        else CandidateEngine(
            config=SearchConfig(
                budget=generation_budget,
                include_exploration=args.selection_mode
                in {"adaptive-calibration", "calibration", "campaign"},
            ),
            observations=engine_observations,
            exclusions=exclusions,
        )
    )
    model_setup_start = time.perf_counter_ns()
    one_shot_cost_model = (
        FeedbackCostModel(observations, hardware)
        if learned_deployment
        else None
    )
    one_shot_effect_model = (
        BankRelativeEffectModel(observations, hardware)
        if learned_deployment
        else None
    )
    one_shot_safety_model = (
        BankRelativeSafetyModel(observations, hardware)
        if learned_deployment
        else None
    )
    adaptive_cost_model = (
        FeedbackCostModel(observations, hardware)
        if args.selection_mode == "adaptive-calibration"
        else None
    )
    adaptive_effect_model = (
        BankRelativeEffectModel(observations, hardware)
        if args.selection_mode == "adaptive-calibration"
        else None
    )
    adaptive_safety_model = (
        BankRelativeSafetyModel(observations, hardware)
        if args.selection_mode == "adaptive-calibration"
        else None
    )
    model_setup_ms = (
        time.perf_counter_ns() - model_setup_start
    ) / 1_000_000.0
    print(
        "TILING_MODEL_SETUP "
        f"strategy={args.selection_mode} "
        f"history_records={len(observations)} "
        f"model_enabled={int(learned_deployment)} "
        f"ms={model_setup_ms:.6f}"
    )
    adaptive_observed_bins = (
        frozenset(
            behavior_key(
                behavior_vector(
                    observation.workload,
                    observation.schedule,
                    hardware,
                )
            )
            for observation in observations
        )
        if args.selection_mode == "adaptive-calibration"
        else frozenset()
    )
    if adaptive_effect_model is not None:
        print(
            "ADAPTIVE_FEEDBACK_MODEL "
            f"effect_rows={len(adaptive_effect_model.rows)} "
            f"effect_workloads={adaptive_effect_model.workloads} "
            f"safety_rows={len(adaptive_safety_model.rows)} "
            f"safety_rejected={adaptive_safety_model.rejected_rows} "
            f"observed_behavior_bins={len(adaptive_observed_bins)}"
        )
    if one_shot_effect_model is not None:
        print(
            "BANK_RELATIVE_MODEL "
            f"rows={len(one_shot_effect_model.rows)} "
            f"workloads={one_shot_effect_model.workloads} "
            f"structured_rows={one_shot_effect_model.structured_rows} "
            f"safety_rows={len(one_shot_safety_model.rows)} "
            f"safety_rejected={one_shot_safety_model.rejected_rows}"
        )
        if args.skip_model_validation:
            print(
                "BANK_RELATIVE_VALIDATION "
                "skipped=1 reason=bounded_full_run_cpu_budget"
            )
        else:
            relative_validation = validate_bank_relative_selector(
                observations, hardware
            )
            print(
                "BANK_RELATIVE_VALIDATION "
                "mode=leave_workload_out_with_bank_incumbent "
                f"groups={relative_validation.groups} "
                "oracle_opportunities="
                f"{relative_validation.oracle_opportunities} "
                f"custom_selections={relative_validation.custom_selections} "
                f"custom_winners={relative_validation.custom_winners} "
                f"custom_regressions={relative_validation.custom_regressions} "
                f"severe_regressions={relative_validation.severe_regressions} "
                "median_selected_ratio="
                f"{relative_validation.median_selected_ratio:.6g} "
                f"p90_selected_ratio="
                f"{relative_validation.p90_selected_ratio:.6g}"
            )

    output_rows: list[dict[str, str]] = []
    all_rows: list[dict[str, str]] = []
    source_counts: Counter[str] = Counter()
    template_counts: Counter[str] = Counter()
    print("SEARCH_FRONTIER_BEGIN")
    for workload in workloads:
        completed_by_template = {
            template: strict_paired_counts[
                (workload.workload_id, template)
            ]
            for template in Template
        }
        calibration_completed = sum(
            completed_by_template.values()
        )
        target_quotas = calibration_quotas.get(
            workload.workload_id
        )
        remaining_target_quotas = (
            {
                template: max(
                    0,
                    quota - completed_by_template[template],
                )
                for template, quota in target_quotas.items()
                if quota > completed_by_template[template]
            }
            if target_quotas is not None
            else None
        )
        calibration_budget = (
            sum(remaining_target_quotas.values())
            if remaining_target_quotas is not None
            else max(0, args.npu_candidates - calibration_completed)
        )
        if args.selection_mode == "adaptive-calibration":
            target_templates = frozenset(target_quotas or {})
            evidence = adaptive_calibration_evidence(
                workload,
                observations,
                target_templates,
            )
            if not evidence.should_refine:
                print(
                    "ADAPTIVE_DECISION "
                    f"{workload.workload_id} "
                    "targets="
                    f"{','.join(sorted(item.value for item in target_templates))} "
                    f"paired={evidence.paired} "
                    f"rejected={evidence.rejected} "
                    "best_ratio="
                    f"{evidence.best_ratio if evidence.best_ratio < float('inf') else 'NA'} "
                    "action=skip_before_host_generation"
                )
                continue
        if (
            args.selection_mode == "calibration"
            and calibration_budget == 0
        ):
            print(
                "SEARCH_WORKLOAD "
                f"{workload.workload_id} "
                f"shape={workload.m}x{workload.n}x{workload.k} "
                f"dtype={workload.dtype} "
                f"trans={int(workload.trans_a)}"
                f"{int(workload.trans_b)} selected=0 "
                f"resume_completed={calibration_completed} "
                "target_quotas="
                f"{target_quotas or {}} "
                "action=skip_completed_calibration_before_host_generation"
            )
            continue
        try:
            official = invoke(workload)
            bank = exact_roundtrip(workload, official.schedule)
        except Exception as exception:
            print(
                "SEARCH_WORKLOAD_BLOCKED "
                f"{workload.workload_id} stage=official_callback "
                f"reason={str(exception)[:240]} action=continue"
            )
            continue
        control_row = candidate_row(
            workload,
            0,
            "bank_seed_control",
            "official_callback_control",
            bank.schedule,
            bank,
        )
        control_row["bank_tiling_signature"] = (
            bank.schedule.signature_text()
        )
        control_row["deployment_strategy"] = args.selection_mode
        output_rows.append(control_row)
        incumbent = Candidate(
            schedule=bank.schedule,
            template=template_of(bank.schedule),
            source="bank_incumbent",
            rationale="exact official RuntimeKb control",
        )
        adaptive_template_quotas = (
            {
                template: args.callback_candidates
                for template in (target_quotas or {})
            }
            if args.selection_mode == "adaptive-calibration"
            else None
        )
        host_tiling_start = time.perf_counter_ns()
        generation_start = host_tiling_start
        if direct_deployment:
            try:
                direct_candidate = direct_rule_candidate(
                    workload,
                    hardware,
                )
            except Exception as exception:
                print(
                    "DIRECT_RULE_BLOCKED "
                    f"{workload.workload_id} "
                    "stage=construction "
                    f"reason={str(exception)[:240]} action=continue"
                )
                continue
            result = None
            ordered = [direct_candidate]
            generated_candidates = 1
        else:
            if engine is None:
                raise RuntimeError("candidate engine was not initialized")
            result = engine.generate(
                workload,
                hardware,
                local_anchor=bank.schedule,
                template_quotas=(
                    remaining_target_quotas
                    if args.selection_mode == "calibration"
                    else adaptive_template_quotas
                ),
            )
            direct_candidates = []
            if learned_deployment:
                try:
                    direct_candidates.append(
                        direct_rule_candidate(workload, hardware)
                    )
                except Exception as exception:
                    print(
                        "ONE_SHOT_DIRECT_RULE_BLOCKED "
                        f"{workload.workload_id} "
                        f"reason={str(exception)[:240]} "
                        "action=continue_with_independent_frontier"
                    )
            ordered = [
                *direct_candidates,
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
            generated_candidates = result.legal_candidates
        generation_ms = (
            time.perf_counter_ns() - generation_start
        ) / 1_000_000.0
        callback_accepted: list[tuple[Candidate, object]] = []
        callback_rejected = 0
        seen: set[tuple[int, ...]] = set()
        callback_start = time.perf_counter_ns()
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
                rejected["bank_tiling_signature"] = (
                    bank.schedule.signature_text()
                )
                all_rows.append(rejected)
                continue
            callback_accepted.append((candidate, callback))
            callback_limit = (
                1
                if direct_deployment
                else generation_budget.callback_candidates
            )
            if len(callback_accepted) >= callback_limit:
                break
        callback_ms = (
            time.perf_counter_ns() - callback_start
        ) / 1_000_000.0
        if direct_deployment and not callback_accepted:
            print(
                "DIRECT_RULE_BLOCKED "
                f"{workload.workload_id} "
                "reason=exact_callback_rejected action=continue"
            )
            continue
        callback_by_signature = {
            candidate.schedule.signature(): callback
            for candidate, callback in callback_accepted
        }
        callback_candidates = [
            candidate for candidate, _ in callback_accepted
        ]
        callback_by_signature[bank.schedule.signature()] = bank
        one_shot_decision = None
        selection_start = time.perf_counter_ns()
        if learned_deployment:
            try:
                one_shot_decision = select_one_shot_candidate(
                    workload,
                    callback_candidates,
                    incumbent,
                    observations,
                    hardware,
                    cost_model=one_shot_cost_model,
                    effect_model=one_shot_effect_model,
                    safety_model=one_shot_safety_model,
                )
            except ValueError as exception:
                print(
                    "ONE_SHOT_BLOCKED "
                    f"{workload.workload_id} "
                    f"reason={str(exception)[:240]} "
                    "deployment=none fallback=disabled action=continue"
                )
                continue
            pending_schedule = unpaired_one_shot.get(
                workload.identity()
            )
            if (
                pending_schedule is not None
                and args.selection_mode != "compact-deployment"
            ):
                pending_candidate = next(
                    (
                        candidate
                        for candidate in callback_candidates
                        if candidate.schedule == pending_schedule
                    ),
                    None,
                )
                if pending_candidate is not None:
                    metrics = dict(pending_candidate.metrics)
                    metrics.update(
                        {
                            "deployment_evidence_strong": 0.0,
                            "deployment_recommended_custom": 0.0,
                            "bank_execution_equivalent": 0.0,
                            "bank_signature_exact": 0.0,
                            "resume_unpaired_remeasurement": 1.0,
                        }
                    )
                    selected = Candidate(
                        schedule=pending_candidate.schedule,
                        template=pending_candidate.template,
                        source="one_shot_research_candidate",
                        rationale=(
                            "repeat the same numerically verified research "
                            "challenger because its paired latency was invalid; "
                            f"generator={pending_candidate.source}"
                        ),
                        acquisition=pending_candidate.acquisition,
                        parent_signatures=(
                            incumbent.schedule.signature(),
                        ),
                        behavior_key=pending_candidate.behavior_key,
                        metrics=metrics,
                    )
                    one_shot_decision = OneShotDecision(
                        candidate=selected,
                        deployment_candidate=incumbent,
                        generator_source=pending_candidate.source,
                        evaluated=one_shot_decision.evaluated,
                        safe_candidates=one_shot_decision.safe_candidates,
                        direct_base_candidates=(
                            one_shot_decision.direct_base_candidates
                        ),
                        transfer_eligible_candidates=(
                            one_shot_decision
                            .transfer_eligible_candidates
                        ),
                        custom_eligible_candidates=(
                            one_shot_decision
                            .custom_eligible_candidates
                        ),
                        bank_equivalent_candidates=(
                            one_shot_decision
                            .bank_equivalent_candidates
                        ),
                        local_candidates=one_shot_decision.local_candidates,
                        selection_policy=(
                            "resume_unpaired_exact_remeasurement"
                        ),
                    )
                else:
                    print(
                        "ONE_SHOT_REMEASURE_SKIPPED "
                        f"{workload.workload_id} "
                        "reason=prior_signature_not_regenerated"
                    )
            research_measurement = (
                one_shot_decision.candidate.source
                == "one_shot_research_candidate"
            )
            if research_measurement and (
                one_shot_decision.deployment_candidate.schedule
                != incumbent.schedule
            ):
                raise ValueError(
                    "one-shot research measurement must retain the bank "
                    "deployment recommendation"
                )
            if not research_measurement and (
                one_shot_decision.deployment_candidate.schedule
                != one_shot_decision.candidate.schedule
            ):
                raise ValueError(
                    "one-shot deployment must use the independently "
                    "generated selection; bank record injection is disabled"
                )
            selected_candidates = [one_shot_decision.candidate]
            racing_plan = None
        elif direct_deployment:
            selected_candidates = callback_candidates
            racing_plan = None
        elif args.selection_mode == "adaptive-calibration":
            selected_candidates = select_adaptive_calibration_candidates(
                workload,
                callback_candidates,
                incumbent,
                observations,
                hardware,
                args.npu_candidates,
                cost_model=adaptive_cost_model,
                effect_model=adaptive_effect_model,
                safety_model=adaptive_safety_model,
                observed_bins=adaptive_observed_bins,
                target_templates=frozenset(target_quotas or {}),
            )
            racing_plan = None
        elif args.selection_mode == "calibration":
            selected_candidates = select_calibration_candidates(
                workload,
                callback_candidates,
                incumbent,
                observations,
                hardware,
                calibration_budget,
                template_quotas=remaining_target_quotas,
            )
            racing_plan = None
        elif args.search_stage == "stage2":
            racing_plan = plan_template_race(
                workload,
                callback_candidates,
                observations,
                args.npu_candidates,
            )
            selected_candidates = select_behavior_coverage(
                workload,
                callback_candidates,
                observations,
                hardware,
                racing_plan.budget,
                probe_templates=True,
                template_probe_floor=1,
                template_quotas=racing_plan.template_quotas,
            )
        else:
            racing_plan = plan_template_race(
                workload,
                callback_candidates,
                (),
                args.npu_candidates,
            )
            selected_candidates = select_behavior_coverage(
                workload,
                callback_candidates,
                observations,
                hardware,
                racing_plan.budget,
                probe_templates=True,
                template_probe_floor=1,
                template_quotas=racing_plan.template_quotas,
                allow_risky_template_probes=True,
            )
        selection_ms = (
            time.perf_counter_ns() - selection_start
        ) / 1_000_000.0
        host_tiling_total_ms = (
            time.perf_counter_ns() - host_tiling_start
        ) / 1_000_000.0
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
            row["bank_tiling_signature"] = (
                bank.schedule.signature_text()
            )
            row.update(
                {
                    "deployment_strategy": args.selection_mode,
                    "host_history_load_ms": (
                        f"{history_load_ms:.6f}"
                    ),
                    "host_model_setup_ms": f"{model_setup_ms:.6f}",
                    "host_generation_ms": f"{generation_ms:.6f}",
                    "host_callback_ms": f"{callback_ms:.6f}",
                    "host_selection_ms": f"{selection_ms:.6f}",
                    "host_tiling_total_ms": (
                        f"{host_tiling_total_ms:.6f}"
                    ),
                    "host_generated_candidates": str(
                        generated_candidates
                    ),
                    "host_callback_candidates": str(
                        len(callback_candidates)
                    ),
                }
            )
            output_rows.append(row)
            all_rows.append(row)
            source_counts[candidate.source] += 1
            template_counts[candidate.template.value] += 1
        reports = (
            ";".join(
                (
                    f"{report.template.value}:raw={report.raw_generated},"
                    f"common={report.common_legal},"
                    f"template={report.template_legal},"
                    f"emitted={report.emitted}"
                )
                for report in result.reports
            )
            if result is not None
            else (
                f"{ordered[0].template.value}:"
                "raw=1,common=1,template=1,emitted=1"
            )
        )
        print(
            "SEARCH_WORKLOAD "
            f"{workload.workload_id} shape={workload.m}x{workload.n}x{workload.k} "
            f"dtype={workload.dtype} trans={int(workload.trans_a)}"
            f"{int(workload.trans_b)} selected={len(accepted)} "
            f"callback_rejected={callback_rejected} "
            "behavior_bins="
            f"{result.behavior_bins if result is not None else 1} "
            f"legal_pool={generated_candidates} "
            "draft_pool="
            f"{result.draft_candidates if result is not None else 1} "
            "excluded="
            f"{result.excluded_fingerprints if result is not None else 0} "
            f"solvers={reports}"
        )
        print(
            "TILING_TIME "
            f"strategy={args.selection_mode} "
            f"workload={workload.workload_id} "
            f"generation_ms={generation_ms:.6f} "
            f"callback_ms={callback_ms:.6f} "
            f"selection_ms={selection_ms:.6f} "
            f"total_ms={host_tiling_total_ms:.6f} "
            f"generated={generated_candidates} "
            f"callback_accepted={len(callback_candidates)} "
            f"selected={len(accepted)}"
        )
        if one_shot_decision is not None:
            selected = one_shot_decision.candidate
            metrics = selected.metrics
            print(
                "ONE_SHOT_DECISION "
                f"{workload.workload_id} "
                f"strategy={args.selection_mode} "
                f"template={selected.template.value} "
                f"evaluated={one_shot_decision.evaluated} "
                f"safe={one_shot_decision.safe_candidates} "
                f"direct_base={one_shot_decision.direct_base_candidates} "
                "transfer_eligible="
                f"{one_shot_decision.transfer_eligible_candidates} "
                "custom_eligible="
                f"{one_shot_decision.custom_eligible_candidates} "
                "bank_equivalent="
                f"{one_shot_decision.bank_equivalent_candidates} "
                "strong_evidence="
                f"{int(metrics.get('deployment_evidence_strong', 0.0))} "
                f"local={one_shot_decision.local_candidates} "
                f"generator={one_shot_decision.generator_source} "
                f"policy={one_shot_decision.selection_policy} "
                "deployment="
                f"{'custom' if metrics.get('deployment_recommended_custom', 0.0) else 'bank'} "
                "bank_signature_exact="
                f"{int(metrics.get('bank_signature_exact', 0.0))} "
                "deployment_signature="
                f"{one_shot_decision.deployment_candidate.schedule.signature_text()} "
                f"score={selected.acquisition:.6g} "
                "predicted_ratio="
                f"{metrics.get('predicted_latency_ratio', 1.0):.6g} "
                "upper_ratio="
                f"{metrics.get('bank_relative_upper_ratio', 1.0):.6g} "
                "effect_samples="
                f"{metrics.get('bank_relative_samples', 0.0):.6g} "
                "effect_behaviors="
                f"{metrics.get('bank_relative_behavior_samples', 0.0):.6g} "
                "effect_uncertainty="
                f"{metrics.get('bank_relative_uncertainty', 1.0):.6g} "
                "effect_support="
                f"{metrics.get('bank_relative_support', 0.0):.6g} "
                "safety_behaviors="
                f"{metrics.get('bank_runtime_behavior_samples', 0.0):.6g} "
                "bank_structure_preserved="
                f"{int(metrics.get('bank_structure_preserved', 0.0))} "
                "bank_subsystems_changed="
                f"{int(metrics.get('bank_changed_subsystems', 0.0))} "
                "bank_transition_risk="
                f"{int(metrics.get('bank_transition_risk_tier', 3.0))} "
                "template_competitive="
                f"{int(metrics.get('template_competitive', 0.0))} "
                "template_hw_opportunity="
                f"{int(metrics.get('template_hardware_opportunity', 0.0))} "
                "active_gain="
                f"{metrics.get('template_active_core_gain', 1.0):.6g} "
                "compute_floor="
                f"{metrics.get('template_compute_floor_ratio', 1.0):.6g} "
                "conservative_floor="
                f"{metrics.get('template_conservative_floor_ratio', 1.0):.6g} "
                "l1_pipeline="
                f"{metrics.get('l1_pipeline_efficiency', 0.0):.6g} "
                "l2_wave="
                f"{metrics.get('l2_wave_efficiency', 0.0):.6g} "
                "l2_pressure="
                f"{metrics.get('l2_capacity_pressure', 0.0):.6g} "
                "runtime_risk="
                f"{metrics.get('runtime_risk_score', 0.5):.6g} "
                f"signature={selected.schedule.signature_text()}"
            )
            expected_candidates = 1
        elif direct_deployment:
            selected = accepted[0][0]
            print(
                "DIRECT_RULE_DECISION "
                f"{workload.workload_id} "
                f"template={selected.template.value} "
                "history_rows_used=0 candidate_pool=1 "
                "formula=source_template_then_geometry_l1_core_l2 "
                "bank_signature_exact="
                f"{int(selected.schedule == bank.schedule)} "
                f"signature={selected.schedule.signature_text()}"
            )
            expected_candidates = 1
        elif args.selection_mode == "campaign":
            evidence = ",".join(
                (
                    f"{item.template.value}:{item.samples}:"
                    f"{item.best_ratio:.6g}:{item.robust_ratio:.6g}:"
                    f"{item.winners}"
                )
                for item in racing_plan.evidence
            ) or "none"
            quotas = ",".join(
                f"{template.value}:{quota}"
                for template, quota in sorted(
                    racing_plan.template_quotas.items(),
                    key=lambda item: item[0].value,
                )
            )
            print(
                "TEMPLATE_RACE "
                f"{workload.workload_id} state={racing_plan.state} "
                f"budget={racing_plan.budget}/{args.npu_candidates} "
                f"quotas={quotas or 'none'} evidence={evidence}"
            )
            expected_candidates = racing_plan.budget
        elif args.selection_mode == "adaptive-calibration":
            expected_candidates = args.npu_candidates
            selected_templates = Counter(
                candidate.template.value
                for candidate in selected_candidates
            )
            selected_sources = Counter(
                candidate.source for candidate in selected_candidates
            )
            selected_bins = {
                candidate.behavior_key
                for candidate in selected_candidates
            }
            adaptive_metrics = (
                selected_candidates[0].metrics
                if selected_candidates
                else {}
            )
            print(
                "ADAPTIVE_PLAN "
                f"{workload.workload_id} selected={len(accepted)} "
                f"requested={args.npu_candidates} "
                "evaluated="
                f"{int(adaptive_metrics.get('adaptive_evaluated', 0.0))} "
                "safe_pool="
                f"{int(adaptive_metrics.get('adaptive_safe_pool', 0.0))} "
                "unsafe_filtered="
                f"{int(adaptive_metrics.get('adaptive_unsafe_filtered', 0.0))} "
                f"behavior_bins={len(selected_bins)} "
                "sources="
                + ",".join(
                    f"{source}:{count}"
                    for source, count in sorted(
                        selected_sources.items()
                    )
                )
                + " templates="
                + ",".join(
                    f"{template}:{count}"
                    for template, count in sorted(
                        selected_templates.items()
                    )
                )
            )
        else:
            expected_candidates = calibration_budget
            selected_templates = Counter(
                candidate.template.value
                for candidate in selected_candidates
            )
            local = sum(
                candidate.source
                == "calibration_local_counterfactual"
                for candidate in selected_candidates
            )
            template_probes = sum(
                candidate.source == "calibration_template_probe"
                for candidate in selected_candidates
            )
            print(
                "CALIBRATION_PLAN "
                f"{workload.workload_id} selected={len(accepted)} "
                f"resume_completed={calibration_completed} "
                f"remaining={calibration_budget} "
                f"local={local} "
                "coupled="
                f"{len(accepted) - local - template_probes} "
                f"template_probes={template_probes} "
                f"bank_template={incumbent.template.value} "
                "target_quotas="
                + (
                    ",".join(
                        f"{template.value}:{quota}"
                        for template, quota in sorted(
                            (target_quotas or {}).items(),
                            key=lambda item: item[0].value,
                        )
                    )
                    or "none"
                )
                + " selected_templates="
                + ",".join(
                    f"{template}:{count}"
                    for template, count in sorted(
                        selected_templates.items()
                    )
                )
            )
            bank_vector = behavior_vector(
                workload, incumbent.schedule, hardware
            )
            for template in sorted(
                {candidate.template for candidate in selected_candidates},
                key=lambda item: item.value,
            ):
                family = [
                    candidate
                    for candidate in selected_candidates
                    if candidate.template == template
                ]
                vectors = [
                    behavior_vector(
                        workload, candidate.schedule, hardware
                    )
                    for candidate in family
                ]
                behavior_bins = {
                    behavior_key(vector) for vector in vectors
                }
                geometries = {
                    (
                        candidate.schedule["baseM"],
                        candidate.schedule["baseN"],
                        candidate.schedule["baseK"],
                    )
                    for candidate in family
                }
                pipelines = {
                    (
                        candidate.schedule["depthA1"],
                        candidate.schedule["depthB1"],
                        candidate.schedule["stepM"],
                        candidate.schedule["stepN"],
                        candidate.schedule["stepKa"],
                        candidate.schedule["stepKb"],
                        candidate.schedule["dbL0A"],
                        candidate.schedule["dbL0B"],
                        candidate.schedule["dbL0C"],
                        candidate.schedule["iterateOrder"],
                    )
                    for candidate in family
                }
                executions = {
                    (
                        candidate.schedule["usedCoreNum"],
                        candidate.schedule["singleCoreM"],
                        candidate.schedule["singleCoreN"],
                        candidate.schedule["singleCoreK"],
                        candidate.schedule["l2MTileCnt"],
                        candidate.schedule["l2NTileCnt"],
                        candidate.schedule["l2MTileBlock"],
                        candidate.schedule["l2NTileBlock"],
                        candidate.schedule["l2IterateOrder"],
                    )
                    for candidate in family
                }
                distances = [
                    behavior_distance(vector, bank_vector)
                    for vector in vectors
                ]
                metric_ranges = {}
                for name in (
                    "active_cores",
                    "core_rounds",
                    "l0_occupancy",
                    "l1_occupancy",
                    "k_passes",
                    "padding_efficiency",
                    "l2_capacity_pressure",
                    "split_reduction_ratio",
                    "traffic_amplification",
                    "full_load_resident_ratio",
                ):
                    values = [
                        vector.metrics.get(name, 0.0)
                        for vector in vectors
                    ]
                    metric_ranges[name] = (
                        min(values, default=0.0),
                        max(values, default=0.0),
                    )
                print(
                    "CALIBRATION_DISTRIBUTION "
                    f"{workload.workload_id} "
                    f"template={template.value} "
                    f"selected={len(family)} "
                    f"behavior_bins={len(behavior_bins)} "
                    f"geometries={len(geometries)} "
                    f"pipelines={len(pipelines)} "
                    f"executions={len(executions)} "
                    f"bank_distance="
                    f"{min(distances, default=0.0):.6g}/"
                    f"{max(distances, default=0.0):.6g} "
                    + " ".join(
                        f"{name}={lower:.6g}/{upper:.6g}"
                        for name, (lower, upper) in metric_ranges.items()
                    )
                )
        if len(accepted) < expected_candidates:
            print(
                f"SEARCH_CAPABILITY_GAP {workload.workload_id} "
                f"requested={expected_candidates} accepted={len(accepted)}"
            )

    appended = 0
    if (
        args.append_candidates is not None
        and args.selection_mode == "campaign"
    ):
        with args.append_candidates.open(
            newline="", encoding="utf-8"
        ) as source:
            previous = list(csv.DictReader(source))
        before = sum(
            row.get("candidate_role") == "searched"
            for row in output_rows
        )
        output_rows = merge_candidate_rows(previous, output_rows)
        after = sum(
            row.get("candidate_role") == "searched"
            for row in output_rows
        )
        appended = after - before

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
        f"appended_candidates={appended} "
        f"sources={dict(sorted(source_counts.items()))} "
        f"templates={dict(sorted(template_counts.items()))} "
        "paired_controls="
        f"{sum(row['candidate_role'] == 'bank_seed_control' for row in output_rows)} "
        f"observations={len(observations)}"
    )
    print("SEARCH_FRONTIER_END")


if __name__ == "__main__":
    main()
