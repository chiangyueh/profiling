from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence, Tuple

from .behavior import behavior_key, behavior_vector, workload_distance
from .contracts import ceil_div, template_of, validate_schedule
from .domain import (
    BehaviorTarget,
    Candidate,
    Hardware,
    MeasuredObservation,
    Schedule,
    Template,
    Workload,
)


Fingerprint = Tuple[int, int, int, str, bool, bool, Tuple[int, ...]]
STRICT_NUMERIC_PREFLIGHT_MODES = {
    "numeric_ones_full_v2",
    "numeric_signed_axes_full_v3",
}


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def fingerprint(workload: Workload, schedule: Schedule) -> Fingerprint:
    return (
        workload.m,
        workload.n,
        workload.k,
        workload.dtype,
        workload.trans_a,
        workload.trans_b,
        schedule.signature(),
    )


def _row_workload(row: dict[str, str], aic_cores: int) -> Workload | None:
    try:
        return Workload(
            workload_id=row.get("workload_id") or row.get("id") or "",
            m=int(row["m"]),
            n=int(row["n"]),
            k=int(row["k"]),
            dtype=row["dtype"].lower(),
            trans_a=_truthy(row.get("trans_a")),
            trans_b=_truthy(row.get("trans_b")),
            max_cores=min(
                int(row.get("max_cores") or aic_cores),
                aic_cores,
            ),
        )
    except (KeyError, ValueError):
        return None


def _row_schedule(row: dict[str, str]) -> Schedule | None:
    signature = row.get("tiling_signature", "")
    if not signature:
        return None
    try:
        return Schedule.from_signature(signature)
    except ValueError:
        return None


def _stable(row: dict[str, str]) -> bool:
    try:
        median_ms = float(row.get("median_ms") or 0)
        stddev_ms = float(row.get("stddev_ms") or 0)
    except ValueError:
        return False
    return (
        math.isfinite(median_ms)
        and math.isfinite(stddev_ms)
        and median_ms > 0
        and 0 <= stddev_ms / median_ms <= 0.05
    )


def _comparison_status(
    baseline: dict[str, str],
    candidate: dict[str, str],
) -> str:
    try:
        baseline_ms = float(baseline.get("median_ms") or 0)
        baseline_stddev = max(
            0.0, float(baseline.get("stddev_ms") or 0)
        )
        candidate_ms = float(candidate.get("median_ms") or 0)
        candidate_stddev = max(
            0.0, float(candidate.get("stddev_ms") or 0)
        )
    except ValueError:
        return "invalid_measurement"
    if baseline_ms <= 0 or candidate_ms <= 0:
        return "invalid_measurement"
    delta_pct = 100.0 * (candidate_ms - baseline_ms) / baseline_ms
    noise_pct = max(
        1.0,
        200.0
        * math.hypot(baseline_stddev, candidate_stddev)
        / baseline_ms,
    )
    if delta_pct < -noise_pct:
        return "improved"
    if delta_pct > noise_pct:
        return "regressed"
    return "within_noise"


def _untrusted_observation_status(row: dict[str, str]) -> bool:
    statuses = " ".join(
        (
            row.get("status_vs_official", ""),
            row.get("status_vs_bank", ""),
        )
    ).lower()
    return "unstable" in statuses or "incoherent" in statuses


def _runtime_rejected(observation: MeasuredObservation) -> bool:
    return observation.source == "runtime_rejected"


def load_feedback(
    *,
    soc: str,
    aic_cores: int,
    profile_paths: Iterable[Path] = (),
    observation_paths: Iterable[Path] = (),
    exclusion_paths: Iterable[Path] = (),
) -> tuple[list[MeasuredObservation], set[Fingerprint]]:
    observations: list[MeasuredObservation] = []
    exclusions: set[Fingerprint] = set()

    for path in exclusion_paths:
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                if row.get("soc") != soc:
                    continue
                try:
                    if int(row.get("aic") or 0) != aic_cores:
                        continue
                except ValueError:
                    continue
                workload = _row_workload(row, aic_cores)
                schedule = _row_schedule(row)
                if workload is not None and schedule is not None:
                    exclusions.add(fingerprint(workload, schedule))

    for path in observation_paths:
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                if row.get("soc") != soc:
                    continue
                try:
                    if int(row.get("aic") or 0) != aic_cores:
                        continue
                    ratio_official = float(row["ratio_vs_official"])
                    ratio_bank = float(row["ratio_vs_bank"])
                except (KeyError, ValueError):
                    continue
                workload = _row_workload(row, aic_cores)
                schedule = _row_schedule(row)
                if (
                    workload is None
                    or schedule is None
                    or not math.isfinite(ratio_official)
                    or not math.isfinite(ratio_bank)
                    or ratio_official <= 0
                    or ratio_bank <= 0
                    or _untrusted_observation_status(row)
                ):
                    continue
                observations.append(
                    MeasuredObservation(
                        workload=workload,
                        schedule=schedule,
                        ratio_vs_official=ratio_official,
                        ratio_vs_bank=ratio_bank,
                        source=row.get("candidate_source", ""),
                        record_id=(
                            f"{row.get('campaign', path.stem)}:"
                            f"{workload.workload_id}"
                        ),
                        status_vs_official=row.get(
                            "status_vs_official", ""
                        ),
                        status_vs_bank=row.get("status_vs_bank", ""),
                        verified=_truthy(row.get("verified")),
                        structured_verified=_truthy(
                            row.get("structured_verified")
                        ),
                    )
                )
                exclusions.add(fingerprint(workload, schedule))

    for path in profile_paths:
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        valid_rows = []
        for row in rows:
            try:
                valid = (
                    row.get("resume_soc") == soc
                    and int(row.get("resume_aic") or 0) == aic_cores
                    and _truthy(row.get("success"))
                )
            except ValueError:
                valid = False
            if valid:
                valid_rows.append(row)
        references: dict[
            tuple[str, str], dict[str, dict[str, str]]
        ] = {}
        for row in valid_rows:
            role = row.get("candidate_role", "")
            if role not in {
                "official_operator_baseline",
                "bank_seed_control",
            }:
                continue
            references.setdefault(
                (
                    row.get("workload_id", ""),
                    row.get("resume_run", ""),
                ),
                {},
            )[role] = row
        for row in valid_rows:
            if (
                row.get("candidate_role") != "searched"
                or not _truthy(row.get("preflight_passed"))
                or not _stable(row)
            ):
                continue
            workload = _row_workload(row, aic_cores)
            schedule = _row_schedule(row)
            if workload is None or schedule is None:
                continue
            exclusions.add(fingerprint(workload, schedule))
            pair = references.get(
                (
                    workload.workload_id,
                    row.get("resume_run", ""),
                ),
                {},
            )
            official = pair.get("official_operator_baseline")
            bank = pair.get("bank_seed_control")
            if (
                official is None
                or bank is None
                or not _stable(official)
                or not _stable(bank)
            ):
                continue
            try:
                candidate_ms = float(row["median_ms"])
                official_ms = float(official["median_ms"])
                bank_ms = float(bank["median_ms"])
            except (KeyError, ValueError):
                continue
            preflight_mode = row.get("preflight_mode")
            observations.append(
                MeasuredObservation(
                    workload=workload,
                    schedule=schedule,
                    ratio_vs_official=candidate_ms / official_ms,
                    ratio_vs_bank=candidate_ms / bank_ms,
                    source=row.get("search_candidate_source", ""),
                    record_id=row.get("resume_record_id") or path.stem,
                    status_vs_official=_comparison_status(
                        official, row
                    ),
                    status_vs_bank=_comparison_status(bank, row),
                    verified=(
                        _truthy(row.get("pair_validated"))
                        and preflight_mode in STRICT_NUMERIC_PREFLIGHT_MODES
                    ),
                    structured_verified=(
                        _truthy(row.get("pair_validated"))
                        and preflight_mode == "numeric_signed_axes_full_v3"
                    ),
                )
            )
    unique: dict[
        tuple[Fingerprint, str], MeasuredObservation
    ] = {}
    for observation in observations:
        unique[
            (fingerprint(observation.workload, observation.schedule), observation.record_id)
        ] = observation
    return list(unique.values()), exclusions


def feedback_targets(
    workload: Workload,
    hardware: Hardware,
    observations: Sequence[MeasuredObservation],
) -> list[BehaviorTarget]:
    targets: list[BehaviorTarget] = []
    same_workload = [
        observation
        for observation in observations
        if observation.workload.identity() == workload.identity()
    ]
    transfer_winners = sorted(
        (
            observation
            for observation in observations
            if observation.workload.identity() != workload.identity()
            and observation.is_verified_winner
            and not _runtime_rejected(observation)
        ),
        key=lambda observation: (
            workload_distance(workload, observation.workload),
            observation.record_id,
        ),
    )[:12]
    for observation in [*same_workload, *transfer_winners]:
        # A poisoned or incomplete NPU output is not a latency target.
        # Exact one-factor runtime counterfactuals are generated separately.
        if _runtime_rejected(observation):
            continue
        if not (observation.is_winner or observation.is_regression):
            continue
        vector = behavior_vector(
            observation.workload, observation.schedule, hardware
        )
        metrics = vector.metrics
        is_transfer = (
            observation.workload.identity() != workload.identity()
        )
        origin = (
            "transfer_winner"
            if is_transfer
            else ("winner" if observation.is_winner else "counterfactual")
        )
        direction = 1.0 if observation.is_winner else -1.0
        targets.append(
            BehaviorTarget(
                template=template_of(observation.schedule),
                l0_occupancy=min(
                    0.98,
                    max(
                        0.08,
                        metrics["l0_occupancy"] + direction * 0.08,
                    ),
                ),
                l1_occupancy=min(
                    0.98,
                    max(
                        0.08,
                        metrics["l1_occupancy"] + direction * 0.08,
                    ),
                ),
                core_rounds=max(
                    1.0,
                    metrics["core_rounds"]
                    + (-1.0 if observation.is_winner else 1.0),
                ),
                k_passes=max(1.0, metrics["k_passes"]),
                padding_efficiency=metrics["padding_efficiency"],
                l2_working_set_ratio=max(
                    0.02, metrics["l2_working_set_ratio"]
                ),
                split_reduction_ratio=metrics["split_reduction_ratio"],
                full_load_resident_ratio=metrics[
                    "full_load_resident_ratio"
                ],
                origin=origin,
            )
        )

    observed_keys = {
        behavior_key(
            behavior_vector(
                observation.workload, observation.schedule, hardware
            )
        )
        for observation in same_workload
    }
    coverage_grid = (
        (0.18, 0.25, 1.0, 2.0),
        (0.40, 0.50, 2.0, 4.0),
        (0.65, 0.70, 4.0, 8.0),
        (0.88, 0.90, 8.0, 16.0),
    )
    for template in (
        Template.BASE,
        Template.SINGLE_CORE_SPLIT_K,
        Template.DETERMINISTIC_SPLIT_K,
        Template.AL1_FULL_LOAD,
        Template.BL1_FULL_LOAD,
    ):
        for l0, l1, rounds, k_passes in coverage_grid:
            coarse_key = (
                template.value,
                int(l0 * 5),
                int(l1 * 5),
                int(math.log2(rounds + 1)),
            )
            if any(
                key[0] == coarse_key[0]
                and key[7] == coarse_key[1]
                and key[8] == coarse_key[2]
                for key in observed_keys
            ):
                continue
            targets.append(
                BehaviorTarget(
                    template=template,
                    l0_occupancy=l0,
                    l1_occupancy=l1,
                    core_rounds=rounds,
                    k_passes=k_passes,
                    origin="unexplored_behavior",
                )
            )
    # Targets are not source quotas; repeated requests are collapsed.
    unique = {
        (
            target.template,
            target.l0_occupancy,
            target.l1_occupancy,
            target.core_rounds,
            target.k_passes,
            target.origin,
        ): target
        for target in targets
    }
    return list(unique.values())


def _l2_mutations(workload: Workload, schedule: Schedule) -> list[Schedule]:
    if schedule["l2MTileBlock"] <= 0 or schedule["l2NTileBlock"] <= 0:
        return []
    m_total = ceil_div(workload.m, schedule["singleCoreM"])
    n_total = ceil_div(workload.n, schedule["singleCoreN"])
    candidates: list[Schedule] = []
    for m_block, n_block in (
        (max(1, schedule["l2MTileBlock"] // 2), schedule["l2NTileBlock"]),
        (min(m_total, schedule["l2MTileBlock"] * 2), schedule["l2NTileBlock"]),
        (schedule["l2MTileBlock"], max(1, schedule["l2NTileBlock"] // 2)),
        (schedule["l2MTileBlock"], min(n_total, schedule["l2NTileBlock"] * 2)),
    ):
        candidates.append(
            schedule.replace(
                l2MTileCnt=ceil_div(m_total, m_block),
                l2NTileCnt=ceil_div(n_total, n_block),
                l2MTileBlock=m_block,
                l2NTileBlock=n_block,
            )
        )
    return candidates


def semantic_mutations(
    workload: Workload,
    hardware: Hardware,
    schedule: Schedule,
    *,
    source: str,
    parent: tuple[int, ...],
) -> list[Candidate]:
    mutations = [
        schedule.replace(iterateOrder=1 - schedule["iterateOrder"]),
        schedule.replace(
            l2IterateOrder=(schedule["l2IterateOrder"] + 1) % 3
        ),
        schedule.replace(dbL0C=1 if schedule["dbL0C"] == 2 else 2),
        *_l2_mutations(workload, schedule),
    ]
    core_limit = min(workload.max_cores, hardware.aic_cores)
    for cores in {
        max(1, schedule["usedCoreNum"] - 1),
        min(core_limit, schedule["usedCoreNum"] + 1),
        core_limit,
    }:
        mutations.append(schedule.replace(usedCoreNum=cores))
    candidates = []
    seen = {schedule.signature()}
    for mutation in mutations:
        if mutation.signature() in seen:
            continue
        seen.add(mutation.signature())
        report = validate_schedule(workload, mutation, hardware)
        if not report.valid:
            continue
        candidates.append(
            Candidate(
                schedule=mutation,
                template=template_of(mutation),
                source=source,
                rationale="legal structural feedback mutation",
                parent_signatures=(parent,),
            )
        )
    return candidates


def feedback_mutations(
    workload: Workload,
    hardware: Hardware,
    observations: Sequence[MeasuredObservation],
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for observation in observations:
        if observation.workload.identity() != workload.identity():
            continue
        if _runtime_rejected(observation):
            # Runtime rejection is evidence for the executability model, not
            # a useful mutation centre. net_log19 rejected 48/50 schedules
            # generated around rejected parents, so expanding that
            # neighbourhood spends NPU budget without supplying a meaningful
            # counterfactual.
            continue
        elif observation.is_winner:
            source = "feedback_winner_mutation"
        elif observation.is_regression:
            source = "feedback_regression_counterfactual"
        else:
            continue
        candidates.extend(
            semantic_mutations(
                workload,
                hardware,
                observation.schedule,
                source=source,
                parent=observation.schedule.signature(),
            )
        )
    return candidates


def local_anchor_mutations(
    workload: Workload,
    hardware: Hardware,
    anchor: Schedule | None,
) -> list[Candidate]:
    if anchor is None:
        return []
    if not validate_schedule(workload, anchor, hardware).valid:
        return []
    return semantic_mutations(
        workload,
        hardware,
        anchor,
        source="local_bank_anchor",
        parent=anchor.signature(),
    )
