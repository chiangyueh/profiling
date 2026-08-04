from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Iterable, Sequence

from .behavior import (
    BehaviorVector,
    FeedbackCostModel,
    behavior_distance,
    behavior_key,
    behavior_vector,
    workload_distance,
)
from .contracts import template_of
from .domain import (
    KNOWLEDGE_FIELDS,
    Candidate,
    Hardware,
    MeasuredObservation,
    Schedule,
    Workload,
)


@dataclass(frozen=True)
class OneShotDecision:
    candidate: Candidate
    evaluated: int
    safe_candidates: int
    direct_base_candidates: int
    transfer_eligible_candidates: int
    custom_eligible_candidates: int
    local_candidates: int
    selection_policy: str


@dataclass(frozen=True)
class BankRelativePrediction:
    samples: int
    robust_ratio: float
    upper_ratio: float
    nearest_distance: float
    support: float


@dataclass(frozen=True)
class BankRelativeValidation:
    groups: int
    oracle_opportunities: int
    custom_selections: int
    custom_winners: int
    custom_regressions: int
    severe_regressions: int
    median_selected_ratio: float
    p90_selected_ratio: float


@dataclass(frozen=True)
class _EffectRow:
    observation: MeasuredObservation
    candidate_vector: BehaviorVector
    bank_vector: BehaviorVector
    changed_fields: frozenset[int]


def _changed_fields(
    bank: Schedule,
    candidate: Schedule,
) -> frozenset[int]:
    return frozenset(
        index
        for index, (bank_value, candidate_value) in enumerate(
            zip(bank.values, candidate.values)
        )
        if bank_value != candidate_value
    )


def _effect_mask_distance(
    left: frozenset[int],
    right: frozenset[int],
) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left ^ right) / len(union)


def _effect_vector_distance(left: _EffectRow, right: _EffectRow) -> float:
    left_delta = tuple(
        candidate - bank
        for candidate, bank in zip(
            left.candidate_vector.values,
            left.bank_vector.values,
        )
    )
    right_delta = tuple(
        candidate - bank
        for candidate, bank in zip(
            right.candidate_vector.values,
            right.bank_vector.values,
        )
    )
    return sum(
        abs(left_value - right_value)
        for left_value, right_value in zip(left_delta, right_delta)
    ) / max(1, len(left_delta))


class BankRelativeEffectModel:
    """Predict candidate/bank latency from paired, verified effects.

    The model never treats an absolute candidate latency from another
    workload as a deployment target. It compares the candidate and its bank
    incumbent as a pair and requires independent workload support.
    """

    def __init__(
        self,
        observations: Sequence[MeasuredObservation],
        hardware: Hardware,
    ) -> None:
        grouped: dict[
            tuple[
                tuple[int, int, int, str, bool, bool, int],
                str,
                tuple[int, ...],
            ],
            list[MeasuredObservation],
        ] = defaultdict(list)
        for observation in observations:
            if (
                observation.source == "runtime_rejected"
                or not observation.verified
                or observation.bank_schedule is None
            ):
                continue
            grouped[
                (
                    observation.workload.identity(),
                    observation.record_id,
                    observation.schedule.signature(),
                )
            ].append(observation)

        rows = []
        for records in grouped.values():
            best_trust = max(record.structured_verified for record in records)
            trusted = [
                record
                for record in records
                if record.structured_verified == best_trust
            ]
            representative = trusted[0]
            ratio = median(record.measured_ratio for record in trusted)
            observation = MeasuredObservation(
                workload=representative.workload,
                schedule=representative.schedule,
                ratio_vs_official=ratio,
                ratio_vs_bank=ratio,
                source=representative.source,
                record_id=representative.record_id,
                status_vs_official=representative.status_vs_official,
                status_vs_bank=representative.status_vs_bank,
                verified=True,
                structured_verified=bool(best_trust),
                bank_schedule=representative.bank_schedule,
            )
            bank_vector = behavior_vector(
                observation.workload,
                observation.bank_schedule,
                hardware,
            )
            candidate_vector = behavior_vector(
                observation.workload,
                observation.schedule,
                hardware,
            )
            rows.append(
                _EffectRow(
                    observation=observation,
                    candidate_vector=candidate_vector,
                    bank_vector=bank_vector,
                    changed_fields=_changed_fields(
                        observation.bank_schedule,
                        observation.schedule,
                    ),
                )
            )
        self.rows = tuple(rows)
        self.workloads = len(
            {row.observation.workload.identity() for row in rows}
        )
        self.structured_rows = sum(
            row.observation.structured_verified for row in rows
        )

    def predict(
        self,
        workload: Workload,
        bank: Schedule,
        candidate: Schedule,
        hardware: Hardware,
        *,
        exclude_workload: (
            tuple[int, int, int, str, bool, bool, int] | None
        ) = None,
    ) -> BankRelativePrediction:
        target_candidate_vector = behavior_vector(
            workload, candidate, hardware
        )
        target_bank_vector = behavior_vector(workload, bank, hardware)
        target = _EffectRow(
            observation=MeasuredObservation(
                workload=workload,
                schedule=candidate,
                ratio_vs_official=1.0,
                ratio_vs_bank=1.0,
                source="target",
                record_id="target",
                bank_schedule=bank,
            ),
            candidate_vector=target_candidate_vector,
            bank_vector=target_bank_vector,
            changed_fields=_changed_fields(bank, candidate),
        )
        bank_template = template_of(bank)
        candidate_template = template_of(candidate)
        nearest_by_workload: dict[
            tuple[int, int, int, str, bool, bool, int],
            tuple[float, float],
        ] = {}
        for row in self.rows:
            observation = row.observation
            identity = observation.workload.identity()
            if identity == exclude_workload:
                continue
            if (
                observation.workload.dtype != workload.dtype
                or observation.workload.trans_a != workload.trans_a
                or observation.workload.trans_b != workload.trans_b
                or template_of(observation.bank_schedule) != bank_template
                or template_of(observation.schedule) != candidate_template
            ):
                continue
            distance = (
                workload_distance(workload, observation.workload)
                + 0.45
                * behavior_distance(
                    target_candidate_vector, row.candidate_vector
                )
                + 0.45
                * behavior_distance(target_bank_vector, row.bank_vector)
                + 0.60 * _effect_mask_distance(
                    target.changed_fields, row.changed_fields
                )
                + 0.40 * _effect_vector_distance(target, row)
            )
            existing = nearest_by_workload.get(identity)
            if existing is None or distance < existing[0]:
                nearest_by_workload[identity] = (
                    distance,
                    observation.measured_ratio,
                )

        nearest = sorted(nearest_by_workload.values())[:8]
        if not nearest:
            return BankRelativePrediction(0, 1.0, math.inf, math.inf, 0.0)
        ratios = sorted(ratio for _, ratio in nearest)
        nearest_distance = nearest[0][0]
        sample_support = min(1.0, len(nearest) / 6.0)
        distance_support = math.exp(-nearest_distance / 1.5)
        return BankRelativePrediction(
            samples=len(nearest),
            robust_ratio=median(ratios),
            upper_ratio=max(ratios),
            nearest_distance=nearest_distance,
            support=sample_support * distance_support,
        )


def _effect_is_deployable(
    prediction: BankRelativePrediction,
    *,
    cross_template: bool,
) -> bool:
    if cross_template:
        return (
            prediction.samples >= 3
            and prediction.robust_ratio <= 0.94
            and prediction.upper_ratio <= 0.96
            and prediction.nearest_distance <= 1.25
            and prediction.support >= 0.20
        )
    return (
        prediction.samples >= 3
        and prediction.robust_ratio <= 0.98
        and prediction.upper_ratio <= 0.99
        and prediction.nearest_distance <= 1.50
        and prediction.support >= 0.15
    )


def validate_bank_relative_selector(
    observations: Sequence[MeasuredObservation],
    hardware: Hardware,
) -> BankRelativeValidation:
    model = BankRelativeEffectModel(observations, hardware)
    groups: dict[
        tuple[
            tuple[int, int, int, str, bool, bool, int],
            str,
        ],
        list[MeasuredObservation],
    ] = defaultdict(list)
    for row in model.rows:
        observation = row.observation
        groups[
            (observation.workload.identity(), observation.record_id)
        ].append(observation)

    selected_ratios = []
    custom_ratios = []
    oracle_opportunities = 0
    for (identity, _), records in groups.items():
        oracle_opportunities += min(
            record.measured_ratio for record in records
        ) <= 0.99
        ranked = []
        for record in records:
            prediction = model.predict(
                record.workload,
                record.bank_schedule,
                record.schedule,
                hardware,
                exclude_workload=identity,
            )
            if not _effect_is_deployable(
                prediction,
                cross_template=(
                    template_of(record.bank_schedule)
                    != template_of(record.schedule)
                ),
            ):
                continue
            ranked.append(
                (
                    prediction.upper_ratio,
                    prediction.robust_ratio,
                    record.schedule.signature(),
                    record.measured_ratio,
                )
            )
        if ranked:
            measured = min(ranked)[3]
            custom_ratios.append(measured)
            selected_ratios.append(measured)
        else:
            selected_ratios.append(1.0)

    ordered = sorted(selected_ratios) or [1.0]
    return BankRelativeValidation(
        groups=len(groups),
        oracle_opportunities=oracle_opportunities,
        custom_selections=len(custom_ratios),
        custom_winners=sum(ratio <= 0.99 for ratio in custom_ratios),
        custom_regressions=sum(ratio >= 1.01 for ratio in custom_ratios),
        severe_regressions=sum(ratio >= 1.10 for ratio in custom_ratios),
        median_selected_ratio=median(ordered),
        p90_selected_ratio=ordered[
            int(0.90 * (len(ordered) - 1))
        ],
    )


def _is_runtime_safe(
    prediction,
) -> bool:
    return not (
        prediction.runtime_risk_support >= 0.10
        and prediction.runtime_risk_score >= 0.35
    )


def _farthest_first(
    workload: Workload,
    candidates: Sequence[Candidate],
    bank: Schedule,
    hardware: Hardware,
    budget: int,
) -> list[Candidate]:
    remaining = list(candidates)
    selected: list[Candidate] = []
    bank_vector = behavior_vector(workload, bank, hardware)
    selected_vectors = [bank_vector]
    while remaining and len(selected) < budget:
        scored = []
        for candidate in remaining:
            vector = behavior_vector(workload, candidate.schedule, hardware)
            novelty = min(
                behavior_distance(vector, existing)
                for existing in selected_vectors
            )
            changed = len(_changed_fields(bank, candidate.schedule))
            scored.append(
                (
                    -novelty,
                    changed,
                    candidate.schedule.signature(),
                    candidate,
                    vector,
                )
            )
        _, _, _, chosen, vector = min(scored)
        selected.append(chosen)
        selected_vectors.append(vector)
        remaining = [
            candidate
            for candidate in remaining
            if candidate.schedule.signature()
            != chosen.schedule.signature()
        ]
    return selected


def select_calibration_candidates(
    workload: Workload,
    candidates: Iterable[Candidate],
    incumbent: Candidate,
    observations: Sequence[MeasuredObservation],
    hardware: Hardware,
    budget: int,
) -> list[Candidate]:
    """Choose controlled bank-relative effects for one calibration run."""

    model = FeedbackCostModel(observations, hardware)
    rated: list[tuple[float, Candidate]] = []
    seen: set[tuple[int, ...]] = set()
    for candidate in candidates:
        if (
            candidate.schedule.signature()
            == incumbent.schedule.signature()
            or candidate.source
            not in {"local_bank_anchor", "contract_coupled_policy"}
        ):
            continue
        signature = candidate.schedule.signature()
        if signature in seen:
            continue
        seen.add(signature)
        vector = behavior_vector(workload, candidate.schedule, hardware)
        runtime = model.predict(
            workload,
            vector,
            exclude_workload=workload.identity(),
        )
        risk = (
            runtime.runtime_risk_support * runtime.runtime_risk_score
        )
        rated.append((risk, candidate))

    # Calibration is where the runtime-risk model itself is corrected.
    # Keep the least risky broad slice, but never let an uncertain model erase
    # the entire exact-callback-accepted experiment.
    retained = []
    retained_per_stratum = max(budget * 2, budget)
    strata = sorted(
        {
            (candidate.source, candidate.template)
            for _, candidate in rated
        },
        key=lambda item: (item[0], item[1].value),
    )
    for source, template in strata:
        retained.extend(
            candidate
            for _, candidate in sorted(
                (
                    item
                    for item in rated
                    if (
                        item[1].source == source
                        and item[1].template == template
                    )
                ),
                key=lambda item: (
                    item[0],
                    item[1].schedule.signature(),
                ),
            )[:retained_per_stratum]
        )
    unique = {
        candidate.schedule.signature(): candidate
        for candidate in retained
    }

    local = [
        candidate
        for candidate in unique.values()
        if (
            candidate.source == "local_bank_anchor"
            and candidate.template == incumbent.template
        )
    ]
    coupled = [
        candidate
        for candidate in unique.values()
        if (
            candidate.source == "contract_coupled_policy"
            and candidate.template == incumbent.template
        )
    ]
    cross_template = [
        candidate
        for candidate in unique.values()
        if candidate.template != incumbent.template
    ]
    cross_budget = min(
        len(cross_template),
        max(1, budget // 3) if cross_template else 0,
    )
    same_template_budget = budget - cross_budget
    local_budget = min(
        len(local), max(1, same_template_budget // 2)
    )
    selected = _farthest_first(
        workload,
        local,
        incumbent.schedule,
        hardware,
        local_budget,
    )
    selected_signatures = {
        candidate.schedule.signature() for candidate in selected
    }
    selected.extend(
        _farthest_first(
            workload,
            [
                candidate
                for candidate in coupled
                if candidate.schedule.signature()
                not in selected_signatures
            ],
            incumbent.schedule,
            hardware,
            same_template_budget - len(selected),
        )
    )
    if len(selected) < same_template_budget:
        selected_signatures = {
            candidate.schedule.signature() for candidate in selected
        }
        selected.extend(
            _farthest_first(
                workload,
                [
                    candidate
                    for candidate in unique.values()
                    if (
                        candidate.template == incumbent.template
                        and candidate.schedule.signature()
                        not in selected_signatures
                    )
                ],
                incumbent.schedule,
                hardware,
                same_template_budget - len(selected),
            )
        )
    if cross_budget:
        cross_groups = {
            template: [
                candidate
                for candidate in cross_template
                if candidate.template == template
            ]
            for template in sorted(
                {candidate.template for candidate in cross_template},
                key=lambda item: item.value,
            )
        }
        cross_quotas = {template: 0 for template in cross_groups}
        templates = list(cross_groups)
        for index in range(cross_budget):
            cross_quotas[templates[index % len(templates)]] += 1
        for template in templates:
            selected.extend(
                _farthest_first(
                    workload,
                    cross_groups[template],
                    incumbent.schedule,
                    hardware,
                    cross_quotas[template],
                )
            )
    if len(selected) < budget:
        selected_signatures = {
            candidate.schedule.signature() for candidate in selected
        }
        selected.extend(
            _farthest_first(
                workload,
                [
                    candidate
                    for candidate in unique.values()
                    if candidate.schedule.signature()
                    not in selected_signatures
                ],
                incumbent.schedule,
                hardware,
                budget - len(selected),
            )
        )

    result = []
    for candidate in selected[:budget]:
        vector = behavior_vector(workload, candidate.schedule, hardware)
        runtime = model.predict(
            workload,
            vector,
            exclude_workload=workload.identity(),
        )
        changed = _changed_fields(incumbent.schedule, candidate.schedule)
        template_switch = candidate.template != incumbent.template
        vector.metrics.update(
            {
                "bank_changed_fields": float(len(changed)),
                "bank_template_switch": float(template_switch),
                "bank_behavior_distance": behavior_distance(
                    vector,
                    behavior_vector(
                        workload, incumbent.schedule, hardware
                    ),
                ),
                "runtime_risk_score": runtime.runtime_risk_score,
                "runtime_risk_support": runtime.runtime_risk_support,
            }
        )
        result.append(
            Candidate(
                schedule=candidate.schedule,
                template=candidate.template,
                source=(
                    "calibration_template_probe"
                    if template_switch
                    else (
                        "calibration_local_counterfactual"
                        if candidate.source == "local_bank_anchor"
                        else "calibration_coupled_counterfactual"
                    )
                ),
                rationale=(
                    "paired bank-relative calibration effect; "
                    f"generator={candidate.source}; changed="
                    + ",".join(
                        KNOWLEDGE_FIELDS[index] for index in sorted(changed)
                    )
                ),
                acquisition=-vector.metrics["bank_behavior_distance"],
                parent_signatures=(incumbent.schedule.signature(),),
                behavior_key=behavior_key(vector),
                metrics=dict(vector.metrics),
            )
        )
    return result


def select_one_shot_candidate(
    workload: Workload,
    candidates: Iterable[Candidate],
    incumbent: Candidate,
    observations: Sequence[MeasuredObservation],
    hardware: Hardware,
    *,
    cost_model: FeedbackCostModel | None = None,
    effect_model: BankRelativeEffectModel | None = None,
    **_ignored,
) -> OneShotDecision:
    """Select one tiling with a measured bank-relative safety criterion."""

    if incumbent.source != "bank_incumbent":
        raise ValueError("one-shot incumbent must be the bank control")
    runtime_model = cost_model or FeedbackCostModel(observations, hardware)
    relative_model = effect_model or BankRelativeEffectModel(
        observations, hardware
    )
    unique: dict[tuple[int, ...], Candidate] = {}
    for candidate in candidates:
        if candidate.source not in {
            "contract_coupled_policy",
            "local_bank_anchor",
        }:
            continue
        if candidate.schedule.signature() == incumbent.schedule.signature():
            continue
        unique.setdefault(candidate.schedule.signature(), candidate)

    evaluated = []
    for candidate in unique.values():
        vector = behavior_vector(workload, candidate.schedule, hardware)
        runtime = runtime_model.predict(
            workload,
            vector,
            exclude_workload=workload.identity(),
        )
        relative = relative_model.predict(
            workload,
            incumbent.schedule,
            candidate.schedule,
            hardware,
            exclude_workload=workload.identity(),
        )
        changed = _changed_fields(incumbent.schedule, candidate.schedule)
        cross_template = candidate.template != incumbent.template
        runtime_safe = _is_runtime_safe(runtime)
        deployable = runtime_safe and _effect_is_deployable(
            relative,
            cross_template=cross_template,
        )
        vector.metrics.update(
            {
                "predicted_latency_ratio": relative.robust_ratio,
                "bank_relative_upper_ratio": relative.upper_ratio,
                "bank_relative_samples": float(relative.samples),
                "bank_relative_nearest_distance": (
                    relative.nearest_distance
                ),
                "bank_relative_support": relative.support,
                "bank_changed_fields": float(len(changed)),
                "runtime_risk_score": runtime.runtime_risk_score,
                "runtime_risk_support": runtime.runtime_risk_support,
            }
        )
        evaluated.append(
            (candidate, vector, relative, runtime_safe, deployable)
        )

    eligible = [item for item in evaluated if item[4]]
    runtime_safe_count = sum(item[3] for item in evaluated)
    direct_base_count = sum(
        item[0].template.value == "BASE"
        and item[0].schedule["singleCoreM"] == item[0].schedule["baseM"]
        and item[0].schedule["singleCoreN"] == item[0].schedule["baseN"]
        for item in evaluated
    )
    local_count = sum(
        item[0].source == "local_bank_anchor" for item in evaluated
    )

    if not eligible:
        vector = behavior_vector(
            workload, incumbent.schedule, hardware
        )
        vector.metrics.update(
            {
                "predicted_latency_ratio": 1.0,
                "bank_relative_upper_ratio": 1.0,
                "bank_relative_samples": 0.0,
                "bank_relative_nearest_distance": math.inf,
                "bank_relative_support": 0.0,
                "bank_changed_fields": 0.0,
                "runtime_risk_score": 0.0,
                "runtime_risk_support": 1.0,
                "one_shot_incumbent_fallback": 1.0,
            }
        )
        selected = Candidate(
            schedule=incumbent.schedule,
            template=incumbent.template,
            source="one_shot_bank_incumbent",
            rationale=(
                "no custom candidate has leave-workload-out paired "
                "evidence sufficient to beat the bank safely"
            ),
            acquisition=0.0,
            parent_signatures=(incumbent.schedule.signature(),),
            behavior_key=behavior_key(vector),
            metrics=dict(vector.metrics),
        )
        return OneShotDecision(
            candidate=selected,
            evaluated=len(evaluated),
            safe_candidates=runtime_safe_count,
            direct_base_candidates=direct_base_count,
            transfer_eligible_candidates=0,
            custom_eligible_candidates=0,
            local_candidates=local_count,
            selection_policy="bank_incumbent",
        )

    def score(item) -> tuple[float, float, tuple[int, ...]]:
        candidate, _, prediction, _, _ = item
        cross_template_penalty = (
            0.02 if candidate.template != incumbent.template else 0.0
        )
        return (
            prediction.upper_ratio + cross_template_penalty,
            prediction.robust_ratio,
            candidate.schedule.signature(),
        )

    original, vector, prediction, _, _ = min(eligible, key=score)
    acquisition = score(
        (original, vector, prediction, True, True)
    )[0]
    vector.metrics.update(
        {
            "one_shot_score": acquisition,
            "one_shot_incumbent_fallback": 0.0,
        }
    )
    selected = Candidate(
        schedule=original.schedule,
        template=original.template,
        source="one_shot_bank_relative",
        rationale=(
            "single deployment decision supported by paired "
            f"candidate/bank effects; generator={original.source}"
        ),
        acquisition=acquisition,
        parent_signatures=(incumbent.schedule.signature(),),
        behavior_key=behavior_key(vector),
        metrics=dict(vector.metrics),
    )
    return OneShotDecision(
        candidate=selected,
        evaluated=len(evaluated),
        safe_candidates=runtime_safe_count,
        direct_base_candidates=direct_base_count,
        transfer_eligible_candidates=sum(
            item[4] and item[0].template != incumbent.template
            for item in evaluated
        ),
        custom_eligible_candidates=len(eligible),
        local_candidates=local_count,
        selection_policy="paired_bank_relative_effect",
    )
