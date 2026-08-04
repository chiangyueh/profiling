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
from .bank_structure import (
    BankTransition,
    bank_transition,
    subsystem_mask_distance,
)
from .contracts import template_of
from .domain import (
    KNOWLEDGE_FIELDS,
    Candidate,
    Hardware,
    MeasuredObservation,
    Schedule,
    Template,
    Workload,
)


@dataclass(frozen=True)
class OneShotDecision:
    candidate: Candidate
    deployment_candidate: Candidate
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
class BankRelativeSafetyPrediction:
    samples: int
    rejected: int
    risk: float
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
class TemplateCalibrationEvidence:
    successful: int
    rejected: int
    winners: int
    regressions: int
    workload_support: int
    median_ratio: float

    @property
    def attempts(self) -> int:
        return self.successful + self.rejected

    @property
    def supported(self) -> bool:
        return (
            self.successful >= 3
            and self.workload_support >= 3
            and self.winners >= 2
            and self.median_ratio <= 0.98
        )

    @property
    def negative(self) -> bool:
        return (
            self.attempts >= 6
            and self.winners == 0
            and (
                self.regressions + self.rejected
            ) / self.attempts >= 0.50
        )


@dataclass(frozen=True)
class _EffectRow:
    observation: MeasuredObservation
    candidate_vector: BehaviorVector
    bank_vector: BehaviorVector
    changed_fields: frozenset[int]
    transition: BankTransition


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


def _bank_relative_distance(target: _EffectRow, row: _EffectRow) -> float:
    return (
        workload_distance(
            target.observation.workload,
            row.observation.workload,
        )
        + 0.45
        * behavior_distance(
            target.candidate_vector, row.candidate_vector
        )
        + 0.45
        * behavior_distance(target.bank_vector, row.bank_vector)
        + 0.60
        * _effect_mask_distance(
            target.changed_fields, row.changed_fields
        )
        + 1.20
        * subsystem_mask_distance(target.transition, row.transition)
        + 0.40 * _effect_vector_distance(target, row)
    )


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
                    transition=bank_transition(
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
            transition=bank_transition(bank, candidate),
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
            distance = _bank_relative_distance(target, row)
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


class BankRelativeSafetyModel:
    """Predict NPU rejection from candidate-to-bank structural effects."""

    def __init__(
        self,
        observations: Sequence[MeasuredObservation],
        hardware: Hardware,
    ) -> None:
        rows = []
        for observation in observations:
            if observation.bank_schedule is None:
                continue
            if (
                observation.source
                not in {"runtime_rejected", "runtime_verified"}
                and not observation.verified
            ):
                continue
            rows.append(
                _EffectRow(
                    observation=observation,
                    candidate_vector=behavior_vector(
                        observation.workload,
                        observation.schedule,
                        hardware,
                    ),
                    bank_vector=behavior_vector(
                        observation.workload,
                        observation.bank_schedule,
                        hardware,
                    ),
                    changed_fields=_changed_fields(
                        observation.bank_schedule,
                        observation.schedule,
                    ),
                    transition=bank_transition(
                        observation.bank_schedule,
                        observation.schedule,
                    ),
                )
            )
        self.rows = tuple(rows)
        self.rejected_rows = sum(
            row.observation.source == "runtime_rejected"
            for row in rows
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
    ) -> BankRelativeSafetyPrediction:
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
            candidate_vector=behavior_vector(
                workload, candidate, hardware
            ),
            bank_vector=behavior_vector(workload, bank, hardware),
            changed_fields=_changed_fields(bank, candidate),
            transition=bank_transition(bank, candidate),
        )
        bank_template = template_of(bank)
        candidate_template = template_of(candidate)
        nearest_by_workload = {}
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
            distance = _bank_relative_distance(target, row)
            existing = nearest_by_workload.get(identity)
            if existing is None or distance < existing[0]:
                nearest_by_workload[identity] = (
                    distance,
                    observation.source == "runtime_rejected",
                )

        nearest = sorted(nearest_by_workload.values())[:8]
        if not nearest:
            return BankRelativeSafetyPrediction(
                0, 0, 0.5, math.inf, 0.0
            )
        weights = [
            math.exp(-distance / 1.25)
            for distance, _ in nearest
        ]
        weight_sum = sum(weights)
        risk = sum(
            weight * float(rejected)
            for weight, (_, rejected) in zip(weights, nearest)
        ) / max(1.0e-9, weight_sum)
        nearest_distance = nearest[0][0]
        support = (
            min(1.0, len(nearest) / 6.0)
            * math.exp(-nearest_distance / 1.5)
        )
        return BankRelativeSafetyPrediction(
            samples=len(nearest),
            rejected=sum(rejected for _, rejected in nearest),
            risk=risk,
            nearest_distance=nearest_distance,
            support=support,
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


def _is_bank_relative_runtime_safe(
    prediction: BankRelativeSafetyPrediction,
) -> bool:
    return not (
        prediction.support >= 0.15
        and prediction.risk >= 0.35
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


def _template_calibration_evidence(
    workload: Workload,
    incumbent: Candidate,
    observations: Sequence[MeasuredObservation],
    templates: set[Template],
) -> dict[Template, TemplateCalibrationEvidence]:
    """Summarize paired template switches without shape-family gates."""

    samples = {
        template: {
            "ratios": [],
            "rejected": 0,
            "workloads": set(),
        }
        for template in templates
    }
    for observation in observations:
        candidate_template = template_of(observation.schedule)
        if candidate_template not in samples:
            continue
        if (
            observation.workload.dtype != workload.dtype
            or observation.workload.trans_a != workload.trans_a
            or observation.workload.trans_b != workload.trans_b
            or observation.bank_schedule is None
            or template_of(observation.bank_schedule)
            != incumbent.template
            or candidate_template
            == template_of(observation.bank_schedule)
        ):
            continue
        item = samples[candidate_template]
        if observation.source == "runtime_rejected":
            item["workloads"].add(observation.workload.identity())
            item["rejected"] += 1
            continue
        if not observation.verified:
            continue
        item["workloads"].add(observation.workload.identity())
        item["ratios"].append(observation.measured_ratio)

    result = {}
    for template, item in samples.items():
        ratios = tuple(item["ratios"])
        result[template] = TemplateCalibrationEvidence(
            successful=len(ratios),
            rejected=item["rejected"],
            winners=sum(ratio <= 0.99 for ratio in ratios),
            regressions=sum(ratio >= 1.01 for ratio in ratios),
            workload_support=len(item["workloads"]),
            median_ratio=median(ratios) if ratios else 1.0,
        )
    return result


def _cross_template_budget(
    budget: int,
    evidence: dict[Template, TemplateCalibrationEvidence],
) -> int:
    if not evidence or budget <= 1:
        return 0
    if any(item.supported for item in evidence.values()):
        return min(budget // 3, max(2, round(budget * 0.25)))
    attempted = [item for item in evidence.values() if item.attempts]
    if attempted and all(item.negative for item in attempted):
        return 1
    return max(1, min(budget // 4, round(budget * 0.20)))


def _template_evidence_order(item) -> tuple:
    template, evidence = item
    state = 0 if evidence.supported else (2 if evidence.negative else 1)
    reject_rate = evidence.rejected / max(1, evidence.attempts)
    return (
        state,
        reject_rate,
        evidence.median_ratio,
        -evidence.workload_support,
        template.value,
    )


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
    cross_evidence = _template_calibration_evidence(
        workload,
        incumbent,
        observations,
        {candidate.template for candidate in cross_template},
    )
    cross_budget = min(
        len(cross_template),
        _cross_template_budget(budget, cross_evidence),
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
        templates = [
            template
            for template, _ in sorted(
                cross_evidence.items(),
                key=_template_evidence_order,
            )
        ]
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
        template_evidence = cross_evidence.get(candidate.template)
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
                "template_evidence_attempts": float(
                    template_evidence.attempts
                    if template_evidence is not None
                    else 0
                ),
                "template_evidence_winners": float(
                    template_evidence.winners
                    if template_evidence is not None
                    else 0
                ),
                "template_evidence_regressions": float(
                    template_evidence.regressions
                    if template_evidence is not None
                    else 0
                ),
                "template_evidence_rejected": float(
                    template_evidence.rejected
                    if template_evidence is not None
                    else 0
                ),
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
    safety_model: BankRelativeSafetyModel | None = None,
    **_ignored,
) -> OneShotDecision:
    """Select exactly one custom tiling using bank-relative evidence.

    The bank is an incumbent and paired measurement control, not a normal
    deployment fallback. A candidate must pass the runtime-safety policy.
    Strong latency evidence changes the confidence tier; it does not decide
    whether the custom search is used at all.
    """

    if incumbent.source != "bank_incumbent":
        raise ValueError("one-shot incumbent must be the bank control")
    runtime_model = cost_model or FeedbackCostModel(observations, hardware)
    relative_model = effect_model or BankRelativeEffectModel(
        observations, hardware
    )
    relative_safety_model = safety_model or BankRelativeSafetyModel(
        observations, hardware
    )
    unique: dict[tuple[int, ...], Candidate] = {}
    for candidate in candidates:
        if candidate.source not in {
            "contract_coupled_policy",
            "contract_global",
            "feedback_regression_counterfactual",
            "feedback_winner_mutation",
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
        relative_safety = relative_safety_model.predict(
            workload,
            incumbent.schedule,
            candidate.schedule,
            hardware,
            exclude_workload=workload.identity(),
        )
        changed = _changed_fields(incumbent.schedule, candidate.schedule)
        transition = bank_transition(
            incumbent.schedule, candidate.schedule
        )
        cross_template = candidate.template != incumbent.template
        runtime_safe = (
            _is_bank_relative_runtime_safe(relative_safety)
            if relative_safety.support >= 0.15
            else _is_runtime_safe(runtime)
        )
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
                "bank_changed_subsystems": float(
                    len(transition.changed_subsystems)
                ),
                "bank_execution_structure_changes": float(
                    len(transition.execution_subsystems)
                ),
                "bank_structure_preserved": float(
                    transition.preserves_execution_structure
                ),
                "bank_transition_risk_tier": float(
                    transition.risk_tier
                ),
                "runtime_risk_score": runtime.runtime_risk_score,
                "runtime_risk_support": runtime.runtime_risk_support,
                "bank_runtime_risk": relative_safety.risk,
                "bank_runtime_risk_support": relative_safety.support,
                "bank_runtime_risk_samples": float(
                    relative_safety.samples
                ),
            }
        )
        evaluated.append(
            (
                candidate,
                vector,
                relative,
                runtime,
                relative_safety,
                runtime_safe,
                deployable,
            )
        )

    strong_evidence = [item for item in evaluated if item[6]]
    eligible = [item for item in evaluated if item[5]]
    runtime_safe_count = sum(item[5] for item in evaluated)
    direct_base_count = sum(
        item[0].template.value == "BASE"
        and item[0].schedule["singleCoreM"] == item[0].schedule["baseM"]
        and item[0].schedule["singleCoreN"] == item[0].schedule["baseN"]
        for item in evaluated
    )
    local_count = sum(
        item[0].source == "local_bank_anchor" for item in evaluated
    )

    if not evaluated:
        raise ValueError(
            "one-shot search produced no independent custom candidate; "
            "bank fallback is disabled"
        )
    if not eligible:
        raise ValueError(
            "one-shot search produced no runtime-safe custom candidate; "
            "bank fallback is disabled"
        )

    def deployment_score(item) -> tuple:
        candidate, _, prediction, _, _, _, _ = item
        transition = bank_transition(
            incumbent.schedule, candidate.schedule
        )
        cross_template_penalty = (
            0.02 if candidate.template != incumbent.template else 0.0
        )
        return (
            prediction.upper_ratio + cross_template_penalty,
            prediction.robust_ratio,
            transition.risk_tier,
            candidate.schedule.signature(),
        )

    def research_score(item) -> tuple:
        (
            candidate,
            vector,
            prediction,
            runtime,
            relative_safety,
            runtime_safe,
            _,
        ) = item
        risk = (
            relative_safety.risk * relative_safety.support
            if relative_safety.support >= 0.15
            else (
                runtime.runtime_risk_score
                * runtime.runtime_risk_support
            )
        )
        predicted = prediction.robust_ratio
        uncertainty = (
            0.12 * (1.0 - prediction.support)
            + 0.03 * min(3.0, prediction.nearest_distance)
            if prediction.samples
            else 0.18
        )
        cross_template = candidate.template != incumbent.template
        transition = bank_transition(
            incumbent.schedule, candidate.schedule
        )
        changed_fields = len(transition.changed_fields)
        structural_penalty = min(
            0.03, 0.004 * max(0, changed_fields - 1)
        )
        return (
            transition.risk_tier,
            cross_template and prediction.samples < 3,
            predicted
            + uncertainty
            + 0.20 * risk
            + 0.03 * cross_template
            + structural_penalty,
            predicted,
            changed_fields,
            vector.metrics.get("analytical_prior", math.inf),
            candidate.schedule.signature(),
        )

    if strong_evidence:
        (
            original,
            vector,
            prediction,
            _,
            _,
            _,
            _,
        ) = min(strong_evidence, key=deployment_score)
        acquisition = deployment_score(
            (original, vector, prediction, None, None, True, True)
        )[0]
        deployment_candidate = original
        source = "one_shot_bank_relative"
        rationale = (
            "single deployment decision supported by paired "
            f"candidate/control effects; generator={original.source}"
        )
        selection_policy = "paired_control_relative_effect"
        deployment_recommended = 1.0
    else:
        (
            original,
            vector,
            prediction,
            runtime,
            relative_safety,
            _,
            _,
        ) = min(eligible, key=research_score)
        acquisition = research_score(
            (
                original,
                vector,
                prediction,
                runtime,
                relative_safety,
                (
                    _is_bank_relative_runtime_safe(relative_safety)
                    if relative_safety.support >= 0.15
                    else _is_runtime_safe(runtime)
                ),
                True,
            )
        )[2]
        deployment_candidate = original
        source = "one_shot_custom_policy"
        rationale = (
            "single runtime-safe custom deployment chosen by expected "
            "bank-relative latency, uncertainty, structural distance, "
            f"and rejection risk; generator={original.source}"
        )
        selection_policy = "risk_bounded_custom_first"
        deployment_recommended = 1.0

    vector.metrics.update(
        {
            "one_shot_score": acquisition,
            "one_shot_incumbent_fallback": 0.0,
            "deployment_recommended_custom": deployment_recommended,
            "deployment_evidence_strong": float(bool(strong_evidence)),
        }
    )
    selected = Candidate(
        schedule=original.schedule,
        template=original.template,
        source=source,
        rationale=rationale,
        acquisition=acquisition,
        parent_signatures=(incumbent.schedule.signature(),),
        behavior_key=behavior_key(vector),
        metrics=dict(vector.metrics),
    )
    return OneShotDecision(
        candidate=selected,
        deployment_candidate=deployment_candidate,
        evaluated=len(evaluated),
        safe_candidates=runtime_safe_count,
        direct_base_candidates=direct_base_count,
        transfer_eligible_candidates=sum(
            item[5] and item[0].template != incumbent.template
            for item in eligible
        ),
        custom_eligible_candidates=len(eligible),
        local_candidates=local_count,
        selection_policy=selection_policy,
    )
