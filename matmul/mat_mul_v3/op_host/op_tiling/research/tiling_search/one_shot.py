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
    INPUT_BYTES,
    KNOWLEDGE_FIELDS,
    Candidate,
    Hardware,
    MeasuredObservation,
    Template,
    Workload,
)
from .ranking import PairwiseLatencyRanker


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


def _normalized_priors(
    values: Sequence[float],
) -> list[float]:
    logs = [math.log1p(max(0.0, value)) for value in values]
    low = min(logs)
    span = max(1.0e-9, max(logs) - low)
    return [(value - low) / span for value in logs]


@dataclass(frozen=True)
class _TransferEvidence:
    samples: int
    winners: int
    robust_ratio: float
    upper_ratio: float
    nearest_distance: float


@dataclass(frozen=True)
class _CounterfactualPair:
    workload: Workload
    template: Template
    field: str
    low_value: int
    high_value: int
    high_over_low: float


@dataclass(frozen=True)
class CounterfactualEvidence:
    samples: int
    robust_ratio: float
    upper_ratio: float
    nearest_distance: float


class CounterfactualPolicyModel:
    """Learn one-field schedule effects from paired NPU observations."""

    def __init__(
        self,
        observations: Sequence[MeasuredObservation],
    ) -> None:
        grouped: dict[
            tuple[
                tuple[int, int, int, str, bool, bool, int],
                str,
            ],
            dict[tuple[int, ...], list[MeasuredObservation]],
        ] = defaultdict(lambda: defaultdict(list))
        for observation in observations:
            if (
                observation.source == "runtime_rejected"
                or not observation.structured_verified
            ):
                continue
            grouped[
                (
                    observation.workload.identity(),
                    observation.record_id,
                )
            ][
                observation.schedule.signature()
            ].append(observation)

        pairs = []
        for signatures in grouped.values():
            representatives = []
            for items in signatures.values():
                representatives.append(
                    (
                        items[0],
                        median(item.measured_ratio for item in items),
                    )
                )
            for index, (left, left_ratio) in enumerate(representatives):
                for right, right_ratio in representatives[index + 1 :]:
                    left_template = template_of(left.schedule)
                    if left_template != template_of(right.schedule):
                        continue
                    changed = [
                        field_index
                        for field_index, (left_value, right_value) in enumerate(
                            zip(
                                left.schedule.values,
                                right.schedule.values,
                            )
                        )
                        if left_value != right_value
                    ]
                    if len(changed) != 1:
                        continue
                    field_index = changed[0]
                    if (
                        left.schedule.values[field_index]
                        < right.schedule.values[field_index]
                    ):
                        low, low_ratio = left, left_ratio
                        high, high_ratio = right, right_ratio
                    else:
                        low, low_ratio = right, right_ratio
                        high, high_ratio = left, left_ratio
                    pairs.append(
                        _CounterfactualPair(
                            workload=left.workload,
                            template=left_template,
                            field=KNOWLEDGE_FIELDS[field_index],
                            low_value=low.schedule.values[field_index],
                            high_value=high.schedule.values[field_index],
                            high_over_low=high_ratio / low_ratio,
                        )
                    )
        self.pairs = tuple(pairs)

    def predict(
        self,
        workload: Workload,
        incumbent: Candidate,
        candidate: Candidate,
    ) -> CounterfactualEvidence:
        changed = [
            index
            for index, (incumbent_value, candidate_value) in enumerate(
                zip(
                    incumbent.schedule.values,
                    candidate.schedule.values,
                )
            )
            if incumbent_value != candidate_value
        ]
        if (
            len(changed) != 1
            or incumbent.template != candidate.template
        ):
            return CounterfactualEvidence(0, 1.0, 1.0, math.inf)
        field_index = changed[0]
        field = KNOWLEDGE_FIELDS[field_index]
        incumbent_value = incumbent.schedule.values[field_index]
        candidate_value = candidate.schedule.values[field_index]
        target_low = min(incumbent_value, candidate_value)
        target_high = max(incumbent_value, candidate_value)
        neighbors = []
        for pair in self.pairs:
            if (
                pair.template != candidate.template
                or pair.workload.dtype != workload.dtype
                or pair.workload.trans_a != workload.trans_a
                or pair.workload.trans_b != workload.trans_b
                or pair.field != field
            ):
                continue
            value_distance = abs(
                math.log1p(pair.low_value) - math.log1p(target_low)
            ) + abs(
                math.log1p(pair.high_value) - math.log1p(target_high)
            )
            distance = (
                workload_distance(workload, pair.workload)
                + 0.5 * value_distance
            )
            effect = pair.high_over_low
            if candidate_value < incumbent_value:
                effect = 1.0 / effect
            neighbors.append((distance, effect))
        nearest = sorted(neighbors)[:12]
        if not nearest:
            return CounterfactualEvidence(0, 1.0, 1.0, math.inf)
        effects = sorted(effect for _, effect in nearest)
        return CounterfactualEvidence(
            samples=len(effects),
            robust_ratio=median(effects),
            upper_ratio=effects[int(0.75 * (len(effects) - 1))],
            nearest_distance=nearest[0][0],
        )


def _is_direct_base(candidate: Candidate) -> bool:
    schedule = candidate.schedule
    return (
        candidate.template == Template.BASE
        and schedule["singleCoreM"] == schedule["baseM"]
        and schedule["singleCoreN"] == schedule["baseN"]
    )


def _transfer_evidence(
    workload: Workload,
    candidate: Candidate,
    observation_vectors: Sequence[
        tuple[MeasuredObservation, BehaviorVector]
    ],
    hardware: Hardware,
) -> _TransferEvidence:
    target = behavior_vector(workload, candidate.schedule, hardware)
    neighbors: list[tuple[float, MeasuredObservation]] = []
    for observation, vector in observation_vectors:
        if (
            observation.workload.identity() == workload.identity()
            or template_of(observation.schedule) != candidate.template
            or observation.workload.dtype != workload.dtype
            or observation.workload.trans_a != workload.trans_a
            or observation.workload.trans_b != workload.trans_b
        ):
            continue
        distance = (
            behavior_distance(target, vector)
            + workload_distance(workload, observation.workload)
        )
        neighbors.append((distance, observation))
    nearest = sorted(
        neighbors,
        key=lambda item: (
            item[0],
            item[1].record_id,
            item[1].schedule.signature(),
        ),
    )[:12]
    if not nearest:
        return _TransferEvidence(0, 0, 1.0, 1.0, math.inf)
    ratios = sorted(
        max(0.10, min(100.0, observation.measured_ratio))
        for _, observation in nearest
    )
    upper_index = int(0.75 * (len(ratios) - 1))
    middle = len(ratios) // 2
    robust = (
        ratios[middle]
        if len(ratios) % 2
        else 0.5 * (ratios[middle - 1] + ratios[middle])
    )
    return _TransferEvidence(
        samples=len(nearest),
        winners=sum(
            observation.is_verified_winner
            for _, observation in nearest
        ),
        robust_ratio=robust,
        upper_ratio=ratios[upper_index],
        nearest_distance=nearest[0][0],
    )


def _hardware_penalty(
    candidate: Candidate,
    metrics: dict[str, float],
    workload: Workload,
) -> float:
    if candidate.template == Template.DETERMINISTIC_SPLIT_K:
        preferred_base_k = 256 // INPUT_BYTES[workload.dtype]
    else:
        preferred_base_k = 64 if workload.dtype in {"fp16", "bf16"} else 32
    base_k_penalty = abs(
        math.log2(
            candidate.schedule["baseK"] / max(1.0, preferred_base_k)
        )
    )
    l0c_penalty = max(0.0, 1.0 - metrics.get("l0c_occupancy", 0.0))
    input_occupancy = min(
        metrics.get("l0a_occupancy", 0.0),
        metrics.get("l0b_occupancy", 0.0),
    )
    input_penalty = max(0.0, 0.50 - input_occupancy)
    l1_pipeline_penalty = max(
        0.0, 0.90 - metrics.get("l1_pipeline_efficiency", 0.0)
    )
    l2_wave_penalty = max(
        0.0, 0.90 - metrics.get("l2_wave_efficiency", 0.0)
    )
    l2_capacity_penalty = max(
        0.0, metrics.get("l2_capacity_pressure", 0.0) - 1.0
    )
    alignment_penalty = max(
        0.0, 0.95 - metrics.get("alignment_efficiency", 0.0)
    )
    return (
        0.25 * base_k_penalty
        + 0.40 * l0c_penalty
        + 0.20 * input_penalty
        + 1.00 * l1_pipeline_penalty
        + 0.60 * l2_wave_penalty
        + 0.50 * l2_capacity_penalty
        + 0.15 * alignment_penalty
    )


def select_one_shot_candidate(
    workload: Workload,
    candidates: Iterable[Candidate],
    incumbent: Candidate,
    observations: Sequence[MeasuredObservation],
    hardware: Hardware,
    *,
    cost_model: FeedbackCostModel | None = None,
    counterfactual_model: CounterfactualPolicyModel | None = None,
    latency_ranker: PairwiseLatencyRanker | None = None,
) -> OneShotDecision:
    """Choose one deployment candidate without target-workload measurements.

    Campaign selection intentionally rewards novelty and template coverage.
    This selector does neither: it uses only leave-target-workload-out latency,
    NPU rejection risk, and the hardware analytical prior.
    """
    if incumbent.source != "bank_incumbent":
        raise ValueError("one-shot incumbent must be the bank control")

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

    model = cost_model or FeedbackCostModel(observations, hardware)
    counterfactuals = counterfactual_model or CounterfactualPolicyModel(
        observations
    )
    ranker = latency_ranker or PairwiseLatencyRanker(
        observations, hardware
    )
    incumbent_vector = behavior_vector(
        workload, incumbent.schedule, hardware
    )
    incumbent_prediction = model.predict(
        workload,
        incumbent_vector,
        exclude_workload=workload.identity(),
        cross_workload_latency_weight=1.0,
    )
    candidate_vectors = [
        (
            candidate,
            behavior_vector(workload, candidate.schedule, hardware),
        )
        for candidate in unique.values()
    ]
    all_direct_base = [
        item for item in candidate_vectors if _is_direct_base(item[0])
    ]
    direct_base = sorted(
        all_direct_base,
        key=lambda item: (
            _hardware_penalty(item[0], item[1].metrics, workload)
            + 0.03
            * math.log1p(item[1].metrics["analytical_prior"]),
            item[0].schedule.signature(),
        ),
    )
    considered = sorted(
        candidate_vectors,
        key=lambda item: (
            _hardware_penalty(item[0], item[1].metrics, workload)
            + 0.03
            * math.log1p(item[1].metrics["analytical_prior"]),
            item[0].schedule.signature(),
        ),
    )
    observation_vectors = [
        (
            observation,
            behavior_vector(
                observation.workload,
                observation.schedule,
                hardware,
            ),
        )
        for observation in observations
        if (
            observation.source != "runtime_rejected"
            and observation.structured_verified
        )
    ]
    direct_base_history = [
        observation
        for observation in observations
        if (
            template_of(observation.schedule) == Template.BASE
            and observation.schedule["singleCoreM"]
            == observation.schedule["baseM"]
            and observation.schedule["singleCoreN"]
            == observation.schedule["baseN"]
        )
    ]
    direct_base_rejections = sum(
        observation.source == "runtime_rejected"
        for observation in direct_base_history
    )
    direct_base_risk = (
        direct_base_rejections + 1.0
    ) / (len(direct_base_history) + 2.0)

    evaluated = []
    for candidate, vector in considered:
        prediction = model.predict(
            workload,
            vector,
            exclude_workload=workload.identity(),
            cross_workload_latency_weight=0.15,
        )
        transfer = _transfer_evidence(
            workload,
            candidate,
            observation_vectors,
            hardware,
        )
        broad_prediction = model.predict(
            workload,
            vector,
            exclude_workload=workload.identity(),
            cross_workload_latency_weight=1.0,
        )
        counterfactual = counterfactuals.predict(
            workload, incumbent, candidate
        )
        pairwise = ranker.compare(
            workload, incumbent, candidate, hardware
        )
        relative_prediction = (
            broad_prediction.latency_ratio
            / max(0.10, incumbent_prediction.latency_ratio)
        )
        effective_risk = (
            min(prediction.runtime_risk_score, direct_base_risk)
            if _is_direct_base(candidate)
            else prediction.runtime_risk_score
        )
        effective_risk_support = (
            max(
                prediction.runtime_risk_support,
                min(1.0, len(direct_base_history) / 32.0),
            )
            if _is_direct_base(candidate)
            else prediction.runtime_risk_support
        )
        vector.metrics.update(
            {
                "predicted_latency_ratio": prediction.latency_ratio,
                "latency_uncertainty": prediction.latency_uncertainty,
                "latency_support": prediction.latency_support,
                "runtime_risk_score": effective_risk,
                "runtime_risk_support": effective_risk_support,
                "model_runtime_risk_score": prediction.runtime_risk_score,
                "direct_base_history": float(len(direct_base_history)),
                "transfer_samples": float(transfer.samples),
                "transfer_winners": float(transfer.winners),
                "transfer_robust_ratio": transfer.robust_ratio,
                "transfer_upper_ratio": transfer.upper_ratio,
                "transfer_nearest_distance": transfer.nearest_distance,
                "direct_base_geometry": float(_is_direct_base(candidate)),
                "relative_model_ratio": relative_prediction,
                "counterfactual_samples": float(counterfactual.samples),
                "counterfactual_robust_ratio": (
                    counterfactual.robust_ratio
                ),
                "counterfactual_upper_ratio": counterfactual.upper_ratio,
                "counterfactual_nearest_distance": (
                    counterfactual.nearest_distance
                ),
                "pairwise_rank_ratio": pairwise.relative_ratio,
                "pairwise_rank_uncertainty": pairwise.uncertainty,
                "pairwise_rank_support": pairwise.support,
                "pairwise_rank_nearest_distance": (
                    pairwise.nearest_workload_distance
                ),
            }
        )
        evaluated.append((candidate, vector, prediction, transfer))

    runtime_safe_candidates = [
        item
        for item in evaluated
        if not (
            item[1].metrics["runtime_risk_support"] >= 0.10
            and item[1].metrics["runtime_risk_score"] >= 0.45
        )
    ]

    transfer_eligible = [
        item
        for item in evaluated
        if item[3].samples >= 8
        and item[3].winners >= 3
        and item[3].robust_ratio <= 0.97
        and item[3].upper_ratio <= 0.99
        and item[3].nearest_distance <= 1.0
        and item[2].latency_ratio <= 0.97
        and item[2].latency_support >= 0.05
        and not (
            item[2].runtime_risk_support >= 0.10
            and item[2].runtime_risk_score >= 0.25
        )
    ]
    exploitation_pool = transfer_eligible

    if not exploitation_pool:
        local_pool = [
            item
            for item in runtime_safe_candidates
            if (
                item[0].source == "local_bank_anchor"
                and item[0].template == incumbent.template
                and item[1].metrics["counterfactual_samples"] >= 2
                and item[1].metrics["counterfactual_upper_ratio"] <= 0.99
                and item[1].metrics[
                    "counterfactual_nearest_distance"
                ]
                <= 2.0
            )
        ]
        coupled_policy_pool = [
            item
            for item in runtime_safe_candidates
            if (
                item[0].source == "contract_coupled_policy"
                and item[0].template == incumbent.template
            )
        ]
        local_signatures = {
            item[0].schedule.signature() for item in local_pool
        }
        policy_pool = [
            *local_pool,
            *(
                item
                for item in coupled_policy_pool
                if item[0].schedule.signature()
                not in local_signatures
            ),
        ]
        if not policy_pool:
            coupled_count = sum(
                item[0].source == "contract_coupled_policy"
                for item in evaluated
            )
            same_template_count = sum(
                item[0].source == "contract_coupled_policy"
                and item[0].template == incumbent.template
                for item in evaluated
            )
            raise ValueError(
                "one-shot selection has no safe deployment-policy "
                "candidate; broad or unsupported-local fallback is "
                "disabled "
                f"(evaluated={len(evaluated)} "
                f"runtime_safe={len(runtime_safe_candidates)} "
                f"coupled={coupled_count} "
                f"same_template={same_template_count} "
                f"incumbent_template={incumbent.template.name} "
                f"evidence_local={len(local_pool)})"
            )
        exploration_pool = policy_pool
        incumbent_hardware_penalty = _hardware_penalty(
            incumbent, incumbent_vector.metrics, workload
        )

        def exploration_score(item) -> tuple[float, tuple[int, ...]]:
            candidate, vector, prediction, _ = item
            metrics = vector.metrics
            counterfactual_support = min(
                1.0, metrics["counterfactual_samples"] / 6.0
            )
            counterfactual_ratio = (
                metrics["counterfactual_upper_ratio"]
                if metrics["counterfactual_samples"] >= 2
                else 1.0
            )
            score = (
                0.30
                * counterfactual_support
                * math.log(max(0.10, counterfactual_ratio))
                + 0.80
                * math.log(
                    max(0.10, metrics["pairwise_rank_ratio"])
                )
                + 0.15
                * (1.0 - metrics["pairwise_rank_support"])
                * metrics["pairwise_rank_uncertainty"]
                + 0.05
                * math.log(
                    max(0.10, metrics["relative_model_ratio"])
                )
                + 0.10
                * max(
                    -0.25,
                    _hardware_penalty(
                        candidate, metrics, workload
                    )
                    - incumbent_hardware_penalty,
                )
                + 0.50
                * metrics["runtime_risk_support"]
                * max(0.0, metrics["runtime_risk_score"] - 0.10)
                + 0.05 * prediction.latency_uncertainty
            )
            return score, candidate.schedule.signature()

        selected_item = min(
            exploration_pool,
            key=exploration_score,
        )
        original, vector, _, _ = selected_item
        score = exploration_score(selected_item)[0]
        policy = (
            "local_counterfactual"
            if original.source == "local_bank_anchor"
            else "coupled_policy_global"
        )
        vector.metrics.update(
            {
                "one_shot_score": score,
                "one_shot_target_observations": 0.0,
                "one_shot_incumbent_fallback": 0.0,
            }
        )
        selected = Candidate(
            schedule=original.schedule,
            template=original.template,
            source=(
                "one_shot_local_exploration"
                if original.source == "local_bank_anchor"
                else "one_shot_coupled_policy"
            ),
            rationale=(
                "single custom deployment decision from bank-relative "
                f"{policy}; generator={original.source}"
            ),
            acquisition=score,
            parent_signatures=original.parent_signatures,
            behavior_key=behavior_key(vector),
            metrics=dict(vector.metrics),
        )
        return OneShotDecision(
            candidate=selected,
            evaluated=len(evaluated),
            safe_candidates=len(runtime_safe_candidates),
            direct_base_candidates=len(direct_base),
            transfer_eligible_candidates=0,
            custom_eligible_candidates=0,
            local_candidates=len(local_pool),
            selection_policy=policy,
        )

    prior_scores = _normalized_priors(
        [
            vector.metrics["analytical_prior"]
            for _, vector, _, _ in exploitation_pool
        ]
    )
    ranked = []
    for (
        candidate,
        vector,
        prediction,
        transfer,
    ), prior_score in zip(
        exploitation_pool,
        prior_scores,
    ):
        hardware_penalty = _hardware_penalty(
            candidate, vector.metrics, workload
        )
        risk_penalty = (
            3.0
            * vector.metrics["runtime_risk_support"]
            * max(0.0, vector.metrics["runtime_risk_score"] - 0.10)
        )
        transfer_ratio = transfer.upper_ratio
        score = (
            math.log(
                max(
                    0.10,
                    min(
                        100.0,
                        max(prediction.latency_ratio, transfer_ratio),
                    ),
                )
            )
            + 0.50
            * math.log(
                max(0.10, vector.metrics["pairwise_rank_ratio"])
            )
            + risk_penalty
            + 0.15 * prior_score
            + 0.75 * hardware_penalty
            + 0.15 * prediction.latency_uncertainty
        )
        vector.metrics.update(
            {
                "one_shot_score": score,
                "one_shot_prior_score": prior_score,
                "one_shot_hardware_penalty": hardware_penalty,
                "one_shot_target_observations": 0.0,
            }
        )
        ranked.append((score, candidate, vector))

    safe = [
        item
        for item in ranked
        if not (
            item[2].metrics["runtime_risk_support"] >= 0.10
            and item[2].metrics["runtime_risk_score"] >= 0.45
        )
    ]
    selection_pool = safe or ranked
    score, original, vector = min(
        selection_pool,
        key=lambda item: (
            item[0],
            item[1].schedule.signature(),
        ),
    )
    selected = Candidate(
        schedule=original.schedule,
        template=original.template,
        source="one_shot_model",
        rationale=(
            "single exploitation decision from independent contract pool; "
            f"generator={original.source}"
        ),
        acquisition=score,
        parent_signatures=(),
        behavior_key=behavior_key(vector),
        metrics=dict(vector.metrics),
    )
    selected.metrics["one_shot_incumbent_fallback"] = 0.0
    return OneShotDecision(
        candidate=selected,
        evaluated=len(ranked),
        safe_candidates=len(safe),
        direct_base_candidates=len(direct_base),
        transfer_eligible_candidates=len(transfer_eligible),
        custom_eligible_candidates=len(exploitation_pool),
        local_candidates=sum(
            item[0].source == "local_bank_anchor"
            for item in evaluated
        ),
        selection_policy="cross_workload_evidence",
    )
