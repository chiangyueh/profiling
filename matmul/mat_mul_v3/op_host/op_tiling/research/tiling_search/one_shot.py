from __future__ import annotations

import math
from dataclasses import dataclass
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
    Candidate,
    Hardware,
    MeasuredObservation,
    Template,
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
    incumbent_fallback: bool


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
    return (
        0.25 * base_k_penalty
        + 0.40 * l0c_penalty
        + 0.20 * input_penalty
    )


def select_one_shot_candidate(
    workload: Workload,
    candidates: Iterable[Candidate],
    incumbent: Candidate,
    observations: Sequence[MeasuredObservation],
    hardware: Hardware,
    *,
    cost_model: FeedbackCostModel | None = None,
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
            "contract_global",
            "local_bank_anchor",
        }:
            continue
        if candidate.schedule.signature() == incumbent.schedule.signature():
            continue
        unique.setdefault(candidate.schedule.signature(), candidate)

    model = cost_model or FeedbackCostModel(observations, hardware)
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
            }
        )
        evaluated.append((candidate, vector, prediction, transfer))

    runtime_safe_candidates = [
        item
        for item in evaluated
        if not (
            item[2].runtime_risk_support >= 0.10
            and item[2].runtime_risk_score >= 0.45
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
        incumbent_vector = behavior_vector(
            workload, incumbent.schedule, hardware
        )
        incumbent_vector.metrics.update(
            {
                "predicted_latency_ratio": 1.0,
                "latency_uncertainty": 0.0,
                "latency_support": 1.0,
                "runtime_risk_score": 0.0,
                "runtime_risk_support": 1.0,
                "one_shot_score": 0.0,
                "one_shot_target_observations": 0.0,
                "one_shot_incumbent_fallback": 1.0,
            }
        )
        selected = Candidate(
            schedule=incumbent.schedule,
            template=incumbent.template,
            source="one_shot_bank_fallback",
            rationale=(
                "bank incumbent retained because no custom candidate has "
                "independent cross-workload replacement evidence"
            ),
            acquisition=0.0,
            parent_signatures=(),
            behavior_key=behavior_key(incumbent_vector),
            metrics=dict(incumbent_vector.metrics),
        )
        return OneShotDecision(
            candidate=selected,
            evaluated=len(evaluated),
            safe_candidates=len(runtime_safe_candidates),
            direct_base_candidates=len(direct_base),
            transfer_eligible_candidates=0,
            custom_eligible_candidates=0,
            incumbent_fallback=True,
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
        incumbent_fallback=False,
    )
