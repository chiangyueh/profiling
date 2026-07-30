from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Iterable, Sequence

from .contracts import (
    align_up,
    ceil_div,
    fix_mode,
    full_load_mode,
    profitability_prior,
    split_mode,
    template_of,
)
from .domain import (
    INPUT_BYTES,
    OUTPUT_BYTES,
    Candidate,
    Hardware,
    MeasuredObservation,
    Schedule,
    Workload,
)


@dataclass(frozen=True)
class BehaviorVector:
    categories: tuple[object, ...]
    values: tuple[float, ...]
    metrics: dict[str, float]


def _safe_log2(value: float) -> float:
    return math.log2(max(value, 1.0e-9))


def behavior_vector(
    workload: Workload,
    schedule: Schedule,
    hardware: Hardware,
) -> BehaviorVector:
    prior, metrics = profitability_prior(workload, schedule, hardware)
    in_bytes = INPUT_BYTES[workload.dtype]
    out_bytes = OUTPUT_BYTES[workload.dtype]
    m_tasks = ceil_div(workload.m, schedule["singleCoreM"])
    n_tasks = ceil_div(workload.n, schedule["singleCoreN"])
    output_tasks = max(1, m_tasks * n_tasks)
    split = split_mode(schedule)
    full = full_load_mode(schedule)
    k_chunks = ceil_div(workload.k, schedule["singleCoreK"])
    if split == 3:
        active_cores = min(schedule["usedCoreNum"], k_chunks)
        core_rounds = ceil_div(k_chunks, max(1, active_cores))
    else:
        active_cores = min(schedule["usedCoreNum"], output_tasks)
        core_rounds = ceil_div(output_tasks, max(1, active_cores))

    padded_m = m_tasks * schedule["singleCoreM"]
    padded_n = n_tasks * schedule["singleCoreN"]
    padded_k = align_up(workload.k, schedule["baseK"])
    padding_efficiency = (
        workload.m
        * workload.n
        * workload.k
        / max(1.0, padded_m * padded_n * padded_k)
    )
    base_tile_ops = (
        2.0
        * schedule["baseM"]
        * schedule["baseN"]
        * schedule["baseK"]
    )
    mte_bytes = (
        schedule["baseM"] * schedule["baseK"] * in_bytes
        + schedule["baseN"] * schedule["baseK"] * in_bytes
        + schedule["baseM"] * schedule["baseN"] * out_bytes
    )
    mte_cube_ratio = mte_bytes / max(1.0, base_tile_ops)

    l2_m_extent = min(
        workload.m,
        schedule["l2MTileBlock"] * schedule["singleCoreM"],
    )
    l2_n_extent = min(
        workload.n,
        schedule["l2NTileBlock"] * schedule["singleCoreN"],
    )
    if schedule["l2MTileBlock"] == 0:
        l2_m_extent = workload.m
        l2_n_extent = workload.n
    l2_working_set = (
        l2_m_extent * workload.k * in_bytes
        + workload.k * l2_n_extent * in_bytes
        + l2_m_extent * l2_n_extent * out_bytes
    )
    l2_working_set_ratio = l2_working_set / max(1.0, hardware.l2_bytes)
    output_bytes = workload.m * workload.n * out_bytes
    reduction_bytes = (
        output_bytes * max(0, schedule["usedCoreNum"] - 1)
        if split == 3
        else 0
    )
    split_reduction_ratio = reduction_bytes / max(1.0, output_bytes)

    resident_bytes = 0
    resident_capacity = 1
    if full == 1:
        resident_bytes = (
            align_up(workload.m, 16)
            * align_up(workload.k, 16)
            * in_bytes
        )
        resident_capacity = hardware.effective_l1_bytes
    elif full == 2:
        resident_bytes = (
            align_up(workload.k, 16)
            * align_up(workload.n, 32 // in_bytes)
            * in_bytes
        )
        resident_capacity = hardware.effective_l1_bytes
    full_load_resident_ratio = resident_bytes / max(1.0, resident_capacity)

    l0_occupancy = max(
        metrics.get("l0a_occupancy", 0.0),
        metrics.get("l0b_occupancy", 0.0),
        metrics.get("l0c_occupancy", 0.0),
    )
    l1_occupancy = metrics.get("l1_occupancy", 0.0)
    k_passes = ceil_div(workload.k, schedule["baseK"])
    metrics.update(
        {
            "active_cores": float(active_cores),
            "core_rounds": float(core_rounds),
            "l0_occupancy": l0_occupancy,
            "l1_occupancy": l1_occupancy,
            "k_passes": float(k_passes),
            "padding_efficiency": padding_efficiency,
            "mte_cube_ratio": mte_cube_ratio,
            "l2_working_set_bytes": float(l2_working_set),
            "l2_working_set_ratio": l2_working_set_ratio,
            "split_reduction_bytes": float(reduction_bytes),
            "split_reduction_ratio": split_reduction_ratio,
            "full_load_resident_ratio": full_load_resident_ratio,
            "analytical_prior": prior,
        }
    )
    categories = (
        template_of(schedule).value,
        schedule["tilingEnable"],
        workload.dtype,
        int(workload.trans_a),
        int(workload.trans_b),
        schedule["iterateOrder"],
        schedule["l2IterateOrder"],
        schedule["dbL0A"],
        schedule["dbL0B"],
        schedule["dbL0C"],
        fix_mode(schedule),
    )
    values = (
        active_cores / max(1.0, min(workload.max_cores, hardware.aic_cores)),
        _safe_log2(core_rounds + 1.0) / 5.0,
        l0_occupancy,
        l1_occupancy,
        _safe_log2(k_passes + 1.0) / 12.0,
        padding_efficiency,
        min(1.0, _safe_log2(1.0 + mte_cube_ratio * 256.0) / 8.0),
        min(4.0, l2_working_set_ratio) / 4.0,
        min(32.0, split_reduction_ratio) / 32.0,
        min(1.5, full_load_resident_ratio) / 1.5,
    )
    return BehaviorVector(categories, values, metrics)


def behavior_key(vector: BehaviorVector) -> tuple[object, ...]:
    template, mode, dtype, trans_a, trans_b, *_ = vector.categories
    (
        active,
        rounds,
        l0,
        l1,
        k_passes,
        padding,
        mte_cube,
        l2,
        reduction,
        resident,
    ) = (
        vector.values
    )
    return (
        template,
        mode,
        dtype,
        trans_a,
        trans_b,
        min(3, int(active * 4)),
        min(4, int(rounds * 5)),
        min(4, int(l0 * 5)),
        min(4, int(l1 * 5)),
        min(4, int(k_passes * 5)),
        min(4, int(padding * 5)),
        min(4, int(mte_cube * 5)),
        min(4, int(l2 * 5)),
        min(3, int(reduction * 4)),
        min(3, int(resident * 4)),
    )


def behavior_distance(left: BehaviorVector, right: BehaviorVector) -> float:
    category_penalty = 0.0
    for index, (left_value, right_value) in enumerate(
        zip(left.categories, right.categories)
    ):
        if left_value == right_value:
            continue
        category_penalty += 2.0 if index < 5 else 0.2
    numeric = math.sqrt(
        sum(
            (left_value - right_value) ** 2
            for left_value, right_value in zip(left.values, right.values)
        )
    )
    return category_penalty + numeric


def workload_distance(left: Workload, right: Workload) -> float:
    categorical = (
        (0.0 if left.dtype == right.dtype else 1.5)
        + (0.0 if left.trans_a == right.trans_a else 0.8)
        + (0.0 if left.trans_b == right.trans_b else 0.8)
    )
    numeric = math.sqrt(
        sum(
            (
                _safe_log2(left_value + 1.0)
                - _safe_log2(right_value + 1.0)
            )
            ** 2
            for left_value, right_value in (
                (left.m, right.m),
                (left.n, right.n),
                (left.k, right.k),
            )
        )
    )
    return categorical + numeric / 4.0


def _predict_feedback(
    workload: Workload,
    vector: BehaviorVector,
    observation_vectors: Sequence[
        tuple[MeasuredObservation, BehaviorVector]
    ],
) -> tuple[float, float]:
    neighbors: list[tuple[float, float]] = []
    for observation, observation_vector in observation_vectors:
        distance = behavior_distance(vector, observation_vector)
        distance += workload_distance(workload, observation.workload)
        neighbors.append((distance, observation.measured_ratio))
    if not neighbors:
        return 1.0, 1.0
    nearest = sorted(neighbors)[: min(7, len(neighbors))]
    weights = [1.0 / max(0.05, distance) for distance, _ in nearest]
    prediction = sum(
        weight * ratio for weight, (_, ratio) in zip(weights, nearest)
    ) / sum(weights)
    spread = median(abs(ratio - prediction) for _, ratio in nearest)
    uncertainty = min(1.0, nearest[0][0] / 4.0 + spread)
    return prediction, uncertainty


def score_candidates(
    workload: Workload,
    candidates: Iterable[Candidate],
    observations: Sequence[MeasuredObservation],
    hardware: Hardware,
) -> list[tuple[Candidate, BehaviorVector, float, float]]:
    evaluated: list[tuple[Candidate, BehaviorVector, float, float]] = []
    observation_vectors = [
        (
            observation,
            behavior_vector(
                observation.workload, observation.schedule, hardware
            ),
        )
        for observation in observations
    ]
    for candidate in candidates:
        vector = behavior_vector(workload, candidate.schedule, hardware)
        prediction, uncertainty = _predict_feedback(
            workload, vector, observation_vectors
        )
        evaluated.append((candidate, vector, prediction, uncertainty))
    if not evaluated:
        return []
    priors = sorted(
        math.log1p(item[1].metrics["analytical_prior"])
        for item in evaluated
    )
    low = priors[0]
    span = max(1.0e-9, priors[-1] - low)
    scored: list[tuple[Candidate, BehaviorVector, float, float]] = []
    for candidate, vector, prediction, uncertainty in evaluated:
        prior = (
            math.log1p(vector.metrics["analytical_prior"]) - low
        ) / span
        # The analytical model is a weak hardware prior. Paired NPU feedback
        # must dominate once it exists; otherwise model error repeatedly
        # outranks measured winners and known execution failures.
        exploitation = 0.20 * prior + 0.80 * min(3.0, prediction) / 3.0
        intervention_bonus = {
            "feedback_winner_mutation": 0.25,
            "feedback_regression_counterfactual": 0.15,
        }.get(candidate.source, 0.0)
        acquisition = (
            exploitation - 0.18 * uncertainty - intervention_bonus
        )
        scored.append((candidate, vector, acquisition, uncertainty))
    return sorted(
        scored,
        key=lambda item: (
            item[2],
            item[0].schedule.signature(),
        ),
    )


def select_behavior_coverage(
    workload: Workload,
    candidates: Iterable[Candidate],
    observations: Sequence[MeasuredObservation],
    hardware: Hardware,
    limit: int,
) -> list[Candidate]:
    unique: dict[tuple[int, ...], Candidate] = {}
    for candidate in candidates:
        unique.setdefault(candidate.schedule.signature(), candidate)
    scored = score_candidates(
        workload, unique.values(), observations, hardware
    )
    if len(scored) <= limit:
        return [
            candidate.with_selection(
                acquisition=acquisition,
                behavior_key=behavior_key(vector),
                metrics=vector.metrics,
            )
            for candidate, vector, acquisition, _ in scored
        ]

    selected: list[tuple[Candidate, BehaviorVector, float, float]] = []
    selected_signatures: set[tuple[int, ...]] = set()

    # One callback capability probe per generated kernel family is retained.
    for template in sorted({item[0].template for item in scored}, key=str):
        representative = next(
            item for item in scored if item[0].template == template
        )
        selected.append(representative)
        selected_signatures.add(representative[0].schedule.signature())

    exploitation_limit = max(len(selected), limit // 2)
    for item in scored:
        if len(selected) >= exploitation_limit:
            break
        signature = item[0].schedule.signature()
        if signature in selected_signatures:
            continue
        selected.append(item)
        selected_signatures.add(signature)

    observation_vectors = [
        behavior_vector(
            observation.workload, observation.schedule, hardware
        )
        for observation in observations
    ]
    # Maintain each remaining candidate's nearest selected/observed distance.
    # Recomputing every pair after each selection is O(N * limit^2) and makes
    # a deliberately broad host pool needlessly expensive.
    nearest_distance: dict[tuple[int, ...], float] = {}
    initial_references = [
        *(item[1] for item in selected),
        *observation_vectors,
    ]
    for item in scored:
        signature = item[0].schedule.signature()
        if signature in selected_signatures:
            continue
        nearest_distance[signature] = min(
            (
                behavior_distance(item[1], reference)
                for reference in initial_references
            ),
            default=4.0,
        )

    while len(selected) < limit:
        best = None
        best_key = None
        for item in scored:
            signature = item[0].schedule.signature()
            if signature in selected_signatures:
                continue
            novelty = nearest_distance[signature]
            key = (novelty - 0.15 * item[2], -item[2])
            if best_key is None or key > best_key:
                best = item
                best_key = key
        if best is None:
            break
        selected.append(best)
        best_signature = best[0].schedule.signature()
        selected_signatures.add(best_signature)
        nearest_distance.pop(best_signature, None)
        for item in scored:
            signature = item[0].schedule.signature()
            if signature in selected_signatures:
                continue
            nearest_distance[signature] = min(
                nearest_distance[signature],
                behavior_distance(item[1], best[1]),
            )

    return [
        candidate.with_selection(
            acquisition=acquisition,
            behavior_key=behavior_key(vector),
            metrics=vector.metrics,
        )
        for candidate, vector, acquisition, _ in sorted(
            selected,
            key=lambda item: (
                item[2],
                item[0].schedule.signature(),
            ),
        )
    ]
