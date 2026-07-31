from __future__ import annotations

import math
from collections import defaultdict
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


@dataclass(frozen=True)
class FeedbackPrediction:
    latency_ratio: float
    latency_uncertainty: float
    latency_support: float
    runtime_risk_score: float
    runtime_risk_support: float


@dataclass(frozen=True)
class ModelValidation:
    mode: str
    latency_samples: int
    runtime_samples: int
    latency_spearman: float
    latency_mae: float
    latency_log_mae: float
    latency_median_factor: float
    latency_p90_factor: float
    pairwise_accuracy: float
    pairwise_comparisons: int
    top_quartile_recall: float
    best_candidate_percentile: float
    runtime_risk_auc: float
    analytical_spearman: float
    analytical_pairwise_accuracy: float


@dataclass(frozen=True)
class _Evidence:
    workload: Workload
    vector: BehaviorVector
    signature: tuple[int, ...]
    latency_ratio: float | None
    runtime_reject_rate: float


@dataclass(frozen=True)
class _LocalLinearModel:
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    templates: frozenset[str]
    residual: float
    samples: int

    def normalized_values(self, values: Sequence[float]) -> list[float]:
        return [
            (value - mean) / scale
            for value, mean, scale in zip(
                values, self.means, self.scales
            )
        ]

    def predict(self, values: Sequence[float]) -> float:
        normalized = self.normalized_values(values)
        log_ratio = self.coefficients[0] + sum(
            coefficient * value
            for coefficient, value in zip(
                self.coefficients[1:], normalized
            )
        )
        return math.exp(max(math.log(0.1), min(math.log(100.0), log_ratio)))

    def extrapolation(self, values: Sequence[float]) -> float:
        normalized = self.normalized_values(values)
        return math.sqrt(
            sum(min(16.0, value * value) for value in normalized)
            / max(1, len(normalized))
        )


def _safe_log2(value: float) -> float:
    return math.log2(max(value, 1.0e-9))


def _solve_linear_system(
    matrix: list[list[float]],
    values: list[float],
) -> list[float] | None:
    size = len(values)
    augmented = [
        [*matrix[row], values[row]] for row in range(size)
    ]
    for column in range(size):
        pivot = max(
            range(column, size),
            key=lambda row: abs(augmented[row][column]),
        )
        if abs(augmented[pivot][column]) < 1.0e-12:
            return None
        augmented[column], augmented[pivot] = (
            augmented[pivot],
            augmented[column],
        )
        divisor = augmented[column][column]
        augmented[column] = [
            value / divisor for value in augmented[column]
        ]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0:
                continue
            augmented[row] = [
                left - factor * right
                for left, right in zip(
                    augmented[row], augmented[column]
                )
            ]
    return [augmented[row][-1] for row in range(size)]


def _fit_local_model(
    evidence: Sequence[_Evidence],
) -> _LocalLinearModel | None:
    usable = [
        item for item in evidence if item.latency_ratio is not None
    ]
    if len(usable) < 6:
        return None
    feature_count = len(usable[0].vector.values)
    means = [
        sum(item.vector.values[index] for item in usable) / len(usable)
        for index in range(feature_count)
    ]
    scales = []
    for index, mean in enumerate(means):
        variance = sum(
            (item.vector.values[index] - mean) ** 2
            for item in usable
        ) / len(usable)
        scales.append(max(1.0e-6, math.sqrt(variance)))
    design = [
        [
            1.0,
            *(
                (value - mean) / scale
                for value, mean, scale in zip(
                    item.vector.values, means, scales
                )
            ),
        ]
        for item in usable
    ]
    targets = [
        math.log(max(0.1, min(100.0, item.latency_ratio or 1.0)))
        for item in usable
    ]
    columns = feature_count + 1
    normal = [
        [
            sum(row[left] * row[right] for row in design)
            for right in range(columns)
        ]
        for left in range(columns)
    ]
    right_hand = [
        sum(row[column] * target for row, target in zip(design, targets))
        for column in range(columns)
    ]
    # A compact ridge model is used only after enough same-workload NPU
    # samples exist. It learns local ordering without claiming cross-shape
    # generalization.
    for index in range(1, columns):
        normal[index][index] += 1.0
    coefficients = _solve_linear_system(normal, right_hand)
    if coefficients is None:
        return None
    residual = median(
        abs(
            target
            - (
                coefficients[0]
                + sum(
                    coefficient * value
                    for coefficient, value in zip(
                        coefficients[1:], row[1:]
                    )
                )
            )
        )
        for row, target in zip(design, targets)
    )
    return _LocalLinearModel(
        tuple(means),
        tuple(scales),
        tuple(coefficients),
        frozenset(str(item.vector.categories[0]) for item in usable),
        residual,
        len(usable),
    )


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
        core_rounds = (
            output_tasks
            * ceil_div(k_chunks, max(1, active_cores))
        )
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
    output_elements = workload.m * workload.n
    output_bytes = workload.m * workload.n * out_bytes
    reduction_bytes = 0.0
    if split == 3:
        # Each participating AIC writes an FP32 partial to workspace; AIV then
        # reads all partials and writes the final output.
        reduction_bytes = (
            2.0 * active_cores * output_elements * 4
        )
    split_reduction_ratio = reduction_bytes / max(1.0, output_bytes)
    split_k_chunks = k_chunks if split in (2, 3) else 1
    if split == 2:
        output_write_multiplier = 2 * split_k_chunks - 1
    elif split == 3:
        output_write_multiplier = (
            output_bytes + reduction_bytes
        ) / max(1.0, output_bytes)
    else:
        output_write_multiplier = 1
    inner_m_extent = max(
        1, schedule["stepM"] * schedule["baseM"]
    )
    inner_n_extent = max(
        1, schedule["stepN"] * schedule["baseN"]
    )
    inner_m_loops = ceil_div(schedule["singleCoreM"], inner_m_extent)
    inner_n_loops = ceil_div(schedule["singleCoreN"], inner_n_extent)
    input_lower_bound = (
        workload.m * workload.k * in_bytes
        + workload.k * workload.n * in_bytes
    )
    partitioned_input_bytes = (
        padded_m * workload.k * in_bytes * n_tasks
        + workload.k * padded_n * in_bytes * m_tasks
    )
    estimated_traffic_bytes = (
        partitioned_input_bytes
        + output_bytes * output_write_multiplier
    )
    traffic_lower_bound = input_lower_bound + output_bytes
    traffic_amplification = estimated_traffic_bytes / max(
        1.0, traffic_lower_bound
    )
    atomic_output_ratio = 0.0
    if split == 2:
        atomic_output_ratio = (
            output_bytes * max(0, output_write_multiplier - 1)
            / max(1.0, input_lower_bound + output_bytes)
        )

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
            "split_k_chunks": float(split_k_chunks),
            "output_write_multiplier": float(output_write_multiplier),
            "inner_m_loops": float(inner_m_loops),
            "inner_n_loops": float(inner_n_loops),
            "estimated_traffic_bytes": float(estimated_traffic_bytes),
            "traffic_amplification": traffic_amplification,
            "atomic_output_ratio": atomic_output_ratio,
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
        _safe_log2(split_k_chunks + 1.0) / 10.0,
        min(16.0, traffic_amplification) / 16.0,
        min(16.0, atomic_output_ratio) / 16.0,
        _safe_log2(inner_m_loops + 1.0) / 6.0,
        _safe_log2(inner_n_loops + 1.0) / 6.0,
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
        split_chunks,
        traffic,
        atomic_output,
        inner_m,
        inner_n,
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
        min(4, int(split_chunks * 8)),
        min(4, int(traffic * 8)),
        min(4, int(atomic_output * 8)),
        min(4, int(inner_m * 6)),
        min(4, int(inner_n * 6)),
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


def _observation_key(
    observation: MeasuredObservation,
) -> tuple[tuple[int, int, int, str, bool, bool, int], tuple[int, ...]]:
    return observation.workload.identity(), observation.schedule.signature()


class FeedbackCostModel:
    """Two-stage feedback model: NPU executability risk, then latency."""

    def __init__(
        self,
        observations: Sequence[MeasuredObservation],
        hardware: Hardware,
    ) -> None:
        grouped: dict[
            tuple[
                tuple[int, int, int, str, bool, bool, int],
                tuple[int, ...],
            ],
            list[MeasuredObservation],
        ] = defaultdict(list)
        for observation in observations:
            grouped[_observation_key(observation)].append(observation)

        evidence: list[_Evidence] = []
        for (_, signature), records in grouped.items():
            successful = [
                record.measured_ratio
                for record in records
                if record.source != "runtime_rejected"
            ]
            reject_count = sum(
                record.source == "runtime_rejected" for record in records
            )
            representative = records[-1]
            evidence.append(
                _Evidence(
                    workload=representative.workload,
                    vector=behavior_vector(
                        representative.workload,
                        representative.schedule,
                        hardware,
                    ),
                    signature=signature,
                    latency_ratio=(
                        median(successful) if successful else None
                    ),
                    runtime_reject_rate=reject_count / len(records),
                )
            )
        self.evidence = tuple(evidence)
        by_workload: dict[
            tuple[int, int, int, str, bool, bool, int],
            list[_Evidence],
        ] = defaultdict(list)
        for item in self.evidence:
            by_workload[item.workload.identity()].append(item)
        self.local_models = {
            workload: model
            for workload, items in by_workload.items()
            if (model := _fit_local_model(items)) is not None
        }

    def predict(
        self,
        workload: Workload,
        vector: BehaviorVector,
        *,
        exclude_keys: frozenset[
            tuple[
                tuple[int, int, int, str, bool, bool, int],
                tuple[int, ...],
            ]
        ] = frozenset(),
        exclude_workload: (
            tuple[int, int, int, str, bool, bool, int] | None
        ) = None,
    ) -> FeedbackPrediction:
        neighbors: list[tuple[float, _Evidence]] = []
        for item in self.evidence:
            key = item.workload.identity(), item.signature
            if key in exclude_keys:
                continue
            if (
                exclude_workload is not None
                and item.workload.identity() == exclude_workload
            ):
                continue
            distance = behavior_distance(vector, item.vector)
            distance += workload_distance(workload, item.workload)
            neighbors.append((distance, item))

        latency_neighbors = sorted(
            (
                (distance, item)
                for distance, item in neighbors
                if item.latency_ratio is not None
            ),
            key=lambda item: item[0],
        )[:5]
        if latency_neighbors:
            latency_weights = [
                math.exp(-distance)
                for distance, _ in latency_neighbors
            ]
            weight_sum = sum(latency_weights)
            log_prediction = sum(
                weight
                * math.log(
                    max(0.1, min(10.0, item.latency_ratio or 1.0))
                )
                for weight, (_, item) in zip(
                    latency_weights, latency_neighbors
                )
            ) / max(1.0e-12, weight_sum)
            latency_ratio = math.exp(log_prediction)
            log_spread = median(
                abs(
                    math.log(
                        max(0.1, min(10.0, item.latency_ratio or 1.0))
                    )
                    - log_prediction
                )
                for _, item in latency_neighbors
            )
            effective_samples = (
                weight_sum * weight_sum
                / max(
                    1.0e-12,
                    sum(weight * weight for weight in latency_weights),
                )
            )
            latency_support = min(
                1.0,
                math.exp(-latency_neighbors[0][0] / 4.0)
                * effective_samples
                / 5.0,
            )
            latency_uncertainty = min(
                1.0,
                latency_neighbors[0][0] / 4.0
                + log_spread
                + 0.25 / math.sqrt(max(1.0, effective_samples)),
            )
        else:
            latency_ratio = 1.0
            latency_uncertainty = 1.0
            latency_support = 0.0

        local_model = None
        if exclude_workload is None:
            if exclude_keys:
                local_evidence = [
                    item
                    for item in self.evidence
                    if item.workload.identity() == workload.identity()
                    and (item.workload.identity(), item.signature)
                    not in exclude_keys
                ]
                local_model = _fit_local_model(local_evidence)
            else:
                local_model = self.local_models.get(workload.identity())
        if (
            local_model is not None
            and str(vector.categories[0]) not in local_model.templates
        ):
            local_model = None
        if local_model is not None:
            local_ratio = local_model.predict(vector.values)
            extrapolation = local_model.extrapolation(vector.values)
            local_support = (
                min(1.0, local_model.samples / 16.0)
                * math.exp(-max(0.0, extrapolation - 2.0))
            )
            # Same-workload measurements are the strongest evidence available
            # for active search. Cross-workload neighbors retain a small role
            # and out-of-distribution schedules do not inherit false certainty.
            local_weight = 0.80 * local_support
            latency_ratio = math.exp(
                local_weight * math.log(local_ratio)
                + (1.0 - local_weight)
                * math.log(max(0.1, latency_ratio))
            )
            latency_support = max(latency_support, local_support)
            if local_support >= 0.25:
                latency_uncertainty = min(
                    latency_uncertainty,
                    min(
                        1.0,
                        local_model.residual
                        + 0.25 / math.sqrt(local_model.samples)
                        + 0.25 * (1.0 - local_support),
                    ),
                )

        risk_neighbors = sorted(
            neighbors, key=lambda item: item[0]
        )[:7]
        if risk_neighbors:
            risk_weights = [
                math.exp(-distance)
                for distance, _ in risk_neighbors
            ]
            risk_weight_sum = sum(risk_weights)
            # A neutral prior avoids interpreting the deliberately sampled
            # reject-heavy campaign as the population rejection rate.
            runtime_risk_score = (
                0.5
                + sum(
                    weight * item.runtime_reject_rate
                    for weight, (_, item) in zip(
                        risk_weights, risk_neighbors
                    )
                )
            ) / max(1.0e-12, 1.0 + risk_weight_sum)
            effective_risk_samples = (
                risk_weight_sum * risk_weight_sum
                / max(
                    1.0e-12,
                    sum(weight * weight for weight in risk_weights),
                )
            )
            runtime_risk_support = min(
                1.0,
                math.exp(-risk_neighbors[0][0] / 4.0)
                * effective_risk_samples
                / 7.0,
            )
        else:
            runtime_risk_score = 0.5
            runtime_risk_support = 0.0

        return FeedbackPrediction(
            latency_ratio=latency_ratio,
            latency_uncertainty=latency_uncertainty,
            latency_support=latency_support,
            runtime_risk_score=runtime_risk_score,
            runtime_risk_support=runtime_risk_support,
        )


def _average_ranks(values: Sequence[float]) -> list[float]:
    return [
        sum(other < value for other in values)
        + (sum(other == value for other in values) + 1) / 2.0
        for value in values
    ]


def _correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 2 or len(left) != len(right):
        return math.nan
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    covariance = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left_ranks, right_ranks)
    )
    left_variance = sum(
        (value - left_mean) ** 2 for value in left_ranks
    )
    right_variance = sum(
        (value - right_mean) ** 2 for value in right_ranks
    )
    denominator = math.sqrt(left_variance * right_variance)
    return covariance / denominator if denominator else math.nan


def _pairwise_accuracy(
    predictions: Sequence[float],
    actual: Sequence[float],
    workloads: Sequence[
        tuple[int, int, int, str, bool, bool, int]
    ],
) -> tuple[float, int]:
    correct = 0
    comparisons = 0
    for left in range(len(predictions)):
        for right in range(left):
            if workloads[left] != workloads[right]:
                continue
            predicted_delta = predictions[left] - predictions[right]
            actual_delta = actual[left] - actual[right]
            if predicted_delta == 0 or actual_delta == 0:
                continue
            comparisons += 1
            correct += predicted_delta * actual_delta > 0
    return (
        correct / comparisons if comparisons else math.nan,
        comparisons,
    )


def _binary_auc(
    scores: Sequence[float],
    labels: Sequence[bool],
) -> float:
    positives = [
        score for score, label in zip(scores, labels) if label
    ]
    negatives = [
        score for score, label in zip(scores, labels) if not label
    ]
    if not positives or not negatives:
        return math.nan
    ordering = sum(
        (positive > negative) + 0.5 * (positive == negative)
        for positive in positives
        for negative in negatives
    )
    return ordering / (len(positives) * len(negatives))


def _top_k_metrics(
    predictions: Sequence[float],
    actual: Sequence[float],
    workloads: Sequence[
        tuple[int, int, int, str, bool, bool, int]
    ],
) -> tuple[float, float]:
    grouped: dict[
        tuple[int, int, int, str, bool, bool, int],
        list[int],
    ] = defaultdict(list)
    for index, workload in enumerate(workloads):
        grouped[workload].append(index)
    recalls = []
    best_percentiles = []
    for indices in grouped.values():
        if len(indices) < 4:
            continue
        count = max(1, ceil_div(len(indices), 4))
        predicted_order = sorted(indices, key=lambda index: predictions[index])
        actual_order = sorted(indices, key=lambda index: actual[index])
        predicted_top = set(predicted_order[:count])
        actual_top = set(actual_order[:count])
        recalls.append(len(predicted_top & actual_top) / count)
        best_index = actual_order[0]
        best_percentiles.append(
            predicted_order.index(best_index) / max(1, len(indices) - 1)
        )
    return (
        sum(recalls) / len(recalls) if recalls else math.nan,
        (
            sum(best_percentiles) / len(best_percentiles)
            if best_percentiles
            else math.nan
        ),
    )


def validate_feedback_model(
    observations: Sequence[MeasuredObservation],
    hardware: Hardware,
    *,
    leave_workload_out: bool,
) -> ModelValidation:
    model = FeedbackCostModel(observations, hardware)
    latency_predictions: list[float] = []
    latency_actual: list[float] = []
    analytical_predictions: list[float] = []
    latency_workloads: list[
        tuple[int, int, int, str, bool, bool, int]
    ] = []
    runtime_scores: list[float] = []
    runtime_labels: list[bool] = []

    for item in model.evidence:
        key = item.workload.identity(), item.signature
        prediction = model.predict(
            item.workload,
            item.vector,
            exclude_keys=frozenset({key}),
            exclude_workload=(
                item.workload.identity() if leave_workload_out else None
            ),
        )
        if item.latency_ratio is not None:
            latency_predictions.append(prediction.latency_ratio)
            latency_actual.append(item.latency_ratio)
            analytical_predictions.append(
                item.vector.metrics["analytical_prior"]
            )
            latency_workloads.append(item.workload.identity())
        runtime_scores.append(prediction.runtime_risk_score)
        runtime_labels.append(item.runtime_reject_rate >= 0.5)

    pairwise, comparisons = _pairwise_accuracy(
        latency_predictions,
        latency_actual,
        latency_workloads,
    )
    analytical_pairwise, _ = _pairwise_accuracy(
        analytical_predictions,
        latency_actual,
        latency_workloads,
    )
    top_quartile_recall, best_candidate_percentile = _top_k_metrics(
        latency_predictions,
        latency_actual,
        latency_workloads,
    )
    latency_log_errors = sorted(
        abs(
            math.log(max(0.1, predicted))
            - math.log(max(0.1, actual))
        )
        for predicted, actual in zip(
            latency_predictions, latency_actual
        )
    )
    return ModelValidation(
        mode=(
            "leave_workload_out"
            if leave_workload_out
            else "leave_fingerprint_out"
        ),
        latency_samples=len(latency_actual),
        runtime_samples=len(runtime_labels),
        latency_spearman=_correlation(
            latency_predictions, latency_actual
        ),
        latency_mae=(
            sum(
                abs(predicted - actual)
                for predicted, actual in zip(
                    latency_predictions, latency_actual
                )
            )
            / len(latency_actual)
            if latency_actual
            else math.nan
        ),
        latency_log_mae=(
            sum(
                abs(
                    math.log(max(0.1, predicted))
                    - math.log(max(0.1, actual))
                )
                for predicted, actual in zip(
                    latency_predictions, latency_actual
                )
            )
            / len(latency_actual)
            if latency_actual
            else math.nan
        ),
        latency_median_factor=(
            math.exp(median(latency_log_errors))
            if latency_log_errors
            else math.nan
        ),
        latency_p90_factor=(
            math.exp(
                latency_log_errors[
                    int(0.9 * (len(latency_log_errors) - 1))
                ]
            )
            if latency_log_errors
            else math.nan
        ),
        pairwise_accuracy=pairwise,
        pairwise_comparisons=comparisons,
        top_quartile_recall=top_quartile_recall,
        best_candidate_percentile=best_candidate_percentile,
        runtime_risk_auc=_binary_auc(runtime_scores, runtime_labels),
        analytical_spearman=_correlation(
            analytical_predictions, latency_actual
        ),
        analytical_pairwise_accuracy=analytical_pairwise,
    )


def score_candidates(
    workload: Workload,
    candidates: Iterable[Candidate],
    observations: Sequence[MeasuredObservation],
    hardware: Hardware,
) -> list[tuple[Candidate, BehaviorVector, float, float]]:
    evaluated: list[tuple[Candidate, BehaviorVector, float, float]] = []
    model = FeedbackCostModel(observations, hardware)
    for candidate in candidates:
        vector = behavior_vector(workload, candidate.schedule, hardware)
        prediction = model.predict(workload, vector)
        vector.metrics.update(
            {
                "predicted_latency_ratio": prediction.latency_ratio,
                "latency_uncertainty": prediction.latency_uncertainty,
                "latency_support": prediction.latency_support,
                "runtime_risk_score": prediction.runtime_risk_score,
                "runtime_risk_support": prediction.runtime_risk_support,
            }
        )
        evaluated.append(
            (
                candidate,
                vector,
                prediction.latency_ratio,
                prediction.latency_uncertainty,
            )
        )
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
        risk = vector.metrics["runtime_risk_score"]
        risk_support = vector.metrics["runtime_risk_support"]
        latency_support = vector.metrics["latency_support"]
        exploitation = (
            latency_support
            * math.log(max(0.25, min(4.0, prediction)))
            + 0.80 * max(0.0, risk - 0.5) * risk_support
            + 0.05 * prior
        )
        intervention_bonus = {
            "feedback_winner_mutation": 0.25,
            "feedback_regression_counterfactual": 0.15,
        }.get(candidate.source, 0.0)
        acquisition = (
            exploitation - 0.08 * uncertainty - intervention_bonus
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
        known_high_risk = sum(
            item[1].metrics.get("runtime_risk_score", 0.5) >= 0.75
            and item[1].metrics.get("runtime_risk_support", 0.0) >= 0.25
            for item in selected
        )
        high_risk_budget = max(1, limit // 8)
        for item in scored:
            signature = item[0].schedule.signature()
            if signature in selected_signatures:
                continue
            risk = item[1].metrics.get("runtime_risk_score", 0.5)
            risk_support = item[1].metrics.get(
                "runtime_risk_support", 0.0
            )
            if (
                risk >= 0.75
                and risk_support >= 0.25
                and known_high_risk >= high_risk_budget
            ):
                continue
            novelty = nearest_distance[signature]
            key = (
                novelty
                - 0.15 * item[2]
                - 0.35 * max(0.0, risk - 0.5) * risk_support,
                -item[2],
            )
            if best_key is None or key > best_key:
                best = item
                best_key = key
        if best is None:
            # High-risk schedules are retained as bounded probes rather than
            # being made unreachable by the learned model.
            remaining = [
                item
                for item in scored
                if item[0].schedule.signature()
                not in selected_signatures
            ]
            if not remaining:
                break
            best = min(remaining, key=lambda item: item[2])
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
