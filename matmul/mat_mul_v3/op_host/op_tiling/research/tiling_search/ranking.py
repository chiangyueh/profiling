from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Iterable, Sequence

import numpy as np

from .behavior import behavior_vector, workload_distance
from .contracts import ceil_div, template_of
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
class PairwiseLatencyPrediction:
    relative_ratio: float
    uncertainty: float
    support: float
    nearest_workload_distance: float


@dataclass(frozen=True)
class PairwiseRankValidation:
    folds: int
    groups: int
    informative_pairs: int
    pairwise_accuracy: float
    one_field_pairs: int
    one_field_accuracy: float
    base_one_field_pairs: int
    base_one_field_accuracy: float
    top_quartile_recall: float
    best_candidate_percentile: float
    median_top1_regret: float
    p90_top1_regret: float


@dataclass(frozen=True)
class _RankRow:
    workload: Workload
    schedule: Schedule
    log_ratio: float
    reliability: float


def _safe_log2(value: float) -> float:
    return math.log2(max(1.0e-9, value))


def _measurement_reliability(observation: MeasuredObservation) -> float:
    if observation.structured_verified:
        return 1.0
    if observation.verified:
        return 0.75
    return 0.20


def _layout(workload: Workload) -> int:
    return 2 * int(workload.trans_a) + int(workload.trans_b)


def _one_hot(index: int, size: int) -> list[float]:
    return [float(position == index) for position in range(size)]


def _tail_efficiency(size: int, tile: int) -> float:
    return size / max(1.0, ceil_div(size, tile) * tile)


def latency_rank_features(
    workload: Workload,
    schedule: Schedule,
    hardware: Hardware,
) -> tuple[float, ...]:
    """Candidate features used only for within-workload latency ordering."""

    vector = behavior_vector(workload, schedule, hardware)
    metrics = vector.metrics
    template_values = tuple(Template)
    template_index = template_values.index(template_of(schedule))
    dtype_index = ("fp16", "bf16", "fp32").index(workload.dtype)
    layout_index = _layout(workload)

    m_tasks = ceil_div(workload.m, schedule["singleCoreM"])
    n_tasks = ceil_div(workload.n, schedule["singleCoreN"])
    output_tasks = max(1, m_tasks * n_tasks)
    active_cores = max(1.0, metrics["active_cores"])
    core_rounds = max(1.0, metrics["core_rounds"])
    wave_efficiency = output_tasks / max(
        1.0, active_cores * core_rounds
    )

    dynamic = [
        *vector.values,
        min(1.0, wave_efficiency),
        _safe_log2(metrics["analytical_prior"] + 1.0) / 16.0,
        _safe_log2(metrics["estimated_traffic_bytes"] + 1.0) / 40.0,
        _safe_log2(metrics["l2_working_set_bytes"] + 1.0) / 32.0,
        min(16.0, metrics["traffic_amplification"]) / 16.0,
        min(64.0, metrics["output_write_multiplier"]) / 64.0,
        metrics.get("l0a_occupancy", 0.0),
        metrics.get("l0b_occupancy", 0.0),
        metrics.get("l0c_occupancy", 0.0),
        _tail_efficiency(workload.m, schedule["baseM"]),
        _tail_efficiency(workload.n, schedule["baseN"]),
        _tail_efficiency(workload.k, schedule["baseK"]),
        _tail_efficiency(workload.m, schedule["singleCoreM"]),
        _tail_efficiency(workload.n, schedule["singleCoreN"]),
        _tail_efficiency(workload.k, schedule["singleCoreK"]),
    ]

    field_scales = {
        "usedCoreNum": 5.0,
        "singleCoreM": 14.0,
        "singleCoreN": 14.0,
        "singleCoreK": 18.0,
        "baseM": 10.0,
        "baseN": 10.0,
        "baseK": 10.0,
        "depthA1": 8.0,
        "depthB1": 8.0,
        "stepM": 5.0,
        "stepN": 5.0,
        "iterateOrder": 1.0,
        "stepKa": 8.0,
        "stepKb": 8.0,
        "dbL0A": 2.0,
        "dbL0B": 2.0,
        "dbL0C": 2.0,
        "l2MTileCnt": 10.0,
        "l2NTileCnt": 10.0,
        "l2MTileBlock": 10.0,
        "l2NTileBlock": 10.0,
        "l2IterateOrder": 2.0,
        "tilingEnable": 4.0,
    }
    schedule_features = [
        (
            float(schedule[field]) / field_scales[field]
            if field in {"iterateOrder", "l2IterateOrder", "tilingEnable"}
            else _safe_log2(schedule[field] + 1.0) / field_scales[field]
        )
        for field in KNOWLEDGE_FIELDS
    ]
    relative_geometry = [
        _safe_log2(schedule["baseM"] / max(1.0, workload.m)),
        _safe_log2(schedule["baseN"] / max(1.0, workload.n)),
        _safe_log2(schedule["baseK"] / max(1.0, workload.k)),
        _safe_log2(schedule["singleCoreM"] / max(1.0, workload.m)),
        _safe_log2(schedule["singleCoreN"] / max(1.0, workload.n)),
        _safe_log2(schedule["singleCoreK"] / max(1.0, workload.k)),
        _safe_log2(
            schedule["singleCoreM"] / max(1.0, schedule["baseM"])
        ),
        _safe_log2(
            schedule["singleCoreN"] / max(1.0, schedule["baseN"])
        ),
        _safe_log2(
            schedule["singleCoreK"] / max(1.0, schedule["baseK"])
        ),
        _safe_log2(m_tasks + 1.0) / 10.0,
        _safe_log2(n_tasks + 1.0) / 10.0,
        _safe_log2(output_tasks + 1.0) / 20.0,
    ]

    template_hot = _one_hot(template_index, len(template_values))
    dtype_hot = _one_hot(dtype_index, 3)
    layout_hot = _one_hot(layout_index, 4)
    interaction_base = dynamic[:15]
    interactions = [
        value * enabled
        for enabled in template_hot
        for value in interaction_base
    ]
    interactions.extend(
        value * enabled
        for enabled in dtype_hot
        for value in dynamic[:10]
    )
    interactions.extend(
        value * enabled
        for enabled in layout_hot
        for value in dynamic[:10]
    )
    return tuple(
        [
            *dynamic,
            *schedule_features,
            *relative_geometry,
            *template_hot,
            *dtype_hot,
            *layout_hot,
            *interactions,
        ]
    )


def _training_groups(
    observations: Sequence[MeasuredObservation],
    *,
    excluded_workloads: frozenset[
        tuple[int, int, int, str, bool, bool, int]
    ] = frozenset(),
) -> list[list[_RankRow]]:
    grouped: dict[
        tuple[
            tuple[int, int, int, str, bool, bool, int],
            str,
        ],
        dict[tuple[int, ...], list[MeasuredObservation]],
    ] = defaultdict(lambda: defaultdict(list))
    for observation in observations:
        if (
            observation.source
            in {"runtime_rejected", "runtime_verified"}
            or observation.workload.identity() in excluded_workloads
        ):
            continue
        grouped[
            (observation.workload.identity(), observation.record_id)
        ][observation.schedule.signature()].append(observation)

    result: list[list[_RankRow]] = []
    for signatures in grouped.values():
        rows = []
        for records in signatures.values():
            best_reliability = max(
                _measurement_reliability(record) for record in records
            )
            trusted = [
                record
                for record in records
                if _measurement_reliability(record) == best_reliability
            ]
            rows.append(
                _RankRow(
                    workload=trusted[0].workload,
                    schedule=trusted[0].schedule,
                    log_ratio=median(
                        math.log(
                            max(0.10, min(100.0, record.measured_ratio))
                        )
                        for record in trusted
                    ),
                    reliability=best_reliability,
                )
            )
        if len(rows) >= 2:
            result.append(rows)
    return result


class PairwiseLatencyRanker:
    """Learn candidate ordering from paired measurements within each run."""

    def __init__(
        self,
        observations: Sequence[MeasuredObservation],
        hardware: Hardware,
        *,
        excluded_workloads: frozenset[
            tuple[int, int, int, str, bool, bool, int]
        ] = frozenset(),
        ridge: float = 0.05,
    ) -> None:
        groups = _training_groups(
            observations, excluded_workloads=excluded_workloads
        )
        centered_features = []
        centered_targets = []
        sample_weights = []
        training_workloads: dict[
            tuple[int, int, int, str, bool, bool, int], Workload
        ] = {}
        support_groups: dict[
            tuple[str, bool, bool, str], set[
                tuple[
                    tuple[int, int, int, str, bool, bool, int],
                    int,
                ]
            ],
        ] = defaultdict(set)

        informative_groups = 0
        for group_index, rows in enumerate(groups):
            features = np.asarray(
                [
                    latency_rank_features(
                        row.workload, row.schedule, hardware
                    )
                    for row in rows
                ],
                dtype=np.float64,
            )
            targets = np.asarray(
                [row.log_ratio for row in rows], dtype=np.float64
            )
            reliabilities = np.asarray(
                [row.reliability for row in rows], dtype=np.float64
            )
            if float(np.max(targets) - np.min(targets)) < math.log(1.005):
                continue
            informative_groups += 1
            weight_sum = float(np.sum(reliabilities))
            feature_mean = np.average(
                features, axis=0, weights=reliabilities
            )
            target_mean = float(
                np.average(targets, weights=reliabilities)
            )
            group_mass = (
                math.sqrt(len(rows))
                * float(np.mean(reliabilities))
            )
            weights = reliabilities / weight_sum * group_mass
            centered_features.append(features - feature_mean)
            centered_targets.append(targets - target_mean)
            sample_weights.append(weights)
            for row in rows:
                identity = row.workload.identity()
                training_workloads[identity] = row.workload
                support_groups[
                    (
                        row.workload.dtype,
                        row.workload.trans_a,
                        row.workload.trans_b,
                        template_of(row.schedule).value,
                    )
                ].add((identity, group_index))

        self.groups = informative_groups
        self.training_workloads = tuple(training_workloads.values())
        self.support_groups = {
            key: len(items) for key, items in support_groups.items()
        }
        if not centered_features:
            self.feature_mean = np.zeros(1, dtype=np.float64)
            self.feature_scale = np.ones(1, dtype=np.float64)
            self.coefficients = np.zeros(1, dtype=np.float64)
            self.residual = 1.0
            self.samples = 0
            return

        feature_matrix = np.concatenate(centered_features, axis=0)
        targets = np.concatenate(centered_targets)
        weights = np.concatenate(sample_weights)
        self.samples = len(targets)
        self.feature_mean = np.average(
            feature_matrix, axis=0, weights=weights
        )
        variance = np.average(
            (feature_matrix - self.feature_mean) ** 2,
            axis=0,
            weights=weights,
        )
        self.feature_scale = np.maximum(1.0e-6, np.sqrt(variance))
        normalized = (
            feature_matrix - self.feature_mean
        ) / self.feature_scale
        weighted = normalized * weights[:, np.newaxis]
        gram = normalized.T @ weighted
        regularization = ridge * max(1.0, float(np.sum(weights)))
        gram.flat[:: gram.shape[0] + 1] += regularization
        right_hand = normalized.T @ (weights * targets)
        try:
            self.coefficients = np.linalg.solve(gram, right_hand)
        except np.linalg.LinAlgError:
            self.coefficients = np.linalg.lstsq(
                gram, right_hand, rcond=None
            )[0]
        errors = normalized @ self.coefficients - targets
        self.residual = math.sqrt(
            float(np.average(errors * errors, weights=weights))
        )

    def score(
        self,
        workload: Workload,
        schedule: Schedule,
        hardware: Hardware,
    ) -> float:
        if self.samples == 0:
            return 0.0
        features = np.asarray(
            latency_rank_features(workload, schedule, hardware),
            dtype=np.float64,
        )
        normalized = (
            features - self.feature_mean
        ) / self.feature_scale
        return float(normalized @ self.coefficients)

    def compare(
        self,
        workload: Workload,
        incumbent: Candidate,
        candidate: Candidate,
        hardware: Hardware,
    ) -> PairwiseLatencyPrediction:
        if self.samples == 0:
            return PairwiseLatencyPrediction(1.0, 1.0, 0.0, math.inf)
        log_ratio = (
            self.score(workload, candidate.schedule, hardware)
            - self.score(workload, incumbent.schedule, hardware)
        )
        matching_groups = self.support_groups.get(
            (
                workload.dtype,
                workload.trans_a,
                workload.trans_b,
                candidate.template.value,
            ),
            0,
        )
        compatible = [
            item
            for item in self.training_workloads
            if (
                item.dtype == workload.dtype
                and item.trans_a == workload.trans_a
                and item.trans_b == workload.trans_b
            )
        ]
        nearest = min(
            (
                workload_distance(workload, item)
                for item in compatible
            ),
            default=math.inf,
        )
        distance_support = (
            math.exp(-nearest / 2.0) if math.isfinite(nearest) else 0.0
        )
        sample_support = min(
            1.0, math.log1p(matching_groups) / math.log(33.0)
        )
        support = sample_support * distance_support
        uncertainty = min(
            1.0,
            self.residual
            + (0.12 * nearest if math.isfinite(nearest) else 1.0)
            + 0.20 / math.sqrt(max(1.0, matching_groups)),
        )
        return PairwiseLatencyPrediction(
            relative_ratio=math.exp(
                max(math.log(0.10), min(math.log(10.0), log_ratio))
            ),
            uncertainty=uncertainty,
            support=support,
            nearest_workload_distance=nearest,
        )


def validate_pairwise_ranker(
    observations: Sequence[MeasuredObservation],
    hardware: Hardware,
    *,
    folds: int = 5,
    ridge: float = 0.05,
) -> PairwiseRankValidation:
    groups = _training_groups(observations)
    identities = sorted(
        {row.workload.identity() for group in groups for row in group}
    )
    fold_count = max(2, min(folds, len(identities)))
    identity_folds = {
        identity: index % fold_count
        for index, identity in enumerate(identities)
    }
    correct = 0
    comparisons = 0
    one_field_correct = 0
    one_field_pairs = 0
    base_one_field_correct = 0
    base_one_field_pairs = 0
    recalls = []
    best_percentiles = []
    regrets = []
    evaluated_groups = 0

    for fold in range(fold_count):
        held_out = frozenset(
            identity
            for identity, assigned in identity_folds.items()
            if assigned == fold
        )
        model = PairwiseLatencyRanker(
            observations,
            hardware,
            excluded_workloads=held_out,
            ridge=ridge,
        )
        for rows in groups:
            if rows[0].workload.identity() not in held_out:
                continue
            scores = [
                model.score(row.workload, row.schedule, hardware)
                for row in rows
            ]
            actual = [row.log_ratio for row in rows]
            informative = 0
            for left in range(len(rows)):
                for right in range(left):
                    actual_delta = actual[left] - actual[right]
                    if abs(actual_delta) < math.log(1.01):
                        continue
                    predicted_delta = scores[left] - scores[right]
                    if predicted_delta == 0:
                        continue
                    informative += 1
                    comparisons += 1
                    ordering_correct = (
                        predicted_delta * actual_delta > 0
                    )
                    correct += ordering_correct
                    if (
                        template_of(rows[left].schedule)
                        == template_of(rows[right].schedule)
                        and sum(
                            left_value != right_value
                            for left_value, right_value in zip(
                                rows[left].schedule.values,
                                rows[right].schedule.values,
                            )
                        )
                        == 1
                    ):
                        one_field_pairs += 1
                        one_field_correct += ordering_correct
                        if (
                            template_of(rows[left].schedule)
                            == Template.BASE
                        ):
                            base_one_field_pairs += 1
                            base_one_field_correct += ordering_correct
            if informative == 0:
                continue
            evaluated_groups += 1
            predicted_order = sorted(
                range(len(rows)), key=lambda index: scores[index]
            )
            actual_order = sorted(
                range(len(rows)), key=lambda index: actual[index]
            )
            count = max(1, ceil_div(len(rows), 4))
            recalls.append(
                len(
                    set(predicted_order[:count])
                    & set(actual_order[:count])
                )
                / count
            )
            best_index = actual_order[0]
            best_percentiles.append(
                predicted_order.index(best_index)
                / max(1, len(rows) - 1)
            )
            regrets.append(
                math.exp(
                    actual[predicted_order[0]] - actual[best_index]
                )
            )

    ordered_regrets = sorted(regrets)
    return PairwiseRankValidation(
        folds=fold_count,
        groups=evaluated_groups,
        informative_pairs=comparisons,
        pairwise_accuracy=correct / comparisons if comparisons else math.nan,
        one_field_pairs=one_field_pairs,
        one_field_accuracy=(
            one_field_correct / one_field_pairs
            if one_field_pairs
            else math.nan
        ),
        base_one_field_pairs=base_one_field_pairs,
        base_one_field_accuracy=(
            base_one_field_correct / base_one_field_pairs
            if base_one_field_pairs
            else math.nan
        ),
        top_quartile_recall=(
            sum(recalls) / len(recalls) if recalls else math.nan
        ),
        best_candidate_percentile=(
            sum(best_percentiles) / len(best_percentiles)
            if best_percentiles
            else math.nan
        ),
        median_top1_regret=(
            median(ordered_regrets) if ordered_regrets else math.nan
        ),
        p90_top1_regret=(
            ordered_regrets[
                int(0.9 * (len(ordered_regrets) - 1))
            ]
            if ordered_regrets
            else math.nan
        ),
    )
