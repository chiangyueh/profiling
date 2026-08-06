from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Sequence

from .behavior import behavior_key, behavior_vector
from .contracts import ceil_div, template_of, validate_schedule
from .domain import (
    INPUT_BYTES,
    OUTPUT_BYTES,
    Candidate,
    Hardware,
    MeasuredObservation,
    Schedule,
    Template,
    Workload,
)
from .solvers.base_policy import (
    l1_pipeline_variant,
    upstream_base_geometry_policy,
    upstream_base_l2_policy,
)
from .solvers.common import make_schedule


@dataclass(frozen=True)
class DirectBaseEvidence:
    paired_base_records: int
    unique_base_records: int
    base_workloads: int
    winner_workloads: int
    bank_base_workloads: int
    geometry_policy_matches: int
    l1_policy_matches: int
    core_policy_matches: int
    whole_l2_bank_workloads: int
    partitioned_l2_bank_workloads: int
    resident_l2_threshold_bytes: float


def fit_direct_base_evidence(
    observations: Sequence[MeasuredObservation],
    hardware: Hardware,
) -> DirectBaseEvidence:
    """Distill strict paired BASE evidence into one global hardware policy.

    Candidate measurements quantify BASE outcomes. Bank records validate the
    source-derived geometry/L1/core formulas and calibrate the transition
    from whole-resident to partitioned L2. No workload names or manually
    defined shape families participate.
    """

    paired = [
        observation
        for observation in observations
        if observation.verified
        and observation.source
        not in {"runtime_rejected", "runtime_verified"}
        and template_of(observation.schedule) == Template.BASE
    ]
    unique: dict[
        tuple[
            tuple[int, int, int, str, bool, bool, int],
            tuple[int, ...],
        ],
        list[MeasuredObservation],
    ] = {}
    for observation in paired:
        unique.setdefault(
            (
                observation.workload.identity(),
                observation.schedule.signature(),
            ),
            [],
        ).append(observation)

    best_by_workload: dict[
        tuple[int, int, int, str, bool, bool, int],
        float,
    ] = {}
    for records in unique.values():
        workload_identity = records[0].workload.identity()
        ratio = median(
            record.measured_ratio for record in records
        )
        best_by_workload[workload_identity] = min(
            ratio,
            best_by_workload.get(workload_identity, float("inf")),
        )

    bank_by_workload: dict[
        tuple[int, int, int, str, bool, bool, int],
        tuple[Workload, Schedule],
    ] = {}
    for observation in observations:
        bank = observation.bank_schedule
        if (
            observation.verified
            and bank is not None
            and template_of(bank) == Template.BASE
        ):
            bank_by_workload[observation.workload.identity()] = (
                observation.workload,
                bank,
            )

    geometry_policy_matches = 0
    l1_policy_matches = 0
    core_policy_matches = 0
    whole_l2_sizes = []
    partitioned_l2_sizes = []
    for workload, schedule in bank_by_workload.values():
        geometry_key = (
            schedule["baseM"],
            schedule["baseN"],
            schedule["baseK"],
            schedule["dbL0A"],
            schedule["dbL0B"],
            schedule["dbL0C"],
        )
        if geometry_key == upstream_base_geometry_policy(
            workload, hardware
        ):
            geometry_policy_matches += 1

        l1_key = (
            schedule["depthA1"],
            schedule["depthB1"],
            schedule["stepM"],
            schedule["stepN"],
            schedule["stepKa"],
            schedule["stepKb"],
        )
        expected_l1 = l1_pipeline_variant(
            workload,
            hardware,
            base_m=schedule["baseM"],
            base_n=schedule["baseN"],
            base_k=schedule["baseK"],
        )
        if l1_key == expected_l1:
            l1_policy_matches += 1

        expected_core = (
            schedule["baseM"],
            schedule["baseN"],
            min(workload.max_cores, hardware.aic_cores),
        )
        if expected_core == (
            schedule["singleCoreM"],
            schedule["singleCoreN"],
            schedule["usedCoreNum"],
        ):
            core_policy_matches += 1

        m_tasks = ceil_div(workload.m, schedule["singleCoreM"])
        n_tasks = ceil_div(workload.n, schedule["singleCoreN"])
        l2_key = (
            schedule["l2MTileCnt"],
            schedule["l2NTileCnt"],
            schedule["l2MTileBlock"],
            schedule["l2NTileBlock"],
        )
        total_bytes = (
            workload.m * workload.k * INPUT_BYTES[workload.dtype]
            + workload.k * workload.n * INPUT_BYTES[workload.dtype]
            + workload.m
            * workload.n
            * OUTPUT_BYTES[workload.dtype]
        )
        if l2_key == (1, 1, m_tasks, n_tasks):
            whole_l2_sizes.append(total_bytes)
        else:
            partitioned_l2_sizes.append(total_bytes)

    reference_threshold = min(
        float(hardware.l2_bytes),
        100.0
        * 1024.0
        * 1024.0
        * hardware.l2_bytes
        / (192.0 * 1024.0 * 1024.0),
    )
    resident_threshold = reference_threshold
    if (
        whole_l2_sizes
        and partitioned_l2_sizes
        and max(whole_l2_sizes) < min(partitioned_l2_sizes)
    ):
        resident_threshold = math.sqrt(
            max(whole_l2_sizes) * min(partitioned_l2_sizes)
        )

    return DirectBaseEvidence(
        paired_base_records=len(paired),
        unique_base_records=len(unique),
        base_workloads=len(best_by_workload),
        winner_workloads=sum(
            ratio < 0.99 for ratio in best_by_workload.values()
        ),
        bank_base_workloads=len(bank_by_workload),
        geometry_policy_matches=geometry_policy_matches,
        l1_policy_matches=l1_policy_matches,
        core_policy_matches=core_policy_matches,
        whole_l2_bank_workloads=len(whole_l2_sizes),
        partitioned_l2_bank_workloads=len(partitioned_l2_sizes),
        resident_l2_threshold_bytes=resident_threshold,
    )


def direct_base_candidate(
    workload: Workload,
    hardware: Hardware,
    evidence: DirectBaseEvidence | None = None,
) -> Candidate:
    """Build one history-distilled BASE record without a bank lookup.

    Strict paired history validates the source-derived geometry/L1/core
    formulas and calibrates the L2 resident boundary. The resulting path
    constructs and sends exactly one complete 23-field record.
    """

    (
        base_m,
        base_n,
        base_k,
        db_a,
        db_b,
        db_c,
    ) = upstream_base_geometry_policy(workload, hardware)

    l1 = l1_pipeline_variant(
        workload,
        hardware,
        base_m=base_m,
        base_n=base_n,
        base_k=base_k,
    )
    if l1 is None:
        raise ValueError("direct BASE policy cannot construct an L1 pipeline")
    depth_a, depth_b, step_m, step_n, step_ka, step_kb = l1

    single_m, single_n = base_m, base_n
    used_cores = min(workload.max_cores, hardware.aic_cores)

    l2, l2_mode = upstream_base_l2_policy(
        workload,
        hardware,
        base_m=base_m,
        base_n=base_n,
        resident_threshold_bytes=(
            evidence.resident_l2_threshold_bytes
            if evidence is not None
            else None
        ),
    )
    (
        l2_m_count,
        l2_n_count,
        l2_m_block,
        l2_n_block,
        l2_order,
    ) = l2
    schedule = make_schedule(
        usedCoreNum=used_cores,
        singleCoreM=single_m,
        singleCoreN=single_n,
        singleCoreK=workload.k,
        baseM=base_m,
        baseN=base_n,
        baseK=base_k,
        depthA1=depth_a,
        depthB1=depth_b,
        stepM=step_m,
        stepN=step_n,
        iterateOrder=0,
        stepKa=step_ka,
        stepKb=step_kb,
        dbL0A=db_a,
        dbL0B=db_b,
        dbL0C=db_c,
        l2MTileCnt=l2_m_count,
        l2NTileCnt=l2_n_count,
        l2MTileBlock=l2_m_block,
        l2NTileBlock=l2_n_block,
        l2IterateOrder=l2_order,
        tilingEnable=0,
    )
    contract = validate_schedule(workload, schedule, hardware)
    if not contract.valid:
        raise ValueError(
            "direct BASE policy violates contract: "
            + ",".join(contract.violations)
        )

    vector = behavior_vector(workload, schedule, hardware)
    metrics = dict(vector.metrics)
    metrics.update(
        {
            "model_enabled": 0.0,
            "deployment_recommended_custom": 1.0,
            "history_rows_used": float(
                evidence.paired_base_records if evidence is not None else 0
            ),
            "candidate_pool_size": 1.0,
            "policy_upstream_geometry": 1.0,
            "policy_upstream_l1": 1.0,
            "policy_upstream_core_grid": 1.0,
            "policy_l2_mode_whole": float(
                l2_mode == "whole_resident"
            ),
            "policy_l2_mode_cache": float(
                l2_mode == "cache_partitioned"
            ),
            "policy_l2_mode_fallback": float(
                l2_mode == "fixed_fallback"
            ),
            "policy_l2_resident_threshold_bytes": float(
                evidence.resident_l2_threshold_bytes
                if evidence is not None
                else 0.0
            ),
        }
    )
    if evidence is not None:
        metrics.update(
            {
                "policy_evidence_unique_base_records": float(
                    evidence.unique_base_records
                ),
                "policy_evidence_base_workloads": float(
                    evidence.base_workloads
                ),
                "policy_evidence_winner_workloads": float(
                    evidence.winner_workloads
                ),
                "policy_evidence_bank_base_workloads": float(
                    evidence.bank_base_workloads
                ),
                "policy_evidence_geometry_matches": float(
                    evidence.geometry_policy_matches
                ),
                "policy_evidence_l1_matches": float(
                    evidence.l1_policy_matches
                ),
                "policy_evidence_core_matches": float(
                    evidence.core_policy_matches
                ),
            }
        )
    return Candidate(
        schedule=schedule,
        template=Template.BASE,
        source="direct_base_policy",
        rationale=(
            "single history-distilled BASE record using upstream-coupled "
            f"geometry/L1/core and {l2_mode} L2 policy"
        ),
        behavior_key=behavior_key(vector),
        metrics=metrics,
    )
