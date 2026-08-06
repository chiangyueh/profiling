from __future__ import annotations

from .behavior import behavior_key, behavior_vector
from .contracts import validate_schedule
from .domain import (
    Candidate,
    Hardware,
    Template,
    Workload,
)
from .solvers.base_policy import (
    l1_pipeline_variant,
    upstream_base_geometry_policy,
    upstream_base_l2_policy,
)
from .solvers.common import make_schedule


DIRECT_BASE_RULE_VERSION = "base_rule_v1"
DIRECT_BASE_L2_RESIDENT_RATIO = 0.5105593326137546
DIRECT_BASE_AUDIT_PAIRED_RECORDS = 498
DIRECT_BASE_AUDIT_UNIQUE_RECORDS = 373
DIRECT_BASE_AUDIT_WORKLOADS = 78
DIRECT_BASE_AUDIT_WINNER_WORKLOADS = 15
DIRECT_BASE_AUDIT_BANK_WORKLOADS = 58


def direct_base_candidate(
    workload: Workload,
    hardware: Hardware,
) -> Candidate:
    """Build one fixed-rule BASE record without history or a bank lookup.

    Historical NPU data was used offline to validate the geometry/L1/core
    formulas and freeze the L2 resident ratio. Deployment performs no model
    fit, candidate enumeration, ranking, or feedback loading.
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
            DIRECT_BASE_L2_RESIDENT_RATIO * hardware.l2_bytes
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
            "history_rows_used": 0.0,
            "candidate_pool_size": 1.0,
            "rule_version": 1.0,
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
                DIRECT_BASE_L2_RESIDENT_RATIO * hardware.l2_bytes
            ),
            "rule_audit_paired_records": float(
                DIRECT_BASE_AUDIT_PAIRED_RECORDS
            ),
            "rule_audit_unique_records": float(
                DIRECT_BASE_AUDIT_UNIQUE_RECORDS
            ),
            "rule_audit_workloads": float(
                DIRECT_BASE_AUDIT_WORKLOADS
            ),
            "rule_audit_winner_workloads": float(
                DIRECT_BASE_AUDIT_WINNER_WORKLOADS
            ),
            "rule_audit_bank_workloads": float(
                DIRECT_BASE_AUDIT_BANK_WORKLOADS
            ),
        }
    )
    return Candidate(
        schedule=schedule,
        template=Template.BASE,
        source="direct_base_policy",
        rationale=(
            "single fixed-rule BASE record using source-coupled "
            f"geometry/L1/core and {l2_mode} L2 policy"
        ),
        behavior_key=behavior_key(vector),
        metrics=metrics,
    )
