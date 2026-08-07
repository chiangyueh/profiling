from __future__ import annotations

from .behavior import behavior_key, behavior_vector
from .contracts import align_up, ceil_div, validate_schedule
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


DIRECT_BASE_RULE_VERSION = "template_rule_v2"
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


def _is_256b_aligned(elements: int, element_bytes: int) -> bool:
    return elements * element_bytes % 256 == 0


def _use_deterministic_split_k(
    workload: Workload,
    hardware: Hardware,
) -> tuple[bool, str]:
    """Distill the source-level split-K profitability decision.

    These conditions select a kernel execution structure. They do not use a
    workload name, history row, fitted model, or RuntimeKb geometry.
    """

    cores = min(workload.max_cores, hardware.aic_cores)
    m_blocks = ceil_div(workload.m, 128)
    n_blocks = ceil_div(workload.n, 128)
    output_blocks = m_blocks * n_blocks
    if (
        workload.k >= cores * 384
        and output_blocks <= cores
        and not (not workload.trans_a and workload.trans_b)
    ):
        return True, "underfilled_output_k_parallelism"

    if (
        workload.dtype == "fp32"
        and workload.m <= 64
        and workload.n <= 64
        and workload.k >= 6144
    ):
        return True, "fp32_small_output_k_parallelism"

    if (
        workload.dtype == "fp32"
        and workload.trans_a
        and not workload.trans_b
        and 1000 <= workload.k <= 4608
        and workload.m <= 256
        and workload.n <= 256
    ):
        return True, "fp32_tn_small_output"

    return False, "base_output_parallelism_sufficient"


def _prefer_nk_split_layout(workload: Workload) -> bool:
    in_bytes = 4 if workload.dtype == "fp32" else 2
    n_32b_aligned = workload.n * in_bytes % 32 == 0
    if workload.m <= workload.n or not n_32b_aligned:
        return False
    if workload.m < 128:
        return False
    if (
        _is_256b_aligned(workload.n, in_bytes)
        and not _is_256b_aligned(workload.m, in_bytes)
    ):
        return False
    if (
        not _is_256b_aligned(workload.m, in_bytes)
        and workload.n * in_bytes <= 256
    ):
        return False
    if workload.m >= 2048 and workload.n == 16:
        return False
    return True


def direct_rule_candidate(
    workload: Workload,
    hardware: Hardware,
) -> Candidate:
    """Return one template-aware schedule using fixed source-derived rules."""

    use_split_k, reason = _use_deterministic_split_k(workload, hardware)
    if not use_split_k:
        base = direct_base_candidate(workload, hardware)
        return Candidate(
            schedule=base.schedule,
            template=base.template,
            source="direct_rule_policy",
            rationale=f"fixed template rule selected BASE: {reason}",
            behavior_key=base.behavior_key,
            metrics={
                **base.metrics,
                "rule_version": 2.0,
                "policy_template_base": 1.0,
                "policy_template_deterministic_split_k": 0.0,
            },
        )

    in_bytes = 4 if workload.dtype == "fp32" else 2
    base_k = 256 // in_bytes
    single_k = 3 * base_k
    k_chunks = ceil_div(workload.k, single_k)
    used_cores = min(
        workload.max_cores, hardware.aic_cores, k_chunks
    )
    prefer_nk = _prefer_nk_split_layout(workload)
    if prefer_nk:
        single_m = 384
        single_n = max(128, align_up(workload.n, 16))
        step_m, step_n = 3, 1
        depth_a, depth_b = 9, 6
        iterate_order, l2_order = 1, 0
    else:
        single_m = max(128, align_up(workload.m, 16))
        single_n = 384
        step_m, step_n = 1, 3
        depth_a, depth_b = 6, 9
        iterate_order, l2_order = 0, 1
    schedule = make_schedule(
        usedCoreNum=used_cores,
        singleCoreM=single_m,
        singleCoreN=single_n,
        singleCoreK=single_k,
        baseM=128,
        baseN=128,
        baseK=base_k,
        depthA1=depth_a,
        depthB1=depth_b,
        stepM=step_m,
        stepN=step_n,
        iterateOrder=iterate_order,
        stepKa=3,
        stepKb=3,
        dbL0A=2,
        dbL0B=2,
        dbL0C=2,
        l2MTileCnt=1,
        l2NTileCnt=1,
        l2MTileBlock=max(1, ceil_div(workload.m, single_m)),
        l2NTileBlock=max(1, ceil_div(workload.n, single_n)),
        l2IterateOrder=l2_order,
        tilingEnable=3,
    )
    contract = validate_schedule(workload, schedule, hardware)
    if not contract.valid:
        raise ValueError(
            "direct deterministic split-K policy violates contract: "
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
            "rule_version": 2.0,
            "policy_template_base": 0.0,
            "policy_template_deterministic_split_k": 1.0,
        }
    )
    return Candidate(
        schedule=schedule,
        template=Template.DETERMINISTIC_SPLIT_K,
        source="direct_rule_policy",
        rationale=(
            "fixed template rule selected deterministic split-K: "
            f"{reason}"
        ),
        behavior_key=behavior_key(vector),
        metrics=metrics,
    )
