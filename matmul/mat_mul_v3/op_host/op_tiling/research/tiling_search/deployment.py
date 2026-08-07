from __future__ import annotations

from .behavior import behavior_key, behavior_vector
from .contracts import (
    align_up,
    bl1_core_split_applicable,
    ceil_div,
    validate_schedule,
)
from .domain import (
    Candidate,
    Hardware,
    INPUT_BYTES,
    OUTPUT_BYTES,
    Template,
    Workload,
)
from .solvers.base_policy import (
    l1_pipeline_variant,
    upstream_base_geometry_policy,
    upstream_base_l2_policy,
)
from .solvers.common import make_schedule


DIRECT_BASE_RULE_VERSION = "template_rule_v6"
DIRECT_BASE_L2_RESIDENT_RATIO = 0.5105593326137546
DIRECT_BASE_AUDIT_PAIRED_RECORDS = 398
DIRECT_BASE_AUDIT_UNIQUE_RECORDS = 317
DIRECT_BASE_AUDIT_WORKLOADS = 79
DIRECT_BASE_AUDIT_WINNER_WORKLOADS = 2
DIRECT_BASE_AUDIT_BANK_WORKLOADS = 48
DIRECT_RULE_AUDIT_RECORDS = 7850
DIRECT_RULE_AUDIT_UNIQUE_RECORDS = 7534
DIRECT_RULE_AUDIT_WORKLOADS = 241
DIRECT_RULE_TRUSTED_WINNER_WORKLOADS = 17
DIRECT_RULE_TRUSTED_WINNER_TEMPLATE_MATCHES = 14
DIRECT_RULE_TRUSTED_WINNER_EXECUTION_EQUIVALENT = 5
DIRECT_RULE_TRUSTED_WINNER_STRUCTURAL_NEAR = 6
DIRECT_RULE_TEMPLATE_AUDIT = {
    Template.BASE: (128, 11, 89, 28),
    Template.AL1_FULL_LOAD: (8, 3, 4, 1),
    Template.BL1_FULL_LOAD: (30, 1, 22, 7),
    Template.BL1_FULL_LOAD_FIXPIPE: (12, 0, 12, 0),
    Template.BL1_FULL_LOAD_VEC_NZ2ND: (11, 1, 10, 0),
    Template.DETERMINISTIC_SPLIT_K: (68, 4, 24, 40),
    Template.SINGLE_CORE_SPLIT_K: (98, 3, 48, 47),
}


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
    """Distill source contracts and trusted paired split-K outcomes.

    These conditions select a kernel execution structure. They do not use a
    workload name, history row, fitted model, or RuntimeKb geometry.
    """

    cores = min(workload.max_cores, hardware.aic_cores)
    m_blocks = ceil_div(workload.m, 128)
    n_blocks = ceil_div(workload.n, 128)
    output_blocks = m_blocks * n_blocks
    if (
        workload.k >= cores * 384
        and output_blocks < cores
        and not (not workload.trans_a and workload.trans_b)
    ):
        return True, "underfilled_output_k_parallelism"

    if (
        workload.k >= cores * 384
        and workload.k >= 8 * (workload.m + workload.n)
        and not (not workload.trans_a and workload.trans_b)
    ):
        return True, "reduction_dominated_k_parallelism"

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


def _candidate(
    workload: Workload,
    hardware: Hardware,
    schedule,
    template: Template,
    reason: str,
) -> Candidate:
    contract = validate_schedule(workload, schedule, hardware)
    if not contract.valid:
        raise ValueError(
            f"direct {template.value} policy violates contract: "
            + ",".join(contract.violations)
        )
    vector = behavior_vector(workload, schedule, hardware)
    metrics = dict(vector.metrics)
    audited, winners, within_noise, regressions = (
        DIRECT_RULE_TEMPLATE_AUDIT[template]
    )
    metrics.update(
        {
            "model_enabled": 0.0,
            "deployment_recommended_custom": 1.0,
            "history_rows_used": 0.0,
            "candidate_pool_size": 1.0,
            "rule_version": 6.0,
            "rule_audit_total_records": float(
                DIRECT_RULE_AUDIT_RECORDS
            ),
            "rule_audit_unique_records": float(
                DIRECT_RULE_AUDIT_UNIQUE_RECORDS
            ),
            "rule_audit_workloads": float(
                DIRECT_RULE_AUDIT_WORKLOADS
            ),
            "rule_audit_template_workloads": float(audited),
            "rule_audit_template_winners": float(winners),
            "rule_audit_template_within_noise": float(within_noise),
            "rule_audit_template_regressions": float(regressions),
            "policy_template_base": float(template == Template.BASE),
            "policy_template_al1_full_load": float(
                template == Template.AL1_FULL_LOAD
            ),
            "policy_template_bl1_full_load": float(
                template
                in {
                    Template.BL1_FULL_LOAD,
                    Template.BL1_FULL_LOAD_FIXPIPE,
                    Template.BL1_FULL_LOAD_VEC_NZ2ND,
                }
            ),
            "policy_template_single_core_split_k": float(
                template == Template.SINGLE_CORE_SPLIT_K
            ),
            "policy_template_deterministic_split_k": float(
                template == Template.DETERMINISTIC_SPLIT_K
            ),
        }
    )
    return Candidate(
        schedule=schedule,
        template=template,
        source="direct_rule_policy",
        rationale=(
            "fixed source-and-offline-evidence-derived template rule: "
            f"{reason}"
        ),
        behavior_key=behavior_key(vector),
        metrics=metrics,
    )


def _al1_candidate(
    workload: Workload,
    hardware: Hardware,
) -> Candidate | None:
    """Port the 8.5 DoAL1FullLoadTiling constructor and gate."""

    if (
        workload.dtype != "fp32"
        or workload.trans_a
        or not workload.trans_b
        or workload.m > 16
        or workload.n <= 16
        or workload.n > 16 * min(workload.max_cores, hardware.aic_cores)
        or workload.k < 4096
        or workload.k % 128
    ):
        return None
    base_m = base_n = 16
    base_k = 256
    step_k = ceil_div(workload.k, base_k)
    resident_a = align_up(workload.m, 16) * align_up(
        workload.k, 8
    ) * INPUT_BYTES[workload.dtype]
    staged_b = (
        base_n * base_k * 2 * INPUT_BYTES[workload.dtype]
    )
    if resident_a + staged_b > hardware.effective_l1_bytes:
        return None
    schedule = make_schedule(
        usedCoreNum=min(
            workload.max_cores,
            hardware.aic_cores,
            ceil_div(workload.n, base_n),
        ),
        singleCoreM=workload.m,
        singleCoreN=base_n,
        singleCoreK=workload.k,
        baseM=base_m,
        baseN=base_n,
        baseK=base_k,
        depthA1=step_k,
        depthB1=2,
        stepM=1,
        stepN=1,
        iterateOrder=0,
        stepKa=step_k,
        stepKb=1,
        dbL0A=2,
        dbL0B=2,
        dbL0C=2,
        l2MTileCnt=1,
        l2NTileCnt=1,
        l2MTileBlock=1,
        l2NTileBlock=ceil_div(workload.n, base_n),
        l2IterateOrder=1,
        tilingEnable=10,
    )
    return _candidate(
        workload,
        hardware,
        schedule,
        Template.AL1_FULL_LOAD,
        "upstream_al1_resident_a",
    )


def _bl1_fix_candidate(
    workload: Workload,
    hardware: Hardware,
) -> Candidate | None:
    """Port NeedSolveFixBound and DoBL1FullloadWithFixpipeTiling."""

    if workload.trans_a or workload.k > 256 or workload.n >= 256:
        return None
    out_alignment = 256 // OUTPUT_BYTES[workload.dtype]
    if (
        workload.n % out_alignment == 0
        or out_alignment % workload.n == 0
        or workload.m
        < min(workload.max_cores, hardware.aic_cores) * 512
    ):
        return None
    c0 = 32 // INPUT_BYTES[workload.dtype]
    if workload.n < c0 and workload.k < c0:
        return None
    if workload.dtype in {"fp16", "bf16"}:
        mode = 1020
        template = Template.BL1_FULL_LOAD_FIXPIPE
    elif (
        workload.dtype == "fp32"
        and workload.k % c0 == 0
        and workload.n <= 192
    ):
        mode = 2020
        template = Template.BL1_FULL_LOAD_VEC_NZ2ND
    else:
        return None
    if template == Template.BL1_FULL_LOAD_VEC_NZ2ND:
        base_n = align_up(workload.n, 16)
        ub_bytes = hardware.ub_bytes or hardware.l0c_bytes
        base_m = min(
            hardware.l0c_capacity(1) // max(1, base_n * 4),
            ub_bytes // max(1, base_n * OUTPUT_BYTES[workload.dtype]),
            hardware.l0a_bytes
            // max(1, 2 * INPUT_BYTES[workload.dtype] * 32),
        )
        base_m = base_m // 128 * 128
        if base_m < 128:
            return None
        base_k = min(
            hardware.l0a_bytes
            // max(1, 2 * INPUT_BYTES[workload.dtype] * base_m),
            hardware.l0b_bytes
            // max(1, 2 * INPUT_BYTES[workload.dtype] * base_n),
        )
        base_k = base_k // 16 * 16
        if base_k < 16:
            return None
        depth_b = ceil_div(workload.k, base_k)
        depth_a = (
            hardware.effective_l1_bytes
            // INPUT_BYTES[workload.dtype]
            - depth_b * base_n * base_k
        ) // max(1, base_m * base_k)
        depth_a = min(depth_a, depth_b)
        if depth_a <= 0:
            return None
        geometry = {
            "baseM": base_m,
            "baseN": base_n,
            "baseK": base_k,
            "depthA1": min(depth_a, 8),
            "depthB1": depth_b,
            "stepKa": 4 if depth_a >= 8 else depth_a,
            "stepKb": depth_b,
        }
    else:
        from .contracts import bl1_fix_geometry

        geometry = bl1_fix_geometry(workload, hardware)
    if geometry is None:
        return None
    schedule = make_schedule(
        usedCoreNum=min(
            workload.max_cores,
            hardware.aic_cores,
            ceil_div(workload.m, geometry["baseM"]),
        ),
        singleCoreM=geometry["baseM"],
        singleCoreN=geometry["baseN"],
        singleCoreK=workload.k,
        baseM=geometry["baseM"],
        baseN=geometry["baseN"],
        baseK=geometry["baseK"],
        depthA1=geometry["depthA1"],
        depthB1=geometry["depthB1"],
        stepM=1,
        stepN=1,
        iterateOrder=0,
        stepKa=geometry["stepKa"],
        stepKb=geometry["stepKb"],
        dbL0A=2,
        dbL0B=2,
        dbL0C=1,
        l2MTileCnt=1,
        l2NTileCnt=1,
        l2MTileBlock=0,
        l2NTileBlock=0,
        l2IterateOrder=0,
        tilingEnable=mode,
    )
    return _candidate(
        workload,
        hardware,
        schedule,
        template,
        "upstream_bl1_fixpipe_bound",
    )


def _bl1_candidate(
    workload: Workload,
    hardware: Hardware,
) -> Candidate | None:
    """Construct the upstream BL1-resident policy without bank geometry."""

    if bl1_core_split_applicable(workload):
        _, _, base_k, db_a, db_b, _ = (
            upstream_base_geometry_policy(workload, hardware)
        )
        base_m = 128
        base_n = workload.n // 2
        single_m = align_up(ceil_div(workload.m, 12), base_m)
        step_k = ceil_div(workload.k, base_k)
        schedule = make_schedule(
            usedCoreNum=min(
                workload.max_cores,
                hardware.aic_cores,
                2 * ceil_div(workload.m, single_m),
            ),
            singleCoreM=single_m,
            singleCoreN=base_n,
            singleCoreK=workload.k,
            baseM=base_m,
            baseN=base_n,
            baseK=base_k,
            depthA1=2 * step_k,
            depthB1=step_k,
            stepM=1,
            stepN=1,
            iterateOrder=0,
            stepKa=step_k,
            stepKb=step_k,
            dbL0A=db_a,
            dbL0B=db_b,
            dbL0C=1,
            l2MTileCnt=1,
            l2NTileCnt=1,
            l2MTileBlock=ceil_div(workload.m, single_m),
            l2NTileBlock=1,
            l2IterateOrder=1,
            tilingEnable=20,
        )
        return _candidate(
            workload,
            hardware,
            schedule,
            Template.BL1_FULL_LOAD,
            "upstream_bl1_n512_core_split",
        )

    in_bytes = INPUT_BYTES[workload.dtype]
    supported_n = {32, 64, 96, 128, 160, 192, 224, 256, 384}
    on_the_fly = (
        workload.n in supported_n
        and (
            not workload.trans_b
            or workload.k * in_bytes in supported_n
        )
    )
    c0 = 32 // in_bytes
    inner_a = workload.m if workload.trans_a else workload.k
    outer_a = workload.k if workload.trans_a else workload.m
    vnchw_a = (
        workload.dtype == "fp32"
        and outer_a >= 72368
        and 1 < inner_a <= c0
    )
    valid_mk = (
        workload.m > 16 * max(workload.k, workload.n)
        and workload.k <= 256
    )
    resident_b = workload.k * workload.n * in_bytes
    if (
        not valid_mk
        or not (
            on_the_fly
            or (
                vnchw_a
                and resident_b < hardware.effective_l1_bytes // 2
            )
        )
    ):
        return None
    base_m, base_n, base_k, db_a, db_b, db_c = (
        upstream_base_geometry_policy(workload, hardware)
    )
    base_k = max(base_k, 128 // in_bytes)
    n_alignment = 16 if workload.trans_b else 32 // in_bytes
    base_n = align_up(min(workload.n, base_n), n_alignment)
    step_n = ceil_div(workload.n, base_n)
    step_k = ceil_div(workload.k, base_k)
    depth_a = 2 * step_k
    depth_b = step_n * step_k
    while base_m >= 16:
        l1_bytes = base_k * (
            depth_a * base_m + depth_b * base_n
        ) * in_bytes
        if l1_bytes <= hardware.effective_l1_bytes:
            break
        base_m //= 2
    if base_m < 16:
        return None
    db_c = (
        2
        if base_m * base_n * 4 * 2 <= hardware.l0c_bytes
        else 1
    )
    single_m = 2 * base_m
    schedule = make_schedule(
        usedCoreNum=min(
            workload.max_cores,
            hardware.aic_cores,
            ceil_div(workload.m, single_m),
        ),
        singleCoreM=single_m,
        singleCoreN=workload.n,
        singleCoreK=workload.k,
        baseM=base_m,
        baseN=base_n,
        baseK=base_k,
        depthA1=depth_a,
        depthB1=depth_b,
        stepM=1,
        stepN=step_n,
        iterateOrder=0,
        stepKa=step_k,
        stepKb=step_k,
        dbL0A=db_a,
        dbL0B=db_b,
        dbL0C=db_c,
        l2MTileCnt=1,
        l2NTileCnt=1,
        l2MTileBlock=ceil_div(workload.m, single_m),
        l2NTileBlock=1,
        l2IterateOrder=1,
        tilingEnable=20,
    )
    return _candidate(
        workload,
        hardware,
        schedule,
        Template.BL1_FULL_LOAD,
        "upstream_bl1_resident_b",
    )


def _ceil_core_factor(value: int, cores: int) -> int:
    factors = {
        20: (1, 2, 4, 5, 10, 20),
        24: (1, 2, 3, 4, 6, 8, 12, 24),
        32: (1, 2, 4, 8, 16, 32),
    }.get(cores, tuple(range(1, cores + 1)))
    for factor in factors:
        if value <= factor:
            return factor
    tail = value % cores
    return (
        ceil_div(value, cores) * cores
        if tail > cores // 2
        else max(cores, value // cores * cores)
    )


def _single_split_applicable(
    workload: Workload,
    hardware: Hardware,
) -> bool:
    cores = min(workload.max_cores, hardware.aic_cores)
    if workload.k >= 27392:
        return True
    base_m, base_n, _, _, _, _ = upstream_base_geometry_policy(
        workload, hardware
    )
    _, l2_mode = upstream_base_l2_policy(
        workload,
        hardware,
        base_m=base_m,
        base_n=base_n,
        resident_threshold_bytes=(
            DIRECT_BASE_L2_RESIDENT_RATIO * hardware.l2_bytes
        ),
    )
    enough_cube_work = (
        workload.m * workload.k >= 5 * (3 * 128) ** 2
        and workload.n >= 1024
        and workload.m >= 3 * 128
        and workload.k >= 3 * 128
        and workload.m * workload.n
        >= 1024 * (3 * 128) * cores
    )
    if (
        not workload.trans_a
        and not workload.trans_b
        and l2_mode != "cache_partitioned"
        and enough_cube_work
    ):
        return True
    if (
        workload.trans_a
        and not workload.trans_b
        and workload.k >= 11000
        and workload.n * INPUT_BYTES[workload.dtype] % 256 == 0
        and (
            (workload.m % 8192 == 0 and workload.n >= 6144)
            or (workload.n % 8192 == 0 and workload.m >= 6144)
        )
    ):
        return True
    return False


def _single_split_candidate(
    workload: Workload,
    hardware: Hardware,
) -> Candidate | None:
    """Port the 20-AIC single-core split-K partition procedure."""

    if not _single_split_applicable(workload, hardware):
        return None
    in_bytes = INPUT_BYTES[workload.dtype]
    cores = min(workload.max_cores, hardware.aic_cores)
    base_k = 256 // in_bytes
    if workload.k <= 4 * base_k:
        return None
    step_m, step_n, step_k = 3, 1, 3
    depth_a, depth_b, order = 9, 6, 1
    m_tile = ceil_div(workload.m, step_m * 128)
    n_tile = ceil_div(workload.n, 3072)
    if m_tile * n_tile < cores:
        step_m, step_n, step_k = 2, 1, 4
        depth_a = depth_b = 8
    if workload.m < 384 or workload.n < 384:
        step_m = step_n = 1
        step_k = 4
        depth_a = depth_b = 8
    m_align = (256 if workload.trans_a else 32) // in_bytes
    n_align = 256 // in_bytes
    m_tile = ceil_div(workload.m, step_m * 128)
    n_tile = _ceil_core_factor(
        ceil_div(workload.n, 3072), cores
    )
    choices: list[tuple[int, int, int]] = []
    if m_tile * n_tile >= cores:
        m_parts = max(1, cores // max(1, n_tile))
        single_m = min(
            workload.m,
            align_up(ceil_div(workload.m, m_parts), m_align),
        )
        single_n = min(
            workload.n,
            align_up(ceil_div(workload.n, n_tile), n_align),
        )
        used = min(
            cores,
            ceil_div(workload.m, single_m)
            * ceil_div(workload.n, single_n),
        )
        choices.append((used, single_m, single_n))
    total = 0
    probe_n = max(1, n_tile)
    while probe_n <= cores:
        probe_n = _ceil_core_factor(probe_n, cores)
        m_parts = max(1, cores // max(1, probe_n))
        single_n = align_up(
            ceil_div(workload.n, probe_n), n_align
        )
        single_m = align_up(
            ceil_div(workload.m, m_parts), m_align
        )
        used = ceil_div(workload.m, single_m) * ceil_div(
            workload.n, single_n
        )
        if used > total:
            total = used
            choices.append(
                (
                    min(cores, used),
                    min(single_m, workload.m),
                    min(single_n, workload.n),
                )
            )
        if probe_n == cores:
            break
        probe_n += 1
    if not choices:
        return None
    used, single_m, single_n = max(
        choices, key=lambda item: (item[0], -item[1] * item[2])
    )
    average = (
        workload.m
        * workload.n
        / max(1, single_m * single_n * cores)
    )
    if (
        single_n < 512
        or average < 0.70
        or (
            workload.n * in_bytes % 256 == 0
            and 896 <= workload.n <= 2048
            and (single_n <= 640 or average < 0.85)
        )
    ):
        return None
    if (
        single_m == 384
        and not (
            workload.dtype == "fp32"
            and workload.n * in_bytes % 256
        )
    ):
        step_m, step_n, step_k = 3, 1, 3
        depth_a, depth_b, order = 9, 6, 1
    else:
        step_m, step_n, step_k = 1, 3, 3
        depth_a, depth_b, order = 6, 9, 0
    single_m = max(single_m, step_m * 128)
    single_n = max(single_n, step_n * 128)
    single_k = step_k * base_k
    schedule = make_schedule(
        usedCoreNum=used,
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
        iterateOrder=order,
        stepKa=step_k,
        stepKb=step_k,
        dbL0A=2,
        dbL0B=2,
        dbL0C=2,
        l2MTileCnt=1,
        l2NTileCnt=1,
        l2MTileBlock=max(1, ceil_div(workload.m, single_m)),
        l2NTileBlock=max(1, ceil_div(workload.n, single_n)),
        l2IterateOrder=0,
        tilingEnable=2,
    )
    return _candidate(
        workload,
        hardware,
        schedule,
        Template.SINGLE_CORE_SPLIT_K,
        "upstream_single_core_split_k_partition",
    )


def _deterministic_after_single_reject(
    workload: Workload,
    hardware: Hardware,
) -> tuple[bool, str]:
    """Use deterministic split-K only for a genuinely underfilled grid.

    The source callback selects this path while the 128x128 output block grid
    cannot fill the AICs. Once the grid reaches the AIC count, BASE output
    parallelism is sufficient and a rejected single-core split must not imply
    deterministic split-K.
    """

    if workload.k < 27392:
        return False, "deterministic_high_k_threshold_not_met"
    m_count = ceil_div(workload.m, 128)
    n_count = ceil_div(workload.n, 128)
    grid = m_count * n_count
    cores = min(workload.max_cores, hardware.aic_cores)
    if grid >= cores:
        return False, "output_grid_has_sufficient_parallelism"
    return True, "single_split_rejected_high_k_deterministic"


def _prefer_nk_split_layout(workload: Workload) -> bool:
    in_bytes = 4 if workload.dtype == "fp32" else 2
    n_32b_aligned = workload.n * in_bytes % 32 == 0
    if not n_32b_aligned:
        return False
    if workload.m < 128 or workload.n < 128:
        return False
    if workload.m >= 2048 and workload.n == 16:
        return False
    return True


def direct_rule_candidate(
    workload: Workload,
    hardware: Hardware,
) -> Candidate:
    """Return one schedule from the complete fixed template rule tree.

    RuntimeKb is intentionally absent.  The ordering mirrors upstream's
    kernel-specific opportunities, while the deterministic split-K rule is
    evaluated before single-core split-K only when output parallelism is
    provably insufficient.
    """

    fix = _bl1_fix_candidate(workload, hardware)
    if fix is not None:
        return fix
    al1 = _al1_candidate(workload, hardware)
    if al1 is not None:
        return al1
    bl1 = _bl1_candidate(workload, hardware)
    if bl1 is not None:
        return bl1

    use_split_k, reason = _use_deterministic_split_k(workload, hardware)
    if not use_split_k:
        single = _single_split_candidate(workload, hardware)
        if single is not None:
            return single
        use_split_k, reason = _deterministic_after_single_reject(
            workload, hardware
        )
    if not use_split_k:
        base = direct_base_candidate(workload, hardware)
        return _candidate(
            workload,
            hardware,
            base.schedule,
            Template.BASE,
            "upstream_base_after_no_special_template",
        )

    in_bytes = INPUT_BYTES[workload.dtype]
    base_k = 256 // in_bytes
    single_k = 3 * base_k
    k_chunks = ceil_div(workload.k, single_k)
    used_cores = min(
        workload.max_cores,
        hardware.aic_cores,
        k_chunks,
        4
        * ceil_div(workload.m, 128)
        * ceil_div(workload.n, 128),
    )
    prefer_nk = _prefer_nk_split_layout(workload)
    if prefer_nk:
        single_m = workload.m
        single_n = 384
        step_m, step_n = 1, 3
        depth_a, depth_b = 6, 9
        iterate_order, l2_order = 0, 0
        per_core_l2 = hardware.l2_bytes * 7 // 10 // max(
            1, hardware.aic_cores
        )
        fixed_n = min(single_n, workload.n)
        available = per_core_l2 - single_k * fixed_n * in_bytes
        divisor = single_k * in_bytes + fixed_n * 4
        if available > 0 and workload.m > available // divisor:
            split = align_up(max(128, available // divisor), 128)
            count = ceil_div(workload.m, split)
            single_m = align_up(ceil_div(workload.m, count), 128)
    else:
        single_m = 384
        single_n = workload.n
        step_m, step_n = 3, 1
        depth_a, depth_b = 9, 6
        iterate_order, l2_order = 1, 1
        per_core_l2 = hardware.l2_bytes * 7 // 10 // max(
            1, hardware.aic_cores
        )
        fixed_m = min(single_m, workload.m)
        available = per_core_l2 - single_k * fixed_m * in_bytes
        divisor = single_k * in_bytes + fixed_m * 4
        if available > 0 and workload.n > available // divisor:
            split = align_up(max(128, available // divisor), 128)
            count = ceil_div(workload.n, split)
            single_n = align_up(ceil_div(workload.n, count), 128)
    if (
        workload.dtype == "fp32"
        and workload.m <= 64
        and workload.n <= 64
    ):
        used_cores = min(
            used_cores,
            8
            + max(0, (workload.k - 6144) // 1024)
            + ceil_div(workload.m, 16)
            + ceil_div(workload.n, 16),
        )
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
    return _candidate(
        workload,
        hardware,
        schedule,
        Template.DETERMINISTIC_SPLIT_K,
        f"upstream_deterministic_split_k:{reason}",
    )
