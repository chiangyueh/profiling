from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .domain import (
    INPUT_BYTES,
    KNOWLEDGE_FIELDS,
    OUTPUT_BYTES,
    Hardware,
    Schedule,
    Template,
    Workload,
)


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def align_up(value: int, alignment: int) -> int:
    return ceil_div(value, alignment) * alignment


def base_k_alignment(workload: Workload) -> int:
    if workload.dtype == "fp32" and not workload.trans_a and workload.trans_b:
        return 8
    return 16


def bl1_fix_mode_supported(workload: Workload, mode: int) -> bool:
    """Return whether the compiled kernel key supports this fix-output mode."""

    if mode == 1:
        return workload.dtype in {"fp16", "bf16"}
    if mode != 2:
        return False
    return workload.dtype == "fp32" and not workload.trans_a


def bl1_official_fix_applicable(workload: Workload, mode: int) -> bool:
    """Mirror the shape portion of upstream NeedSolveFixBound.

    This is used to emit the official constructor anchor. It is not a kernel
    legality gate: NPU evidence shows that the compiled kernels support a
    broader BL1-resident schedule space.
    """

    if not bl1_fix_mode_supported(workload, mode):
        return False
    if workload.trans_a or workload.k > 256:
        return False
    output_alignment = 256 // OUTPUT_BYTES[workload.dtype]
    fixpipe_bound = (
        workload.n < 256
        and workload.n % output_alignment != 0
        and output_alignment % workload.n != 0
    )
    if not fixpipe_bound:
        return False
    if mode == 2:
        c0 = 32 // INPUT_BYTES[workload.dtype]
        return (
            workload.k % c0 == 0
            and workload.n <= 192
        )
    return True


def bl1_fix_geometry(
    workload: Workload,
    hardware: Hardware,
) -> dict[str, int] | None:
    """Reconstruct DoBL1FullloadWithFixpipeTiling without bank geometry."""

    in_bytes = INPUT_BYTES[workload.dtype]
    base_n = align_up(workload.n, 16)
    if base_n <= 0:
        return None
    base_m_l0c = hardware.l0c_bytes // max(1, base_n * 4)
    ub_bytes = hardware.ub_bytes or hardware.l0c_bytes
    base_m_ub = ub_bytes // max(1, 256 * OUTPUT_BYTES[workload.dtype])
    base_m = min(base_m_l0c, base_m_ub)
    base_m = base_m // 128 * 128
    if base_m < 128:
        return None
    base_ka = hardware.l0a_bytes // max(1, 2 * in_bytes * base_m)
    base_kb = hardware.l0b_bytes // max(1, 2 * in_bytes * base_n)
    base_k = min(base_ka, base_kb) // 16 * 16
    if base_k < 16:
        return None
    depth_b = ceil_div(workload.k, base_k)
    depth_a = (
        hardware.effective_l1_bytes // in_bytes
        - depth_b * base_n * base_k
    ) // max(1, base_m * base_k)
    depth_a = min(depth_a, depth_b)
    if depth_a <= 0:
        return None
    step_ka = depth_a
    if depth_a >= 8:
        depth_a = 8
        step_ka = 4
    return {
        "baseM": base_m,
        "baseN": base_n,
        "baseK": base_k,
        "depthA1": depth_a,
        "depthB1": depth_b,
        "stepKa": step_ka,
        "stepKb": depth_b,
    }


def split_mode(schedule: Schedule) -> int:
    return schedule["tilingEnable"] % 10


def full_load_mode(schedule: Schedule) -> int:
    return (schedule["tilingEnable"] // 10) % 10


def fix_mode(schedule: Schedule) -> int:
    return (schedule["tilingEnable"] // 1000) % 10


def special_mode(schedule: Schedule) -> int:
    return (schedule["tilingEnable"] // 10000) % 10


def template_of(schedule: Schedule) -> Template:
    split = split_mode(schedule)
    full = full_load_mode(schedule)
    fix = fix_mode(schedule)
    if split == 3:
        return Template.DETERMINISTIC_SPLIT_K
    if split == 2:
        return Template.SINGLE_CORE_SPLIT_K
    if full == 1:
        return Template.AL1_FULL_LOAD
    if full == 2 and fix == 1:
        return Template.BL1_FULL_LOAD_FIXPIPE
    if full == 2 and fix == 2:
        return Template.BL1_FULL_LOAD_VEC_NZ2ND
    if full == 2:
        return Template.BL1_FULL_LOAD
    return Template.BASE


@dataclass(frozen=True)
class ContractReport:
    valid: bool
    violations: tuple[str, ...]
    metrics: dict[str, float]


def _l2_schedule_valid(workload: Workload, schedule: Schedule) -> bool:
    m_total = ceil_div(workload.m, schedule["singleCoreM"])
    n_total = ceil_div(workload.n, schedule["singleCoreN"])
    m_block = schedule["l2MTileBlock"]
    n_block = schedule["l2NTileBlock"]
    if m_block == 0 or n_block == 0:
        return (
            m_block == 0
            and n_block == 0
            and schedule["l2MTileCnt"] == 1
            and schedule["l2NTileCnt"] == 1
        )
    if (
        schedule["l2MTileCnt"] != ceil_div(m_total, m_block)
        or schedule["l2NTileCnt"] != ceil_div(n_total, n_block)
    ):
        return False
    m_tail = m_total - (schedule["l2MTileCnt"] - 1) * m_block
    n_tail = n_total - (schedule["l2NTileCnt"] - 1) * n_block
    return 1 <= m_tail <= m_block and 1 <= n_tail <= n_block


def common_hardware_contract(
    workload: Workload,
    schedule: Schedule,
    hardware: Hardware,
) -> ContractReport:
    violations: list[str] = []
    if workload.dtype not in INPUT_BYTES:
        violations.append("unsupported_dtype")
        return ContractReport(False, tuple(violations), {})

    positive = (
        "usedCoreNum",
        "singleCoreM",
        "singleCoreN",
        "singleCoreK",
        "baseM",
        "baseN",
        "baseK",
        "depthA1",
        "depthB1",
        "stepM",
        "stepN",
        "stepKa",
        "stepKb",
        "dbL0A",
        "dbL0B",
        "dbL0C",
        "l2MTileCnt",
        "l2NTileCnt",
    )
    if any(schedule[field] <= 0 for field in positive):
        violations.append("non_positive_field")
    if schedule["l2MTileBlock"] < 0 or schedule["l2NTileBlock"] < 0:
        violations.append("negative_l2_block")
    if schedule["usedCoreNum"] > min(workload.max_cores, hardware.aic_cores):
        violations.append("core_limit")
    if schedule["iterateOrder"] not in (0, 1):
        violations.append("iterate_order")
    if schedule["l2IterateOrder"] not in (0, 1, 2):
        violations.append("l2_iterate_order")
    if any(
        schedule[field] not in (1, 2)
        for field in ("dbL0A", "dbL0B", "dbL0C")
    ):
        violations.append("double_buffer")

    split = split_mode(schedule)
    full = full_load_mode(schedule)
    fix = fix_mode(schedule)
    if special_mode(schedule) != 0:
        violations.append("special_mode")
    if split not in (0, 2, 3) or full not in (0, 1, 2) or fix not in (0, 1, 2):
        violations.append("tiling_enable_range")
    if split and (full or fix):
        violations.append("mixed_split_full_load")
    if full == 0 and fix:
        violations.append("fix_without_full_load")
    if full == 1 and fix:
        violations.append("fix_with_al1")

    k0 = base_k_alignment(workload)
    if (
        schedule["baseM"] % 16
        or schedule["baseN"] % 16
        or schedule["baseK"] % k0
    ):
        violations.append("base_alignment")

    in_bytes = INPUT_BYTES[workload.dtype]
    l0a = (
        schedule["baseM"]
        * schedule["baseK"]
        * in_bytes
        * schedule["dbL0A"]
    )
    l0b = (
        schedule["baseN"]
        * schedule["baseK"]
        * in_bytes
        * schedule["dbL0B"]
    )
    l0c = (
        schedule["baseM"]
        * schedule["baseN"]
        * 4
        * schedule["dbL0C"]
    )
    l0c_capacity = hardware.l0c_capacity(schedule["dbL0C"])
    if l0a > hardware.l0a_bytes:
        violations.append("l0a_capacity")
    if l0b > hardware.l0b_bytes:
        violations.append("l0b_capacity")
    if l0c > l0c_capacity:
        violations.append("l0c_capacity")

    one_a = schedule["stepM"] * schedule["stepKa"]
    one_b = schedule["stepN"] * schedule["stepKb"]
    if one_a <= 0 or schedule["depthA1"] % one_a:
        violations.append("depth_a_factor")
    elif schedule["depthA1"] // one_a not in (1, 2):
        violations.append("depth_a_buffers")
    if one_b <= 0 or schedule["depthB1"] % one_b:
        violations.append("depth_b_factor")
    elif schedule["depthB1"] // one_b not in (1, 2):
        violations.append("depth_b_buffers")

    a1 = (
        schedule["baseM"]
        * schedule["baseK"]
        * schedule["depthA1"]
        * in_bytes
    )
    b1 = (
        schedule["baseN"]
        * schedule["baseK"]
        * schedule["depthB1"]
        * in_bytes
    )
    if a1 + b1 > hardware.effective_l1_bytes:
        violations.append("l1_capacity")
    if (
        schedule["stepKa"] % schedule["stepKb"]
        and schedule["stepKb"] % schedule["stepKa"]
    ):
        violations.append("incompatible_k_steps")

    metrics = {
        "l0a_bytes": float(l0a),
        "l0b_bytes": float(l0b),
        "l0c_bytes": float(l0c),
        "l1_bytes": float(a1 + b1),
        "l0a_occupancy": l0a / max(1.0, hardware.l0a_bytes),
        "l0b_occupancy": l0b / max(1.0, hardware.l0b_bytes),
        "l0c_occupancy": l0c / max(1.0, l0c_capacity),
        "l1_occupancy": (a1 + b1) / max(1.0, hardware.effective_l1_bytes),
    }
    return ContractReport(not violations, tuple(violations), metrics)


def _base_contract(
    workload: Workload,
    schedule: Schedule,
    hardware: Hardware,
) -> list[str]:
    del hardware
    violations: list[str] = []
    k0 = base_k_alignment(workload)
    if schedule["singleCoreK"] != workload.k:
        violations.append("base_single_core_k")
    if schedule["baseM"] > align_up(schedule["singleCoreM"], 16):
        violations.append("base_m_exceeds_core")
    if schedule["baseN"] > align_up(schedule["singleCoreN"], 16):
        violations.append("base_n_exceeds_core")
    if schedule["baseK"] > align_up(schedule["singleCoreK"], k0):
        violations.append("base_k_exceeds_core")
    if schedule["stepM"] * schedule["baseM"] > align_up(
        schedule["singleCoreM"], 16
    ):
        violations.append("step_m_exceeds_core")
    if schedule["stepN"] * schedule["baseN"] > align_up(
        schedule["singleCoreN"], 16
    ):
        violations.append("step_n_exceeds_core")
    if not _l2_schedule_valid(workload, schedule):
        violations.append("l2_coverage")
    return violations


def _single_core_split_k_contract(
    workload: Workload,
    schedule: Schedule,
    hardware: Hardware,
) -> list[str]:
    del hardware
    violations: list[str] = []
    if schedule["stepKa"] != schedule["stepKb"]:
        violations.append("split_k_step_mismatch")
    if schedule["singleCoreK"] != schedule["stepKa"] * schedule["baseK"]:
        violations.append("split_k_extent")
    if schedule["singleCoreK"] >= workload.k:
        violations.append("split_k_requires_multiple_chunks")
    if ceil_div(workload.k, schedule["singleCoreK"]) < 2:
        violations.append("split_k_chunk_count")
    if schedule["singleCoreM"] < schedule["stepM"] * schedule["baseM"]:
        violations.append("split_k_inner_m")
    if schedule["singleCoreN"] < schedule["stepN"] * schedule["baseN"]:
        violations.append("split_k_inner_n")
    # MatmulSingleCoreSplitKBaseBlock reads only calOrder from
    # L2cacheTilePara. The count/block fields are carried by the common
    # record, but do not describe singleCoreM/N coverage in this kernel.
    if schedule["l2MTileBlock"] <= 0 or schedule["l2NTileBlock"] <= 0:
        violations.append("split_k_l2_block")
    return violations


def _deterministic_split_k_contract(
    workload: Workload,
    schedule: Schedule,
    hardware: Hardware,
) -> list[str]:
    del hardware
    violations: list[str] = []
    expected_base_k = 256 // INPUT_BYTES[workload.dtype]
    if (
        schedule["baseM"],
        schedule["baseN"],
        schedule["baseK"],
    ) != (128, 128, expected_base_k):
        violations.append("det_static_base")
    mk33 = (
        schedule["stepM"],
        schedule["stepN"],
        schedule["stepKa"],
        schedule["stepKb"],
        schedule["depthA1"],
        schedule["depthB1"],
        schedule["singleCoreM"],
        schedule["iterateOrder"],
    ) == (3, 1, 3, 3, 9, 6, 384, 1)
    nk33 = (
        schedule["stepM"],
        schedule["stepN"],
        schedule["stepKa"],
        schedule["stepKb"],
        schedule["depthA1"],
        schedule["depthB1"],
        schedule["singleCoreN"],
        schedule["iterateOrder"],
    ) == (1, 3, 3, 3, 6, 9, 384, 0)
    if not (mk33 or nk33):
        violations.append("det_static_algorithm")
    if schedule["singleCoreK"] != 3 * schedule["baseK"]:
        violations.append("det_k_extent")
    k_chunks = ceil_div(workload.k, schedule["singleCoreK"])
    if k_chunks < 2:
        violations.append("det_k_chunks")
    if schedule["usedCoreNum"] > k_chunks:
        violations.append("det_core_chunks")
    # Deterministic split-K does not consume L2cacheTilePara. Keep nonzero
    # block values for callback compatibility without imposing BASE-kernel
    # coverage semantics on this template.
    if schedule["l2MTileBlock"] <= 0 or schedule["l2NTileBlock"] <= 0:
        violations.append("det_l2_block")
    return violations


def _al1_contract(
    workload: Workload,
    schedule: Schedule,
    hardware: Hardware,
) -> list[str]:
    violations: list[str] = []
    in_bytes = INPUT_BYTES[workload.dtype]
    if schedule["singleCoreM"] < workload.m:
        violations.append("al1_m_residency")
    if schedule["singleCoreK"] < workload.k:
        violations.append("al1_k_extent")
    if schedule["stepM"] != 1 or schedule["stepN"] != 1:
        violations.append("al1_step_mn")
    expected_step_k = ceil_div(workload.k, schedule["baseK"])
    if schedule["stepKa"] != expected_step_k or schedule["stepKb"] != 1:
        violations.append("al1_k_steps")
    if schedule["depthA1"] not in (
        schedule["stepKa"],
        2 * schedule["stepKa"],
    ):
        violations.append("al1_depth")
    resident_a = (
        align_up(workload.m, 16)
        * align_up(workload.k, base_k_alignment(workload))
        * in_bytes
    )
    staged_b = (
        schedule["baseN"]
        * schedule["baseK"]
        * schedule["depthB1"]
        * in_bytes
    )
    if resident_a + staged_b > hardware.effective_l1_bytes:
        violations.append("al1_resident_capacity")
    if not _l2_schedule_valid(workload, schedule):
        violations.append("al1_l2_coverage")
    return violations


def _bl1_contract(
    workload: Workload,
    schedule: Schedule,
    hardware: Hardware,
) -> list[str]:
    violations: list[str] = []
    in_bytes = INPUT_BYTES[workload.dtype]
    fix = fix_mode(schedule)
    if fix:
        if not bl1_fix_mode_supported(workload, fix):
            violations.append("bl1_fix_kernel_regime")
        expected_cores = min(
            workload.max_cores,
            hardware.aic_cores,
            ceil_div(workload.m, schedule["singleCoreM"]),
        )
        if (
            schedule["singleCoreM"],
            schedule["singleCoreN"],
            schedule["singleCoreK"],
        ) != (
            schedule["baseM"],
            align_up(workload.n, 16),
            workload.k,
        ):
            violations.append("bl1_fix_output_geometry")
        if schedule["usedCoreNum"] != expected_cores:
            violations.append("bl1_fix_core_contract")
        expected_step_n = ceil_div(workload.n, schedule["baseN"])
        expected_step_k = ceil_div(workload.k, schedule["baseK"])
        regular_a_pipeline = (
            schedule["stepKa"] == expected_step_k
            and schedule["depthA1"]
            in (expected_step_k, 2 * expected_step_k)
        )
        official_large_k_pipeline = (
            expected_step_k >= 8
            and schedule["stepKa"] == 4
            and schedule["depthA1"] == 8
        )
        if (
            schedule["iterateOrder"],
            schedule["dbL0A"],
            schedule["dbL0B"],
            schedule["dbL0C"],
            schedule["l2IterateOrder"],
        ) != (
            0,
            2,
            2,
            1,
            0,
        ):
            violations.append("bl1_fix_pipeline_contract")
        if (
            schedule["stepM"],
            schedule["stepN"],
            schedule["stepKb"],
        ) != (
            1,
            expected_step_n,
            expected_step_k,
        ):
            violations.append("bl1_fix_step_contract")
        if not regular_a_pipeline and not official_large_k_pipeline:
            violations.append("bl1_fix_a_pipeline")
        if schedule["depthB1"] not in (
            expected_step_n * expected_step_k,
            2 * expected_step_n * expected_step_k,
        ):
            violations.append("bl1_fix_b_pipeline")
        resident_b = (
            align_up(workload.k, base_k_alignment(workload))
            * align_up(workload.n, 32 // in_bytes)
            * in_bytes
        )
        staged_a = (
            schedule["baseM"]
            * schedule["baseK"]
            * schedule["depthA1"]
            * in_bytes
        )
        if resident_b + staged_a > hardware.effective_l1_bytes:
            violations.append("bl1_fix_resident_capacity")
        if (
            schedule["l2MTileCnt"],
            schedule["l2NTileCnt"],
            schedule["l2MTileBlock"],
            schedule["l2NTileBlock"],
        ) != (1, 1, 0, 0):
            violations.append("bl1_fix_l2_contract")
        return violations

    if schedule["singleCoreK"] < workload.k:
        violations.append("bl1_k_extent")
    if schedule["singleCoreN"] < workload.n:
        violations.append("bl1_n_residency")
    if schedule["stepM"] != 1:
        violations.append("bl1_step_m")
    expected_step_n = ceil_div(workload.n, schedule["baseN"])
    expected_step_k = ceil_div(workload.k, schedule["baseK"])
    if schedule["stepN"] != expected_step_n:
        violations.append("bl1_step_n")
    if (
        schedule["stepKa"] != expected_step_k
        or schedule["stepKb"] != expected_step_k
    ):
        violations.append("bl1_step_k")
    if schedule["depthB1"] not in (
        schedule["stepN"] * schedule["stepKb"],
        2 * schedule["stepN"] * schedule["stepKb"],
    ):
        violations.append("bl1_depth")
    resident_b = (
        align_up(workload.k, base_k_alignment(workload))
        * align_up(workload.n, 32 // in_bytes)
        * in_bytes
    )
    staged_a = (
        schedule["baseM"]
        * schedule["baseK"]
        * schedule["depthA1"]
        * in_bytes
    )
    if resident_b + staged_a > hardware.effective_l1_bytes:
        violations.append("bl1_resident_capacity")
    expected_cores = min(
        workload.max_cores,
        hardware.aic_cores,
        ceil_div(workload.m, 2 * schedule["baseM"]),
    )
    if schedule["singleCoreM"] != 2 * schedule["baseM"]:
        violations.append("bl1_standard_single_m")
    if schedule["singleCoreN"] != workload.n:
        violations.append("bl1_standard_single_n")
    if schedule["depthA1"] != 2 * schedule["stepKa"]:
        violations.append("bl1_standard_a_pipeline")
    if schedule["depthB1"] != schedule["stepN"] * schedule["stepKb"]:
        violations.append("bl1_standard_b_pipeline")
    if schedule["usedCoreNum"] != expected_cores:
        violations.append("bl1_standard_core_contract")
    if (
        schedule["l2MTileCnt"],
        schedule["l2NTileCnt"],
        schedule["l2MTileBlock"],
        schedule["l2NTileBlock"],
        schedule["l2IterateOrder"],
    ) != (
        1,
        1,
        ceil_div(workload.m, schedule["singleCoreM"]),
        1,
        1,
    ):
        violations.append("bl1_standard_l2_contract")
    return violations


_TEMPLATE_CONTRACTS: dict[
    Template, Callable[[Workload, Schedule, Hardware], list[str]]
] = {
    Template.BASE: _base_contract,
    Template.SINGLE_CORE_SPLIT_K: _single_core_split_k_contract,
    Template.DETERMINISTIC_SPLIT_K: _deterministic_split_k_contract,
    Template.AL1_FULL_LOAD: _al1_contract,
    Template.BL1_FULL_LOAD: _bl1_contract,
    Template.BL1_FULL_LOAD_FIXPIPE: _bl1_contract,
    Template.BL1_FULL_LOAD_VEC_NZ2ND: _bl1_contract,
}


def template_kernel_contract(
    workload: Workload,
    schedule: Schedule,
    hardware: Hardware,
) -> ContractReport:
    template = template_of(schedule)
    violations = _TEMPLATE_CONTRACTS[template](workload, schedule, hardware)
    return ContractReport(not violations, tuple(violations), {})


def validate_schedule(
    workload: Workload,
    schedule: Schedule,
    hardware: Hardware,
) -> ContractReport:
    common = common_hardware_contract(workload, schedule, hardware)
    if not common.valid:
        return common
    template = template_kernel_contract(workload, schedule, hardware)
    return ContractReport(
        template.valid,
        template.violations,
        common.metrics,
    )


def profitability_prior(
    workload: Workload,
    schedule: Schedule,
    hardware: Hardware,
) -> tuple[float, dict[str, float]]:
    common = common_hardware_contract(workload, schedule, hardware)
    m_tasks = ceil_div(workload.m, schedule["singleCoreM"])
    n_tasks = ceil_div(workload.n, schedule["singleCoreN"])
    output_tasks = m_tasks * n_tasks
    split = split_mode(schedule)
    split_k_chunks = (
        ceil_div(workload.k, schedule["singleCoreK"])
        if split in (2, 3)
        else 1
    )
    if split == 3:
        active = min(schedule["usedCoreNum"], split_k_chunks)
        rounds = (
            output_tasks
            * ceil_div(split_k_chunks, max(1, active))
        )
    else:
        active = min(schedule["usedCoreNum"], max(1, output_tasks))
        rounds = ceil_div(output_tasks, max(1, active))
    padded_m = m_tasks * schedule["singleCoreM"]
    padded_n = n_tasks * schedule["singleCoreN"]
    padded_k = align_up(workload.k, schedule["baseK"])
    padding_efficiency = (
        workload.m
        * workload.n
        * workload.k
        / max(1.0, padded_m * padded_n * padded_k)
    )
    l0_occupancy = max(
        common.metrics.get("l0a_occupancy", 0.0),
        common.metrics.get("l0b_occupancy", 0.0),
        common.metrics.get("l0c_occupancy", 0.0),
    )
    l1_occupancy = common.metrics.get("l1_occupancy", 0.0)
    k_passes = ceil_div(workload.k, schedule["baseK"])
    in_bytes = INPUT_BYTES[workload.dtype]
    out_bytes = OUTPUT_BYTES[workload.dtype]
    input_lower_bound = (
        workload.m * workload.k * in_bytes
        + workload.k * workload.n * in_bytes
    )
    output_bytes = workload.m * workload.n * out_bytes
    if split == 2:
        output_write_multiplier = 2 * split_k_chunks - 1
    elif split == 3:
        reduction_bytes = (
            2.0 * active * workload.m * workload.n * 4
        )
        output_write_multiplier = (
            output_bytes + reduction_bytes
        ) / max(1.0, output_bytes)
    else:
        output_write_multiplier = 1
    partitioned_input_bytes = (
        padded_m * workload.k * in_bytes * n_tasks
        + workload.k * padded_n * in_bytes * m_tasks
    )
    traffic_amplification = (
        partitioned_input_bytes
        + output_bytes * output_write_multiplier
    ) / max(1.0, input_lower_bound + output_bytes)
    core_utilization = active / max(
        1.0, min(workload.max_cores, hardware.aic_cores)
    )
    occupancy_penalty = abs(l0_occupancy - 0.75) + 0.25 * abs(
        l1_occupancy - 0.65
    )
    ideal_active_cores = max(
        1.0, min(workload.max_cores, hardware.aic_cores)
    )
    compute_amplification = (
        ideal_active_cores
        / max(1.0, active)
        / max(0.05, padding_efficiency)
    )
    score = (
        compute_amplification
        * (1.0 + 0.12 * occupancy_penalty)
        * (1.0 + 0.002 * k_passes)
        * (1.0 + 0.15 * max(0.0, traffic_amplification - 1.0))
    )
    if split == 2:
        # A single-core split does not create additional output-tile
        # parallelism. Once the output grid already fills the AICs, repeated
        # partial writes are pure overhead. This remains a ranking prior, not
        # an applicability gate, so split-K probes stay reachable.
        output_parallelism = min(
            1.0,
            output_tasks
            / max(1.0, min(workload.max_cores, hardware.aic_cores)),
        )
        score *= 1.0 + min(32.0, split_k_chunks - 1.0) * (
            0.05 + 0.20 * output_parallelism
        )
    metrics = dict(common.metrics)
    metrics.update(
        {
            "output_tasks": float(output_tasks),
            "active_cores": float(active),
            "core_rounds": float(rounds),
            "core_utilization": core_utilization,
            "compute_amplification": compute_amplification,
            "padding_efficiency": padding_efficiency,
            "k_passes": float(k_passes),
            "split_k_chunks": float(split_k_chunks),
            "output_write_multiplier": float(output_write_multiplier),
            "traffic_amplification": traffic_amplification,
            "profitability_prior": score,
        }
    )
    return score, metrics
