from __future__ import annotations

import math
from functools import lru_cache

from ..contracts import align_up, base_k_alignment, ceil_div
from ..domain import INPUT_BYTES, OUTPUT_BYTES, Hardware, Workload


_REFERENCE_L2_BYTES = 192 * 1024 * 1024
_REFERENCE_L2_WORKING_SET = 100 * 1024 * 1024
_CALC_M_BASIC = (
    (128, 256, 128),
    (128, 128, 256),
    (64, 512, 64),
)
_CALC_MN_BASIC = (
    (64, 64, 512),
    (128, 128, 256),
    (128, 256, 128),
)


def _cal_base_size(
    count: int,
    total_cores: int,
    size: int,
    maximum: int,
) -> int:
    cores = max(1, total_cores // count) if count else 1
    return min(align_up(max(1, size // cores), 16), maximum)


def _cal_base_mn(
    workload: Workload,
    hardware: Hardware,
    base_m: int,
    base_n: int,
    maximum_m: int,
    maximum_n: int,
) -> tuple[int, int]:
    m_cores = ceil_div(workload.m, maximum_m)
    n_cores = ceil_div(workload.n, maximum_n)
    if m_cores < n_cores:
        n_cores = max(1, hardware.aic_cores // m_cores)
        base_n = min(
            align_up(max(1, workload.n // n_cores), 16),
            maximum_n,
        )
    else:
        m_cores = max(1, hardware.aic_cores // n_cores)
        base_m = min(
            align_up(max(1, workload.m // m_cores), 16),
            maximum_m,
        )
    return (
        align_up(min(workload.m, base_m), 16),
        align_up(min(workload.n, base_n), 16),
    )


def _select_formulaic_base(
    workload: Workload,
    hardware: Hardware,
    options: tuple[tuple[int, int, int], ...],
    *,
    balance_m_only: bool,
) -> tuple[int, int, int]:
    selected = None
    for maximum_m, maximum_n, base_k_bytes in options:
        if balance_m_only:
            n_cores = ceil_div(workload.n, maximum_n)
            base_m = _cal_base_size(
                n_cores,
                hardware.aic_cores,
                workload.m,
                maximum_m,
            )
            base_n = maximum_n
        else:
            base_m, base_n = _cal_base_mn(
                workload,
                hardware,
                maximum_m,
                maximum_n,
                maximum_m,
                maximum_n,
            )
        m_cores = ceil_div(workload.m, base_m)
        n_cores = ceil_div(workload.n, base_n)
        tail = m_cores * n_cores % hardware.aic_cores
        load = (
            (base_m + base_n)
            * (m_cores * n_cores // hardware.aic_cores)
            + ((base_m + base_n) if tail else 0)
        )
        value = (load, base_m, base_n, base_k_bytes)
        if selected is None or value[0] < selected[0]:
            selected = value
    if selected is None:
        raise ValueError("formulaic BASE policy has no geometry")
    return selected[1], selected[2], selected[3]


def _balance_base_geometry(
    workload: Workload,
    hardware: Hardware,
    basic_block_m: int,
) -> tuple[int, int, int]:
    in_bytes = INPUT_BYTES[workload.dtype]
    if workload.m >= basic_block_m:
        m_cores = max(1, ceil_div(workload.m, basic_block_m))
        n_cores = max(1, hardware.aic_cores // m_cores)
        base_m = basic_block_m
        base_n = min(
            align_up(ceil_div(workload.n, n_cores), 16),
            256,
        )
        base_k_a = (
            hardware.l0a_bytes // 2 // in_bytes // 256
        )
        base_k_b = (
            hardware.l0b_bytes // (2 * in_bytes * base_n)
        )
    else:
        maximum_n_floor = (
            basic_block_m if workload.m >= 64 else 1024
        )
        base_m = align_up(workload.m, 16)
        n_tile = max(1, workload.n // hardware.aic_cores)
        base_n = max(align_up(n_tile, 16), maximum_n_floor)
        maximum_n = (
            hardware.l0c_bytes // (4 * base_m) // 16 * 16
        )
        base_n = min(base_n, maximum_n)
        n_cores = ceil_div(workload.n, base_n)
        tail_cores = (
            ceil_div(workload.n, 256) % hardware.aic_cores
            if base_n > 256
            else 0
        )
        if n_cores % hardware.aic_cores < tail_cores:
            base_n = 256
        base_k_a = (
            hardware.l0a_bytes // (2 * in_bytes * base_m)
        )
        base_k_b = (
            hardware.l0b_bytes // (2 * in_bytes * base_n)
        )
    base_k = min(base_k_a, base_k_b) // 16 * 16
    if (
        min(base_m, base_n, base_k) <= 0
        or base_m % 16
        or base_n % 16
        or base_k % 16
    ):
        return 128, 256, 128 // in_bytes
    return base_m, base_n, base_k


def upstream_base_geometry_policy(
    workload: Workload,
    hardware: Hardware,
) -> tuple[int, int, int, int, int, int]:
    """Port the coupled 8.5 BASE and small-shape geometry policy.

    The formulas correspond to SetBaseBlockTiling, CalBase,
    CalBaseMBaseN, BalanceBaseBlockTiling, and DoSmallShapeTiling.
    They depend on shape, layout, dtype, core count, and L0 capacity,
    not on a RuntimeKb record or a hand-written workload family.
    """

    in_bytes = INPUT_BYTES[workload.dtype]
    base_k = 128 // in_bytes
    base_m, base_n = 128, 256
    if hardware.l0c_bytes >= 256 * 1024:
        base_m = 256
    elif workload.trans_a and workload.trans_b:
        base_m, base_n = 256, 128
    elif workload.trans_a or workload.trans_b:
        large_m = (
            workload.m >= (hardware.aic_cores // 2) * 256
        )
        large_n = (
            workload.n >= (hardware.aic_cores // 2) * 256
        )
        if large_m and large_n:
            cost_m_large = (
                ceil_div(workload.m, 256) * workload.n
                + ceil_div(workload.n, 128) * workload.m
            )
            cost_n_large = (
                ceil_div(workload.m, 128) * workload.n
                + ceil_div(workload.n, 256) * workload.m
            )
            if cost_m_large < cost_n_large:
                base_m, base_n = 256, 128
        elif large_m or (
            not large_n and workload.m > workload.n
        ):
            base_m, base_n = 256, 128

    basic_block_m = (
        256 if hardware.l0c_bytes >= 256 * 1024 else 128
    )
    aligned_m = align_up(workload.m, basic_block_m)
    aligned_n = align_up(workload.n, 256)
    small_block_count = (
        aligned_m // base_m * (aligned_n // base_n)
        < hardware.aic_cores
    )
    if small_block_count or workload.m < 256 or workload.n < 256:
        m_aligned = workload.m * in_bytes % 256 == 0
        k_aligned = workload.k * in_bytes % 256 == 0
        n_aligned = workload.n * in_bytes % 256 == 0
        if not workload.trans_a and not workload.trans_b:
            base_m, base_n, base_k_bytes = _select_formulaic_base(
                workload,
                hardware,
                _CALC_M_BASIC,
                balance_m_only=True,
            )
            base_k = base_k_bytes // in_bytes
            if not k_aligned and not n_aligned:
                base_m, base_n, base_k = _balance_base_geometry(
                    workload, hardware, basic_block_m
                )
            elif not k_aligned:
                base_n = 256
                base_k = 128 // in_bytes
                base_m = _cal_base_size(
                    ceil_div(workload.n, 256),
                    hardware.aic_cores,
                    workload.m,
                    basic_block_m,
                )
            elif not n_aligned:
                base_k = 128 // in_bytes
                base_m, base_n = _cal_base_mn(
                    workload,
                    hardware,
                    base_m,
                    base_n,
                    128,
                    256,
                )
        elif workload.trans_a and not workload.trans_b:
            m_cores = ceil_div(workload.m, base_m)
            n_cores = ceil_div(workload.n, base_n)
            if not m_aligned and not n_aligned:
                base_m, base_n, base_k = _balance_base_geometry(
                    workload, hardware, basic_block_m
                )
            elif not m_aligned:
                base_m = _cal_base_size(
                    n_cores,
                    hardware.aic_cores,
                    workload.m,
                    basic_block_m,
                )
            elif not n_aligned:
                base_n = _cal_base_size(
                    m_cores,
                    hardware.aic_cores,
                    workload.n,
                    256,
                )
        elif not workload.trans_a and workload.trans_b:
            base_m, base_n, base_k_bytes = _select_formulaic_base(
                workload,
                hardware,
                _CALC_MN_BASIC,
                balance_m_only=False,
            )
            base_k = base_k_bytes // in_bytes
            if not k_aligned:
                base_m, base_n, base_k = _balance_base_geometry(
                    workload, hardware, basic_block_m
                )
        else:
            m_cores = ceil_div(workload.m, base_m)
            base_n = _cal_base_size(
                m_cores,
                hardware.aic_cores,
                workload.n,
                256,
            )
            if not m_aligned and k_aligned:
                base_m, base_n, base_k = _balance_base_geometry(
                    workload, hardware, basic_block_m
                )
            elif not m_aligned:
                base_m, base_n = _cal_base_mn(
                    workload,
                    hardware,
                    base_m,
                    base_n,
                    128,
                    256,
                )

        if 128 < workload.m < 256 and base_n > 128:
            base_n = 128
        elif 128 < workload.n < 256 and base_m > 128:
            base_m = 128
        elif base_m * base_n > 32768:
            if 128 < base_m < 256:
                base_m = 128
            if 128 < base_n < 256:
                base_n = 128
        if base_m * base_n > 32768:
            base_m = 32768 // base_n

    base_k = min(
        base_k,
        align_up(workload.k, base_k_alignment(workload)),
    )
    return base_m, base_n, base_k, 2, 2, 1


def base_geometry_variants(
    workload: Workload,
    hardware: Hardware,
) -> tuple[tuple[int, int, int, int, int, int], ...]:
    """Return capacity-valid BASE geometries, led by upstream policy.

    The first geometry follows SetBaseBlockTiling. Remaining entries are
    structural alternatives, not independent field combinations.
    """

    in_bytes = INPUT_BYTES[workload.dtype]
    base_k = 128 // in_bytes
    policy = upstream_base_geometry_policy(workload, hardware)
    preferred = (128, 256)
    if workload.trans_a and workload.trans_b:
        preferred = (256, 128)
    elif workload.trans_a or workload.trans_b:
        large_m = workload.m >= (hardware.aic_cores // 2) * 256
        large_n = workload.n >= (hardware.aic_cores // 2) * 256
        if large_m and large_n:
            cost_m_large = (
                ceil_div(workload.m, 256) * workload.n
                + ceil_div(workload.n, 128) * workload.m
            )
            cost_n_large = (
                ceil_div(workload.m, 128) * workload.n
                + ceil_div(workload.n, 256) * workload.m
            )
            if cost_m_large < cost_n_large:
                preferred = (256, 128)
        elif large_m or (not large_n and workload.m > workload.n):
            preferred = (256, 128)

    tail_m = max(16, align_up(min(workload.m, 256), 16))
    tail_n = max(16, align_up(min(workload.n, 256), 16))
    geometries = [
        preferred,
        (preferred[1], preferred[0]),
        (128, 128),
        (64, 256),
        (256, 64),
        (128, tail_n),
        (tail_m, 128),
        (tail_m, tail_n),
        (tail_m, 32),
        (64, 64),
        (32, tail_n),
    ]
    geometry_seen = set()
    unique_geometries = []
    for geometry in geometries:
        if geometry in geometry_seen:
            continue
        geometry_seen.add(geometry)
        unique_geometries.append(geometry)
    geometries = unique_geometries

    supplemental = {
        (128, tail_n),
        (tail_m, 128),
        (tail_m, tail_n),
        (tail_m, 32),
        (64, 64),
        (32, tail_n),
    }
    geometry_k = [
        ((policy[0], policy[1]), policy[2]),
        *((geometry, base_k) for geometry in geometries),
        *(
            (geometry, candidate_k)
            for geometry in geometries
            if geometry in supplemental
            for candidate_k in (2 * base_k, 4 * base_k)
        ),
    ]

    result = [policy]
    seen = {policy}
    for (base_m, base_n), candidate_k in geometry_k:
        db_a = (
            2
            if base_m * candidate_k * in_bytes * 2
            <= hardware.l0a_bytes
            else 1
        )
        db_b = (
            2
            if base_n * candidate_k * in_bytes * 2
            <= hardware.l0b_bytes
            else 1
        )
        db_c = (
            2
            if base_m * base_n * 4 * 2 <= hardware.l0c_bytes
            else 1
        )
        for buffering in (
            (db_a, db_b, db_c),
            (db_a, db_b, 1),
        ):
            spec = (
                base_m,
                base_n,
                candidate_k,
                *buffering,
            )
            if spec in seen:
                continue
            seen.add(spec)
            candidate_db_a, candidate_db_b, candidate_db_c = buffering
            if (
                base_m
                * candidate_k
                * in_bytes
                * candidate_db_a
                > hardware.l0a_bytes
                or base_n
                * candidate_k
                * in_bytes
                * candidate_db_b
                > hardware.l0b_bytes
                or base_m
                * base_n
                * 4
                * candidate_db_c
                > hardware.l0c_capacity(candidate_db_c)
            ):
                continue
            result.append(spec)
    return tuple(result)


def _align_l1_step_k(
    step_k: int,
    *,
    workload_k: int,
    base_k: int,
    in_bytes: int,
) -> int:
    if step_k * base_k >= workload_k:
        return max(1, step_k)
    transfer_bytes = step_k * base_k * in_bytes
    alignment = 0
    if transfer_bytes > 512:
        alignment = 512
    elif transfer_bytes > 256:
        alignment = 256
    if alignment and alignment % (base_k * in_bytes) == 0:
        while (
            step_k > 1
            and step_k * base_k * in_bytes % alignment
        ):
            step_k -= 1
    return max(1, step_k)


def l1_pipeline_variant(
    workload: Workload,
    hardware: Hardware,
    *,
    base_m: int,
    base_n: int,
    base_k: int,
) -> tuple[int, int, int, int, int, int] | None:
    """Reproduce the coupled CalL1Tiling pipeline geometry."""

    in_bytes = INPUT_BYTES[workload.dtype]
    total_l1 = hardware.effective_l1_bytes
    depth_a = total_l1 // 2 // base_m // base_k // in_bytes
    depth_b = total_l1 // 2 // base_n // base_k // in_bytes
    if depth_a <= 0 or depth_b <= 0:
        return None
    size_a = depth_a * base_m * base_k * in_bytes
    size_b = depth_b * base_n * base_k * in_bytes
    if size_a + size_b > total_l1:
        if base_m <= base_n:
            depth_a //= 2
        else:
            depth_b //= 2
    step_ka = _align_l1_step_k(
        max(1, depth_a // 2),
        workload_k=workload.k,
        base_k=base_k,
        in_bytes=in_bytes,
    )
    step_kb = _align_l1_step_k(
        max(1, depth_b // 2),
        workload_k=workload.k,
        base_k=base_k,
        in_bytes=in_bytes,
    )
    if step_ka >= step_kb:
        step_ka = max(step_kb, step_ka // step_kb * step_kb)
    else:
        step_kb = max(step_ka, step_kb // step_ka * step_ka)
    depth_a = 2 * step_ka
    depth_b = 2 * step_kb
    if (
        depth_a * base_m * base_k * in_bytes
        + depth_b * base_n * base_k * in_bytes
        > total_l1
    ):
        return None
    return depth_a, depth_b, 1, 1, step_ka, step_kb


def l1_pipeline_variants(
    workload: Workload,
    hardware: Hardware,
    *,
    base_m: int,
    base_n: int,
    base_k: int,
) -> tuple[tuple[int, int, int, int, int, int], ...]:
    """Return the L1 policies implemented by the two BASE kernel paths.

    The V200 path uses depth 6 and stepK 3 for ND inputs. Newer paths
    derive the depth from available L1. Both records occur in the 910B3
    RuntimeKb, so the independent solver must expose both without using
    workload-family gates.
    """

    capacity_policy = l1_pipeline_variant(
        workload,
        hardware,
        base_m=base_m,
        base_n=base_n,
        base_k=base_k,
    )
    kernel_policies = []
    if capacity_policy is not None:
        kernel_policies.append(capacity_policy)
    kernel_policies.append((6, 6, 1, 1, 3, 3))

    result = []
    seen = set()
    in_bytes = INPUT_BYTES[workload.dtype]
    for policy in kernel_policies:
        if policy in seen:
            continue
        seen.add(policy)
        depth_a, depth_b, _, _, _, _ = policy
        if (
            depth_a * base_m * base_k * in_bytes
            + depth_b * base_n * base_k * in_bytes
            <= hardware.effective_l1_bytes
        ):
            result.append(policy)
    return tuple(result)


def core_partition_variants(
    workload: Workload,
    hardware: Hardware,
    *,
    minimum_m: int,
    minimum_n: int,
    alignment_m: int = 16,
    alignment_n: int = 16,
    rounds: tuple[int, ...] = (1, 2),
) -> tuple[tuple[int, int, int], ...]:
    """Factor the output grid while preserving full-core wave efficiency."""

    core_limit = min(workload.max_cores, hardware.aic_cores)
    values = set()
    for wave_count in rounds:
        target_tasks = max(1, core_limit * wave_count)
        for m_parts in range(1, target_tasks + 1):
            n_parts = ceil_div(target_tasks, m_parts)
            for candidate_n_parts in {
                max(1, n_parts - 1),
                n_parts,
                n_parts + 1,
            }:
                single_m = max(
                    minimum_m,
                    align_up(
                        ceil_div(workload.m, m_parts), alignment_m
                    ),
                )
                single_n = max(
                    minimum_n,
                    align_up(
                        ceil_div(workload.n, candidate_n_parts),
                        alignment_n,
                    ),
                )
                m_tasks = ceil_div(workload.m, single_m)
                n_tasks = ceil_div(workload.n, single_n)
                tasks = max(1, m_tasks * n_tasks)
                cores = min(core_limit, tasks)
                values.add((single_m, single_n, cores))

    def score(value: tuple[int, int, int]) -> tuple[float, ...]:
        single_m, single_n, cores = value
        tasks = ceil_div(workload.m, single_m) * ceil_div(
            workload.n, single_n
        )
        waves = ceil_div(tasks, cores)
        wave_efficiency = tasks / max(1.0, waves * cores)
        padding = (
            workload.m
            * workload.n
            / max(
                1.0,
                ceil_div(workload.m, single_m)
                * single_m
                * ceil_div(workload.n, single_n)
                * single_n,
            )
        )
        return (-wave_efficiency, -padding, waves, single_m * single_n)

    direct = (
        minimum_m,
        minimum_n,
        min(
            core_limit,
            ceil_div(workload.m, minimum_m)
            * ceil_div(workload.n, minimum_n),
        ),
    )
    ordered = [direct]
    paired_l2_direct = (
        minimum_m,
        minimum_n,
        min(core_limit, 2, direct[2]),
    )
    if paired_l2_direct != direct:
        ordered.append(paired_l2_direct)
    full_core_direct = (minimum_m, minimum_n, core_limit)
    if full_core_direct not in ordered:
        ordered.append(full_core_direct)
    ordered.extend(
        value
        for value in sorted(values, key=score)
        if value not in {direct, paired_l2_direct, full_core_direct}
    )
    return tuple(ordered)


def _l2_target_bytes(hardware: Hardware) -> float:
    scaled = (
        _REFERENCE_L2_WORKING_SET
        * hardware.l2_bytes
        / _REFERENCE_L2_BYTES
    )
    return max(1.0, min(float(hardware.l2_bytes), scaled))


def upstream_base_l2_policy(
    workload: Workload,
    hardware: Hardware,
    *,
    base_m: int,
    base_n: int,
    resident_threshold_bytes: float | None = None,
) -> tuple[tuple[int, int, int, int, int], str]:
    """Reconstruct the coupled 8.5 BASE L2 policy for a 20-AIC target.

    This is the integer policy implemented by InitL2SplitParams, CalcTile,
    and DoL2CacheTiling. A BASE-only request keeps the whole output grid in
    one L2 group while its working set fits. Larger problems use CalcTile;
    the 4-by-(AIC/4) layout is only the final fallback when CalcTile cannot
    find a capacity- and tail-valid split.
    """

    in_bytes = INPUT_BYTES[workload.dtype]
    out_bytes = OUTPUT_BYTES[workload.dtype]
    threshold = (
        _l2_target_bytes(hardware)
        if resident_threshold_bytes is None
        else max(
            1.0,
            min(float(hardware.l2_bytes), resident_threshold_bytes),
        )
    )
    m_tasks = ceil_div(workload.m, base_m)
    n_tasks = ceil_div(workload.n, base_n)
    whole = (1, 1, m_tasks, n_tasks, 0)
    total_bytes = (
        workload.m * workload.k * in_bytes
        + workload.k * workload.n * in_bytes
        + workload.m * workload.n * out_bytes
    )
    if total_bytes <= threshold:
        return whole, "whole_resident"

    if base_n >= base_m:
        out_base, inner_base = base_n, base_m
        out_value, inner_value = workload.n, workload.m
        inner_bad = workload.trans_a
        output_is_n = True
    else:
        out_base, inner_base = base_m, base_n
        out_value, inner_value = workload.m, workload.n
        inner_bad = not workload.trans_b
        output_is_n = False

    if hardware.aic_cores == 20:
        max_conflict_dim = 5
        min_conflict_dim = 4
    else:
        max_conflict_dim = min(hardware.aic_cores, 6)
        min_conflict_dim = min(hardware.aic_cores, 3)
    inner_max_conflict = (
        min_conflict_dim if inner_bad else max_conflict_dim
    )
    outer_min_use_dim = max(
        1, hardware.aic_cores // max(1, max_conflict_dim)
    )
    inner_min_use_dim = max(
        1, hardware.aic_cores // max(1, inner_max_conflict)
    )

    selected: tuple[int, int, int, int] | None = None
    out_conflict = 0
    inner_conflict = 0
    for outer_use_dim in range(
        hardware.aic_cores, outer_min_use_dim - 1, -1
    ):
        for inner_use_dim in range(
            hardware.aic_cores, inner_min_use_dim - 1, -1
        ):
            out_tile = max(
                1, out_value // (out_base * outer_use_dim)
            )
            inner_tile = max(
                1, inner_value // (inner_base * inner_use_dim)
            )
            out_split = align_up(
                ceil_div(out_value, out_tile), out_base
            )
            inner_split = align_up(
                ceil_div(inner_value, inner_tile), inner_base
            )
            split_bytes = (
                out_split * workload.k * in_bytes
                + workload.k * inner_split * in_bytes
                + out_split * inner_split * out_bytes
            )
            if split_bytes > threshold:
                continue

            out_tail_value = (
                (out_value + out_split - 1) % out_split
            ) + 1
            inner_tail_value = (
                (inner_value + inner_split - 1) % inner_split
            ) + 1
            out_tail_count = ceil_div(out_tail_value, out_base)
            inner_tail_count = ceil_div(
                inner_tail_value, inner_base
            )
            if (
                out_tail_count * max_conflict_dim
                < hardware.aic_cores
                or inner_tail_count * inner_max_conflict
                < hardware.aic_cores
            ):
                continue

            next_out_conflict = ceil_div(
                hardware.aic_cores, out_tail_count
            )
            next_inner_conflict = ceil_div(
                hardware.aic_cores, inner_tail_count
            )
            if (
                selected is None
                or (
                    out_conflict >= next_out_conflict
                    and inner_conflict >= next_inner_conflict
                )
            ):
                selected = (
                    out_tile,
                    inner_tile,
                    out_split,
                    inner_split,
                )
                out_conflict = next_out_conflict
                inner_conflict = next_inner_conflict

    if selected is None:
        m_block = min(4, m_tasks)
        n_block = min(
            max(1, hardware.aic_cores // 4), n_tasks
        )
        return (
            ceil_div(m_tasks, m_block),
            ceil_div(n_tasks, n_block),
            m_block,
            n_block,
            0,
        ), "fixed_fallback"

    _, _, out_split, inner_split = selected
    if output_is_n:
        n_split, m_split = out_split, inner_split
    else:
        m_split, n_split = out_split, inner_split
    m_block = ceil_div(m_split, base_m)
    n_block = ceil_div(n_split, base_n)
    return (
        ceil_div(workload.m, m_block * base_m),
        ceil_div(workload.n, n_block * base_n),
        m_block,
        n_block,
        0,
    ), "cache_partitioned"


@lru_cache(maxsize=4096)
def l2_policy_variants(
    workload: Workload,
    hardware: Hardware,
    *,
    single_m: int,
    single_n: int,
    used_cores: int,
) -> tuple[tuple[int, int, int, int, int], ...]:
    """Enumerate coherent L2 tiles using upstream capacity and tail signals."""

    m_total = ceil_div(workload.m, single_m)
    n_total = ceil_div(workload.n, single_n)
    in_bytes = INPUT_BYTES[workload.dtype]
    target_bytes = _l2_target_bytes(hardware)
    core_count = max(1, used_cores)

    def working_set(m_block: int, n_block: int) -> float:
        m_extent = min(workload.m, m_block * single_m)
        n_extent = min(workload.n, n_block * single_n)
        return float(
            m_extent * workload.k * in_bytes
            + workload.k * n_extent * in_bytes
            + m_extent * n_extent * in_bytes
        )

    candidates = []
    for m_block in range(1, m_total + 1):
        for n_block in range(1, n_total + 1):
            resident = working_set(m_block, n_block)
            if resident > target_bytes:
                continue
            m_count = ceil_div(m_total, m_block)
            n_count = ceil_div(n_total, n_block)
            m_tail = m_total - (m_count - 1) * m_block
            n_tail = n_total - (n_count - 1) * n_block
            slot_count = 0
            task_count = 0
            for m_size, m_repeats in (
                (m_block, max(0, m_count - 1)),
                (m_tail, 1),
            ):
                for n_size, n_repeats in (
                    (n_block, max(0, n_count - 1)),
                    (n_tail, 1),
                ):
                    repeats = m_repeats * n_repeats
                    if repeats == 0:
                        continue
                    tasks = m_size * n_size
                    task_count += repeats * tasks
                    slot_count += (
                        repeats * ceil_div(tasks, core_count) * core_count
                    )
            wave_efficiency = task_count / max(1.0, slot_count)
            tail_efficiency = min(
                m_tail / max(1.0, m_block),
                n_tail / max(1.0, n_block),
            )
            small_tail = (
                m_tail * min(5, core_count) < core_count
                or n_tail * min(4, core_count) < core_count
            )
            capacity_distance = abs(
                math.log(max(1.0e-9, resident / target_bytes))
            )
            candidates.append(
                (
                    (
                        float(small_tail),
                        -wave_efficiency,
                        -tail_efficiency,
                        capacity_distance,
                        m_count * n_count,
                    ),
                    (m_count, n_count, m_block, n_block, 0),
                )
            )

    balanced_m = min(m_total, 4)
    balanced_n = min(n_total, max(1, core_count // max(1, balanced_m)))
    balanced_partition = (
        ceil_div(m_total, balanced_m),
        ceil_div(n_total, balanced_n),
        balanced_m,
        balanced_n,
        0,
    )
    paired_n_partition = (
        m_total,
        ceil_div(n_total, min(2, n_total)),
        1,
        min(2, n_total),
        0,
    )
    whole = (1, 1, m_total, n_total, 0)
    if working_set(m_total, n_total) <= target_bytes:
        ordered = [balanced_partition]
        ordered.extend(value for _, value in sorted(candidates))
    else:
        ordered = [value for _, value in sorted(candidates)]
        ordered.append(balanced_partition)
    if core_count == 2:
        ordered.insert(0, paired_n_partition)
    ordered.append(whole)

    result = []
    seen = set()
    for value in ordered:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
        if len(result) >= 64:
            break
    return tuple(result)
