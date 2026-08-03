from __future__ import annotations

import math
from functools import lru_cache

from ..contracts import align_up, ceil_div
from ..domain import INPUT_BYTES, Hardware, Workload


_REFERENCE_L2_BYTES = 192 * 1024 * 1024
_REFERENCE_L2_WORKING_SET = 100 * 1024 * 1024


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

    geometries = [
        preferred,
        (preferred[1], preferred[0]),
        (128, 128),
        (64, 256),
        (256, 64),
    ]
    result = []
    seen = set()
    for base_m, base_n in geometries:
        if (base_m, base_n) in seen:
            continue
        seen.add((base_m, base_n))
        db_a = (
            2
            if base_m * base_k * in_bytes * 2 <= hardware.l0a_bytes
            else 1
        )
        db_b = (
            2
            if base_n * base_k * in_bytes * 2 <= hardware.l0b_bytes
            else 1
        )
        db_c = (
            2
            if base_m * base_n * 4 * 2 <= hardware.l0c_bytes
            else 1
        )
        if (
            base_m * base_k * in_bytes * db_a > hardware.l0a_bytes
            or base_n * base_k * in_bytes * db_b > hardware.l0b_bytes
            or base_m * base_n * 4 * db_c > hardware.l0c_bytes
        ):
            continue
        result.append((base_m, base_n, base_k, db_a, db_b, db_c))
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
    ordered.extend(
        value
        for value in sorted(values, key=score)
        if value != direct
    )
    return tuple(ordered)


def _l2_target_bytes(hardware: Hardware) -> float:
    scaled = (
        _REFERENCE_L2_WORKING_SET
        * hardware.l2_bytes
        / _REFERENCE_L2_BYTES
    )
    return max(1.0, min(float(hardware.l2_bytes), scaled))


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
    whole = (1, 1, m_total, n_total, 0)
    if working_set(m_total, n_total) <= target_bytes:
        ordered = [balanced_partition]
        ordered.extend(value for _, value in sorted(candidates))
    else:
        ordered = [value for _, value in sorted(candidates)]
        ordered.append(balanced_partition)
    ordered.append(whole)

    result = []
    seen = set()
    for value in ordered:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
        if len(result) >= 16:
            break
    return tuple(result)
