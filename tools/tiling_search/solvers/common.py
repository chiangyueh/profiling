from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence

from ..contracts import align_up, base_k_alignment, ceil_div
from ..domain import (
    INPUT_BYTES,
    BehaviorTarget,
    Hardware,
    Schedule,
    Workload,
)


def _aligned_values(limit: int, alignment: int) -> list[int]:
    canonical = (
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        10,
        12,
        14,
        16,
        20,
        24,
        28,
        32,
        40,
        48,
        56,
        64,
        80,
        96,
        112,
        128,
    )
    values = {
        alignment * factor
        for factor in canonical
        if alignment * factor <= limit
    }
    values.add(alignment)
    values.add(max(alignment, limit // alignment * alignment))
    for numerator, denominator in ((1, 4), (1, 2), (3, 4)):
        value = limit * numerator // denominator
        values.add(max(alignment, value // alignment * alignment))
    return sorted(value for value in values if value > 0)


def tile_limits(
    workload: Workload,
    hardware: Hardware,
) -> tuple[int, int, int]:
    in_bytes = INPUT_BYTES[workload.dtype]
    k_align = base_k_alignment(workload)
    max_m = min(
        hardware.l0a_bytes // max(1, k_align * in_bytes),
        hardware.l0c_bytes // (16 * 4),
    )
    max_n = min(
        hardware.l0b_bytes // max(1, k_align * in_bytes),
        hardware.l0c_bytes // (16 * 4),
    )
    max_k = min(
        hardware.l0a_bytes // (16 * in_bytes),
        hardware.l0b_bytes // (16 * in_bytes),
    )
    return (
        max(16, max_m // 16 * 16),
        max(16, max_n // 16 * 16),
        max(k_align, max_k // k_align * k_align),
    )


def _target_tile_values(
    workload: Workload,
    hardware: Hardware,
    targets: Sequence[BehaviorTarget],
) -> tuple[set[int], set[int], set[int]]:
    in_bytes = INPUT_BYTES[workload.dtype]
    k_align = base_k_alignment(workload)
    m_values: set[int] = set()
    n_values: set[int] = set()
    k_values: set[int] = set()
    max_m, max_n, max_k = tile_limits(workload, hardware)
    for target in targets:
        occupancy = target.l0_occupancy
        if occupancy is None:
            continue
        occupancy = min(0.98, max(0.05, occupancy))
        for db in (1, 2):
            for base_k in _aligned_values(max_k, k_align):
                m_extent = int(
                    occupancy * hardware.l0a_bytes
                    / max(1, base_k * in_bytes * db)
                )
                n_extent = int(
                    occupancy * hardware.l0b_bytes
                    / max(1, base_k * in_bytes * db)
                )
                m_values.add(max(16, min(max_m, align_up(m_extent, 16))))
                n_values.add(max(16, min(max_n, align_up(n_extent, 16))))
                k_values.add(base_k)
    return m_values, n_values, k_values


def tile_specs(
    workload: Workload,
    hardware: Hardware,
    targets: Sequence[BehaviorTarget],
) -> list[tuple[int, int, int, int, int, int]]:
    in_bytes = INPUT_BYTES[workload.dtype]
    k_align = base_k_alignment(workload)
    max_m, max_n, hardware_max_k = tile_limits(workload, hardware)
    max_k = min(hardware_max_k, align_up(workload.k, k_align))
    m_values = set(_aligned_values(max_m, 16))
    n_values = set(_aligned_values(max_n, 16))
    k_values = set(_aligned_values(max_k, k_align))
    target_m, target_n, target_k = _target_tile_values(
        workload, hardware, targets
    )
    m_values.update(target_m)
    n_values.update(target_n)
    k_values.update(value for value in target_k if value <= max_k)

    specs: list[
        tuple[tuple[int, int, int, int], tuple[int, int, int, int, int, int]]
    ] = []
    for base_m in sorted(m_values):
        for base_n in sorted(n_values):
            for base_k in sorted(k_values):
                for db_a, db_b, db_c in (
                    (1, 1, 1),
                    (2, 2, 1),
                    (2, 2, 2),
                    (1, 2, 1),
                    (2, 1, 1),
                ):
                    l0a = base_m * base_k * in_bytes * db_a
                    l0b = base_n * base_k * in_bytes * db_b
                    l0c = base_m * base_n * 4 * db_c
                    if (
                        l0a > hardware.l0a_bytes
                        or l0b > hardware.l0b_bytes
                        or l0c > hardware.l0c_bytes
                    ):
                        continue
                    occupancy = max(
                        l0a / hardware.l0a_bytes,
                        l0b / hardware.l0b_bytes,
                        l0c / hardware.l0c_bytes,
                    )
                    bucket = min(9, int(occupancy * 10))
                    aspect = min(
                        7,
                        int(abs(math.log2(base_m / max(1, base_n))) * 2),
                    )
                    k_bucket = min(7, int(math.log2(max(1, base_k / k_align))))
                    db_bucket = db_a + 2 * db_b + 4 * db_c
                    specs.append(
                        (
                            (bucket, aspect, k_bucket, db_bucket),
                            (base_m, base_n, base_k, db_a, db_b, db_c),
                        )
                    )

    # Interleave hardware behavior strata so a lazy prefix is not dominated by
    # small tiles or one buffering mode.
    buckets: dict[tuple[int, int, int, int], list[tuple[int, ...]]] = {}
    for key, spec in specs:
        buckets.setdefault(key, []).append(spec)
    ordered: list[tuple[int, int, int, int, int, int]] = []
    keys = sorted(buckets)
    while any(buckets.values()):
        for key in keys:
            if buckets[key]:
                ordered.append(buckets[key].pop(len(buckets[key]) // 2))
    return ordered


def partition_geometries(
    workload: Workload,
    hardware: Hardware,
    base_m: int,
    base_n: int,
    targets: Sequence[BehaviorTarget],
) -> list[tuple[int, int, int]]:
    core_limit = min(workload.max_cores, hardware.aic_cores)
    target_rounds = {1, 2, 4, 8}
    target_rounds.update(
        max(1, int(round(target.core_rounds)))
        for target in targets
        if target.core_rounds is not None
    )
    geometries: set[tuple[int, int, int]] = set()
    for rounds in sorted(target_rounds):
        task_target = max(1, core_limit * rounds)
        for m_parts in range(1, min(task_target, core_limit * 2) + 1):
            n_parts = ceil_div(task_target, m_parts)
            for adjusted_n in {max(1, n_parts - 1), n_parts, n_parts + 1}:
                single_m = max(
                    base_m,
                    align_up(ceil_div(workload.m, m_parts), 16),
                )
                single_n = max(
                    base_n,
                    align_up(ceil_div(workload.n, adjusted_n), 16),
                )
                m_tasks = ceil_div(workload.m, single_m)
                n_tasks = ceil_div(workload.n, single_n)
                tasks = max(1, m_tasks * n_tasks)
                geometries.add((single_m, single_n, min(core_limit, tasks)))
    return sorted(
        geometries,
        key=lambda value: (
            ceil_div(
                ceil_div(workload.m, value[0])
                * ceil_div(workload.n, value[1]),
                value[2],
            ),
            abs(value[0] / max(1, value[1]) - workload.m / max(1, workload.n)),
            value,
        ),
    )


def l1_variants(
    workload: Workload,
    hardware: Hardware,
    *,
    base_m: int,
    base_n: int,
    base_k: int,
    single_m: int,
    single_n: int,
    targets: Sequence[BehaviorTarget],
) -> list[tuple[int, int, int, int, int, int]]:
    in_bytes = INPUT_BYTES[workload.dtype]
    m_steps = {
        1,
        min(2, max(1, ceil_div(single_m, base_m))),
        min(4, max(1, ceil_div(single_m, base_m))),
    }
    n_steps = {
        1,
        min(2, max(1, ceil_div(single_n, base_n))),
        min(4, max(1, ceil_div(single_n, base_n))),
    }
    k_tiles = max(1, ceil_div(workload.k, base_k))
    k_steps = {1, min(2, k_tiles), min(4, k_tiles), min(8, k_tiles)}
    variants: list[tuple[int, int, int, int, int, int]] = []
    for step_m in sorted(m_steps):
        for step_n in sorted(n_steps):
            for step_k in sorted(k_steps):
                for a_buffers in (1, 2):
                    for b_buffers in (1, 2):
                        depth_a = step_m * step_k * a_buffers
                        depth_b = step_n * step_k * b_buffers
                        l1_bytes = (
                            base_m * base_k * depth_a * in_bytes
                            + base_n * base_k * depth_b * in_bytes
                        )
                        if l1_bytes > hardware.effective_l1_bytes:
                            continue
                        variants.append(
                            (
                                depth_a,
                                depth_b,
                                step_m,
                                step_n,
                                step_k,
                                step_k,
                            )
                        )
    if not variants:
        return []

    def distance(variant: tuple[int, ...]) -> tuple[float, tuple[int, ...]]:
        depth_a, depth_b, _, _, _, _ = variant
        occupancy = (
            base_m * base_k * depth_a * in_bytes
            + base_n * base_k * depth_b * in_bytes
        ) / hardware.effective_l1_bytes
        requested = [
            abs(occupancy - target.l1_occupancy)
            for target in targets
            if target.l1_occupancy is not None
        ]
        return (min(requested, default=abs(occupancy - 0.65)), variant)

    return sorted(set(variants), key=distance)


def l2_variants(
    workload: Workload,
    hardware: Hardware,
    single_m: int,
    single_n: int,
    targets: Sequence[BehaviorTarget],
) -> list[tuple[int, int, int, int, int]]:
    m_total = ceil_div(workload.m, single_m)
    n_total = ceil_div(workload.n, single_n)
    in_bytes = INPUT_BYTES[workload.dtype]
    target_ratios = {
        min(1.0, max(0.02, target.l2_working_set_ratio))
        for target in targets
        if target.l2_working_set_ratio is not None
    }
    target_ratios.update((0.15, 0.35, 0.65, 0.95))
    variants: set[tuple[int, int, int, int, int]] = set()
    for ratio in sorted(target_ratios):
        budget = max(1.0, hardware.l2_bytes * ratio)
        target_elements = max(
            1,
            int(
                budget
                / max(
                    1,
                    (single_m + single_n) * workload.k * in_bytes
                    + single_m * single_n * in_bytes,
                )
            ),
        )
        m_options = {
            1,
            m_total,
            max(1, min(m_total, int(math.sqrt(target_elements)))),
        }
        for m_block in m_options:
            n_block = max(1, min(n_total, ceil_div(target_elements, m_block)))
            for order in (0, 1, 2):
                variants.add(
                    (
                        ceil_div(m_total, m_block),
                        ceil_div(n_total, n_block),
                        m_block,
                        n_block,
                        order,
                    )
                )
    variants.add((1, 1, m_total, n_total, 0))
    variants.add((1, 1, m_total, n_total, 1))
    return sorted(variants)


def interleave(iterables: Iterable[Iterable[Schedule]]) -> Iterator[Schedule]:
    iterators = [iter(values) for values in iterables]
    while iterators:
        remaining = []
        for iterator in iterators:
            try:
                yield next(iterator)
                remaining.append(iterator)
            except StopIteration:
                pass
        iterators = remaining


def make_schedule(**values: int) -> Schedule:
    return Schedule.from_mapping(values)
