from __future__ import annotations

from collections.abc import Iterator, Sequence

from ..contracts import align_up, ceil_div
from ..domain import (
    INPUT_BYTES,
    BehaviorTarget,
    Hardware,
    Schedule,
    Template,
    Workload,
)
from .common import (
    l2_variants,
    make_schedule,
    partition_geometries,
    tile_specs,
)


class SingleCoreSplitKSolver:
    template = Template.SINGLE_CORE_SPLIT_K

    def generate(
        self,
        workload: Workload,
        hardware: Hardware,
        targets: Sequence[BehaviorTarget] = (),
    ) -> Iterator[Schedule]:
        specs = tile_specs(workload, hardware, targets)
        for index, (
            base_m,
            base_n,
            base_k,
            db_a,
            db_b,
            db_c,
        ) in enumerate(specs):
            max_step_k = max(1, (workload.k - 1) // base_k)
            step_k_values = {
                value
                for value in (1, 2, 3, 4, 6, 8)
                if value <= max_step_k
            }
            step_k_values.update(
                max(
                    1,
                    min(max_step_k, int(round(target.k_passes or 1))),
                )
                for target in targets
                if target.k_passes is not None
            )
            for step_k in sorted(step_k_values):
                single_k = step_k * base_k
                if single_k >= workload.k:
                    continue
                step_m = (1, 2, 3, 4)[index % 4]
                step_n = (1, 3, 2, 4)[index % 4]
                inner_m = step_m * base_m
                inner_n = step_n * base_n
                geometries = partition_geometries(
                    workload, hardware, inner_m, inner_n, targets
                )
                if not geometries:
                    continue
                single_m, single_n, cores = geometries[index % len(geometries)]
                for a_buffers, b_buffers in ((1, 1), (2, 1), (1, 2), (2, 2)):
                    l2 = l2_variants(
                        workload, hardware, single_m, single_n, targets
                    )
                    if not l2:
                        continue
                    (
                        l2_m_count,
                        l2_n_count,
                        l2_m_block,
                        l2_n_block,
                        l2_order,
                    ) = l2[(index + a_buffers + b_buffers) % len(l2)]
                    yield make_schedule(
                        usedCoreNum=cores,
                        singleCoreM=max(single_m, inner_m),
                        singleCoreN=max(single_n, inner_n),
                        singleCoreK=single_k,
                        baseM=base_m,
                        baseN=base_n,
                        baseK=base_k,
                        depthA1=step_m * step_k * a_buffers,
                        depthB1=step_n * step_k * b_buffers,
                        stepM=step_m,
                        stepN=step_n,
                        iterateOrder=(index + step_k) % 2,
                        stepKa=step_k,
                        stepKb=step_k,
                        dbL0A=db_a,
                        dbL0B=db_b,
                        dbL0C=db_c,
                        l2MTileCnt=l2_m_count,
                        l2NTileCnt=l2_n_count,
                        l2MTileBlock=l2_m_block,
                        l2NTileBlock=l2_n_block,
                        l2IterateOrder=l2_order,
                        tilingEnable=2,
                    )


class DeterministicSplitKSolver:
    template = Template.DETERMINISTIC_SPLIT_K

    def generate(
        self,
        workload: Workload,
        hardware: Hardware,
        targets: Sequence[BehaviorTarget] = (),
    ) -> Iterator[Schedule]:
        del targets
        in_bytes = INPUT_BYTES[workload.dtype]
        base_k = 256 // in_bytes
        single_k = 3 * base_k
        k_chunks = ceil_div(workload.k, single_k)
        core_limit = min(workload.max_cores, hardware.aic_cores, k_chunks)
        if k_chunks < 2 or core_limit <= 0:
            return
        layouts = (
            (384, max(128, align_up(workload.n, 16)), 3, 1, 9, 6, 1, 0),
            (max(128, align_up(workload.m, 16)), 384, 1, 3, 6, 9, 0, 1),
        )
        core_values = sorted(
            {
                1,
                min(2, core_limit),
                min(4, core_limit),
                min(8, core_limit),
                min(16, core_limit),
                core_limit,
            }
        )
        for (
            single_m,
            single_n,
            step_m,
            step_n,
            depth_a,
            depth_b,
            order,
            l2_order,
        ) in layouts:
            m_tasks = ceil_div(workload.m, single_m)
            n_tasks = ceil_div(workload.n, single_n)
            for cores in core_values:
                yield make_schedule(
                    usedCoreNum=cores,
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
                    stepKa=3,
                    stepKb=3,
                    dbL0A=2,
                    dbL0B=2,
                    dbL0C=2,
                    l2MTileCnt=1,
                    l2NTileCnt=1,
                    l2MTileBlock=max(1, m_tasks),
                    l2NTileBlock=max(1, n_tasks),
                    l2IterateOrder=l2_order,
                    tilingEnable=3,
                )
