from __future__ import annotations

from collections.abc import Iterator, Sequence

from ..contracts import align_up, base_k_alignment, ceil_div
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


class Al1FullLoadSolver:
    template = Template.AL1_FULL_LOAD

    def generate(
        self,
        workload: Workload,
        hardware: Hardware,
        targets: Sequence[BehaviorTarget] = (),
    ) -> Iterator[Schedule]:
        in_bytes = INPUT_BYTES[workload.dtype]
        resident_a = (
            align_up(workload.m, 16)
            * align_up(workload.k, base_k_alignment(workload))
            * in_bytes
        )
        if resident_a >= hardware.effective_l1_bytes:
            return
        for index, (
            base_m,
            base_n,
            base_k,
            db_a,
            db_b,
            db_c,
        ) in enumerate(tile_specs(workload, hardware, targets)):
            if base_m > align_up(workload.m, 16):
                continue
            step_ka = ceil_div(workload.k, base_k)
            for n_parts in range(
                1, min(workload.max_cores, hardware.aic_cores) + 1
            ):
                single_m = max(base_m, align_up(workload.m, 16))
                single_n = max(
                    base_n,
                    align_up(ceil_div(workload.n, n_parts), 16),
                )
                tasks = ceil_div(workload.n, single_n)
                l2 = l2_variants(
                    workload, hardware, single_m, single_n, targets
                )
                for depth_b in (1, 2):
                    if not l2:
                        continue
                    (
                        l2_m_count,
                        l2_n_count,
                        l2_m_block,
                        l2_n_block,
                        l2_order,
                    ) = l2[(index + depth_b) % len(l2)]
                    yield make_schedule(
                        usedCoreNum=min(
                            workload.max_cores, hardware.aic_cores, tasks
                        ),
                        singleCoreM=single_m,
                        singleCoreN=single_n,
                        singleCoreK=workload.k,
                        baseM=base_m,
                        baseN=base_n,
                        baseK=base_k,
                        depthA1=step_ka,
                        depthB1=depth_b,
                        stepM=1,
                        stepN=1,
                        iterateOrder=index % 2,
                        stepKa=step_ka,
                        stepKb=1,
                        dbL0A=db_a,
                        dbL0B=db_b,
                        dbL0C=db_c,
                        l2MTileCnt=l2_m_count,
                        l2NTileCnt=l2_n_count,
                        l2MTileBlock=l2_m_block,
                        l2NTileBlock=l2_n_block,
                        l2IterateOrder=l2_order,
                        tilingEnable=10,
                    )


class Bl1FullLoadSolver:
    template = Template.BL1_FULL_LOAD

    def generate(
        self,
        workload: Workload,
        hardware: Hardware,
        targets: Sequence[BehaviorTarget] = (),
    ) -> Iterator[Schedule]:
        in_bytes = INPUT_BYTES[workload.dtype]
        resident_b = (
            align_up(workload.k, base_k_alignment(workload))
            * align_up(workload.n, 32 // in_bytes)
            * in_bytes
        )
        if resident_b >= hardware.effective_l1_bytes:
            return
        base_k = 64
        max_base_m = (
            hardware.l0a_bytes
            // max(1, base_k * in_bytes * 2)
            // 16
            * 16
        )
        max_base_n = min(
            hardware.l0b_bytes // max(1, base_k * in_bytes * 2),
            hardware.l0c_bytes // max(1, max_base_m * 4),
        )
        max_base_n = max_base_n // 16 * 16
        if max_base_m >= 16 and max_base_n >= 16:
            base_m = min(max_base_m, align_up(workload.m, 16))
            base_n = min(max_base_n, align_up(workload.n, 16))
            step_n = ceil_div(workload.n, base_n)
            step_k = ceil_div(workload.k, base_k)
            cores = min(
                workload.max_cores,
                hardware.aic_cores,
                ceil_div(workload.m, base_m),
            )
            modes = [1020]
            if workload.dtype == "fp32" and not workload.trans_a:
                modes.append(2020)
            for mode in modes:
                yield make_schedule(
                    usedCoreNum=max(1, cores),
                    singleCoreM=base_m,
                    singleCoreN=max(base_n, align_up(workload.n, 16)),
                    singleCoreK=workload.k,
                    baseM=base_m,
                    baseN=base_n,
                    baseK=base_k,
                    depthA1=step_k,
                    depthB1=step_n * step_k,
                    stepM=1,
                    stepN=step_n,
                    iterateOrder=0,
                    stepKa=step_k,
                    stepKb=step_k,
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
        for index, (
            base_m,
            base_n,
            base_k,
            db_a,
            db_b,
            db_c,
        ) in enumerate(tile_specs(workload, hardware, targets)):
            step_n = ceil_div(workload.n, base_n)
            step_k = ceil_div(workload.k, base_k)
            single_n = max(base_n, align_up(workload.n, 16))
            geometries = partition_geometries(
                workload, hardware, base_m, single_n, targets
            )
            if not geometries:
                continue
            single_m, _, cores = geometries[index % len(geometries)]
            l2 = l2_variants(
                workload, hardware, single_m, single_n, targets
            )
            if not l2:
                continue
            for mode in (20,):
                for a_buffers in (1, 2):
                    for b_buffers in (1, 2):
                        (
                            l2_m_count,
                            l2_n_count,
                            l2_m_block,
                            l2_n_block,
                            l2_order,
                        ) = l2[
                            (index + a_buffers + b_buffers) % len(l2)
                        ]
                        yield make_schedule(
                            usedCoreNum=cores,
                            singleCoreM=single_m,
                            singleCoreN=single_n,
                            singleCoreK=workload.k,
                            baseM=base_m,
                            baseN=base_n,
                            baseK=base_k,
                            depthA1=step_k * a_buffers,
                            depthB1=step_n * step_k * b_buffers,
                            stepM=1,
                            stepN=step_n,
                            iterateOrder=index % 2,
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
                            tilingEnable=mode,
                        )
