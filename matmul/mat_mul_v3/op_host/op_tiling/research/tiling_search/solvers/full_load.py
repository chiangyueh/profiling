from __future__ import annotations

from collections.abc import Iterator, Sequence

from ..contracts import (
    align_up,
    base_k_alignment,
    bl1_fix_geometry,
    bl1_fix_mode_supported,
    bl1_official_fix_applicable,
    ceil_div,
)
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
    tile_specs,
)


class Al1FullLoadSolver:
    template = Template.AL1_FULL_LOAD
    source = "contract_global"

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
    source = "contract_global"

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

        fix_geometry = bl1_fix_geometry(workload, hardware)
        if fix_geometry is not None:
            for fix, mode in ((1, 1020), (2, 2020)):
                if not bl1_official_fix_applicable(workload, fix):
                    continue
                base_m = fix_geometry["baseM"]
                yield make_schedule(
                    usedCoreNum=min(
                        workload.max_cores,
                        hardware.aic_cores,
                        ceil_div(workload.m, base_m),
                    ),
                    singleCoreM=base_m,
                    singleCoreN=fix_geometry["baseN"],
                    singleCoreK=workload.k,
                    baseM=base_m,
                    baseN=fix_geometry["baseN"],
                    baseK=fix_geometry["baseK"],
                    depthA1=fix_geometry["depthA1"],
                    depthB1=fix_geometry["depthB1"],
                    stepM=1,
                    stepN=1,
                    iterateOrder=0,
                    stepKa=fix_geometry["stepKa"],
                    stepKb=fix_geometry["stepKb"],
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

        # The compiled fix-output kernels support a broader BL1-resident
        # geometry than the official constructor's profitability gate. Build
        # a bounded capacity-derived grid so a rejected geometry has genuinely
        # different base-K and L0 shapes available on the next NPU pass.
        k_alignment = base_k_alignment(workload)
        fix_base_ks = tuple(
            value
            for value in (
                k_alignment,
                2 * k_alignment,
                4 * k_alignment,
                8 * k_alignment,
            )
            if value <= align_up(workload.k, k_alignment)
        )
        for fix_base_k in fix_base_ks:
            max_base_m = min(
                align_up(workload.m, 16),
                hardware.l0a_bytes
                // max(1, fix_base_k * in_bytes * 2)
                // 16
                * 16,
            )
            max_base_n = min(
                align_up(workload.n, 16),
                hardware.l0b_bytes
                // max(1, fix_base_k * in_bytes * 2)
                // 16
                * 16,
            )
            base_m_values = tuple(
                dict.fromkeys(
                    value // 16 * 16
                    for value in (max_base_m, max_base_m // 2)
                    if value >= 16
                )
            )
            for fix_base_m in base_m_values:
                l0c_base_n = (
                    hardware.l0c_bytes
                    // max(1, fix_base_m * 4)
                    // 16
                    * 16
                )
                capped_base_n = min(max_base_n, l0c_base_n)
                base_n_values = tuple(
                    dict.fromkeys(
                        value // 16 * 16
                        for value in (
                            capped_base_n,
                            capped_base_n // 2,
                        )
                        if value >= 16
                    )
                )
                for fix_base_n in base_n_values:
                    step_n = ceil_div(workload.n, fix_base_n)
                    step_k = ceil_div(workload.k, fix_base_k)
                    cores = min(
                        workload.max_cores,
                        hardware.aic_cores,
                        ceil_div(workload.m, fix_base_m),
                    )
                    for fix, mode in ((1, 1020), (2, 2020)):
                        if not bl1_fix_mode_supported(workload, fix):
                            continue
                        yield make_schedule(
                            usedCoreNum=max(1, cores),
                            singleCoreM=fix_base_m,
                            singleCoreN=align_up(workload.n, 16),
                            singleCoreK=workload.k,
                            baseM=fix_base_m,
                            baseN=fix_base_n,
                            baseK=fix_base_k,
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

        n_alignment = 16 if workload.trans_b else 32 // in_bytes
        for index, (
            base_m,
            base_n,
            base_k,
            db_a,
            db_b,
            db_c,
        ) in enumerate(tile_specs(workload, hardware, targets)):
            base_n = min(base_n, align_up(workload.n, n_alignment))
            base_n = align_up(base_n, n_alignment)
            step_n = ceil_div(workload.n, base_n)
            step_k = ceil_div(workload.k, base_k)
            depth_a = 2 * step_k
            depth_b = step_n * step_k
            l1_bytes = base_k * (
                depth_a * base_m + depth_b * base_n
            ) * in_bytes
            if l1_bytes > hardware.effective_l1_bytes:
                continue
            single_m = 2 * base_m
            cores = min(
                workload.max_cores,
                hardware.aic_cores,
                ceil_div(workload.m, single_m),
            )
            yield make_schedule(
                usedCoreNum=max(1, cores),
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
                iterateOrder=index % 2,
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
