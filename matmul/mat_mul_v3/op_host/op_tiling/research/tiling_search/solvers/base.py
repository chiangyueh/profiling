from __future__ import annotations

from collections.abc import Iterator, Sequence

from ..contracts import ceil_div
from ..domain import BehaviorTarget, Hardware, Schedule, Template, Workload
from .common import (
    l1_variants,
    l2_variants,
    make_schedule,
    partition_geometries,
    tile_specs,
)
from .base_policy import (
    base_geometry_variants,
    core_partition_variants,
    l1_pipeline_variant,
    l2_policy_variants,
)


class BaseSolver:
    template = Template.BASE

    def generate(
        self,
        workload: Workload,
        hardware: Hardware,
        targets: Sequence[BehaviorTarget] = (),
    ) -> Iterator[Schedule]:
        # Lead with coupled schedules reconstructed from the upstream BASE
        # policy. The broad contract space below remains available for
        # exploration, but cannot dominate a lazy prefix with arbitrary
        # combinations of L1, L2 and buffering fields.
        for (
            base_m,
            base_n,
            base_k,
            db_a,
            db_b,
            db_c,
        ) in base_geometry_variants(workload, hardware):
            l1 = l1_pipeline_variant(
                workload,
                hardware,
                base_m=base_m,
                base_n=base_n,
                base_k=base_k,
            )
            if l1 is None:
                continue
            (
                depth_a,
                depth_b,
                step_m,
                step_n,
                step_ka,
                step_kb,
            ) = l1
            geometries = core_partition_variants(
                workload,
                hardware,
                minimum_m=base_m,
                minimum_n=base_n,
            )
            for single_m, single_n, cores in geometries[:4]:
                for (
                    l2_m_count,
                    l2_n_count,
                    l2_m_block,
                    l2_n_block,
                    l2_order,
                ) in l2_policy_variants(
                    workload,
                    hardware,
                    single_m=single_m,
                    single_n=single_n,
                    used_cores=cores,
                )[:8]:
                    yield make_schedule(
                        usedCoreNum=cores,
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

        specs = tile_specs(workload, hardware, targets)
        for variant_round in range(4):
            for index, (
                base_m,
                base_n,
                base_k,
                db_a,
                db_b,
                db_c,
            ) in enumerate(specs):
                if variant_round == 0:
                    single_m = base_m
                    single_n = base_n
                    output_tasks = (
                        ceil_div(workload.m, single_m)
                        * ceil_div(workload.n, single_n)
                    )
                    cores = min(
                        workload.max_cores,
                        hardware.aic_cores,
                        output_tasks,
                    )
                else:
                    geometries = partition_geometries(
                        workload, hardware, base_m, base_n, targets
                    )
                    if not geometries:
                        continue
                    single_m, single_n, cores = geometries[
                        (index + variant_round - 1) % len(geometries)
                    ]
                l1 = l1_variants(
                    workload,
                    hardware,
                    base_m=base_m,
                    base_n=base_n,
                    base_k=base_k,
                    single_m=single_m,
                    single_n=single_n,
                    targets=targets,
                )
                l2 = l2_variants(
                    workload, hardware, single_m, single_n, targets
                )
                if not l1 or not l2:
                    continue
                (
                    depth_a,
                    depth_b,
                    step_m,
                    step_n,
                    step_ka,
                    step_kb,
                ) = l1[(index + variant_round) % len(l1)]
                (
                    l2_m_count,
                    l2_n_count,
                    l2_m_block,
                    l2_n_block,
                    l2_order,
                ) = l2[(index * 3 + variant_round) % len(l2)]
                yield make_schedule(
                    usedCoreNum=cores,
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
                    iterateOrder=(index + variant_round) % 2,
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
