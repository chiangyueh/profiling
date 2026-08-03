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
from .base_policy import core_partition_variants, l2_policy_variants
from .common import make_schedule


class SingleCoreSplitKSolver:
    template = Template.SINGLE_CORE_SPLIT_K

    def generate(
        self,
        workload: Workload,
        hardware: Hardware,
        targets: Sequence[BehaviorTarget] = (),
    ) -> Iterator[Schedule]:
        del targets
        in_bytes = INPUT_BYTES[workload.dtype]
        base_k = 256 // in_bytes
        if 3 * base_k >= workload.k:
            return
        alignment_m = max(
            16, (512 if workload.trans_a else 256) // in_bytes
        )
        alignment_n = max(16, 512 // in_bytes)
        layouts = (
            # MK33 keeps three M blocks resident and double-buffers B.
            (3, 1, 3, 9, 6, 1, 384, 128),
            # NK33 is the transposed data-reuse policy.
            (1, 3, 3, 6, 9, 0, 128, 384),
            # Upstream SetBasicBlockOf24 variants used when the 3x3 output
            # grid cannot fill the machine or one output axis is small.
            (2, 1, 4, 8, 8, 1, 256, 128),
            (1, 1, 4, 8, 8, 1, 128, 128),
        )
        for (
            step_m,
            step_n,
            step_k,
            depth_a,
            depth_b,
            iterate_order,
            inner_m,
            inner_n,
        ) in layouts:
            single_k = step_k * base_k
            if single_k >= workload.k:
                continue
            geometries = core_partition_variants(
                workload,
                hardware,
                minimum_m=inner_m,
                minimum_n=inner_n,
                alignment_m=alignment_m,
                alignment_n=alignment_n,
                rounds=(1, 2, 3),
            )
            for single_m, single_n, cores in geometries[:12]:
                l2 = l2_policy_variants(
                    workload,
                    hardware,
                    single_m=single_m,
                    single_n=single_n,
                    used_cores=cores,
                )
                for (
                    l2_m_count,
                    l2_n_count,
                    l2_m_block,
                    l2_n_block,
                    l2_order,
                ) in l2[:4]:
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
                        iterateOrder=iterate_order,
                        stepKa=step_k,
                        stepKb=step_k,
                        dbL0A=2,
                        dbL0B=2,
                        dbL0C=2,
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
        core_values = [
            core_limit,
            *sorted(
                {
                1,
                min(2, core_limit),
                min(4, core_limit),
                min(8, core_limit),
                min(16, core_limit),
                }
            ),
        ]
        core_values = list(dict.fromkeys(core_values))
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
