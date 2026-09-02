"""Translate the public CANN MatMulV3 schedule ABI into the generic IR.

This module is an adapter, not a second cost model.  It contains no latency
constants and does not inspect operator names.  The generic simulator still
receives only an operator graph, a numeric schedule and a hardware profile.
"""

from __future__ import annotations

from collections.abc import Mapping

from .hardware import Hardware
from .ir import MemorySpace
from .operators import matmul
from .schedule import TilingPlan
from .simulator import SimulationResult, ceil_div, simulate


def _integer(values: Mapping[str, int], name: str) -> int:
    value = int(values[name])
    if value <= 0:
        raise ValueError(f"MatMulV3 {name} must be positive")
    return value


def plan_from_cann(
    m: int,
    n: int,
    k: int,
    knowledge: Mapping[str, int],
) -> TilingPlan:
    """Map a CANN 8.1 BASE or deterministic Split-K record to the IR.

    The integer ``tilingEnable`` selects schedule semantics; it is not used as
    a fitted cost label.  Full-load and single-core Split-K paths are rejected
    until their distinct dataflow can be expressed exactly.
    """

    mode = int(knowledge["tilingEnable"])
    if mode not in (0, 3):
        raise ValueError("unsupported MatMulV3 execution graph")
    base_m = _integer(knowledge, "baseM")
    base_n = _integer(knowledge, "baseN")
    base_k = _integer(knowledge, "baseK")
    single_m = _integer(knowledge, "singleCoreM")
    single_n = _integer(knowledge, "singleCoreN")
    single_k = _integer(knowledge, "singleCoreK")
    if mode == 0 and single_k != k:
        raise ValueError("BASE singleCoreK must cover the complete reduction")
    reduction_parts = ceil_div(k, single_k) if mode == 3 else 1

    l2_m = _integer(knowledge, "l2MTileBlock") * single_m
    l2_n = _integer(knowledge, "l2NTileBlock") * single_n
    l2_order = int(knowledge["l2IterateOrder"])
    cube_order = int(knowledge["iterateOrder"])
    # tileL2cacheTiling.calOrder drives the BASE block scheduler.  Value zero
    # selects its default staggered schedule; it does not delegate L2 order to
    # TCubeTiling.iterateOrder.  Falling back to the Cube field made identical
    # L2 schedules receive different cache traffic in the simulator.
    traversal = ("n", "m") if l2_order == 2 else ("m", "n")
    a_packets = _integer(knowledge, "depthA1") // (
        _integer(knowledge, "stepM") * _integer(knowledge, "stepKa")
    )
    b_packets = _integer(knowledge, "depthB1") // (
        _integer(knowledge, "stepN") * _integer(knowledge, "stepKb")
    )
    l1_buffers = 2 if min(a_packets, b_packets) >= 2 else 1
    return TilingPlan(
        algorithm=0,
        axis_tiles=(("m", base_m), ("n", base_n), ("k", base_k)),
        task_tiles=(("m", single_m), ("n", single_n), ("k", single_k)),
        cache_tiles=(
            ("m", max(min(single_m, m), min(l2_m, m))),
            ("n", max(min(single_n, n), min(l2_n, n))),
            ("k", k),
        ),
        used_cores=_integer(knowledge, "usedCoreNum"),
        reduction_parts=(("k", reduction_parts),),
        buffers=(
            (MemorySpace.L1, l1_buffers),
            (MemorySpace.L0A, _integer(knowledge, "dbL0A")),
            (MemorySpace.L0B, _integer(knowledge, "dbL0B")),
            (MemorySpace.L0C, _integer(knowledge, "dbL0C")),
        ),
        traversal=traversal,
    )


def base_plan_from_cann(
    m: int,
    n: int,
    k: int,
    knowledge: Mapping[str, int],
) -> TilingPlan:
    """Compatibility name retained for callers that already pass BASE."""

    plan = plan_from_cann(m, n, k, knowledge)
    if int(knowledge["tilingEnable"]) != 0:
        raise ValueError("base_plan_from_cann requires MatMulV3 BASE")
    return plan


def simulate_cann_base(
    m: int,
    n: int,
    k: int,
    dtype: str,
    trans_a: bool,
    trans_b: bool,
    knowledge: Mapping[str, int],
    hardware: Hardware,
) -> SimulationResult:
    plan = base_plan_from_cann(m, n, k, knowledge)
    return simulate(
        matmul(m, n, k, dtype, trans_a=trans_a, trans_b=trans_b),
        plan,
        hardware,
    )
