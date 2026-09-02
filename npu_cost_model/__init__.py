"""Declarative NPU operator IR, legal tiling solver and cycle simulator."""

from .hardware import Hardware, ResourceRate, ascend_910b3
from .cann_matmul import base_plan_from_cann, plan_from_cann, simulate_cann_base
from .ir import (
    Access,
    AccessMode,
    AccessPattern,
    Algorithm,
    Axis,
    AxisKind,
    MemorySpace,
    Operator,
    Primitive,
    Resource,
    Stage,
    StageScope,
    Tensor,
)
from .schedule import RankedTiling, ScheduleSpace, SearchPolicy, SolveResult, TilingPlan
from .simulator import SimulationResult, simulate
from .solver import generate_plans, plan_space_size, solve

__all__ = [
    "Access",
    "AccessMode",
    "AccessPattern",
    "Algorithm",
    "Axis",
    "AxisKind",
    "Hardware",
    "MemorySpace",
    "Operator",
    "Primitive",
    "RankedTiling",
    "Resource",
    "ResourceRate",
    "ScheduleSpace",
    "SearchPolicy",
    "SimulationResult",
    "SolveResult",
    "Stage",
    "StageScope",
    "Tensor",
    "TilingPlan",
    "ascend_910b3",
    "base_plan_from_cann",
    "plan_from_cann",
    "generate_plans",
    "plan_space_size",
    "simulate",
    "simulate_cann_base",
    "solve",
]
