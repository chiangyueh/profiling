"""Schedule and tiling parameter types for the generic solver."""

from __future__ import annotations

from dataclasses import dataclass, field

from .ir import MemorySpace


@dataclass(frozen=True)
class TilingPlan:
    """A complete, serializable schedule candidate.

    ``algorithm`` is an index into ``Operator.algorithms``.  It is not used as
    a cost label; it selects the hardware-stage graph that is being scheduled.
    Reduction parallelism is represented numerically in ``reduction_parts``.
    """

    algorithm: int
    axis_tiles: tuple[tuple[str, int], ...]
    used_cores: int
    task_tiles: tuple[tuple[str, int], ...] = ()
    cache_tiles: tuple[tuple[str, int], ...] = ()
    reduction_parts: tuple[tuple[str, int], ...] = ()
    buffers: tuple[tuple[MemorySpace, int], ...] = ()
    traversal: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        tile_names = [name for name, _ in self.axis_tiles]
        task_tile_names = [name for name, _ in self.task_tiles]
        cache_tile_names = [name for name, _ in self.cache_tiles]
        reduction_names = [name for name, _ in self.reduction_parts]
        buffer_names = [space for space, _ in self.buffers]
        if self.algorithm < 0:
            raise ValueError("algorithm index must be non-negative")
        if self.used_cores <= 0:
            raise ValueError("used_cores must be positive")
        if len(tile_names) != len(set(tile_names)):
            raise ValueError("axis_tiles contains duplicate axes")
        if len(task_tile_names) != len(set(task_tile_names)):
            raise ValueError("task_tiles contains duplicate axes")
        if len(cache_tile_names) != len(set(cache_tile_names)):
            raise ValueError("cache_tiles contains duplicate axes")
        if len(reduction_names) != len(set(reduction_names)):
            raise ValueError("reduction_parts contains duplicate axes")
        if len(buffer_names) != len(set(buffer_names)):
            raise ValueError("buffers contains duplicate spaces")
        if any(value <= 0 for _, value in self.axis_tiles):
            raise ValueError("tile sizes must be positive")
        if any(value <= 0 for _, value in self.task_tiles):
            raise ValueError("task tile sizes must be positive")
        if any(value <= 0 for _, value in self.cache_tiles):
            raise ValueError("cache tile sizes must be positive")
        if any(value <= 0 for _, value in self.reduction_parts):
            raise ValueError("reduction partitions must be positive")
        if any(value not in (1, 2) for _, value in self.buffers):
            raise ValueError("buffer counts must be one or two")

    @property
    def tiles(self) -> dict[str, int]:
        return dict(self.axis_tiles)

    @property
    def reductions(self) -> dict[str, int]:
        return dict(self.reduction_parts)

    @property
    def tasks(self) -> dict[str, int]:
        return dict(self.task_tiles) or self.tiles

    @property
    def caches(self) -> dict[str, int]:
        return dict(self.cache_tiles) or self.tasks

    @property
    def buffer_counts(self) -> dict[MemorySpace, int]:
        return dict(self.buffers)

    def as_dict(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "axis_tiles": dict(self.axis_tiles),
            "task_tiles": self.tasks,
            "cache_tiles": self.caches,
            "used_cores": self.used_cores,
            "reduction_parts": dict(self.reduction_parts),
            "buffers": {space.value: count for space, count in self.buffers},
            "traversal": list(self.traversal),
        }


@dataclass(frozen=True)
class ScheduleSpace:
    """Declarative bounds for legal schedule generation.

    Empty option lists ask the solver to derive hardware breakpoints from the
    axis extent/alignment and available cores. ``coupled_task_axes`` declares
    backend schedules whose inner and per-core task geometry is the same.
    """

    tile_options: tuple[tuple[str, tuple[int, ...]], ...] = ()
    task_tile_options: tuple[tuple[str, tuple[int, ...]], ...] = ()
    cache_tile_options: tuple[tuple[str, tuple[int, ...]], ...] = ()
    core_options: tuple[int, ...] = ()
    reduction_options: tuple[tuple[str, tuple[int, ...]], ...] = ()
    buffer_options: tuple[tuple[int, ...], ...] = ()
    traversal_options: tuple[tuple[str, ...], ...] = ()
    coupled_task_axes: tuple[str, ...] = ()
    max_axis_values: int = 8

    def __post_init__(self) -> None:
        if self.max_axis_values <= 0:
            raise ValueError("max_axis_values must be positive")
        if len(self.coupled_task_axes) != len(set(self.coupled_task_axes)):
            raise ValueError("coupled_task_axes contains duplicate axes")
        if any(value <= 0 for value in self.core_options):
            raise ValueError("core options must be positive")
        if any(
            value <= 0
            for _, values in self.tile_options
            for value in values
        ):
            raise ValueError("tile options must be positive")
        if any(
            value <= 0
            for _, values in self.task_tile_options
            for value in values
        ):
            raise ValueError("task tile options must be positive")
        if any(
            value <= 0
            for _, values in self.cache_tile_options
            for value in values
        ):
            raise ValueError("cache tile options must be positive")
        if any(
            value <= 0
            for _, values in self.reduction_options
            for value in values
        ):
            raise ValueError("reduction options must be positive")
        if any(value not in (1, 2) for values in self.buffer_options for value in values):
            raise ValueError("buffer options must contain only one or two")

    @property
    def tiles(self) -> dict[str, tuple[int, ...]]:
        return dict(self.tile_options)

    @property
    def reductions(self) -> dict[str, tuple[int, ...]]:
        return dict(self.reduction_options)

    @property
    def task_tiles(self) -> dict[str, tuple[int, ...]]:
        return dict(self.task_tile_options)

    @property
    def cache_tiles(self) -> dict[str, tuple[int, ...]]:
        return dict(self.cache_tile_options)


@dataclass(frozen=True)
class SearchPolicy:
    top_k: int = 20
    max_evaluations: int = 100_000
    include_single_buffer: bool = True

    def __post_init__(self) -> None:
        if self.top_k <= 0 or self.max_evaluations <= 0:
            raise ValueError("search limits must be positive")


@dataclass(frozen=True)
class RankedTiling:
    rank: int
    plan: TilingPlan
    cycles: float
    critical_core_cycles: float
    hbm_cycles: float
    l2_cycles: float
    shared_resource_cycles: float
    bottleneck: str
    active_cores: int
    workspace_bytes: int
    gm_read_bytes: float
    gm_write_bytes: float
    l2_bytes: float
    peak_memory_bytes: tuple[tuple[MemorySpace, int], ...]
    resource_cycles: tuple[tuple[str, float], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "cycles": self.cycles,
            "critical_core_cycles": self.critical_core_cycles,
            "hbm_cycles": self.hbm_cycles,
            "l2_cycles": self.l2_cycles,
            "shared_resource_cycles": self.shared_resource_cycles,
            "bottleneck": self.bottleneck,
            "active_cores": self.active_cores,
            "workspace_bytes": self.workspace_bytes,
            "gm_read_bytes": self.gm_read_bytes,
            "gm_write_bytes": self.gm_write_bytes,
            "l2_bytes": self.l2_bytes,
            "peak_memory_bytes": {
                space.value: value for space, value in self.peak_memory_bytes
            },
            "resource_cycles": dict(self.resource_cycles),
            "tiling": self.plan.as_dict(),
        }


@dataclass
class SolveResult:
    ranked: list[RankedTiling]
    evaluated: int
    legal: int
    rejected: int
    exhaustive: bool
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    search_metadata: dict[str, object] = field(default_factory=dict)

    @property
    def best(self) -> RankedTiling:
        if not self.ranked:
            raise RuntimeError("solver produced no legal tiling")
        return self.ranked[0]


@dataclass(frozen=True)
class IdealRegion:
    """Finite schedule neighbourhood around hardware-projected optima.

    ``anchors`` are selected from capacity, issue-width, core-wave, reuse,
    pipeline and reduction projections. ``plans`` contains those anchors and
    one declared schedule transition around each anchor. Neither collection
    is a random sample or a fixed per-kernel quota.
    """

    plans: tuple[TilingPlan, ...]
    anchors: tuple[TilingPlan, ...]
    evaluated: int
    legal: int
    rejected: int
    exhaustive: bool
    rejection_reasons: tuple[tuple[str, int], ...] = ()
    algorithm_anchor_counts: tuple[tuple[int, int], ...] = ()
    algorithm_region_counts: tuple[tuple[int, int], ...] = ()
