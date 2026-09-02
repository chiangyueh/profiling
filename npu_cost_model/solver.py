"""Legal schedule generation and parameter-only cost ranking."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from heapq import heappush, heapreplace
from itertools import product
from math import floor, gcd, prod

from .hardware import Hardware
from .ir import Axis, MemorySpace, Operator
from .schedule import (
    RankedTiling,
    ScheduleSpace,
    SearchPolicy,
    SolveResult,
    TilingPlan,
)
from .simulator import align_up, ceil_div, simulate


def _spread(values: list[int], limit: int) -> tuple[int, ...]:
    values = sorted(set(values))
    if len(values) <= limit:
        return tuple(values)
    if limit == 1:
        return (values[-1],)
    selected = {
        values[round(index * (len(values) - 1) / (limit - 1))]
        for index in range(limit)
    }
    return tuple(sorted(selected))


def automatic_tile_values(axis: Axis, core_count: int, limit: int) -> tuple[int, ...]:
    """Derive alignment and parallelism breakpoints without latency data."""

    if axis.tile_values:
        values = [
            value
            for value in axis.tile_values
            if value % axis.alignment == 0
        ]
        return _spread(values, limit)

    alignment = axis.alignment
    values = {alignment, align_up(axis.extent, alignment)}
    value = alignment
    while value < axis.extent:
        values.add(value)
        value *= 2
    for divisor in (2, 3, 4, 5, 8, 16, core_count):
        values.add(align_up(ceil_div(axis.extent, divisor), alignment))
    return _spread([value for value in values if value > 0], limit)


def automatic_task_tile_values(
    axis: Axis,
    inner_tile: int,
    core_count: int,
    limit: int,
) -> tuple[int, ...]:
    """Derive core-task breakpoints independently from local-memory tiles."""

    alignment = axis.alignment
    values = {inner_tile, align_up(axis.extent, alignment)}
    value = alignment
    while value < axis.extent:
        values.add(value)
        value *= 2
    for divisor in (2, 3, 4, 5, 8, 10, 16, core_count):
        values.add(align_up(ceil_div(axis.extent, divisor), alignment))
    for multiplier in (2, 3, 4, 8):
        values.add(align_up(inner_tile * multiplier, alignment))
    minimum = min(inner_tile, axis.extent)
    maximum = max(inner_tile, align_up(axis.extent, alignment))
    return _spread(
        [value for value in values if minimum <= value <= maximum], limit
    )


def automatic_cache_tile_values(
    axis: Axis,
    task_tile: int,
    core_count: int,
    limit: int,
) -> tuple[int, ...]:
    """Derive L2 scheduling-group breakpoints from task geometry."""

    alignment = axis.alignment
    values = {task_tile, align_up(axis.extent, alignment)}
    for divisor in (2, 3, 4, 5, 8, 10, 16, core_count):
        values.add(align_up(ceil_div(axis.extent, divisor), alignment))
    for multiplier in (2, 3, 4, 8):
        values.add(align_up(task_tile * multiplier, alignment))
    minimum = min(task_tile, axis.extent)
    maximum = max(task_tile, align_up(axis.extent, alignment))
    return _spread(
        [value for value in values if minimum <= value <= maximum], limit
    )


def _algorithm_tile_options(
    operator: Operator,
    algorithm_index: int,
    hardware: Hardware,
    space: ScheduleSpace,
) -> dict[str, tuple[int, ...]]:
    algorithm = operator.algorithms[algorithm_index]
    core_count = hardware.core_count(algorithm.core_resource)
    explicit = space.tiles
    result: dict[str, tuple[int, ...]] = {}
    for axis in operator.axes:
        values = explicit.get(axis.name)
        if values is None:
            values = automatic_tile_values(axis, core_count, space.max_axis_values)
        legal = tuple(sorted(set(
            value for value in values
            if value > 0 and value % axis.alignment == 0
        )))
        if not legal:
            raise ValueError(f"axis {axis.name} has no aligned tile option")
        result[axis.name] = legal
    return result


def _algorithm_tile_levels(
    operator: Operator,
    algorithm_index: int,
    hardware: Hardware,
    space: ScheduleSpace,
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    algorithm = operator.algorithms[algorithm_index]
    core_count = hardware.core_count(algorithm.core_resource)
    inner_options = _algorithm_tile_options(
        operator, algorithm_index, hardware, space
    )
    explicit_tasks = space.task_tiles
    explicit_caches = space.cache_tiles
    all_levels: list[tuple[tuple[int, int, int], ...]] = []
    for axis in operator.axes:
        explicit = explicit_tasks.get(axis.name)
        explicit_cache = explicit_caches.get(axis.name)
        levels: list[tuple[int, int, int]] = []
        for inner in inner_options[axis.name]:
            if explicit is not None:
                tasks = tuple(sorted(set(
                    value
                    for value in explicit
                    if value % axis.alignment == 0
                    and value >= min(inner, axis.extent)
                )))
            elif axis.independent_task_tiling and axis.name in algorithm.output_axes:
                tasks = automatic_task_tile_values(
                    axis, inner, core_count, space.max_axis_values
                )
            else:
                tasks = (inner,)
            for task in tasks:
                if explicit_cache is not None:
                    caches = tuple(sorted(set(
                        value
                        for value in explicit_cache
                        if value >= min(task, axis.extent)
                    )))
                elif (
                    axis.independent_cache_tiling
                    and axis.name in algorithm.output_axes
                ):
                    caches = automatic_cache_tile_values(
                        axis, task, core_count, space.max_axis_values
                    )
                else:
                    caches = (task,)
                levels.extend((inner, task, cache) for cache in caches)
        if not levels:
            raise ValueError(
                f"axis {axis.name} has no legal inner/task/cache tile combination"
            )
        all_levels.append(tuple(dict.fromkeys(levels)))
    return tuple(all_levels)


def _core_options(operator: Operator, algorithm_index: int, hardware: Hardware,
                  space: ScheduleSpace) -> tuple[int, ...]:
    maximum = hardware.core_count(operator.algorithms[algorithm_index].core_resource)
    if space.core_options:
        return tuple(sorted(set(value for value in space.core_options if value <= maximum)))
    values = {1, maximum}
    power = 1
    while power <= maximum:
        values.add(power)
        power *= 2
    for divisor in (2, 3, 4):
        values.add(max(1, maximum // divisor))
        values.add(max(1, ceil_div(maximum, divisor)))
    return tuple(sorted(value for value in values if value <= maximum))


def _reduction_options(
    operator: Operator,
    algorithm_index: int,
    hardware: Hardware,
    space: ScheduleSpace,
) -> dict[str, tuple[int, ...]]:
    algorithm = operator.algorithms[algorithm_index]
    explicit = space.reductions
    maximum = hardware.core_count(algorithm.core_resource)
    result: dict[str, tuple[int, ...]] = {}
    for axis_name in algorithm.reduction_axes:
        if not algorithm.parallel_reduction:
            result[axis_name] = (1,)
            continue
        values = explicit.get(axis_name)
        if values is None:
            values = _core_options(operator, algorithm_index, hardware, space)
        result[axis_name] = tuple(sorted(set(
            value for value in values if 1 <= value <= maximum
        )))
    return result


def _buffer_profiles(algorithm_spaces: tuple[MemorySpace, ...],
                     space: ScheduleSpace,
                     policy: SearchPolicy) -> tuple[tuple[tuple[MemorySpace, int], ...], ...]:
    if not algorithm_spaces:
        return ((),)
    configured = space.buffer_options or tuple(
        product((1, 2), repeat=len(algorithm_spaces))
    )
    profiles: list[tuple[tuple[MemorySpace, int], ...]] = []
    for values in configured:
        if len(values) == 1:
            values = values * len(algorithm_spaces)
        if len(values) != len(algorithm_spaces):
            raise ValueError("buffer option width must match algorithm buffered_spaces")
        if not policy.include_single_buffer and all(value == 1 for value in values):
            continue
        profiles.append(tuple(zip(algorithm_spaces, values)))
    return tuple(profiles or [tuple((item, 1) for item in algorithm_spaces)])


@dataclass(frozen=True)
class _PlanSpace:
    algorithm_index: int
    axis_levels: tuple[tuple[tuple[int, int, int], ...], ...]
    reduction_values: tuple[tuple[int, ...], ...]
    core_values: tuple[int, ...]
    buffer_values: tuple[tuple[tuple[MemorySpace, int], ...], ...]
    traversal_values: tuple[tuple[str, ...], ...]

    @property
    def dimensions(self) -> tuple[tuple[object, ...], ...]:
        return (
            *self.axis_levels,
            *self.reduction_values,
            self.core_values,
            self.buffer_values,
            self.traversal_values,
        )

    @property
    def size(self) -> int:
        return prod(len(values) for values in self.dimensions)


def _plan_spaces(
    operator: Operator,
    hardware: Hardware,
    space: ScheduleSpace,
    policy: SearchPolicy,
) -> tuple[_PlanSpace, ...]:
    result: list[_PlanSpace] = []
    for algorithm_index, algorithm in enumerate(operator.algorithms):
        reductions = _reduction_options(
            operator, algorithm_index, hardware, space
        )
        buffers = _buffer_profiles(algorithm.buffered_spaces, space, policy)
        traversals = space.traversal_options or (
            tuple(algorithm.output_axes),
            tuple(reversed(algorithm.output_axes)),
        )
        traversals = tuple(dict.fromkeys(traversals))
        if any(
            set(value) != set(algorithm.output_axes)
            or len(value) != len(algorithm.output_axes)
            for value in traversals
        ):
            raise ValueError("every traversal option must permute the output axes")
        plan_space = _PlanSpace(
            algorithm_index=algorithm_index,
            axis_levels=_algorithm_tile_levels(
                operator, algorithm_index, hardware, space
            ),
            reduction_values=tuple(
                reductions[axis] for axis in algorithm.reduction_axes
            ),
            core_values=_core_options(
                operator, algorithm_index, hardware, space
            ),
            buffer_values=buffers,
            traversal_values=traversals,
        )
        if plan_space.size:
            result.append(plan_space)
    return tuple(result)


def _decode_choice(plan_space: _PlanSpace, flat_index: int) -> tuple[object, ...]:
    dimensions = plan_space.dimensions
    values: list[object] = [None] * len(dimensions)
    for index in range(len(dimensions) - 1, -1, -1):
        dimension = dimensions[index]
        flat_index, offset = divmod(flat_index, len(dimension))
        values[index] = dimension[offset]
    return tuple(values)


def _allocations(counts: tuple[int, ...], budget: int) -> tuple[int, ...]:
    total = sum(counts)
    if total <= budget:
        return counts
    if not counts:
        return ()
    if budget < len(counts):
        return tuple(1 if index < budget else 0 for index in range(len(counts)))

    targets = [budget * count / total for count in counts]
    result = [min(count, max(1, floor(target))) for count, target in zip(counts, targets)]
    while sum(result) < budget:
        candidates = [
            index for index, count in enumerate(counts) if result[index] < count
        ]
        selected = max(candidates, key=lambda index: targets[index] - result[index])
        result[selected] += 1
    while sum(result) > budget:
        candidates = [index for index, value in enumerate(result) if value > 1]
        selected = min(candidates, key=lambda index: targets[index] - result[index])
        result[selected] -= 1
    return tuple(result)


def _sample_indices(size: int, count: int):
    """Traverse a capped Cartesian space without prefix or radix bias."""

    if count >= size:
        yield from range(size)
        return
    # The golden-ratio rotation is deterministic and avoids repeatedly
    # landing on the same low-radix choices (buffer/traversal) or only the
    # first high-radix choices (large tiles).  It is a search ordering, not a
    # hardware or latency coefficient.
    step = max(1, int(size * 0.6180339887498949))
    while gcd(step, size) != 1:
        step += 1
    for sample in range(count):
        yield sample * step % size


def generate_plans(
    operator: Operator,
    hardware: Hardware,
    space: ScheduleSpace | None = None,
    policy: SearchPolicy | None = None,
):
    """Yield deterministic numeric tiling candidates.

    No candidate depends on historical measurements, an installed callback or
    an operator-name dispatch.  The algorithm graph and axis declarations are
    part of the supplied Python IR.
    """

    space = space or ScheduleSpace()
    policy = policy or SearchPolicy()
    plan_spaces = _plan_spaces(operator, hardware, space, policy)
    allocations = _allocations(
        tuple(item.size for item in plan_spaces), policy.max_evaluations
    )
    axis_names = tuple(axis.name for axis in operator.axes)
    for plan_space, allocation in zip(plan_spaces, allocations):
        algorithm = operator.algorithms[plan_space.algorithm_index]
        for flat_index in _sample_indices(plan_space.size, allocation):
            choice = _decode_choice(plan_space, flat_index)
            axis_count = len(axis_names)
            reduction_count = len(algorithm.reduction_axes)
            levels = choice[:axis_count]
            reduction_choice = choice[
                axis_count:axis_count + reduction_count
            ]
            used_cores = choice[axis_count + reduction_count]
            buffer_items = choice[axis_count + reduction_count + 1]
            traversal = choice[axis_count + reduction_count + 2]
            yield TilingPlan(
                algorithm=plan_space.algorithm_index,
                axis_tiles=tuple(
                    (name, level[0]) for name, level in zip(axis_names, levels)
                ),
                task_tiles=tuple(
                    (name, level[1]) for name, level in zip(axis_names, levels)
                ),
                cache_tiles=tuple(
                    (name, level[2]) for name, level in zip(axis_names, levels)
                ),
                used_cores=used_cores,
                reduction_parts=tuple(
                    zip(algorithm.reduction_axes, reduction_choice)
                ),
                buffers=buffer_items,
                traversal=traversal,
            )


def plan_space_size(
    operator: Operator,
    hardware: Hardware,
    space: ScheduleSpace | None = None,
    policy: SearchPolicy | None = None,
) -> int:
    """Return the deterministic search-space cardinality before its cap."""

    space = space or ScheduleSpace()
    policy = policy or SearchPolicy()
    return sum(
        item.size for item in _plan_spaces(operator, hardware, space, policy)
    )


def solve(
    operator: Operator,
    hardware: Hardware,
    space: ScheduleSpace | None = None,
    policy: SearchPolicy | None = None,
) -> SolveResult:
    """Generate legal tilings, simulate cycles and return the fastest plans."""

    space = space or ScheduleSpace()
    policy = policy or SearchPolicy()
    evaluated = legal = 0
    reasons: Counter[str] = Counter()
    # Keep only the requested frontier.  A full search may contain millions
    # of candidates; ranking must not retain every SimulationResult in RAM.
    frontier: list[tuple[float, int, TilingPlan, object]] = []
    for plan in generate_plans(operator, hardware, space, policy):
        evaluated += 1
        result = simulate(operator, plan, hardware)
        if not result.valid:
            reasons[result.error] += 1
            continue
        legal += 1
        entry = (-result.total_cycles, -evaluated, plan, result)
        if len(frontier) < policy.top_k:
            heappush(frontier, entry)
        elif result.total_cycles < -frontier[0][0]:
            heapreplace(frontier, entry)

    scored = [(-entry[0], entry[2], entry[3]) for entry in frontier]
    scored.sort(key=lambda item: (
        item[0],
        item[1].used_cores,
        item[1].axis_tiles,
        item[1].task_tiles,
        item[1].cache_tiles,
        item[1].reduction_parts,
        tuple((space.value, value) for space, value in item[1].buffers),
        item[1].traversal,
    ))
    ranked: list[RankedTiling] = []
    for rank, (_, plan, result) in enumerate(scored[:policy.top_k], 1):
        ranked.append(
            RankedTiling(
                rank=rank,
                plan=plan,
                cycles=result.total_cycles,
                critical_core_cycles=result.critical_core_cycles,
                hbm_cycles=result.hbm_cycles,
                l2_cycles=result.l2_cycles,
                shared_resource_cycles=result.shared_resource_cycles,
                bottleneck=result.bottleneck,
                active_cores=result.active_cores,
                workspace_bytes=result.workspace_bytes,
                gm_read_bytes=result.gm_read_bytes,
                gm_write_bytes=result.gm_write_bytes,
                l2_bytes=result.l2_bytes,
                peak_memory_bytes=result.peak_memory_bytes,
                resource_cycles=tuple(
                    (resource.value, cycles)
                    for resource, cycles in result.resource_cycles
                ),
            )
        )
    exhaustive = plan_space_size(operator, hardware, space, policy) <= policy.max_evaluations
    return SolveResult(
        ranked=ranked,
        evaluated=evaluated,
        legal=legal,
        rejected=evaluated - legal,
        exhaustive=exhaustive,
        rejection_reasons=dict(reasons),
    )
