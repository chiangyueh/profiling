"""Legal schedule generation and parameter-only cost ranking."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from heapq import heappush, heapreplace
from itertools import product
from math import floor, gcd, prod

from .hardware import Hardware
from .ir import AccessMode, Axis, MemorySpace, Operator
from .schedule import (
    RankedTiling,
    IdealRegion,
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
    preserve_declared: bool = False,
) -> dict[str, tuple[int, ...]]:
    algorithm = operator.algorithms[algorithm_index]
    core_count = hardware.core_count(algorithm.core_resource)
    explicit = space.tiles
    result: dict[str, tuple[int, ...]] = {}
    for axis in operator.axes:
        values = explicit.get(axis.name)
        if values is None:
            if axis.tile_values:
                # Frontends use tile_values to declare their complete legal
                # hardware transition lattice.  Local ideal-region search
                # walks adjacent values and therefore needs no arbitrary
                # subsampling of that lattice.
                values = (
                    axis.tile_values
                    if preserve_declared
                    else _spread(list(axis.tile_values), space.max_axis_values)
                )
            else:
                values = automatic_tile_values(
                    axis, core_count, space.max_axis_values
                )
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
    preserve_declared: bool = False,
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    algorithm = operator.algorithms[algorithm_index]
    core_count = hardware.core_count(algorithm.core_resource)
    inner_options = _algorithm_tile_options(
        operator, algorithm_index, hardware, space, preserve_declared
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


def _transition_traversals(
    algorithm,
    space: ScheduleSpace,
) -> tuple[tuple[str, ...], ...]:
    """Return traversal changes implied by the output iteration domain."""

    if space.traversal_options:
        values = space.traversal_options
    else:
        axes = tuple(algorithm.output_axes)
        if not axes:
            values = ((),)
        else:
            # Put axes with the largest declared read-reuse opportunity on
            # the inner side, and retain both declared memory directions.
            # This is derived from Access.axes rather than operator identity.
            reuse = {axis: 0 for axis in axes}
            for stage in algorithm.stages:
                for access in stage.accesses:
                    if access.mode != AccessMode.READ:
                        continue
                    for axis in axes:
                        if axis not in access.axes:
                            reuse[axis] += 1
            reuse_order = tuple(sorted(
                axes, key=lambda axis: (reuse[axis], axes.index(axis))
            ))
            values = (axes, tuple(reversed(axes)), reuse_order)
    result = tuple(dict.fromkeys(values))
    if any(
        set(value) != set(algorithm.output_axes)
        or len(value) != len(algorithm.output_axes)
        for value in result
    ):
        raise ValueError("every traversal option must permute the output axes")
    return result


def _transition_buffers(
    algorithm,
    space: ScheduleSpace,
    policy: SearchPolicy,
) -> tuple[tuple[tuple[MemorySpace, int], ...], ...]:
    """Derive buffer profiles from producer/consumer boundaries."""

    if space.buffer_options:
        return _buffer_profiles(algorithm.buffered_spaces, space, policy)
    spaces = tuple(algorithm.buffered_spaces)
    if not spaces:
        return ((),)
    single = {item: 1 for item in spaces}
    profiles: list[tuple[tuple[MemorySpace, int], ...]] = []
    if policy.include_single_buffer:
        profiles.append(tuple(single.items()))
    for boundary in algorithm.pipeline_boundaries or tuple((item,) for item in spaces):
        values = dict(single)
        for item in boundary:
            values[item] = 2
        profiles.append(tuple(values.items()))
    profiles.append(tuple((item, 2) for item in spaces))
    return tuple(dict.fromkeys(profiles))


def _level_distance(
    candidate: tuple[int, int, int],
    target: tuple[int, int, int],
) -> float:
    return sum(
        abs(candidate_value - target_value) / max(1, target_value)
        for candidate_value, target_value in zip(candidate, target)
    )


def _nearest_level(
    levels: tuple[tuple[int, int, int], ...],
    target: tuple[int, int, int],
) -> tuple[int, int, int]:
    return min(levels, key=lambda value: (_level_distance(value, target), value))


def _axis_level_neighbours(
    levels: tuple[tuple[int, int, int], ...],
    current: tuple[int, int, int],
) -> tuple[tuple[int, int, int], ...]:
    """Move by one legal inner/task/cache transition on one level."""

    result: list[tuple[int, int, int]] = []
    for component in range(3):
        values = sorted(set(level[component] for level in levels))
        position = values.index(current[component])
        for next_position in (position - 1, position + 1):
            if not 0 <= next_position < len(values):
                continue
            target = list(current)
            target[component] = values[next_position]
            compatible = tuple(
                level for level in levels if level[component] == target[component]
            )
            if compatible:
                result.append(_nearest_level(compatible, tuple(target)))
    return tuple(dict.fromkeys(result))


def _output_task_count(operator: Operator, algorithm, plan: TilingPlan) -> int:
    return prod(
        ceil_div(operator.axis(axis).extent, plan.tasks[axis])
        for axis in algorithm.output_axes
    ) if algorithm.output_axes else 1


def _total_task_count(operator: Operator, algorithm, plan: TilingPlan) -> int:
    return _output_task_count(operator, algorithm, plan) * prod(
        plan.reductions.get(axis, 1) for axis in algorithm.reduction_axes
    )


def _useful_core_values(
    total_tasks: int,
    allowed: tuple[int, ...],
) -> tuple[int, ...]:
    """Core counts at the two wave transitions surrounding saturation."""

    legal = tuple(value for value in allowed if value <= total_tasks)
    if not legal:
        return (min(allowed),)
    best = legal[-1]
    waves = ceil_div(total_tasks, best)
    transition = ceil_div(total_tasks, waves)
    position = min(
        range(len(legal)), key=lambda index: abs(legal[index] - transition)
    )
    indexes = {len(legal) - 1, position}
    if position:
        indexes.add(position - 1)
    if position + 1 < len(legal):
        indexes.add(position + 1)
    return tuple(legal[index] for index in sorted(indexes))


def _reduction_transition_values(
    operator: Operator,
    algorithm_index: int,
    plan: TilingPlan,
    hardware: Hardware,
    space: ScheduleSpace,
    axis_name: str,
) -> tuple[int, ...]:
    """Reduction partitions around the core-fill transition."""

    algorithm = operator.algorithms[algorithm_index]
    if not algorithm.parallel_reduction:
        return (1,)
    chunks = ceil_div(operator.axis(axis_name).extent, plan.tiles[axis_name])
    maximum = min(hardware.core_count(algorithm.core_resource), chunks)
    explicit = space.reductions.get(axis_name)
    if explicit is not None:
        allowed = tuple(sorted(set(value for value in explicit if value <= maximum)))
    else:
        allowed = tuple(sorted(set(
            value for value in (*_core_options(operator, algorithm_index, hardware, space), maximum)
            if value <= maximum
        )))
    if not allowed:
        return (1,)
    output_tasks = _output_task_count(operator, algorithm, plan)
    required = min(maximum, max(1, ceil_div(
        hardware.core_count(algorithm.core_resource), output_tasks
    )))
    position = min(
        range(len(allowed)), key=lambda index: abs(allowed[index] - required)
    )
    indexes = {0, position, len(allowed) - 1}
    if position:
        indexes.add(position - 1)
    if position + 1 < len(allowed):
        indexes.add(position + 1)
    return tuple(allowed[index] for index in sorted(indexes))


def _reduction_fill_profile(
    operator: Operator,
    algorithm_index: int,
    plan: TilingPlan,
    hardware: Hardware,
    space: ScheduleSpace,
) -> tuple[tuple[str, int], ...]:
    """Distribute only the missing core parallelism over reduction axes."""

    algorithm = operator.algorithms[algorithm_index]
    remaining = max(1, ceil_div(
        hardware.core_count(algorithm.core_resource),
        _output_task_count(operator, algorithm, plan),
    ))
    profile: list[tuple[str, int]] = []
    current = plan
    for axis_name in algorithm.reduction_axes:
        options = _reduction_transition_values(
            operator, algorithm_index, current, hardware, space, axis_name
        )
        value = min(options, key=lambda item: (abs(item - remaining), item))
        profile.append((axis_name, value))
        remaining = max(1, ceil_div(remaining, value))
        current = _replace_plan(current, reduction_parts=tuple(profile))
    return tuple(profile)


def _reduction_chunk_profile(
    operator: Operator,
    algorithm_index: int,
    plan: TilingPlan,
    hardware: Hardware,
    space: ScheduleSpace,
) -> tuple[tuple[str, int], ...]:
    """Expose one full core wave of legal reduction chunks.

    This second hardware anchor is required even when output tiles already
    fill the cores: shortening a very deep reduction changes the live cache
    footprint and pipeline length.  The partition product is bounded by the
    physical core wave rather than by an operator-specific Split-K count.
    """

    algorithm = operator.algorithms[algorithm_index]
    remaining = hardware.core_count(algorithm.core_resource)
    profile: list[tuple[str, int]] = []
    current = plan
    for axis_name in algorithm.reduction_axes:
        options = _reduction_transition_values(
            operator, algorithm_index, current, hardware, space, axis_name
        )
        eligible = tuple(value for value in options if value <= remaining)
        value = eligible[-1] if eligible else 1
        profile.append((axis_name, value))
        remaining = max(1, remaining // value)
        current = _replace_plan(current, reduction_parts=tuple(profile))
    return tuple(profile)


def _plan_from_levels(
    algorithm_index: int,
    axis_names: tuple[str, ...],
    levels: tuple[tuple[int, int, int], ...],
    reductions: tuple[tuple[str, int], ...],
    used_cores: int,
    buffers: tuple[tuple[MemorySpace, int], ...],
    traversal: tuple[str, ...],
) -> TilingPlan:
    return TilingPlan(
        algorithm=algorithm_index,
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
        reduction_parts=reductions,
        buffers=buffers,
        traversal=traversal,
    )


def _replace_plan(plan: TilingPlan, **changes) -> TilingPlan:
    values = {
        "algorithm": plan.algorithm,
        "axis_tiles": plan.axis_tiles,
        "task_tiles": plan.task_tiles,
        "cache_tiles": plan.cache_tiles,
        "used_cores": plan.used_cores,
        "reduction_parts": plan.reduction_parts,
        "buffers": plan.buffers,
        "traversal": plan.traversal,
    }
    values.update(changes)
    return TilingPlan(**values)


def _ideal_neighbours(
    operator: Operator,
    hardware: Hardware,
    space: ScheduleSpace,
    policy: SearchPolicy,
    levels_by_axis: tuple[tuple[tuple[int, int, int], ...], ...],
    buffers: tuple[tuple[tuple[MemorySpace, int], ...], ...],
    traversals: tuple[tuple[str, ...], ...],
    plan: TilingPlan,
    *,
    vary_reductions: bool = True,
) -> tuple[TilingPlan, ...]:
    algorithm = operator.algorithms[plan.algorithm]
    axis_names = tuple(axis.name for axis in operator.axes)
    current_levels = tuple(
        (plan.tiles[name], plan.tasks[name], plan.caches[name])
        for name in axis_names
    )
    result: list[TilingPlan] = []
    for axis_index, level_options in enumerate(levels_by_axis):
        for replacement in _axis_level_neighbours(
            level_options, current_levels[axis_index]
        ):
            levels = list(current_levels)
            levels[axis_index] = replacement
            candidate = _plan_from_levels(
                plan.algorithm, axis_names, tuple(levels), plan.reduction_parts,
                plan.used_cores, plan.buffers, plan.traversal,
            )
            result.append(candidate)

    if vary_reductions:
        reductions = dict(plan.reduction_parts)
        for axis_name in algorithm.reduction_axes:
            values = _reduction_transition_values(
                operator, plan.algorithm, plan, hardware, space, axis_name
            )
            current = reductions.get(axis_name, 1)
            # A changed inner reduction tile can remove the old partition
            # count from the current transition set. Include its nearest
            # transition.
            position = min(
                range(len(values)), key=lambda index: abs(values[index] - current)
            )
            for next_position in (position - 1, position, position + 1):
                if not 0 <= next_position < len(values):
                    continue
                value = values[next_position]
                if value == current:
                    continue
                changed = dict(reductions)
                changed[axis_name] = value
                result.append(_replace_plan(
                    plan,
                    reduction_parts=tuple(
                        (axis, changed.get(axis, 1))
                        for axis in algorithm.reduction_axes
                    ),
                ))

    core_allowed = _core_options(operator, plan.algorithm, hardware, space)
    for value in _useful_core_values(
        _total_task_count(operator, algorithm, plan), core_allowed
    ):
        if value != plan.used_cores:
            result.append(_replace_plan(plan, used_cores=value))
    result.extend(
        _replace_plan(plan, buffers=value)
        for value in buffers if value != plan.buffers
    )
    result.extend(
        _replace_plan(plan, traversal=value)
        for value in traversals if value != plan.traversal
    )
    return tuple(dict.fromkeys(result))


def derive_ideal_region(
    operator: Operator,
    hardware: Hardware,
    space: ScheduleSpace | None = None,
    policy: SearchPolicy | None = None,
) -> IdealRegion:
    """Find hardware-model optima, then expand one legal transition around them.

    The same implementation consumes every algorithm graph declared by the
    operator IR.  It never branches on operator/algorithm names and never
    reads measured latency, RuntimeKb, callbacks or calibration tables.
    """

    space = space or ScheduleSpace()
    policy = policy or SearchPolicy()
    axis_names = tuple(axis.name for axis in operator.axes)
    cache: dict[TilingPlan, object] = {}
    reasons: Counter[str] = Counter()
    stopped = False

    def evaluate(plan: TilingPlan):
        nonlocal stopped
        if plan in cache:
            return cache[plan]
        if len(cache) >= policy.max_evaluations:
            stopped = True
            return None
        value = simulate(operator, plan, hardware)
        cache[plan] = value
        if not value.valid:
            reasons[value.error] += 1
        return value

    anchors: list[TilingPlan] = []
    algorithm_anchor_counts: Counter[int] = Counter()
    algorithm_region_counts: Counter[int] = Counter()
    algorithm_contexts: dict[int, tuple] = {}

    for algorithm_index, algorithm in enumerate(operator.algorithms):
        levels_by_axis = _algorithm_tile_levels(
            operator, algorithm_index, hardware, space, True
        )
        buffers = _transition_buffers(algorithm, space, policy)
        traversals = _transition_traversals(algorithm, space)
        core_allowed = _core_options(operator, algorithm_index, hardware, space)
        algorithm_contexts[algorithm_index] = (
            levels_by_axis, buffers, traversals
        )

        output_rank = max(1, len(algorithm.output_axes))
        axis_seeds: list[tuple[tuple[int, int, int], ...]] = []
        low = tuple(min(levels) for levels in levels_by_axis)
        high = tuple(max(levels) for levels in levels_by_axis)
        parallel: list[tuple[int, int, int]] = []
        for axis, levels in zip(operator.axes, levels_by_axis):
            if axis.name in algorithm.output_axes:
                chunks = max(1, round(
                    hardware.core_count(algorithm.core_resource)
                    ** (1.0 / output_rank)
                ))
                task_target = align_up(ceil_div(axis.extent, chunks), axis.alignment)
                inner_target = min(task_target, max(level[0] for level in levels))
            else:
                inner_target = max(level[0] for level in levels)
                task_target = inner_target
            parallel.append(_nearest_level(
                levels, (inner_target, task_target, task_target)
            ))
        axis_seeds.extend((low, tuple(parallel), high))

        for levels in tuple(dict.fromkeys(axis_seeds)):
            prototype = _plan_from_levels(
                algorithm_index, axis_names, levels,
                tuple((axis, 1) for axis in algorithm.reduction_axes),
                1, buffers[0], traversals[0],
            )
            reduction_profiles = [prototype.reduction_parts]
            if algorithm.parallel_reduction:
                reduction_profiles.append(_reduction_fill_profile(
                    operator, algorithm_index, prototype, hardware, space
                ))
                reduction_profiles.append(_reduction_chunk_profile(
                    operator, algorithm_index, prototype, hardware, space
                ))
            for reduction_profile in tuple(dict.fromkeys(reduction_profiles)):
                # Buffer and traversal alternatives are coordinates in every
                # neighbourhood, so multiplying all of them into the starting
                # set adds repeated walks without exposing a new basin.
                candidate = _replace_plan(
                    prototype,
                    reduction_parts=reduction_profile,
                    buffers=buffers[0],
                    traversal=traversals[0],
                )
                candidate = _replace_plan(
                    candidate,
                    used_cores=_useful_core_values(
                        _total_task_count(operator, algorithm, candidate),
                        core_allowed,
                    )[-1],
                )
                current = candidate
                current_result = evaluate(current)
                while current_result is not None:
                    neighbourhood = _ideal_neighbours(
                        operator, hardware, space, policy,
                        levels_by_axis, buffers, traversals, current,
                        vary_reductions=False,
                    )
                    scored = []
                    for neighbour in neighbourhood:
                        result = evaluate(neighbour)
                        if result is not None and result.valid:
                            scored.append((result.total_cycles, neighbour, result))
                    if not scored:
                        break
                    scored.sort(key=lambda item: (item[0], item[1].as_dict().__repr__()))
                    best_cycles, best_plan, best_result = scored[0]
                    if current_result.valid and best_cycles >= current_result.total_cycles:
                        break
                    current, current_result = best_plan, best_result
                if current_result is not None and current_result.valid:
                    anchors.append(current)
                if stopped:
                    break
            if stopped:
                break

        unique_algorithm_anchors = tuple(dict.fromkeys(
            plan for plan in anchors if plan.algorithm == algorithm_index
        ))
        algorithm_anchor_counts[algorithm_index] = len(unique_algorithm_anchors)
        if stopped:
            break

    anchors = list(dict.fromkeys(anchors))
    region: list[TilingPlan] = []
    for anchor in anchors:
        levels_by_axis, buffers, traversals = algorithm_contexts[anchor.algorithm]
        for plan in (anchor, *_ideal_neighbours(
            operator, hardware, space, policy,
            levels_by_axis, buffers, traversals, anchor,
        )):
            result = evaluate(plan)
            if result is not None and result.valid:
                region.append(plan)
    region = list(dict.fromkeys(region))
    for plan in region:
        algorithm_region_counts[plan.algorithm] += 1
    legal = sum(value.valid for value in cache.values())
    return IdealRegion(
        plans=tuple(region),
        anchors=tuple(anchors),
        evaluated=len(cache),
        legal=legal,
        rejected=len(cache) - legal,
        exhaustive=not stopped,
        rejection_reasons=tuple(sorted(reasons.items())),
        algorithm_anchor_counts=tuple(sorted(algorithm_anchor_counts.items())),
        algorithm_region_counts=tuple(sorted(algorithm_region_counts.items())),
    )


def solve_ideal_region(
    operator: Operator,
    hardware: Hardware,
    space: ScheduleSpace | None = None,
    policy: SearchPolicy | None = None,
) -> SolveResult:
    """Rank the finite, hardware-derived ideal schedule neighbourhood."""

    space = space or ScheduleSpace()
    policy = policy or SearchPolicy()
    region = derive_ideal_region(operator, hardware, space, policy)
    scored = []
    for plan in region.plans:
        result = simulate(operator, plan, hardware)
        if result.valid:
            scored.append((result.total_cycles, plan, result))
    scored.sort(key=lambda item: (item[0], item[1].as_dict().__repr__()))
    ranked: list[RankedTiling] = []
    for rank, (_, plan, result) in enumerate(scored[:policy.top_k], 1):
        ranked.append(RankedTiling(
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
        ))
    return SolveResult(
        ranked=ranked,
        evaluated=region.evaluated,
        legal=region.legal,
        rejected=region.rejected,
        exhaustive=region.exhaustive,
        rejection_reasons=dict(region.rejection_reasons),
        search_metadata={
            "region_plan_count": len(region.plans),
            "anchor_count": len(region.anchors),
            "algorithm_anchor_counts": dict(region.algorithm_anchor_counts),
            "algorithm_region_counts": dict(region.algorithm_region_counts),
            "method": "hardware_transition_local_optima",
        },
    )
