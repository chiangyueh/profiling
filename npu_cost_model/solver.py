"""Legal schedule generation and parameter-only cost ranking."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from heapq import heappush, heapreplace
from itertools import product
from math import floor, gcd, prod

from .hardware import Hardware
from .ir import AccessMode, Axis, MemorySpace, Operator, dtype_bytes
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
    unknown_coupled = set(space.coupled_task_axes) - {
        axis.name for axis in operator.axes
    }
    if unknown_coupled:
        raise ValueError(
            "coupled task axes are not present in the operator: "
            + ", ".join(sorted(unknown_coupled))
        )
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
            if axis.name in space.coupled_task_axes:
                tasks = (inner,)
            elif explicit is not None:
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


def _level_for_targets(
    levels: tuple[tuple[int, int, int], ...],
    inner: int,
    task: int,
    cache: int,
) -> tuple[int, int, int]:
    """Select the declared legal level nearest three physical targets."""

    same_inner = tuple(level for level in levels if level[0] == inner)
    return _nearest_level(same_inner or levels, (inner, task, cache))


def _local_memory_pressure(
    operator: Operator,
    algorithm,
    axis_names: tuple[str, ...],
    levels: tuple[tuple[int, int, int], ...],
    buffers: tuple[tuple[MemorySpace, int], ...],
    reductions: tuple[tuple[str, int], ...],
    hardware: Hardware,
) -> float:
    """Return the largest scratchpad-capacity ratio for an inner tile."""

    tiles = {
        name: level[0] for name, level in zip(axis_names, levels)
    }
    buffer_counts = dict(buffers)
    allocations: dict[tuple[str, MemorySpace], int] = {}
    for stage in algorithm.stages:
        for access in stage.accesses:
            tensor = operator.tensor(access.tensor)
            element_bytes = dtype_bytes(access.local_dtype or tensor.dtype)
            elements = prod(
                min(operator.axis(axis).extent, tiles[axis])
                for axis in access.axes
            ) if access.axes else 1
            for memory in access.path:
                if memory in (MemorySpace.GM, MemorySpace.L2):
                    continue
                key = (tensor.name, memory)
                allocations[key] = max(
                    allocations.get(key, 0), elements * element_bytes
                )
    usage: dict[MemorySpace, int] = {}
    for (_, memory), byte_count in allocations.items():
        usage[memory] = usage.get(memory, 0) + (
            byte_count * buffer_counts.get(memory, 1)
        )
    if prod(value for _, value in reductions) > 1:
        output_elements = prod(tiles[axis] for axis in algorithm.output_axes)
        usage[MemorySpace.UB] = usage.get(MemorySpace.UB, 0) + (
            2 * output_elements * dtype_bytes(algorithm.partial_dtype)
        )
    return max(
        (
            byte_count / capacity
            for memory, byte_count in usage.items()
            if (capacity := hardware.capacities.get(memory, 0)) > 0
        ),
        default=0.0,
    )


def _fit_inner_capacity(
    operator: Operator,
    algorithm,
    axis_names: tuple[str, ...],
    levels_by_axis: tuple[tuple[tuple[int, int, int], ...], ...],
    levels: tuple[tuple[int, int, int], ...],
    buffers: tuple[tuple[MemorySpace, int], ...],
    reductions: tuple[tuple[str, int], ...],
    hardware: Hardware,
) -> tuple[tuple[int, int, int], ...]:
    """Project a target geometry onto all declared scratchpad constraints."""

    current = levels
    pressure = _local_memory_pressure(
        operator, algorithm, axis_names, current, buffers, reductions, hardware
    )
    while pressure > 1.0:
        alternatives: list[
            tuple[float, int, tuple[tuple[int, int, int], ...]]
        ] = []
        for axis_index, options in enumerate(levels_by_axis):
            inner_values = sorted(set(level[0] for level in options))
            position = inner_values.index(current[axis_index][0])
            if position == 0:
                continue
            inner = inner_values[position - 1]
            changed = list(current)
            changed[axis_index] = _level_for_targets(
                options,
                inner,
                max(inner, current[axis_index][1]),
                max(inner, current[axis_index][2]),
            )
            changed_tuple = tuple(changed)
            next_pressure = _local_memory_pressure(
                operator, algorithm, axis_names, changed_tuple,
                buffers, reductions, hardware,
            )
            retained_volume = prod(level[0] for level in changed_tuple)
            alternatives.append(
                (next_pressure, -retained_volume, changed_tuple)
            )
        if not alternatives:
            break
        next_pressure, _, current = min(alternatives)
        if next_pressure >= pressure and current == levels:
            break
        pressure = next_pressure
    return current


def _balanced_output_parts(
    operator: Operator,
    algorithm,
    levels: tuple[tuple[int, int, int], ...],
    core_count: int,
) -> dict[str, int]:
    """Factor one physical core wave over output axes by useful extent."""

    by_name = {
        axis.name: level for axis, level in zip(operator.axes, levels)
    }
    parts = {axis: 1 for axis in algorithm.output_axes}
    limits = {
        axis: ceil_div(
            operator.axis(axis).extent,
            min(operator.axis(axis).extent, by_name[axis][0]),
        )
        for axis in algorithm.output_axes
    }
    while prod(parts.values()) < core_count:
        candidates = tuple(
            axis for axis in algorithm.output_axes
            if parts[axis] < limits[axis]
        )
        if not candidates:
            break
        selected = max(
            candidates,
            key=lambda axis: (
                operator.axis(axis).extent
                / (parts[axis] * by_name[axis][0]),
                -algorithm.output_axes.index(axis),
            ),
        )
        parts[selected] += 1
    return parts


def _output_part_profiles(
    operator: Operator,
    algorithm,
    levels_by_axis: tuple[tuple[tuple[int, int, int], ...], ...],
    levels: tuple[tuple[int, int, int], ...],
    core_count: int,
    coupled_task_axes: frozenset[str],
) -> tuple[tuple[tuple[str, int], ...], ...]:
    """Derive exact core-wave grids without a tile Cartesian product."""

    by_name = {
        axis.name: level for axis, level in zip(operator.axes, levels)
    }
    axes = algorithm.output_axes
    options_by_name = {
        axis.name: options
        for axis, options in zip(operator.axes, levels_by_axis)
    }
    limits = {}
    for axis in axes:
        inner = (
            min(level[0] for level in options_by_name[axis])
            if axis in coupled_task_axes else by_name[axis][0]
        )
        limits[axis] = ceil_div(
            operator.axis(axis).extent,
            min(operator.axis(axis).extent, inner),
        )
    profiles: list[tuple[tuple[str, int], ...]] = []

    def add(parts: dict[str, int]) -> None:
        profile = tuple((axis, parts.get(axis, 1)) for axis in axes)
        if profile not in profiles:
            profiles.append(profile)

    add(_balanced_output_parts(
        operator, algorithm, levels, core_count
    ))
    for left_index, left in enumerate(axes):
        if core_count <= limits[left]:
            add({left: core_count})
        for right in axes[left_index + 1:]:
            for left_parts in range(1, core_count + 1):
                if core_count % left_parts:
                    continue
                right_parts = core_count // left_parts
                if left_parts <= limits[left] and right_parts <= limits[right]:
                    add({left: left_parts, right: right_parts})
    return tuple(profiles)


def _retarget_tasks(
    operator: Operator,
    algorithm,
    levels_by_axis: tuple[tuple[tuple[int, int, int], ...], ...],
    levels: tuple[tuple[int, int, int], ...],
    output_parts: tuple[tuple[str, int], ...],
    coupled_task_axes: frozenset[str],
) -> tuple[tuple[int, int, int], ...]:
    parts = dict(output_parts)
    result: list[tuple[int, int, int]] = []
    for axis, options, level in zip(operator.axes, levels_by_axis, levels):
        inner = level[0]
        task = (
            align_up(ceil_div(axis.extent, parts[axis.name]), axis.alignment)
            if axis.name in parts else inner
        )
        if axis.name in parts and axis.name in coupled_task_axes:
            result.append(_nearest_level(options, (task, task, task)))
        else:
            result.append(_level_for_targets(options, inner, task, task))
    return tuple(result)


def _l2_footprint(
    operator: Operator,
    algorithm,
    axis_names: tuple[str, ...],
    levels: tuple[tuple[int, int, int], ...],
    reductions: tuple[tuple[str, int], ...],
) -> int:
    caches = {
        name: level[2] for name, level in zip(axis_names, levels)
    }
    reduction_values = dict(reductions)
    reduction_count = prod(reduction_values.values()) if reductions else 1
    allocations: dict[str, int] = {}
    for stage in algorithm.stages:
        for access in stage.accesses:
            if MemorySpace.GM not in access.path:
                continue
            tensor = operator.tensor(access.tensor)
            value_dtype = (
                algorithm.partial_dtype
                if access.is_result and reduction_count > 1 else tensor.dtype
            )
            elements = 1
            for axis_name in access.axes:
                axis = operator.axis(axis_name)
                if axis_name in algorithm.output_axes:
                    extent = min(axis.extent, caches[axis_name])
                elif axis_name in algorithm.reduction_axes:
                    extent = ceil_div(
                        axis.extent, reduction_values.get(axis_name, 1)
                    )
                else:
                    extent = axis.extent
                elements *= extent
            allocations[tensor.name] = max(
                allocations.get(tensor.name, 0),
                elements * dtype_bytes(value_dtype),
            )
    return sum(allocations.values())


def _reuse_axis_weights(operator: Operator, algorithm, traversal) -> dict[str, int]:
    """Count source bytes that one larger cache axis can reuse."""

    weights = {axis: 0 for axis in algorithm.output_axes}
    for stage in algorithm.stages:
        for access in stage.accesses:
            if (
                access.mode != AccessMode.READ
                or MemorySpace.GM not in access.path
                or not any(
                    memory in access.path
                    for memory in (MemorySpace.L1, MemorySpace.UB)
                )
            ):
                continue
            tensor = operator.tensor(access.tensor)
            for axis_name in reversed(traversal):
                if axis_name in access.axes:
                    break
                weights[axis_name] += tensor.elements * dtype_bytes(tensor.dtype)
    return weights


def _expand_cache_for_reuse(
    operator: Operator,
    algorithm,
    axis_names: tuple[str, ...],
    levels_by_axis: tuple[tuple[tuple[int, int, int], ...], ...],
    levels: tuple[tuple[int, int, int], ...],
    reductions: tuple[tuple[str, int], ...],
    traversal: tuple[str, ...],
    hardware: Hardware,
) -> tuple[tuple[int, int, int], ...]:
    """Grow only reuse-bearing cache axes up to the physical L2 boundary."""

    capacity = hardware.capacities.get(MemorySpace.L2, 0)
    if capacity <= 0:
        return levels
    weights = _reuse_axis_weights(operator, algorithm, traversal)
    current = levels
    while True:
        alternatives: list[
            tuple[float, int, tuple[tuple[int, int, int], ...]]
        ] = []
        for axis_index, (axis, options) in enumerate(
            zip(operator.axes, levels_by_axis)
        ):
            weight = weights.get(axis.name, 0)
            if weight <= 0:
                continue
            compatible = tuple(
                level for level in options
                if level[0] == current[axis_index][0]
                and level[1] == current[axis_index][1]
            )
            cache_values = sorted(set(level[2] for level in compatible))
            position = cache_values.index(current[axis_index][2])
            if position + 1 == len(cache_values):
                continue
            changed = list(current)
            changed[axis_index] = (
                current[axis_index][0], current[axis_index][1],
                cache_values[position + 1],
            )
            changed_tuple = tuple(changed)
            footprint = _l2_footprint(
                operator, algorithm, axis_names, changed_tuple, reductions
            )
            if footprint <= capacity:
                gain = weight * (
                    cache_values[position + 1] / current[axis_index][2] - 1.0
                )
                alternatives.append((-gain, footprint, changed_tuple))
        if not alternatives:
            return current
        _, _, current = min(alternatives)


def _best_feasible_buffer(
    operator: Operator,
    algorithm,
    axis_names: tuple[str, ...],
    levels: tuple[tuple[int, int, int], ...],
    reductions: tuple[tuple[str, int], ...],
    profiles: tuple[tuple[tuple[MemorySpace, int], ...], ...],
    hardware: Hardware,
) -> tuple[tuple[MemorySpace, int], ...]:
    """Maximize declared pipeline overlap without overflowing scratchpads."""

    boundaries = algorithm.pipeline_boundaries or tuple(
        (memory,) for memory in algorithm.buffered_spaces
    )
    feasible = tuple(
        profile for profile in profiles
        if _local_memory_pressure(
            operator, algorithm, axis_names, levels, profile,
            reductions, hardware,
        ) <= 1.0
    )
    if not feasible:
        return profiles[0]
    return max(
        feasible,
        key=lambda profile: (
            sum(
                all(dict(profile).get(memory, 1) == 2 for memory in boundary)
                for boundary in boundaries
            ),
            -sum(dict(profile).values()),
        ),
    )


def _direct_hardware_anchors(
    operator: Operator,
    hardware: Hardware,
    space: ScheduleSpace,
    algorithm_index: int,
    levels_by_axis: tuple[tuple[tuple[int, int, int], ...], ...],
    buffers: tuple[tuple[tuple[MemorySpace, int], ...], ...],
    traversals: tuple[tuple[str, ...], ...],
) -> tuple[TilingPlan, ...]:
    """Solve capacity, core-wave, reuse and reduction boundaries directly."""

    algorithm = operator.algorithms[algorithm_index]
    axis_names = tuple(axis.name for axis in operator.axes)
    core_allowed = _core_options(operator, algorithm_index, hardware, space)
    core_count = core_allowed[-1]
    single_buffers = min(buffers, key=lambda profile: sum(dict(profile).values()))
    boundaries = algorithm.pipeline_boundaries or tuple(
        (memory,) for memory in algorithm.buffered_spaces
    )
    overlap_buffers = max(
        buffers,
        key=lambda profile: (
            sum(
                all(dict(profile).get(memory, 1) == 2 for memory in boundary)
                for boundary in boundaries
            ),
            -sum(dict(profile).values()),
        ),
    )
    serial = tuple((axis, 1) for axis in algorithm.reduction_axes)

    output_rank = max(1, len(algorithm.output_axes))
    parallel: list[tuple[int, int, int]] = []
    for axis, levels in zip(operator.axes, levels_by_axis):
        if axis.name in algorithm.output_axes:
            chunks = max(1, round(core_count ** (1.0 / output_rank)))
            task_target = align_up(ceil_div(axis.extent, chunks), axis.alignment)
            inner_target = min(task_target, max(level[0] for level in levels))
        else:
            inner_target = max(level[0] for level in levels)
            task_target = inner_target
        parallel.append(_nearest_level(
            levels, (inner_target, task_target, task_target)
        ))
    scheduled_axes = set(algorithm.output_axes) | set(algorithm.reduction_axes)
    issue_geometry = tuple(
        min(levels) if axis.name in scheduled_axes else max(levels)
        for axis, levels in zip(operator.axes, levels_by_axis)
    )
    target_geometries = (
        issue_geometry,
        tuple(parallel),
        tuple(max(levels) for levels in levels_by_axis),
    )
    inner_geometries = tuple(dict.fromkeys(
        _fit_inner_capacity(
            operator, algorithm, axis_names, levels_by_axis, geometry,
            fit_buffers, serial, hardware,
        )
        for geometry in target_geometries
        for fit_buffers in dict.fromkeys((single_buffers, overlap_buffers))
    ))

    result: list[TilingPlan] = []
    coupled_task_axes = frozenset(space.coupled_task_axes)
    for geometry in inner_geometries:
        part_profiles = (
            (),
            *_output_part_profiles(
                operator, algorithm, levels_by_axis, geometry, core_count,
                coupled_task_axes,
            ),
        )
        for output_parts in part_profiles:
            task_levels = _retarget_tasks(
                operator, algorithm, levels_by_axis, geometry,
                output_parts, coupled_task_axes,
            )
            prototype = _plan_from_levels(
                algorithm_index, axis_names, task_levels, serial,
                core_count, single_buffers, traversals[0],
            )
            reduction_profiles = [serial]
            if algorithm.parallel_reduction:
                reduction_profiles.extend((
                    _reduction_fill_profile(
                        operator, algorithm_index, prototype, hardware, space
                    ),
                    _reduction_chunk_profile(
                        operator, algorithm_index, prototype, hardware, space
                    ),
                ))
            for reductions in tuple(dict.fromkeys(reduction_profiles)):
                for traversal in traversals:
                    cache_levels = _expand_cache_for_reuse(
                        operator, algorithm, axis_names, levels_by_axis,
                        task_levels, reductions, traversal, hardware,
                    )
                    buffer_profile = _best_feasible_buffer(
                        operator, algorithm, axis_names, cache_levels,
                        reductions, buffers, hardware,
                    )
                    candidate = _plan_from_levels(
                        algorithm_index, axis_names, cache_levels, reductions,
                        core_count, buffer_profile, traversal,
                    )
                    candidate = _replace_plan(
                        candidate,
                        used_cores=_useful_core_values(
                            _total_task_count(operator, algorithm, candidate),
                            core_allowed,
                        )[-1],
                    )
                    result.append(candidate)
    return tuple(dict.fromkeys(result))


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
    """Project hardware-model optima, then expand one legal transition.

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
        algorithm_contexts[algorithm_index] = (
            levels_by_axis, buffers, traversals
        )
        direct = _direct_hardware_anchors(
            operator, hardware, space, algorithm_index,
            levels_by_axis, buffers, traversals,
        )
        # Keep the best feasible direct solution in every numeric reduction
        # topology.  This retains serial and each independently partitioned
        # reduction graph without assigning names or fixed quotas to them.
        best_by_topology: dict[tuple[object, ...], tuple[float, TilingPlan]] = {}
        for candidate in direct:
            result = evaluate(candidate)
            if result is None:
                break
            if not result.valid:
                continue
            topology = tuple(
                candidate.reductions.get(axis, 1) > 1
                for axis in algorithm.reduction_axes
            )
            key = (result.total_cycles, candidate.as_dict().__repr__())
            previous = best_by_topology.get(topology)
            if previous is None or key < (
                previous[0], previous[1].as_dict().__repr__()
            ):
                best_by_topology[topology] = (
                    result.total_cycles, candidate
                )
        unique_algorithm_anchors = tuple(
            value[1]
            for _, value in sorted(best_by_topology.items())
        )
        anchors.extend(unique_algorithm_anchors)
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
            "method": "hardware_equation_projection",
        },
    )
