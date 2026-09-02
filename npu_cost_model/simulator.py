"""Operator-independent lowering and cycle simulation.

The evaluator consumes only an :class:`Operator`, a numeric
:class:`TilingPlan` and a :class:`Hardware` profile.  Operator names and
algorithm names are metadata; all branching is on declared hardware
semantics such as indirect access, reduction partitions or memory routes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from math import ceil, inf, prod

from .hardware import Hardware
from .ir import (
    Access,
    AccessMode,
    AccessPattern,
    Algorithm,
    MemorySpace,
    Operator,
    Primitive,
    Resource,
    Stage,
    StageScope,
    dtype_bytes,
)
from .schedule import TilingPlan


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def align_up(value: int, alignment: int) -> int:
    return ceil_div(value, alignment) * alignment


@dataclass
class WorkCost:
    valid: bool = True
    error: str = ""
    elapsed_cycles: float = 0.0
    resource_cycles: dict[Resource, float] = field(default_factory=dict)
    gm_read_bytes: float = 0.0
    gm_write_bytes: float = 0.0
    l2_bytes: float = 0.0

    def add(self, other: "WorkCost") -> None:
        self.valid = self.valid and other.valid
        if not self.error and other.error:
            self.error = other.error
        self.elapsed_cycles += other.elapsed_cycles
        self.gm_read_bytes += other.gm_read_bytes
        self.gm_write_bytes += other.gm_write_bytes
        self.l2_bytes += other.l2_bytes
        for resource, cycles in other.resource_cycles.items():
            self.resource_cycles[resource] = (
                self.resource_cycles.get(resource, 0.0) + cycles
            )

    def scaled(self, count: int) -> "WorkCost":
        return WorkCost(
            valid=self.valid,
            error=self.error,
            elapsed_cycles=self.elapsed_cycles * count,
            resource_cycles={
                resource: cycles * count
                for resource, cycles in self.resource_cycles.items()
            },
            gm_read_bytes=self.gm_read_bytes * count,
            gm_write_bytes=self.gm_write_bytes * count,
            l2_bytes=self.l2_bytes * count,
        )


@dataclass(frozen=True)
class SimulationResult:
    valid: bool
    error: str
    total_cycles: float
    critical_core_cycles: float
    average_core_cycles: float
    hbm_cycles: float
    l2_cycles: float
    shared_resource_cycles: float
    bottleneck: str
    launch_cycles: float
    reduction_cycles: float
    active_cores: int
    workspace_bytes: int
    gm_read_bytes: float
    gm_write_bytes: float
    l2_bytes: float
    peak_memory_bytes: tuple[tuple[MemorySpace, int], ...]
    resource_cycles: tuple[tuple[Resource, float], ...]


def _invalid(error: str) -> SimulationResult:
    return SimulationResult(
        valid=False,
        error=error,
        total_cycles=inf,
        critical_core_cycles=inf,
        average_core_cycles=inf,
        hbm_cycles=inf,
        l2_cycles=inf,
        shared_resource_cycles=inf,
        bottleneck="invalid",
        launch_cycles=0.0,
        reduction_cycles=0.0,
        active_cores=0,
        workspace_bytes=0,
        gm_read_bytes=0.0,
        gm_write_bytes=0.0,
        l2_bytes=0.0,
        peak_memory_bytes=(),
        resource_cycles=(),
    )


def _transfer_resource(
    source: MemorySpace,
    destination: MemorySpace,
    mode: AccessMode,
) -> Resource:
    if mode == AccessMode.ATOMIC_ADD:
        return Resource.ATOMIC
    if source == MemorySpace.L1 and destination in (MemorySpace.L0A, MemorySpace.L0B):
        return Resource.MTE1
    if source == MemorySpace.L0C and destination == MemorySpace.GM:
        return Resource.FIXPIPE
    if source == MemorySpace.UB and destination == MemorySpace.GM:
        return Resource.MTE3
    if destination == MemorySpace.GM:
        return Resource.MTE3
    return Resource.MTE2


def _points(extents: dict[str, int], axes: tuple[str, ...]) -> int:
    return prod(extents[name] for name in axes) if axes else 1


def _traffic_bytes(
    access: Access,
    extents: dict[str, int],
    tiles: dict[str, int],
    element_bytes: int,
) -> tuple[int, int, int]:
    """Return service bytes, useful bytes and issued copy requests."""

    elements = _points(extents, access.axes)
    useful = elements * element_bytes
    if access.pattern == AccessPattern.INDIRECT:
        requests = ceil_div(elements, access.coalesced_elements)
        return requests * access.transaction_bytes, useful, requests

    if access.pattern == AccessPattern.STRIDED:
        contiguous = access.contiguous_axes or access.axes[-1:]
        run_elements = _points(extents, contiguous)
        runs = max(1, ceil_div(elements, max(1, run_elements)))
        run_bytes = run_elements * element_bytes
        traffic = runs * align_up(run_bytes, access.transaction_bytes)
        return traffic, useful, runs

    # A direct tiled access is submitted once per logical tile, not once per
    # 32-byte block. Service bytes still include block alignment per request.
    requests = prod(ceil_div(extents[axis], tiles[axis]) for axis in access.axes)
    average_request = ceil_div(useful, max(1, requests))
    traffic = requests * align_up(average_request, access.transaction_bytes)
    return traffic, useful, requests


def _work_item(
    hardware: Hardware,
    resource: Resource,
    *,
    dtype: str | None = None,
    byte_count: float = 0.0,
    operation_count: float = 0.0,
    issue_count: float = 0.0,
    fixed_cycles: float = 0.0,
    latency_waves: float = 0.0,
    route: tuple[MemorySpace, MemorySpace] | None = None,
) -> WorkCost:
    rate = hardware.rate(resource, dtype)
    byte_rate = hardware.route_rate(*route) if route is not None else 0.0
    if byte_rate <= 0.0:
        byte_rate = rate.bytes_per_cycle
    if byte_count > 0.0 and byte_rate <= 0.0:
        return WorkCost(False, f"missing byte rate for {resource.value}")
    if operation_count > 0.0 and rate.operations_per_cycle <= 0.0:
        return WorkCost(False, f"missing operation rate for {resource.value}/{dtype}")
    occupancy = fixed_cycles + issue_count * rate.issue_cycles
    if byte_count > 0.0:
        occupancy += byte_count / byte_rate
    if operation_count > 0.0:
        occupancy += operation_count / rate.operations_per_cycle
    elapsed = occupancy + latency_waves * rate.latency_cycles
    return WorkCost(
        elapsed_cycles=elapsed,
        resource_cycles={resource: occupancy},
    )


def _access_cost(
    operator: Operator,
    algorithm: Algorithm,
    plan: TilingPlan,
    access: Access,
    extents: dict[str, int],
    hardware: Hardware,
    reduction_partitions: int,
    partial_dtype: str,
) -> WorkCost:
    tensor = operator.tensor(access.tensor)
    # A result write becomes a partial-result write when the schedule
    # parallelizes a reduction.
    value_dtype = (
        partial_dtype
        if access.is_result and reduction_partitions > 1
        else tensor.dtype
    )
    element_bytes = dtype_bytes(value_dtype)
    traffic, useful, transactions = _traffic_bytes(
        access, extents, plan.tiles, element_bytes
    )
    port_bytes = traffic
    if access.service_bytes_per_element is not None:
        port_bytes = align_up(
            _points(extents, access.axes) * access.service_bytes_per_element,
            access.transaction_bytes,
        )
    reuse_factor = 1
    if (
        access.mode == AccessMode.READ
        and access.pattern != AccessPattern.INDIRECT
        and any(space in access.path for space in (MemorySpace.L1, MemorySpace.UB))
    ):
        traversal = plan.traversal or algorithm.output_axes
        task_tiles = plan.tasks
        cache_tiles = plan.caches
        for axis_name in reversed(traversal):
            if axis_name in access.axes:
                break
            reuse_factor *= max(
                1, ceil_div(cache_tiles[axis_name], task_tiles[axis_name])
            )
    result = WorkCost()
    path = access.path
    for source, destination in zip(path, path[1:]):
        resource = _transfer_resource(source, destination, access.mode)
        service = float(
            port_bytes
            if access.service_bytes_per_element is not None
            else traffic
            if MemorySpace.GM in (source, destination)
            else useful
        )
        hop_transactions = float(transactions)
        if source == MemorySpace.GM and reuse_factor > 1:
            service /= reuse_factor
            hop_transactions /= reuse_factor
        waves = 1.0
        if access.pattern == AccessPattern.INDIRECT and MemorySpace.GM in (
            source, destination
        ):
            waves = float(ceil(hop_transactions / 16.0))
        if access.mode == AccessMode.ATOMIC_ADD:
            waves *= access.contention_factor
        work = _work_item(
            hardware,
            resource,
            byte_count=service,
            issue_count=hop_transactions,
            latency_waves=waves,
            route=(source, destination),
        )
        if source == MemorySpace.GM:
            work.gm_read_bytes += traffic / reuse_factor
            work.l2_bytes += 2.0 * traffic / reuse_factor
        if destination == MemorySpace.GM:
            work.gm_write_bytes += traffic
            work.l2_bytes += 2.0 * traffic
        if access.mode == AccessMode.ATOMIC_ADD and destination == MemorySpace.GM:
            work.gm_read_bytes += traffic
            work.l2_bytes += 2.0 * traffic
        result.add(work)
    return result


def _primitive_cost(
    primitive: Primitive,
    extents: dict[str, int],
    tiles: dict[str, int],
    hardware: Hardware,
) -> WorkCost:
    work_extents = dict(extents)
    for axis in primitive.padded_axes:
        work_extents[axis] = align_up(extents[axis], tiles[axis])
    points = _points(work_extents, primitive.axes)
    operations = points * primitive.operations_per_point
    issues = ceil_div(points, primitive.issue_elements)
    return _work_item(
        hardware,
        primitive.resource,
        dtype=primitive.dtype,
        operation_count=operations,
        issue_count=float(issues),
        fixed_cycles=primitive.fixed_cycles,
    )


def _stage_cost(
    operator: Operator,
    algorithm: Algorithm,
    plan: TilingPlan,
    stage: Stage,
    extents: dict[str, int],
    hardware: Hardware,
    reduction_partitions: int,
    partial_dtype: str,
) -> WorkCost:
    children = [
        *(
            _access_cost(
                operator,
                algorithm,
                plan,
                access,
                extents,
                hardware,
                reduction_partitions,
                partial_dtype,
            )
            for access in stage.accesses
        ),
        *(
            _primitive_cost(primitive, extents, plan.tiles, hardware)
            for primitive in stage.primitives
        ),
    ]
    result = WorkCost()
    for child in children:
        result.add(child)
    if not result.valid:
        return result
    if stage.concurrent:
        longest = max((child.elapsed_cycles for child in children), default=0.0)
        resource_roof = max(result.resource_cycles.values(), default=0.0)
        result.elapsed_cycles = max(longest, resource_roof)
    return result


def _local_memory(
    operator: Operator,
    algorithm: Algorithm,
    plan: TilingPlan,
    reduction_partitions: int,
) -> dict[MemorySpace, int]:
    tiles = plan.tiles
    buffers = plan.buffer_counts
    allocations: dict[tuple[str, MemorySpace], int] = {}
    for stage in algorithm.stages:
        for access in stage.accesses:
            tensor = operator.tensor(access.tensor)
            local_dtype = access.local_dtype or tensor.dtype
            elements = prod(
                min(operator.axis(axis).extent, tiles[axis])
                for axis in access.axes
            ) if access.axes else 1
            byte_count = elements * dtype_bytes(local_dtype)
            for space in access.path:
                if space in (MemorySpace.GM, MemorySpace.L2):
                    continue
                key = (tensor.name, space)
                allocations[key] = max(allocations.get(key, 0), byte_count)
    peak: dict[MemorySpace, int] = {}
    for (_, space), byte_count in allocations.items():
        peak[space] = peak.get(space, 0) + byte_count * buffers.get(space, 1)

    if reduction_partitions > 1:
        output_tile_elements = prod(tiles[axis] for axis in algorithm.output_axes)
        partial_live = 2 * output_tile_elements * dtype_bytes(algorithm.partial_dtype)
        peak[MemorySpace.UB] = peak.get(MemorySpace.UB, 0) + partial_live
    return peak


def _cache_memory(
    operator: Operator,
    algorithm: Algorithm,
    plan: TilingPlan,
    reduction_partitions: int,
) -> int:
    """Estimate one live L2 schedule group's unique tensor footprint."""

    cache_tiles = plan.caches
    allocations: dict[str, int] = {}
    for stage in algorithm.stages:
        for access in stage.accesses:
            if MemorySpace.GM not in access.path:
                continue
            tensor = operator.tensor(access.tensor)
            value_dtype = (
                algorithm.partial_dtype
                if access.is_result and reduction_partitions > 1
                else tensor.dtype
            )
            elements = 1
            for axis_name in access.axes:
                axis = operator.axis(axis_name)
                if axis_name in algorithm.output_axes:
                    extent = min(axis.extent, cache_tiles[axis_name])
                elif axis_name in algorithm.reduction_axes:
                    extent = ceil_div(axis.extent, reduction_partitions)
                else:
                    extent = axis.extent
                elements *= extent
            allocations[tensor.name] = max(
                allocations.get(tensor.name, 0),
                elements * dtype_bytes(value_dtype),
            )
    return sum(allocations.values())


def _axis_classes(extent: int, tile: int) -> list[tuple[int, int]]:
    full = extent // tile
    tail = extent % tile
    result: list[tuple[int, int]] = []
    if full:
        result.append((tile, full))
    if tail:
        result.append((tail, 1))
    return result


def _partition_classes(extent: int, parts: int) -> list[tuple[int, int]]:
    small = extent // parts
    extra = extent % parts
    result: list[tuple[int, int]] = []
    if parts - extra:
        result.append((small, parts - extra))
    if extra:
        result.append((small + 1, extra))
    return [(size, count) for size, count in result if size > 0 and count > 0]


def _task_classes(
    operator: Operator,
    algorithm: Algorithm,
    plan: TilingPlan,
) -> list[tuple[dict[str, int], int]]:
    tiles = plan.tasks
    reductions = plan.reductions
    dimensions: list[tuple[str, list[tuple[int, int]]]] = []
    scheduled = set(algorithm.output_axes) | set(algorithm.reduction_axes)
    for axis_name in algorithm.output_axes:
        axis = operator.axis(axis_name)
        dimensions.append((axis_name, _axis_classes(axis.extent, tiles[axis_name])))
    for axis_name in algorithm.reduction_axes:
        axis = operator.axis(axis_name)
        dimensions.append(
            (axis_name, _partition_classes(axis.extent, reductions.get(axis_name, 1)))
        )

    classes: list[tuple[dict[str, int], int]] = []
    for combination in product(*(values for _, values in dimensions)):
        extents = {
            name: value[0]
            for (name, _), value in zip(dimensions, combination)
        }
        for axis in operator.axes:
            if axis.name not in scheduled:
                extents[axis.name] = axis.extent
        multiplicity = prod(value[1] for value in combination)
        classes.append((extents, multiplicity))
    return classes


def _kernel_stage_cost(
    operator: Operator,
    algorithm: Algorithm,
    plan: TilingPlan,
    hardware: Hardware,
    active_cores: int,
    reduction_partitions: int,
) -> WorkCost:
    extents = {axis.name: axis.extent for axis in operator.axes}
    total = WorkCost()
    for stage in algorithm.stages:
        if stage.scope != StageScope.KERNEL:
            continue
        total.add(
            _stage_cost(
                operator,
                algorithm,
                plan,
                stage,
                extents,
                hardware,
                reduction_partitions,
                algorithm.partial_dtype,
            )
        )
    if active_cores <= 1:
        return total
    total.elapsed_cycles /= active_cores
    total.gm_read_bytes /= active_cores
    total.gm_write_bytes /= active_cores
    total.l2_bytes /= active_cores
    total.resource_cycles = {
        resource: cycles / active_cores
        for resource, cycles in total.resource_cycles.items()
    }
    return total


def simulate(operator: Operator, plan: TilingPlan, hardware: Hardware) -> SimulationResult:
    if plan.algorithm >= len(operator.algorithms):
        return _invalid("algorithm index is outside the operator IR")
    algorithm = operator.algorithms[plan.algorithm]
    tiles = plan.tiles
    if set(tiles) != {axis.name for axis in operator.axes}:
        return _invalid("tiling must provide exactly one tile for every operator axis")
    for axis in operator.axes:
        tile = tiles[axis.name]
        if tile % axis.alignment != 0:
            return _invalid(f"axis {axis.name} tile violates alignment {axis.alignment}")
    tasks = plan.tasks
    if set(tasks) != {axis.name for axis in operator.axes}:
        return _invalid("task tiling must provide exactly one tile for every operator axis")
    for axis in operator.axes:
        task_tile = tasks[axis.name]
        if task_tile < min(tiles[axis.name], axis.extent):
            return _invalid(f"axis {axis.name} task tile is smaller than its inner tile")
    caches = plan.caches
    if set(caches) != {axis.name for axis in operator.axes}:
        return _invalid("cache tiling must provide exactly one tile for every operator axis")
    for axis in operator.axes:
        if caches[axis.name] < min(tasks[axis.name], axis.extent):
            return _invalid(f"axis {axis.name} cache tile is smaller than its task tile")

    if set(plan.traversal) != set(algorithm.output_axes) or (
        len(plan.traversal) != len(algorithm.output_axes)
    ):
        return _invalid("traversal must be a permutation of the output axes")
    if not set(plan.buffer_counts) <= set(algorithm.buffered_spaces):
        return _invalid("plan buffers a memory space not declared by the algorithm")

    hardware_cores = hardware.core_count(algorithm.core_resource)
    if plan.used_cores > hardware_cores:
        return _invalid(
            f"used_cores exceeds {algorithm.core_resource.value} core count"
        )

    reductions = plan.reductions
    if not set(reductions) <= set(algorithm.reduction_axes):
        return _invalid("plan partitions an axis that is not a reduction")
    total_reduction_partitions = 1
    for axis_name in algorithm.reduction_axes:
        parts = reductions.get(axis_name, 1)
        if parts > 1 and not algorithm.parallel_reduction:
            return _invalid("algorithm does not permit a parallel reduction")
        available_chunks = ceil_div(operator.axis(axis_name).extent, tiles[axis_name])
        if parts > available_chunks:
            return _invalid(f"reduction partitions exceed {axis_name} tile chunks")
        total_reduction_partitions *= parts

    peak = _local_memory(
        operator, algorithm, plan, total_reduction_partitions
    )
    peak[MemorySpace.L2] = _cache_memory(
        operator, algorithm, plan, total_reduction_partitions
    )
    for space, byte_count in peak.items():
        # L2 is a cache, not a statically allocated scratchpad.  A working
        # set larger than L2 is executable and causes refetch traffic; local
        # L1/L0/UB overflows remain hard legality failures.
        if space == MemorySpace.L2:
            continue
        capacity = hardware.capacities.get(space, 0)
        if capacity and byte_count > capacity:
            return _invalid(
                f"{space.value} capacity exceeded: {byte_count}>{capacity}"
            )

    classes = _task_classes(operator, algorithm, plan)
    total_tasks = sum(count for _, count in classes)
    active_cores = min(plan.used_cores, total_tasks)
    if active_cores <= 0:
        return _invalid("schedule has no executable task")

    core_serial = [0.0] * active_cores
    core_resources: list[dict[Resource, float]] = [dict() for _ in range(active_cores)]
    core_fill = [0.0] * active_cores
    core_gm_read = [0.0] * active_cores
    core_gm_write = [0.0] * active_cores
    core_l2 = [0.0] * active_cores
    offset = 0
    task_cache: dict[tuple[tuple[str, int], ...], WorkCost] = {}
    for extents, multiplicity in classes:
        signature = tuple(sorted(extents.items()))
        task = task_cache.get(signature)
        if task is None:
            task = WorkCost()
            for stage in algorithm.stages:
                if stage.scope == StageScope.TASK:
                    task.add(
                        _stage_cost(
                            operator,
                            algorithm,
                            plan,
                            stage,
                            extents,
                            hardware,
                            total_reduction_partitions,
                            algorithm.partial_dtype,
                        )
                    )
            task_cache[signature] = task
        if not task.valid:
            return _invalid(task.error)
        base = multiplicity // active_cores
        extra = multiplicity % active_cores
        for core in range(active_cores):
            count = base + (1 if (core - offset) % active_cores < extra else 0)
            if count <= 0:
                continue
            core_serial[core] += task.elapsed_cycles * count
            core_gm_read[core] += task.gm_read_bytes * count
            core_gm_write[core] += task.gm_write_bytes * count
            core_l2[core] += task.l2_bytes * count
            task_roof = max(task.resource_cycles.values(), default=0.0)
            core_fill[core] = max(core_fill[core], task.elapsed_cycles - task_roof)
            for resource, cycles in task.resource_cycles.items():
                core_resources[core][resource] = (
                    core_resources[core].get(resource, 0.0) + cycles * count
                )
        offset = (offset + extra) % active_cores

    kernel_stage = _kernel_stage_cost(
        operator,
        algorithm,
        plan,
        hardware,
        active_cores,
        total_reduction_partitions,
    )
    if not kernel_stage.valid:
        return _invalid(kernel_stage.error)

    boundaries = algorithm.pipeline_boundaries or tuple(
        (space,) for space in algorithm.buffered_spaces
    )
    satisfied_boundaries = sum(
        all(plan.buffer_counts.get(space, 1) == 2 for space in boundary)
        for boundary in boundaries
    )
    overlap_fraction = (
        satisfied_boundaries / len(boundaries)
        if algorithm.pipeline_capable and boundaries
        else 0.0
    )
    core_cycles: list[float] = []
    aggregate_resources: dict[Resource, float] = {}
    gm_read = gm_write = l2_bytes = 0.0
    for core in range(active_cores):
        for resource, cycles in kernel_stage.resource_cycles.items():
            core_resources[core][resource] = (
                core_resources[core].get(resource, 0.0) + cycles
            )
        core_serial[core] += kernel_stage.elapsed_cycles
        core_gm_read[core] += kernel_stage.gm_read_bytes
        core_gm_write[core] += kernel_stage.gm_write_bytes
        core_l2[core] += kernel_stage.l2_bytes
        if overlap_fraction > 0.0:
            resource_roof = max(core_resources[core].values(), default=0.0)
            fully_pipelined = resource_roof + core_fill[core]
            cycles = core_serial[core] - overlap_fraction * (
                core_serial[core] - fully_pipelined
            )
        else:
            cycles = core_serial[core]
        core_cycles.append(cycles)
        gm_read += core_gm_read[core]
        gm_write += core_gm_write[core]
        l2_bytes += core_l2[core]
        for resource, cycles_value in core_resources[core].items():
            aggregate_resources[resource] = (
                aggregate_resources.get(resource, 0.0) + cycles_value
            )

    workspace_bytes = 0
    reduction_cycles = 0.0
    if total_reduction_partitions > 1:
        if algorithm.result_tensor is None:
            return _invalid("parallel reduction has no result tensor")
        result_tensor = operator.tensor(algorithm.result_tensor)
        output_elements = prod(operator.axis(axis).extent for axis in algorithm.output_axes)
        partial_bytes = output_elements * dtype_bytes(algorithm.partial_dtype)
        workspace_bytes = partial_bytes * total_reduction_partitions
        reduction_read = float(workspace_bytes)
        reduction_write = float(result_tensor.elements * dtype_bytes(result_tensor.dtype))
        reduction_operations = (
            output_elements
            * (total_reduction_partitions - 1)
            * algorithm.reduction_operations_per_element
        )
        reduction_cores = min(
            hardware.core_count(algorithm.reduction_resource),
            max(1, ceil_div(output_elements, 256)),
        )
        operation_rate = hardware.rate(
            algorithm.reduction_resource, algorithm.partial_dtype
        ).operations_per_cycle
        if reduction_operations and operation_rate <= 0.0:
            return _invalid("parallel reduction resource has no operation rate")
        vector_cycles = reduction_operations / max(
            1.0, operation_rate * reduction_cores
        )
        sync_rate = hardware.rate(Resource.SYNC).operations_per_cycle
        sync_cycles = (active_cores + reduction_cores) / max(1.0e-12, sync_rate)
        reduction_cycles = (
            hardware.kernel_launch_cycles + vector_cycles + sync_cycles
        )
        critical_index = max(range(active_cores), key=core_cycles.__getitem__)
        core_cycles[critical_index] += reduction_cycles
        gm_read += reduction_read
        gm_write += reduction_write
        l2_bytes += 2.0 * (reduction_read + reduction_write)
        aggregate_resources[algorithm.reduction_resource] = (
            aggregate_resources.get(algorithm.reduction_resource, 0.0)
            + reduction_operations / max(1.0, operation_rate)
        )
        aggregate_resources[Resource.SYNC] = (
            aggregate_resources.get(Resource.SYNC, 0.0) + sync_cycles
        )

    critical = max(core_cycles)
    average = sum(core_cycles) / active_cores
    l2_capacity = hardware.capacities.get(MemorySpace.L2, 0)
    cache_pressure = (
        max(1.0, peak.get(MemorySpace.L2, 0) / l2_capacity)
        if l2_capacity > 0
        else 1.0
    )
    if cache_pressure > 1.0:
        refetch = gm_read * (cache_pressure - 1.0)
        gm_read += refetch
        l2_bytes += 2.0 * refetch
    hbm_cycles = (
        (gm_read + gm_write) / hardware.aggregate_hbm_bytes_per_cycle
        if hardware.aggregate_hbm_bytes_per_cycle > 0.0
        else 0.0
    )
    l2_cycles = (
        l2_bytes / hardware.aggregate_l2_bytes_per_cycle
        if hardware.aggregate_l2_bytes_per_cycle > 0.0
        else 0.0
    )
    shared_cycles = 0.0
    for resource, cycles in aggregate_resources.items():
        units = hardware.parallel_units.get(resource, 0.0)
        if units > 0.0:
            shared_cycles = max(shared_cycles, cycles / units)
    launch = (
        hardware.kernel_launch_cycles
        + hardware.active_core_launch_cycles * active_cores
    )
    roofs = {
        "critical_core": critical,
        "hbm": hbm_cycles,
        "l2": l2_cycles,
        "shared_resource": shared_cycles,
    }
    bottleneck = max(roofs, key=roofs.__getitem__)
    total = roofs[bottleneck] + launch
    return SimulationResult(
        valid=True,
        error="",
        total_cycles=total,
        critical_core_cycles=critical,
        average_core_cycles=average,
        hbm_cycles=hbm_cycles,
        l2_cycles=l2_cycles,
        shared_resource_cycles=shared_cycles,
        bottleneck=bottleneck,
        launch_cycles=launch,
        reduction_cycles=reduction_cycles,
        active_cores=active_cores,
        workspace_bytes=workspace_bytes,
        gm_read_bytes=gm_read,
        gm_write_bytes=gm_write,
        l2_bytes=l2_bytes,
        peak_memory_bytes=tuple(sorted(peak.items(), key=lambda item: item[0].value)),
        resource_cycles=tuple(sorted(aggregate_resources.items(), key=lambda item: item[0].value)),
    )
