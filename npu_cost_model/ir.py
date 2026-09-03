"""Declarative operator IR for the parameter-only NPU cost simulator.

The IR describes iteration domains, tensor accesses, hardware primitives and
dependencies.  It deliberately contains no measured latency, callback state,
RuntimeKb record or operator-specific cost hook.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import prod
from typing import Iterable


class AxisKind(str, Enum):
    PARALLEL = "parallel"
    REDUCTION = "reduction"


class Resource(str, Enum):
    MTE2 = "mte2"
    MTE1 = "mte1"
    MTE3 = "mte3"
    CUBE = "cube"
    VECTOR = "vector"
    SCALAR = "scalar"
    FIXPIPE = "fixpipe"
    ATOMIC = "atomic"
    SYNC = "sync"


class MemorySpace(str, Enum):
    GM = "gm"
    L2 = "l2"
    L1 = "l1"
    L0A = "l0a"
    L0B = "l0b"
    L0C = "l0c"
    UB = "ub"


class AccessMode(str, Enum):
    READ = "read"
    WRITE = "write"
    ATOMIC_ADD = "atomic_add"


class AccessPattern(str, Enum):
    CONTIGUOUS = "contiguous"
    STRIDED = "strided"
    INDIRECT = "indirect"


class StageScope(str, Enum):
    TASK = "task"
    CORE = "core"
    KERNEL = "kernel"


class TileLevel(str, Enum):
    """Schedule level that determines a core-local allocation's extent."""

    INNER = "inner"
    TASK = "task"
    CACHE = "cache"
    KERNEL = "kernel"


class ReductionProtocol(str, Enum):
    """How independently tiled reduction chunks produce one result.

    These are hardware dataflow semantics, not MatMul template names.  The
    same protocols describe reductions in any operator IR.
    """

    DIRECT = "direct"
    SERIAL_DIRECT = "serial_direct"
    SERIAL_WORKSPACE = "serial_workspace"
    PARALLEL_WORKSPACE = "parallel_workspace"
    PARALLEL_ATOMIC = "parallel_atomic"


@dataclass(frozen=True)
class WorkspaceDimension:
    """One schedule-derived dimension of a temporary GM allocation."""

    axis: str
    level: TileLevel = TileLevel.KERNEL
    alignment: int = 1
    # Some device ABIs reserve a complete padded task slot even when the
    # logical tensor tail is smaller.  Keep clamping as the default for
    # ordinary tensor-shaped scratch, and let the frontend state the padded
    # allocation contract explicitly when the kernel requires it.
    clamp_to_axis: bool = True

    def __post_init__(self) -> None:
        if not self.axis:
            raise ValueError("workspace axis must not be empty")
        if self.alignment <= 0:
            raise ValueError("workspace alignment must be positive")


@dataclass(frozen=True)
class WorkspaceBuffer:
    """Generic temporary-storage and traffic contract.

    It is deliberately expressed in schedule levels and memory routes.  A
    frontend can describe reduction partials, layout-conversion scratch, or
    vector post-processing without teaching the simulator an operator name.
    """

    dimensions: tuple[WorkspaceDimension, ...] = ()
    dtype: str = "uint8"
    copies: int = 1
    per_active_core: bool = False
    per_reduction_partition: bool = False
    fixed_bytes: int = 0
    producer_path: tuple[MemorySpace, ...] = ()
    consumer_path: tuple[MemorySpace, ...] = ()

    def __post_init__(self) -> None:
        names = [dimension.axis for dimension in self.dimensions]
        if len(names) != len(set(names)):
            raise ValueError("workspace dimensions must be unique")
        if self.copies <= 0 or self.fixed_bytes < 0:
            raise ValueError("workspace copies/fixed bytes are invalid")
        dtype_bytes(self.dtype)
        for path in (self.producer_path, self.consumer_path):
            if path and len(path) < 2:
                raise ValueError("workspace memory path needs at least two spaces")


DTYPE_BYTES = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "fp16": 2,
    "bf16": 2,
    "int16": 2,
    "uint16": 2,
    "fp32": 4,
    "int32": 4,
    "uint32": 4,
    "fp64": 8,
    "int64": 8,
    "uint64": 8,
}


def dtype_bytes(dtype: str) -> int:
    try:
        return DTYPE_BYTES[dtype]
    except KeyError as exception:
        raise ValueError(f"unsupported dtype: {dtype}") from exception


@dataclass(frozen=True)
class Axis:
    name: str
    extent: int
    kind: AxisKind = AxisKind.PARALLEL
    alignment: int = 1
    tile_values: tuple[int, ...] = ()
    core_mappable: bool = True
    # A task tile is the iteration region assigned as one schedulable work
    # item; an inner tile is the region resident in local memory at once.
    # Cube contractions normally need both levels, while many vector loops
    # use the same tile at both levels.
    independent_task_tiling: bool = False
    independent_cache_tiling: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("axis name must not be empty")
        if self.extent <= 0:
            raise ValueError(f"axis {self.name} extent must be positive")
        if self.alignment <= 0:
            raise ValueError(f"axis {self.name} alignment must be positive")
        if any(value <= 0 for value in self.tile_values):
            raise ValueError(f"axis {self.name} tile values must be positive")


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple[int, ...]
    dtype: str
    layout: str = "ND"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tensor name must not be empty")
        if not self.shape or any(value <= 0 for value in self.shape):
            raise ValueError(f"tensor {self.name} shape must be non-empty and positive")
        dtype_bytes(self.dtype)

    @property
    def elements(self) -> int:
        return prod(self.shape)


@dataclass(frozen=True)
class Access:
    """One logical tensor access made by a stage.

    ``axes`` names the iteration axes that select distinct tensor elements.
    Axes used by the primitive but absent here are reuse axes.  ``path`` is
    the required hierarchy route and is therefore sufficient to lower the
    access to MTE/FixPipe work without consulting the operator identity.
    """

    tensor: str
    axes: tuple[str, ...]
    mode: AccessMode
    path: tuple[MemorySpace, ...]
    pattern: AccessPattern = AccessPattern.CONTIGUOUS
    contiguous_axes: tuple[str, ...] = ()
    transaction_bytes: int = 32
    coalesced_elements: int = 1
    # Axes whose successive tiles are separated by a producer/consumer
    # dependency.  Their transfer latency cannot be collapsed into one
    # pipeline fill even when the copy engine accepts queued requests.
    dependency_axes: tuple[str, ...] = ()
    contention_factor: float = 1.0
    local_dtype: str | None = None
    # Some engines consume more internal-port bytes than reach GM. FixPipe,
    # for example, reads an FP32 accumulator and writes an FP16 result.
    service_bytes_per_element: int | None = None
    is_result: bool = False
    # By default every local allocation is sized from the inner tile.  A
    # declaration may instead keep a tensor resident for one task, L2 cache
    # group, or the complete kernel.  The simulator consumes only this
    # memory-level declaration and never an operator/template label.
    residency: tuple[tuple[MemorySpace, TileLevel], ...] = ()

    def __post_init__(self) -> None:
        if len(self.path) < 2:
            raise ValueError("an access path needs at least two memory spaces")
        if self.transaction_bytes <= 0:
            raise ValueError("transaction_bytes must be positive")
        if self.coalesced_elements <= 0:
            raise ValueError("coalesced_elements must be positive")
        if len(self.dependency_axes) != len(set(self.dependency_axes)):
            raise ValueError("dependency_axes must be unique")
        if not set(self.dependency_axes) <= set(self.axes):
            raise ValueError("dependency_axes must be accessed tensor axes")
        if self.contention_factor < 1.0:
            raise ValueError("contention_factor must be at least one")
        if self.local_dtype is not None:
            dtype_bytes(self.local_dtype)
        if self.service_bytes_per_element is not None and self.service_bytes_per_element <= 0:
            raise ValueError("service_bytes_per_element must be positive")
        residency_spaces = [space for space, _ in self.residency]
        if len(residency_spaces) != len(set(residency_spaces)):
            raise ValueError("access residency contains duplicate memory spaces")
        if any(space not in self.path for space in residency_spaces):
            raise ValueError("access residency refers to a space outside its path")
        if any(space in (MemorySpace.GM, MemorySpace.L2) for space in residency_spaces):
            raise ValueError("access residency is only for core-local memory")


@dataclass(frozen=True)
class Primitive:
    """A hardware primitive repeated over a set of iteration axes."""

    resource: Resource
    axes: tuple[str, ...]
    operations_per_point: float = 1.0
    issue_elements: int = 1
    dtype: str | None = None
    padded_axes: tuple[str, ...] = ()
    fixed_cycles: float = 0.0

    def __post_init__(self) -> None:
        if self.operations_per_point < 0.0:
            raise ValueError("operations_per_point must be non-negative")
        if self.issue_elements <= 0:
            raise ValueError("issue_elements must be positive")
        if self.fixed_cycles < 0.0:
            raise ValueError("fixed_cycles must be non-negative")
        if self.dtype is not None:
            dtype_bytes(self.dtype)


@dataclass(frozen=True)
class Stage:
    name: str
    accesses: tuple[Access, ...] = ()
    primitives: tuple[Primitive, ...] = ()
    scope: StageScope = StageScope.TASK
    # Work items inside a stage may be submitted concurrently.  They still
    # compete for a shared resource, which the simulator accounts for.
    concurrent: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("stage name must not be empty")
        if not self.accesses and not self.primitives:
            raise ValueError(f"stage {self.name} has no work")


@dataclass(frozen=True)
class Algorithm:
    """An operator implementation expressed only in hardware semantics."""

    stages: tuple[Stage, ...]
    output_axes: tuple[str, ...]
    reduction_axes: tuple[str, ...] = ()
    core_resource: Resource = Resource.VECTOR
    parallel_reduction: bool = False
    # ``None`` preserves the original IR contract: parallel_reduction=True
    # means a workspace/tree finalization, otherwise a direct result.  New
    # graphs declare the protocol explicitly so serial Split-K and atomic
    # Split-K are not mistaken for deterministic workspace reduction.
    reduction_protocol: ReductionProtocol | None = None
    result_tensor: str | None = None
    partial_dtype: str = "fp32"
    reduction_resource: Resource = Resource.VECTOR
    reduction_operations_per_element: float = 1.0
    pipeline_capable: bool = True
    buffered_spaces: tuple[MemorySpace, ...] = (MemorySpace.UB,)
    # Each tuple is one producer/consumer boundary.  A boundary can overlap
    # adjacent task iterations only when all of its storage spaces are
    # double-buffered.  This is schedule dependency metadata, not a cost fit.
    pipeline_boundaries: tuple[tuple[MemorySpace, ...], ...] = ()
    workspace_buffers: tuple[WorkspaceBuffer, ...] = ()
    # Metadata only.  The simulator never branches on this value.
    name: str = "algorithm"

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("algorithm must contain at least one stage")
        if len(self.output_axes) != len(set(self.output_axes)):
            raise ValueError("algorithm output_axes must be unique")
        if len(self.reduction_axes) != len(set(self.reduction_axes)):
            raise ValueError("algorithm reduction_axes must be unique")
        if set(self.output_axes) & set(self.reduction_axes):
            raise ValueError("output and reduction axes must be disjoint")
        if len(self.buffered_spaces) != len(set(self.buffered_spaces)):
            raise ValueError("algorithm buffered_spaces must be unique")
        if any(
            space in (MemorySpace.GM, MemorySpace.L2)
            for space in self.buffered_spaces
        ):
            raise ValueError("only core-local memory may be explicitly buffered")
        if any(not boundary for boundary in self.pipeline_boundaries):
            raise ValueError("pipeline boundaries must not be empty")
        if any(
            not set(boundary) <= set(self.buffered_spaces)
            for boundary in self.pipeline_boundaries
        ):
            raise ValueError("pipeline boundary uses an undeclared buffer space")
        if self.reduction_operations_per_element < 0.0:
            raise ValueError("reduction_operations_per_element must be non-negative")
        dtype_bytes(self.partial_dtype)

    @property
    def effective_reduction_protocol(self) -> ReductionProtocol:
        if self.reduction_protocol is not None:
            return self.reduction_protocol
        return (
            ReductionProtocol.PARALLEL_WORKSPACE
            if self.parallel_reduction
            else ReductionProtocol.DIRECT
        )

    @property
    def permits_reduction_partitioning(self) -> bool:
        return self.effective_reduction_protocol != ReductionProtocol.DIRECT

    @property
    def distributes_reduction_partitions(self) -> bool:
        return self.effective_reduction_protocol in (
            ReductionProtocol.PARALLEL_WORKSPACE,
            ReductionProtocol.PARALLEL_ATOMIC,
        )


@dataclass(frozen=True)
class Operator:
    axes: tuple[Axis, ...]
    tensors: tuple[Tensor, ...]
    algorithms: tuple[Algorithm, ...]
    # Name is diagnostic metadata and is never passed to the cost equations.
    name: str = "operator"
    attributes: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        axis_names = [axis.name for axis in self.axes]
        tensor_names = [tensor.name for tensor in self.tensors]
        if len(axis_names) != len(set(axis_names)):
            raise ValueError("operator axis names must be unique")
        if len(tensor_names) != len(set(tensor_names)):
            raise ValueError("operator tensor names must be unique")
        if not self.algorithms:
            raise ValueError("operator must define at least one algorithm")
        axes = set(axis_names)
        tensors = set(tensor_names)
        for algorithm in self.algorithms:
            referenced_axes = set(algorithm.output_axes) | set(algorithm.reduction_axes)
            if not referenced_axes <= axes:
                raise ValueError("algorithm references an unknown output/reduction axis")
            if algorithm.result_tensor is not None and algorithm.result_tensor not in tensors:
                raise ValueError("algorithm result_tensor is unknown")
            for workspace in algorithm.workspace_buffers:
                if not {item.axis for item in workspace.dimensions} <= axes:
                    raise ValueError("workspace references an unknown iteration axis")
            for stage in algorithm.stages:
                for access in stage.accesses:
                    if access.tensor not in tensors:
                        raise ValueError(f"stage references unknown tensor {access.tensor}")
                    if not set(access.axes) <= axes:
                        raise ValueError("access references an unknown iteration axis")
                    if not set(access.contiguous_axes) <= set(access.axes):
                        raise ValueError("contiguous_axes must be a subset of access axes")
                for primitive in stage.primitives:
                    if not set(primitive.axes) <= axes:
                        raise ValueError("primitive references an unknown iteration axis")
                    if not set(primitive.padded_axes) <= set(primitive.axes):
                        raise ValueError("padded_axes must be a subset of primitive axes")

    def axis(self, name: str) -> Axis:
        for axis in self.axes:
            if axis.name == name:
                return axis
        raise KeyError(name)

    def tensor(self, name: str) -> Tensor:
        for tensor in self.tensors:
            if tensor.name == name:
                return tensor
        raise KeyError(name)


def axes_product(extents: dict[str, int], axes: Iterable[str]) -> int:
    return prod(extents[name] for name in axes)
