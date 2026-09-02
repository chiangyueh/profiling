"""Declarative examples built from the common operator IR.

These functions contain computation and access semantics, not cost formulas.
The simulator never checks the returned operator's name.
"""

from __future__ import annotations

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
    dtype_bytes,
)


def _power_tiles(alignment: int, maximum: int) -> tuple[int, ...]:
    values: list[int] = []
    value = alignment
    while value <= maximum:
        values.append(value)
        value *= 2
    return tuple(values)


def matmul(
    m: int,
    n: int,
    k: int,
    dtype: str = "fp16",
    *,
    trans_a: bool = False,
    trans_b: bool = False,
) -> Operator:
    """Matrix contraction whose K reduction may be partitioned numerically."""

    in_bytes = dtype_bytes(dtype)
    k0 = 8 if dtype == "fp32" else 16
    a_shape = (k, m) if trans_a else (m, k)
    b_shape = (n, k) if trans_b else (k, n)
    axes = (
        Axis(
            "m", m, AxisKind.PARALLEL, 16, tuple(range(16, 257, 16)),
            independent_task_tiling=True, independent_cache_tiling=True,
        ),
        Axis(
            "n", n, AxisKind.PARALLEL, 16, tuple(range(16, 257, 16)),
            independent_task_tiling=True, independent_cache_tiling=True,
        ),
        Axis(
            "k", k, AxisKind.REDUCTION, k0,
            tuple(range(k0, 257, k0)), False,
        ),
    )
    tensors = (
        Tensor("A", a_shape, dtype),
        Tensor("B", b_shape, dtype),
        Tensor("C", (m, n), dtype),
    )
    stages = (
        Stage(
            "load_inputs",
            accesses=(
                Access(
                    "A", ("m", "k"), AccessMode.READ,
                    (MemorySpace.GM, MemorySpace.L1, MemorySpace.L0A),
                    pattern=(
                        AccessPattern.STRIDED
                        if trans_a else AccessPattern.CONTIGUOUS
                    ),
                    contiguous_axes=(("m",) if trans_a else ("k",)),
                ),
                Access(
                    "B", ("k", "n"), AccessMode.READ,
                    (MemorySpace.GM, MemorySpace.L1, MemorySpace.L0B),
                    pattern=(
                        AccessPattern.STRIDED
                        if trans_b else AccessPattern.CONTIGUOUS
                    ),
                    contiguous_axes=(("k",) if trans_b else ("n",)),
                ),
            ),
            concurrent=True,
        ),
        Stage(
            "matrix_multiply_accumulate",
            primitives=(
                Primitive(
                    Resource.CUBE,
                    ("m", "n", "k"),
                    operations_per_point=1.0,
                    issue_elements=16 * 16 * k0,
                    dtype=dtype,
                    padded_axes=("m", "n", "k"),
                ),
            ),
        ),
        Stage(
            "write_result",
            accesses=(
                Access(
                    "C", ("m", "n"), AccessMode.WRITE,
                    (MemorySpace.L0C, MemorySpace.GM),
                    local_dtype="fp32",
                    service_bytes_per_element=4 + in_bytes,
                    is_result=True,
                ),
            ),
        ),
    )
    algorithm = Algorithm(
        stages=stages,
        output_axes=("m", "n"),
        reduction_axes=("k",),
        core_resource=Resource.CUBE,
        parallel_reduction=True,
        result_tensor="C",
        partial_dtype="fp32",
        reduction_resource=Resource.VECTOR,
        reduction_operations_per_element=1.0,
        pipeline_capable=True,
        buffered_spaces=(
            MemorySpace.L1,
            MemorySpace.L0A,
            MemorySpace.L0B,
            MemorySpace.L0C,
        ),
        pipeline_boundaries=(
            (MemorySpace.L0A, MemorySpace.L0B),
            (MemorySpace.L0C,),
        ),
        name="cube_contraction",
    )
    return Operator(
        axes=axes,
        tensors=tensors,
        algorithms=(algorithm,),
        name="matmul",
        attributes=(("trans_a", str(int(trans_a))), ("trans_b", str(int(trans_b)))),
    )


def elementwise_add(shape: tuple[int, ...], dtype: str = "fp16") -> Operator:
    if not shape:
        raise ValueError("elementwise shape must not be empty")
    lanes = max(1, 256 // (8 * dtype_bytes(dtype)))
    axes = tuple(
        Axis(f"d{index}", extent, tile_values=_power_tiles(1, min(1024, extent)))
        for index, extent in enumerate(shape)
    )
    names = tuple(axis.name for axis in axes)
    tensors = (
        Tensor("A", shape, dtype),
        Tensor("B", shape, dtype),
        Tensor("C", shape, dtype),
    )
    algorithm = Algorithm(
        stages=(
            Stage(
                "load",
                accesses=(
                    Access("A", names, AccessMode.READ, (MemorySpace.GM, MemorySpace.UB)),
                    Access("B", names, AccessMode.READ, (MemorySpace.GM, MemorySpace.UB)),
                ),
                concurrent=True,
            ),
            Stage(
                "add",
                primitives=(Primitive(
                    Resource.VECTOR, names, 1.0, lanes, dtype=dtype
                ),),
            ),
            Stage(
                "store",
                accesses=(Access(
                    "C", names, AccessMode.WRITE,
                    (MemorySpace.UB, MemorySpace.GM), is_result=True,
                ),),
            ),
        ),
        output_axes=names,
        core_resource=Resource.VECTOR,
        result_tensor="C",
        pipeline_capable=True,
        buffered_spaces=(MemorySpace.UB,),
        name="vector_add",
    )
    return Operator(axes, tensors, (algorithm,), name="elementwise_add")


def reduce_sum(
    shape: tuple[int, ...],
    axis: int,
    dtype: str = "fp16",
) -> Operator:
    if not shape:
        raise ValueError("reduce_sum shape must not be empty")
    normalized = axis % len(shape)
    axes = tuple(
        Axis(
            f"d{index}",
            extent,
            AxisKind.REDUCTION if index == normalized else AxisKind.PARALLEL,
            tile_values=_power_tiles(1, min(1024, extent)),
            core_mappable=index != normalized,
        )
        for index, extent in enumerate(shape)
    )
    names = tuple(item.name for item in axes)
    reduction_name = axes[normalized].name
    output_names = tuple(name for name in names if name != reduction_name)
    output_shape = tuple(
        value for index, value in enumerate(shape) if index != normalized
    ) or (1,)
    tensors = (
        Tensor("X", shape, dtype),
        Tensor("Y", output_shape, dtype),
    )
    algorithm = Algorithm(
        stages=(
            Stage(
                "load",
                accesses=(Access(
                    "X", names, AccessMode.READ, (MemorySpace.GM, MemorySpace.UB)
                ),),
            ),
            Stage(
                "reduce",
                primitives=(Primitive(
                    Resource.VECTOR, names, 1.0,
                    max(1, 256 // (8 * dtype_bytes(dtype))), dtype=dtype,
                ),),
            ),
            Stage(
                "store",
                accesses=(Access(
                    "Y", output_names, AccessMode.WRITE,
                    (MemorySpace.UB, MemorySpace.GM), is_result=True,
                ),),
            ),
        ),
        output_axes=output_names,
        reduction_axes=(reduction_name,),
        core_resource=Resource.VECTOR,
        parallel_reduction=True,
        result_tensor="Y",
        partial_dtype="fp32",
        reduction_resource=Resource.VECTOR,
        buffered_spaces=(MemorySpace.UB,),
        name="vector_reduction",
    )
    return Operator(axes, tensors, (algorithm,), name="reduce_sum")


def gather_elements(
    source_shape: tuple[int, ...],
    index_shape: tuple[int, ...],
    axis: int,
    dtype: str = "fp16",
    index_dtype: str = "int32",
    *,
    coalesced_elements: int = 1,
) -> Operator:
    if not source_shape or len(source_shape) != len(index_shape):
        raise ValueError("source and index shapes must have the same non-zero rank")
    normalized = axis % len(source_shape)
    axes = tuple(
        Axis(
            f"d{index}", extent,
            tile_values=_power_tiles(1, min(1024, extent)),
        )
        for index, extent in enumerate(index_shape)
    )
    names = tuple(item.name for item in axes)
    tensors = (
        Tensor("X", source_shape, dtype),
        Tensor("indices", index_shape, index_dtype),
        Tensor("Y", index_shape, dtype),
    )
    algorithm = Algorithm(
        stages=(
            Stage(
                "load_indices_and_values",
                accesses=(
                    Access(
                        "indices", names, AccessMode.READ,
                        (MemorySpace.GM, MemorySpace.UB),
                    ),
                    Access(
                        "X", names, AccessMode.READ,
                        (MemorySpace.GM, MemorySpace.UB),
                        pattern=AccessPattern.INDIRECT,
                        coalesced_elements=coalesced_elements,
                    ),
                ),
                concurrent=True,
            ),
            Stage(
                "index_address",
                primitives=(Primitive(
                    Resource.SCALAR,
                    names,
                    operations_per_point=6.0 + len(source_shape),
                    issue_elements=1,
                    dtype=index_dtype,
                ),),
            ),
            Stage(
                "store",
                accesses=(Access(
                    "Y", names, AccessMode.WRITE,
                    (MemorySpace.UB, MemorySpace.GM), is_result=True,
                ),),
            ),
        ),
        output_axes=names,
        core_resource=Resource.VECTOR,
        result_tensor="Y",
        pipeline_capable=True,
        buffered_spaces=(MemorySpace.UB,),
        name="indexed_read",
    )
    return Operator(
        axes,
        tensors,
        (algorithm,),
        name="gather_elements",
        attributes=(("axis", str(normalized)),),
    )


def scatter_add(
    shape: tuple[int, ...],
    dtype: str = "fp16",
    index_dtype: str = "int32",
    *,
    collision_group: int = 1,
) -> Operator:
    """In-place indexed update expressed through indirect atomic writes."""

    if not shape:
        raise ValueError("scatter shape must not be empty")
    if collision_group <= 0:
        raise ValueError("collision_group must be positive")
    axes = tuple(
        Axis(
            f"d{index}", extent,
            tile_values=_power_tiles(1, min(1024, extent)),
        )
        for index, extent in enumerate(shape)
    )
    names = tuple(item.name for item in axes)
    tensors = (
        Tensor("indices", shape, index_dtype),
        Tensor("updates", shape, dtype),
        Tensor("Y", shape, dtype),
    )
    algorithm = Algorithm(
        stages=(
            Stage(
                "load_indices_and_updates",
                accesses=(
                    Access("indices", names, AccessMode.READ, (MemorySpace.GM, MemorySpace.UB)),
                    Access("updates", names, AccessMode.READ, (MemorySpace.GM, MemorySpace.UB)),
                ),
                concurrent=True,
            ),
            Stage(
                "address_and_add",
                primitives=(Primitive(
                    Resource.SCALAR, names, 8.0, 1, dtype=index_dtype
                ),),
            ),
            Stage(
                "atomic_store",
                accesses=(Access(
                    "Y", names, AccessMode.ATOMIC_ADD,
                    (MemorySpace.UB, MemorySpace.GM),
                    pattern=AccessPattern.INDIRECT,
                    contention_factor=float(collision_group),
                    is_result=True,
                ),),
            ),
        ),
        output_axes=names,
        core_resource=Resource.VECTOR,
        result_tensor="Y",
        pipeline_capable=True,
        buffered_spaces=(MemorySpace.UB,),
        name="indexed_atomic_update",
    )
    return Operator(axes, tensors, (algorithm,), name="scatter_add")


def scatter_elements(
    output_shape: tuple[int, ...],
    index_shape: tuple[int, ...],
    axis: int,
    dtype: str = "fp16",
    index_dtype: str = "int32",
    *,
    reduction: str = "assign",
    collision_group: int = 1,
) -> Operator:
    """Copy an input and apply indirect element updates.

    Dedicated unscheduled axes describe the one-time input initialization;
    update axes describe the independently tiled indexed work.  The common
    simulator therefore sees both traffic phases without recognizing the
    operator name.
    """

    if (
        not output_shape
        or len(output_shape) != len(index_shape)
        or any(value <= 0 for value in output_shape + index_shape)
    ):
        raise ValueError("scatter output/index shapes must have equal non-zero rank")
    if reduction not in ("assign", "add"):
        raise ValueError("scatter reduction must be assign or add")
    if collision_group <= 0:
        raise ValueError("collision_group must be positive")
    normalized = axis % len(output_shape)
    update_axes = tuple(
        Axis(
            f"u{index}", extent,
            tile_values=_power_tiles(1, min(1024, extent)),
        )
        for index, extent in enumerate(index_shape)
    )
    init_axes = tuple(
        Axis(
            f"y{index}", extent,
            tile_values=_power_tiles(1, min(1024, extent)),
            core_mappable=False,
        )
        for index, extent in enumerate(output_shape)
    )
    updates = tuple(axis.name for axis in update_axes)
    initialization = tuple(axis.name for axis in init_axes)
    tensors = (
        Tensor("X", output_shape, dtype),
        Tensor("indices", index_shape, index_dtype),
        Tensor("updates", index_shape, dtype),
        Tensor("Y", output_shape, dtype),
    )
    write_mode = (
        AccessMode.ATOMIC_ADD if reduction == "add" else AccessMode.WRITE
    )
    algorithm = Algorithm(
        stages=(
            Stage(
                "initialize_output",
                accesses=(
                    Access(
                        "X", initialization, AccessMode.READ,
                        (MemorySpace.GM, MemorySpace.UB),
                    ),
                    Access(
                        "Y", initialization, AccessMode.WRITE,
                        (MemorySpace.UB, MemorySpace.GM),
                    ),
                ),
                scope=StageScope.KERNEL,
            ),
            Stage(
                "load_indices_and_updates",
                accesses=(
                    Access(
                        "indices", updates, AccessMode.READ,
                        (MemorySpace.GM, MemorySpace.UB),
                    ),
                    Access(
                        "updates", updates, AccessMode.READ,
                        (MemorySpace.GM, MemorySpace.UB),
                    ),
                ),
                concurrent=True,
            ),
            Stage(
                "address",
                primitives=(Primitive(
                    Resource.SCALAR, updates, 7.0 + len(output_shape), 1,
                    dtype=index_dtype,
                ),),
            ),
            Stage(
                "indirect_update",
                accesses=(Access(
                    "Y", updates, write_mode,
                    (MemorySpace.UB, MemorySpace.GM),
                    pattern=AccessPattern.INDIRECT,
                    contention_factor=float(collision_group),
                    is_result=True,
                ),),
            ),
        ),
        output_axes=updates,
        core_resource=Resource.VECTOR,
        result_tensor="Y",
        pipeline_capable=True,
        buffered_spaces=(MemorySpace.UB,),
        pipeline_boundaries=((MemorySpace.UB,),),
        name="copy_then_indexed_update",
    )
    return Operator(
        update_axes + init_axes,
        tensors,
        (algorithm,),
        name="scatter_elements",
        attributes=(("axis", str(normalized)), ("reduction", reduction)),
    )


def flash_attention_forward(
    batch: int,
    heads: int,
    q_seq: int,
    kv_seq: int,
    head_dim: int,
    dtype: str = "fp16",
) -> Operator:
    """Declarative fused QK-softmax-PV execution graph.

    This is an example lowering for equal Q/KV head counts.  It demonstrates
    that a fused operator is composed from the same memory, Cube and Vector
    resources; it does not introduce an attention-specific cost equation.
    """

    if min(batch, heads, q_seq, kv_seq, head_dim) <= 0:
        raise ValueError("attention dimensions must be positive")
    k0 = 8 if dtype == "fp32" else 16
    axes = (
        Axis("b", batch, tile_values=(1,)),
        Axis("h", heads, tile_values=(1,)),
        Axis(
            "q", q_seq, alignment=16,
            tile_values=_power_tiles(16, min(256, max(16, q_seq))),
            independent_task_tiling=True,
            independent_cache_tiling=True,
        ),
        Axis(
            "s", kv_seq, AxisKind.REDUCTION, 16,
            _power_tiles(16, min(256, max(16, kv_seq))), False,
        ),
        Axis(
            "ki", head_dim, AxisKind.REDUCTION, k0,
            _power_tiles(k0, min(256, max(k0, head_dim))), False,
        ),
        Axis(
            "o", head_dim, alignment=16,
            tile_values=_power_tiles(16, min(256, max(16, head_dim))),
            independent_task_tiling=True,
            independent_cache_tiling=True,
        ),
    )
    tensors = (
        Tensor("Q", (batch, heads, q_seq, head_dim), dtype),
        Tensor("K", (batch, heads, kv_seq, head_dim), dtype),
        Tensor("V", (batch, heads, kv_seq, head_dim), dtype),
        Tensor("O", (batch, heads, q_seq, head_dim), dtype),
    )
    in_bytes = dtype_bytes(dtype)
    algorithm = Algorithm(
        stages=(
            Stage(
                "load_qk",
                accesses=(
                    Access(
                        "Q", ("b", "h", "q", "ki"), AccessMode.READ,
                        (MemorySpace.GM, MemorySpace.L1, MemorySpace.L0A),
                    ),
                    Access(
                        "K", ("b", "h", "s", "ki"), AccessMode.READ,
                        (MemorySpace.GM, MemorySpace.L1, MemorySpace.L0B),
                    ),
                ),
                concurrent=True,
            ),
            Stage(
                "qk_cube",
                primitives=(Primitive(
                    Resource.CUBE,
                    ("b", "h", "q", "s", "ki"),
                    issue_elements=16 * 16 * k0,
                    dtype=dtype,
                    padded_axes=("q", "s", "ki"),
                ),),
            ),
            Stage(
                "online_softmax",
                primitives=(Primitive(
                    Resource.VECTOR,
                    ("b", "h", "q", "s"),
                    operations_per_point=9.0,
                    issue_elements=max(1, 32 // in_bytes),
                    dtype=dtype,
                    padded_axes=("q", "s"),
                ),),
            ),
            Stage(
                "load_v",
                accesses=(Access(
                    "V", ("b", "h", "s", "o"), AccessMode.READ,
                    (MemorySpace.GM, MemorySpace.L1, MemorySpace.L0B),
                ),),
            ),
            Stage(
                "pv_cube",
                primitives=(Primitive(
                    Resource.CUBE,
                    ("b", "h", "q", "s", "o"),
                    issue_elements=16 * 16 * k0,
                    dtype=dtype,
                    padded_axes=("q", "s", "o"),
                ),),
            ),
            Stage(
                "write_output",
                accesses=(Access(
                    "O", ("b", "h", "q", "o"), AccessMode.WRITE,
                    (MemorySpace.L0C, MemorySpace.GM),
                    local_dtype="fp32",
                    service_bytes_per_element=4 + in_bytes,
                    is_result=True,
                ),),
            ),
        ),
        output_axes=("b", "h", "q", "o"),
        reduction_axes=("s", "ki"),
        core_resource=Resource.CUBE,
        parallel_reduction=False,
        result_tensor="O",
        pipeline_capable=True,
        buffered_spaces=(
            MemorySpace.L1,
            MemorySpace.L0A,
            MemorySpace.L0B,
            MemorySpace.L0C,
        ),
        pipeline_boundaries=(
            (MemorySpace.L0A, MemorySpace.L0B),
            (MemorySpace.L0C,),
        ),
        name="fused_qk_softmax_pv",
    )
    return Operator(axes, tensors, (algorithm,), name="flash_attention_forward")


def flash_attention_score_grad(
    batch: int,
    q_heads: int,
    kv_heads: int,
    q_seq: int,
    kv_seq: int,
    head_dim: int,
    dtype: str = "fp16",
) -> Operator:
    """Declarative FlashAttention score-backward hardware graph.

    Q heads are represented as ``kv_head × group`` so GQA/MQA reuse and the
    dK/dV cross-group accumulation are explicit hardware semantics.
    """

    if min(batch, q_heads, kv_heads, q_seq, kv_seq, head_dim) <= 0:
        raise ValueError("attention-gradient dimensions must be positive")
    if q_heads % kv_heads:
        raise ValueError("q_heads must be divisible by kv_heads")
    group = q_heads // kv_heads
    k0 = 8 if dtype == "fp32" else 16
    axes = (
        Axis("b", batch, tile_values=(1,)),
        Axis("hk", kv_heads, tile_values=(1,)),
        Axis("g", group, tile_values=(1,)),
        Axis(
            "q", q_seq, alignment=16,
            tile_values=_power_tiles(16, min(256, max(16, q_seq))),
            independent_task_tiling=True,
            independent_cache_tiling=True,
        ),
        Axis(
            "s", kv_seq, AxisKind.REDUCTION, 16,
            _power_tiles(16, min(256, max(16, kv_seq))), False,
        ),
        Axis(
            "di", head_dim, AxisKind.REDUCTION, k0,
            _power_tiles(k0, min(256, max(k0, head_dim))), False,
        ),
        Axis(
            "do", head_dim, alignment=16,
            tile_values=(((head_dim + 15) // 16) * 16,),
        ),
    )
    q_shape = (batch, kv_heads, group, q_seq, head_dim)
    kv_shape = (batch, kv_heads, kv_seq, head_dim)
    tensors = (
        Tensor("Q", q_shape, dtype),
        Tensor("K", kv_shape, dtype),
        Tensor("V", kv_shape, dtype),
        Tensor("dO", q_shape, dtype),
        Tensor("dQ", q_shape, dtype),
        Tensor("dK", kv_shape, dtype),
        Tensor("dV", kv_shape, dtype),
    )
    in_bytes = dtype_bytes(dtype)
    algorithm = Algorithm(
        stages=(
            Stage(
                "load_backward_operands",
                accesses=(
                    Access(
                        "Q", ("b", "hk", "g", "q", "di"), AccessMode.READ,
                        (MemorySpace.GM, MemorySpace.L1, MemorySpace.L0A),
                    ),
                    Access(
                        "dO", ("b", "hk", "g", "q", "di"), AccessMode.READ,
                        (MemorySpace.GM, MemorySpace.L1, MemorySpace.L0A),
                    ),
                    Access(
                        "K", ("b", "hk", "s", "di"), AccessMode.READ,
                        (MemorySpace.GM, MemorySpace.L1, MemorySpace.L0B),
                    ),
                    Access(
                        "V", ("b", "hk", "s", "di"), AccessMode.READ,
                        (MemorySpace.GM, MemorySpace.L1, MemorySpace.L0B),
                    ),
                ),
                concurrent=True,
            ),
            Stage(
                "backward_cube_contractions",
                primitives=(Primitive(
                    Resource.CUBE,
                    ("b", "hk", "g", "q", "s", "di"),
                    operations_per_point=4.0,
                    issue_elements=16 * 16 * k0,
                    dtype=dtype,
                    padded_axes=("q", "s", "di"),
                ),),
            ),
            Stage(
                "softmax_backward_vector",
                primitives=(Primitive(
                    Resource.VECTOR,
                    ("b", "hk", "g", "q", "s"),
                    operations_per_point=6.0,
                    issue_elements=max(1, 32 // in_bytes),
                    dtype=dtype,
                    padded_axes=("q", "s"),
                ),),
            ),
            Stage(
                "write_gradients",
                accesses=(
                    Access(
                        "dQ", ("b", "hk", "g", "q", "do"),
                        AccessMode.WRITE, (MemorySpace.L0C, MemorySpace.GM),
                        local_dtype="fp32",
                        service_bytes_per_element=4 + in_bytes,
                        is_result=True,
                    ),
                    Access(
                        "dK", ("b", "hk", "s", "do"),
                        AccessMode.ATOMIC_ADD,
                        (MemorySpace.L0C, MemorySpace.GM),
                        local_dtype="fp32",
                    ),
                    Access(
                        "dV", ("b", "hk", "s", "do"),
                        AccessMode.ATOMIC_ADD,
                        (MemorySpace.L0C, MemorySpace.GM),
                        local_dtype="fp32",
                    ),
                ),
                concurrent=True,
            ),
        ),
        output_axes=("b", "hk", "g", "q", "do"),
        reduction_axes=("s", "di"),
        core_resource=Resource.CUBE,
        parallel_reduction=False,
        result_tensor="dQ",
        pipeline_capable=True,
        buffered_spaces=(
            MemorySpace.L1,
            MemorySpace.L0A,
            MemorySpace.L0B,
            MemorySpace.L0C,
        ),
        pipeline_boundaries=(
            (MemorySpace.L0A, MemorySpace.L0B),
            (MemorySpace.L0C,),
        ),
        name="attention_score_backward",
    )
    return Operator(axes, tensors, (algorithm,), name="flash_attention_score_grad")
