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
    ReductionProtocol,
    Resource,
    Stage,
    StageScope,
    Tensor,
    TileLevel,
    WorkspaceBuffer,
    WorkspaceDimension,
    dtype_bytes,
)
from .matmul_layout import LayoutConversion, source_layout_conversion


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
    a_layout: str = "ND",
    b_layout: str = "ND",
    output_dtype: str | None = None,
    has_bias: bool = False,
) -> Operator:
    """C220 matrix contraction expressed as its complete dataflow set.

    Kernel-family names below are diagnostic metadata.  The simulator sees
    only memory paths, residency levels, primitives and reduction protocols.
    Frontends may therefore describe another contraction implementation with
    the same building blocks without adding a cost-model branch.
    """

    in_bytes = dtype_bytes(dtype)
    out_dtype = output_dtype or dtype
    # TCube's FP32 K0=8 format is available only for ND x transposed-ND.
    # Every other input-layout combination is lowered with K0=16.  Keeping
    # this in the operator-to-hardware lowering prevents the generic solver
    # from projecting an ideal region that the backend ABI later rejects.
    k0 = 8 if dtype == "fp32" and not trans_a and trans_b else 16
    a_shape = (k, m) if trans_a else (m, k)
    b_shape = (n, k) if trans_b else (k, n)
    axes = (
        Axis(
            "m", m, AxisKind.PARALLEL, 16,
            independent_task_tiling=True, independent_cache_tiling=True,
        ),
        Axis(
            "n", n, AxisKind.PARALLEL, 16,
            independent_task_tiling=True, independent_cache_tiling=True,
        ),
        Axis(
            "k", k, AxisKind.REDUCTION, k0,
            independent_task_tiling=False,
        ),
    )
    tensors = [
        Tensor("A", a_shape, dtype),
        Tensor("B", b_shape, dtype),
        Tensor("C", (m, n), out_dtype),
        # Logical GM temporary used only by the FixPipe/NZ2ND graphs.  Its
        # allocated padding is declared separately by WorkspaceBuffer; its
        # accesses let the generic simulator count actual route traffic and
        # copy requests rather than mistaking reserved ping-pong capacity for
        # bytes transferred.
        Tensor("OutputWorkspace", (m, n), out_dtype),
    ]
    if has_bias:
        tensors.append(Tensor("Bias", (n,), out_dtype))

    a_pattern = AccessPattern.STRIDED if trans_a else AccessPattern.CONTIGUOUS
    b_pattern = AccessPattern.STRIDED if trans_b else AccessPattern.CONTIGUOUS
    a_contiguous = ("m",) if trans_a else ("k",)
    b_contiguous = ("k",) if trans_b else ("n",)
    conversions = {
        name: source_layout_conversion(
            m, n, k, dtype, trans_a, trans_b,
            a_layout=a_layout,
            b_layout=b_layout,
            graph_name=name,
        )
        for name in (
            "base",
            "single_core_split_k",
            "deterministic_split_k",
            "al1_full_load",
            "bl1_full_load",
            "bl1_full_load_fixpipe",
            "bl1_full_load_vec_nz2nd",
        )
    }

    def input_accesses(
        *,
        resident_a: bool = False,
        resident_b: bool = False,
        l1_only: bool = False,
    ) -> tuple[Access, ...]:
        result: list[Access] = []
        if not resident_a:
            result.append(Access(
                "A", ("m", "k"), AccessMode.READ,
                (
                    (MemorySpace.L1, MemorySpace.L0A)
                    if l1_only
                    else (MemorySpace.GM, MemorySpace.L1, MemorySpace.L0A)
                ),
                pattern=a_pattern,
                contiguous_axes=a_contiguous,
                dependency_axes=("k",),
            ))
        else:
            result.append(Access(
                "A", ("m", "k"), AccessMode.READ,
                (MemorySpace.L1, MemorySpace.L0A),
                pattern=a_pattern,
                contiguous_axes=a_contiguous,
                dependency_axes=("k",),
            ))
        if not resident_b:
            result.append(Access(
                "B", ("k", "n"), AccessMode.READ,
                (
                    (MemorySpace.L1, MemorySpace.L0B)
                    if l1_only
                    else (MemorySpace.GM, MemorySpace.L1, MemorySpace.L0B)
                ),
                pattern=b_pattern,
                contiguous_axes=b_contiguous,
                dependency_axes=("k",),
            ))
        else:
            result.append(Access(
                "B", ("k", "n"), AccessMode.READ,
                (MemorySpace.L1, MemorySpace.L0B),
                pattern=b_pattern,
                contiguous_axes=b_contiguous,
                dependency_axes=("k",),
            ))
        if has_bias:
            result.append(Access(
                "Bias", ("n",), AccessMode.READ,
                (MemorySpace.GM, MemorySpace.L1),
            ))
        return tuple(result)

    def conversion_stages(conversion: LayoutConversion) -> tuple[Stage, ...]:
        lanes = max(1, 256 // (8 * in_bytes))
        stages: list[Stage] = []
        for tensor, access_axes, enabled in (
            ("A", ("m", "k"), conversion.a),
            ("B", ("k", "n"), conversion.b),
        ):
            if not enabled:
                continue
            stages.append(Stage(
                f"stream_{tensor.lower()}_nd_to_nz",
                accesses=(
                    Access(tensor, access_axes, AccessMode.READ,
                           (MemorySpace.GM, MemorySpace.UB)),
                    Access(tensor, access_axes, AccessMode.WRITE,
                           (MemorySpace.UB, MemorySpace.GM)),
                ),
                primitives=(Primitive(
                    Resource.VECTOR, access_axes,
                    operations_per_point=1.0,
                    issue_elements=lanes,
                    dtype=dtype,
                ),),
                scope=StageScope.KERNEL,
                concurrent=True,
            ))
        return tuple(stages)

    def output_stages(
        *,
        atomic: bool = False,
        vector_output: bool = False,
        workspace_alignment: int = 1,
        invocation_axes: tuple[str, ...] = (),
        repeated_atomic_axis: str | None = None,
    ) -> tuple[Stage, ...]:
        if vector_output:
            return (
                Stage(
                    "write_output_workspace",
                    accesses=(Access(
                        "OutputWorkspace", ("m", "n"), AccessMode.WRITE,
                        (MemorySpace.L0C, MemorySpace.GM),
                        transaction_bytes=(
                            workspace_alignment * dtype_bytes(out_dtype)
                        ),
                        local_dtype="fp32",
                        service_bytes_per_element=4 + dtype_bytes(out_dtype),
                    ),),
                    invocation_axes=invocation_axes,
                ),
                Stage(
                    "vector_layout_and_cast",
                    accesses=(
                        Access(
                            "OutputWorkspace", ("m", "n"), AccessMode.READ,
                            (MemorySpace.GM, MemorySpace.UB),
                            transaction_bytes=(
                                workspace_alignment * dtype_bytes(out_dtype)
                            ),
                        ),
                        Access(
                            "C", ("m", "n"), AccessMode.WRITE,
                            (MemorySpace.UB, MemorySpace.GM),
                            is_result=True,
                        ),
                    ),
                    primitives=(Primitive(
                        Resource.VECTOR,
                        ("m", "n"),
                        operations_per_point=1.0,
                        issue_elements=max(1, 256 // (8 * dtype_bytes(out_dtype))),
                        dtype=out_dtype,
                    ),),
                    invocation_axes=invocation_axes,
                ),
            )
        return (Stage(
            "write_result",
            accesses=(Access(
                "C", ("m", "n"),
                AccessMode.ATOMIC_ADD
                if atomic or repeated_atomic_axis is not None
                else AccessMode.WRITE,
                (MemorySpace.L0C, MemorySpace.GM),
                local_dtype="fp32",
                service_bytes_per_element=4 + dtype_bytes(out_dtype),
                is_result=True,
                first_iteration_mode=(
                    AccessMode.WRITE
                    if repeated_atomic_axis is not None else None
                ),
                mode_switch_axis=repeated_atomic_axis,
            ),),
            invocation_axes=invocation_axes,
        ),)

    def graph(
        name: str,
        protocol: ReductionProtocol,
        *,
        resident_a: bool = False,
        resident_b: bool = False,
        staged_gm_to_l1: bool = False,
        vector_output: bool = False,
        workspace_alignment: int = 1,
        atomic: bool = False,
        workspace_buffers: tuple[WorkspaceBuffer, ...] = (),
        coupled_task_axes: tuple[str, ...] = (),
    ) -> Algorithm:
        conversion = conversions[name]
        serial_reduction = protocol in (
            ReductionProtocol.SERIAL_DIRECT,
            ReductionProtocol.SERIAL_WORKSPACE,
        )
        invocation_axes = ("m", "n", "k") if serial_reduction else ()
        stages: list[Stage] = list(conversion_stages(conversion))
        if atomic:
            stages.append(Stage(
                "clear_atomic_destination",
                accesses=(Access(
                    "C", ("m", "n"), AccessMode.WRITE,
                    (MemorySpace.UB, MemorySpace.GM),
                ),),
                scope=StageScope.KERNEL,
            ))
        if resident_a:
            stages.append(Stage(
                "load_resident_a",
                accesses=(Access(
                    "A", ("m", "k"), AccessMode.READ,
                    (MemorySpace.GM, MemorySpace.L1),
                    pattern=a_pattern,
                    contiguous_axes=a_contiguous,
                    residency=((MemorySpace.L1, TileLevel.TASK),),
                ),),
                scope=StageScope.CORE,
            ))
        if resident_b:
            stages.append(Stage(
                "load_resident_b",
                accesses=(Access(
                    "B", ("k", "n"), AccessMode.READ,
                    (MemorySpace.GM, MemorySpace.L1),
                    pattern=b_pattern,
                    contiguous_axes=b_contiguous,
                    residency=((MemorySpace.L1, TileLevel.TASK),),
                ),),
                scope=StageScope.CORE,
            ))
        if staged_gm_to_l1:
            stages.extend((
                Stage(
                    "stage_inputs_to_l1",
                    accesses=(
                        Access("A", ("m", "k"), AccessMode.READ,
                               (MemorySpace.GM, MemorySpace.L1),
                               pattern=a_pattern, contiguous_axes=a_contiguous,
                               dependency_axes=("k",)),
                        Access("B", ("k", "n"), AccessMode.READ,
                               (MemorySpace.GM, MemorySpace.L1),
                               pattern=b_pattern, contiguous_axes=b_contiguous,
                               dependency_axes=("k",)),
                    ),
                    concurrent=True,
                    invocation_axes=invocation_axes,
                ),
                Stage(
                    "stage_l1_to_l0",
                    accesses=input_accesses(l1_only=True),
                    concurrent=True,
                    invocation_axes=invocation_axes,
                ),
            ))
        else:
            stages.append(Stage(
                "load_inputs",
                accesses=input_accesses(
                    resident_a=resident_a,
                    resident_b=resident_b,
                ),
                concurrent=True,
                invocation_axes=invocation_axes,
            ))
        stages.append(Stage(
            "matrix_multiply_accumulate",
            primitives=(Primitive(
                Resource.CUBE,
                ("m", "n", "k"),
                operations_per_point=1.0,
                issue_elements=16 * 16 * k0,
                dtype=dtype,
                padded_axes=("m", "n", "k"),
                command_axes=("m", "n", "k"),
            ),),
            invocation_axes=invocation_axes,
        ))
        stages.extend(output_stages(
            atomic=atomic,
            vector_output=vector_output,
            workspace_alignment=workspace_alignment,
            invocation_axes=invocation_axes,
            repeated_atomic_axis="k" if serial_reduction else None,
        ))
        buffered = [MemorySpace.L1, MemorySpace.L0A,
                    MemorySpace.L0B, MemorySpace.L0C]
        if conversion.a or conversion.b or vector_output:
            buffered.append(MemorySpace.UB)
        return Algorithm(
            stages=tuple(stages),
            output_axes=("m", "n"),
            reduction_axes=("k",),
            core_resource=Resource.CUBE,
            parallel_reduction=(
                protocol in (
                    ReductionProtocol.PARALLEL_WORKSPACE,
                    ReductionProtocol.PARALLEL_ATOMIC,
                )
            ),
            reduction_protocol=protocol,
            result_tensor="C",
            partial_dtype="fp32",
            reduction_resource=Resource.VECTOR,
            reduction_operations_per_element=1.0,
            pipeline_capable=True,
            buffered_spaces=tuple(buffered),
            pipeline_boundaries=((MemorySpace.L0A, MemorySpace.L0B),),
            workspace_buffers=workspace_buffers,
            coupled_task_axes=coupled_task_axes,
            name=name,
        )

    system_workspace = WorkspaceBuffer(fixed_bytes=20 * 1024 * 1024)
    def conversion_workspace(
        conversion: LayoutConversion,
    ) -> tuple[WorkspaceBuffer, ...]:
        result: list[WorkspaceBuffer] = []
        if conversion.a:
            result.append(WorkspaceBuffer(
            dimensions=(
                WorkspaceDimension("m", alignment=(32 // in_bytes if trans_a else 16)),
                WorkspaceDimension("k", alignment=(16 if trans_a else 32 // in_bytes)),
            ),
            dtype=dtype,
            ))
        if conversion.b:
            result.append(WorkspaceBuffer(
            dimensions=(
                WorkspaceDimension("k", alignment=(32 // in_bytes if trans_b else 16)),
                WorkspaceDimension("n", alignment=(16 if trans_b else 32 // in_bytes)),
            ),
            dtype=dtype,
            ))
        return tuple(result)

    common_workspace = (
        system_workspace, *conversion_workspace(conversions["base"])
    )
    serial_workspace = (
        system_workspace,
        *conversion_workspace(conversions["single_core_split_k"]),
        WorkspaceBuffer(
        dimensions=(
            WorkspaceDimension("m"),
            WorkspaceDimension("n", alignment=256 // in_bytes),
        ),
        dtype="fp32",
        ),
    )
    deterministic_workspace = (
        system_workspace,
        *conversion_workspace(conversions["deterministic_split_k"]),
        WorkspaceBuffer(
        dimensions=(
            WorkspaceDimension("m", TileLevel.TASK, clamp_to_axis=False),
            WorkspaceDimension("n", TileLevel.TASK, clamp_to_axis=False),
        ),
        dtype="fp32",
        copies=2,
        per_active_core=True,
        ),
    )
    fixpipe_prefix = (
        system_workspace,
        *conversion_workspace(conversions["bl1_full_load_fixpipe"]),
    )
    fixpipe_workspace = (*fixpipe_prefix, WorkspaceBuffer(
        dimensions=(
            WorkspaceDimension("m", TileLevel.INNER),
            WorkspaceDimension("n", TileLevel.KERNEL, 512 // dtype_bytes(out_dtype)),
        ),
        dtype=out_dtype,
        copies=2,
        per_active_core=True,
    ))
    nz2nd_workspace = (
        system_workspace,
        *conversion_workspace(conversions["bl1_full_load_vec_nz2nd"]),
        WorkspaceBuffer(
        dimensions=(
            WorkspaceDimension("m", TileLevel.INNER),
            WorkspaceDimension("n", TileLevel.KERNEL, 16),
        ),
        dtype=out_dtype,
        copies=2,
        per_active_core=True,
        ),
    )

    # These are exactly the seven execution families dispatched by the
    # pinned CANN 8.1 Ascend910B3 mat_mul_v3.cpp.  Aligned/unaligned suffixes
    # are layout-conversion variants of the same dataflow and are selected at
    # the ABI boundary, not separate cost functions.
    algorithms: list[Algorithm] = [
        graph(
            "base", ReductionProtocol.DIRECT,
            workspace_buffers=common_workspace,
            coupled_task_axes=("m", "n"),
        ),
        graph(
              "single_core_split_k",
              ReductionProtocol.SERIAL_DIRECT if out_dtype == "fp32" else ReductionProtocol.SERIAL_WORKSPACE,
              workspace_buffers=serial_workspace),
        graph("deterministic_split_k", ReductionProtocol.PARALLEL_WORKSPACE,
              workspace_buffers=deterministic_workspace),
        graph("bl1_full_load", ReductionProtocol.DIRECT, resident_b=True,
              workspace_buffers=(
                  system_workspace,
                  *conversion_workspace(conversions["bl1_full_load"]),
              )),
        graph(
            "bl1_full_load_fixpipe",
            ReductionProtocol.DIRECT,
            resident_b=True,
            vector_output=True,
            workspace_alignment=512 // dtype_bytes(out_dtype),
            workspace_buffers=fixpipe_workspace,
        ),
    ]
    # AL1 full load is the sole FP32-NT-only family in the CANN 8.1 source.
    if dtype == "fp32" and not trans_a and trans_b:
        algorithms.append(
            graph(
                "al1_full_load", ReductionProtocol.DIRECT, resident_a=True,
                workspace_buffers=(
                    system_workspace,
                    *conversion_workspace(conversions["al1_full_load"]),
                ),
            )
        )
    # The vector NZ2ND variant is selected from FixPipe by actual conversion
    # state, K alignment, and N extent; transpose is not part of its source
    # predicate.
    vector_conversion = conversions["bl1_full_load_vec_nz2nd"]
    if (
        dtype == "fp32"
        and not vector_conversion.a
        and k % 8 == 0
        and n <= 192
    ):
        algorithms.append(
            graph(
                "bl1_full_load_vec_nz2nd",
                ReductionProtocol.DIRECT,
                resident_b=True,
                vector_output=True,
                workspace_alignment=16,
                workspace_buffers=nz2nd_workspace,
            )
        )

    return Operator(
        axes=axes,
        tensors=tuple(tensors),
        algorithms=tuple(algorithms),
        name="matmul",
        attributes=(
            ("trans_a", str(int(trans_a))),
            ("trans_b", str(int(trans_b))),
            ("a_layout", a_layout),
            ("b_layout", b_layout),
            ("output_dtype", out_dtype),
            ("has_bias", str(int(has_bias))),
        ),
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
