#!/usr/bin/env python3
from __future__ import annotations

import inspect
from dataclasses import replace

from npu_cost_model import (
    Access,
    AccessMode,
    Algorithm,
    Axis,
    MemorySpace,
    Operator,
    Resource,
    ScheduleSpace,
    SearchPolicy,
    Stage,
    Tensor,
    TilingPlan,
    ascend_910b3,
    derive_ideal_region,
    generate_plans,
    simulate,
    solve,
    solve_ideal_region,
)
from npu_cost_model.operators import (
    elementwise_add,
    flash_attention_forward,
    flash_attention_score_grad,
    gather_elements,
    matmul,
    reduce_sum,
    scatter_elements,
)


HARDWARE = ascend_910b3()


def _copy_operator(name: str = "user_defined_copy") -> Operator:
    axis = Axis("x", 257, tile_values=(16, 32, 64, 128))
    algorithm = Algorithm(
        stages=(
            Stage(
                "read",
                accesses=(
                    Access(
                        "X",
                        ("x",),
                        AccessMode.READ,
                        (MemorySpace.GM, MemorySpace.UB),
                    ),
                ),
            ),
            Stage(
                "write",
                accesses=(
                    Access(
                        "Y",
                        ("x",),
                        AccessMode.WRITE,
                        (MemorySpace.UB, MemorySpace.GM),
                        is_result=True,
                    ),
                ),
            ),
        ),
        output_axes=("x",),
        core_resource=Resource.VECTOR,
        result_tensor="Y",
        buffered_spaces=(MemorySpace.UB,),
        name="arbitrary_frontend_lowering",
    )
    return Operator(
        axes=(axis,),
        tensors=(Tensor("X", (257,), "fp16"), Tensor("Y", (257,), "fp16")),
        algorithms=(algorithm,),
        name=name,
    )


def test_a_new_operator_can_be_defined_only_with_the_public_ir() -> None:
    result = solve(
        _copy_operator(),
        HARDWARE,
        ScheduleSpace(
            core_options=(1, 8, 40),
            buffer_options=((1,), (2,)),
            traversal_options=(("x",),),
        ),
        SearchPolicy(top_k=4, max_evaluations=1000),
    )
    assert result.exhaustive
    assert result.legal > 1
    assert result.ranked
    assert [item.cycles for item in result.ranked] == sorted(
        item.cycles for item in result.ranked
    )


def test_cost_is_invariant_to_operator_and_algorithm_names() -> None:
    original = _copy_operator("first_name")
    renamed_algorithm = replace(original.algorithms[0], name="unrelated_name")
    renamed = replace(
        original,
        algorithms=(renamed_algorithm,),
        name="second_name",
    )
    plan = TilingPlan(
        algorithm=0,
        axis_tiles=(("x", 64),),
        used_cores=8,
        buffers=((MemorySpace.UB, 2),),
        traversal=("x",),
    )
    first = simulate(original, plan, HARDWARE)
    second = simulate(renamed, plan, HARDWARE)
    assert first.valid and second.valid
    assert first == second


def test_generic_legality_rejects_bad_alignment_capacity_and_traversal() -> None:
    operator = matmul(128, 128, 512)
    base = dict(
        algorithm=0,
        axis_tiles=(("m", 32), ("n", 64), ("k", 64)),
        used_cores=20,
        reduction_parts=(("k", 1),),
        buffers=(),
        traversal=("m", "n"),
    )
    assert simulate(operator, TilingPlan(**base), HARDWARE).valid

    misaligned = dict(base)
    misaligned["axis_tiles"] = (("m", 24), ("n", 64), ("k", 64))
    assert "alignment" in simulate(
        operator, TilingPlan(**misaligned), HARDWARE
    ).error

    bad_traversal = dict(base)
    bad_traversal["traversal"] = ("m", "k")
    assert "permutation" in simulate(
        operator, TilingPlan(**bad_traversal), HARDWARE
    ).error

    huge = elementwise_add((131072,), "fp16")
    huge_plan = TilingPlan(
        algorithm=0,
        axis_tiles=(("d0", 131072),),
        used_cores=1,
        buffers=((MemorySpace.UB, 2),),
        traversal=("d0",),
    )
    capacity = simulate(huge, huge_plan, HARDWARE)
    assert not capacity.valid
    assert "ub capacity exceeded" in capacity.error


def test_parallel_reduction_is_a_numeric_schedule_choice() -> None:
    operator = matmul(16, 16, 16384)
    common = dict(
        algorithm=0,
        axis_tiles=(("m", 16), ("n", 16), ("k", 128)),
        used_cores=20,
        buffers=(),
        traversal=("m", "n"),
    )
    serial = simulate(
        operator,
        TilingPlan(**common, reduction_parts=(("k", 1),)),
        HARDWARE,
    )
    partitioned = simulate(
        operator,
        TilingPlan(**common, reduction_parts=(("k", 20),)),
        HARDWARE,
    )
    assert serial.valid and partitioned.valid
    assert serial.active_cores == 1
    assert partitioned.active_cores == 20
    assert serial.workspace_bytes == 0
    assert partitioned.workspace_bytes == 20 * 16 * 16 * 4
    assert partitioned.total_cycles < serial.total_cycles


def test_same_shape_different_semantics_take_different_hardware_paths() -> None:
    shape = (64, 64)
    direct = elementwise_add(shape)
    indirect = gather_elements(shape, shape, axis=1)
    plan = TilingPlan(
        algorithm=0,
        axis_tiles=(("d0", 16), ("d1", 64)),
        used_cores=20,
        buffers=((MemorySpace.UB, 1),),
        traversal=("d0", "d1"),
    )
    direct_cost = simulate(direct, plan, HARDWARE)
    indirect_cost = simulate(indirect, plan, HARDWARE)
    assert direct_cost.valid and indirect_cost.valid
    assert indirect_cost.total_cycles > direct_cost.total_cycles
    assert dict(indirect_cost.resource_cycles)[Resource.SCALAR] > 0.0


def test_generated_matmul_plans_are_aligned_and_solver_returns_legal_top_k() -> None:
    operator = matmul(96, 160, 1024)
    space = ScheduleSpace(
        tile_options=(
            ("m", (16, 32, 48)),
            ("n", (16, 64, 128)),
            ("k", (16, 64, 128)),
        ),
        task_tile_options=(
            ("m", (48, 96)),
            ("n", (80, 160)),
        ),
        cache_tile_options=(("m", (96,)), ("n", (160,))),
        core_options=(1, 8, 20),
        reduction_options=(("k", (1, 4, 20)),),
        buffer_options=((1, 1, 1, 1), (2, 2, 2, 1)),
        traversal_options=(("m", "n"), ("n", "m")),
    )
    policy = SearchPolicy(top_k=8, max_evaluations=10000)
    plans = list(generate_plans(operator, HARDWARE, space, policy))
    assert plans
    assert all(
        plan.tiles[axis.name] % axis.alignment == 0
        for plan in plans
        for axis in operator.axes
    )
    result = solve(operator, HARDWARE, space, policy)
    assert result.exhaustive
    assert len(result.ranked) == 8
    assert result.legal + result.rejected == result.evaluated
    assert all(simulate(operator, item.plan, HARDWARE).valid for item in result.ranked)


def test_public_solver_has_no_history_or_callback_input() -> None:
    for function in (
        generate_plans, derive_ideal_region, solve, solve_ideal_region, simulate
    ):
        parameters = inspect.signature(function).parameters
        assert "history" not in parameters
        assert "callback" not in parameters
        assert "runtime_kb" not in parameters


def test_capped_search_covers_every_hardware_dimension_not_a_prefix() -> None:
    plans = list(
        generate_plans(
            matmul(1024, 1024, 4096),
            HARDWARE,
            policy=SearchPolicy(top_k=2, max_evaluations=200),
        )
    )
    assert len(plans) == len(set(plans)) == 200
    assert len({plan.tiles["m"] for plan in plans}) > 2
    assert len({plan.tasks["m"] for plan in plans}) > 2
    assert len({plan.caches["m"] for plan in plans}) > 2
    assert {plan.traversal for plan in plans} == {("m", "n"), ("n", "m")}
    assert {plan.used_cores for plan in plans} >= {1, 20}
    assert {plan.reductions["k"] for plan in plans} >= {1, 20}
    assert len({plan.buffers for plan in plans}) > 2


def test_ideal_region_is_local_bounded_and_uses_every_declared_algorithm() -> None:
    original = _copy_operator()
    second = replace(original.algorithms[0], name="second_execution_graph")
    operator = replace(original, algorithms=(original.algorithms[0], second))
    region = derive_ideal_region(operator, HARDWARE)
    assert region.exhaustive
    assert region.plans
    assert len(region.plans) < 100
    assert dict(region.algorithm_anchor_counts).keys() == {0, 1}
    assert dict(region.algorithm_region_counts).keys() == {0, 1}
    assert all(simulate(operator, plan, HARDWARE).valid for plan in region.plans)


def test_reduction_schedule_is_discovered_without_a_named_splitk_path() -> None:
    operator = matmul(128, 128, 16384)
    region = derive_ideal_region(operator, HARDWARE)
    partitions = {plan.reductions["k"] for plan in region.plans}
    assert 1 in partitions
    assert max(partitions) > 1
    renamed = replace(
        operator,
        algorithms=(replace(operator.algorithms[0], name="renamed"),),
        name="renamed_operator",
    )
    renamed_region = derive_ideal_region(renamed, HARDWARE)
    assert region.plans == renamed_region.plans


def test_l2_traversal_and_pipeline_boundaries_change_cycles() -> None:
    operator = matmul(512, 128, 1024)
    common = dict(
        algorithm=0,
        axis_tiles=(("m", 64), ("n", 64), ("k", 64)),
        task_tiles=(("m", 128), ("n", 128), ("k", 1024)),
        cache_tiles=(("m", 512), ("n", 128), ("k", 1024)),
        used_cores=2,
        reduction_parts=(("k", 1),),
    )
    row_first = simulate(
        operator,
        TilingPlan(
            **common,
            buffers=((MemorySpace.L0A, 1), (MemorySpace.L0B, 1), (MemorySpace.L0C, 1)),
            traversal=("m", "n"),
        ),
        HARDWARE,
    )
    column_first = simulate(
        operator,
        TilingPlan(
            **common,
            buffers=((MemorySpace.L0A, 1), (MemorySpace.L0B, 1), (MemorySpace.L0C, 1)),
            traversal=("n", "m"),
        ),
        HARDWARE,
    )
    pipelined = simulate(
        operator,
        TilingPlan(
            **common,
            buffers=((MemorySpace.L0A, 2), (MemorySpace.L0B, 2), (MemorySpace.L0C, 2)),
            traversal=("n", "m"),
        ),
        HARDWARE,
    )
    assert row_first.valid and column_first.valid and pipelined.valid
    assert row_first.total_cycles != column_first.total_cycles
    assert pipelined.total_cycles < column_first.total_cycles


def test_fused_attention_uses_the_same_generic_solver() -> None:
    operator = flash_attention_forward(1, 8, 128, 256, 64)
    space = ScheduleSpace(
        tile_options=(
            ("b", (1,)),
            ("h", (1,)),
            ("q", (32, 64)),
            ("s", (32, 64)),
            ("ki", (16, 32)),
            ("o", (32, 64)),
        ),
        task_tile_options=(("q", (64, 128)), ("o", (64,))),
        cache_tile_options=(("q", (128,)), ("o", (64,))),
        core_options=(1, 8, 20),
        buffer_options=((1, 1, 1, 1), (1, 2, 2, 2)),
        traversal_options=(("b", "h", "q", "o"),),
    )
    result = solve(
        operator,
        HARDWARE,
        space,
        SearchPolicy(top_k=4, max_evaluations=10000),
    )
    assert result.exhaustive
    assert len(result.ranked) == 4
    resources = dict(result.best.resource_cycles)
    assert resources[Resource.CUBE.value] > 0.0
    assert resources[Resource.VECTOR.value] > 0.0


def test_scatter_models_initial_copy_and_atomic_contention_from_ir() -> None:
    low_collision = scatter_elements(
        (1024, 17), (1024, 1), 1, reduction="add", collision_group=1
    )
    high_collision = scatter_elements(
        (1024, 17), (1024, 1), 1, reduction="add", collision_group=8
    )
    axis_tiles = tuple((axis.name, min(axis.extent, 64)) for axis in low_collision.axes)
    plan = TilingPlan(
        algorithm=0,
        axis_tiles=axis_tiles,
        used_cores=20,
        buffers=((MemorySpace.UB, 2),),
        traversal=("u0", "u1"),
    )
    low = simulate(low_collision, plan, HARDWARE)
    high = simulate(high_collision, plan, HARDWARE)
    assert low.valid and high.valid
    assert low.gm_read_bytes >= 1024 * 17 * 2
    assert low.gm_write_bytes >= 1024 * 17 * 2
    assert high.total_cycles > low.total_cycles


def test_attention_grad_exposes_gqa_reuse_cube_vector_and_atomic_paths() -> None:
    operator = flash_attention_score_grad(1, 8, 2, 64, 128, 64)
    plan = TilingPlan(
        algorithm=0,
        axis_tiles=(
            ("b", 1), ("hk", 1), ("g", 1), ("q", 32),
            ("s", 64), ("di", 32), ("do", 64),
        ),
        task_tiles=(
            ("b", 1), ("hk", 1), ("g", 1), ("q", 64),
            ("s", 64), ("di", 32), ("do", 64),
        ),
        cache_tiles=(
            ("b", 1), ("hk", 1), ("g", 4), ("q", 64),
            ("s", 64), ("di", 32), ("do", 64),
        ),
        used_cores=20,
        buffers=(
            (MemorySpace.L1, 1), (MemorySpace.L0A, 2),
            (MemorySpace.L0B, 2), (MemorySpace.L0C, 2),
        ),
        traversal=("b", "hk", "g", "q", "do"),
    )
    result = simulate(operator, plan, HARDWARE)
    assert result.valid
    resources = dict(result.resource_cycles)
    assert resources[Resource.CUBE] > 0.0
    assert resources[Resource.VECTOR] > 0.0
    assert resources[Resource.ATOMIC] > 0.0


def test_scalar_reduction_is_a_legal_empty_output_domain() -> None:
    result = solve(
        reduce_sum((4096,), 0),
        HARDWARE,
        ScheduleSpace(
            tile_options=(("d0", (64, 256)),),
            core_options=(1, 8, 40),
            reduction_options=(("d0", (1, 8, 40)),),
            buffer_options=((1,), (2,)),
            traversal_options=((),),
        ),
        SearchPolicy(top_k=3, max_evaluations=1000),
    )
    assert result.exhaustive
    assert result.ranked
