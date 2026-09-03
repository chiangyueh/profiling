#!/usr/bin/env python3
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import analyze_matmul_model_validation as analysis
import generate_matmul_model_validation_candidates as candidates
import generate_matmul_model_validation_workloads as workloads
import refine_matmul_v3_candidates as old
from npu_cost_model import (
    CANN81_MATMUL_FAMILIES,
    CANN81_MATMUL_KERNEL_SUFFIXES,
    MemorySpace,
    TilingPlan,
    kernel_suffix,
    lower_plan_to_cann,
    plan_from_cann,
    simulate,
    source_kernel_suffix,
    validate_cann_tiling,
)
from profile_official_tilings import IncrementalJsonl


HARDWARE = old.Hardware(
    20, 64 * 1024, 64 * 1024, 128 * 1024, 511.75 * 1024,
    192 * 1024 * 1024, 110.0, 32.0,
)


def test_cann81_dispatch_suffixes_and_families_are_complete() -> None:
    keys = (
        (0, 0, "base"),
        (0, 1, "base"),
        (2, 0, "single_core_split_k"),
        (2, 1, "single_core_split_k"),
        (3, 0, "deterministic_split_k"),
        (3, 1, "deterministic_split_k"),
        (10, 1, "al1_full_load"),
        (20, 0, "bl1_full_load"),
        (20, 1, "bl1_full_load"),
        (1020, 0, "bl1_full_load_fixpipe"),
        (1020, 1, "bl1_full_load_fixpipe"),
        (2020, 1, "bl1_full_load_vec_nz2nd"),
    )
    observed_suffixes = []
    observed_families = set()
    for enabled, mix, family in keys:
        knowledge = {"tilingEnable": enabled, "mixNd2Nz": mix}
        observed_suffixes.append(kernel_suffix(knowledge, aligned=bool(mix)))
        assert candidates.execution_mode_name(knowledge) == family
        observed_families.add(family)
    assert tuple(observed_suffixes) == CANN81_MATMUL_KERNEL_SUFFIXES
    assert observed_families == set(CANN81_MATMUL_FAMILIES)


def test_all_cann81_kernel_suffixes_are_generated_from_source_rules() -> None:
    cases = (
        (512, 512, 512, "fp16", False, False),
        (257, 1009, 4097, "fp16", False, False),
        (128, 17, 16384, "fp16", False, False),
        (16, 320, 4096, "fp32", False, True),
        (8192, 64, 32, "fp16", False, False),
        (28672, 64, 17, "fp16", False, False),
        (16384, 17, 32, "fp16", False, False),
        (24576, 17, 7, "fp32", False, False),
        (16384, 17, 8, "fp32", False, False),
    )
    suffixes: set[int] = set()
    families: set[str] = set()
    for sequence, (m, n, k, dtype, trans_a, trans_b) in enumerate(cases):
        workload = old.Workload(
            f"suffix_{sequence}", m, n, k, dtype,
            trans_a, trans_b, HARDWARE.aic_cores,
        )
        for knowledge in candidates.proposal_space(workload, HARDWARE):
            suffixes.add(source_kernel_suffix(
                m, n, k, dtype, trans_a, trans_b, knowledge,
            ))
            families.add(candidates.execution_mode_name(knowledge))
    assert suffixes == set(CANN81_MATMUL_KERNEL_SUFFIXES)
    assert families == set(CANN81_MATMUL_FAMILIES)


@pytest.mark.parametrize(
    "family,shape,dtype,trans_a,trans_b",
    (
        ("base", (512, 512, 512), "fp16", False, False),
        ("single_core_split_k", (128, 128, 16384), "fp16", False, False),
        ("deterministic_split_k", (128, 128, 16384), "fp16", False, False),
        ("al1_full_load", (16, 320, 4096), "fp32", False, True),
        ("bl1_full_load", (65536, 128, 128), "fp16", False, False),
        ("bl1_full_load_fixpipe", (65536, 7, 16), "fp16", False, False),
        ("bl1_full_load_vec_nz2nd", (65536, 7, 8), "fp32", False, True),
    ),
)
def test_each_cann81_family_is_generated_lowered_validated_and_simulated(
    family: str,
    shape: tuple[int, int, int],
    dtype: str,
    trans_a: bool,
    trans_b: bool,
) -> None:
    workload = old.Workload(
        family, *shape, dtype, trans_a, trans_b, HARDWARE.aic_cores
    )
    proposals = candidates.proposal_space(workload, HARDWARE)
    family_rows = [
        row for row in proposals
        if candidates.execution_mode_name(row) == family
    ]
    assert family_rows, f"no generated CANN row for {family}"
    generic = candidates.generic_hardware(HARDWARE)
    operator = candidates.matmul(
        *shape, dtype, trans_a=trans_a, trans_b=trans_b
    )
    for knowledge in family_rows:
        assert not validate_cann_tiling(
            *shape, dtype, trans_a, trans_b, knowledge, generic
        )
        plan = plan_from_cann(
            *shape, knowledge, dtype=dtype,
            trans_a=trans_a, trans_b=trans_b,
        )
        assert simulate(operator, plan, generic).valid


def test_workspace_equations_come_from_schedule_dimensions() -> None:
    workload = old.Workload(
        "det", 128, 128, 16384, "fp16", False, False, 20
    )
    proposal = next(
        row for row in candidates.proposal_space(workload, HARDWARE)
        if candidates.execution_mode_name(row) == "deterministic_split_k"
    )
    generic = candidates.generic_hardware(HARDWARE)
    operator = candidates.matmul(128, 128, 16384, "fp16")
    plan = plan_from_cann(128, 128, 16384, proposal)
    result = simulate(operator, plan, generic)
    expected = (
        20 * 1024 * 1024
        + result.active_cores
        * proposal["singleCoreM"]
        * proposal["singleCoreN"]
        * 2
        * 4
    )
    assert result.workspace_bytes == expected


def test_catalog_has_200_plus_unique_shapes_and_colleague_anchors() -> None:
    rows = workloads.build_catalog()
    assert len(rows) == 240
    assert len({(row["m"], row["n"], row["k"]) for row in rows}) == 240
    assert [(row["m"], row["n"], row["k"]) for row in rows[:3]] == [
        ("2048", "1536", "7168"),
        ("2048", "7168", "2048"),
        ("4096", "512", "7168"),
    ]
    assert {row["max_cores"] for row in rows} == {"20"}
    assert all("search_core_cap" not in row for row in rows)
    assert {row["search_family"] for row in rows} == {
        "hardware_ideal_region"
    }
    coverage = ";".join(row["coverage"] for row in rows[:200])
    for name in (
        "cann81_al1_full_load",
        "cann81_bl1_full_load",
        "cann81_bl1_fixpipe",
        "cann81_bl1_vec_nz2nd",
        "cann81_mixed_nd2nz",
        "cann81_splitk_mixed_nd2nz",
        "cann81_bl1_mixed_nd2nz",
        "cann81_fixpipe_mixed_nd2nz",
    ):
        assert name in coverage


def test_hardware_simulator_ranks_the_full_legal_pool(monkeypatch) -> None:
    workload = old.Workload(
        "anchor", 2048, 1536, 7168, "fp16", False, False, 20
    )
    proposals = candidates.proposal_space(workload, HARDWARE)
    def old_model_must_not_run(*_args, **_kwargs):
        raise AssertionError("old analytical cost model was called")

    monkeypatch.setattr(old, "analytical_score", old_model_must_not_run)
    ranked, new_ns = candidates.ranked_pool(
        workload, proposals, HARDWARE
    )
    assert len(proposals) > 1
    assert len(ranked) == len(proposals)
    assert new_ns > 0
    # This geometry exposes at least one task per physical AIC.  Lower-core
    # copies of the same geometry are dominated, so the finite region keeps
    # the full physical core count instead of manufacturing core caps.
    core_counts = {row["knowledge"]["usedCoreNum"] for row in ranked}
    assert max(core_counts) == HARDWARE.aic_cores
    assert min(core_counts) >= 1
    assert all(row["selection"] == "new_hardware_simulator" for row in ranked)


def test_base_abi_materialization_preserves_task_geometry_and_k_extent() -> None:
    workload = old.Workload(
        "short_k", 96, 80, 128, "fp16", False, False, 20
    )
    plan = TilingPlan(
        algorithm=0,
        axis_tiles=(("m", 16), ("n", 16), ("k", 256)),
        task_tiles=(("m", 96), ("n", 80), ("k", 256)),
        cache_tiles=(("m", 96), ("n", 80), ("k", 256)),
        used_cores=1,
        reduction_parts=(("k", 1),),
        buffers=(
            (MemorySpace.L1, 1),
            (MemorySpace.L0A, 1),
            (MemorySpace.L0B, 1),
            (MemorySpace.L0C, 1),
        ),
        traversal=("m", "n"),
    )
    knowledge = lower_plan_to_cann(
        workload.m, workload.n, workload.k, workload.dtype,
        workload.trans_a, workload.trans_b, plan,
        candidates.generic_hardware(HARDWARE),
    )
    assert knowledge["singleCoreM"] == knowledge["baseM"] == 16
    assert knowledge["singleCoreN"] == knowledge["baseN"] == 16
    assert knowledge["baseK"] == 128


def test_paired_result_uses_measured_noise_threshold() -> None:
    official = {"median_ms": "1", "stddev_ms": "0.001"}
    close = {"median_ms": "1.005", "stddev_ms": "0.001"}
    faster = {"median_ms": "0.97", "stddev_ms": "0.001"}
    assert analysis.paired_result(official, close)["verdict"] == "within_noise"
    assert analysis.paired_result(official, faster)["verdict"] == "improved"


def test_deterministic_splitk_is_lowered_as_numeric_reduction_parts() -> None:
    workload = old.Workload(
        "split", 128, 128, 16384, "fp16", False, False, 20
    )
    proposals = candidates.proposal_space_for(
        workload,
        None,
        HARDWARE,
        "deterministic_split_k",
    )
    proposal = next(
        item for item in proposals if item["tilingEnable"] == 3
    )
    plan = plan_from_cann(
        workload.m, workload.n, workload.k, proposal
    )
    assert plan.algorithm == 2
    assert plan.reductions["k"] > 1
    assert plan.used_cores == proposal["usedCoreNum"]


def test_old_model_does_not_cancel_splitk_parallelism_with_base_l2_penalty() -> None:
    workload = old.Workload(
        "split_parallelism", 128, 128, 32768, "fp16", False, False, 20
    )
    knowledge = {
        "usedCoreNum": 4,
        "singleCoreM": 128,
        "singleCoreN": 128,
        "singleCoreK": 384,
        "baseM": 128,
        "baseN": 128,
        "baseK": 128,
        "depthA1": 9,
        "depthB1": 6,
        "stepM": 3,
        "stepN": 1,
        "iterateOrder": 0,
        "stepKa": 3,
        "stepKb": 3,
        "dbL0A": 2,
        "dbL0B": 2,
        "dbL0C": 2,
        "l2MTileCnt": 1,
        "l2NTileCnt": 1,
        "l2MTileBlock": 1,
        "l2NTileBlock": 1,
        "l2IterateOrder": 0,
        "tilingEnable": 3,
    }
    four_core = old.analytical_score(workload, knowledge, HARDWARE)
    twenty_core = old.analytical_score(
        workload, dict(knowledge, usedCoreNum=20), HARDWARE
    )
    assert twenty_core.cycles < four_core.cycles


def test_matmul_generation_has_no_fixed_base_or_splitk_quota() -> None:
    source = Path(candidates.__file__).read_text(encoding="utf-8")
    assert "GEOMETRY_LIMIT" not in source
    assert "deterministic_splitk_candidates" not in source
    assert "splitk-workloads" not in source
    workload = old.Workload(
        "deep", 128, 128, 16384, "fp16", False, False, 20
    )
    proposals, region = candidates.derive_proposals(workload, HARDWARE)
    assert region.exhaustive
    assert region.evaluated <= 10_000
    assert region.plans
    assert {item["tilingEnable"] for item in proposals} == {0, 2, 3}


def test_capacity_frontier_retargets_cores_to_changed_task_grid() -> None:
    workload = old.Workload(
        "short", 96, 80, 128, "fp16", False, False, 20
    )
    proposals = candidates.proposal_space(workload, HARDWARE)
    three_task = [
        item for item in proposals
        if item["tilingEnable"] == 0
        and old.ceil_div(workload.m, item["singleCoreM"])
        * old.ceil_div(workload.n, item["singleCoreN"]) == 3
    ]
    assert three_task
    assert 3 in {item["usedCoreNum"] for item in three_task}
    assert all(item["usedCoreNum"] <= 3 for item in three_task)


def test_ideal_region_ignores_cartesian_axis_sampling_limit() -> None:
    workload = old.Workload(
        "axis_limit", 512, 768, 4096, "fp16", False, False, 20
    )
    generic = candidates.generic_hardware(HARDWARE)
    operator = candidates.matmul(
        workload.m, workload.n, workload.k, workload.dtype
    )
    policy = candidates.SearchPolicy(top_k=1, max_evaluations=10_000)
    small = candidates.derive_ideal_region(
        operator,
        generic,
        candidates.ScheduleSpace(
            core_options=tuple(range(1, 21)),
            coupled_task_axes=("m", "n"),
            max_axis_values=1,
        ),
        policy,
    )
    large = candidates.derive_ideal_region(
        operator,
        generic,
        candidates.ScheduleSpace(
            core_options=tuple(range(1, 21)),
            coupled_task_axes=("m", "n"),
            max_axis_values=128,
        ),
        policy,
    )
    assert small.exhaustive and large.exhaustive
    assert small.plans == large.plans


def test_active_generator_never_returns_a_truncated_ideal_region(monkeypatch) -> None:
    original = candidates.derive_ideal_region

    def truncated(*args, **kwargs):
        return replace(original(*args, **kwargs), exhaustive=False)

    monkeypatch.setattr(candidates, "derive_ideal_region", truncated)
    workload = old.Workload(
        "bounded", 128, 128, 1024, "fp16", False, False, 20
    )
    with pytest.raises(old.SearchError, match="truncated search result"):
        candidates.derive_proposals(workload, HARDWARE)


def test_execution_identity_is_model_owned_not_callback_owned() -> None:
    workload = old.Workload(
        "identity", 128, 128, 1024, "fp16", False, False, 20
    )
    knowledge = candidates.proposal_space(workload, HARDWARE)[0]
    row: dict[str, str] = {}
    candidates.attach_execution_identity(row, workload, knowledge)
    assert len(row["model_schedule_sha256"]) == 64
    assert row["model_kernel_family"] == old.template_name(knowledge)
    assert not any(name.startswith("callback_") for name in row)


def test_active_generator_has_no_official_tiler_or_runtimekb_route() -> None:
    source = Path(candidates.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "old.invoke_official_callback",
        "old.validate_callback",
        "tbe.common.utils",
        "_RT_BANK_CACHE",
    ):
        assert forbidden not in source


def test_fp32_k_alignment_follows_input_layout_contract() -> None:
    nt = old.Workload("nt", 128, 128, 1024, "fp32", False, True, 20)
    tn = old.Workload("tn", 128, 128, 1024, "fp32", True, False, 20)
    assert old.base_k_alignment(nt) == 8
    assert old.base_k_alignment(tn) == 16
    assert all(item["baseK"] % 8 == 0 for item in candidates.proposal_space(nt, HARDWARE))
    assert all(item["baseK"] % 16 == 0 for item in candidates.proposal_space(tn, HARDWARE))


def test_incremental_log_resumes_without_duplicate_records(tmp_path: Path) -> None:
    writer = IncrementalJsonl(tmp_path, 256)
    writer.write("a", {"record_type": "one", "payload": "x" * 100})
    writer.close()
    writer = IncrementalJsonl(tmp_path, 256)
    writer.write("a", {"record_type": "duplicate"})
    writer.write("b", {"record_type": "two", "payload": "y" * 100})
    writer.close()
    text = "".join(path.read_text() for path in sorted(tmp_path.glob("*.log")))
    assert text.count('"record_key":"a"') == 1
    assert text.count('"record_key":"b"') == 1
    assert all(path.stat().st_size <= 256 for path in tmp_path.glob("*.log"))
