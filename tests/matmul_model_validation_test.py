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
from npu_cost_model import MemorySpace, TilingPlan, plan_from_cann
from profile_official_tilings import IncrementalJsonl


HARDWARE = old.Hardware(
    20, 64 * 1024, 64 * 1024, 128 * 1024, 511.75 * 1024,
    192 * 1024 * 1024, 110.0, 32.0,
)


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
    assert min(core_counts) >= 19
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
    knowledge = candidates.make_base_from_plan(workload, HARDWARE, plan)
    assert knowledge is not None
    assert knowledge["singleCoreM"] == knowledge["baseM"] == 96
    assert knowledge["singleCoreN"] == knowledge["baseN"] == 80
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
    assert plan.algorithm == 0
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
    assert {item["tilingEnable"] for item in proposals} == {0, 3}


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
    assert {item["usedCoreNum"] for item in three_task} == {3}


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
