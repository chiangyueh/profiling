#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import analyze_matmul_model_validation as analysis
import generate_matmul_model_validation_candidates as candidates
import generate_matmul_model_validation_workloads as workloads
import refine_matmul_v3_candidates as old
from npu_cost_model import plan_from_cann
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
    assert len({row["search_core_cap"] for row in rows}) == 7
    assert sum(row["search_family"] == "base" for row in rows) == 180
    assert sum(
        row["search_family"] == "deterministic_split_k" for row in rows
    ) == 60


def test_shared_pool_varies_real_core_count_and_both_models_rank_it() -> None:
    workload = old.Workload(
        "anchor", 2048, 1536, 7168, "fp16", False, False, 20
    )
    proposals = candidates.proposal_space(workload, HARDWARE, 20)
    ranked, old_ns, new_ns = candidates.ranked_pool(
        workload, proposals, HARDWARE
    )
    assert len(proposals) >= 31
    assert len(ranked) == len(proposals)
    assert old_ns > 0 and new_ns > 0
    assert len({row["knowledge"]["usedCoreNum"] for row in ranked[:24]}) >= 3
    assert any(row["selection"] == "old_frontier" for row in ranked[:24])
    assert any(row["selection"] == "new_frontier" for row in ranked[:24])
    assert any("hardware_coverage" in row["selection"] for row in ranked[:24])


def test_noise_filtered_pairs_do_not_score_indistinguishable_latency() -> None:
    correct, comparable = analysis.pair_accuracy(
        [1.0, 2.0, 3.0],
        [1.0, 1.005, 1.2],
        [0.001, 0.001, 0.001],
    )
    assert (correct, comparable) == (2, 2)


def test_deterministic_splitk_is_lowered_as_numeric_reduction_parts() -> None:
    workload = old.Workload(
        "split", 128, 128, 16384, "fp16", False, False, 20
    )
    proposal = candidates.proposal_space_for(
        workload,
        None,
        HARDWARE,
        20,
        "deterministic_split_k",
    )[0]
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
