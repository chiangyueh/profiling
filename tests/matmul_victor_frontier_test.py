#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import generate_matmul_victor_frontier_candidates as candidates
import generate_matmul_victor_frontier_workloads as workloads


def test_victor_workload_contract() -> None:
    rows = workloads.build_workloads()
    assert [
        (int(row["m"]), int(row["n"]), int(row["k"])) for row in rows
    ] == [
        (2048, 1536, 7168),
        (2048, 7168, 2048),
        (4096, 512, 7168),
    ]
    assert all(int(row["required_successful_tilings"]) == 720 for row in rows)


def test_frontier_is_frozen_by_structure_before_model_scoring() -> None:
    platform = candidates.old.Hardware(
        20, 64 * 1024, 64 * 1024, 128 * 1024, 524032,
        192 * 1024 * 1024, 16.0, 8.0,
    )
    hardware = candidates.generic_hardware(platform)
    for metadata in workloads.build_workloads():
        workload = candidates.old.Workload(
            metadata["workload_id"], int(metadata["m"]), int(metadata["n"]),
            int(metadata["k"]), metadata["dtype"], False, False, 20,
        )
        proposed, anchors, _ = candidates.proposed_candidates(
            workload, platform, hardware
        )
        formal, reserves = candidates.select_fixed_design(
            proposed, anchors, 720
        )
        assert len(formal) == 720
        assert len(reserves) == 32
        assert len({candidates.signature(row["knowledge"]) for row in (*formal, *reserves)}) == 752
        assert not any("simulation" in row or "model_rank" in row for row in formal)
        assert {
            candidates.execution_mode_name(row["knowledge"]) for row in formal
        } == {"base", "single_core_split_k", "deterministic_split_k"}
        assert {
            row["knowledge"]["usedCoreNum"] for row in formal
        } == set(range(1, 21))
        assert {
            row["knowledge"]["baseK"] for row in formal
        } == {16, 32, 64, 96, 128, 192, 256}
