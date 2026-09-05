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
    metadata = workloads.build_workloads()[0]
    workload = candidates.old.Workload(
        metadata["workload_id"], int(metadata["m"]), int(metadata["n"]),
        int(metadata["k"]), metadata["dtype"], False, False, 20,
    )
    knowledge = candidates.base_knowledge(workload, 32, 192, 32, 20)
    anchor = {
        "knowledge": knowledge,
        "design_role": "official_source_anchor",
        "controlled_factor": "official_source_route",
        "pair_id": "source_000", "factor_signature": "ALL@20",
        "hardware_stratum": candidates.hardware_stratum(
            workload, knowledge, hardware
        ),
        "source_raw_tiling_hex": "00" * 272,
        "source_route": "ALL+BASE", "source_core_cap": "20",
    }
    proposed, _ = candidates.source_frontier_candidates(
        workload, [anchor], platform, hardware
    )
    formal, reserves = candidates.select_fixed_design(
        proposed, [anchor], 720
    )
    assert len(formal) == 720
    assert len(reserves) == 32
    assert formal[0] is anchor
    assert not any("simulation" in row or "model_rank" in row for row in formal)
    assert {row["knowledge"]["usedCoreNum"] for row in formal} == set(range(1, 21))
    assert {row["controlled_factor"] for row in formal} >= {
        "official_source_route", "source_mnk_geometry", "core_parallelism",
        "k_pipeline", "l0_buffering", "l2_partition", "cube_traversal",
    }


def test_source_key_decodes_all_execution_graph_digits() -> None:
    words = [0] * 68
    words[0:15] = [20, 128, 128, 32768, 32768, 128, 128, 128,
                   128, 128, 128, 2, 2, 1, 1]
    words[17] = 0
    words[26:28] = [1, 1]
    words[30:33] = [2, 2, 2]
    words[50:55] = [1, 1, 1, 1, 0]
    import struct
    knowledge = candidates.knowledge_from_source(
        struct.pack("<68I", *words).hex(),
        10_000_000_000_000_000_000 + 10201,
    )
    assert knowledge["tilingEnable"] == 1020
