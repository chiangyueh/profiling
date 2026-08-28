from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import generate_matmul_controlled_candidates as candidates
import generate_matmul_controlled_workloads as catalog
import refine_matmul_v3_candidates as contract


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def legal_seed(workload: contract.Workload) -> dict[str, int]:
    return {
        "usedCoreNum": 20,
        "singleCoreM": 128,
        "singleCoreN": 256,
        "singleCoreK": workload.k,
        "baseM": 128,
        "baseN": 256,
        "baseK": 64,
        "depthA1": 16,
        "depthB1": 8,
        "stepM": 1,
        "stepN": 1,
        "iterateOrder": 0,
        "stepKa": 8,
        "stepKb": 4,
        "dbL0A": 2,
        "dbL0B": 2,
        "dbL0C": 1,
        "l2MTileCnt": 1,
        "l2NTileCnt": 1,
        "l2MTileBlock": contract.ceil_div(workload.m, 128),
        "l2NTileBlock": contract.ceil_div(workload.n, 256),
        "l2IterateOrder": 0,
        "tilingEnable": 0,
    }


def test_catalog_and_controlled_spaces_cover_every_block() -> None:
    rows = catalog.build_catalog()
    assert len(rows) == 250
    assert Counter(row["experiment_block"] for row in rows) == catalog.BLOCK_COUNTS
    assert len(
        {
            (
                row["m"], row["n"], row["k"], row["dtype"],
                row["trans_a"], row["trans_b"],
            )
            for row in rows
        }
    ) == 250

    hardware = contract.Hardware(
        20, 65536, 65536, 131072, 524288, 201326592, 110.0, 32.0
    )
    representatives = {
        "l2": contract.Workload("l2", 2048, 2048, 4096, "fp16", False, False, 20),
        "concurrency": contract.Workload(
            "concurrency", 1024, 1024, 1024, "fp16", False, False, 20
        ),
        "buffer": contract.Workload(
            "buffer", 1024, 1024, 512, "fp16", False, False, 20
        ),
        "splitk": contract.Workload(
            "splitk", 512, 512, 32768, "fp16", False, False, 20
        ),
    }
    for block, workload in representatives.items():
        knowledge = legal_seed(workload)
        assert contract.hard_legal(workload, knowledge, hardware)
        seed = SimpleNamespace(bank=SimpleNamespace(knowledge=knowledge))
        proposals = candidates.proposals_for(block, workload, seed, hardware)
        assert len(proposals) >= 28
        assert len(
            {contract.knowledge_signature(proposal[0]) for proposal in proposals}
        ) == len(proposals)


def test_exporter_enforces_exact_250_by_20_gate(tmp_path: Path) -> None:
    workloads = catalog.build_catalog()
    candidate_rows: list[dict[str, str]] = []
    profile_rows: list[dict[str, str]] = []
    sample_rows: list[dict[str, str]] = []
    official_rows: list[dict[str, str]] = []
    official_samples: list[dict[str, str]] = []
    knowledge = legal_seed(
        contract.Workload("test", 128, 256, 512, "fp16", False, False, 20)
    )
    candidate_mapping = {
        "usedCoreNum": "used_core_num",
        "singleCoreM": "single_core_m",
        "singleCoreN": "single_core_n",
        "singleCoreK": "single_core_k",
        "baseM": "base_m",
        "baseN": "base_n",
        "baseK": "base_k",
        "depthA1": "depth_a1",
        "depthB1": "depth_b1",
        "stepM": "step_m",
        "stepN": "step_n",
        "iterateOrder": "iterate_order",
        "stepKa": "step_ka",
        "stepKb": "step_kb",
        "dbL0A": "db_l0a",
        "dbL0B": "db_l0b",
        "dbL0C": "db_l0c",
        "l2MTileCnt": "bank_l2_m_tile_count",
        "l2NTileCnt": "bank_l2_n_tile_count",
        "l2MTileBlock": "bank_l2_m_tile_block",
        "l2NTileBlock": "bank_l2_n_tile_block",
        "l2IterateOrder": "bank_l2_iterate_order",
        "tilingEnable": "bank_tiling_enable",
    }
    for workload in workloads:
        workload_id = workload["workload_id"]
        official_rows.append(
            {
                "workload_id": workload_id,
                "success": "1",
                "preflight_passed": "1",
                "preflight_mode": "numeric_signed_axes_full_v3",
                "median_ms": "1",
                "stddev_ms": "0.01",
            }
        )
        official_samples.append(
            {
                "workload_id": workload_id,
                "rank": "-1",
                "candidate_role": "official_operator_baseline",
                "sample": "0",
                "latency_ms": "1",
            }
        )
        for rank in range(20):
            role = "bank_seed_control" if rank == 0 else "searched"
            candidate = {
                "workload_id": workload_id,
                "rank": str(rank),
                "candidate_role": role,
                "pair_id": f"{workload_id}:pair:{rank // 2}",
                "changed_factor": workload["experiment_block"],
                "pair_variant": str(rank % 2),
                "controlled_sequence": str(rank),
                "callback_tiling_sha256": f"sha-{workload_id}-{rank}",
                "callback_tiling_key": "1",
                "callback_kernel_family": "BASE",
                "callback_kernel_variant": "BASE_ALIGNED",
            }
            candidate.update(
                {column: str(knowledge[field]) for field, column in candidate_mapping.items()}
            )
            candidate_rows.append(candidate)
            profile_rows.append(
                {
                    "workload_id": workload_id,
                    "rank": str(rank),
                    "candidate_role": role,
                    "success": "1",
                    "preflight_passed": "1",
                    "preflight_mode": "numeric_signed_axes_full_v3",
                    "median_ms": "1",
                    "stddev_ms": "0.01",
                    "min_ms": "0.99",
                    "max_ms": "1.01",
                    "tiling_signature": f"{workload_id}:{rank}",
                }
            )
            sample_rows.append(
                {
                    "workload_id": workload_id,
                    "rank": str(rank),
                    "candidate_role": role,
                    "sample": "0",
                    "latency_ms": "1",
                }
            )

    paths = {
        "workloads": tmp_path / "workloads.csv",
        "candidates": tmp_path / "candidates.csv",
        "profile": tmp_path / "profile.csv",
        "samples": tmp_path / "samples.csv",
        "official_profile": tmp_path / "official_profile.csv",
        "official_samples": tmp_path / "official_samples.csv",
    }
    for name, rows in (
        ("workloads", workloads),
        ("candidates", candidate_rows),
        ("profile", profile_rows),
        ("samples", sample_rows),
        ("official_profile", official_rows),
        ("official_samples", official_samples),
    ):
        write_csv(paths[name], rows)
    log_directory = tmp_path / "logs"
    command = [
        sys.executable,
        str(ROOT / "tools" / "export_matmul_controlled_frontier.py"),
    ]
    for name, path in paths.items():
        command.extend((f"--{name.replace('_', '-')}", str(path)))
    command.extend(
        (
            "--log-directory", str(log_directory),
            "--soc", "Ascend910B3",
            "--aic-cores", "20",
        )
    )
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    records = []
    for path in sorted(log_directory.glob("*.log")):
        records.extend(json.loads(line) for line in path.read_text().splitlines())
        assert path.stat().st_size <= 50 * 1024 * 1024
    formal = [row for row in records if row["record_type"] == "formal_latency_candidate"]
    assert len(formal) == 5000
    summary = [row for row in records if row["record_type"] == "campaign_summary"]
    assert summary == [
        {
            "schema": "matmul_controlled_frontier_measurement_v1",
            "record_type": "campaign_summary",
            "status": "complete",
            "semantic_shapes": 250,
            "formal_latency_count": 5000,
            "formal_latency_count_by_block": {
                "l2": 1600,
                "concurrency": 1200,
                "buffer": 1200,
                "splitk": 1000,
            },
            "all_groups_have_20_distinct_correct_tilings": True,
            "log_file_count": 1,
            "log_rotation_max_bytes": 50 * 1024 * 1024,
        }
    ]
