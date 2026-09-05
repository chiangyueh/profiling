#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import generate_matmul_hardware_calibration_workloads as workloads
import generate_matmul_hardware_calibration_candidates as candidates
import profile_direct_matmul as direct_profile


def test_frozen_campaign_contract() -> None:
    rows = workloads.build_workloads()
    assert len(rows) == 70
    assert len({row["workload_id"] for row in rows}) == 70
    assert len({
        (row["m"], row["n"], row["k"], row["dtype"],
         row["trans_a"], row["trans_b"])
        for row in rows
    }) == 70
    assert Counter(row["calibration_partition"] for row in rows) == {
        "calibration": 49,
        "holdout": 21,
    }
    assert sum(int(row["required_successful_tilings"]) for row in rows) == 2185
    assert Counter(
        row["coverage_intent"] for row in rows
    ) == {
        "base": 19,
        "single_core_split_k": 13,
        "deterministic_split_k": 11,
        "al1_full_load": 6,
        "bl1_full_load": 7,
        "bl1_full_load_fixpipe": 7,
        "bl1_full_load_vec_nz2nd": 7,
    }


def test_jsonl_measurements_are_recoverable(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    measurement = {
        "workload_id": "w0",
        "rank": "1",
        "candidate_role": "searched",
        "success": "1",
        "preflight_passed": "1",
        "median_ms": "0.5",
        "model_schedule_sha256": "abc",
    }
    record = {
        "record_type": "candidate_measurement",
        "measurement": measurement,
        "samples_ms": [0.4, 0.5, 0.6],
    }
    (logs / "1.log").write_text(json.dumps(record) + "\n", encoding="utf-8")
    profiles = tmp_path / "profiles.csv"
    samples = tmp_path / "samples.csv"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "restore_matmul_measurement_logs.py"),
            "--log-directory", str(logs),
            "--profile-output", str(profiles),
            "--samples-output", str(samples),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "profiles=1 samples=3" in completed.stdout
    with profiles.open(newline="", encoding="utf-8") as stream:
        restored_profiles = list(csv.DictReader(stream))
    with samples.open(newline="", encoding="utf-8") as stream:
        restored_samples = list(csv.DictReader(stream))
    assert restored_profiles[0]["model_schedule_sha256"] == "abc"
    assert [row["latency_ms"] for row in restored_samples] == ["0.4", "0.5", "0.6"]


def test_direct_resume_does_not_repeat_numeric_failures(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    manifest = {("w0", "7"): {"workload_id": "w0", "rank": "7"}}
    failure = {
        "schema": direct_profile.SCHEMA,
        "record_type": "candidate_failure",
        "candidate": {"workload_id": "w0", "rank": "7"},
        "runner": {"status": "failed", "error": "numeric mismatch"},
    }
    (logs / "1.log").write_text(json.dumps(failure) + "\n", encoding="utf-8")

    completed, samples, attempted = direct_profile.load_completed(logs, manifest)

    assert completed == {}
    assert samples == {}
    assert attempted == {("w0", "7")}


def _pool_item(family: str, rank: int, core_count: int) -> dict:
    knowledge = {field: 1 for field in candidates.old.KNOWLEDGE_FIELDS}
    knowledge.update({
        "usedCoreNum": core_count,
        "tilingEnable": {
            "base": 0,
            "single_core_split_k": 2,
            "deterministic_split_k": 3,
        }[family],
    })
    return {
        "family": family,
        "knowledge": knowledge,
        "model_rank": rank,
    }


def test_controlled_selection_does_not_force_unrelated_graphs() -> None:
    pool = [
        _pool_item("base", 1, 20),
        _pool_item("deterministic_split_k", 2, 19),
        _pool_item("single_core_split_k", 3, 18),
        _pool_item("base", 4, 17),
        _pool_item("base", 5, 16),
    ]

    selected, _ = candidates.select_controlled(
        pool, count=3, reserves=1, coverage_intent="base"
    )

    assert [item["family"] for item, _ in selected] == ["base"] * 3
    assert all(reason != "execution_graph" for _, reason in selected)


def test_controlled_selection_keeps_real_global_model_optimum() -> None:
    pool = [
        _pool_item("deterministic_split_k", 1, 19),
        _pool_item("base", 2, 20),
        _pool_item("base", 3, 18),
        _pool_item("base", 4, 16),
        _pool_item("base", 5, 14),
    ]

    selected, _ = candidates.select_controlled(
        pool, count=3, reserves=1, coverage_intent="base"
    )

    assert selected[0][0]["family"] == "deterministic_split_k"
    assert [item["family"] for item, _ in selected[1:]] == ["base", "base"]
    assert selected[1][1] == "coverage_anchor"
