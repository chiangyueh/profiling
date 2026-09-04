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


def test_frozen_campaign_contract() -> None:
    rows = workloads.build_workloads()
    assert len(rows) == 70
    assert len({row["workload_id"] for row in rows}) == 70
    assert Counter(row["calibration_partition"] for row in rows) == {
        "calibration": 49,
        "holdout": 21,
    }
    assert sum(int(row["required_successful_tilings"]) for row in rows) == 2185
    assert Counter(
        row["target_kernel_family"] for row in rows
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
