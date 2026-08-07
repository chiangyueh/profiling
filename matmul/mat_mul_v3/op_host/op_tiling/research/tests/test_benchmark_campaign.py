from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path


RESEARCH = Path(__file__).resolve().parents[1]
ROOT = RESEARCH.parents[4]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(RESEARCH))
MANIFEST = load_module(
    "benchmark_manifest_under_test",
    RESEARCH / "benchmark_manifest.py",
)
PROFILE = load_module(
    "benchmark_profile_under_test",
    RESEARCH / "profile.py",
)


def test_manifest_has_balanced_nonrandom_coverage() -> None:
    cases = MANIFEST.build_cases(20)
    assert len(cases) == 192
    assert Counter(case.benchmark_group for case in cases) == {
        "balanced": 24,
        "aspect": 24,
        "underfilled": 24,
        "reduction": 24,
        "alignment": 24,
        "l1_residency": 24,
        "l2_wave": 24,
        "dtype_layout": 24,
    }
    assert Counter((case.trans_a, case.trans_b) for case in cases) == {
        (0, 0): 48,
        (1, 0): 48,
        (0, 1): 48,
        (1, 1): 48,
    }
    dtype_counts = Counter(case.dtype for case in cases)
    assert dtype_counts["fp16"] >= 90
    assert dtype_counts["bf16"] >= 48
    assert dtype_counts["fp32"] >= 40
    identities = {
        (
            case.m,
            case.n,
            case.k,
            case.dtype,
            case.trans_a,
            case.trans_b,
        )
        for case in cases
    }
    assert len(identities) == len(cases)


def test_manifest_stays_inside_full_numeric_preflight_contract() -> None:
    for case in MANIFEST.build_cases(20):
        element_bytes = 4 if case.dtype == "fp32" else 2
        input_bytes = (case.m * case.k + case.k * case.n) * element_bytes
        assert input_bytes <= 512 * 1024 * 1024
        assert case.k <= 60000


def test_profile_record_contains_replayable_measurement_context() -> None:
    source = {
        "workload_id": "bench",
        "m": "128",
        "n": "128",
        "k": "4096",
        "dtype": "fp16",
        "trans_a": "0",
        "trans_b": "0",
        "max_cores": "20",
        "rank": "1",
        "candidate_role": "searched",
        "candidate_source": "contract_global",
        "search_template": "BASE",
        "search_rationale": "coverage",
        "search_behavior_key": "[1,2]",
        "search_behavior_metrics": '{"active_cores":20}',
        "tiling_signature": ":".join("1" for _ in range(23)),
        "callback_tiling_key": "100000000000000000",
        "callback_block_dim": "20",
        "callback_workspace_bytes": "0",
        "callback_tiling_sha256": "abc",
    }
    measured = {
        "success": "1",
        "preflight_passed": "1",
        "preflight_mode": "numeric_signed_axes_full_v3",
        "median_ms": "1.0",
        "stddev_ms": "0.01",
        "min_ms": "0.98",
        "mean_ms": "1.0",
        "p95_ms": "1.02",
        "max_ms": "1.03",
        "raw_samples_json": "[0.98,1.0,1.02]",
        "runner_wall_ms": "12.5",
    }
    official = dict(measured, median_ms="1.1")
    bank = dict(measured, median_ms="1.05")
    row = PROFILE.profile_record(
        source,
        measured,
        record_id="record",
        run_id="run",
        soc="Ascend910B3",
        aic=20,
        toolkit="8.1",
        benchmark_stage="stage1",
        official=official,
        bank=bank,
        official_post=official,
        bank_post=bank,
        pair_validated=True,
    )
    assert row["benchmark_stage"] == "stage1"
    assert row["raw_samples_json"] == "[0.98,1.0,1.02]"
    assert row["official_profile_json"]
    assert row["bank_profile_json"]
    assert row["callback_tiling_sha256"] == "abc"
    assert row["failure_stage"] == ""


def test_full_command_is_fixed_two_stage_campaign() -> None:
    script = (ROOT / "run_npu.sh").read_text(encoding="utf-8")
    assert "STAGE1_NPU=64" in script
    assert "STAGE2_NPU=32" in script
    assert "--fixed-campaign-budget" in script
    assert '--resume-feedback "${RESUME}"' in script
    assert "BENCHMARK_RECORDS" not in script
    assert "npu_dual_model" not in script
