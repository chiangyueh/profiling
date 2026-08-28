#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

from scripts.gen_data import gen_golden_data
from tiling import base


class ExactCandidateValidator(base.BaseValidator):
    """Do not repair the candidate being audited."""

    def is_valid(self, params: list[base.BaseParam]) -> bool:
        return True

    def repair(self, params: list[base.BaseParam]) -> list[base.BaseParam]:
        return params

    def get_combinations(self, num: int, const_params: list[base.BaseParam]):
        raise NotImplementedError


class ShapeSweepReal(base.BaseAlgoReal):
    audit_algorithm = "shape_sweep"


def reviewed_shapes() -> list[tuple[int, int, int]]:
    shapes = []
    for index in range(200):
        m = 32 + 16 * (index % 20)
        n = 32 + 16 * (index // 20)
        k = 64 + 32 * ((7 * index + 3) % 13)
        shapes.append((m, n, k))
    if len(shapes) != 200 or len(set(shapes)) != 200:
        raise RuntimeError("the final audit must contain exactly 200 distinct shapes")
    return shapes


def _yes_geometry(index: int, m: int, n: int) -> dict[str, int]:
    base_geometries = (
        (128, 128, 64),
        (256, 64, 64),
        (64, 256, 64),
        (512, 32, 32),
        (32, 512, 32),
    )
    base_m, base_n, base_k = base_geometries[(index // 2) % len(base_geometries)]
    target_m = base_m if index % 4 == 0 else max(16, base_m // 2)
    target_n = base_n if index % 6 == 0 else max(16, base_n // 2)
    return {
        "MM_BASE_M": base_m,
        "MM_BASE_N": base_n,
        "MM_BASE_K": base_k,
        "MM_SINGLE_M": min(m, target_m),
        "MM_SINGLE_N": min(n, target_n),
    }


def _no_geometry(index: int, m: int, n: int) -> dict[str, int]:
    if index % 4 == 1:
        return {
            "MM_BASE_M": 16,
            "MM_BASE_N": 128,
            "MM_BASE_K": 64,
            "MM_SINGLE_M": 32,
            "MM_SINGLE_N": min(n, 128),
        }
    return {
        "MM_BASE_M": 128,
        "MM_BASE_N": 16,
        "MM_BASE_K": 64,
        "MM_SINGLE_M": min(m, 128),
        "MM_SINGLE_N": 32,
    }


def candidate_params(index: int, shape: tuple[int, int, int]) -> list[base.BaseParam]:
    m, n, k = shape
    geometry = _yes_geometry(index, m, n) if index % 2 == 0 else _no_geometry(index, m, n)
    values = {
        "MM_M": m,
        "MM_N": n,
        "MM_K": k,
        **geometry,
        "MM_STEP_M": 1,
        "MM_STEP_N": 1,
        "MM_STEP_Ka": 4,
        "MM_STEP_Kb": 4,
        "MM_DB_L0A": 2,
        "MM_DB_L0B": 2,
        "MM_DB_L0C": 2,
        "MM_ITER_ORDER": 0,
    }
    return [base.BaseParam(name, value, True) for name, value in values.items()]


def load_candidate_rows(log_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(log_dir.glob("*.log"), key=lambda item: int(item.stem)):
        with path.open(encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                if record.get("record_type") == "candidate_audit":
                    rows.append(record)
    return rows


def summarize(log_dir: Path, summary_path: Path) -> dict:
    rows = load_candidate_rows(log_dir)
    classifications = Counter(row["classification"] for row in rows)
    decisions = Counter(
        "accepted" if row["cost_model_filter"]["accepted"] else "rejected"
        for row in rows
    )
    numeric = Counter(row["numeric"]["status"] for row in rows)
    shapes = {
        (
            row["cost_model_filter"]["materialized_tiling"]["M"],
            row["cost_model_filter"]["materialized_tiling"]["N"],
            row["cost_model_filter"]["materialized_tiling"]["K"],
        )
        for row in rows
    }
    summary = {
        "schema": "matmul_cost_filter_shape_audit_summary_v1",
        "record_type": "run_summary",
        "run_id": os.environ.get("MATMUL_AUDIT_RUN_ID"),
        "total": len(rows),
        "unique_shapes": len(shapes),
        "classifications": dict(sorted(classifications.items())),
        "filter_decisions": dict(sorted(decisions.items())),
        "numeric": dict(sorted(numeric.items())),
        "latency_available": sum(row["execution"]["latency_us"] is not None for row in rows),
        "attention": {
            "filter_false_negative": classifications["rejected_correct_filter_false_negative"],
            "filter_false_positive": classifications["accepted_wrong_filter_false_positive"],
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    log_dir = Path(os.environ["MATMUL_AUDIT_LOG_DIR"])
    shapes = reviewed_shapes()
    runner = ShapeSweepReal(
        is_stop=lambda _results: False,
        validator=ExactCandidateValidator(),
    )
    print("MATMUL_FILTER_SHAPE_AUDIT_BEGIN shapes=200 accepted=100 rejected=100")
    for index, shape in enumerate(shapes):
        gen_golden_data(*shape, announce=False)
        runner._duration(candidate_params(index, shape))
    summary = summarize(log_dir, log_dir.parent / "summary.json")
    if summary["total"] != 200 or summary["unique_shapes"] != 200:
        raise RuntimeError(f"incomplete final shape audit: {summary}")
    base.BaseAlgo._append_audit(summary)
    print("MATMUL_FILTER_SHAPE_AUDIT_SUMMARY " + json.dumps(summary, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
