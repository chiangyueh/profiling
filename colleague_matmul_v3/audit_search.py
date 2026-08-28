#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.gen_data import gen_golden_data
from tiling import base, ga, limits, pso, sa


class GaReal(base.BaseAlgoReal, ga.GaAlgo):
    audit_algorithm = "ga"


class PsoReal(base.BaseAlgoReal, pso.PsoAlgo):
    audit_algorithm = "pso"


class SaReal(base.BaseAlgoReal, sa.SaAlgo):
    audit_algorithm = "sa"


class BoundaryReal(base.BaseAlgoReal):
    audit_algorithm = "boundary"


def domains() -> dict[str, list[int]]:
    return {
        "MM_BASE_M": [16, 32, 64, 128, 256, 512],
        "MM_BASE_N": [16, 32, 64, 128, 256, 512],
        "MM_BASE_K": [16, 32, 64, 96, 128, 256, 512],
        "MM_SINGLE_M": [16, 32, 64, 128, 176, 192, 256, 512],
        "MM_SINGLE_N": [16, 32, 64, 128, 176, 192, 256, 512],
        "MM_STEP_Ka": [1, 2, 4, 8, 12, 16],
        "MM_STEP_Kb": [1, 2, 4, 8, 12, 16],
    }


def fixed_params() -> list[base.BaseParam]:
    return [
        base.BaseParam("MM_M", 512, True),
        base.BaseParam("MM_N", 512, True),
        base.BaseParam("MM_K", 512, True),
        base.BaseParam("MM_STEP_M", 1, True),
        base.BaseParam("MM_STEP_N", 1, True),
        base.BaseParam("MM_DB_L0A", 2, True),
        base.BaseParam("MM_DB_L0B", 2, True),
        base.BaseParam("MM_DB_L0C", 2, True),
        base.BaseParam("MM_ITER_ORDER", 0, True),
    ]


def boundary_params(**overrides: int) -> list[base.BaseParam]:
    values = {
        "MM_M": 512,
        "MM_N": 512,
        "MM_K": 512,
        "MM_BASE_M": 128,
        "MM_BASE_N": 128,
        "MM_BASE_K": 64,
        "MM_SINGLE_M": 128,
        "MM_SINGLE_N": 128,
        "MM_STEP_M": 1,
        "MM_STEP_N": 1,
        "MM_STEP_Ka": 4,
        "MM_STEP_Kb": 4,
        "MM_DB_L0A": 2,
        "MM_DB_L0B": 2,
        "MM_DB_L0C": 2,
        "MM_ITER_ORDER": 0,
    }
    values.update(overrides)
    return [base.BaseParam(name, value, True) for name, value in values.items()]


def run_boundary_controls(state_dir: Path) -> None:
    # These controls guarantee that an audit contains both sides of the filter,
    # independently of where a stochastic search happens to initialize.
    controls = [
        boundary_params(),
        boundary_params(MM_BASE_M=256, MM_BASE_N=64, MM_SINGLE_M=256, MM_SINGLE_N=64),
        boundary_params(MM_BASE_M=64, MM_BASE_N=256, MM_SINGLE_M=64, MM_SINGLE_N=256),
        boundary_params(
            MM_BASE_M=512, MM_BASE_N=32, MM_BASE_K=32,
            MM_SINGLE_M=512, MM_SINGLE_N=32,
        ),
        boundary_params(
            MM_BASE_M=32, MM_BASE_N=512, MM_BASE_K=32,
            MM_SINGLE_M=32, MM_SINGLE_N=512,
        ),
        boundary_params(MM_BASE_M=16, MM_BASE_N=32, MM_SINGLE_M=176, MM_SINGLE_N=192),
        boundary_params(MM_BASE_M=64, MM_SINGLE_M=128),
        boundary_params(MM_BASE_N=64, MM_SINGLE_N=128),
        boundary_params(MM_BASE_M=256, MM_SINGLE_M=128, MM_DB_L0C=1),
        boundary_params(MM_BASE_N=256, MM_SINGLE_N=128, MM_DB_L0C=1),
    ]
    probe = BoundaryReal(
        is_stop=lambda _results: False,
        validator=legacy_validator(genetic=False),
        cache_path=str(state_dir / "boundary_cache.json"),
    )
    probe.audit_algorithm = "boundary"
    print("MATMUL_FILTER_AUDIT_ALGORITHM_BEGIN algorithm=boundary")
    for params in controls:
        probe._duration(params)
    print("MATMUL_FILTER_AUDIT_ALGORITHM_END algorithm=boundary")


def legacy_validator(*, genetic: bool):
    constraint = limits.MatmulLimits(
        max_cores=24,
        L0A_size=64 * 1024,
        L0B_size=64 * 1024,
        L0C_size=128 * 1024,
        L1_size=512 * 1024,
        domains=domains(),
        dtype_size=2,
    )
    if genetic:
        return ga.GaMatmulValidator(constraint)
    from tiling.validators import MatmulValidator
    return MatmulValidator(constraint)


def run_campaign(rounds: int, state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    gen_golden_data(512, 512, 512)
    stop_never = lambda _results: False
    run_boundary_controls(state_dir)

    print("MATMUL_FILTER_AUDIT_ALGORITHM_BEGIN algorithm=ga")
    ga_algo = GaReal(
        is_stop=stop_never,
        validator=legacy_validator(genetic=True),
        input_params=fixed_params(),
        pop_size=6,
        mut_rate=0.20,
        tournament_k=3,
        cache_path=str(state_dir / "ga_cache.json"),
    )
    ga_algo.audit_algorithm = "ga"
    for _ in range(rounds):
        ga_algo.run()
    print("MATMUL_FILTER_AUDIT_ALGORITHM_END algorithm=ga")

    print("MATMUL_FILTER_AUDIT_ALGORITHM_BEGIN algorithm=pso")
    pso_algo = PsoReal(
        is_stop=stop_never,
        validator=legacy_validator(genetic=False),
        input_params=fixed_params(),
        swarm_size=6,
        cache_path=str(state_dir / "pso_cache.json"),
    )
    pso_algo.audit_algorithm = "pso"
    for _ in range(rounds):
        pso_algo.run()
    print("MATMUL_FILTER_AUDIT_ALGORITHM_END algorithm=pso")

    print("MATMUL_FILTER_AUDIT_ALGORITHM_BEGIN algorithm=sa")
    sa_algo = SaReal(
        is_stop=stop_never,
        validator=legacy_validator(genetic=False),
        input_params=fixed_params(),
        cache_path=str(state_dir / "sa_cache.json"),
    )
    sa_algo.audit_algorithm = "sa"
    for _ in range(rounds * 6):
        sa_algo.run()
    print("MATMUL_FILTER_AUDIT_ALGORITHM_END algorithm=sa")


def summarize(
    log_dir: Path,
    summary_path: Path,
    *,
    seed: int | None = None,
    rounds: int | None = None,
) -> dict:
    counts: Counter[str] = Counter()
    by_algorithm: dict[str, Counter[str]] = {}
    total = 0
    for path in sorted(log_dir.glob("*.log"), key=lambda item: int(item.stem)):
        with path.open(encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                if record.get("record_type") != "candidate_audit":
                    continue
                total += 1
                classification = record["classification"]
                algorithm = record["algorithm"]
                counts[classification] += 1
                by_algorithm.setdefault(algorithm, Counter())[classification] += 1
    summary = {
        "schema": "matmul_cost_filter_audit_summary_v1",
        "record_type": "run_summary",
        "run_id": os.environ.get("MATMUL_AUDIT_RUN_ID"),
        "shape": {"M": 512, "N": 512, "K": 512},
        "seed": seed,
        "rounds": rounds,
        "total": total,
        "counts": dict(sorted(counts.items())),
        "by_algorithm": {
            name: dict(sorted(values.items()))
            for name, values in sorted(by_algorithm.items())
        },
        "attention": {
            "filter_false_negative": counts["rejected_correct_filter_false_negative"],
            "filter_false_positive": counts["accepted_wrong_filter_false_positive"],
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    seed = int(os.environ.get("MATMUL_AUDIT_SEED", "20260828"))
    rounds = int(os.environ.get("MATMUL_AUDIT_ROUNDS", "4"))
    log_dir = Path(os.environ["MATMUL_AUDIT_LOG_DIR"])
    state_dir = Path(os.environ["MATMUL_AUDIT_STATE_DIR"])
    random.seed(seed)
    np.random.seed(seed)
    run_campaign(rounds, state_dir)
    summary = summarize(
        log_dir,
        log_dir.parent / "summary.json",
        seed=seed,
        rounds=rounds,
    )
    base.BaseAlgo._append_audit(summary)
    print("MATMUL_FILTER_AUDIT_SUMMARY " + json.dumps(summary, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
