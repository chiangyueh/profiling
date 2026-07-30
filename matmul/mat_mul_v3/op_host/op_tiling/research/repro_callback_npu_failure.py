#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from callback import exact_roundtrip, invoke
from profile import (
    ProfileError,
    create_bank,
    discover_bank,
    run_runner,
    truthy,
)
from tiling_search.domain import KNOWLEDGE_FIELDS, Schedule, Workload


WORKLOAD_ID = "callback_ok_npu_undercoverage_tiny"
WORKLOAD = Workload(
    workload_id=WORKLOAD_ID,
    m=32,
    n=32,
    k=128,
    dtype="fp16",
    trans_a=False,
    trans_b=False,
    max_cores=20,
)

# This is a valid 23-field BASE record as far as the host callback and
# RuntimeKb lookup are concerned. It deliberately assigns only one 16x16
# output block to a 32x32 output so the NPU preflight can expose the missing
# runtime coverage.
SCHEDULE = Schedule.from_signature(
    "1:16:16:128:16:16:32:2:2:1:1:0:1:1:1:1:1:1:1:1:1:0:0"
)


def schedule_row() -> dict[str, str]:
    row = {
        "workload_id": WORKLOAD.workload_id,
        "m": str(WORKLOAD.m),
        "n": str(WORKLOAD.n),
        "k": str(WORKLOAD.k),
        "dtype": WORKLOAD.dtype,
        "trans_a": "1" if WORKLOAD.trans_a else "0",
        "trans_b": "1" if WORKLOAD.trans_b else "0",
    }
    row.update(
        {
            "used_core_num": str(SCHEDULE["usedCoreNum"]),
            "single_core_m": str(SCHEDULE["singleCoreM"]),
            "single_core_n": str(SCHEDULE["singleCoreN"]),
            "single_core_k": str(SCHEDULE["singleCoreK"]),
            "base_m": str(SCHEDULE["baseM"]),
            "base_n": str(SCHEDULE["baseN"]),
            "base_k": str(SCHEDULE["baseK"]),
            "depth_a1": str(SCHEDULE["depthA1"]),
            "depth_b1": str(SCHEDULE["depthB1"]),
            "step_m": str(SCHEDULE["stepM"]),
            "step_n": str(SCHEDULE["stepN"]),
            "iterate_order": str(SCHEDULE["iterateOrder"]),
            "step_ka": str(SCHEDULE["stepKa"]),
            "step_kb": str(SCHEDULE["stepKb"]),
            "db_l0a": str(SCHEDULE["dbL0A"]),
            "db_l0b": str(SCHEDULE["dbL0B"]),
            "db_l0c": str(SCHEDULE["dbL0C"]),
            "l2_m_tile_count": str(SCHEDULE["l2MTileCnt"]),
            "l2_n_tile_count": str(SCHEDULE["l2NTileCnt"]),
            "l2_m_tile_block": str(SCHEDULE["l2MTileBlock"]),
            "l2_n_tile_block": str(SCHEDULE["l2NTileBlock"]),
            "l2_iterate_order": str(SCHEDULE["l2IterateOrder"]),
            "tiling_enable": str(SCHEDULE["tilingEnable"]),
        }
    )
    return row


def write_workload_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=[
                "workload_id",
                "m",
                "n",
                "k",
                "dtype",
                "trans_a",
                "trans_b",
            ],
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(schedule_row())


def run_host_callback() -> None:
    # CANN initializes the Python RuntimeKb cache on the first callback. The
    # normal profiling path also performs this unmodified control lookup first.
    invoke(WORKLOAD)
    callback = exact_roundtrip(WORKLOAD, SCHEDULE)
    if callback.schedule.signature() != SCHEDULE.signature():
        raise RuntimeError("host callback did not preserve the exact schedule")
    print("HOST_CALLBACK_PASS")
    print(
        f"  workload={WORKLOAD_ID} "
        f"shape={WORKLOAD.m}x{WORKLOAD.n}x{WORKLOAD.k} "
        f"dtype={WORKLOAD.dtype} trans=00"
    )
    print("  host_gemm_executed=0")
    print(f"  exact_fields={len(KNOWLEDGE_FIELDS)}")
    print(f"  block_dim={callback.block_dim} tiling_key={callback.key}")
    print(f"  signature={SCHEDULE.signature_text()}")


def run_npu(args: argparse.Namespace) -> None:
    results = args.results.resolve()
    results.mkdir(parents=True, exist_ok=True)
    workload_csv = results / "workload.csv"
    write_workload_csv(workload_csv)

    env = dict(os.environ)
    schema_dir = results / "schema"
    schema_dir.mkdir(parents=True, exist_ok=True)
    spec = discover_bank(
        args.cann_root.resolve(),
        args.soc,
        args.aic,
        args.probe.resolve(),
        schema_dir,
        env,
    )
    candidate_env = create_bank(
        schedule_row(),
        spec,
        args.probe.resolve(),
        results / "runtime_bank",
        env,
    )
    print("RUNTIME_KB_PASS")
    print(
        f"  soc={spec.soc} aic={spec.aic} "
        f"knowledge_fields={len(KNOWLEDGE_FIELDS)} found=1"
    )

    try:
        measured = run_runner(
            args.runner.resolve(),
            workload_csv,
            WORKLOAD_ID,
            candidate_env,
            results / "npu_runner",
            timeout=args.timeout,
            warmup=0,
            repeat=1,
            samples=1,
        )
    except ProfileError as exception:
        measured = {
            "success": "0",
            "preflight_passed": "0",
            "preflight_mode": "runner_failed",
            "error": str(exception),
        }
    error = measured.get("error", "")
    evidence = {
        "workload_id": WORKLOAD_ID,
        "shape": [WORKLOAD.m, WORKLOAD.n, WORKLOAD.k],
        "dtype": WORKLOAD.dtype,
        "trans_a": WORKLOAD.trans_a,
        "trans_b": WORKLOAD.trans_b,
        "tiling_signature": SCHEDULE.signature_text(),
        "host_callback_exact_fields": len(KNOWLEDGE_FIELDS),
        "runtime_kb_found": True,
        "npu_success": truthy(measured.get("success")),
        "npu_preflight_passed": truthy(measured.get("preflight_passed")),
        "npu_preflight_mode": measured.get("preflight_mode", ""),
        "npu_error": error,
    }
    (results / "repro_result.json").write_text(
        json.dumps(evidence, indent=2) + "\n",
        encoding="utf-8",
    )

    if evidence["npu_success"] or evidence["npu_preflight_passed"]:
        print("REPRO_NOT_CONFIRMED")
        print("  callback and RuntimeKb accepted the record, but NPU passed")
        raise SystemExit(1)

    expected_failure = (
        "official numeric preflight failed" in error
        or "official output coverage failed" in error
    )
    if not expected_failure:
        print("REPRO_INCONCLUSIVE")
        print("  NPU execution did not reach the expected output check")
        print(f"  mode={evidence['npu_preflight_mode']}")
        print(f"  error={error}")
        raise SystemExit(1)

    print("NPU_PREFLIGHT_EXPECTED_FAILURE")
    print(f"  mode={evidence['npu_preflight_mode']}")
    print(f"  error={error}")
    print("REPRO_CONFIRMED")
    print("  callback_exact_roundtrip=1 runtime_kb_found=1 npu_preflight_passed=0")
    print(f"  evidence={results / 'repro_result.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("host", "npu"), required=True)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--cann-root", type=Path)
    parser.add_argument("--soc")
    parser.add_argument("--aic", type=int)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--timeout", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "host":
        run_host_callback()
        return
    missing = [
        name
        for name in ("runner", "probe", "cann_root", "soc", "aic", "results")
        if getattr(args, name) is None
    ]
    if missing:
        raise SystemExit("missing NPU arguments: " + ",".join(missing))
    run_npu(args)


if __name__ == "__main__":
    main()
