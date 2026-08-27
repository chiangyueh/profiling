#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import profile_official_tilings as bank_api
import refine_matmul_v3_candidates as validator


WORKLOAD_ID = "bench_v2_127_l1_residency"
HISTORY_RECORD_ID = (
    "6ae3c1554538f198a3ad6fb1958fb555921cd23b497308c582aad6a2216bc9e6"
)


def candidate_row() -> dict[str, str]:
    values = (
        16, 64, 112, 24576, 64, 64, 256, 6, 6, 1, 1, 0,
        3, 3, 2, 2, 1, 1, 1, 2, 8, 0, 0,
    )
    if len(values) != len(validator.KNOWLEDGE_FIELDS):
        raise RuntimeError("counterexample knowledge schema length mismatch")
    knowledge = dict(zip(validator.KNOWLEDGE_FIELDS, values))
    row = {
        "workload_id": WORKLOAD_ID,
        "m": "128",
        "n": "896",
        "k": "24576",
        "dtype": "bf16",
        "trans_a": "1",
        "trans_b": "1",
        "max_cores": "20",
        "candidate_role": "searched",
        "valid": "1",
        "official_return": "0",
        "official_core_num": "16",
        "execution_mode": "base_iterate_all",
        "callback_kernel_suffix": "1",
        "used_core_num": str(knowledge["usedCoreNum"]),
        "single_core_m": str(knowledge["singleCoreM"]),
        "single_core_n": str(knowledge["singleCoreN"]),
        "single_core_k": str(knowledge["singleCoreK"]),
        "base_m": str(knowledge["baseM"]),
        "base_n": str(knowledge["baseN"]),
        "base_k": str(knowledge["baseK"]),
        "depth_a1": str(knowledge["depthA1"]),
        "depth_b1": str(knowledge["depthB1"]),
        "step_m": str(knowledge["stepM"]),
        "step_n": str(knowledge["stepN"]),
        "iterate_order": str(knowledge["iterateOrder"]),
        "step_ka": str(knowledge["stepKa"]),
        "step_kb": str(knowledge["stepKb"]),
        "db_l0a": str(knowledge["dbL0A"]),
        "db_l0b": str(knowledge["dbL0B"]),
        "db_l0c": str(knowledge["dbL0C"]),
        "bank_l2_m_tile_count": str(knowledge["l2MTileCnt"]),
        "bank_l2_n_tile_count": str(knowledge["l2NTileCnt"]),
        "bank_l2_m_tile_block": str(knowledge["l2MTileBlock"]),
        "bank_l2_n_tile_block": str(knowledge["l2NTileBlock"]),
        "bank_l2_iterate_order": str(knowledge["l2IterateOrder"]),
        "bank_tiling_enable": str(knowledge["tilingEnable"]),
    }
    return row


def write_unchecked_bank(
    row: dict[str, str],
    spec: bank_api.BankSpec,
    probe: Path,
    bank_root: Path,
    cache_root: Path,
) -> tuple[dict[str, str], str]:
    """Replay one known-bad historical row without passing the validator."""
    info = bank_api.make_info(row)
    knowledge = bank_api.make_knowledge(row)
    key_path = bank_root / "matmul_v3_input.bin"
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_bytes(bank_api.pack_info(info))
    record_id = bank_api.probe_hash(probe, key_path)
    record = {
        "id": record_id,
        "info_dict": info,
        "knowledge": knowledge,
        "op": "MatMulV3",
        "version": spec.version,
    }
    bank_file = bank_root / spec.soc / "unified_bank" / spec.filename
    bank_file.parent.mkdir(parents=True, exist_ok=True)
    bank_file.write_text(
        json.dumps(record, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["TUNE_BANK_PATH"] = str(bank_root)
    env["ASCEND_CACHE_PATH"] = str(cache_root)
    query = subprocess.run(
        [str(probe), "--query", str(key_path), spec.soc, str(spec.aic_cores)],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if query.returncode != 0 or "found=1" not in query.stdout:
        raise RuntimeError(
            "private RuntimeKb record was not selected by the bank probe: " +
            query.stdout.strip()
        )
    return env, query.stdout.strip()


def write_workload_csv(path: Path, row: dict[str, str]) -> None:
    fields = ["workload_id", "m", "n", "k", "dtype", "trans_a", "trans_b"]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerow({field: row[field] for field in fields})


def read_only_profile_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 1:
        raise RuntimeError(f"expected one runner profile row, observed {len(rows)}")
    return rows[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--bank-probe", type=Path, required=True)
    parser.add_argument("--cann-root", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--soc", default="Ascend910B3")
    parser.add_argument("--aic-cores", type=int, default=20)
    args = parser.parse_args()

    args.state_dir.mkdir(parents=True, exist_ok=True)
    row = candidate_row()
    spec = bank_api.discover_bank(
        args.cann_root,
        args.soc,
        args.aic_cores,
        64 * 1024,
        64 * 1024,
        256 * 1024,
        512 * 1024,
        args.bank_probe,
        args.state_dir,
    )

    knowledge = bank_api.make_knowledge(row)
    workload = validator.Workload(
        workload_id=row["workload_id"],
        m=int(row["m"]),
        n=int(row["n"]),
        k=int(row["k"]),
        dtype=row["dtype"],
        trans_a=True,
        trans_b=True,
        max_cores=20,
    )
    hardware = validator.Hardware(
        aic_cores=args.aic_cores,
        l0a_bytes=spec.l0a_bytes,
        l0b_bytes=spec.l0b_bytes,
        l0c_bytes=spec.l0c_bytes,
        l1_bytes=spec.l1_bytes,
        l2_bytes=192 * 1024 * 1024,
        l2_bytes_per_cycle_per_core=1.0,
        hbm_bytes_per_cycle_per_core=1.0,
    )
    if validator.hard_legal(workload, knowledge, hardware):
        raise RuntimeError("current hard_legal unexpectedly accepted the counterexample")
    if not (
        knowledge["singleCoreN"] > knowledge["baseN"] and
        validator.l2_base_schedule_legal(workload, knowledge)
    ):
        raise RuntimeError("counterexample no longer isolates the BASE N geometry rule")
    try:
        bank_api.validate_candidate(row, spec)
    except bank_api.ProfileError as exception:
        validator_error = str(exception)
    else:
        raise RuntimeError("profile admission path unexpectedly accepted the counterexample")

    bank_root = args.state_dir / "private_bank"
    cache_root = args.state_dir / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    env, query = write_unchecked_bank(
        row, spec, args.bank_probe, bank_root, cache_root
    )
    workload_csv = args.state_dir / "workload.csv"
    profile_csv = args.state_dir / "profile.csv"
    samples_csv = args.state_dir / "samples.csv"
    runner_log = args.state_dir / "runner.log"
    write_workload_csv(workload_csv, row)
    command = [
        str(args.runner),
        "--candidates", str(workload_csv),
        "--output", str(profile_csv),
        "--samples-output", str(samples_csv),
        "--device", "0",
        "--warmup", "0",
        "--repeat", "1",
        "--samples", "1",
        "--numeric-preflight-max-mib", "64",
        "--structured-full-preflight",
        "--preflight-only",
    ]
    completed = subprocess.run(
        command,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    runner_log.write_text(completed.stdout, encoding="utf-8")
    profile = read_only_profile_row(profile_csv)
    mismatch = profile.get("error", "")
    if (
        completed.returncode == 0 or
        profile.get("preflight_passed") not in {"0", "false"} or
        "official structured numeric preflight failed" not in mismatch
    ):
        raise RuntimeError(
            "the unchecked candidate did not reproduce its historical numeric "
            f"failure; rc={completed.returncode} error={mismatch!r}; "
            f"details={runner_log}"
        )

    validator_line = inspect.getsourcelines(validator.hard_legal)[1]
    print(
        "MATMUL_RUNTIMEKB_REPLAY_ROUTE "
        f"status=passed runtime_kb_found=1 callback=official_aclnnMatmul "
        f"record_id={HISTORY_RECORD_ID}"
    )
    print(
        "MATMUL_RUNTIMEKB_REPLAY_UNCHECKED "
        f"status=wrong_output shape=128x896x24576 dtype=bf16 trans=11 "
        f"detail={json.dumps(mismatch, ensure_ascii=False)}"
    )
    print(
        "MATMUL_RUNTIMEKB_REPLAY_CHECKED "
        "status=rejected reason=BASE_SINGLE_CORE_N_EXCEEDS_BASE_N "
        f"singleCoreN={knowledge['singleCoreN']} baseN={knowledge['baseN']} "
        "kernel_launched=0"
    )
    print(
        "MATMUL_RUNTIMEKB_REPLAY_VALIDATOR "
        f"source={Path(validator.__file__).resolve()}:{validator_line} "
        f"entry=hard_legal admission={json.dumps(validator_error)}"
    )
    print(f"MATMUL_RUNTIMEKB_REPLAY_LOG {runner_log}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exception:
        print(f"MATMUL_RUNTIMEKB_REPLAY_FAILED {exception}", file=sys.stderr)
        raise SystemExit(1)
