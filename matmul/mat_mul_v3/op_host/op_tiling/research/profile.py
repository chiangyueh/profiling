#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import signal
import struct
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


KNOWLEDGE_COLUMNS = {
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
    "l2MTileCnt": "l2_m_tile_count",
    "l2NTileCnt": "l2_n_tile_count",
    "l2MTileBlock": "l2_m_tile_block",
    "l2NTileBlock": "l2_n_tile_block",
    "l2IterateOrder": "l2_iterate_order",
    "tilingEnable": "tiling_enable",
}

MEASUREMENT_COLUMNS = [
    "record_id",
    "run_id",
    "soc",
    "aic",
    "toolkit",
    "workload_id",
    "m",
    "n",
    "k",
    "dtype",
    "trans_a",
    "trans_b",
    "rank",
    "candidate_role",
    "candidate_source",
    "search_template",
    "search_acquisition",
    "search_behavior_metrics",
    "tiling_signature",
    "success",
    "preflight_passed",
    "preflight_mode",
    "error",
    "median_ms",
    "stddev_ms",
    "official_ms",
    "official_stddev_ms",
    "bank_ms",
    "bank_stddev_ms",
    "speedup_vs_official",
    "speedup_vs_bank",
    "status_vs_official",
    "status_vs_bank",
    "pair_validated",
    "official_post_ms",
    "bank_post_ms",
    "official_drift_pct",
    "bank_drift_pct",
]

STRICT_NUMERIC_PREFLIGHT_MODES = {
    "numeric_ones_full_v2",
    "numeric_signed_axes_full_v3",
}

SUMMARY_COLUMNS = [
    "workload_id",
    "m",
    "n",
    "k",
    "dtype",
    "trans_a",
    "trans_b",
    "searched",
    "successful",
    "runtime_rejected",
    "best_rank",
    "best_template",
    "best_source",
    "best_signature",
    "best_ms",
    "official_ms",
    "bank_ms",
    "speedup_vs_official",
    "speedup_vs_bank",
    "status_vs_official",
    "status_vs_bank",
    "optimization_result",
]


class ProfileError(RuntimeError):
    pass


@dataclass(frozen=True)
class BankSpec:
    soc: str
    aic: int
    filename: str
    version: int


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def write_csv(
    path: Path,
    columns: list[str],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=columns, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
        destination.flush()
        os.fsync(destination.fileno())
    temporary.replace(path)


def pack_info(info: dict) -> bytes:
    values = [
        int(info["m"]),
        int(info["k"]),
        int(info["n"]),
        int(info["batch_a1"]),
        int(info["batch_a2"]),
        int(info["batch_a3"]),
        int(info["batch_a4"]),
        int(info["batch_b1"]),
        int(info["batch_b2"]),
        int(info["batch_b3"]),
        int(info["batch_b4"]),
        float(info["l1_fused_num"]),
        float(info["aub_double_num"]),
        float(info["bub_double_num"]),
        float(info["fused_double_operand_num"]),
        int(info["a_dtype"]),
        int(info["b_dtype"]),
        int(info["out_dtype"]),
        int(info["a_format"]),
        int(info["b_format"]),
        int(info["out_format"]),
        bool(info["trans_a_flag"]),
        bool(info["trans_b_flag"]),
        bool(info["bias_flag"]),
        bool(info["reserved_bool"]),
        bool(info["m_align_flag"]),
        bool(info["k_align_flag"]),
        bool(info["n_align_flag"]),
        int(info["reserved_params1"]),
        int(info["reserved_params2"]),
        int(info["reserved_params3"]),
        int(info["reserved_params4"]),
        int(info["reserved_params5"]),
        int(info["reserved_params6"]),
    ]
    result = struct.pack("<11q4f6i7?6Q", *values)
    if len(result) != 183:
        raise ProfileError(f"MatMulV3 key has {len(result)} bytes, expected 183")
    return result


def make_info(row: dict[str, str]) -> dict:
    dtype = row["dtype"].lower()
    if dtype not in {"fp16", "bf16", "fp32"}:
        raise ProfileError(f"unsupported dtype={dtype}")
    dtype_code = 0 if dtype == "fp32" else 1
    k_alignment = 8 if dtype == "fp32" else 16
    m = int(row["m"])
    n = int(row["n"])
    k = int(row["k"])
    aligned_m = align_up(m, 16)
    aligned_n = align_up(n, 16)
    aligned_k = align_up(k, k_alignment)
    return {
        "a_dtype": dtype_code,
        "a_format": 2,
        "aub_double_num": 1.0,
        "b_dtype": dtype_code,
        "b_format": 2,
        "batch_a1": 1,
        "batch_a2": 1,
        "batch_a3": 1,
        "batch_a4": 1,
        "batch_b1": 1,
        "batch_b2": 1,
        "batch_b3": 1,
        "batch_b4": 1,
        "bias_flag": False,
        "bub_double_num": 1.0,
        "fused_double_operand_num": 0.0,
        "k": aligned_k,
        "k_align_flag": k == aligned_k,
        "l1_fused_num": 0.0,
        "m": aligned_m,
        "m_align_flag": m == aligned_m,
        "n": aligned_n,
        "n_align_flag": n == aligned_n,
        "out_dtype": dtype_code,
        "out_format": 2,
        "reserved_bool": False,
        "reserved_params1": 0,
        "reserved_params2": 0,
        "reserved_params3": 0,
        "reserved_params4": 0,
        "reserved_params5": 0,
        "reserved_params6": 0,
        "trans_a_flag": truthy(row.get("trans_a")),
        "trans_b_flag": truthy(row.get("trans_b")),
    }


def make_knowledge(row: dict[str, str]) -> dict[str, int]:
    try:
        return {
            field: int(row[column])
            for field, column in KNOWLEDGE_COLUMNS.items()
        }
    except (KeyError, ValueError) as exception:
        raise ProfileError(
            f"{row.get('workload_id')} has incomplete 23-field schedule"
        ) from exception


def probe_hash(probe: Path, key_path: Path, env: dict[str, str]) -> int:
    result = subprocess.run(
        [str(probe), "--hash", str(key_path)],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode:
        raise ProfileError(result.stdout.strip())
    return int(result.stdout.strip())


def discover_bank(
    cann_root: Path,
    soc: str,
    aic: int,
    probe: Path,
    work_dir: Path,
    env: dict[str, str],
) -> BankSpec:
    bank_dir = (
        cann_root
        / "opp/built-in/data/op"
        / soc
        / "unified_bank"
    )
    filename = f"{soc}_{aic}_AiCore_MatMulV3_runtime_kb.json"
    path = bank_dir / filename
    if not path.is_file():
        raise ProfileError(f"installed MatMulV3 runtime bank is missing: {path}")
    first = next(
        line for line in path.read_text(encoding="utf-8").splitlines() if line
    )
    record = json.loads(first)
    if record.get("op") != "MatMulV3":
        raise ProfileError(f"unexpected runtime bank op in {path}")
    if set(record.get("knowledge", {})) != set(KNOWLEDGE_COLUMNS):
        raise ProfileError("installed RuntimeKb does not use the official 23 fields")
    key_path = work_dir / "schema_key.bin"
    key_path.write_bytes(pack_info(record["info_dict"]))
    computed = probe_hash(probe, key_path, env)
    if computed != int(record["id"]):
        raise ProfileError(
            f"RuntimeKb ABI mismatch stored={record['id']} computed={computed}"
        )
    return BankSpec(soc, aic, filename, int(record.get("version", 0)))


def create_bank(
    row: dict[str, str],
    spec: BankSpec,
    probe: Path,
    root: Path,
    env: dict[str, str],
) -> dict[str, str]:
    info = make_info(row)
    knowledge = make_knowledge(row)
    key = pack_info(info)
    key_path = root / "input.bin"
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_bytes(key)
    record = {
        "id": probe_hash(probe, key_path, env),
        "info_dict": info,
        "knowledge": knowledge,
        "op": "MatMulV3",
        "version": spec.version,
    }
    bank_path = root / spec.soc / "unified_bank" / spec.filename
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    bank_path.write_text(
        json.dumps(record, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    candidate_env = dict(env)
    candidate_env["TUNE_BANK_PATH"] = str(root)
    candidate_env["ASCEND_CACHE_PATH"] = str(root / "cache")
    query = subprocess.run(
        [
            str(probe),
            "--query",
            str(key_path),
            spec.soc,
            str(spec.aic),
        ],
        env=candidate_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if query.returncode or "found=1" not in query.stdout:
        raise ProfileError(
            "RuntimeKb rejected exact candidate: " + query.stdout.strip()
        )
    return candidate_env


def run_runner(
    runner: Path,
    candidates: Path,
    workload_id: str,
    env: dict[str, str],
    work_dir: Path,
    timeout: int,
    warmup: int,
    repeat: int,
    samples: int,
    numeric_preflight_max_mib: int,
    require_numeric_preflight: bool,
) -> dict[str, str]:
    work_dir.mkdir(parents=True, exist_ok=True)
    output = work_dir / "profile.csv"
    sample_output = work_dir / "samples.csv"
    command = [
        str(runner),
        "--candidates",
        str(candidates),
        "--output",
        str(output),
        "--samples-output",
        str(sample_output),
        "--only-workload",
        workload_id,
        "--warmup",
        str(warmup),
        "--repeat",
        str(repeat),
        "--samples",
        str(samples),
        "--numeric-preflight-max-mib",
        str(numeric_preflight_max_mib),
    ]
    if require_numeric_preflight:
        command.append("--require-numeric-preflight")
    process = subprocess.Popen(
        command,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        stdout, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
        return {
            "success": "0",
            "preflight_passed": "0",
            "preflight_mode": "timeout",
            "error": f"MatMulV3 timeout after {timeout}s",
            "median_ms": "",
            "stddev_ms": "",
        }
    if not output.is_file():
        return {
            "success": "0",
            "preflight_passed": "0",
            "preflight_mode": "runner_failed",
            "error": stdout.strip()[-1000:],
            "median_ms": "",
            "stddev_ms": "",
        }
    rows = read_csv(output)
    if not rows:
        raise ProfileError(f"runner returned no row for {workload_id}")
    row = rows[0]
    if process.returncode and truthy(row.get("success")):
        row["success"] = "0"
        row["error"] = stdout.strip()[-1000:]
    return row


def comparison(
    candidate_ms: float,
    candidate_std: float,
    baseline_ms: float,
    baseline_std: float,
) -> tuple[float, str]:
    speedup = baseline_ms / candidate_ms
    delta = 100.0 * (candidate_ms - baseline_ms) / baseline_ms
    noise = max(
        1.0,
        200.0 * math.hypot(candidate_std, baseline_std) / baseline_ms,
    )
    if delta < -noise:
        return speedup, "improved"
    if delta > noise:
        return speedup, "regressed"
    return speedup, "within_noise"


def measurement_reusable(row: dict[str, str]) -> bool:
    if not truthy(row.get("success")):
        return row.get("preflight_mode") not in {
            "",
            "baseline_drift",
            "provisional",
            "runner_failed",
        }
    return (
        truthy(row.get("pair_validated"))
        and row.get("preflight_mode") in STRICT_NUMERIC_PREFLIGHT_MODES
        and bool(row.get("official_ms"))
        and bool(row.get("bank_ms"))
    )


def measurement_completed(row: dict[str, str]) -> bool:
    """Return whether an exact fingerprint must never be measured again."""
    if not truthy(row.get("success")):
        return measurement_reusable(row)
    return (
        truthy(row.get("preflight_passed"))
        and row.get("preflight_mode") in STRICT_NUMERIC_PREFLIGHT_MODES
        and bool(row.get("median_ms"))
    )


def baseline_drift_pct(
    before: dict[str, str],
    after: dict[str, str],
) -> float:
    before_ms = float(before["median_ms"])
    after_ms = float(after["median_ms"])
    return 100.0 * abs(after_ms - before_ms) / min(before_ms, after_ms)


def conservative_reference(
    before: dict[str, str],
    after: dict[str, str],
) -> dict[str, str]:
    result = dict(before)
    result["median_ms"] = f"{min(float(before['median_ms']), float(after['median_ms'])):.12g}"
    result["stddev_ms"] = f"{max(float(before['stddev_ms']), float(after['stddev_ms'])):.12g}"
    return result


def measurement_key(
    soc: str,
    aic: int,
    toolkit: str,
    row: dict[str, str],
) -> str:
    value = "|".join(
        (
            soc,
            str(aic),
            toolkit,
            row["workload_id"],
            row["m"],
            row["n"],
            row["k"],
            row["dtype"],
            row["trans_a"],
            row["trans_b"],
            row["candidate_role"],
            row["tiling_signature"],
        )
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def profile_record(
    source: dict[str, str],
    measured: dict[str, str],
    *,
    record_id: str,
    run_id: str,
    soc: str,
    aic: int,
    toolkit: str,
    official: dict[str, str] | None = None,
    bank: dict[str, str] | None = None,
    official_post: dict[str, str] | None = None,
    bank_post: dict[str, str] | None = None,
    pair_validated: bool = False,
) -> dict[str, str]:
    row = {column: "" for column in MEASUREMENT_COLUMNS}
    for column in (
        "workload_id",
        "m",
        "n",
        "k",
        "dtype",
        "trans_a",
        "trans_b",
        "rank",
        "candidate_role",
        "candidate_source",
        "search_template",
        "search_acquisition",
        "search_behavior_metrics",
        "tiling_signature",
    ):
        row[column] = source.get(column, "")
    row.update(
        {
            "record_id": record_id,
            "run_id": run_id,
            "soc": soc,
            "aic": str(aic),
            "toolkit": toolkit,
            "success": measured.get("success", "0"),
            "preflight_passed": measured.get("preflight_passed", "0"),
            "preflight_mode": measured.get("preflight_mode", ""),
            "error": measured.get("error", ""),
            "median_ms": measured.get("median_ms", ""),
            "stddev_ms": measured.get("stddev_ms", ""),
            "pair_validated": str(int(pair_validated)),
        }
    )
    if official_post is not None and bank_post is not None:
        row.update(
            {
                "official_post_ms": official_post.get("median_ms", ""),
                "bank_post_ms": bank_post.get("median_ms", ""),
                "official_drift_pct": (
                    f"{baseline_drift_pct(official, official_post):.12g}"
                    if official is not None
                    else ""
                ),
                "bank_drift_pct": (
                    f"{baseline_drift_pct(bank, bank_post):.12g}"
                    if bank is not None
                    else ""
                ),
            }
        )
    if (
        source.get("candidate_role") == "searched"
        and official is not None
        and bank is not None
        and truthy(measured.get("success"))
    ):
        candidate_ms = float(measured["median_ms"])
        candidate_std = float(measured["stddev_ms"])
        official_ms = float(official["median_ms"])
        official_std = float(official["stddev_ms"])
        bank_ms = float(bank["median_ms"])
        bank_std = float(bank["stddev_ms"])
        speedup_official, status_official = comparison(
            candidate_ms, candidate_std, official_ms, official_std
        )
        speedup_bank, status_bank = comparison(
            candidate_ms, candidate_std, bank_ms, bank_std
        )
        row.update(
            {
                "official_ms": f"{official_ms:.12g}",
                "official_stddev_ms": f"{official_std:.12g}",
                "bank_ms": f"{bank_ms:.12g}",
                "bank_stddev_ms": f"{bank_std:.12g}",
                "speedup_vs_official": f"{speedup_official:.12g}",
                "speedup_vs_bank": f"{speedup_bank:.12g}",
                "status_vs_official": status_official,
                "status_vs_bank": status_bank,
            }
        )
    return row


def summarize(
    candidates: list[dict[str, str]],
    records: dict[str, dict[str, str]],
    soc: str,
    aic: int,
    toolkit: str,
) -> list[dict[str, str]]:
    workloads: dict[str, list[dict[str, str]]] = {}
    for row in candidates:
        if row["candidate_role"] == "searched":
            workloads.setdefault(row["workload_id"], []).append(row)
    summaries = []
    for workload_id, rows in workloads.items():
        measured = [
            records.get(measurement_key(soc, aic, toolkit, row))
            for row in rows
        ]
        successful = [
            row for row in measured
            if row is not None
            and truthy(row.get("success"))
            and measurement_reusable(row)
        ]
        rejected = sum(row is not None and not truthy(row.get("success")) for row in measured)
        def paired_ratio(row: dict[str, str]) -> tuple[float, float]:
            candidate_ms = float(row["median_ms"])
            return (
                max(
                    candidate_ms / float(row["official_ms"]),
                    candidate_ms / float(row["bank_ms"]),
                ),
                candidate_ms,
            )

        best = min(successful, key=paired_ratio, default=None)
        source = rows[0]
        summary = {column: "" for column in SUMMARY_COLUMNS}
        summary.update(
            {
                "workload_id": workload_id,
                "m": source["m"],
                "n": source["n"],
                "k": source["k"],
                "dtype": source["dtype"],
                "trans_a": source["trans_a"],
                "trans_b": source["trans_b"],
                "searched": str(len(rows)),
                "successful": str(len(successful)),
                "runtime_rejected": str(rejected),
            }
        )
        if best is None:
            summary["optimization_result"] = "no_successful_candidate"
        else:
            summary.update(
                {
                    "best_rank": best["rank"],
                    "best_template": best["search_template"],
                    "best_source": best["candidate_source"],
                    "best_signature": best["tiling_signature"],
                    "best_ms": best["median_ms"],
                    "official_ms": best["official_ms"],
                    "bank_ms": best["bank_ms"],
                    "speedup_vs_official": best["speedup_vs_official"],
                    "speedup_vs_bank": best["speedup_vs_bank"],
                    "status_vs_official": best["status_vs_official"],
                    "status_vs_bank": best["status_vs_bank"],
                    "optimization_result": (
                        "improved"
                        if best["status_vs_official"] == "improved"
                        and best["status_vs_bank"] == "improved"
                        else "not_improved"
                    ),
                }
            )
        summaries.append(summary)
        print(
            "WORKLOAD_RESULT "
            f"{workload_id} best_rank={summary['best_rank'] or 'none'} "
            f"best_ms={summary['best_ms'] or 'NA'} "
            f"official_ms={summary['official_ms'] or 'NA'} "
            f"bank_ms={summary['bank_ms'] or 'NA'} "
            f"speedup={summary['speedup_vs_official'] or 'NA'} "
            f"speedup_vs_bank={summary['speedup_vs_bank'] or 'NA'} "
            f"optimization_result={summary['optimization_result']}"
        )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--candidate-results", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--cann-root", type=Path, required=True)
    parser.add_argument("--soc", required=True)
    parser.add_argument("--aic", type=int, required=True)
    parser.add_argument("--toolkit", required=True)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--baseline-repeat", type=int, default=30)
    parser.add_argument("--baseline-samples", type=int, default=9)
    parser.add_argument("--numeric-preflight-max-mib", type=int, default=256)
    parser.add_argument("--baseline-drift-pct", type=float, default=3.0)
    parser.add_argument("--pair-block-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pair_block_size <= 0:
        raise ProfileError("--pair-block-size must be positive")
    if args.baseline_repeat <= 0 or args.baseline_samples <= 0:
        raise ProfileError("baseline repeat and samples must be positive")
    env = os.environ.copy()
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.candidates)
    controls = {
        row["workload_id"]: row
        for row in rows
        if row["candidate_role"] == "bank_seed_control"
    }
    searched = [row for row in rows if row["candidate_role"] == "searched"]
    existing = read_csv(args.resume) if args.resume.is_file() else []
    records = {
        row["record_id"]: row
        for row in existing
        if measurement_completed(row)
    }
    run_id = time.strftime("%Y%m%d_%H%M%S")

    with tempfile.TemporaryDirectory(
        prefix=".matmul_v3_profile_", dir=args.summary.parent
    ) as temporary:
        work = Path(temporary)
        spec = discover_bank(
            args.cann_root, args.soc, args.aic, args.probe, work, env
        )
        print(
            f"bank_schema: soc={spec.soc} aic={spec.aic} "
            "input_bytes=183 knowledge_fields=23"
        )
        by_workload: dict[str, list[dict[str, str]]] = {}
        for row in searched:
            by_workload.setdefault(row["workload_id"], []).append(row)
        pending_total = sum(
            measurement_key(
                args.soc, args.aic, args.toolkit, row
            ) not in records
            for row in searched
        )
        print(
            f"profile_plan: workloads={len(by_workload)} "
            f"searched_candidates={len(searched)} "
            f"npu_searched_pending={pending_total} "
            f"resume_exact={len(records)} "
            f"paired_measurement=blocked_pre_post:{args.pair_block_size} "
            f"baseline_sampling={args.baseline_repeat}x"
            f"{args.baseline_samples} numeric_preflight=full"
        )

        for workload_id, workload_rows in by_workload.items():
            pending = [
                row
                for row in workload_rows
                if measurement_key(
                    args.soc, args.aic, args.toolkit, row
                ) not in records
            ]
            if not pending:
                continue
            if workload_id not in controls:
                raise ProfileError(f"{workload_id}: missing bank control")
            control = controls[workload_id]
            pair_dir = work / workload_id
            empty_bank = pair_dir / "empty_bank"
            empty_bank.mkdir(parents=True, exist_ok=True)
            official_env = dict(env)
            official_env["TUNE_BANK_PATH"] = str(empty_bank)
            official_env["ASCEND_CACHE_PATH"] = str(pair_dir / "official_cache")
            bank_env = create_bank(
                control,
                spec,
                args.probe,
                pair_dir / "bank_control",
                env,
            )
            official_before: dict[str, str] | None = None
            bank_before: dict[str, str] | None = None
            for block_start in range(
                0, len(pending), args.pair_block_size
            ):
                block = pending[
                    block_start:block_start + args.pair_block_size
                ]
                block_number = block_start // args.pair_block_size + 1
                block_dir = pair_dir / f"pair_block_{block_number}"
                if official_before is None:
                    official_before = run_runner(
                        args.runner,
                        args.candidates,
                        workload_id,
                        official_env,
                        block_dir / "official_pre",
                        args.timeout,
                        args.warmup,
                        args.baseline_repeat,
                        args.baseline_samples,
                        args.numeric_preflight_max_mib,
                        True,
                    )
                official = official_before
                if not truthy(official.get("success")):
                    raise ProfileError(
                        f"{workload_id}: official baseline failed: "
                        f"{official.get('error')}"
                    )
                if bank_before is None:
                    bank_before = run_runner(
                        args.runner,
                        args.candidates,
                        workload_id,
                        bank_env,
                        block_dir / "bank_pre",
                        args.timeout,
                        args.warmup,
                        args.baseline_repeat,
                        args.baseline_samples,
                        args.numeric_preflight_max_mib,
                        True,
                    )
                bank = bank_before
                if not truthy(bank.get("success")):
                    raise ProfileError(
                        f"{workload_id}: bank control failed: "
                        f"{bank.get('error')}"
                    )

                block_measurements: list[
                    tuple[dict[str, str], dict[str, str], str, int]
                ] = []
                for block_offset, candidate in enumerate(block):
                    index = block_start + block_offset + 1
                    record_id = measurement_key(
                        args.soc, args.aic, args.toolkit, candidate
                    )
                    candidate_dir = (
                        block_dir / f"candidate_{candidate['rank']}"
                    )
                    try:
                        candidate_env = create_bank(
                            candidate,
                            spec,
                            args.probe,
                            candidate_dir / "bank",
                            env,
                        )
                        measured = run_runner(
                            args.runner,
                            args.candidates,
                            workload_id,
                            candidate_env,
                            candidate_dir / "runner",
                            args.timeout,
                            args.warmup,
                            args.repeat,
                            args.samples,
                            args.numeric_preflight_max_mib,
                            True,
                        )
                    except Exception as exception:
                        measured = {
                            "success": "0",
                            "preflight_passed": "0",
                            "preflight_mode": "runtime_rejected",
                            "error": str(exception),
                            "median_ms": "",
                            "stddev_ms": "",
                        }
                    # Persist exact correctness/executability immediately. If
                    # the process is interrupted before post controls, this
                    # fingerprint remains completed but is never rankable.
                    provisional = profile_record(
                        candidate,
                        measured,
                        record_id=record_id,
                        run_id=run_id,
                        soc=args.soc,
                        aic=args.aic,
                        toolkit=args.toolkit,
                        official=official,
                        bank=bank,
                    )
                    records[record_id] = provisional
                    block_measurements.append(
                        (candidate, measured, record_id, index)
                    )
                    write_csv(
                        args.resume,
                        MEASUREMENT_COLUMNS,
                        list(records.values()),
                    )

                official_post = run_runner(
                    args.runner,
                    args.candidates,
                    workload_id,
                    official_env,
                    block_dir / "official_post",
                    args.timeout,
                    args.warmup,
                    args.baseline_repeat,
                    args.baseline_samples,
                    args.numeric_preflight_max_mib,
                    True,
                )
                bank_post = run_runner(
                    args.runner,
                    args.candidates,
                    workload_id,
                    bank_env,
                    block_dir / "bank_post",
                    args.timeout,
                    args.warmup,
                    args.baseline_repeat,
                    args.baseline_samples,
                    args.numeric_preflight_max_mib,
                    True,
                )
                if not truthy(official_post.get("success")):
                    raise ProfileError(
                        f"{workload_id}: post official baseline failed: "
                        f"{official_post.get('error')}"
                    )
                if not truthy(bank_post.get("success")):
                    raise ProfileError(
                        f"{workload_id}: post bank control failed: "
                        f"{bank_post.get('error')}"
                    )
                # The post controls are already the nearest measurements
                # before the next block. Reusing them avoids an immediate
                # duplicate control run without weakening the bracket.
                official_before = official_post
                bank_before = bank_post
                official_drift = baseline_drift_pct(
                    official, official_post
                )
                bank_drift = baseline_drift_pct(bank, bank_post)
                pair_valid = (
                    official_drift <= args.baseline_drift_pct
                    and bank_drift <= args.baseline_drift_pct
                )
                print(
                    f"PAIR_REFERENCE {workload_id} "
                    f"block={block_number} candidates={len(block)} "
                    f"official_ms={official['median_ms']}/"
                    f"{official_post['median_ms']} "
                    f"bank_ms={bank['median_ms']}/"
                    f"{bank_post['median_ms']} "
                    f"drift_pct={official_drift:.6g}/{bank_drift:.6g} "
                    f"validated={int(pair_valid)}"
                )
                official_reference = conservative_reference(
                    official, official_post
                )
                bank_reference = conservative_reference(bank, bank_post)
                for candidate, measured, record_id, index in block_measurements:
                    record = profile_record(
                        candidate,
                        measured,
                        record_id=record_id,
                        run_id=run_id,
                        soc=args.soc,
                        aic=args.aic,
                        toolkit=args.toolkit,
                        official=official_reference,
                        bank=bank_reference,
                        official_post=official_post,
                        bank_post=bank_post,
                        pair_validated=pair_valid,
                    )
                    record["official_drift_pct"] = (
                        f"{official_drift:.12g}"
                    )
                    record["bank_drift_pct"] = f"{bank_drift:.12g}"
                    if (
                        truthy(record["success"])
                        and not pair_valid
                    ):
                        record["error"] = (
                            "latency_untrusted_baseline_drift "
                            f"official={official_drift:.6g}% "
                            f"bank={bank_drift:.6g}%"
                        )
                    records[record_id] = record
                    metrics = json.loads(
                        candidate.get("search_behavior_metrics") or "{}"
                    )
                    if truthy(record["success"]) and pair_valid:
                        print(
                            f"candidate_done [{index}/{len(pending)}] "
                            f"{workload_id} rank={candidate['rank']} "
                            f"tpl={candidate['search_template']} "
                            f"source={candidate['candidate_source']} "
                            f"ms={record['median_ms']} "
                            "speedup_vs_official="
                            f"{record['speedup_vs_official']} "
                            f"speedup_vs_bank={record['speedup_vs_bank']} "
                            "status_vs_official="
                            f"{record['status_vs_official']} "
                            f"status_vs_bank={record['status_vs_bank']} "
                            "model_ratio="
                            f"{metrics.get('predicted_latency_ratio', '')} "
                            "model_support="
                            f"{metrics.get('latency_support', '')} "
                            "runtime_risk="
                            f"{metrics.get('runtime_risk_score', '')} "
                            f"signature={candidate['tiling_signature']}"
                        )
                    elif truthy(record["success"]):
                        print(
                            f"candidate_unpaired [{index}/{len(pending)}] "
                            f"{workload_id} rank={candidate['rank']} "
                            f"tpl={candidate['search_template']} "
                            f"source={candidate['candidate_source']} "
                            f"ms={record['median_ms']} "
                            f"signature={candidate['tiling_signature']} "
                            "action=retain_preflight_exclude_latency"
                        )
                    else:
                        print(
                            f"candidate_rejected [{index}/{len(pending)}] "
                            f"{workload_id} rank={candidate['rank']} "
                            f"tpl={candidate['search_template']} "
                            f"source={candidate['candidate_source']} "
                            "runtime_risk="
                            f"{metrics.get('runtime_risk_score', '')} "
                            f"signature={candidate['tiling_signature']} "
                            f"reason={record['error'][:240]}"
                        )
                write_csv(
                    args.resume,
                    MEASUREMENT_COLUMNS,
                    list(records.values()),
                )

    summaries = summarize(
        rows, records, args.soc, args.aic, args.toolkit
    )
    write_csv(args.summary, SUMMARY_COLUMNS, summaries)
    write_csv(
        args.candidate_results,
        MEASUREMENT_COLUMNS,
        [
            records[measurement_key(args.soc, args.aic, args.toolkit, row)]
            for row in searched
            if measurement_key(args.soc, args.aic, args.toolkit, row) in records
        ],
    )
    improved = sum(
        row["optimization_result"] == "improved" for row in summaries
    )
    not_improved = sum(
        row["optimization_result"] == "not_improved" for row in summaries
    )
    print(
        f"RESULT_TOTAL workloads={len(summaries)} improved={improved} "
        f"not_improved={not_improved} "
        f"other={len(summaries) - improved - not_improved}"
    )
    print(
        f"profile_npu completed summary={args.summary} "
        f"candidates={args.candidate_results} resume={args.resume}"
    )


if __name__ == "__main__":
    main()
