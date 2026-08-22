#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import signal
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import refine_matmul_v3_candidates as matmul_contract


PROFILE_COLUMNS = [
    "workload_id", "rank", "source", "candidate_role",
    "m", "n", "k", "dtype", "trans_a", "trans_b", "execution_mode",
    "used_core_num", "hint_single_core_m", "hint_single_core_n", "hint_single_core_k",
    "hint_base_m", "hint_base_n", "hint_base_k",
    "official_base_m", "official_base_n", "official_base_k",
    "official_core_num", "official_m_dim", "official_n_dim",
    "kernel_template", "kernel_single_core_m", "kernel_single_core_n",
    "kernel_single_core_k", "m_base_blocks", "n_base_blocks",
    "m_base_tail", "n_base_tail", "l2_m_tile_count", "l2_n_tile_count",
    "l2_m_tile_block", "l2_n_tile_block", "l2_iterate_order",
    "proxy_total", "success", "preflight_passed", "preflight_mode", "error",
    "min_ms", "mean_ms", "median_ms", "stddev_ms", "p95_ms", "max_ms", "tflops",
    "warmup", "repeat", "samples", "tiling_signature", "tiling_bin",
    "search_guidance", "search_bottleneck", "search_rationale",
    "search_transition_gain", "search_resume_policy", "search_stop_reason",
    "search_model_cycles",
    "search_model_raw_ratio_vs_bank_seed",
    "search_model_ratio_vs_bank_seed", "search_model_calibration",
    "search_model_confidence", "search_history_match",
    "callback_tiling_sha256", "callback_derived_diff_vs_default",
    "callback_derived_diff_vs_bank_seed",
    "measurement_source",
]
SAMPLE_COLUMNS = ["workload_id", "rank", "candidate_role", "sample", "latency_ms"]
RESUME_METADATA_COLUMNS = [
    "resume_soc", "resume_aic", "resume_run", "resume_record_id",
]

INFO_KEYS = {
    "a_dtype", "a_format", "aub_double_num", "b_dtype", "b_format",
    "batch_a1", "batch_a2", "batch_a3", "batch_a4",
    "batch_b1", "batch_b2", "batch_b3", "batch_b4",
    "bias_flag", "bub_double_num", "fused_double_operand_num",
    "k", "k_align_flag", "l1_fused_num", "m", "m_align_flag",
    "n", "n_align_flag", "out_dtype", "out_format", "reserved_bool",
    "reserved_params1", "reserved_params2", "reserved_params3",
    "reserved_params4", "reserved_params5", "reserved_params6",
    "trans_a_flag", "trans_b_flag",
}
KNOWLEDGE_KEYS = {
    "usedCoreNum", "singleCoreM", "singleCoreN", "singleCoreK",
    "baseM", "baseN", "baseK", "depthA1", "depthB1", "stepM", "stepN",
    "iterateOrder", "stepKa", "stepKb", "dbL0A", "dbL0B", "dbL0C",
    "l2MTileCnt", "l2NTileCnt", "l2MTileBlock", "l2NTileBlock",
    "l2IterateOrder", "tilingEnable",
}
KNOWLEDGE_BANK_COLUMNS = {
    "l2MTileCnt": "bank_l2_m_tile_count",
    "l2NTileCnt": "bank_l2_n_tile_count",
    "l2MTileBlock": "bank_l2_m_tile_block",
    "l2NTileBlock": "bank_l2_n_tile_block",
    "l2IterateOrder": "bank_l2_iterate_order",
    "tilingEnable": "bank_tiling_enable",
}
TUNING_BANK_ROLES = {"searched", "bank_seed_control"}


class ProfileError(RuntimeError):
    pass


class RunnerProfileError(ProfileError):
    def __init__(
        self,
        message: str,
        row: dict[str, str],
        samples: list[dict[str, str]],
        output: str,
        return_code: int,
    ) -> None:
        super().__init__(message)
        self.row = row
        self.samples = samples
        self.output = output
        self.return_code = return_code


@dataclass(frozen=True)
class BankSpec:
    soc: str
    aic_cores: int
    l0a_bytes: int
    l0b_bytes: int
    l0c_bytes: int
    l1_bytes: int
    filename: str
    version: int
    template: dict


def truthy(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").lower() in {"1", "true", "yes", "on"}


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def prepare_measurement_run_dir(
    candidate_root: Path,
    history_record: dict[str, str] | None,
) -> Path | None:
    if history_record is not None:
        return None
    run_dir = candidate_root / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def as_float(row: dict[str, str], name: str) -> float:
    try:
        return float(row.get(name, ""))
    except (TypeError, ValueError):
        return float("nan")


def compact(value: float) -> str:
    return f"{value:.6g}" if math.isfinite(value) else "NA"


def comparison_status(
    baseline: dict[str, str], candidate: dict[str, str]
) -> tuple[float, float, float, str]:
    baseline_ms = as_float(baseline, "median_ms")
    candidate_ms = as_float(candidate, "median_ms")
    if baseline_ms <= 0 or candidate_ms <= 0:
        return float("nan"), float("nan"), float("nan"), "invalid_measurement"
    baseline_stddev = max(0.0, as_float(baseline, "stddev_ms"))
    candidate_stddev = max(0.0, as_float(candidate, "stddev_ms"))
    if not math.isfinite(baseline_stddev):
        baseline_stddev = 0.0
    if not math.isfinite(candidate_stddev):
        candidate_stddev = 0.0
    speedup = baseline_ms / candidate_ms
    delta_pct = 100.0 * (candidate_ms - baseline_ms) / baseline_ms
    noise_pct = max(
        1.0,
        200.0 * math.hypot(baseline_stddev, candidate_stddev) / baseline_ms,
    )
    if delta_pct < -noise_pct:
        status = "improved"
    elif delta_pct > noise_pct:
        status = "regressed"
    else:
        status = "within_noise"
    return speedup, delta_pct, noise_pct, status


def dual_baseline_optimization_status(
    official_status: str,
    bank_status: str,
) -> str:
    return (
        "improved"
        if official_status == "improved" and bank_status == "improved"
        else "not_improved"
    )


def exact_resume_guard_missing(
    workloads: list[dict[str, str]],
    candidates: list[dict[str, str]],
    trusted_official_workloads: set[str],
    trusted_candidate_keys: set[tuple[str, str, str]],
    protected_prefix: int,
) -> tuple[
    set[str],
    list[str],
    list[tuple[str, str, str]],
]:
    if protected_prefix > len(workloads):
        raise ProfileError(
            "exact-resume prefix exceeds the workload count: "
            f"{protected_prefix}>{len(workloads)}"
        )
    protected_ids = {
        workload["workload_id"]
        for workload in workloads[:protected_prefix]
        if workload["dtype"].lower() in {"fp16", "bf16", "fp32"}
    }
    missing_baselines = sorted(
        protected_ids - trusted_official_workloads
    )
    missing_candidates = sorted(
        (
            candidate["workload_id"],
            candidate["candidate_role"],
            candidate["rank"],
        )
        for candidate in candidates
        if candidate["workload_id"] in protected_ids
        and candidate.get(
            "search_resume_policy", "require_existing"
        ) != "allow_new"
        and (
            candidate["workload_id"],
            candidate["candidate_role"],
            candidate["rank"],
        ) not in trusted_candidate_keys
    )
    return protected_ids, missing_baselines, missing_candidates


def force_paired_measurements_for_new_search(
    candidates: list[dict[str, str]],
    history_baselines: dict[str, dict[str, str]],
    history_assignments: dict[tuple[str, str, str], dict[str, str]],
) -> tuple[set[str], int, int]:
    """Remeasure each new search action with controls from the same run.

    Absolute ACL event times can shift across runs. Reusing an old candidate
    while measuring fresh controls can therefore reverse the reported result.
    Only workloads with at least one genuinely new searched schedule are
    affected; completed workloads remain resumable.
    """
    allow_new_workloads = {
        candidate["workload_id"]
        for candidate in candidates
        if candidate.get("candidate_role") == "searched"
        and candidate.get("search_resume_policy") == "allow_new"
    }
    controls = {
        candidate["workload_id"]: (
            candidate["workload_id"],
            candidate["candidate_role"],
            candidate["rank"],
        )
        for candidate in candidates
        if candidate.get("candidate_role") == "bank_seed_control"
    }
    new_workloads: set[str] = set()
    for workload_id in allow_new_workloads:
        searched_keys = [
            (
                candidate["workload_id"],
                candidate["candidate_role"],
                candidate["rank"],
            )
            for candidate in candidates
            if candidate["workload_id"] == workload_id
            and candidate.get("candidate_role") == "searched"
        ]
        assigned = [
            history_assignments[key]
            for key in searched_keys
            if key in history_assignments
        ]
        if len(assigned) != len(searched_keys):
            new_workloads.add(workload_id)
            continue
        baseline = history_baselines.get(workload_id)
        control_key = controls.get(workload_id)
        control = (
            history_assignments.get(control_key)
            if control_key is not None
            else None
        )
        records = [*assigned, baseline, control]
        run_ids = {
            (record or {}).get("resume_run")
            or (record or {}).get("run_id")
            for record in records
        }
        if (
            None in records
            or "" in run_ids
            or None in run_ids
            or len(run_ids) != 1
        ):
            new_workloads.add(workload_id)
    removed_baselines = 0
    removed_schedules = 0
    for workload_id in new_workloads:
        if history_baselines.pop(workload_id, None) is not None:
            removed_baselines += 1
        for assignment_key in list(history_assignments):
            if assignment_key[0] != workload_id:
                continue
            del history_assignments[assignment_key]
            removed_schedules += 1
    return new_workloads, removed_baselines, removed_schedules


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ProfileError(f"{path}: missing CSV header")
        rows = list(reader)
    for line, row in enumerate(rows, 2):
        if None in row:
            raise ProfileError(f"{path}:{line}: row has more values than its header")
    return rows


def load_workloads(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    if not rows:
        raise ProfileError(f"{path}: no workloads")
    id_key = "workload_id" if "workload_id" in rows[0] else "id"
    required = {id_key, "m", "n", "k", "dtype", "trans_a", "trans_b"}
    missing = required - set(rows[0])
    if missing:
        raise ProfileError(f"{path}: missing workload columns: {sorted(missing)}")
    workloads: list[dict[str, str]] = []
    seen: set[str] = set()
    for source in rows:
        workload = dict(source)
        workload["workload_id"] = source[id_key]
        if workload["workload_id"] in seen:
            continue
        seen.add(workload["workload_id"])
        workloads.append(workload)
    return workloads


def historical_profile(
    workload: dict[str, str],
    history: dict[str, str],
    source: str,
    role: str,
) -> dict[str, str]:
    row = {column: "" for column in PROFILE_COLUMNS}
    median_ms = history.get("median_ms", "")
    stddev_ms = history.get("stddev_ms", "") or "0"
    tflops = history.get("tflops", "")
    if not tflops and median_ms:
        operations = (
            2.0
            * int(workload["m"])
            * int(workload["n"])
            * int(workload["k"])
        )
        tflops = f"{operations / (float(median_ms) * 1.0e9):.12g}"
    row.update(
        {
            "workload_id": workload["workload_id"],
            "rank": history.get("rank", ""),
            "source": source,
            "candidate_role": role,
            "m": workload["m"],
            "n": workload["n"],
            "k": workload["k"],
            "dtype": workload["dtype"],
            "trans_a": workload["trans_a"],
            "trans_b": workload["trans_b"],
            "success": "1",
            "preflight_passed": "1",
            "preflight_mode": "historical_npu_measurement",
            "error": "",
            "min_ms": median_ms,
            "mean_ms": median_ms,
            "median_ms": median_ms,
            "stddev_ms": stddev_ms,
            "p95_ms": median_ms,
            "max_ms": median_ms,
            "tflops": tflops,
            "warmup": "historical",
            "repeat": "historical",
            "samples": "historical",
            "measurement_source": (
                f"history:{history.get('run_id', 'unknown')}:"
                f"{history.get('record_key', 'unknown')}"
            ),
        }
    )
    return row


def historical_official_profile(
    workload: dict[str, str],
    history: dict[str, str],
) -> dict[str, str]:
    row = historical_profile(
        workload,
        history,
        "installed_aclnn_matmul",
        "official_operator_baseline",
    )
    row["rank"] = "-1"
    return row


def historical_baseline_matches_workload(
    workload: dict[str, str],
    history: dict[str, str],
) -> bool:
    """Require an unambiguous workload identity before reusing OCR latency."""
    expected_shape = (
        f"{workload['m']}x{workload['n']}x{workload['k']}"
    )
    return (
        history.get("workload_id") == workload["workload_id"]
        and history.get("shape") == expected_shape
        and history.get("dtype", "").lower()
        == workload["dtype"].lower()
        and history.get("trans_a") == workload["trans_a"]
        and history.get("trans_b") == workload["trans_b"]
    )


def candidate_history_fingerprint(
    candidate: dict[str, str],
    knowledge: dict,
) -> tuple[str, ...]:
    template = kernel_template_name(knowledge)
    if template.startswith("MatMulV3_"):
        template = template[len("MatMulV3_"):]
    m_blocks = math.ceil(
        int(candidate["m"]) / int(knowledge["singleCoreM"])
    )
    n_blocks = math.ceil(
        int(candidate["n"]) / int(knowledge["singleCoreN"])
    )
    return (
        candidate.get("candidate_role", "searched"),
        candidate["workload_id"],
        f"{candidate['m']}x{candidate['n']}x{candidate['k']}",
        candidate["dtype"].lower(),
        template,
        f"{knowledge['baseM']}x{knowledge['baseN']}x{knowledge['baseK']}",
        (
            f"{knowledge['singleCoreM']}x{knowledge['singleCoreN']}x"
            f"{knowledge['singleCoreK']}"
        ),
        str(knowledge["usedCoreNum"]),
        f"{m_blocks}x{n_blocks}",
        (
            f"{knowledge['l2MTileCnt']}x{knowledge['l2NTileCnt']}"
            f"({knowledge['l2MTileBlock']}x{knowledge['l2NTileBlock']})"
        ),
        str(knowledge["iterateOrder"]),
        f"{knowledge['depthA1']}x{knowledge['depthB1']}",
        f"{knowledge['dbL0A']}x{knowledge['dbL0B']}x{knowledge['dbL0C']}",
        str(knowledge["l2IterateOrder"]),
        candidate.get("trans_a", ""),
        candidate.get("trans_b", ""),
    )


def load_history(
    path: Path | None,
    soc: str,
    aic_cores: int,
) -> tuple[
    dict[str, dict[str, str]],
    dict[tuple[str, ...], list[dict[str, str]]],
]:
    if path is None or not path.is_file():
        return {}, {}
    baselines: dict[str, dict[str, str]] = {}
    candidates: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in read_csv(path):
        if (
            not truthy(row.get("ocr_complete"))
            or row.get("soc") != soc
            or int(row.get("aic") or 0) != aic_cores
            or as_float(row, "median_ms") <= 0
        ):
            continue
        if row.get("record_type") == "baseline":
            baselines[row["workload_id"]] = row
        elif (
            row.get("record_type") in {"candidate", "bank_control"}
            and row.get("preflight_contract") == "grid9_v1"
        ):
            role = (
                "bank_seed_control"
                if row.get("record_type") == "bank_control"
                else "searched"
            )
            key = (
                role,
                row.get("workload_id", ""),
                row.get("shape", ""),
                row.get("dtype", "").lower(),
                row.get("template", ""),
                row.get("T", ""),
                row.get("S", ""),
                row.get("C", ""),
                row.get("G", ""),
                row.get("L2", ""),
                row.get("I", ""),
                row.get("L1", ""),
                row.get("DB", ""),
                row.get("L2O", ""),
                row.get("trans_a", ""),
                row.get("trans_b", ""),
            )
            candidates.setdefault(key, []).append(row)
    for records in candidates.values():
        records.sort(key=lambda row: int(row.get("rank") or 0))
    return baselines, candidates


def exact_profile_fingerprint(
    row: dict[str, str],
) -> tuple[str, ...] | None:
    role = row.get("candidate_role", "")
    signature = row.get("tiling_signature", "")
    callback_sha = row.get("callback_tiling_sha256", "").lower()
    template = (
        row.get("kernel_template", "")
        or row.get("search_template", "")
        or row.get("callback_kernel_family", "")
    )
    if template.startswith("MatMulV3_"):
        template = template[len("MatMulV3_"):]
    try:
        signature_words = [int(value) for value in signature.split(":")]
    except ValueError:
        return None
    if (
        role not in TUNING_BANK_ROLES
        or len(signature_words) != len(matmul_contract.KNOWLEDGE_FIELDS)
        or len(callback_sha) != 64
        or any(character not in "0123456789abcdef" for character in callback_sha)
    ):
        return None
    required = (
        "workload_id", "m", "n", "k", "dtype", "trans_a", "trans_b",
    )
    if any(not row.get(field, "") for field in required) or not template:
        return None
    return (
        role,
        row["workload_id"],
        row["m"],
        row["n"],
        row["k"],
        row["dtype"].lower(),
        row["trans_a"],
        row["trans_b"],
        template,
        signature,
        callback_sha,
    )


def exact_official_fingerprint(
    row: dict[str, str],
) -> tuple[str, ...] | None:
    if (
        row.get("candidate_role") != "official_operator_baseline"
        or row.get("source") != "installed_aclnn_matmul"
    ):
        return None
    required = (
        "workload_id", "m", "n", "k", "dtype", "trans_a", "trans_b",
    )
    if any(not row.get(field, "") for field in required):
        return None
    return (
        row["workload_id"],
        row["m"],
        row["n"],
        row["k"],
        row["dtype"].lower(),
        row["trans_a"],
        row["trans_b"],
    )


def resumable_profile_fingerprint(
    row: dict[str, str],
) -> tuple[str, ...] | None:
    tiling_key = exact_profile_fingerprint(row)
    if tiling_key is not None:
        return ("tiling", *tiling_key)
    official_key = exact_official_fingerprint(row)
    if official_key is not None:
        return ("official", *official_key)
    return None


def load_exact_profile_history(
    paths: list[Path],
    soc: str,
    aic_cores: int,
) -> dict[tuple[str, ...], list[dict[str, str]]]:
    history: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for path in paths:
        if not path.is_file():
            continue
        for source in read_csv(path):
            try:
                source_aic = int(source.get("resume_aic") or 0)
            except ValueError:
                continue
            if (
                source.get("resume_soc") != soc
                or source_aic != aic_cores
                or not truthy(source.get("success"))
                or not truthy(source.get("preflight_passed"))
                or as_float(source, "median_ms") <= 0
            ):
                continue
            key = exact_profile_fingerprint(source)
            if key is None:
                continue
            row = dict(source)
            row["run_id"] = (
                source.get("resume_run")
                or path.stem
            )
            row["record_key"] = (
                source.get("resume_record_id")
                or f"resume:{source.get('workload_id', '')}:"
                f"{source.get('candidate_role', '')}:"
                f"{source.get('rank', '')}"
            )
            history.setdefault(key, []).append(row)
    for records in history.values():
        records.sort(key=lambda row: int(row.get("rank") or 0))
    return history


def load_exact_official_history(
    paths: list[Path],
    soc: str,
    aic_cores: int,
) -> dict[tuple[str, ...], list[dict[str, str]]]:
    history: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for path in paths:
        if not path.is_file():
            continue
        for source in read_csv(path):
            try:
                source_aic = int(source.get("resume_aic") or 0)
            except ValueError:
                continue
            if (
                source.get("resume_soc") != soc
                or source_aic != aic_cores
                or not truthy(source.get("success"))
                or as_float(source, "median_ms") <= 0
            ):
                continue
            key = exact_official_fingerprint(source)
            if key is None:
                continue
            row = dict(source)
            row["run_id"] = source.get("resume_run") or path.stem
            row["record_key"] = (
                source.get("resume_record_id")
                or f"resume:official:{source.get('workload_id', '')}"
            )
            history.setdefault(key, []).append(row)
    return history


def merge_exact_profile_history(
    output: Path,
    sources: list[Path],
    soc: str,
    aic_cores: int,
    run_id: str,
) -> int:
    records: dict[tuple[str, ...], dict[str, str]] = {}
    for path in sources:
        if not path.is_file():
            continue
        for source in read_csv(path):
            source_soc = source.get("resume_soc", "")
            source_aic = source.get("resume_aic", "")
            try:
                source_aic_value = int(source_aic or 0)
            except ValueError:
                continue
            if (
                source_soc and source_soc != soc
            ) or (
                source_aic and source_aic_value != aic_cores
            ):
                continue
            if (
                not truthy(source.get("success"))
                or as_float(source, "median_ms") <= 0
            ):
                continue
            key = resumable_profile_fingerprint(source)
            if key is None:
                continue
            if (
                source.get("candidate_role") in TUNING_BANK_ROLES
                and not truthy(source.get("preflight_passed"))
            ):
                continue
            row = {
                column: source.get(column, "")
                for column in PROFILE_COLUMNS
            }
            row.update(
                {
                    "resume_soc": soc,
                    "resume_aic": str(aic_cores),
                    "resume_run": (
                        source.get("resume_run")
                        or run_id
                    ),
                    "resume_record_id": (
                        source.get("resume_record_id")
                        or source.get("measurement_source")
                        or f"{path.name}:{source.get('workload_id', '')}:"
                        f"{source.get('candidate_role', '')}:"
                        f"{source.get('rank', '')}"
                    ),
                }
            )
            records[key] = row
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = [*PROFILE_COLUMNS, *RESUME_METADATA_COLUMNS]
    with tempfile.NamedTemporaryFile(
        "w",
        newline="",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        delete=False,
    ) as temporary:
        writer = csv.DictWriter(temporary, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records.values())
        temporary_path = Path(temporary.name)
    temporary_path.replace(output)
    return len(records)


def write_header(path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        csv.DictWriter(destination, fieldnames=columns).writeheader()


def append_row(path: Path, columns: list[str], row: dict[str, str]) -> None:
    with path.open("a", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns, extrasaction="ignore")
        writer.writerow(row)
        destination.flush()


def append_samples(
    path: Path,
    rows: list[dict[str, str]],
    rank: str,
    candidate_role: str,
) -> None:
    with path.open("a", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=SAMPLE_COLUMNS)
        for source in rows:
            row = dict(source)
            row["rank"] = rank
            row["candidate_role"] = candidate_role
            writer.writerow(row)
        destination.flush()


def pack_info(info: dict) -> bytes:
    values = [
        int(info["m"]), int(info["k"]), int(info["n"]),
        int(info["batch_a1"]), int(info["batch_a2"]),
        int(info["batch_a3"]), int(info["batch_a4"]),
        int(info["batch_b1"]), int(info["batch_b2"]),
        int(info["batch_b3"]), int(info["batch_b4"]),
        float(info["l1_fused_num"]), float(info["aub_double_num"]),
        float(info["bub_double_num"]), float(info["fused_double_operand_num"]),
        int(info["a_dtype"]), int(info["b_dtype"]), int(info["out_dtype"]),
        int(info["a_format"]), int(info["b_format"]), int(info["out_format"]),
        bool(info["trans_a_flag"]), bool(info["trans_b_flag"]),
        bool(info["bias_flag"]), bool(info["reserved_bool"]),
        bool(info["m_align_flag"]), bool(info["k_align_flag"]),
        bool(info["n_align_flag"]),
        int(info["reserved_params1"]), int(info["reserved_params2"]),
        int(info["reserved_params3"]), int(info["reserved_params4"]),
        int(info["reserved_params5"]), int(info["reserved_params6"]),
    ]
    packed = struct.pack("<11q4f6i7?6Q", *values)
    if len(packed) != 183:
        raise ProfileError(f"internal MatMulV3 key size error: {len(packed)}")
    return packed


def probe_hash(probe: Path, key_path: Path) -> int:
    result = subprocess.run(
        [str(probe), "--hash", str(key_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise ProfileError(
            f"RuntimeKb::CommonHash failed rc={result.returncode}: "
            f"{result.stdout.strip()}"
        )
    try:
        return int(result.stdout.strip())
    except ValueError as exception:
        raise ProfileError(f"invalid tiling-bank hash output: {result.stdout!r}") from exception


def discover_bank(
    cann_root: Path,
    soc: str,
    aic_cores: int,
    l0a_bytes: int,
    l0b_bytes: int,
    l0c_bytes: int,
    l1_bytes: int,
    probe: Path,
    work_dir: Path,
) -> BankSpec:
    bank_dir = cann_root / "opp" / "built-in" / "data" / "op" / soc / "unified_bank"
    expected = bank_dir / f"{soc}_{aic_cores}_AiCore_MatMulV3_runtime_kb.json"
    if not expected.is_file():
        available = sorted(path.name for path in bank_dir.glob("*MatMulV3_runtime_kb.json"))
        raise ProfileError(
            "installed CANN has no matching MatMulV3 runtime bank: "
            f"expected={expected}; available={available}"
        )
    with expected.open(encoding="utf-8") as source:
        first_line = next((line for line in source if line.strip()), "")
    if not first_line:
        raise ProfileError(f"installed MatMulV3 runtime bank is empty: {expected}")
    template = json.loads(first_line)
    if template.get("op") != "MatMulV3":
        raise ProfileError(f"unexpected runtime-bank op in {expected}")
    info = template.get("info_dict")
    knowledge = template.get("knowledge")
    if not isinstance(info, dict) or set(info) != INFO_KEYS:
        raise ProfileError(
            "unsupported CANN MatMulV3 input-key schema; "
            f"missing={sorted(INFO_KEYS - set(info or {}))} "
            f"extra={sorted(set(info or {}) - INFO_KEYS)}"
        )
    if not isinstance(knowledge, dict) or set(knowledge) != KNOWLEDGE_KEYS:
        raise ProfileError(
            "unsupported CANN MatMulV3 knowledge schema; "
            f"missing={sorted(KNOWLEDGE_KEYS - set(knowledge or {}))} "
            f"extra={sorted(set(knowledge or {}) - KNOWLEDGE_KEYS)}"
        )

    key_path = work_dir / "installed_bank_key.bin"
    key_path.write_bytes(pack_info(info))
    observed_hash = probe_hash(probe, key_path)
    expected_hash = int(template.get("id", -1))
    if observed_hash != expected_hash:
        raise ProfileError(
            "CANN MatMulV3 runtime-bank ABI mismatch: "
            f"stored_id={expected_hash} computed_id={observed_hash}"
        )
    return BankSpec(
        soc=soc,
        aic_cores=aic_cores,
        l0a_bytes=l0a_bytes,
        l0b_bytes=l0b_bytes,
        l0c_bytes=l0c_bytes,
        l1_bytes=l1_bytes,
        filename=expected.name,
        version=int(template.get("version", 0)),
        template=template,
    )


def make_info(row: dict[str, str]) -> dict:
    dtype = row["dtype"].lower()
    if dtype not in {"fp16", "bf16", "fp32"}:
        raise ProfileError(f"MatMulV3 tuning bank does not support dtype={dtype}")
    dtype_code = 0 if dtype == "fp32" else 1
    reduce_alignment = 8 if dtype == "fp32" else 16
    m = int(row["m"])
    n = int(row["n"])
    k = int(row["k"])
    aligned_m = align_up(m, 16)
    aligned_n = align_up(n, 16)
    aligned_k = align_up(k, reduce_alignment)
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


def require_int(row: dict[str, str], name: str, minimum: int = 1) -> int:
    try:
        value = int(row[name])
    except (KeyError, ValueError) as exception:
        raise ProfileError(f"candidate has invalid {name}={row.get(name)!r}") from exception
    if value < minimum:
        raise ProfileError(f"candidate has invalid {name}={value}")
    return value


def make_knowledge(row: dict[str, str]) -> dict:
    knowledge = {
        "usedCoreNum": require_int(row, "used_core_num"),
        "singleCoreM": require_int(row, "single_core_m"),
        "singleCoreN": require_int(row, "single_core_n"),
        "singleCoreK": require_int(row, "single_core_k"),
        "baseM": require_int(row, "base_m"),
        "baseN": require_int(row, "base_n"),
        "baseK": require_int(row, "base_k"),
        "depthA1": require_int(row, "depth_a1"),
        "depthB1": require_int(row, "depth_b1"),
        "stepM": require_int(row, "step_m"),
        "stepN": require_int(row, "step_n"),
        "iterateOrder": require_int(row, "iterate_order", 0),
        "stepKa": require_int(row, "step_ka"),
        "stepKb": require_int(row, "step_kb"),
        "dbL0A": require_int(row, "db_l0a"),
        "dbL0B": require_int(row, "db_l0b"),
        "dbL0C": require_int(row, "db_l0c"),
    }
    for field, column in KNOWLEDGE_BANK_COLUMNS.items():
        minimum = (
            0
            if field in {
                "l2MTileBlock", "l2NTileBlock",
                "l2IterateOrder", "tilingEnable",
            }
            else 1
        )
        knowledge[field] = require_int(row, column, minimum)
    return knowledge


def validate_candidate(row: dict[str, str], spec: BankSpec) -> None:
    if (
        row.get("candidate_role") not in TUNING_BANK_ROLES
        or not truthy(row.get("valid"))
    ):
        raise ProfileError(
            "only valid searched/control candidates may enter the tuning bank"
        )
    if int(row.get("official_return", "-1")) == -1:
        raise ProfileError("candidate was rejected by MultiCoreMatmulTiling")

    knowledge = make_knowledge(row)
    used_cores = knowledge["usedCoreNum"]
    official_cores = require_int(row, "official_core_num")
    if used_cores != official_cores:
        raise ProfileError(
            f"used_core_num={used_cores} differs from official_core_num={official_cores}"
        )
    if used_cores > spec.aic_cores:
        raise ProfileError(
            f"used_core_num={used_cores} exceeds MatMulV3 AIC count={spec.aic_cores}"
        )

    workload = matmul_contract.Workload(
        workload_id=row["workload_id"],
        m=int(row["m"]),
        n=int(row["n"]),
        k=int(row["k"]),
        dtype=row["dtype"].lower(),
        trans_a=truthy(row.get("trans_a")),
        trans_b=truthy(row.get("trans_b")),
        max_cores=int(row.get("max_cores") or spec.aic_cores),
    )
    hardware = matmul_contract.Hardware(
        aic_cores=spec.aic_cores,
        l0a_bytes=spec.l0a_bytes,
        l0b_bytes=spec.l0b_bytes,
        l0c_bytes=spec.l0c_bytes,
        l1_bytes=spec.l1_bytes,
        l2_bytes=192 * 1024 * 1024,
        l2_bytes_per_cycle_per_core=1.0,
        hbm_bytes_per_cycle_per_core=1.0,
    )
    if not matmul_contract.hard_legal(workload, knowledge, hardware):
        raise ProfileError(
            "candidate violates the exact CANN 8.1 MatMulV3 template contract"
        )

    family = matmul_contract.template_name(knowledge)
    expected_modes = {
        "BASE": "base_iterate_all",
        "SINGLE_CORE_SPLIT_K": "single_core_split_k",
        "DETERMINISTIC_SPLIT_K": "deterministic_split_k",
        "AL1_FULL_LOAD": "al1_full_load",
        "BL1_FULL_LOAD": "bl1_full_load",
        "BL1_FULL_LOAD_FIXPIPE": "bl1_full_load_fixpipe",
        "BL1_FULL_LOAD_VEC_NZ2ND": "bl1_full_load_vec_nz2nd",
    }
    mode = row.get("execution_mode", "")
    if mode != expected_modes[family]:
        raise ProfileError(
            f"execution_mode={mode!r} does not match MatMulV3 family={family}"
        )

    suffix = require_int(row, "callback_kernel_suffix", 0)
    try:
        callback_family = matmul_contract.CANN81_KERNEL_VARIANTS[suffix][1]
    except KeyError as exception:
        raise ProfileError(
            f"installed CANN 8.1 MatMulV3 has no kernel suffix={suffix}"
        ) from exception
    if callback_family != family:
        raise ProfileError(
            f"callback kernel suffix={suffix} selects {callback_family}, "
            f"not requested family={family}"
        )


def create_candidate_bank(
    row: dict[str, str],
    spec: BankSpec,
    probe: Path,
    root: Path,
    cache_root: Path,
) -> tuple[dict[str, str], str, dict]:
    validate_candidate(row, spec)
    info = make_info(row)
    knowledge = make_knowledge(row)
    key = pack_info(info)
    key_path = root / "matmul_v3_input.bin"
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_bytes(key)
    record_id = probe_hash(probe, key_path)
    record = {
        "id": record_id,
        "info_dict": info,
        "knowledge": knowledge,
        "op": "MatMulV3",
        "version": spec.version,
    }
    bank_file = root / spec.soc / "unified_bank" / spec.filename
    bank_file.parent.mkdir(parents=True, exist_ok=True)
    bank_file.write_text(
        json.dumps(record, separators=(",", ":"), ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["TUNE_BANK_PATH"] = str(root)
    env["ASCEND_CACHE_PATH"] = str(cache_root)
    query = subprocess.run(
        [
            str(probe), "--query", str(key_path),
            spec.soc, str(spec.aic_cores),
        ],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    query_output = query.stdout.strip()
    if query.returncode != 0:
        raise ProfileError(
            "CANN rejected generated MatMulV3 tuning-bank record: "
            f"rc={query.returncode} output={query_output}"
        )
    if "found=1" in query_output:
        query_mode = "runtime_kb"
    else:
        raise ProfileError(
            "generated MatMulV3 bank record was not found by RuntimeKb: "
            f"output={query_output}"
        )
    return env, query_mode, knowledge


def failed_runner_row(workload_id: str, message: str) -> dict[str, str]:
    row = {column: "" for column in PROFILE_COLUMNS}
    row.update({
        "workload_id": workload_id,
        "success": "0",
        "preflight_passed": "0",
        "error": message,
    })
    return row


def signal_process_group(process: subprocess.Popen, signal_number: int) -> None:
    try:
        os.killpg(process.pid, signal_number)
    except ProcessLookupError:
        pass


def run_logged_process(
    command: list[str],
    env: dict[str, str],
    log_path: Path,
    timeout_sec: int,
) -> tuple[int, str, bool]:
    """Run one isolated candidate without reusing a subprocess pipe."""
    timed_out = False
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as runner_log:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=runner_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            process.wait(timeout=timeout_sec if timeout_sec > 0 else None)
        except subprocess.TimeoutExpired:
            timed_out = True
            signal_process_group(process, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                signal_process_group(process, signal.SIGKILL)
                process.wait()
        except BaseException:
            if process.poll() is None:
                signal_process_group(process, signal.SIGKILL)
                process.wait()
            raise
    output = log_path.read_text(encoding="utf-8", errors="replace")
    return process.returncode, output, timed_out


def run_official(
    runner: Path,
    workload_source: Path,
    workload_id: str,
    output_dir: Path,
    env: dict[str, str],
    args: argparse.Namespace,
) -> tuple[dict[str, str], list[dict[str, str]]]:
    profile_path = output_dir / "profile.csv"
    samples_path = output_dir / "samples.csv"
    command = [
        str(runner),
        "--candidates", str(workload_source),
        "--output", str(profile_path),
        "--samples-output", str(samples_path),
        "--device", str(args.device),
        "--warmup", str(args.warmup),
        "--repeat", str(args.repeat),
        "--samples", str(args.samples),
        "--numeric-preflight-max-mib", str(args.numeric_preflight_max_mib),
        "--only-workload", workload_id,
    ]
    try:
        return_code, output, timed_out = run_logged_process(
            command, env, output_dir / "runner.log", args.timeout_sec
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exception:
        message = (
            "host runner orchestration failed "
            f"({type(exception).__name__}): {exception}"
        )
        raise RunnerProfileError(
            message,
            failed_runner_row(workload_id, message),
            [],
            "",
            125,
        ) from exception

    if timed_out:
        tail = " | ".join(output.splitlines()[-6:])
        message = f"official MatMulV3 timeout after {args.timeout_sec}s"
        if tail:
            message += f"; {tail}"
        sample_rows: list[dict[str, str]] = []
        try:
            sample_rows = read_csv(samples_path) if samples_path.is_file() else []
        except (OSError, ProfileError):
            pass
        raise RunnerProfileError(
            message,
            failed_runner_row(workload_id, message),
            sample_rows,
            output,
            return_code if return_code is not None else 124,
        )

    try:
        profile_rows = read_csv(profile_path) if profile_path.is_file() else []
        sample_rows = read_csv(samples_path) if samples_path.is_file() else []
    except (OSError, ProfileError, ValueError, csv.Error) as exception:
        tail = " | ".join(output.splitlines()[-8:])
        message = (
            "official runner result parse failed "
            f"({type(exception).__name__}: {exception}, rc={return_code})"
        )
        if tail:
            message += f"; {tail}"
        raise RunnerProfileError(
            message,
            failed_runner_row(workload_id, message),
            [],
            output,
            return_code,
        ) from exception
    if len(profile_rows) != 1:
        tail = " | ".join(output.splitlines()[-8:])
        message = (
            f"official runner produced {len(profile_rows)} rows "
            f"(rc={return_code}); {tail}"
        )
        raise RunnerProfileError(
            message,
            failed_runner_row(workload_id, message),
            sample_rows,
            output,
            return_code,
        )
    row = profile_rows[0]
    if row.get("workload_id") != workload_id:
        raise ProfileError("official runner returned the wrong workload")
    if return_code != 0 or not truthy(row.get("success")):
        raise RunnerProfileError(
            row.get("error") or f"official MatMulV3 runner rc={return_code}",
            row,
            sample_rows,
            output,
            return_code,
        )
    return row, sample_rows


def candidate_profile(
    candidate: dict[str, str] | None,
    measured: dict[str, str],
    rank: str,
    source: str,
    role: str,
    knowledge: dict | None = None,
) -> dict[str, str]:
    result = dict(measured)
    result.update({
        "rank": rank,
        "source": source,
        "candidate_role": role,
    })
    if candidate is None:
        return result
    # A timed-out runner has no profile CSV, so its synthetic failure row does
    # not contain shape fields. Keep every candidate row self-identifying so a
    # rejected candidate cannot invalidate ranking for the whole workload.
    for field in (
        "workload_id", "m", "n", "k", "dtype", "trans_a", "trans_b",
    ):
        value = candidate.get(field, "")
        if value != "":
            result[field] = value
    mapping = {
        "execution_mode": "execution_mode",
        "used_core_num": "used_core_num",
        "hint_single_core_m": "candidate_single_core_m",
        "hint_single_core_n": "candidate_single_core_n",
        "hint_single_core_k": "candidate_single_core_k",
        "hint_base_m": "candidate_base_m",
        "hint_base_n": "candidate_base_n",
        "hint_base_k": "candidate_base_k",
        "official_base_m": "base_m",
        "official_base_n": "base_n",
        "official_base_k": "base_k",
        "official_core_num": "official_core_num",
        "official_m_dim": "official_m_dim",
        "official_n_dim": "official_n_dim",
        "proxy_total": "proxy_total",
        "tiling_signature": "tiling_signature",
        "tiling_bin": "tiling_bin",
        "search_guidance": "search_guidance",
        "search_bottleneck": "search_bottleneck",
        "search_rationale": "search_rationale",
        "search_transition_gain": "search_transition_gain",
        "search_resume_policy": "search_resume_policy",
        "search_stop_reason": "search_stop_reason",
        "search_model_cycles": "search_model_cycles",
        "search_model_raw_ratio_vs_bank_seed": "search_model_raw_ratio_vs_bank_seed",
        "search_model_ratio_vs_bank_seed": "search_model_ratio_vs_bank_seed",
        "search_model_calibration": "search_model_calibration",
        "search_model_confidence": "search_model_confidence",
        "search_history_match": "search_history_match",
        "callback_tiling_sha256": "callback_tiling_sha256",
        "callback_derived_diff_vs_default": "callback_derived_diff_vs_default",
        "callback_derived_diff_vs_bank_seed": "callback_derived_diff_vs_bank_seed",
    }
    for destination, origin in mapping.items():
        result[destination] = candidate.get(origin, "")
    if knowledge is not None:
        single_m = int(knowledge["singleCoreM"])
        single_n = int(knowledge["singleCoreN"])
        m = int(candidate["m"])
        n = int(candidate["n"])
        m_blocks = math.ceil(m / single_m)
        n_blocks = math.ceil(n / single_n)
        result.update({
            "used_core_num": str(knowledge["usedCoreNum"]),
            "kernel_template": kernel_template_name(knowledge),
            "kernel_single_core_m": str(knowledge["singleCoreM"]),
            "kernel_single_core_n": str(knowledge["singleCoreN"]),
            "kernel_single_core_k": str(knowledge["singleCoreK"]),
            "m_base_blocks": str(m_blocks),
            "n_base_blocks": str(n_blocks),
            "m_base_tail": str(m - (m_blocks - 1) * single_m),
            "n_base_tail": str(n - (n_blocks - 1) * single_n),
            "l2_m_tile_count": str(knowledge["l2MTileCnt"]),
            "l2_n_tile_count": str(knowledge["l2NTileCnt"]),
            "l2_m_tile_block": str(knowledge["l2MTileBlock"]),
            "l2_n_tile_block": str(knowledge["l2NTileBlock"]),
            "l2_iterate_order": str(knowledge["l2IterateOrder"]),
        })
    return result


def kernel_template_name(knowledge: dict) -> str:
    return f"MatMulV3_{matmul_contract.template_name(knowledge)}"


def isolated_candidate_failure(error: RunnerProfileError) -> bool:
    detail = f"{error}\n{error.output}".lower()
    infrastructure_markers = (
        "host runner orchestration failed",
        "aclinit",
        "aclrtsetdevice",
        "aclrtcreatecontext",
        "aclrtcreatestream",
        "aclrtbinaryloadfromfile",
        "registerascendbinary",
        "symbol lookup error",
        "error while loading shared libraries",
        "507008",
        "107000",
    )
    return not any(marker in detail for marker in infrastructure_markers)


def schedule_text(candidate: dict[str, str], knowledge: dict) -> str:
    m_blocks = math.ceil(int(candidate["m"]) / int(knowledge["singleCoreM"]))
    n_blocks = math.ceil(int(candidate["n"]) / int(knowledge["singleCoreN"]))
    template = kernel_template_name(knowledge)
    if template.startswith("MatMulV3_"):
        template = template[len("MatMulV3_"):]
    return (
        f"tpl={template} "
        f"T={knowledge['baseM']}x{knowledge['baseN']}x{knowledge['baseK']} "
        f"S={knowledge['singleCoreM']}x{knowledge['singleCoreN']}x"
        f"{knowledge['singleCoreK']} C={knowledge['usedCoreNum']} "
        f"G={m_blocks}x{n_blocks} "
        f"L2={knowledge['l2MTileCnt']}x{knowledge['l2NTileCnt']}"
        f"({knowledge['l2MTileBlock']}x{knowledge['l2NTileBlock']}) "
        f"I={knowledge['iterateOrder']} "
        f"L1={knowledge['depthA1']}x{knowledge['depthB1']} "
        f"DB={knowledge['dbL0A']}x{knowledge['dbL0B']}x{knowledge['dbL0C']}"
    )


def print_tiling_failure(
    candidate: dict[str, str],
    knowledge: dict,
    error: RunnerProfileError,
) -> None:
    message = str(error)
    index_match = re.search(r"C index=(\d+)", message)
    observed_match = re.search(r"observed=([^,\s]+)", message)
    index = int(index_match.group(1)) if index_match else None
    n = int(candidate["n"])
    row = index // n if index is not None else None
    column = index % n if index is not None else None
    lower_message = message.lower()
    if "host runner orchestration failed" in lower_message:
        classification = "host_process_control_failure"
        evidence = "the host profiler failed while starting or waiting for the isolated runner"
        contract = "static=passed runtime_kb=found npu_preflight=unknown"
    elif "timeout" in lower_message:
        classification = "candidate_timeout"
        evidence = "the isolated official MatMulV3 process made no completion progress"
        contract = "static=passed runtime_kb=found npu_preflight=timeout"
    elif index is not None and observed_match and observed_match.group(1).lower().startswith("0x5a"):
        classification = "output_not_written"
        evidence = "output still contains the 0x5a preflight poison value"
        contract = "static=passed runtime_kb=found npu_preflight=failed"
    elif "coverage failed" in message:
        classification = "output_coverage_failure"
        evidence = "at least one sampled output element was not produced"
        contract = "static=passed runtime_kb=found npu_preflight=failed"
    elif "507015" in message:
        classification = "aicore_exception"
        evidence = "the candidate raised an AICore exception during synchronization"
        contract = "static=passed runtime_kb=found npu_preflight=failed"
    else:
        classification = "candidate_execution_failure"
        evidence = "the official MatMulV3 candidate process returned failure"
        contract = "static=passed runtime_kb=found npu_preflight=failed"

    print("TILING_ERROR_BEGIN", flush=True)
    print(
        f"workload={candidate['workload_id']} rank={candidate['rank']} "
        f"shape={candidate['m']}x{candidate['n']}x{candidate['k']} "
        f"dtype={candidate['dtype']} trans={candidate['trans_a']}{candidate['trans_b']}",
        flush=True,
    )
    print(f"schedule: {schedule_text(candidate, knowledge)}", flush=True)
    print(
        "l1: "
        f"depth={knowledge['depthA1']}x{knowledge['depthB1']} "
        f"stepMN={knowledge['stepM']}x{knowledge['stepN']} "
        f"stepK={knowledge['stepKa']}x{knowledge['stepKb']} "
        f"db={knowledge['dbL0A']}x{knowledge['dbL0B']}x{knowledge['dbL0C']}",
        flush=True,
    )
    if "preflight" in lower_message:
        stage = "preflight"
    elif "warmup" in lower_message:
        stage = "warmup"
    elif "sample" in lower_message:
        stage = "timing"
    elif "timeout" in lower_message:
        stage = "timeout"
    else:
        stage = "runner"
    print(f"failure: stage={stage} classification={classification}", flush=True)
    print(f"error: {message}", flush=True)
    if index is not None:
        print(
            f"location: C_index={index} row={row} col={column} "
            f"observed={observed_match.group(1) if observed_match else 'unknown'}",
            flush=True,
        )
    print(f"evidence: {evidence}", flush=True)
    print(f"contract: {contract}", flush=True)
    output_tail = [
        line.strip()
        for line in error.output.splitlines()
        if line.strip()
    ][-3:]
    if output_tail:
        print(f"runner_tail: {' | '.join(output_tail)}", flush=True)
    print("TILING_ERROR_END", flush=True)


def unsupported_profile(
    workload: dict[str, str],
    rank: str,
    source: str,
    role: str,
    args: argparse.Namespace,
) -> dict[str, str]:
    row = {column: "" for column in PROFILE_COLUMNS}
    row.update({
        "workload_id": workload["workload_id"],
        "rank": rank,
        "source": source,
        "candidate_role": role,
        "m": workload["m"],
        "n": workload["n"],
        "k": workload["k"],
        "dtype": workload["dtype"],
        "trans_a": workload["trans_a"],
        "trans_b": workload["trans_b"],
        "execution_mode": "unsupported",
        "success": "0",
        "preflight_passed": "0",
        "error": f"unsupported_by_MatMulV3:{workload['dtype']}",
        "min_ms": "0",
        "mean_ms": "0",
        "median_ms": "0",
        "stddev_ms": "0",
        "p95_ms": "0",
        "max_ms": "0",
        "tflops": "0",
        "warmup": str(args.warmup),
        "repeat": str(args.repeat),
        "samples": str(args.samples),
    })
    return row


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile searched tilings through the official CANN MatMulV3 kernel."
    )
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--bank-probe", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--custom-output", type=Path, required=True)
    parser.add_argument("--custom-samples-output", type=Path, required=True)
    parser.add_argument("--official-output", type=Path, required=True)
    parser.add_argument("--official-samples-output", type=Path, required=True)
    parser.add_argument("--history", type=Path)
    parser.add_argument(
        "--profile-history",
        type=Path,
        action="append",
        default=[],
        help="reuse exact successful profile rows from an earlier local run",
    )
    parser.add_argument("--cann-root", type=Path, required=True)
    parser.add_argument("--soc", required=True)
    parser.add_argument("--aic-cores", type=int, required=True)
    parser.add_argument("--l0a-bytes", type=int, required=True)
    parser.add_argument("--l0b-bytes", type=int, required=True)
    parser.add_argument("--l0c-bytes", type=int, required=True)
    parser.add_argument("--l1-bytes", type=int, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--rank-limit", type=int, default=20)
    parser.add_argument("--numeric-preflight-max-mib", type=int, default=4)
    parser.add_argument(
        "--skip-bank-seed-control",
        action="store_true",
        help="profile only searched solver candidates, not a bank-seed control",
    )
    parser.add_argument("--timeout-sec", type=int, default=60)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate every candidate against CANN RuntimeKb without using the NPU",
    )
    parser.add_argument(
        "--require-exact-resume-prefix",
        type=int,
        default=0,
        help=(
            "require exact baseline/control/candidate resume records for the "
            "first N workloads before any NPU profiling"
        ),
    )
    args = parser.parse_args()

    for executable in (args.runner, args.bank_probe):
        executable = executable.resolve()
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ProfileError(f"required executable is missing: {executable}")
    args.runner = args.runner.resolve()
    args.bank_probe = args.bank_probe.resolve()
    args.candidates = args.candidates.resolve()
    args.workloads = args.workloads.resolve()
    args.cann_root = args.cann_root.resolve()
    if args.history is not None:
        args.history = args.history.resolve()
    args.profile_history = [
        path.resolve() for path in args.profile_history
    ]
    if (
        args.aic_cores <= 0
        or args.rank_limit <= 0
        or min(
            args.l0a_bytes,
            args.l0b_bytes,
            args.l0c_bytes,
            args.l1_bytes,
        ) <= 0
    ):
        raise ProfileError("platform capacities and rank-limit must be positive")
    if args.repeat <= 0 or args.samples <= 0 or args.warmup < 0:
        raise ProfileError("warmup/repeat/samples values are invalid")

    candidates = read_csv(args.candidates)
    workloads = load_workloads(args.workloads)
    history_baselines, history_candidates = load_history(
        args.history, args.soc, args.aic_cores
    )
    history_baselines = {
        workload["workload_id"]: history_baselines[workload["workload_id"]]
        for workload in workloads
        if workload["workload_id"] in history_baselines
        and historical_baseline_matches_workload(
            workload,
            history_baselines[workload["workload_id"]],
        )
    }
    exact_profile_history = load_exact_profile_history(
        args.profile_history, args.soc, args.aic_cores
    )
    exact_official_history = load_exact_official_history(
        args.profile_history, args.soc, args.aic_cores
    )
    exact_official_assignments = 0
    for workload in workloads:
        lookup = dict(workload)
        lookup.update(
            {
                "source": "installed_aclnn_matmul",
                "candidate_role": "official_operator_baseline",
            }
        )
        key = exact_official_fingerprint(lookup)
        old_records = exact_official_history.get(key, []) if key else []
        if not old_records:
            continue
        history_baselines[workload["workload_id"]] = old_records[0]
        exact_official_assignments += 1
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in candidates:
        grouped.setdefault(row["workload_id"], []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row.get("rank", "0")))

    write_header(args.custom_output, PROFILE_COLUMNS)
    write_header(args.custom_samples_output, SAMPLE_COLUMNS)
    write_header(args.official_output, PROFILE_COLUMNS)
    write_header(args.official_samples_output, SAMPLE_COLUMNS)

    supported_candidates = [
        row
        for row in candidates
        if (
            (
                row.get("candidate_role") == "searched"
                and 0 < int(row.get("rank", "0")) <= args.rank_limit
            )
            or (
                row.get("candidate_role") == "bank_seed_control"
                and int(row.get("rank", "-1")) == 0
                and not args.skip_bank_seed_control
            )
        )
        and row.get("dtype", "").lower() in {"fp16", "bf16", "fp32"}
    ]
    current_history_groups: dict[
        tuple[str, ...], list[tuple[dict[str, str], dict]]
    ] = {}
    for candidate in supported_candidates:
        knowledge = make_knowledge(candidate)
        key = candidate_history_fingerprint(candidate, knowledge)
        current_history_groups.setdefault(key, []).append(
            (candidate, knowledge)
        )
    history_assignments: dict[
        tuple[str, str, str], dict[str, str]
    ] = {}
    exact_history_assignments = 0
    for candidate in supported_candidates:
        key = exact_profile_fingerprint(candidate)
        old_records = exact_profile_history.get(key, []) if key else []
        if not old_records:
            continue
        assignment_key = (
            candidate["workload_id"],
            candidate["candidate_role"],
            candidate["rank"],
        )
        history_assignments[assignment_key] = old_records[0]
        exact_history_assignments += 1
    for key, current_records in current_history_groups.items():
        old_records = history_candidates.get(key, [])
        current_records.sort(
            key=lambda item: (
                item[1]["l2IterateOrder"],
                item[1]["iterateOrder"],
                int(item[0].get("rank") or 0),
            )
        )
        for (candidate, _), history_record in zip(
            current_records, old_records
        ):
            assignment_key = (
                candidate["workload_id"],
                candidate["candidate_role"],
                candidate["rank"],
            )
            if assignment_key not in history_assignments:
                history_assignments[assignment_key] = history_record

    (
        paired_measurement_workloads,
        paired_baselines_removed,
        paired_schedules_removed,
    ) = force_paired_measurements_for_new_search(
        supported_candidates,
        history_baselines,
        history_assignments,
    )

    if args.require_exact_resume_prefix:
        (
            protected_ids,
            missing_baselines,
            missing_schedules,
        ) = exact_resume_guard_missing(
            workloads,
            supported_candidates,
            set(history_baselines),
            set(history_assignments),
            args.require_exact_resume_prefix,
        )
        if missing_baselines or missing_schedules:
            details: list[str] = []
            if missing_baselines:
                details.append(
                    "baselines=" + ",".join(missing_baselines[:5])
                )
            if missing_schedules:
                details.append(
                    "schedules="
                    + ",".join(
                        f"{workload_id}:{role}:{rank}"
                        for workload_id, role, rank
                        in missing_schedules[:5]
                    )
                )
            raise ProfileError(
                "resume guard found prior workloads without a trusted "
                "identity-complete measurement; retain "
                "results/npu_full_resume.csv ("
                + "; ".join(details)
                + ")"
            )
        print(
            "resume_guard: "
            f"protected_workloads={args.require_exact_resume_prefix} "
            f"trusted_baselines={len(protected_ids)} "
            "missing=0 action=no_prior_npu_remeasurement",
            flush=True,
        )

    with tempfile.TemporaryDirectory(
        prefix="matmul_v3_bank_", dir=str(args.custom_output.parent)
    ) as temporary:
        work_dir = Path(temporary)
        cache_root = work_dir / "empty_cache"
        cache_root.mkdir()
        spec = discover_bank(
            args.cann_root,
            args.soc,
            args.aic_cores,
            args.l0a_bytes,
            args.l0b_bytes,
            args.l0c_bytes,
            args.l1_bytes,
            args.bank_probe,
            work_dir,
        )
        print(
            f"bank_schema: soc={spec.soc} aic={spec.aic_cores} "
            "input_bytes=183 knowledge_fields=23",
            flush=True,
        )

        prepared_candidates: dict[
            tuple[str, str, str], tuple[dict[str, str] | None, dict]
        ] = {}
        query_modes: set[str] = set()
        history_prepared = 0
        bank_prepared = 0
        for validation_index, candidate in enumerate(supported_candidates, 1):
            workload_id = candidate["workload_id"]
            rank = candidate["rank"]
            role = candidate["candidate_role"]
            knowledge = make_knowledge(candidate)
            if (workload_id, role, rank) in history_assignments:
                validate_candidate(candidate, spec)
                prepared_candidates[(workload_id, role, rank)] = (
                    None,
                    knowledge,
                )
                history_prepared += 1
                continue
            candidate_root = (
                work_dir / "candidate" /
                f"{safe_name(workload_id)}_{safe_name(role)}_rank{rank}"
            )
            bank_root = candidate_root / "bank"
            candidate_cache = candidate_root / "cache"
            candidate_cache.mkdir(parents=True)
            try:
                candidate_env, query_mode, knowledge = create_candidate_bank(
                    candidate,
                    spec,
                    args.bank_probe,
                    bank_root,
                    candidate_cache,
                )
            except ProfileError as exception:
                raise ProfileError(
                    f"{workload_id} role={role} rank={rank}: {exception}"
                ) from exception
            prepared_candidates[(workload_id, role, rank)] = (
                candidate_env,
                knowledge,
            )
            bank_prepared += 1
            query_modes.add(query_mode)
            if (
                validation_index == 1
                or validation_index == len(supported_candidates)
                or validation_index % 50 == 0
            ):
                print(
                    f"bank_lookup: [{validation_index}/{len(supported_candidates)}] "
                    f"{workload_id} role={role} rank={rank}",
                    flush=True,
                )
        print(
            f"bank_records_prepared: {bank_prepared} "
            f"history_records_reused={history_prepared} "
            f"lookup={'+'.join(sorted(query_modes)) or 'none'}"
            f"{'(found=1)' if query_modes else ''} "
            "execution_check=NPU_preflight",
            flush=True,
        )
        if args.validate_only:
            return 0

        empty_bank = work_dir / "empty_bank"
        empty_bank.mkdir()
        baseline_env = os.environ.copy()
        baseline_env["TUNE_BANK_PATH"] = str(empty_bank)
        baseline_env["ASCEND_CACHE_PATH"] = str(cache_root)

        candidate_index = 0
        runtime_rejected = 0
        unsupported = 0
        workloads_with_search = len(
            {
                row["workload_id"]
                for row in supported_candidates
                if row.get("candidate_role") == "searched"
            }
        )
        searched_count = sum(
            row.get("candidate_role") == "searched"
            for row in supported_candidates
        )
        control_count = sum(
            row.get("candidate_role") == "bank_seed_control"
            for row in supported_candidates
        )
        pending_searched = sum(
            row.get("candidate_role") == "searched"
            and (
                row["workload_id"],
                row["candidate_role"],
                row["rank"],
            ) not in history_assignments
            for row in supported_candidates
        )
        pending_controls = sum(
            row.get("candidate_role") == "bank_seed_control"
            and (
                row["workload_id"],
                row["candidate_role"],
                row["rank"],
            ) not in history_assignments
            for row in supported_candidates
        )
        pending_baselines = sum(
            workload["dtype"].lower() in {"fp16", "bf16", "fp32"}
            and workload["workload_id"] not in history_baselines
            for workload in workloads
        )
        print(
            f"profile_plan: workloads={len(workloads)} "
            f"searched_candidates={searched_count} "
            f"bank_seed_controls={control_count} "
            f"workloads_with_search={workloads_with_search} "
            f"workloads_without_searched_candidate="
            f"{len(workloads) - workloads_with_search} "
            f"history_baselines={len(history_baselines)} "
            f"history_candidate_schedules={len(history_candidates)} "
            f"resume_exact_baselines={len(exact_official_history)} "
            f"resume_exact_baselines_assigned={exact_official_assignments} "
            f"resume_exact_schedules={len(exact_profile_history)} "
            f"resume_exact_assigned={exact_history_assignments} "
            f"history_candidates_assigned={len(history_assignments)} "
            f"npu_searched_pending={pending_searched} "
            f"npu_controls_pending={pending_controls} "
            f"npu_official_baselines_pending={pending_baselines} "
            f"paired_measurement_workloads={len(paired_measurement_workloads)} "
            f"paired_baselines_remeasured={paired_baselines_removed} "
            f"paired_schedules_remeasured={paired_schedules_removed}",
            flush=True,
        )
        for workload_index, workload in enumerate(workloads, 1):
            workload_id = workload["workload_id"]
            rows = grouped.get(workload_id, [])
            if workload["dtype"].lower() not in {"fp16", "bf16", "fp32"}:
                unsupported += 1
                official = unsupported_profile(
                    workload, "-1", "installed_aclnn_matmul",
                    "official_operator_baseline", args,
                )
                api_auto = unsupported_profile(
                    workload, "0", "official_default", "api_auto_baseline", args
                )
                append_row(args.official_output, PROFILE_COLUMNS, official)
                append_row(args.custom_output, PROFILE_COLUMNS, api_auto)
                print(
                    f"workload_skip: [{workload_index}/{len(workloads)}] "
                    f"{workload_id} dtype={workload['dtype']} not supported by MatMulV3",
                    flush=True,
                )
                continue

            baseline_dir = work_dir / "baseline" / safe_name(workload_id)
            baseline_dir.mkdir(parents=True)
            history_baseline = history_baselines.get(workload_id)
            if history_baseline is not None:
                measured = historical_official_profile(
                    workload, history_baseline
                )
                samples = []
                print(
                    f"official_history_reuse {workload_id} "
                    f"run={history_baseline.get('run_id', 'unknown')} "
                    f"ms={history_baseline.get('median_ms', '')}",
                    flush=True,
                )
            else:
                try:
                    measured, samples = run_official(
                        args.runner, args.workloads, workload_id,
                        baseline_dir, baseline_env, args,
                    )
                except RunnerProfileError as exception:
                    append_row(
                        args.official_output,
                        PROFILE_COLUMNS,
                        exception.row,
                    )
                    print("OFFICIAL_BASELINE_ERROR_BEGIN", flush=True)
                    print(
                        f"workload={workload_id} "
                        f"shape={workload['m']}x{workload['n']}x{workload['k']} "
                        f"dtype={workload['dtype']}",
                        flush=True,
                    )
                    print(f"error={exception}", flush=True)
                    print("OFFICIAL_BASELINE_ERROR_END", flush=True)
                    raise ProfileError(
                        f"installed official MatMulV3 baseline failed for {workload_id}"
                    ) from exception
            append_row(args.official_output, PROFILE_COLUMNS, measured)
            append_samples(
                args.official_samples_output, samples,
                "-1", "official_operator_baseline",
            )
            api_auto = candidate_profile(
                None, measured, "0",
                "official_default", "api_auto_baseline",
            )
            append_row(args.custom_output, PROFILE_COLUMNS, api_auto)
            append_samples(
                args.custom_samples_output, samples, "0", "api_auto_baseline"
            )
            official_baseline = measured
            print(
                f"WORKLOAD [{workload_index}/{len(workloads)}] {workload_id} "
                f"shape={workload['m']}x{workload['n']}x{workload['k']} "
                f"dtype={workload['dtype']} "
                f"official_ms={compact(as_float(measured, 'median_ms'))} "
                f"std={compact(as_float(measured, 'stddev_ms'))} "
                f"tflops={compact(as_float(measured, 'tflops'))}",
                flush=True,
            )

            controls = [] if args.skip_bank_seed_control else [
                row
                for row in rows
                if row.get("candidate_role") == "bank_seed_control"
                and int(row.get("rank", "-1")) == 0
            ]
            if len(controls) > 1:
                raise ProfileError(
                    f"{workload_id}: multiple bank_seed_control rows"
                )
            bank_control_profile: dict[str, str] | None = None
            if controls:
                control = controls[0]
                control_rank = control["rank"]
                control_role = control["candidate_role"]
                control_root = (
                    work_dir / "candidate" /
                    f"{safe_name(workload_id)}_{safe_name(control_role)}"
                    f"_rank{control_rank}"
                )
                control_env, control_knowledge = prepared_candidates[
                    (workload_id, control_role, control_rank)
                ]
                history_control = history_assignments.get(
                    (workload_id, control_role, control_rank)
                )
                control_run_dir = prepare_measurement_run_dir(
                    control_root, history_control
                )
                print(
                    f"bank_control_start {workload_id} "
                    f"{schedule_text(control, control_knowledge)}",
                    flush=True,
                )
                if history_control is not None:
                    control_measured = historical_profile(
                        workload,
                        history_control,
                        "historical_bank_seed_control",
                        control_role,
                    )
                    control_samples = []
                    print(
                        f"bank_control_history_reuse {workload_id} "
                        f"run={history_control.get('run_id', 'unknown')} "
                        f"ms={history_control.get('median_ms', '')}",
                        flush=True,
                    )
                else:
                    if control_env is None:
                        raise ProfileError(
                            f"{workload_id}: missing bank environment for "
                            "an unmeasured bank control"
                        )
                    if control_run_dir is None:
                        raise ProfileError(
                            f"{workload_id}: missing run directory for "
                            "an unmeasured bank control"
                        )
                    try:
                        control_measured, control_samples = run_official(
                            args.runner,
                            args.workloads,
                            workload_id,
                            control_run_dir,
                            control_env,
                            args,
                        )
                    except RunnerProfileError as exception:
                        failed = candidate_profile(
                            control,
                            exception.row,
                            control_rank,
                            control.get(
                                "source", "official_seed_bank_roundtrip"
                            ),
                            control_role,
                            control_knowledge,
                        )
                        append_row(
                            args.custom_output, PROFILE_COLUMNS, failed
                        )
                        append_samples(
                            args.custom_samples_output,
                            exception.samples,
                            control_rank,
                            control_role,
                        )
                        print_tiling_failure(
                            control, control_knowledge, exception
                        )
                        raise ProfileError(
                            f"bank seed control failed for {workload_id}"
                        ) from exception
                bank_control_profile = candidate_profile(
                    control,
                    control_measured,
                    control_rank,
                    control.get("source", "official_seed_bank_roundtrip"),
                    control_role,
                    control_knowledge,
                )
                append_row(
                    args.custom_output,
                    PROFILE_COLUMNS,
                    bank_control_profile,
                )
                append_samples(
                    args.custom_samples_output,
                    control_samples,
                    control_rank,
                    control_role,
                )
                control_speedup, control_delta, _, control_status = (
                    comparison_status(
                        official_baseline, bank_control_profile
                    )
                )
                print(
                    f"bank_control_done {workload_id} "
                    f"ms={compact(as_float(bank_control_profile, 'median_ms'))} "
                    f"official_ms={compact(as_float(official_baseline, 'median_ms'))} "
                    f"official_over_bank={compact(control_speedup)} "
                    f"delta={compact(control_delta)}% status={control_status}",
                    flush=True,
                )

            searched = [
                row
                for row in rows
                if row.get("candidate_role") == "searched"
                and 0 < int(row.get("rank", "0")) <= args.rank_limit
            ]
            best_ms = float("inf")
            best_rank = ""
            best_bank_status = ""
            best_official_status = ""
            for candidate in searched:
                candidate_index += 1
                rank = candidate["rank"]
                candidate_root = (
                    work_dir / "candidate" /
                    f"{safe_name(workload_id)}_searched_rank{rank}"
                )
                candidate_env, knowledge = prepared_candidates[
                    (workload_id, "searched", rank)
                ]
                history_candidate = history_assignments.get(
                    (workload_id, "searched", rank)
                )
                run_dir = prepare_measurement_run_dir(
                    candidate_root, history_candidate
                )
                show_progress = (
                    rank == "1"
                    or candidate_index == searched_count
                    or candidate_index % args.progress_every == 0
                )
                if show_progress:
                    print(
                        f"candidate_start [{candidate_index}/{searched_count}] "
                        f"{workload_id} rank={rank} "
                        f"cause={candidate.get('search_bottleneck', '')}/"
                        f"{candidate.get('search_guidance', '')} "
                        f"model={candidate.get('search_model_ratio_vs_bank_seed', '')} "
                        f"resume={candidate.get('search_resume_policy', '')} "
                        f"{schedule_text(candidate, knowledge)}",
                        flush=True,
                    )
                if history_candidate is not None:
                    measured = historical_profile(
                        workload,
                        history_candidate,
                        "historical_searched_tiling",
                        "searched",
                    )
                    samples = []
                    print(
                        f"candidate_history_reuse [{candidate_index}/"
                        f"{searched_count}] {workload_id} rank={rank} "
                        f"old_rank={history_candidate.get('rank', '')} "
                        f"run={history_candidate.get('run_id', 'unknown')} "
                        f"ms={history_candidate.get('median_ms', '')}",
                        flush=True,
                    )
                else:
                    if candidate_env is None:
                        raise ProfileError(
                            f"{workload_id} rank={rank}: missing bank "
                            "environment for an unmeasured candidate"
                        )
                    if run_dir is None:
                        raise ProfileError(
                            f"{workload_id} rank={rank}: missing run "
                            "directory for an unmeasured candidate"
                        )
                    try:
                        measured, samples = run_official(
                            args.runner,
                            args.workloads,
                            workload_id,
                            run_dir,
                            candidate_env,
                            args,
                        )
                    except RunnerProfileError as exception:
                        failed = candidate_profile(
                            candidate,
                            exception.row,
                            rank,
                            candidate.get("source", "searched"),
                            "searched",
                            knowledge,
                        )
                        append_row(
                            args.custom_output, PROFILE_COLUMNS, failed
                        )
                        append_samples(
                            args.custom_samples_output,
                            exception.samples,
                            rank,
                            "searched",
                        )
                        print_tiling_failure(
                            candidate, knowledge, exception
                        )
                        if not isolated_candidate_failure(exception):
                            print(
                                f"candidate_abort [{candidate_index}/"
                                f"{searched_count}] {workload_id} rank={rank} "
                                "action=stop_infrastructure_failure",
                                flush=True,
                            )
                            raise ProfileError(
                                "profiling infrastructure failed at "
                                f"{workload_id} rank={rank}: {exception}"
                            ) from exception
                        runtime_rejected += 1
                        print(
                            f"candidate_rejected [{candidate_index}/"
                            f"{searched_count}] {workload_id} rank={rank} "
                            "action=discard_and_continue",
                            flush=True,
                        )
                        continue
                profiled = candidate_profile(
                    candidate, measured, rank,
                    candidate.get("source", "searched"), "searched", knowledge,
                )
                append_row(args.custom_output, PROFILE_COLUMNS, profiled)
                append_samples(
                    args.custom_samples_output, samples, rank, "searched"
                )
                candidate_ms = as_float(profiled, "median_ms")
                is_new_best = candidate_ms < best_ms
                if is_new_best:
                    best_ms = candidate_ms
                    best_rank = rank
                official_speedup, official_delta, _, official_status = (
                    comparison_status(official_baseline, profiled)
                )
                optimization_baseline = (
                    bank_control_profile
                    if bank_control_profile is not None
                    else official_baseline
                )
                speedup, delta_pct, noise_pct, status = comparison_status(
                    optimization_baseline, profiled
                )
                if is_new_best:
                    best_bank_status = status
                    best_official_status = official_status
                if show_progress or is_new_best:
                    marker = "best" if is_new_best else "measured"
                    bank_speedup = (
                        as_float(bank_control_profile, "median_ms")
                        / candidate_ms
                        if bank_control_profile is not None
                        else float("nan")
                    )
                    print(
                        f"candidate_done [{candidate_index}/{searched_count}] "
                        f"{workload_id} rank={rank} {schedule_text(candidate, knowledge)} "
                        f"ms={compact(candidate_ms)} std={compact(as_float(profiled, 'stddev_ms'))} "
                        f"speedup_vs_official={compact(official_speedup)} "
                        f"speedup_vs_bank={compact(bank_speedup)} "
                        f"model_ratio={candidate.get('search_model_ratio_vs_bank_seed', 'NA')} "
                        f"delta_vs_bank={compact(delta_pct)}% "
                        f"noise={compact(noise_pct)}% "
                        f"status_vs_bank={status} "
                        f"status_vs_official={official_status} "
                        f"delta_vs_official={compact(official_delta)}% "
                        f"mark={marker}",
                        flush=True,
                    )
            bank_control_ms = (
                as_float(bank_control_profile, "median_ms")
                if bank_control_profile is not None
                else float("nan")
            )
            has_best = math.isfinite(best_ms)
            print(
                f"WORKLOAD_RESULT {workload_id} best_rank={best_rank or 'none'} "
                f"best_ms={compact(best_ms) if has_best else 'NA'} "
                f"bank_control_ms={compact(bank_control_ms)} "
                f"official_ms={compact(as_float(official_baseline, 'median_ms'))} "
                f"speedup={compact(as_float(official_baseline, 'median_ms') / best_ms if has_best else float('nan'))} "
                f"speedup_vs_bank={compact(bank_control_ms / best_ms if has_best else float('nan'))} "
                f"status_vs_bank={best_bank_status or 'no_searched_candidate'} "
                f"status_vs_official={best_official_status or 'no_searched_candidate'} "
                f"optimization_result="
                f"{dual_baseline_optimization_status(best_official_status, best_bank_status)}",
                flush=True,
            )

        print(
            f"official_tiling_profile completed "
            f"baselines={len(workloads) - unsupported} "
            f"searched={candidate_index} runtime_rejected={runtime_rejected} "
            f"unsupported={unsupported}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ProfileError, OSError, ValueError, KeyError, json.JSONDecodeError) as exception:
        print(f"fatal: {exception}", flush=True)
        raise SystemExit(1)
