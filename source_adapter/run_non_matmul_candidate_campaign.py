#!/usr/bin/env python3
"""Run finite, source-attested non-MatMul tiling candidates on one NPU.

FlashAttentionScoreGrad runs the installed operator once as the reference, then
asks each separately installed original-strategy custom OPP overlay to generate
one raw tiling identity without launching a kernel. Exact duplicate raw
identities are linked rather than timed again; every distinct identity is run
once with deterministic non-zero input and must match complete dq/dk/dv bytes
before its device-event latency is admitted. Other operators in this first
source catalog expose one source-native path per semantic workload; they are
recorded once, never inflated into fake candidates.

There are no host timeouts, no forced process termination, no CPU/simulator
latency, random shapes, cost-model selection, RuntimeKb, callback timing, or
CCE measurement lookup. A worker failure is recorded and unrelated workloads
continue. Temporary reference bytes live only under /tmp and are removed by the
temporary-directory context when the process exits normally.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CATALOG_PATH = ROOT / "non_matmul_candidate_catalog.py"
LOCK_PATH = ROOT / "non_matmul_source_lock.json"
LOCK = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
SCHEMA = "non_matmul_source_candidate_measurement_v1"


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_catalog() -> list[dict[str, Any]]:
    spec = importlib.util.spec_from_file_location("non_matmul_candidate_catalog", CATALOG_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load source candidate catalog")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = module.catalog()
    audit = module.audit(rows)
    if audit["matmul_included"] or audit["maximum_original_source_candidate_attempts"] > audit["maximum_total_candidate_records"]:
        raise RuntimeError("invalid source candidate catalog contract")
    return rows


def parse_worker_result(output: str) -> dict[str, Any] | None:
    for line in reversed(output.splitlines()):
        marker = "MULTIOP_NPU_RESULT "
        if line.startswith(marker):
            try:
                return json.loads(line[len(marker):])
            except json.JSONDecodeError:
                return None
    return None


def worker_args(runner: Path, workload: dict[str, Any], device: int, warmup: int, samples: int) -> list[str]:
    arguments = [
        str(runner), "--workload-id", workload["workload_id"], "--op", workload["op"],
        "--device", str(device), "--warmup", str(warmup), "--samples", str(samples),
        "--expected-soc", "Ascend910B3",
    ]
    if workload["op"] in ("flash_attention_score_grad", "fused_infer_attention_score"):
        fields = ("batch", "q_heads", "kv_heads", "q_seq", "kv_seq", "head_dim", "layout", "dtype")
    elif workload["op"] == "gather_elements":
        fields = ("shape", "index_shape", "axis", "dtype", "index_dtype")
    elif workload["op"] == "scatter_elements":
        fields = ("shape", "index_shape", "axis", "dtype", "index_dtype", "reduce")
    else:
        raise RuntimeError("catalog contains unsupported source campaign op: {}".format(workload["op"]))
    for field in fields:
        value = workload[field]
        rendered = ",".join(str(item) for item in value) if isinstance(value, list) else str(value)
        arguments.extend(("--" + field.replace("_", "-"), rendered))
    return arguments


def run_worker(arguments: list[str], environment: dict[str, str]) -> tuple[dict[str, Any], str, float, int]:
    started = time.monotonic()
    completed = subprocess.run(arguments, text=True, capture_output=True, env=environment, check=False)
    elapsed_ms = (time.monotonic() - started) * 1000.0
    output = completed.stdout + completed.stderr
    parsed = parse_worker_result(completed.stdout)
    if parsed is None:
        parsed = {
            "status": "failed",
            "error": "worker emitted no parseable MULTIOP_NPU_RESULT",
        }
    return parsed, output, elapsed_ms, completed.returncode


def read_progress(path: Path) -> set[str]:
    """Read terminal result identities already recorded in JSONL.

    A failed source attempt is terminal just like a successful one. This keeps
    the campaign genuinely finite: retries after a transient device failure
    require a deliberate new progress path instead of silently appending an
    unbounded sequence of duplicate failure rows.
    """
    completed: set[str] = set()
    if not path.is_file():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("schema") == SCHEMA and item.get("status") in ("success", "failed"):
                completed.add(str(item.get("record_key")))
    return completed


def emit(handle: Any, item: dict[str, Any]) -> None:
    encoded = json.dumps(item, ensure_ascii=False, sort_keys=True)
    print("SOURCE_CANDIDATE_RESULT " + encoded, flush=True)
    handle.write(encoded + "\n")
    handle.flush()


def emit_reference_blocked_candidates(handle: Any, workload: dict[str, Any],
                                      candidates: list[tuple[dict[str, Any], dict[str, Any]]],
                                      runner_hash: str, source_rule: str,
                                      counts: dict[str, int]) -> None:
    """Close candidate rows that cannot run without an installed reference.

    This is explicitly *not* a source-strategy rejection. It records why the
    strategy was not launched, lets unrelated workloads continue, and retains
    a hard finite record ceiling.
    """
    for candidate, _ in candidates:
        key = stable_hash({"workload": workload, "candidate": candidate, "runner": runner_hash})
        emit(handle, {
            "schema": SCHEMA, "record_key": key, "status": "failed", "workload": workload,
            "candidate": candidate,
            "result": {"status": "not_started", "error": "installed reference unavailable"},
            "runner_rc": None, "worker_wall_ms": None,
            "rejection_reason": "not_started_because_installed_reference_failed",
            "source_rule": source_rule,
        })
        counts["failed"] += 1


def validate_custom_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = ("operator", "strategy_class", "strategy_priority", "official_commit", "custom_opp_root", "source_package_sha256",
                "source_tiling_observation_enabled", "overlay_manifest_sha256")
    if any(key not in manifest for key in required):
        raise RuntimeError("invalid custom OPP manifest: {}".format(path))
    if manifest["operator"] != "FlashAttentionScoreGrad" or manifest["official_commit"] != LOCK["sources"]["cann_ops_adv"]["commit"]:
        raise RuntimeError("custom OPP manifest does not match the pinned FASG source: {}".format(path))
    if manifest["source_tiling_observation_enabled"] is not True or not manifest["overlay_manifest_sha256"]:
        raise RuntimeError("custom OPP package does not attest raw source tiling observation: {}".format(path))
    root = Path(manifest["custom_opp_root"])
    vendor = root / "vendors" / str(manifest["vendor"])
    if not vendor.is_dir():
        raise RuntimeError("custom OPP vendor directory is missing: {}".format(vendor))
    manifest["manifest_path"] = str(path)
    return manifest


def compact_failure(worker_output: str) -> str:
    lines = [line for line in worker_output.splitlines() if line.startswith("MULTIOP_NPU_RESULT ")]
    if lines:
        return lines[-1][:4000]
    return worker_output[-4000:]


def read_tiling_observations(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("schema") == "fasg_raw_tiling_observation_v1":
                rows.append(item)
    return rows


def tiling_identity(item: dict[str, Any]) -> str:
    required = ("tiling_key", "block_dim", "raw_bytes", "raw_fnv1a64")
    if any(key not in item for key in required):
        raise RuntimeError("source tiling audit row is incomplete")
    return ":".join(str(item[key]) for key in required)


def one_successful_tiling_observation(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    rows = read_tiling_observations(path)
    if not rows:
        return None, "isolated custom source emitted no raw-tiling observation"
    identities = {tiling_identity(row) for row in rows}
    if len(identities) != 1:
        return None, "one original context emitted multiple raw tiling identities"
    if any(int(row.get("status", -1)) != 0 for row in rows):
        return None, "original source tiling returned a non-success graph status"
    observation = dict(rows[-1])
    observation["observation_count"] = len(rows)
    observation["raw_tiling_identity"] = next(iter(identities))
    return observation, None


def summarize_recorded_source_tilings(path: Path) -> dict[str, int]:
    generated = 0
    timed = 0
    raw_identities: set[tuple[str, str]] = set()
    duplicates = 0
    if not path.is_file():
        return {"original_strategy_successes": 0, "distinct_raw_tilings": 0,
                "duplicate_strategy_attempts": 0, "distinct_raw_tilings_timed": 0}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("schema") != SCHEMA or row.get("workload", {}).get("op") != "flash_attention_score_grad":
                continue
            if row.get("candidate", {}).get("kind") != "isolated_original_source_strategy":
                continue
            observation = row.get("source_tiling_observation")
            if not isinstance(observation, dict) or "raw_tiling_identity" not in observation:
                continue
            generated += 1
            key = (str(row["workload"]["workload_id"]), str(observation["raw_tiling_identity"]))
            raw_identities.add(key)
            if row.get("result", {}).get("status") == "deduplicated_same_raw_tiling":
                duplicates += 1
            if isinstance(row.get("execution_tiling_observation"), dict):
                timed += 1
    return {
        "original_strategy_successes": generated,
        "distinct_raw_tilings": len(raw_identities),
        "duplicate_strategy_attempts": duplicates,
        "distinct_raw_tilings_timed": timed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--progress", required=True, type=Path)
    parser.add_argument("--device", required=True, type=int,
                        help="logical device id after ASCEND_RT_VISIBLE_DEVICES mapping")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--fasg-custom-opp-manifest", action="append", default=[], type=Path,
                        help="repeat once per isolated original FASG strategy package")
    args = parser.parse_args()
    if not args.runner.is_file():
        raise RuntimeError("worker runner does not exist: {}".format(args.runner))
    if args.device < 0 or args.warmup < 0 or args.samples < 1:
        raise RuntimeError("invalid device/warmup/samples")
    custom = [validate_custom_manifest(path) for path in args.fasg_custom_opp_manifest]
    classes = [str(item["strategy_class"]) for item in custom]
    if len(classes) != len(set(classes)):
        raise RuntimeError("duplicate FASG original strategy manifests")
    expected_strategies = int(LOCK["operators"]["flash_attention_score_grad"]["expected_strategy_count"])
    if len(custom) != expected_strategies:
        raise RuntimeError(
            "expected manifests for all {} original FASG strategies, received {}".format(expected_strategies, len(custom)))
    workloads = load_catalog()
    maximum_rows = sum(
        1 + expected_strategies if workload["op"] == "flash_attention_score_grad" else 1
        for workload in workloads
    )
    maximum_budget = int(LOCK["collection_contract"]["maximum_total_candidate_records"])
    if maximum_rows > maximum_budget:
        raise RuntimeError("source-derived result plan exceeds the campaign record ceiling")
    args.progress.parent.mkdir(parents=True, exist_ok=True)
    completed = read_progress(args.progress)
    base_environment = dict(os.environ)
    base_environment.pop("ASCEND_CUSTOM_OPP_PATH", None)
    base_environment.pop("FASG_TILING_AUDIT_PATH", None)
    runner_hash = digest_file(args.runner)
    plan = {
        "schema": SCHEMA,
        "matmul_included": False,
        "worker_runner_sha256": runner_hash,
        "semantic_workloads": len(workloads),
        "fasg_original_strategy_packages": len(custom),
        "measurement": {"warmup": args.warmup, "samples": args.samples, "timing": "device_event_only"},
        "historical_latency_or_tiling_records_read": 0,
        "cce_data_read": 0,
        "runtime_kb_or_callback_read": 0,
        "forced_worker_timeouts": 0,
        "temporary_reference_storage": "/tmp only; removed after each FASG workload",
        "maximum_result_rows": maximum_rows,
        "retry_policy": "terminal exact records, including failures, are not rerun; use a new progress path for a deliberate retry",
    }
    print("SOURCE_CANDIDATE_CAMPAIGN_BEGIN " + json.dumps(plan, ensure_ascii=False, sort_keys=True), flush=True)
    counts = {
        "success": 0, "failed": 0, "resumed": 0, "output_rejected": 0,
        "fasg_original_strategy_tiling_success": 0,
        "fasg_distinct_raw_tilings": 0,
        "fasg_duplicate_raw_tiling_attempts": 0,
        "fasg_distinct_raw_tilings_timed": 0,
    }
    with args.progress.open("a", encoding="utf-8") as progress:
        for workload in workloads:
            if workload["op"] != "flash_attention_score_grad":
                candidate = {"kind": "source_native_single_path", "id": workload["op"] + "_original"}
                key = stable_hash({"workload": workload, "candidate": candidate, "runner": runner_hash})
                if key in completed:
                    counts["resumed"] += 1
                    continue
                result, worker_output, wall_ms, rc = run_worker(
                    worker_args(args.runner, workload, args.device, args.warmup, args.samples), base_environment)
                status = "success" if rc == 0 and result.get("status") == "success" else "failed"
                row = {
                    "schema": SCHEMA, "record_key": key, "status": status, "workload": workload,
                    "candidate": candidate, "result": result, "runner_rc": rc, "worker_wall_ms": wall_ms,
                    "source_rule": "one original source-native path; no synthetic alternate tiling",
                }
                if status != "success": row["worker_failure"] = compact_failure(worker_output)
                emit(progress, row)
                counts[status] += 1
                continue

            package_candidates = [
                ({
                    "kind": "isolated_original_source_strategy",
                    "id": package["strategy_class"],
                    "priority": package["strategy_priority"],
                    "official_commit": package["official_commit"],
                    "custom_opp_root": package["custom_opp_root"],
                    "source_package_sha256": package["source_package_sha256"],
                }, package)
                for package in custom
            ]
            pending_candidates = [
                (candidate, package)
                for candidate, package in package_candidates
                if stable_hash({"workload": workload, "candidate": candidate, "runner": runner_hash}) not in completed
            ]
            if not pending_candidates:
                counts["resumed"] += len(package_candidates)
                continue
            with tempfile.TemporaryDirectory(prefix="fasg_source_reference_") as temp:
                reference_path = Path(temp) / "reference.bin"
                reference_candidate = {"kind": "installed_reference", "id": "installed_reference"}
                reference_key = stable_hash({"workload": workload, "candidate": reference_candidate, "runner": runner_hash})
                reference_result: dict[str, Any] | None = None
                if reference_key not in completed:
                    command = worker_args(args.runner, workload, args.device, args.warmup, args.samples)
                    command.extend(("--write-reference", str(reference_path)))
                    reference_result, output, wall_ms, rc = run_worker(command, base_environment)
                    status = "success" if rc == 0 and reference_result.get("status") == "success" and reference_path.is_file() else "failed"
                    row = {
                        "schema": SCHEMA, "record_key": reference_key, "status": status, "workload": workload,
                        "candidate": reference_candidate, "result": reference_result, "runner_rc": rc,
                        "worker_wall_ms": wall_ms, "source_rule": "installed original reference only; not a searched candidate",
                    }
                    if status != "success": row["worker_failure"] = compact_failure(output)
                    emit(progress, row)
                    counts[status] += 1
                    if status != "success":
                        emit_reference_blocked_candidates(
                            progress, workload, pending_candidates, runner_hash,
                            "candidate was not launched because the installed reference failed", counts)
                        continue
                else:
                    # Reference bytes are intentionally temporary, so a resumed
                    # FASG workload must regenerate them without recording a
                    # second reference-latency row.
                    command = worker_args(args.runner, workload, args.device, args.warmup, args.samples)
                    command.extend(("--write-reference", str(reference_path)))
                    reference_result, output, _, rc = run_worker(command, base_environment)
                    if rc != 0 or reference_result.get("status") != "success" or not reference_path.is_file():
                        emit_reference_blocked_candidates(
                            progress, workload, pending_candidates, runner_hash,
                            "candidate was not launched because temporary installed-reference regeneration failed", counts)
                        continue
                    counts["resumed"] += 1

                reference_ms = reference_result.get("median_ms") if reference_result else None
                raw_identity_owners: dict[str, dict[str, Any]] = {}
                for candidate, package in pending_candidates:
                    key = stable_hash({"workload": workload, "candidate": candidate, "runner": runner_hash})
                    environment = dict(base_environment)
                    environment["ASCEND_CUSTOM_OPP_PATH"] = str(package["custom_opp_root"])

                    # First invoke original host-side tiling once, with no
                    # workspace allocation or kernel launch. The overlay audit
                    # must produce one stable raw identity; otherwise the
                    # candidate cannot be called source-attested.
                    discovery_path = Path(temp) / ("discovery_" + hashlib.sha256(key.encode()).hexdigest() + ".jsonl")
                    environment["FASG_TILING_AUDIT_PATH"] = str(discovery_path)
                    discovery_command = worker_args(args.runner, workload, args.device, args.warmup, args.samples)
                    discovery_command.append("--source-tiling-only")
                    discovery_result, discovery_output, discovery_wall_ms, discovery_rc = run_worker(
                        discovery_command, environment)
                    observation, observation_error = one_successful_tiling_observation(discovery_path)
                    if discovery_rc != 0 or discovery_result.get("status") != "success" or observation is None:
                        row = {
                            "schema": SCHEMA, "record_key": key, "status": "failed", "workload": workload,
                            "candidate": candidate, "result": {"discovery": discovery_result},
                            "runner_rc": discovery_rc, "worker_wall_ms": discovery_wall_ms,
                            "source_tiling_observation": read_tiling_observations(discovery_path),
                            "rejection_reason": observation_error or "original source tiling discovery worker failed",
                            "source_rule": "unchanged original strategy must expose one successful raw tiling identity before latency measurement",
                        }
                        row["worker_failure"] = compact_failure(discovery_output)
                        emit(progress, row)
                        counts["failed"] += 1
                        continue

                    identity = str(observation["raw_tiling_identity"])
                    counts["fasg_original_strategy_tiling_success"] += 1
                    owner = raw_identity_owners.get(identity)
                    if owner is not None:
                        counts["fasg_duplicate_raw_tiling_attempts"] += 1
                        status = str(owner["status"])
                        row = {
                            "schema": SCHEMA, "record_key": key, "status": status, "workload": workload,
                            "candidate": candidate,
                            "result": {"discovery": discovery_result, "status": "deduplicated_same_raw_tiling"},
                            "runner_rc": discovery_rc, "worker_wall_ms": discovery_wall_ms,
                            "source_tiling_observation": observation,
                            "duplicate_of_record_key": owner["record_key"],
                            "reference_median_ms": reference_ms,
                            "latency_admitted": status == "success",
                            "source_rule": "same observed original raw tiling identity; device timing was retained only from its first source strategy",
                        }
                        emit(progress, row)
                        counts[status] += 1
                        continue

                    counts["fasg_distinct_raw_tilings"] += 1
                    execution_path = Path(temp) / ("execution_" + hashlib.sha256(key.encode()).hexdigest() + ".jsonl")
                    environment["FASG_TILING_AUDIT_PATH"] = str(execution_path)
                    command = worker_args(args.runner, workload, args.device, args.warmup, args.samples)
                    command.extend(("--compare-reference", str(reference_path)))
                    result, output, wall_ms, rc = run_worker(command, environment)
                    matched = bool(result.get("output_reference_checked")) and bool(result.get("output_reference_equal"))
                    execution_observation, execution_error = one_successful_tiling_observation(execution_path)
                    same_identity = execution_observation is not None and execution_observation.get("raw_tiling_identity") == identity
                    status = "success" if (
                        rc == 0 and result.get("status") == "success" and matched and same_identity
                    ) else "failed"
                    row = {
                        "schema": SCHEMA, "record_key": key, "status": status, "workload": workload,
                        "candidate": candidate, "result": result, "runner_rc": rc, "worker_wall_ms": wall_ms,
                        "reference_median_ms": reference_ms,
                        "source_tiling_observation": observation,
                        "execution_tiling_observation": execution_observation,
                        "discovery_wall_ms": discovery_wall_ms,
                        "latency_admitted": status == "success",
                        "source_rule": "one isolated unchanged original strategy registration; raw identity must match between discovery and timed execution",
                    }
                    if status != "success":
                        row["worker_failure"] = compact_failure(output)
                        if rc == 0 and result.get("status") == "success" and not matched:
                            row["rejection_reason"] = "full_dq_dk_dv_output_differs_from_installed_reference"
                            counts["output_rejected"] += 1
                        elif execution_error is not None:
                            row["rejection_reason"] = execution_error
                        elif not same_identity:
                            row["rejection_reason"] = "timed execution raw tiling identity differs from the discovered original source identity"
                    emit(progress, row)
                    counts[status] += 1
                    counts["fasg_distinct_raw_tilings_timed"] += 1
                    raw_identity_owners[identity] = {"record_key": key, "status": status}
    print("SOURCE_CANDIDATE_CAMPAIGN_END " + json.dumps({
        "schema": SCHEMA, "matmul_included": False, "counts_this_invocation": counts,
        "recorded_source_tiling_summary": summarize_recorded_source_tilings(args.progress),
        "status": "complete_with_per_candidate_failures_recorded",
    }, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print("SOURCE_CANDIDATE_CAMPAIGN_FATAL " + json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)
