#!/usr/bin/env python3
"""Collect exactly 20,000 source-derived non-MatMul latency records.

The source inventory contains eight FlashAttentionScoreGrad registrations. A
three-strategy subset has overlapping original capability on the reviewed
BNSD/fp16 lattice. Each source strategy emits an unchanged raw tiling first.
A formal FASG group is admitted only when at least two distinct raw tilings
execute and exactly match the installed operator's full dq/dk/dv output.

The JSONL contains compact group results, never traces. Its ``valid_latency``
arrays contain exactly 20,000 device-event measurements at completion. No tile
field is enumerated or edited; no model, callback, RuntimeKb, CCE data, CPU or
simulator timing, host timeout, or forced worker kill is used.
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
LOCK = json.loads((ROOT / "non_matmul_source_lock.json").read_text(encoding="utf-8"))
SCHEMA = "non_matmul_source_candidate_measurement_v2"


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_catalog() -> tuple[list[dict[str, Any]], dict[str, Any], Any]:
    spec = importlib.util.spec_from_file_location("non_matmul_candidate_catalog", CATALOG_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load source candidate catalog")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = module.catalog()
    audit = module.audit(rows)
    if audit["matmul_included"]:
        raise RuntimeError("MatMul is forbidden in this campaign")
    if int(audit["formal_latency_target"]) != 20_000:
        raise RuntimeError("formal latency target must be exactly 20,000")
    return rows, audit, module


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
        raise RuntimeError("unsupported source campaign op: {}".format(workload["op"]))
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
        parsed = {"status": "failed", "error": "worker emitted no parseable MULTIOP_NPU_RESULT"}
    return parsed, output, elapsed_ms, completed.returncode


def compact_failure(output: str) -> str:
    lines = [line for line in output.splitlines() if line.startswith("MULTIOP_NPU_RESULT ")]
    return (lines[-1] if lines else output[-4000:])[:4000]


def read_progress(path: Path) -> tuple[set[str], int, int]:
    completed: set[str] = set()
    admitted = 0
    rejected = 0
    if not path.is_file():
        return completed, admitted, rejected
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("schema") != SCHEMA or row.get("status") not in ("admitted", "rejected"):
                continue
            key = row.get("group_key")
            if not isinstance(key, str):
                continue
            completed.add(key)
            if row["status"] == "admitted":
                admitted += int(row.get("valid_latency_count", 0))
            else:
                rejected += 1
    return completed, admitted, rejected


def emit(handle: Any, row: dict[str, Any]) -> None:
    encoded = json.dumps(row, ensure_ascii=False, sort_keys=True)
    print("SOURCE_TILING_GROUP_RESULT " + encoded, flush=True)
    handle.write(encoded + "\n")
    handle.flush()


def validate_custom_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = (
        "operator", "strategy_class", "strategy_priority", "official_commit", "custom_opp_root",
        "source_package_sha256", "source_tiling_observation_enabled", "overlay_manifest_sha256",
    )
    if any(key not in manifest for key in required):
        raise RuntimeError("invalid custom OPP manifest: {}".format(path))
    if manifest["operator"] != "FlashAttentionScoreGrad":
        raise RuntimeError("custom OPP package is not FlashAttentionScoreGrad")
    if manifest["official_commit"] != LOCK["sources"]["cann_ops_adv"]["commit"]:
        raise RuntimeError("custom OPP package source differs from the pinned official source")
    if manifest["source_tiling_observation_enabled"] is not True or not manifest["overlay_manifest_sha256"]:
        raise RuntimeError("custom OPP package does not attest raw source tiling observation")
    root = Path(manifest["custom_opp_root"])
    if not (root / "vendors" / str(manifest["vendor"])).is_dir():
        raise RuntimeError("isolated custom OPP vendor is missing")
    manifest["manifest_path"] = str(path)
    return manifest


def read_tiling_observations(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    result: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("schema") == "fasg_raw_tiling_observation_v1":
                result.append(item)
    return result


def raw_identity(item: dict[str, Any]) -> str:
    fields = ("tiling_key", "block_dim", "raw_bytes", "raw_fnv1a64")
    if any(field not in item for field in fields):
        raise RuntimeError("source raw-tiling observation is incomplete")
    return ":".join(str(item[field]) for field in fields)


def successful_observation(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    rows = read_tiling_observations(path)
    if not rows:
        return None, "custom source emitted no raw-tiling observation"
    identities = {raw_identity(row) for row in rows}
    if len(identities) != 1:
        return None, "one original strategy context emitted multiple raw identities"
    if any(int(row.get("status", -1)) != 0 for row in rows):
        return None, "original strategy returned a non-success graph status"
    output = dict(rows[-1])
    output["raw_tiling_identity"] = next(iter(identities))
    output["observation_count"] = len(rows)
    return output, None


def candidate_descriptor(package: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "isolated_original_source_strategy",
        "id": package["strategy_class"],
        "priority": package["strategy_priority"],
        "official_commit": package["official_commit"],
        "source_package_sha256": package["source_package_sha256"],
    }


def summarize(path: Path) -> dict[str, int]:
    completed, admitted, rejected = read_progress(path)
    fasg_groups = 0
    raw_tiling_sum = 0
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("schema") != SCHEMA or row.get("workload", {}).get("op") != "flash_attention_score_grad":
                    continue
                raw_tiling_sum += int(row.get("source_discovery", {}).get("distinct_raw_tilings", 0))
                if row.get("status") == "admitted":
                    fasg_groups += 1
    return {
        "completed_groups": len(completed),
        "formal_valid_latency_records": admitted,
        "rejected_groups": rejected,
        "admitted_fasg_multi_tiling_groups": fasg_groups,
        "sum_of_group_distinct_raw_tilings": raw_tiling_sum,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--progress", required=True, type=Path)
    parser.add_argument("--device", required=True, type=int)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--fasg-custom-opp-manifest", action="append", default=[], type=Path)
    args = parser.parse_args()
    if not args.runner.is_file():
        raise RuntimeError("worker runner does not exist: {}".format(args.runner))
    if args.device < 0 or args.warmup < 0 or args.samples < 1:
        raise RuntimeError("invalid device/warmup/samples")

    workloads, audit, catalog_module = load_catalog()
    target = int(audit["formal_latency_target"])
    required_classes = tuple(catalog_module.FASG_MULTI_TILING_SOURCE_CLASSES)
    packages = [validate_custom_manifest(path) for path in args.fasg_custom_opp_manifest]
    by_class = {str(package["strategy_class"]): package for package in packages}
    if len(by_class) != len(packages) or set(by_class) != set(required_classes):
        raise RuntimeError("expected exactly the source-overlapping FASG strategies: {}".format(
            ", ".join(required_classes)))
    packages = [by_class[name] for name in required_classes]

    args.progress.parent.mkdir(parents=True, exist_ok=True)
    completed, admitted, prior_rejected = read_progress(args.progress)
    if admitted > target:
        raise RuntimeError("existing progress exceeds the 20,000 formal-record target")
    base_environment = dict(os.environ)
    base_environment.pop("ASCEND_CUSTOM_OPP_PATH", None)
    base_environment.pop("FASG_TILING_AUDIT_PATH", None)
    runner_hash = digest_file(args.runner)
    print("SOURCE_TILING_CAMPAIGN_BEGIN " + json.dumps({
        "schema": SCHEMA, "matmul_included": False, "formal_latency_target": target,
        "formal_latency_records_already_present": admitted,
        "semantic_workloads": len(workloads),
        "full_fasg_source_registry_count": int(audit["full_fasg_original_strategy_registry_count"]),
        "source_multi_tiling_subset": list(required_classes),
        "per_fasg_shape_minimum_distinct_raw_tilings": 2,
        "source_discovery_upper_bound": int(audit["source_discovery_upper_bound"]),
        "timing": {"warmup": args.warmup, "samples": args.samples, "kind": "device_event_only"},
        "no_cpu_or_simulator_latency": True, "no_host_timeout_or_forced_worker_kill": True,
        "historical_latency_or_tiling_records_read": 0, "cce_data_or_cost_model_read": 0,
        "temporary_reference_storage": "/tmp only", "resume": "admitted/rejected groups are terminal",
    }, ensure_ascii=False, sort_keys=True), flush=True)

    other_workloads = [workload for workload in workloads if workload["op"] != "flash_attention_score_grad"]
    fasg_workloads = [workload for workload in workloads if workload["op"] == "flash_attention_score_grad"]
    with args.progress.open("a", encoding="utf-8") as progress:
        for workload in other_workloads:
            if admitted >= target:
                break
            group_key = stable_hash({"workload": workload, "runner": runner_hash, "kind": "single_original_path"})
            if group_key in completed:
                continue
            result, output, wall_ms, rc = run_worker(
                worker_args(args.runner, workload, args.device, args.warmup, args.samples), base_environment)
            if rc == 0 and result.get("status") == "success":
                emit(progress, {
                    "schema": SCHEMA, "group_key": group_key, "status": "admitted", "workload": workload,
                    "valid_latency_count": 1,
                    "valid_latency": [{"candidate": {"kind": "source_native_single_path", "id": workload["op"] + "_original"},
                                       "result": result, "runner_rc": rc, "worker_wall_ms": wall_ms}],
                    "source_discovery": {"original_source_paths": 1, "distinct_raw_tilings": 1},
                    "source_rule": "the source exposes one eligible semantic path; no alternative is invented",
                })
                admitted += 1
            else:
                emit(progress, {
                    "schema": SCHEMA, "group_key": group_key, "status": "rejected", "workload": workload,
                    "valid_latency_count": 0, "rejection_reason": "single original source path did not complete",
                    "failure": compact_failure(output), "runner_rc": rc, "worker_wall_ms": wall_ms,
                })
            completed.add(group_key)

        for workload in fasg_workloads:
            if admitted >= target:
                break
            group_key = stable_hash({"workload": workload, "runner": runner_hash,
                                     "strategy_classes": required_classes, "kind": "source_multi_tiling_group"})
            if group_key in completed:
                continue
            remaining = target - admitted
            if remaining < 2:
                # A FASG group is never reduced to one candidate merely to
                # fill a counter: that would violate the per-shape
                # multi-tiling contract. This state cannot arise in a fresh
                # campaign (the single-path lane runs first), but it prevents
                # an externally malformed resume file from exceeding target.
                break
            # Pairs enforce the multi-tiling requirement. A triple repairs
            # parity if a naturally single-path workload was rejected.
            needed = 3 if remaining % 2 else 2
            with tempfile.TemporaryDirectory(prefix="fasg_source_reference_") as temp:
                reference_path = Path(temp) / "reference.bin"
                command = worker_args(args.runner, workload, args.device, args.warmup, args.samples)
                command.extend(("--write-reference", str(reference_path)))
                reference, reference_output, reference_wall_ms, reference_rc = run_worker(command, base_environment)
                if reference_rc != 0 or reference.get("status") != "success" or not reference_path.is_file():
                    emit(progress, {
                        "schema": SCHEMA, "group_key": group_key, "status": "rejected", "workload": workload,
                        "valid_latency_count": 0, "rejection_reason": "installed reference did not complete",
                        "failure": compact_failure(reference_output), "runner_rc": reference_rc,
                        "worker_wall_ms": reference_wall_ms,
                    })
                    completed.add(group_key)
                    continue

                discovered: list[tuple[dict[str, Any], dict[str, Any], dict[str, str], float]] = []
                identities: set[str] = set()
                discovery_failures: list[str] = []
                for package in packages:
                    candidate = candidate_descriptor(package)
                    candidate_key = stable_hash({"group": group_key, "candidate": candidate})
                    environment = dict(base_environment)
                    environment["ASCEND_CUSTOM_OPP_PATH"] = str(package["custom_opp_root"])
                    audit_path = Path(temp) / ("discover_" + candidate_key + ".jsonl")
                    environment["FASG_TILING_AUDIT_PATH"] = str(audit_path)
                    command = worker_args(args.runner, workload, args.device, args.warmup, args.samples)
                    command.append("--source-tiling-only")
                    result, output, wall_ms, rc = run_worker(command, environment)
                    observation, reason = successful_observation(audit_path)
                    if rc != 0 or result.get("status") != "success" or observation is None:
                        discovery_failures.append(candidate["id"] + ": " + (reason or compact_failure(output)))
                        continue
                    identity = str(observation["raw_tiling_identity"])
                    if identity not in identities:
                        identities.add(identity)
                        discovered.append((candidate, observation, environment, wall_ms))

                discovery = {
                    "attempted_original_strategies": len(packages),
                    "successful_original_strategies": len(discovered),
                    "distinct_raw_tilings": len(identities),
                    "full_registry_strategy_count": int(audit["full_fasg_original_strategy_registry_count"]),
                }
                if len(discovered) < needed:
                    emit(progress, {
                        "schema": SCHEMA, "group_key": group_key, "status": "rejected", "workload": workload,
                        "valid_latency_count": 0, "source_discovery": discovery,
                        "rejection_reason": "fewer than {} distinct raw tilings from eligible original strategies".format(needed),
                        "discovery_failures": discovery_failures[:3],
                    })
                    completed.add(group_key)
                    continue

                valid: list[dict[str, Any]] = []
                execution_failures: list[str] = []
                for candidate, observation, environment, discovery_wall_ms in discovered:
                    if len(valid) >= needed:
                        break
                    candidate_key = stable_hash({"group": group_key, "candidate": candidate})
                    audit_path = Path(temp) / ("execute_" + candidate_key + ".jsonl")
                    environment = dict(environment)
                    environment["FASG_TILING_AUDIT_PATH"] = str(audit_path)
                    command = worker_args(args.runner, workload, args.device, args.warmup, args.samples)
                    command.extend(("--compare-reference", str(reference_path)))
                    result, output, wall_ms, rc = run_worker(command, environment)
                    execution_observation, reason = successful_observation(audit_path)
                    matched = bool(result.get("output_reference_checked")) and bool(result.get("output_reference_equal"))
                    same_identity = execution_observation is not None and (
                        execution_observation.get("raw_tiling_identity") == observation["raw_tiling_identity"])
                    if rc != 0 or result.get("status") != "success" or not matched or not same_identity:
                        execution_failures.append(candidate["id"] + ": " + (
                            reason or ("output differs from installed reference" if not matched else compact_failure(output))))
                        continue
                    valid.append({
                        "candidate": candidate, "result": result, "runner_rc": rc, "worker_wall_ms": wall_ms,
                        "discovery_wall_ms": discovery_wall_ms, "source_tiling_observation": observation,
                        "execution_tiling_observation": execution_observation,
                        "reference_median_ms": reference.get("median_ms"),
                    })

                if len(valid) < needed:
                    emit(progress, {
                        "schema": SCHEMA, "group_key": group_key, "status": "rejected", "workload": workload,
                        "valid_latency_count": 0, "source_discovery": discovery,
                        "rejection_reason": "fewer than {} distinct source tilings completed with exact output equality".format(needed),
                        "execution_failures": execution_failures[:3],
                    })
                    completed.add(group_key)
                    continue

                emit(progress, {
                    "schema": SCHEMA, "group_key": group_key, "status": "admitted", "workload": workload,
                    "valid_latency_count": needed, "valid_latency": valid[:needed], "source_discovery": discovery,
                    "source_rule": "each retained entry came from an unchanged original source strategy; all retained outputs exactly match the installed reference",
                })
                admitted += needed
                completed.add(group_key)

    result = summarize(args.progress)
    status = "complete" if result["formal_valid_latency_records"] == target else "reserve_exhausted_before_target"
    print("SOURCE_TILING_CAMPAIGN_END " + json.dumps({
        "schema": SCHEMA, "status": status, "matmul_included": False,
        "formal_latency_target": target, "summary": result, "prior_rejected_groups": prior_rejected,
    }, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if status == "complete" else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print("SOURCE_TILING_CAMPAIGN_FATAL " + json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)
