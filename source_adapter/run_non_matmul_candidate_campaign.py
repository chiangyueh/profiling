#!/usr/bin/env python3
"""Collect output-validated native GatherElements source-tiling contexts.

Every formal group evaluates the finite set of bounded inputs to CANN's
original dynamic GatherElements source.  The source then produces the runtime
flow-table normally; this controller never enumerates, edits, or replays its
fields. A group is admitted only when twenty distinct source contexts launch,
exactly match the installed reference, and complete device-event measurement.
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
SCHEMA = "gather_elements_native_dynamic_measurement_v4"
SOURCE_BACKEND = "acl_op_compiler_custom_opp_real_npu"
# The campaign is intentionally append-only so an interrupted physical-NPU
# run can resume without losing its completed groups.  A single log is kept
# below this limit; the next numeric log is opened before an oversized write.
MAX_LOG_BYTES = 50 * 1024 * 1024
# This executable campaign has one deliberate scope.  Keeping only the
# native CANN operator name prevents an old GatherElementsV2 compatibility
# route from being accepted by accident.
OPERATOR_RUNTIME_NAMES = {"GatherElements": "gather_elements"}
RUNTIME_OPERATOR_NAMES = {value: key for key, value in OPERATOR_RUNTIME_NAMES.items()}


def digest_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_catalog() -> tuple[list[dict[str, Any]], dict[str, Any], Any]:
    spec = importlib.util.spec_from_file_location("non_matmul_candidate_catalog", CATALOG_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load non-MatMul candidate catalog")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    workloads = module.catalog()
    audit = module.audit(workloads)
    if audit.get("matmul_included"):
        raise RuntimeError("catalog unexpectedly contains MatMul")
    return workloads, audit, module


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
    values = [str(runner), "--workload-id", workload["workload_id"], "--op", workload["op"],
              "--device", str(device), "--warmup", str(warmup), "--samples", str(samples),
              "--expected-soc", "Ascend910B3"]
    if workload["op"] in ("flash_attention_score_grad", "fused_infer_attention_score"):
        fields = ("batch", "q_heads", "kv_heads", "q_seq", "kv_seq", "head_dim", "layout", "dtype")
    elif workload["op"] == "gather_elements":
        fields = ("shape", "index_shape", "axis", "dtype", "index_dtype")
    elif workload["op"] == "scatter_elements":
        fields = ("shape", "index_shape", "axis", "dtype", "index_dtype", "reduce")
    else:
        raise RuntimeError("unsupported runtime op: {}".format(workload["op"]))
    for field in fields:
        item = workload[field]
        rendered = ",".join(map(str, item)) if isinstance(item, list) else str(item)
        values += ["--" + field.replace("_", "-"), rendered]
    return values


def run_worker(arguments: list[str], environment: dict[str, str]) -> tuple[dict[str, Any], str, float, int]:
    # No subprocess timeout: a forced host kill can poison a real device task.
    started = time.monotonic()
    done = subprocess.run(arguments, text=True, capture_output=True, env=environment, check=False)
    output = done.stdout + done.stderr
    result = parse_worker_result(done.stdout) or {"status": "failed", "error": "worker emitted no MULTIOP_NPU_RESULT"}
    return result, output, (time.monotonic() - started) * 1000.0, done.returncode


def compact_failure(output: str) -> str:
    records = [line for line in output.splitlines() if line.startswith("MULTIOP_NPU_RESULT ")]
    return (records[-1] if records else output[-2500:])[:2500]


def runner_failure(result: dict[str, Any], output: str) -> str:
    """Prefer the runner's real error over a later missing-audit symptom."""
    error = result.get("error")
    return str(error) if error else compact_failure(output)


class RotatingJsonl:
    """Append JSON lines to numbered logs without ever exceeding 50 MiB."""

    def __init__(self, directory: Path, maximum_bytes: int = MAX_LOG_BYTES) -> None:
        self.directory = directory
        self.maximum_bytes = maximum_bytes
        self.directory.mkdir(parents=True, exist_ok=True)
        numbers = [int(path.stem) for path in self.directory.glob("*.log") if path.stem.isdigit()]
        self.index = max(numbers, default=1)

    def append(self, row: dict[str, Any]) -> Path:
        encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        if len(encoded) > self.maximum_bytes:
            raise RuntimeError("one campaign record exceeds the 50 MiB log limit")
        path = self.directory / f"{self.index}.log"
        existing = path.stat().st_size if path.is_file() else 0
        if existing and existing + len(encoded) > self.maximum_bytes:
            self.index += 1
            path = self.directory / f"{self.index}.log"
        with path.open("ab") as handle:
            handle.write(encoded)
            handle.flush()
        return path


def emit(writer: RotatingJsonl, row: dict[str, Any]) -> None:
    """Write every formal point and every candidate rejection as log records.

    The workload summary is deliberately separate from its formal candidates:
    this makes a missing candidate observable rather than silently hidden in
    a nested progress blob, while resume still keys solely on the summary.
    """
    common = {"schema": row["schema"], "group_key": row["group_key"], "workload": row["workload"]}
    for rank, candidate in enumerate(row.get("valid_latency", []), start=1):
        writer.append({**common, "record_type": "formal_latency_candidate", "rank": rank,
                       "candidate": candidate})
    for phase in ("discovery", "verification", "measurement"):
        for detail in row.get(f"{phase}_failures", []):
            writer.append({**common, "record_type": "candidate_rejected", "phase": phase,
                           "reason": detail})
    summary = dict(row)
    summary.pop("valid_latency", None)
    summary["record_type"] = "workload"
    writer.append(summary)
    # Keep the terminal useful during a long background campaign. Full
    # identities, output checks, latency samples, and rejection causes stay
    # in the rotating logs rather than being duplicated to the terminal.
    workload = row.get("workload", {})
    discovery = row.get("source_discovery", {})
    summary = {
        "schema": row.get("schema"), "status": row.get("status"),
        "op": workload.get("op"), "workload_id": workload.get("workload_id"),
        "valid_latency_count": row.get("valid_latency_count", 0),
        "distinct_raw_tilings": discovery.get("distinct_raw_tilings"),
        "attempted_source_contexts": discovery.get("attempted_source_contexts"),
        "rejection_reason": row.get("rejection_reason"),
    }
    print("SOURCE_TILING_GROUP_RESULT " + json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


def iter_log_rows(directory: Path) -> Any:
    if not directory.is_dir():
        return
    paths = sorted((path for path in directory.glob("*.log") if path.stem.isdigit()), key=lambda path: int(path.stem))
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # A host interruption can leave only the final line
                    # incomplete.  Earlier append-only records remain valid.
                    continue


def read_progress(directory: Path) -> tuple[set[str], dict[str, int], int]:
    completed: set[str] = set()
    admitted: dict[str, int] = {}
    rejected = 0
    for row in iter_log_rows(directory):
        workload = row.get("workload", {})
        if (row.get("schema") != SCHEMA or row.get("record_type") != "workload" or
                row.get("status") not in ("admitted", "rejected", "budget_skipped") or
                not isinstance(row.get("group_key"), str) or not isinstance(workload, dict) or
                workload.get("op") not in RUNTIME_OPERATOR_NAMES):
            continue
        completed.add(row["group_key"])
        if row["status"] == "admitted":
            op = str(workload["op"])
            admitted[op] = admitted.get(op, 0) + int(row.get("valid_latency_count", 0))
        elif row["status"] == "rejected":
            rejected += 1
    return completed, admitted, rejected


def validate_source_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("missing native GatherElements overlay manifest: {}".format(path))
    item: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    required = ("schema", "operator", "runtime_op", "source_kind", "cann_root", "cann_version_file_sha256",
                "installed_source", "installed_source_sha256", "installed_opp_root", "custom_opp_root", "vendor", "vendor_impl_directory", "vendor_root",
                "source_file", "source_file_sha256", "instrumentation", "hardware_envelope_heuristic",
                "strategy_algorithm_changes", "kernel_algorithm_changes", "formal_data_gate")
    if any(key not in item for key in required) or item["schema"] != "gather_elements_native_dynamic_overlay_v4":
        raise RuntimeError("invalid native CANN GatherElements source-overlay manifest: {}".format(path))
    source_file = Path(str(item["source_file"]))
    if not source_file.is_file() or digest_file(source_file) != str(item["source_file_sha256"]):
        raise RuntimeError("native GatherElements source overlay is missing or mismatched: {}".format(source_file))
    runtime_op = OPERATOR_RUNTIME_NAMES.get(str(item["operator"]))
    if runtime_op is None:
        raise RuntimeError("native overlay is not an allowed GatherElements source: {}".format(item["operator"]))
    inst = item["instrumentation"]
    if (not isinstance(inst, dict) or inst.get("enabled") is not True or inst.get("mutates_tiling_context") is not False or
            not inst.get("audit_schema") or not inst.get("audit_environment") or not inst.get("source_budget_environment") or
            inst.get("dispatch_environment") != "GATHER_ELEMENTS_SOURCE_DISPATCH" or
            inst.get("dispatch_value") != "aclop_compile_and_execute"):
        raise RuntimeError("native overlay does not attest its original-source candidate axes")
    envelope = item["hardware_envelope_heuristic"]
    if (not isinstance(envelope, dict) or envelope.get("enabled") is not True or not envelope.get("environment") or
            not envelope.get("audit_field") or tuple(envelope.get("divisors", ())) != (2, 4, 8) or
            int(envelope.get("max_anchors", 0)) < 1):
        raise RuntimeError("native overlay hardware-envelope provenance is invalid")
    vendor_root = Path(str(item["vendor_root"]))
    custom_root = Path(str(item["custom_opp_root"]))
    installed_root = Path(str(item["installed_opp_root"]))
    expected_impl = str(item["vendor"]) + "_impl"
    source_parent = vendor_root / "op_impl" / "ai_core" / "tbe" / str(item["vendor_impl_directory"]) / "dynamic"
    if (not custom_root.is_dir() or not installed_root.is_dir() or not vendor_root.is_dir() or
            vendor_root != custom_root / "vendors" / str(item["vendor"]) or
            str(item["vendor_impl_directory"]) != expected_impl or not source_parent.is_dir()):
        raise RuntimeError("native GatherElements private custom-OPP layout is incomplete")
    if item["strategy_algorithm_changes"] is not False or item["kernel_algorithm_changes"] is not False:
        raise RuntimeError("native GatherElements overlay is not source-preserving")
    item["runtime_op"] = runtime_op
    item["manifest_path"] = str(path)
    return item


def read_observations(path: Path, schema: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if value.get("schema") == schema:
            rows.append(value)
    return rows


def source_tiling_identity(observation: dict[str, Any]) -> str:
    fields = ("source_variant_sha256", "aiv_core_cap", "ub_cap_divisor")
    if any(field not in observation for field in fields):
        raise RuntimeError("source audit omitted its native tiling-context identity")
    return ":".join(str(observation[field]) for field in fields)


def successful_observation(path: Path, package: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    rows = read_observations(path, str(package["instrumentation"]["audit_schema"]))
    if not rows:
        return None, "original-source overlay emitted no audit observation"
    identities = {source_tiling_identity(row) for row in rows}
    if len(identities) != 1:
        return None, "one source execution emitted multiple tiling-context identities"
    if any(int(row.get("status", -1)) != 0 for row in rows):
        return None, "original source tiler returned a non-success status"
    result = dict(rows[-1])
    result["source_tiling_identity"] = next(iter(identities))
    result["observation_count"] = len(rows)
    return result, None


def candidate_descriptor(package: dict[str, Any], cap: int, l2_divisor: int = 1) -> dict[str, Any]:
    if l2_divisor not in (1, 2, 4, 8):
        raise RuntimeError("invalid source L2-envelope divisor")
    heuristic = l2_divisor != 1
    envelope = package["hardware_envelope_heuristic"]
    return {
        "kind": "hardware_rule_heuristic_from_original_source_context" if heuristic else "original_source_tiler_with_runtime_bounded_core_budget",
        "origin": "hardware_rule_heuristic" if heuristic else "original_source",
        "id": package.get("strategy_class") or "original_semantic_dispatch",
        "priority": package.get("strategy_priority"), "aiv_core_cap": int(cap),
        "hardware_envelope_resource": envelope.get("resource"), "hardware_envelope_divisor": int(l2_divisor),
        "source_file_sha256": package["source_file_sha256"],
        "source_kind": package["source_kind"],
    }


def candidate_label(candidate: dict[str, Any]) -> str:
    resource = candidate.get("hardware_envelope_resource") or "none"
    return "{}@aiv{}@{}/{}".format(candidate["id"], candidate["aiv_core_cap"], resource, candidate["hardware_envelope_divisor"])


def context_matches(observation: dict[str, Any] | None, candidate: dict[str, Any], package: dict[str, Any]) -> bool:
    if observation is None or str(observation.get("aiv_core_cap")) != str(candidate["aiv_core_cap"]):
        return False
    envelope = package["hardware_envelope_heuristic"]
    if envelope["enabled"]:
        return str(observation.get(str(envelope["audit_field"]), "1")) == str(candidate["hardware_envelope_divisor"])
    return candidate["hardware_envelope_divisor"] == 1


def source_environment(base: dict[str, str], package: dict[str, Any], candidate: dict[str, Any], audit: Path) -> dict[str, str]:
    environment = dict(base)
    # This is CANN 8.1's normal custom-OPP loader contract: retain the real
    # installed OPP root and add exactly one private vendor directory for the
    # source candidate.  The worker process is isolated, so it cannot leak
    # this selection to other users or later reference workers.
    environment["ASCEND_OPP_PATH"] = str(package["installed_opp_root"])
    environment["ASCEND_CUSTOM_OPP_PATH"] = str(package["vendor_root"])
    environment[str(package["instrumentation"]["audit_environment"])] = str(audit)
    environment[str(package["instrumentation"]["dispatch_environment"])] = str(package["instrumentation"]["dispatch_value"])
    environment[str(package["instrumentation"]["source_budget_environment"])] = str(candidate["aiv_core_cap"])
    envelope = package["hardware_envelope_heuristic"]
    if envelope["enabled"]:
        environment[str(envelope["environment"])] = str(candidate["hardware_envelope_divisor"])
    return environment


def source_audit_emitted(path: Path, package: dict[str, Any], candidate: dict[str, Any]) -> tuple[bool, str]:
    """Confirm that this process really loaded the source host tiler.

    A non-success tiling status is allowed here: the preflight is about the
    deployment/audit path, not about admitting a particular tiling.  It must,
    however, emit the requested context and a complete source identity.
    """
    rows = read_observations(path, str(package["instrumentation"]["audit_schema"]))
    if not rows:
        return False, "original-source overlay emitted no audit observation"
    for row in rows:
        try:
            source_tiling_identity(row)
        except RuntimeError:
            continue
        if context_matches(row, candidate, package):
            return True, ""
    return False, "source audit did not identify the requested source context"


def source_audit_preflight(args: Any, planned: dict[str, list[dict[str, Any]]],
                           packages: dict[str, list[dict[str, Any]]], caps: tuple[int, ...],
                           base_env: dict[str, str]) -> list[dict[str, Any]]:
    """Run the smallest real-NPU deployment gate before formal measurement.

    It covers one installed viability launch and one source-tiling audit for
    the selected operator.  A failed gate ends the campaign before any
    semantic-shape search can create repeated rejects.
    """
    checks: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="source_tiling_preflight_") as temporary:
        root = Path(temporary)
        for op in sorted(planned):
            workload = planned[op][0]
            result, output, wall, rc = run_worker(worker_args(args.runner, workload, args.device, 0, 0), base_env)
            reference_ok = rc == 0 and result.get("status") == "success"
            checks.append({"operator": op, "workload_id": workload["workload_id"], "kind": "installed_reference_viability",
                           "status": "passed" if reference_ok else "failed", "worker_wall_ms": wall,
                           "worker_return_code": rc, "worker_status": result.get("status"),
                           "runner_error": None if reference_ok else result.get("error"),
                           "failure": None if reference_ok else runner_failure(result, output)})
            if not reference_ok:
                continue
            for package in packages[op]:
                # A 20-core native source launch is the deployment gate.  The
                # lower caps are evaluated only after this path is proven.
                candidate = candidate_descriptor(package, caps[-1])
                audit = root / (op + "_" + stable_hash({"package": package["source_file_sha256"], "candidate": candidate}) + ".jsonl")
                result, output, wall, rc = run_worker(
                    worker_args(args.runner, workload, args.device, 0, 0) + ["--source-tiling-only", "1"],
                    source_environment(base_env, package, candidate, audit))
                observed, reason = source_audit_emitted(audit, package, candidate)
                source_ok = (observed and rc == 0 and result.get("status") == "success" and
                             result.get("backend") == SOURCE_BACKEND)
                checks.append({"operator": op, "workload_id": workload["workload_id"], "kind": "source_tiler_audit",
                               "strategy": candidate["id"], "aiv_core_cap": candidate["aiv_core_cap"],
                               "status": "passed" if source_ok else "failed", "worker_return_code": rc,
                               "worker_status": result.get("status"), "worker_wall_ms": wall,
                               "runner_backend": result.get("backend"),
                               "runner_error": None if source_ok else result.get("error"),
                               "failure": None if source_ok else (
                                   runner_failure(result, output) if (rc != 0 or result.get("status") != "success")
                                   else (reason if not observed else
                                         ("runner used unexpected backend: {}".format(result.get("backend"))
                                          if result.get("backend") != SOURCE_BACKEND else compact_failure(output))))})
    return checks


def plan_packages(manifests: list[dict[str, Any]], selected_operator: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for package in manifests:
        grouped.setdefault(str(package["runtime_op"]), []).append(package)
    if set(grouped) != {selected_operator}:
        raise RuntimeError("this campaign accepts exactly one {} source package; found={}".format(
            selected_operator, sorted(grouped)))
    if (len(grouped[selected_operator]) != 1 or
            grouped[selected_operator][0]["hardware_envelope_heuristic"]["enabled"] is not True):
        raise RuntimeError("GatherElements requires one original semantic dispatcher plus its declared conditional UB envelope")
    return grouped


def source_supported_workload(workload: dict[str, Any]) -> bool:
    """Filter before NPU work using the native GatherElements declaration.

    Its source declaration accepts x in fp16/bf16/fp32/int32 and index in
    int32.  Sending CANN-8.1's installed-only int64 index variants through
    the native overlay would manufacture predictable rejects and cannot
    contribute to the 5,000 real source-kernel measurements.
    """
    return (workload.get("op") == "gather_elements" and
            workload.get("dtype") in ("fp16", "bf16", "fp32", "int32") and
            workload.get("index_dtype") == "int32")


def select_heuristic_anchors(discovered: dict[str, dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    by_id: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for item in discovered.values():
        for origin in item["source_origins"]:
            if origin["hardware_envelope_divisor"] != 1:
                continue
            key = stable_hash(origin)
            if key not in seen:
                seen.add(key)
                by_id.setdefault(str(origin["id"]), []).append(origin)
    for values in by_id.values():
        values.sort(key=lambda row: (int(row["aiv_core_cap"]), str(row["id"])))
    selected: list[dict[str, Any]] = []
    while len(selected) < maximum and any(by_id.values()):
        for name in sorted(by_id):
            if by_id[name] and len(selected) < maximum:
                selected.append(by_id[name].pop(0))
    return selected


def discover_group(args: Any, workload: dict[str, Any], packages: list[dict[str, Any]], caps: tuple[int, ...], minimum: int,
                   base_env: dict[str, str], group_key: str, temp: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any], list[str]]:
    discovered: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    successful_contexts = original_contexts = heuristic_contexts = 0
    def report(anchors: list[dict[str, Any]], source_audit_missing: bool = False) -> dict[str, Any]:
        return {"attempted_source_contexts": original_contexts + heuristic_contexts,
                "attempted_original_source_contexts": original_contexts,
                "attempted_hardware_rule_heuristic_contexts": heuristic_contexts,
                "successful_source_contexts": successful_contexts, "distinct_raw_tilings": len(discovered),
                "heuristic_anchor_count": len(anchors), "source_audit_missing": source_audit_missing}

    def attempt(package: dict[str, Any], candidate: dict[str, Any], kind: str) -> bool:
        nonlocal successful_contexts, original_contexts, heuristic_contexts
        if kind == "original": original_contexts += 1
        else: heuristic_contexts += 1
        audit = temp / ("discover_" + stable_hash({"g": group_key, "c": candidate}) + ".jsonl")
        result, output, wall, rc = run_worker(worker_args(args.runner, workload, args.device, 0, 0) + ["--source-tiling-only", "1"],
                                              source_environment(base_env, package, candidate, audit))
        observed, reason = successful_observation(audit, package)
        if result.get("backend") != SOURCE_BACKEND:
            failures.append(candidate_label(candidate) + ": runner used unexpected backend {}".format(result.get("backend")))
            return True
        if reason == "original-source overlay emitted no audit observation":
            failures.append(candidate_label(candidate) + ": " + reason)
            # This is a deployment/instrumentation failure, not an invalid
            # tiling. Retrying its remaining caps would only create duplicate
            # rejection records and cannot produce a formal candidate.
            return True
        if rc != 0 or result.get("status") != "success" or not context_matches(observed, candidate, package):
            failures.append(candidate_label(candidate) + ": " + (reason or compact_failure(output)))
            return False
        successful_contexts += 1
        identity = str(observed["source_tiling_identity"])
        entry = discovered.setdefault(identity, {"candidate": candidate, "package": package, "source_origins": [],
                                                 "source_tiling_observation": observed, "discovery_wall_ms": wall})
        entry["source_origins"].append(candidate)
        return False
    for package in packages:
        for cap in caps:
            if attempt(package, candidate_descriptor(package, cap), "original"):
                return discovered, report([], source_audit_missing=True), failures
    anchors: list[dict[str, Any]] = []
    envelope = packages[0]["hardware_envelope_heuristic"]
    if len(discovered) < minimum and envelope["enabled"] and all(package["hardware_envelope_heuristic"] == envelope for package in packages):
        anchors = select_heuristic_anchors(discovered, int(envelope["max_anchors"]))
        by_id = {
            str(package.get("strategy_class") or "original_semantic_dispatch"): package
            for package in packages
        }
        for anchor in anchors:
            for divisor in envelope["divisors"]:
                package = by_id[str(anchor["id"])]
                if attempt(package, candidate_descriptor(package, int(anchor["aiv_core_cap"]), int(divisor)), "heuristic"):
                    return discovered, report(anchors, source_audit_missing=True), failures
    return discovered, report(anchors), failures


def execute_group(args: Any, workload: dict[str, Any], packages: list[dict[str, Any]], caps: tuple[int, ...], minimum: int,
                  base_env: dict[str, str], group_key: str, temp: Path) -> tuple[str, dict[str, Any]]:
    reference_path = temp / "reference.bin"
    reference, output, wall, rc = run_worker(worker_args(args.runner, workload, args.device, 0, 0) + ["--write-reference", str(reference_path)], base_env)
    if rc != 0 or reference.get("status") != "success" or not reference_path.is_file():
        return "rejected", {"rejection_reason": "installed reference execution failed", "failure": compact_failure(output), "worker_wall_ms": wall}
    discovered, discovery, failures = discover_group(args, workload, packages, caps, minimum, base_env, group_key, temp)
    if discovery.get("source_audit_missing"):
        return "rejected", {"rejection_reason": "source tiling audit was absent; this is a deployment failure, not a tiling rejection",
                              "source_discovery": discovery, "discovery_failures": failures}
    if len(discovered) < minimum:
        return "rejected", {"rejection_reason": "fewer than 20 distinct native source-tiling contexts from the complete source search", "source_discovery": discovery, "discovery_failures": failures}
    verified: list[dict[str, Any]] = []
    verification_failures: list[str] = []
    for identity, item in discovered.items():
        candidate, package = item["candidate"], item["package"]
        audit = temp / ("verify_" + stable_hash({"g": group_key, "c": candidate}) + ".jsonl")
        result, output, wall, rc = run_worker(worker_args(args.runner, workload, args.device, 0, 0) + ["--compare-reference", str(reference_path)],
                                              source_environment(base_env, package, candidate, audit))
        observed, reason = successful_observation(audit, package)
        equal = bool(result.get("output_reference_checked")) and bool(result.get("output_reference_equal"))
        if (rc != 0 or result.get("status") != "success" or result.get("backend") != SOURCE_BACKEND or not equal or
                not context_matches(observed, candidate, package) or observed.get("source_tiling_identity") != identity):
            verification_failures.append(candidate_label(candidate) + ": " + (
                reason or ("unexpected backend {}".format(result.get("backend"))
                           if result.get("backend") != SOURCE_BACKEND else "output/reference or source identity mismatch")))
            continue
        verified.append({**item, "verification_result": result, "verification_wall_ms": wall, "verification_tiling_observation": observed})
    if len(verified) < minimum:
        return "rejected", {"rejection_reason": "fewer than 20 distinct source tilings passed exact output validation", "source_discovery": discovery,
                              "successful_verified_distinct_tilings": len(verified), "discovery_failures": failures,
                              "verification_failures": verification_failures}
    measured: list[dict[str, Any]] = []
    measurement_failures: list[str] = []
    # Discovery and reference validation above intentionally process the
    # complete raw source set.  Timing exactly twenty deterministic identities
    # per admitted shape keeps the 5,000-record contract exact while retaining
    # a non-random ranking set. If a timed candidate fails, the next validated
    # identity replaces it; failures never count.
    verified.sort(key=lambda item: str(item["source_tiling_observation"]["source_tiling_identity"]))
    for item in verified:
        if len(measured) == minimum:
            break
        candidate, package = item["candidate"], item["package"]
        audit = temp / ("measure_" + stable_hash({"g": group_key, "c": candidate}) + ".jsonl")
        result, output, wall, rc = run_worker(worker_args(args.runner, workload, args.device, args.warmup, args.samples) + ["--compare-reference", str(reference_path)],
                                              source_environment(base_env, package, candidate, audit))
        observed, reason = successful_observation(audit, package)
        equal = bool(result.get("output_reference_checked")) and bool(result.get("output_reference_equal"))
        if (rc != 0 or result.get("status") != "success" or result.get("backend") != SOURCE_BACKEND or not equal or
                not context_matches(observed, candidate, package) or
                observed.get("source_tiling_identity") != item["source_tiling_observation"]["source_tiling_identity"]):
            measurement_failures.append(candidate_label(candidate) + ": " + (
                reason or ("unexpected backend {}".format(result.get("backend"))
                           if result.get("backend") != SOURCE_BACKEND else
                           "measurement output/reference or source identity mismatch")))
            continue
        compact = {key: value for key, value in item.items() if key != "package"}
        compact.update({"latency_result": result, "latency_tiling_observation": observed, "latency_worker_wall_ms": wall})
        measured.append(compact)
    # A source candidate that later fails execution is rejected individually;
    # it does not erase other legal candidates.  The shape is useful only
    # when it still has the requested twenty successful, measured tilings.
    if len(measured) < minimum:
        return "rejected", {"rejection_reason": "fewer than 20 source tilings completed real measurement", "source_discovery": discovery,
                              "successful_verified_distinct_tilings": len(verified), "successful_measured_distinct_tilings": len(measured),
                              "discovery_failures": failures, "verification_failures": verification_failures,
                              "measurement_failures": measurement_failures}
    return "admitted", {"valid_latency_count": len(measured), "valid_latency": measured, "source_discovery": discovery,
                         "successful_verified_distinct_tilings": len(verified),
                         "discovery_failures": failures, "verification_failures": verification_failures,
                         "measurement_failures": measurement_failures,
                         "rejected_candidate_count": len(failures) + len(verification_failures) + len(measurement_failures),
                         "source_rule": "complete native-source discovery and validation, then deterministic timing of 20 validated source contexts; the declared UB envelope is used only when core-budget contexts are below 20; failed candidates do not count; no flow-table field generation"}


def summarize(directory: Path) -> dict[str, Any]:
    _, per_op, rejected = read_progress(directory)
    return {"formal_valid_latency_records": sum(per_op.values()), "formal_valid_latency_records_per_op": per_op, "rejected_groups": rejected}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--log-dir", required=True, type=Path,
                        help="append-only numbered JSONL logs; each file is capped at 50 MiB")
    parser.add_argument("--device", required=True, type=int)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--operator", choices=("gather_elements",), required=True)
    parser.add_argument("--record-target", type=int, required=True)
    parser.add_argument("--source-package-manifest", action="append", default=[], type=Path)
    args = parser.parse_args()
    if not args.runner.is_file() or args.device < 0 or args.warmup < 0 or args.samples < 1:
        raise RuntimeError("invalid worker runner or measurement arguments")
    workloads, audit, catalog = load_catalog()
    minimum = int(catalog.MIN_SUCCESSFUL_TILINGS_PER_SHAPE)
    caps = tuple(int(value) for value in catalog.SOURCE_AIV_CAPS)
    budgets = {args.operator: args.record_target}
    if minimum != 20 or not caps or args.record_target < minimum or args.record_target % minimum:
        raise RuntimeError("GatherElements requires an exact target of whole 20-tiling groups")
    packages = plan_packages([validate_source_manifest(path) for path in args.source_package_manifest], args.operator)
    writer = RotatingJsonl(args.log_dir)
    completed, admitted_by_op, prior_rejected = read_progress(args.log_dir)
    if any(admitted_by_op.get(op, 0) > budgets[op] for op in budgets):
        raise RuntimeError("existing progress exceeds a per-operator formal record ceiling")
    base_env = dict(os.environ)
    base_env.pop("ASCEND_CUSTOM_OPP_PATH", None)
    for package_set in packages.values():
        for package in package_set:
            base_env.pop(str(package["instrumentation"]["audit_environment"]), None)
            base_env.pop(str(package["instrumentation"]["source_budget_environment"]), None)
            base_env.pop(str(package["instrumentation"]["dispatch_environment"]), None)
    for package_set in packages.values():
        for package in package_set:
            envelope = package["hardware_envelope_heuristic"]
            if envelope["enabled"]:
                base_env.pop(str(envelope["environment"]), None)
    runner_hash = digest_file(args.runner)
    planned = {args.operator: [row for row in workloads if source_supported_workload(row)]}
    if len(planned[args.operator]) < args.record_target // minimum:
        raise RuntimeError("GatherElements catalog does not contain enough reviewed shapes for the requested record target")
    begin = {
        "schema": SCHEMA, "matmul_included": False, "operator": args.operator,
        "formal_latency_ceiling": args.record_target, "formal_record_budget_per_op": budgets,
        "per_shape_minimum_successful_distinct_tilings": minimum,
        "candidate_collection": "complete original-source discovery and exact-output validation; deterministic 20-identity timing set per admitted shape; declared local hardware-rule capacity envelope only after original set is below 20",
        "operators": {op: {"semantic_workloads": len(planned[op]), "source_package_count": len(packages[op])} for op in budgets},
        "historical_latency_or_tiling_records_read": 0, "cce_data_or_cost_model_read": 0,
        "timing": {"warmup": args.warmup, "samples": args.samples, "kind": "device_event_only"},
        "no_host_timeout_or_forced_worker_kill": True, "temporary_reference_storage": "/tmp only",
        "preflight": "one installed viability launch per operator plus one source-audit call per host-tiler package; any failure stops before semantic-shape discovery",
        "log_directory": str(args.log_dir), "log_rotation_max_bytes": MAX_LOG_BYTES,
    }
    writer.append({**begin, "record_type": "campaign_begin"})
    print("SOURCE_TILING_CAMPAIGN_BEGIN " + json.dumps(begin, ensure_ascii=False, sort_keys=True), flush=True)
    preflight = source_audit_preflight(args, planned, packages, caps, base_env)
    for check in preflight:
        writer.append({"schema": SCHEMA, "record_type": "campaign_preflight", **check})
    preflight_failures = [check for check in preflight if check["status"] != "passed"]
    if preflight_failures:
        failure = {"schema": SCHEMA, "record_type": "campaign_preflight_failed", "status": "failed",
                   "failure_count": len(preflight_failures), "checks": preflight_failures,
                   "reason": "no semantic workload discovery was started because installed viability or source-audit loading failed"}
        writer.append(failure)
        print("SOURCE_TILING_CAMPAIGN_PREFLIGHT_FAILED " + json.dumps(failure, ensure_ascii=False, sort_keys=True), flush=True)
        return 2
    print("SOURCE_TILING_CAMPAIGN_PREFLIGHT_PASSED " + json.dumps({
        "checks": len(preflight), "source_tiler_audits": sum(check["kind"] == "source_tiler_audit" for check in preflight),
        "installed_reference_viability": sum(check["kind"] == "installed_reference_viability" for check in preflight),
    }, sort_keys=True), flush=True)
    op = args.operator
    for workload in planned[op]:
        if admitted_by_op.get(op, 0) >= budgets[op]:
            break
        group_key = stable_hash({"workload": workload, "runner_sha256": runner_hash,
                                 "package_manifests": [p["source_file_sha256"] for p in packages[op]],
                                 "source_aiv_caps": caps, "minimum": minimum, "schema": SCHEMA})
        if group_key in completed:
            continue
        remaining = budgets[op] - admitted_by_op.get(op, 0)
        if remaining < minimum:
            break
        with tempfile.TemporaryDirectory(prefix=op + "_source_tiling_") as temporary:
            status, details = execute_group(args, workload, packages[op], caps, minimum,
                                            base_env, group_key, Path(temporary))
        row = {"schema": SCHEMA, "group_key": group_key, "status": status, "workload": workload, **details}
        if status != "admitted":
            row.setdefault("valid_latency_count", 0)
        emit(writer, row)
        completed.add(group_key)
        if status == "admitted":
            admitted_by_op[op] = admitted_by_op.get(op, 0) + int(details["valid_latency_count"])
    end = {"schema": SCHEMA, "matmul_included": False,
          "operator": args.operator,
          "status": "complete" if admitted_by_op.get(args.operator, 0) == args.record_target else "completed_under_target",
          "summary": summarize(args.log_dir), "prior_rejected_groups": prior_rejected,
          "log_directory": str(args.log_dir), "log_rotation_max_bytes": MAX_LOG_BYTES}
    writer.append({**end, "record_type": "campaign_end"})
    print("SOURCE_TILING_CAMPAIGN_END " + json.dumps(end, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print("SOURCE_TILING_CAMPAIGN_FATAL " + json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)
