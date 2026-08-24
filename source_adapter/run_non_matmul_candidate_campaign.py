#!/usr/bin/env python3
"""Collect output-validated private GatherElementsV2 source compile contexts.

Every formal group evaluates the finite set of bounded inputs to the private
CANN package's original GatherElementsV2 host tiler. The controller never
enumerates, edits, or replays opaque raw-tiling fields. A group is admitted
only when twenty distinct source contexts launch, exactly match the installed
reference, and complete device-event measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CATALOG_PATH = ROOT / "non_matmul_candidate_catalog.py"
LOCK = json.loads((ROOT / "non_matmul_source_lock.json").read_text(encoding="utf-8"))
SCHEMA = "gather_elements_v2_cann81_prebuilt_measurement_v10"
SOURCE_BACKEND = "private_cann81_prebuilt_ascendc_aclnn_real_npu"
SOURCE_OPERATOR_TYPE = "GatherElementsV2"
# The campaign is intentionally append-only so an interrupted physical-NPU
# run can resume without losing its completed groups.  A single log is kept
# below this limit; the next numeric log is opened before an oversized write.
MAX_LOG_BYTES = 50 * 1024 * 1024
# This executable campaign has one deliberate scope.  The generic compiler
# must call the generated CANN 8.1 C++ ACLNN entry point and its precompiled
# device kernel, not the installed Python GatherElements implementation.
OPERATOR_RUNTIME_NAMES = {"GatherElementsV2": "gather_elements"}
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


def parse_worker_stage(output: str) -> dict[str, Any] | None:
    marker = "MULTIOP_NPU_STAGE "
    for line in reversed(output.splitlines()):
        if line.startswith(marker):
            try:
                return json.loads(line[len(marker):])
            except json.JSONDecodeError:
                return {"stage": "malformed_stage_record"}
    return None


def worker_termination(return_code: int, output: str) -> tuple[str, dict[str, Any] | None]:
    stage = parse_worker_stage(output)
    if return_code < 0:
        number = -return_code
        try:
            name = signal.Signals(number).name
        except ValueError:
            name = "SIGNAL"
        reason = "worker terminated by {}({})".format(name, number)
    else:
        reason = "worker exited with code {} without MULTIOP_NPU_RESULT".format(return_code)
    if stage and stage.get("stage"):
        reason += " after stage={}".format(stage["stage"])
        if stage.get("detail"):
            reason += " detail={}".format(stage["detail"])
    return reason, stage


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


def source_worker_args(runner: Path, workload: dict[str, Any], device: int, warmup: int, samples: int) -> list[str]:
    # This is a precompiled C++/Ascend-C package: there are no Python/TBE
    # compiler children to drain.  Use the same isolated-worker lifetime as
    # the working MatMul path; the process exits after flushing its result and
    # never resets the device or unloads CANN's process-local OpAPI state.
    return worker_args(runner, workload, device, warmup, samples)


def run_worker(arguments: list[str], environment: dict[str, str]) -> tuple[dict[str, Any], str, float, int]:
    # No subprocess timeout: a forced host kill can poison a real device task.
    started = time.monotonic()
    done = subprocess.run(arguments, text=True, capture_output=True, env=environment, check=False)
    output = done.stdout + done.stderr
    result = parse_worker_result(done.stdout)
    if result is None:
        reason, stage = worker_termination(done.returncode, output)
        result = {"status": "failed", "error": reason, "last_stage": stage}
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
        "distinct_source_compile_contexts": discovery.get("distinct_source_compile_contexts"),
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
        raise RuntimeError("missing private GatherElementsV2 package manifest: {}".format(path))
    item: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    required = ("schema", "operator", "source_operator_type", "runtime_op", "source_kind", "cann_root", "cann_version_file_sha256",
                "project_root", "package_root", "source_file", "source_file_sha256", "op_api_library", "op_api_library_sha256",
                "op_proto_library", "op_proto_library_sha256",
                "op_tiling_library", "op_tiling_library_sha256", "ops_config", "ops_config_sha256",
                "kernel_binary_root", "kernel_binary_info_config", "kernel_binary_info_config_sha256",
                "kernel_operator_config", "kernel_operator_config_sha256", "precompiled_device_kernels",
                "runtime_python_compilation", "build_cann_version", "instrumentation",
                "hardware_envelope_heuristic", "strategy_algorithm_changes", "kernel_algorithm_changes", "formal_data_gate")
    if (any(key not in item for key in required) or
            item["schema"] != "gather_elements_v2_cann81_prebuilt_package_v2" or
            item.get("source_operator_type") != SOURCE_OPERATOR_TYPE or
            item.get("operator") != SOURCE_OPERATOR_TYPE or
            item.get("runtime_op") != "gather_elements"):
        raise RuntimeError("invalid private GatherElementsV2 CANN package manifest: {}".format(path))
    source_file = Path(str(item["source_file"]))
    if not source_file.is_file() or digest_file(source_file) != str(item["source_file_sha256"]):
        raise RuntimeError("private GatherElementsV2 host tiler is missing or mismatched: {}".format(source_file))
    package_root = Path(str(item["package_root"]))
    project_root = Path(str(item["project_root"]))
    if not package_root.is_dir() or not project_root.is_dir():
        raise RuntimeError("private GatherElementsV2 package/project root is absent")
    for value_key, hash_key in (("op_api_library", "op_api_library_sha256"),
                                ("op_proto_library", "op_proto_library_sha256"),
                                ("op_tiling_library", "op_tiling_library_sha256"),
                                ("ops_config", "ops_config_sha256"),
                                ("kernel_binary_info_config", "kernel_binary_info_config_sha256"),
                                ("kernel_operator_config", "kernel_operator_config_sha256")):
        artifact = Path(str(item[value_key]))
        if not artifact.is_file() or digest_file(artifact) != str(item[hash_key]) or package_root not in artifact.parents:
            raise RuntimeError("private GatherElementsV2 package artifact is missing or mismatched: {}".format(artifact))
    kernel_root = Path(str(item["kernel_binary_root"]))
    kernels = item["precompiled_device_kernels"]
    if (not kernel_root.is_dir() or package_root not in kernel_root.parents or not isinstance(kernels, list) or
            len(kernels) != 4 or item["runtime_python_compilation"] is not False or
            item["build_cann_version"] != "8.1.RC1"):
        raise RuntimeError("private package is not a complete CANN 8.1 precompiled kernel package")
    for kernel in kernels:
        if not isinstance(kernel, dict):
            raise RuntimeError("invalid precompiled GatherElementsV2 kernel manifest entry")
        for value_key, hash_key in (("object", "object_sha256"), ("metadata", "metadata_sha256")):
            artifact = Path(str(kernel.get(value_key, "")))
            if (not artifact.is_file() or digest_file(artifact) != str(kernel.get(hash_key)) or
                    kernel_root not in artifact.parents):
                raise RuntimeError("precompiled GatherElementsV2 kernel is missing or mismatched: {}".format(artifact))
    runtime_op = OPERATOR_RUNTIME_NAMES.get(str(item["operator"]))
    if runtime_op is None:
        raise RuntimeError("private package is not an allowed GatherElements source: {}".format(item["operator"]))
    inst = item["instrumentation"]
    if (not isinstance(inst, dict) or inst.get("enabled") is not True or inst.get("mutates_tiling_context") is not False or
            not inst.get("audit_schema") or not inst.get("audit_environment") or not inst.get("source_budget_environment") or
            inst.get("dispatch_environment") != "GATHER_ELEMENTS_SOURCE_DISPATCH" or
            inst.get("dispatch_value") != "cann81_prebuilt_aclnn" or
            inst.get("opapi_library_environment") != "GATHER_ELEMENTS_SOURCE_OPAPI_LIBRARY"):
        raise RuntimeError("private package does not attest its original-source candidate axes")
    envelope = item["hardware_envelope_heuristic"]
    if (not isinstance(envelope, dict) or envelope.get("enabled") is not True or
            envelope.get("environment") != "GATHER_ELEMENTS_SOURCE_UB_DIVISOR" or
            not envelope.get("audit_field") or tuple(envelope.get("divisors", ())) != (2, 4, 8) or
            int(envelope.get("max_anchors", 0)) < 1):
        raise RuntimeError("private package hardware-envelope provenance is invalid")
    config = json.loads(Path(str(item["ops_config"])).read_text(encoding="utf-8"))
    entry = config.get(SOURCE_OPERATOR_TYPE)
    if not isinstance(entry, dict) or entry.get("opFile", {}).get("value") != "gather_elements_v2" or \
            entry.get("opInterface", {}).get("value") != "gather_elements_v2":
        raise RuntimeError("private package lacks GatherElementsV2 generic-dispatch config")
    version_file = Path(str(item["cann_root"])) / "opp" / "version.info"
    if not version_file.is_file() or digest_file(version_file) != str(item["cann_version_file_sha256"]):
        raise RuntimeError("private package was built against a different or missing CANN OPP installation")
    if item["strategy_algorithm_changes"] is not False or item["kernel_algorithm_changes"] is not False:
        raise RuntimeError("private GatherElementsV2 package is not source-preserving")
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


def source_compile_context_identity(observation: dict[str, Any]) -> str:
    fields = ("source_compile_context_sha256", "aiv_core_cap", "ub_cap_divisor", "compile_info_vars")
    if any(field not in observation for field in fields):
        raise RuntimeError("source audit omitted its native compile-context identity")
    return stable_hash({field: observation[field] for field in fields})


def successful_observation(path: Path, package: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    all_rows = read_observations(path, str(package["instrumentation"]["audit_schema"]))
    rows = [row for row in all_rows if row.get("event") == "tiling_generated"]
    if not rows:
        return None, "private GatherElementsV2 host tiler emitted no raw-tiling audit"
    identities = {source_compile_context_identity(row) for row in rows}
    if len(identities) != 1:
        return None, "one source execution emitted multiple compile-context identities"
    if any(int(row.get("status", -1)) != 0 for row in rows):
        return None, "original source tiler returned a non-success status"
    result = dict(rows[-1])
    result["source_compile_context_identity"] = next(iter(identities))
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
    if (observation is None or observation.get("event") != "tiling_generated" or
            observation.get("operator_type") != SOURCE_OPERATOR_TYPE or
            str(observation.get("aiv_core_cap")) != str(candidate["aiv_core_cap"])):
        return False
    envelope = package["hardware_envelope_heuristic"]
    if envelope["enabled"]:
        return str(observation.get(str(envelope["audit_field"]), "1")) == str(candidate["hardware_envelope_divisor"])
    return candidate["hardware_envelope_divisor"] == 1


def source_environment(base: dict[str, str], package: dict[str, Any], candidate: dict[str, Any], audit: Path) -> dict[str, str]:
    environment = dict(base)
    # This mapping is passed only to one subprocess.  It keeps the installed
    # OPP root and adds one checkout-local vendor package through CANN's
    # supported custom-OPP variable; it does not alter the login shell, CANN,
    # or another user's process/environment.
    package_root = Path(str(package["package_root"]))
    op_api = Path(str(package["op_api_library"]))
    if op_api.parent != package_root / "op_api/lib":
        raise RuntimeError("private GatherElementsV2 op-api library is outside its vendor package")
    environment["ASCEND_OPP_PATH"] = str(Path(str(package["cann_root"])) / "opp")
    # CANN's generated set_env.bash sets this to the individual vendor
    # package (not its packages/ parent) and prepends this exact op_api/lib
    # directory to the dynamic loader path.  Reproduce that contract only for
    # this worker process; never source or install anything into global CANN.
    environment["ASCEND_CUSTOM_OPP_PATH"] = str(package_root)
    previous_loader_path = environment.get("LD_LIBRARY_PATH", "")
    environment["LD_LIBRARY_PATH"] = str(op_api.parent) + (":" + previous_loader_path if previous_loader_path else "")
    environment[str(package["instrumentation"]["opapi_library_environment"])] = str(op_api)
    environment["GATHER_ELEMENTS_SOURCE_OPERATOR_TYPE"] = SOURCE_OPERATOR_TYPE
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
    all_rows = read_observations(path, str(package["instrumentation"]["audit_schema"]))
    rows = [row for row in all_rows if row.get("event") == "tiling_generated"]
    if not rows:
        return False, "private GatherElementsV2 host tiler emitted no raw-tiling audit"
    for row in rows:
        try:
            source_compile_context_identity(row)
        except RuntimeError:
            continue
        if context_matches(row, candidate, package):
            return True, ""
    return False, "source audit did not identify the requested source context"


def source_audit_preflight(args: Any, planned: dict[str, list[dict[str, Any]]],
                           packages: dict[str, list[dict[str, Any]]], caps: tuple[int, ...],
                           base_env: dict[str, str]) -> list[dict[str, Any]]:
    """Prove each boundary from the working ACLNN path to the custom kernel.

    Each boundary runs in its own process, so a host-library or device crash
    cannot erase the last known-good boundary.  The gates are deliberately
    ordered: installed reference launch, private OpAPI load/symbol lookup,
    private C++ GetWorkspace/host tiler, then precompiled kernel launch plus
    exact output comparison.
    """
    checks: list[dict[str, Any]] = []
    temporary_root = args.log_dir.parent / "tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="source_tiling_preflight_", dir=str(temporary_root)) as temporary:
        root = Path(temporary)
        for op in sorted(planned):
            workload = planned[op][0]
            reference = root / (op + "_installed_reference.bin")
            result, output, wall, rc = run_worker(
                worker_args(args.runner, workload, args.device, 0, 0) + ["--write-reference", str(reference)], base_env)
            reference_ok = (rc == 0 and result.get("status") == "success" and reference.is_file() and
                            int(result.get("output_bytes") or 0) > 0)
            checks.append({"operator": op, "workload_id": workload["workload_id"], "kind": "installed_reference_viability",
                           "status": "passed" if reference_ok else "failed", "worker_wall_ms": wall,
                           "worker_return_code": rc, "worker_status": result.get("status"),
                           "last_stage": parse_worker_stage(output),
                           "runner_error": None if reference_ok else result.get("error"),
                           "failure": None if reference_ok else runner_failure(result, output)})
            if not reference_ok:
                continue
            for package in packages[op]:
                candidate = candidate_descriptor(package, caps[-1])

                load_audit = root / (op + "_load_unused.jsonl")
                result, output, wall, rc = run_worker(
                    source_worker_args(args.runner, workload, args.device, 0, 0) + ["--source-load-only", "1"],
                    source_environment(base_env, package, candidate, load_audit))
                load_ok = (rc == 0 and result.get("status") == "success" and result.get("backend") == SOURCE_BACKEND)
                checks.append({"operator": op, "workload_id": workload["workload_id"], "kind": "private_opapi_load",
                               "strategy": candidate["id"], "aiv_core_cap": candidate["aiv_core_cap"],
                               "status": "passed" if load_ok else "failed", "worker_return_code": rc,
                               "worker_status": result.get("status"), "worker_wall_ms": wall,
                               "runner_backend": result.get("backend"), "last_stage": parse_worker_stage(output),
                               "runner_error": None if load_ok else result.get("error"),
                               "failure": None if load_ok else runner_failure(result, output)})
                if not load_ok:
                    continue

                tiling_audit = root / (op + "_host_tiling_" + stable_hash(candidate) + ".jsonl")
                result, output, wall, rc = run_worker(
                    source_worker_args(args.runner, workload, args.device, 0, 0) + ["--source-host-tiling-only", "1"],
                    source_environment(base_env, package, candidate, tiling_audit))
                observed, reason = source_audit_emitted(tiling_audit, package, candidate)
                tiling_ok = (observed and rc == 0 and result.get("status") == "success" and
                             result.get("backend") == SOURCE_BACKEND)
                checks.append({"operator": op, "workload_id": workload["workload_id"], "kind": "private_host_tiling",
                               "strategy": candidate["id"], "aiv_core_cap": candidate["aiv_core_cap"],
                               "status": "passed" if tiling_ok else "failed", "worker_return_code": rc,
                               "worker_status": result.get("status"), "worker_wall_ms": wall,
                               "runner_backend": result.get("backend"), "last_stage": parse_worker_stage(output),
                               "runner_error": None if tiling_ok else result.get("error"),
                               "failure": None if tiling_ok else (
                                   runner_failure(result, output) if (rc != 0 or result.get("status") != "success") else reason)})
                if not tiling_ok:
                    continue

                launch_audit = root / (op + "_kernel_launch_" + stable_hash(candidate) + ".jsonl")
                result, output, wall, rc = run_worker(
                    source_worker_args(args.runner, workload, args.device, 0, 0) +
                    ["--source-tiling-only", "1", "--compare-reference", str(reference)],
                    source_environment(base_env, package, candidate, launch_audit))
                launch_observed, launch_reason = source_audit_emitted(launch_audit, package, candidate)
                launch_ok = (launch_observed and rc == 0 and result.get("status") == "success" and
                             result.get("backend") == SOURCE_BACKEND and
                             result.get("output_reference_checked") is True and
                             result.get("output_reference_equal") is True)
                if not launch_ok and rc == 0 and result.get("status") == "success":
                    if not launch_observed:
                        launch_failure = launch_reason
                    elif result.get("backend") != SOURCE_BACKEND:
                        launch_failure = "runner used unexpected backend: {}".format(result.get("backend"))
                    else:
                        launch_failure = "private precompiled kernel output did not exactly match installed aclnnGather"
                else:
                    launch_failure = None if launch_ok else runner_failure(result, output)
                checks.append({"operator": op, "workload_id": workload["workload_id"],
                               "kind": "private_precompiled_kernel_launch",
                               "strategy": candidate["id"], "aiv_core_cap": candidate["aiv_core_cap"],
                               "status": "passed" if launch_ok else "failed", "worker_return_code": rc,
                               "worker_status": result.get("status"), "worker_wall_ms": wall,
                               "runner_backend": result.get("backend"), "last_stage": parse_worker_stage(output),
                               "output_reference_checked": result.get("output_reference_checked"),
                               "output_reference_equal": result.get("output_reference_equal"),
                               "runner_error": None if launch_ok else result.get("error"),
                               "failure": launch_failure})
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
    """Filter before NPU work using the private GatherElementsV2 declaration.

    Its source declaration accepts x in fp16/bf16/fp32/int32 and index in
    int32.  Sending CANN-8.1's installed-only int64 index variants through
    the private 910B package would manufacture predictable rejects and cannot
    contribute to the 5,000 real source-kernel measurements.  At least twenty
    output-parallel positions are also required: otherwise all caps above the
    available parallelism collapse to the same raw tiling and cannot satisfy
    the requested 20-way ranking set.
    """
    if not (workload.get("op") == "gather_elements" and
            workload.get("dtype") in ("fp16", "bf16", "fp32", "int32") and
            workload.get("index_dtype") == "int32"):
        return False
    shape = workload.get("index_shape")
    axis = workload.get("axis")
    if not isinstance(shape, list) or not shape or not isinstance(axis, int):
        return False
    normalized_axis = axis % len(shape)
    output_parallelism = 1
    for index, value in enumerate(shape):
        if not isinstance(value, int) or value <= 0:
            return False
        if index != normalized_axis:
            output_parallelism *= value
    return output_parallelism >= 20


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
                "successful_source_contexts": successful_contexts, "distinct_source_compile_contexts": len(discovered),
                "heuristic_anchor_count": len(anchors), "source_audit_missing": source_audit_missing}

    def attempt(package: dict[str, Any], candidate: dict[str, Any], kind: str) -> bool:
        nonlocal successful_contexts, original_contexts, heuristic_contexts
        if kind == "original": original_contexts += 1
        else: heuristic_contexts += 1
        audit = temp / ("discover_" + stable_hash({"g": group_key, "c": candidate}) + ".jsonl")
        result, output, wall, rc = run_worker(source_worker_args(args.runner, workload, args.device, 0, 0) + ["--source-tiling-only", "1"],
                                              source_environment(base_env, package, candidate, audit))
        observed, reason = successful_observation(audit, package)
        if result.get("backend") != SOURCE_BACKEND:
            failures.append(candidate_label(candidate) + ": runner used unexpected backend {}".format(result.get("backend")))
            return True
        if reason == "private GatherElementsV2 host tiler emitted no raw-tiling audit":
            failures.append(candidate_label(candidate) + ": " + reason)
            # This is a deployment/instrumentation failure, not an invalid
            # tiling. Retrying its remaining caps would only create duplicate
            # rejection records and cannot produce a formal candidate.
            return True
        if rc != 0 or result.get("status") != "success" or not context_matches(observed, candidate, package):
            failures.append(candidate_label(candidate) + ": " + (reason or compact_failure(output)))
            return False
        successful_contexts += 1
        identity = str(observed["source_compile_context_identity"])
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
        return "rejected", {"rejection_reason": "source compile-context audit was absent; this is a deployment failure, not a candidate rejection",
                              "source_discovery": discovery, "discovery_failures": failures}
    if len(discovered) < minimum:
        return "rejected", {"rejection_reason": "fewer than 20 distinct native source compile contexts from the complete source search", "source_discovery": discovery, "discovery_failures": failures}
    verified: list[dict[str, Any]] = []
    verification_failures: list[str] = []
    for identity, item in discovered.items():
        candidate, package = item["candidate"], item["package"]
        audit = temp / ("verify_" + stable_hash({"g": group_key, "c": candidate}) + ".jsonl")
        result, output, wall, rc = run_worker(source_worker_args(args.runner, workload, args.device, 0, 0) + ["--compare-reference", str(reference_path)],
                                              source_environment(base_env, package, candidate, audit))
        observed, reason = successful_observation(audit, package)
        equal = bool(result.get("output_reference_checked")) and bool(result.get("output_reference_equal"))
        if (rc != 0 or result.get("status") != "success" or result.get("backend") != SOURCE_BACKEND or not equal or
                not context_matches(observed, candidate, package) or observed.get("source_compile_context_identity") != identity):
            verification_failures.append(candidate_label(candidate) + ": " + (
                reason or ("unexpected backend {}".format(result.get("backend"))
                           if result.get("backend") != SOURCE_BACKEND else "output/reference or source identity mismatch")))
            continue
        verified.append({**item, "verification_result": result, "verification_wall_ms": wall, "verification_tiling_observation": observed})
    if len(verified) < minimum:
        return "rejected", {"rejection_reason": "fewer than 20 distinct source compile contexts passed exact output validation", "source_discovery": discovery,
                              "successful_verified_distinct_tilings": len(verified), "discovery_failures": failures,
                              "verification_failures": verification_failures}
    measured: list[dict[str, Any]] = []
    measurement_failures: list[str] = []
    # Discovery and reference validation above intentionally process the
    # complete raw source set.  Timing exactly twenty deterministic identities
    # per admitted shape keeps the 5,000-record contract exact while retaining
    # a non-random ranking set. If a timed candidate fails, the next validated
    # identity replaces it; failures never count.
    verified.sort(key=lambda item: str(item["source_tiling_observation"]["source_compile_context_identity"]))
    for item in verified:
        if len(measured) == minimum:
            break
        candidate, package = item["candidate"], item["package"]
        audit = temp / ("measure_" + stable_hash({"g": group_key, "c": candidate}) + ".jsonl")
        result, output, wall, rc = run_worker(source_worker_args(args.runner, workload, args.device, args.warmup, args.samples) + ["--compare-reference", str(reference_path)],
                                              source_environment(base_env, package, candidate, audit))
        observed, reason = successful_observation(audit, package)
        equal = bool(result.get("output_reference_checked")) and bool(result.get("output_reference_equal"))
        if (rc != 0 or result.get("status") != "success" or result.get("backend") != SOURCE_BACKEND or not equal or
                not context_matches(observed, candidate, package) or
                observed.get("source_compile_context_identity") != item["source_tiling_observation"]["source_compile_context_identity"]):
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
        return "rejected", {"rejection_reason": "fewer than 20 source compile contexts completed real measurement", "source_discovery": discovery,
                              "successful_verified_distinct_tilings": len(verified), "successful_measured_distinct_tilings": len(measured),
                              "discovery_failures": failures, "verification_failures": verification_failures,
                              "measurement_failures": measurement_failures}
    return "admitted", {"valid_latency_count": len(measured), "valid_latency": measured, "source_discovery": discovery,
                         "successful_verified_distinct_tilings": len(verified),
                         "discovery_failures": failures, "verification_failures": verification_failures,
                         "measurement_failures": measurement_failures,
                         "rejected_candidate_count": len(failures) + len(verification_failures) + len(measurement_failures),
                         "source_rule": "complete private GatherElementsV2 source discovery and validation, then deterministic timing of 20 validated source contexts; the declared UB envelope is used only when core-budget contexts are below 20; failed candidates do not count; no flow-table field generation"}


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
    temporary_root = args.log_dir.parent / "tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    completed, admitted_by_op, prior_rejected = read_progress(args.log_dir)
    if any(admitted_by_op.get(op, 0) > budgets[op] for op in budgets):
        raise RuntimeError("existing progress exceeds a per-operator formal record ceiling")
    base_env = dict(os.environ)
    base_env.pop("ASCEND_CUSTOM_OPP_PATH", None)
    base_env.pop("GATHER_ELEMENTS_SOURCE_OPERATOR_TYPE", None)
    base_env.pop("GATHER_ELEMENTS_SOURCE_OPAPI_LIBRARY", None)
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
        "no_host_timeout_or_forced_worker_kill": True, "temporary_reference_storage": str(temporary_root),
        "preflight": "ordered isolated gates: installed ACLNN reference, private C++ OpAPI load, private GetWorkspace/host tiler, precompiled kernel launch, exact output equality",
        "log_directory": str(args.log_dir), "log_rotation_max_bytes": MAX_LOG_BYTES,
    }
    writer.append({**begin, "record_type": "campaign_begin"})
    print("SOURCE_TILING_CAMPAIGN_BEGIN " + json.dumps({
        "operator": args.operator,
        "target_records": args.record_target,
        "minimum_contexts_per_shape": minimum,
        "reviewed_shapes": len(planned[args.operator]),
        "logs": str(args.log_dir),
    }, ensure_ascii=False, sort_keys=True), flush=True)
    preflight = source_audit_preflight(args, planned, packages, caps, base_env)
    for check in preflight:
        writer.append({"schema": SCHEMA, "record_type": "campaign_preflight", **check})
    preflight_failures = [check for check in preflight if check["status"] != "passed"]
    if preflight_failures:
        failure = {"schema": SCHEMA, "record_type": "campaign_preflight_failed", "status": "failed",
                   "failure_count": len(preflight_failures), "checks": preflight_failures,
                   "reason": "no semantic workload discovery was started because installed viability or source-audit loading failed"}
        writer.append(failure)
        first = preflight_failures[0]
        print("SOURCE_TILING_CAMPAIGN_PREFLIGHT_FAILED " + json.dumps({
            "kind": first.get("kind"), "worker_return_code": first.get("worker_return_code"),
            "runner_backend": first.get("runner_backend"), "failure": first.get("failure"),
            "last_stage": first.get("last_stage"),
            "logs": str(args.log_dir)
        }, ensure_ascii=False, sort_keys=True), flush=True)
        return 2
    print("SOURCE_TILING_CAMPAIGN_PREFLIGHT_PASSED " + json.dumps({
        "checks": [check["kind"] for check in preflight],
        "installed_reference_viability": sum(check["kind"] == "installed_reference_viability" for check in preflight),
        "private_opapi_load": sum(check["kind"] == "private_opapi_load" for check in preflight),
        "private_host_tiling": sum(check["kind"] == "private_host_tiling" for check in preflight),
        "private_precompiled_kernel_launch": sum(
            check["kind"] == "private_precompiled_kernel_launch" for check in preflight),
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
        with tempfile.TemporaryDirectory(prefix=op + "_source_tiling_", dir=str(temporary_root)) as temporary:
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
    print("SOURCE_TILING_CAMPAIGN_END " + json.dumps({
        "operator": args.operator, "status": end["status"],
        "formal_valid_latency_records": end["summary"]["formal_valid_latency_records"],
        "rejected_groups": end["summary"]["rejected_groups"], "logs": str(args.log_dir)
    }, ensure_ascii=False, sort_keys=True), flush=True)
    # A successful process must mean that the requested formal dataset exists,
    # not merely that the finite catalog has been exhausted.  This prevents an
    # under-target set of rejected or incomplete shapes from looking complete.
    return 0 if end["status"] == "complete" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print("SOURCE_TILING_CAMPAIGN_FATAL " + json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)
