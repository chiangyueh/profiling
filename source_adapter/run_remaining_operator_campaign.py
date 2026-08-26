#!/usr/bin/env python3
"""Collect 5,000 validated native CANN-8.1 tiling latencies for one operator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import remaining_operator_candidate_catalog as catalog_module
import run_non_matmul_candidate_campaign as campaign


LOCK = json.loads((ROOT / "remaining_operators_cann81_lock.json").read_text(encoding="utf-8"))
OPERATOR_TYPES = {
    "flash_attention_score_grad": "FlashAttentionScoreGrad",
    "fused_infer_attention_score": "FusedInferAttentionScore",
}
BACKENDS = {
    "flash_attention_score_grad": "private_cann81_fasg_prebuilt_aclnn_real_npu",
    "fused_infer_attention_score": "private_cann81_fias_prebuilt_aclnn_real_npu",
}


def digest_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def worker_args(runner: Path, workload: dict[str, Any], device: int, warmup: int, samples: int) -> list[str]:
    values = [str(runner), "--workload-id", workload["workload_id"], "--op", workload["op"],
              "--device", str(device), "--warmup", str(warmup), "--samples", str(samples),
              "--expected-soc", "Ascend910B3"]
    fields = ("batch", "q_heads", "kv_heads", "q_seq", "kv_seq", "head_dim", "dtype", "layout")
    for field in fields:
        item = workload[field]
        rendered = ",".join(map(str, item)) if isinstance(item, list) else str(item)
        values += ["--" + field.replace("_", "-"), rendered]
    return values


def validate_manifest(path: Path, selected: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError("private package manifest is absent: {}".format(path))
    item = json.loads(path.read_text(encoding="utf-8"))
    if (item.get("schema") != "remaining_operator_cann81_prebuilt_package_v2" or
            item.get("device_kernel_origin") != "installed_cann81_binary_package_private_copy"):
        raise RuntimeError("invalid complete CANN-8.1 attention package manifest: {}".format(path))
    if (item.get("runtime_op") != selected or item.get("operator") != OPERATOR_TYPES[selected] or
            item.get("build_cann_version") != "8.1.RC1" or
            item.get("runtime_python_compilation") is not False or
            item.get("strategy_algorithm_changes") is not False or
            item.get("kernel_algorithm_changes") is not False):
        raise RuntimeError("invalid native CANN-8.1 package manifest: {}".format(path))
    package_root = Path(item["package_root"])
    if not package_root.is_dir():
        raise RuntimeError("private package root is absent")
    artifacts = [("source_file", "source_file_sha256"),
                 ("op_tiling_library", "op_tiling_library_sha256")]
    artifacts += [("op_api_library", "op_api_library_sha256"),
                  ("op_proto_library", "op_proto_library_sha256"),
                  ("ops_config", "ops_config_sha256"),
                  ("installed_kernel_copy_manifest", "installed_kernel_copy_manifest_sha256")]
    for value_key, hash_key in artifacts:
        artifact = Path(item[value_key])
        if not artifact.is_file() or digest_file(artifact) != item[hash_key]:
            raise RuntimeError("private package artifact is missing or mismatched: {}".format(artifact))
    kernels = item.get("precompiled_device_kernels")
    if not isinstance(kernels, list) or not kernels:
        raise RuntimeError("private package has no precompiled Ascend910B kernels")
    for kernel in kernels:
        for value_key, hash_key in (("object", "object_sha256"), ("metadata", "metadata_sha256")):
            artifact = Path(kernel[value_key])
            if not artifact.is_file() or digest_file(artifact) != kernel[hash_key]:
                raise RuntimeError("precompiled kernel artifact is missing or mismatched")
    version = Path(item["cann_root"]) / "opp/version.info"
    if not version.is_file() or digest_file(version) != item["cann_version_file_sha256"]:
        raise RuntimeError("package CANN root has changed")
    instrumentation = item.get("instrumentation")
    envelope = item.get("hardware_envelope_heuristic")
    if (not isinstance(instrumentation, dict) or instrumentation.get("enabled") is not True or
            instrumentation.get("mutates_tiling_context") is not False or
            not isinstance(envelope, dict) or tuple(envelope.get("divisors", ())) != (2, 4, 8)):
        raise RuntimeError("package source-candidate provenance is incomplete")
    item["manifest_path"] = str(path)
    return item


def normalize_observation(row: dict[str, Any], package: dict[str, Any]) -> dict[str, Any]:
    value = dict(row)
    value.setdefault("event", "tiling_generated")
    value.setdefault("operator_type", package["operator"])
    value.setdefault("source_compile_context_sha256", package["official_tiling_source_sha256"])
    if "compile_info_vars" not in value:
        value["compile_info_vars"] = {
            "tiling_key": value.get("tiling_key"), "block_dim": value.get("block_dim"),
            "raw_tiling_bytes": value.get("raw_bytes"), "raw_tiling_fnv1a64": str(value.get("raw_fnv1a64")),
        }
    compile_info = value["compile_info_vars"]
    value.setdefault("tiling", {
        "raw_tiling_bytes": compile_info.get("raw_tiling_bytes", value.get("raw_bytes")),
        "raw_tiling_fnv1a64": str(compile_info.get("raw_tiling_fnv1a64", value.get("raw_fnv1a64"))),
    })
    return value


def observations(path: Path, package: dict[str, Any]) -> list[dict[str, Any]]:
    return [normalize_observation(row, package) for row in
            campaign.read_observations(path, package["instrumentation"]["audit_schema"])]


def successful_observation(path: Path, package: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    rows = [row for row in observations(path, package) if row.get("event") == "tiling_generated"]
    if not rows:
        return None, "private CANN-8.1 host tiler emitted no raw-tiling audit"
    identities = {campaign.source_compile_context_identity(row) for row in rows}
    if len(identities) != 1:
        return None, "one source execution emitted multiple compile-context identities"
    if any(int(row.get("status", -1)) != 0 for row in rows):
        return None, "original source tiler returned a non-success status"
    result = dict(rows[-1])
    result["source_compile_context_identity"] = next(iter(identities))
    result["observation_count"] = len(rows)
    return result, None


def context_matches(observation: dict[str, Any] | None, candidate: dict[str, Any], package: dict[str, Any]) -> bool:
    if (observation is None or observation.get("operator_type") != package["operator"] or
            observation.get("source_compile_context_sha256") != package["official_tiling_source_sha256"] or
            str(observation.get("aiv_core_cap")) != str(candidate["aiv_core_cap"])):
        return False
    envelope = package["hardware_envelope_heuristic"]
    return str(observation.get(envelope["audit_field"], "1")) == str(candidate["hardware_envelope_divisor"])


def source_environment(base: dict[str, str], package: dict[str, Any], candidate: dict[str, Any], audit: Path) -> dict[str, str]:
    environment = dict(base)
    environment["ASCEND_OPP_PATH"] = str(Path(package["cann_root"]) / "opp")
    inst, envelope = package["instrumentation"], package["hardware_envelope_heuristic"]
    package_root, library = Path(package["package_root"]), Path(package["op_api_library"])
    if library.parent != package_root / "op_api/lib":
        raise RuntimeError("private attention OpAPI is outside its package")
    environment["ASCEND_CUSTOM_OPP_PATH"] = str(package_root)
    library_environment = inst["opapi_library_environment"]
    old_loader = environment.get("LD_LIBRARY_PATH", "")
    environment["LD_LIBRARY_PATH"] = str(library.parent) + (":" + old_loader if old_loader else "")
    environment[library_environment] = str(library)
    environment[inst["audit_environment"]] = str(audit)
    environment[inst["dispatch_environment"]] = inst["dispatch_value"]
    environment[inst["source_budget_environment"]] = str(candidate["aiv_core_cap"])
    environment[envelope["environment"]] = str(candidate["hardware_envelope_divisor"])
    return environment


def source_audit_emitted(path: Path, package: dict[str, Any], candidate: dict[str, Any]) -> tuple[bool, str]:
    rows = observations(path, package)
    for row in rows:
        try:
            campaign.source_compile_context_identity(row)
        except RuntimeError:
            continue
        if context_matches(row, candidate, package):
            return True, ""
    return False, "private host tiler did not audit the requested source context"


def plan_packages(manifests: list[dict[str, Any]], selected: str) -> dict[str, list[dict[str, Any]]]:
    if not manifests or any(item["runtime_op"] != selected for item in manifests):
        raise RuntimeError("package/operator mismatch")
    if selected == "flash_attention_score_grad":
        strategies = {item.get("strategy_class") for item in manifests}
        if len(manifests) != 8 or None in strategies or len(strategies) != 8:
            raise RuntimeError("FASG requires all eight isolated original strategy packages")
    elif len(manifests) != 1:
        raise RuntimeError("{} requires exactly one original dispatcher package".format(selected))
    return {selected: manifests}


def supported(workload: dict[str, Any]) -> bool:
    return (workload["dtype"] in ("fp16", "bf16") and workload["head_dim"] % 16 == 0 and
            workload["q_heads"] % workload["kv_heads"] == 0)


def storage(workload: dict[str, Any]) -> int:
    return workload["batch"] * (workload["q_heads"] * workload["q_seq"] +
                                2 * workload["kv_heads"] * workload["kv_seq"]) * workload["head_dim"]


def remaining_preflight(args: Any, planned: dict[str, list[dict[str, Any]]],
                        packages: dict[str, list[dict[str, Any]]], caps: tuple[int, ...],
                        base_env: dict[str, str]) -> list[dict[str, Any]]:
    """Gate every private source route, then prove one eligible real kernel route.

    Some official FASG templates are intentionally shape- or deterministic-
    only. Requiring every isolated registration to accept one common shape
    would rewrite the official predicate contract. All eight libraries must
    load and construct an executor; the always-capable original base template
    then proves host audit, precompiled launch and exact output equality.
    """
    checks: list[dict[str, Any]] = []
    op = next(iter(planned))
    workload = planned[op][0]
    temporary_root = args.log_dir.parent / "tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="remaining_preflight_", dir=str(temporary_root)) as temporary:
        root = Path(temporary)
        reference = root / "installed_reference.bin"
        result, output, wall, rc = campaign.run_worker(
            worker_args(args.runner, workload, args.device, 0, 0) + ["--write-reference", str(reference)], base_env)
        ok = rc == 0 and result.get("status") == "success" and reference.is_file() and int(result.get("output_bytes") or 0) > 0
        checks.append({"operator": op, "workload_id": workload["workload_id"], "kind": "installed_reference_viability",
                       "status": "passed" if ok else "failed", "worker_wall_ms": wall, "worker_return_code": rc,
                       "worker_status": result.get("status"), "last_stage": campaign.parse_worker_stage(output),
                       "failure": None if ok else campaign.runner_failure(result, output)})
        if not ok:
            return checks
        for package in packages[op]:
            candidate = campaign.candidate_descriptor(package, caps[-1])
            audit = root / ("load_" + campaign.stable_hash(candidate) + ".jsonl")
            result, output, wall, rc = campaign.run_worker(
                worker_args(args.runner, workload, args.device, 0, 0) + ["--source-load-only", "1"],
                source_environment(base_env, package, candidate, audit))
            ok = rc == 0 and result.get("status") == "success" and result.get("backend") == BACKENDS[op]
            load_kind = "private_opapi_load"
            checks.append({"operator": op, "workload_id": workload["workload_id"], "kind": load_kind,
                           "strategy": candidate["id"], "status": "passed" if ok else "failed",
                           "worker_wall_ms": wall, "worker_return_code": rc, "worker_status": result.get("status"),
                           "runner_backend": result.get("backend"), "last_stage": campaign.parse_worker_stage(output),
                           "failure": None if ok else campaign.runner_failure(result, output)})
            if not ok:
                continue
            if (op == "flash_attention_score_grad" and package.get("strategy_class") !=
                    "FlashAttentionScoreGradTilingS1s2Bn2gs1s2"):
                # The other seven official registrations have shape,
                # deterministic, or optional-input predicates. Their legality
                # is evaluated during discovery, not falsified in preflight by
                # forcing one common semantic shape.
                continue
            planning = root / ("planning_" + campaign.stable_hash(candidate) + ".jsonl")
            result, output, wall, rc = campaign.run_worker(
                worker_args(args.runner, workload, args.device, 0, 0) + ["--source-executor-planning-only", "1"],
                source_environment(base_env, package, candidate, planning))
            ok = rc == 0 and result.get("status") == "success" and result.get("backend") == BACKENDS[op]
            checks.append({"operator": op, "workload_id": workload["workload_id"], "kind": "private_executor_planning",
                           "strategy": candidate["id"], "status": "passed" if ok else "failed",
                           "worker_wall_ms": wall, "worker_return_code": rc, "worker_status": result.get("status"),
                           "runner_backend": result.get("backend"), "last_stage": campaign.parse_worker_stage(output),
                           "failure": None if ok else campaign.runner_failure(result, output)})
        if any(item["status"] != "passed" for item in checks):
            return checks
        launch_packages = packages[op]
        if op == "flash_attention_score_grad":
            preferred = [item for item in launch_packages if item.get("strategy_class") ==
                         "FlashAttentionScoreGradTilingS1s2Bn2gs1s2"]
            launch_packages = preferred or launch_packages
        launch_ok = False
        launch_failure = "no eligible private package launched"
        for package in launch_packages:
            candidate = campaign.candidate_descriptor(package, caps[-1])
            audit = root / ("launch_" + campaign.stable_hash(candidate) + ".jsonl")
            result, output, wall, rc = campaign.run_worker(
                worker_args(args.runner, workload, args.device, 0, 0) +
                ["--source-tiling-only", "1", "--compare-reference", str(reference)],
                source_environment(base_env, package, candidate, audit))
            observed, reason = source_audit_emitted(audit, package, candidate)
            launch_ok = (observed and rc == 0 and result.get("status") == "success" and
                         result.get("backend") == BACKENDS[op] and
                         result.get("output_reference_checked") is True and result.get("output_reference_equal") is True)
            launch_failure = None if launch_ok else (reason or campaign.runner_failure(result, output))
            launch_kind = "private_precompiled_kernel_launch"
            checks.append({"operator": op, "workload_id": workload["workload_id"],
                           "kind": launch_kind, "strategy": candidate["id"],
                           "status": "passed" if launch_ok else "failed", "worker_wall_ms": wall,
                           "worker_return_code": rc, "worker_status": result.get("status"),
                           "runner_backend": result.get("backend"), "last_stage": campaign.parse_worker_stage(output),
                           "output_reference_checked": result.get("output_reference_checked"),
                           "output_reference_equal": result.get("output_reference_equal"), "failure": launch_failure})
            if launch_ok:
                break
    return checks


def configure(selected: str) -> None:
    campaign.SCHEMA = "{}_cann81_native_measurement_v1".format(selected)
    campaign.SOURCE_BACKEND = BACKENDS[selected]
    campaign.SOURCE_OPERATOR_TYPE = OPERATOR_TYPES[selected]
    campaign.SOURCE_RULE = ("complete official CANN 8.1 {} source discovery and exact-output validation; "
                            "deterministic timing of 20 validated source contexts; the declared hardware "
                            "capacity envelope is used only when original core-budget contexts are below 20; "
                            "failed candidates do not count; no raw tiling field generation").format(
                                OPERATOR_TYPES[selected])
    campaign.OPERATOR_RUNTIME_NAMES = {OPERATOR_TYPES[selected]: selected}
    campaign.RUNTIME_OPERATOR_NAMES = {selected: OPERATOR_TYPES[selected]}
    campaign.worker_args = worker_args
    campaign.source_worker_args = lambda runner, workload, device, warmup, samples: worker_args(
        runner, workload, device, warmup, samples)
    campaign.successful_observation = successful_observation
    campaign.context_matches = context_matches
    campaign.source_environment = source_environment
    campaign.source_audit_emitted = source_audit_emitted
    campaign.source_audit_preflight = remaining_preflight
    campaign.plan_packages = plan_packages
    campaign.source_supported_workload = supported
    campaign.workload_storage_elements = storage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--device", required=True, type=int)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--operator", required=True, choices=tuple(OPERATOR_TYPES))
    parser.add_argument("--record-target", type=int, default=5000)
    parser.add_argument("--source-package-manifest", action="append", default=[], type=Path)
    args = parser.parse_args()
    if not args.runner.is_file() or args.device < 0 or args.warmup < 0 or args.samples < 1:
        raise RuntimeError("invalid runner or measurement arguments")
    if args.record_target != 5000:
        raise RuntimeError("remaining-operator campaigns have an exact 5,000-record target")
    configure(args.operator)
    workloads = catalog_module.catalog(args.operator)
    minimum, caps = 20, tuple(range(1, 21))
    packages = plan_packages([validate_manifest(path, args.operator) for path in args.source_package_manifest], args.operator)
    writer = campaign.RotatingJsonl(args.log_dir)
    temporary_root = args.log_dir.parent / "tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    completed, admitted_by_op, prior_rejected = campaign.read_progress(args.log_dir)
    base_env = dict(os.environ)
    base_env.pop("ASCEND_CUSTOM_OPP_PATH", None)
    for package in packages[args.operator]:
        inst, envelope = package["instrumentation"], package["hardware_envelope_heuristic"]
        library_environment = inst.get("opapi_library_environment", inst.get("tiling_library_environment"))
        if not isinstance(library_environment, str):
            raise RuntimeError("source library environment is absent")
        environment_keys = [library_environment, inst["audit_environment"],
                            inst["source_budget_environment"], inst["dispatch_environment"],
                            envelope["environment"]]
        for key in environment_keys:
            base_env.pop(key, None)
    planned = {args.operator: sorted((row for row in workloads if supported(row)),
                                     key=lambda row: (storage(row), row["workload_id"]))}
    if len(planned[args.operator]) < 250:
        raise RuntimeError("catalog has fewer than 250 reviewed shapes")
    begin = {"schema": campaign.SCHEMA, "record_type": "campaign_begin", "operator": args.operator,
             "formal_latency_ceiling": 5000, "per_shape_minimum_successful_distinct_tilings": 20,
             "reviewed_shapes": len(planned[args.operator]), "source_package_count": len(packages[args.operator]),
             "historical_latency_or_tiling_records_read": 0, "cce_data_or_cost_model_read": 0,
             "runtime_python_compilation": False, "matmul_included": False,
             "scatter_elements_included": False,
             "timing": {"warmup": args.warmup, "samples": args.samples, "kind": "device_event_only"},
             "log_directory": str(args.log_dir), "log_rotation_max_bytes": campaign.MAX_LOG_BYTES}
    writer.append(begin)
    print("SOURCE_TILING_CAMPAIGN_BEGIN " + json.dumps({"operator": args.operator, "target_records": 5000,
          "minimum_contexts_per_shape": 20, "reviewed_shapes": len(planned[args.operator]),
          "logs": str(args.log_dir)}, sort_keys=True), flush=True)
    preflight = campaign.source_audit_preflight(args, planned, packages, caps, base_env)
    for check in preflight:
        writer.append({"schema": campaign.SCHEMA, "record_type": "campaign_preflight", **check})
    failures = [check for check in preflight if check["status"] != "passed"]
    if failures:
        writer.append({"schema": campaign.SCHEMA, "record_type": "campaign_preflight_failed",
                       "status": "failed", "checks": failures})
        first = failures[0]
        print("SOURCE_TILING_CAMPAIGN_PREFLIGHT_FAILED " + json.dumps({
              "kind": first.get("kind"), "worker_return_code": first.get("worker_return_code"),
              "failure": first.get("failure"), "last_stage": first.get("last_stage"),
              "logs": str(args.log_dir)}, sort_keys=True), flush=True)
        return 2
    print("SOURCE_TILING_CAMPAIGN_PREFLIGHT_PASSED " + json.dumps({
          "checks": [item["kind"] for item in preflight]}, sort_keys=True), flush=True)
    runner_hash = digest_file(args.runner)
    op = args.operator
    for workload in planned[op]:
        if admitted_by_op.get(op, 0) >= 5000:
            break
        group_key = campaign.stable_hash({"workload": workload, "runner_sha256": runner_hash,
                    "packages": [item["source_file_sha256"] for item in packages[op]],
                    "source_aiv_caps": caps, "minimum": minimum, "schema": campaign.SCHEMA})
        if group_key in completed:
            continue
        with tempfile.TemporaryDirectory(prefix=op + "_source_tiling_", dir=str(temporary_root)) as temporary:
            status, details = campaign.execute_group(args, workload, packages[op], caps, minimum,
                                                     base_env, group_key, Path(temporary))
        row = {"schema": campaign.SCHEMA, "group_key": group_key, "status": status,
               "workload": workload, **details}
        if status != "admitted":
            row.setdefault("valid_latency_count", 0)
        campaign.emit(writer, row)
        completed.add(group_key)
        if status == "admitted":
            admitted_by_op[op] = admitted_by_op.get(op, 0) + int(details["valid_latency_count"])
    _, final_counts, rejected = campaign.read_progress(args.log_dir)
    count = final_counts.get(op, 0)
    status = "complete" if count == 5000 else "completed_under_target"
    writer.append({"schema": campaign.SCHEMA, "record_type": "campaign_end", "operator": op,
                   "status": status, "formal_valid_latency_records": count,
                   "rejected_groups": rejected, "prior_rejected_groups": prior_rejected,
                   "matmul_included": False, "scatter_elements_included": False})
    print("SOURCE_TILING_CAMPAIGN_END " + json.dumps({"operator": op, "status": status,
          "formal_valid_latency_records": count, "rejected_groups": rejected,
          "logs": str(args.log_dir)}, sort_keys=True), flush=True)
    return 0 if status == "complete" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print("SOURCE_TILING_CAMPAIGN_FATAL " + json.dumps({"error": str(error)}), file=sys.stderr)
        raise SystemExit(2)
