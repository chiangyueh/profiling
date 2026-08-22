#!/usr/bin/env python3
"""Run the explicit real-NPU multi-operator catalog.

One workload is one process. The worker is never force-killed by this
orchestrator: a hung device operation is evidence that must be left intact for
the administrator to inspect, rather than being converted into a synthetic
timeout. Successful exact workload/spec pairs are durable and are not run
again on the next invocation.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

from workloads import OPS, audit, catalog


RESULT_PREFIX = "MULTIOP_NPU_RESULT "
STAGE_PREFIX = "MULTIOP_NPU_STAGE "
PROGRESS_PREFIX = "MULTIOP_PROGRESS "


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def spec_hash(workload):
    return hashlib.sha256(canonical(workload).encode()).hexdigest()


def command_for(runner, workload, device, warmup, samples):
    command = [
        str(runner),
        "--workload-id", workload["workload_id"],
        "--op", workload["op"],
        "--device", str(device),
        "--expected-soc", "Ascend910B3",
        "--warmup", str(warmup),
        "--samples", str(samples),
    ]
    names = {
        "dtype": "dtype", "m": "m", "n": "n", "k": "k",
        "trans_a": "trans-a", "trans_b": "trans-b", "shape": "shape",
        "perm": "perm", "axis": "axis", "index_shape": "index-shape",
        "index_dtype": "index-dtype", "reduce": "reduce", "layout": "layout",
        "batch": "batch", "q_heads": "q-heads", "kv_heads": "kv-heads",
        "q_seq": "q-seq", "kv_seq": "kv-seq", "head_dim": "head-dim",
    }
    for key, flag in names.items():
        if key not in workload:
            continue
        value = workload[key]
        if isinstance(value, list):
            value = ",".join(map(str, value))
        elif isinstance(value, bool):
            value = int(value)
        command.extend([f"--{flag}", str(value)])
    return command


def parse_result(output):
    records = []
    for line in output.splitlines():
        if line.startswith(RESULT_PREFIX):
            records.append(json.loads(line[len(RESULT_PREFIX):]))
    if len(records) != 1:
        raise ValueError(f"expected one {RESULT_PREFIX.strip()} record, observed {len(records)}")
    return records[0]


def last_npu_stage(output):
    for line in reversed(output.splitlines()):
        if line.startswith(STAGE_PREFIX):
            try:
                return json.loads(line[len(STAGE_PREFIX):]).get("stage")
            except json.JSONDecodeError:
                return "malformed_stage_record"
    return None


def worker_tail(output, limit=600):
    """Keep a small diagnostic only when the worker broke its result contract."""
    text = output.strip().replace("\x00", "")
    return text[-limit:] if text else "worker emitted no output"


def run_one(runner, workload, stage, device, warmup, samples):
    started = time.monotonic()
    command = command_for(runner, workload, device, warmup, samples)
    try:
        # Deliberately no subprocess timeout and no killpg/SIGKILL path. A
        # host-side forced termination was the source of prior misleading
        # failures and can leave a device task in an indeterminate state.
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            env=os.environ.copy(),
            check=False,
        )
        output = completed.stdout or ""
        observed_stage = last_npu_stage(output)
        try:
            result = parse_result(output)
        except Exception as error:
            result = {
                "schema": "multi_op_real_npu_v2",
                "status": "failed",
                "backend": "aclnn_real_npu",
                "workload_id": workload["workload_id"],
                "op": workload["op"],
                "error": f"worker output contract failed: {error}; tail={worker_tail(output)!r}",
            }
        if completed.returncode != 0 and result.get("status") == "success":
            result["status"] = "failed"
            result["error"] = f"worker returned {completed.returncode} after success record"
        result["runner_rc"] = completed.returncode
        if observed_stage is not None:
            result["last_npu_stage"] = observed_stage
    except Exception as error:
        result = {
            "schema": "multi_op_real_npu_v2",
            "status": "failed",
            "backend": "aclnn_real_npu",
            "workload_id": workload["workload_id"],
            "op": workload["op"],
            "runner_rc": 125,
            "error": f"worker process launch failed: {error}",
        }
    result.update({
        "stage": stage,
        "spec_sha256": spec_hash(workload),
        "wall_ms": (time.monotonic() - started) * 1000.0,
        "coverage": workload["coverage"],
    })
    return result


def load_completed(progress_path):
    completed = set()
    if not progress_path.exists():
        return completed
    for line in progress_path.read_text(errors="replace").splitlines():
        if not line.startswith(PROGRESS_PREFIX):
            continue
        try:
            record = json.loads(line[len(PROGRESS_PREFIX):])
        except json.JSONDecodeError:
            continue
        if record.get("status") == "success":
            completed.add((record.get("stage"), record.get("workload_id"), record.get("spec_sha256")))
    return completed


def append_progress(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(PROGRESS_PREFIX + canonical(record) + "\n")
        output.flush()
        os.fsync(output.fileno())


def interleaved(workloads):
    grouped = defaultdict(list)
    for workload in workloads:
        grouped[workload["op"]].append(workload)
    for round_index in range(max(map(len, grouped.values()))):
        for op in OPS:
            if round_index < len(grouped[op]):
                yield grouped[op][round_index]


def emit(kind, value):
    print(kind + " " + canonical(value), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--samples", type=int, default=7)
    args = parser.parse_args()
    if not args.runner.is_file():
        raise SystemExit(f"runner does not exist: {args.runner}")
    if args.warmup < 0 or args.samples < 1:
        raise SystemExit("full campaign requires warmup >= 0 and samples >= 1")

    workloads = catalog()
    completed = load_completed(args.progress)
    emit("MULTIOP_WORKLOAD_AUDIT", audit(workloads))
    emit("MULTIOP_RESUME_AUDIT", {
        "progress": str(args.progress),
        "successful_exact_records": len(completed),
        "policy": "only exact successful stage/workload/spec records are reused",
    })

    outcomes = Counter()
    op_outcomes = defaultdict(Counter)
    viable_ops = set()
    failed_preflights = set()

    # Every operator gets its own preflight. A failed one is recorded and
    # skipped, but does not suppress measurements for the other operators.
    for workload in (item for item in workloads if item["preflight"]):
        key = ("preflight", workload["workload_id"], spec_hash(workload))
        if key in completed:
            viable_ops.add(workload["op"])
            emit("MULTIOP_PREFLIGHT", {
                "op": workload["op"], "workload_id": workload["workload_id"],
                "status": "already_successful",
            })
            continue
        emit("MULTIOP_WORKLOAD_BEGIN", {"stage": "preflight", **workload})
        result = run_one(args.runner, workload, "preflight", args.device, 0, 0)
        emit("MULTIOP_WORKLOAD_END", result)
        append_progress(args.progress, result)
        outcomes[result["status"]] += 1
        op_outcomes[workload["op"]][result["status"]] += 1
        if result["status"] == "success":
            viable_ops.add(workload["op"])
        else:
            failed_preflights.add(workload["op"])

    emit("MULTIOP_PREFLIGHT_SUMMARY", {
        "passed_ops": sorted(viable_ops),
        "failed_ops": sorted(failed_preflights),
        "policy": "failed preflight skips only that operator; all other viable operators continue",
    })

    for workload in interleaved(workloads):
        if workload["op"] not in viable_ops or workload["preflight"]:
            continue
        key = ("full", workload["workload_id"], spec_hash(workload))
        if key in completed:
            continue
        emit("MULTIOP_WORKLOAD_BEGIN", {"stage": "full", **workload})
        result = run_one(args.runner, workload, "full", args.device, args.warmup, args.samples)
        emit("MULTIOP_WORKLOAD_END", result)
        append_progress(args.progress, result)
        outcomes[result["status"]] += 1
        op_outcomes[workload["op"]][result["status"]] += 1

    emit("MULTIOP_CAMPAIGN_SUMMARY", {
        "schema": "multi_op_campaign_summary_v2",
        "backend": "aclnn_real_npu_only",
        "catalog_workloads": len(workloads),
        "preflight_passed_ops": sorted(viable_ops),
        "preflight_failed_ops": sorted(failed_preflights),
        "outcomes_this_invocation": dict(outcomes),
        "outcomes_by_op_this_invocation": {op: dict(op_outcomes[op]) for op in OPS},
        "progress": str(args.progress),
    })


if __name__ == "__main__":
    main()
