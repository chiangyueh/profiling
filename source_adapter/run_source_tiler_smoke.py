#!/usr/bin/env python3
"""Launch one native CANN GatherElements source kernel before collection.

The smoke uses normal ACLNN executor lifecycle (GetWorkspaceSize, launch,
stream synchronization) under the private OPP overlay.  It proves that the
native dynamic source can compile and execute before the long campaign.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT / "run_non_matmul_candidate_campaign.py"


def campaign_module():
    spec = importlib.util.spec_from_file_location("source_tiler_campaign", CAMPAIGN)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load source tiler campaign helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def last_runner_stage(output: str) -> str | None:
    """Return the final runner stage without hiding a native-process crash."""
    stages = [line for line in output.splitlines() if line.startswith("MULTIOP_NPU_STAGE ")]
    return stages[-1] if stages else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--device", required=True, type=int)
    parser.add_argument("--source-package-manifest", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    if not args.runner.is_file() or args.device < 0:
        raise RuntimeError("runner or logical NPU device is invalid")
    module = campaign_module()
    package = module.validate_source_manifest(args.source_package_manifest)
    runtime_op = str(package["runtime_op"])
    workloads, _, _ = module.load_catalog()
    workload = next((item for item in workloads if item["op"] == runtime_op), None)
    if workload is None:
        raise RuntimeError("no legal smoke workload for {}".format(runtime_op))
    # The deployment gate must use the normal hardware core budget.  A cap of
    # one is a later candidate axis, not a valid proxy for whether the native
    # source can load or launch on this 20-core target.
    candidate = module.candidate_descriptor(package, max(module.SOURCE_AIV_CAPS))
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="source_tiler_smoke_", dir=args.work_dir) as temporary:
        audit = Path(temporary) / "audit.jsonl"
        result, output, wall, rc = module.run_worker(
            module.worker_args(args.runner, workload, args.device, 0, 0) + ["--source-tiling-only", "1"],
            module.source_environment(dict(os.environ), package, candidate, audit),
        )
        observed, reason = module.source_audit_emitted(audit, package, candidate)
    passed = bool(observed and rc == 0 and result.get("status") == "success")
    record = {
        "status": "passed" if passed else "failed", "operator": runtime_op,
        "workload_id": workload["workload_id"], "worker_return_code": rc,
        "worker_status": result.get("status"), "worker_wall_ms": wall,
        "failure": None if passed else (reason if not observed else module.compact_failure(output)),
        # ``reason`` can be a secondary consequence of a native crash.  Keep
        # the last real runner transition on the same terminal/log record so
        # an operator does not need to recover a discarded subprocess trace.
        "last_runner_stage": None if passed else last_runner_stage(output),
    }
    print("GATHER_ELEMENTS_NATIVE_SOURCE_SMOKE " + json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print("GATHER_ELEMENTS_NATIVE_SOURCE_SMOKE " + json.dumps({"status": "failed", "failure": str(error)}, sort_keys=True), flush=True)
        raise SystemExit(2)
