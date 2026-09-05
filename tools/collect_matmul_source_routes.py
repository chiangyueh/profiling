#!/usr/bin/env python3
"""Collect every successful original MatMulV3 route/core-cap tiling.

The private host package is process-local.  Executor planning invokes the
original CANN dispatcher but stops before launching a kernel; the instrumented
host tiler writes the exact raw tiling records used as frontier anchors.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


def load_audit(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid source audit JSON at line {line_number}: {error}")
        if row.get("schema") == "matmul_v3_source_route_observation_v1":
            rows.append(row)
    return rows


def complete(rows: list[dict], workload_ids: list[str], max_cores: int) -> bool:
    by_workload: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_workload[str(row.get("workload_id", ""))].append(row)
    return all(
        any(
            row.get("route") == "ALL"
            and int(row.get("core_cap", 0)) == max_cores
            and row.get("status") == "success"
            and row.get("route_matched") is True
            and len(str(row.get("raw_tiling_hex", ""))) == 544
            for row in by_workload[workload_id]
        )
        for workload_id in workload_ids
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--opapi", type=Path, required=True)
    parser.add_argument("--tiling-library", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-cores", type=int, required=True)
    args = parser.parse_args()

    with args.workloads.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        workloads = list(reader)
    if not workloads or args.max_cores <= 0:
        raise RuntimeError("source route collector received an empty workload set or core count")
    workload_ids = [row["workload_id"] for row in workloads]
    args.state_dir.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    existing = load_audit(args.audit)
    if complete(existing, workload_ids, args.max_cores):
        print(f"SOURCE_ROUTE_DISCOVERY cached shapes={len(workloads)}")
        return 0
    args.audit.unlink(missing_ok=True)

    private_lib = args.opapi.resolve()
    private_tiling = args.tiling_library.resolve()
    package_root = args.package_root.resolve()
    if (
        not args.runner.is_file() or not private_lib.is_file()
        or not private_tiling.is_file() or not package_root.is_dir()
    ):
        raise RuntimeError("source route runner or private MatMul host package is missing")
    runner_log = args.state_dir / "source_tiler.log"
    runner_log.unlink(missing_ok=True)
    for index, workload in enumerate(workloads, 1):
        one_workload = args.state_dir / f"workload_{index:02d}.csv"
        with one_workload.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerow(workload)
        env = os.environ.copy()
        env.update({
            "ASCEND_CUSTOM_OPP_PATH": str(package_root),
            "MATMUL_SOURCE_OPAPI_LIBRARY": str(private_lib),
            "MATMUL_SOURCE_TILING_LIBRARY": str(private_tiling),
            "MATMUL_SOURCE_ROUTE_AUDIT_PATH": str(args.audit.resolve()),
            "MATMUL_SOURCE_ROUTE_MAX_CORES": str(args.max_cores),
            "MATMUL_SOURCE_ROUTE_WORKLOAD_ID": workload["workload_id"],
            "LD_LIBRARY_PATH": (
                str(private_lib.parent) + ":" + str(private_tiling.parent) + ":"
                + env.get("LD_LIBRARY_PATH", "")
            ).rstrip(":"),
        })
        command = [
            str(args.runner.resolve()), "--planning-only",
            "--candidates", str(one_workload),
            "--output", str(args.state_dir / f"planning_{index:02d}.csv"),
            "--samples-output", str(args.state_dir / f"planning_samples_{index:02d}.csv"),
            "--device", str(args.device), "--warmup", "0", "--repeat", "1",
            "--samples", "1", "--numeric-preflight-max-mib", "0",
        ]
        result = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
        with runner_log.open("a", encoding="utf-8") as stream:
            stream.write(f"===== {workload['workload_id']} rc={result.returncode} =====\n")
            stream.write(result.stdout)
            stream.write(result.stderr)
        if result.returncode != 0:
            tail = "\n".join((result.stdout + result.stderr).splitlines()[-12:])
            raise RuntimeError(
                f"source route planning failed for {workload['workload_id']} rc={result.returncode}\n{tail}"
            )
        current = load_audit(args.audit)
        if not complete(current, workload_ids[:index], args.max_cores):
            raise RuntimeError(
                f"source route audit lacks a valid ALL@{args.max_cores} anchor for "
                f"{workload['workload_id']}"
            )
        matched = sum(
            row.get("workload_id") == workload["workload_id"]
            and row.get("route_matched") is True
            for row in current
        )
        print(f"SOURCE_ROUTE_DISCOVERY {index}/{len(workloads)} matched={matched}", flush=True)

    rows = load_audit(args.audit)
    if not complete(rows, workload_ids, args.max_cores):
        raise RuntimeError("source route audit is incomplete")
    identities = {
        (row.get("workload_id"), row.get("tiling_key"), row.get("block_dim"), row.get("raw_tiling_hex"))
        for row in rows if row.get("route_matched") is True
    }
    routes = Counter(
        str(row.get("route")) for row in rows if row.get("route_matched") is True
    )
    print(
        "SOURCE_ROUTE_DISCOVERY_DONE "
        f"shapes={len(workloads)} matched_rows={sum(routes.values())} "
        f"distinct_anchors={len(identities)} routes={dict(sorted(routes.items()))}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"fatal: {error}", flush=True)
        raise SystemExit(1)
