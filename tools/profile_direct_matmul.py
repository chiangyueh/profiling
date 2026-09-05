#!/usr/bin/env python3
"""Measure exact solver tilings through direct CANN 8.1 kernel launches."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path

from direct_matmul_tiling import write_manifest


PROFILE_COLUMNS = [
    "workload_id", "rank", "source", "candidate_role",
    "m", "n", "k", "dtype", "trans_a", "trans_b", "execution_mode",
    "used_core_num", "hint_single_core_m", "hint_single_core_n",
    "hint_single_core_k", "hint_base_m", "hint_base_n", "hint_base_k",
    "official_base_m", "official_base_n", "official_base_k",
    "official_core_num", "official_m_dim", "official_n_dim",
    "proxy_total", "success", "preflight_passed", "preflight_mode", "error",
    "min_ms", "mean_ms", "median_ms", "stddev_ms", "p95_ms", "max_ms",
    "tflops", "warmup", "repeat", "samples", "tiling_signature", "tiling_bin",
    "device_prepare_ms", "executor_setup_ms", "numeric_preflight_ms",
    "workspace_bytes", "warmup_wall_ms", "measurement_wall_ms", "runner_total_ms",
    "model_schedule_sha256", "measurement_source", "tiling_applied",
    "full_output_validated", "actual_tiling_sha256", "actual_tiling_fnv1a64",
    "actual_kernel_suffix", "actual_block_dim",
]
SAMPLE_COLUMNS = ["workload_id", "rank", "candidate_role", "sample", "latency_ms"]
SCHEMA = "matmul_direct_candidate_measurement_v1"


def truthy(value: object) -> bool:
    return str(value or "").lower() in {"1", "true", "yes", "on"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


class JsonlLog:
    def __init__(self, directory: Path, max_bytes: int) -> None:
        self.directory = directory
        self.max_bytes = max_bytes
        directory.mkdir(parents=True, exist_ok=True)
        paths = sorted(
            (path for path in directory.glob("[0-9]*.log") if path.stem.isdigit()),
            key=lambda path: int(path.stem),
        )
        for path in paths:
            if path.stat().st_size > max_bytes:
                raise RuntimeError(f"log exceeds 50 MiB contract: {path}")
        self.keys: set[str] = set()
        for path in paths:
            with path.open(encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    try:
                        key = str(json.loads(line).get("record_key", ""))
                    except json.JSONDecodeError:
                        continue
                    if key:
                        self.keys.add(key)
        self.index = int(paths[-1].stem) if paths else 1
        self.path = directory / f"{self.index}.log"
        self.size = self.path.stat().st_size if self.path.exists() else 0
        if self.size:
            with self.path.open("rb") as stream:
                stream.seek(-1, os.SEEK_END)
                complete_tail = stream.read(1) == b"\n"
            if not complete_tail:
                self.index += 1
                self.path = directory / f"{self.index}.log"
                self.size = 0
        self.stream = self.path.open("ab")

    def append(self, record: dict) -> None:
        encoded = (json.dumps(record, separators=(",", ":")) + "\n").encode()
        if len(encoded) > self.max_bytes:
            raise RuntimeError("one JSONL record exceeds log size contract")
        if self.size and self.size + len(encoded) > self.max_bytes:
            self.stream.close()
            self.index += 1
            self.path = self.directory / f"{self.index}.log"
            self.stream = self.path.open("ab")
            self.size = 0
        self.stream.write(encoded)
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.size += len(encoded)

    def append_once(self, key: str, record: dict) -> None:
        if key in self.keys:
            return
        self.append({"record_key": key, **record})
        self.keys.add(key)

    def close(self) -> None:
        self.stream.close()


def load_completed(
    log_directory: Path,
    manifest: dict[tuple[str, str], dict[str, str]],
) -> tuple[
    dict[tuple[str, str], dict[str, str]],
    dict[tuple[str, str], list[float]],
    set[tuple[str, str]],
]:
    completed: dict[tuple[str, str], dict[str, str]] = {}
    samples: dict[tuple[str, str], list[float]] = {}
    attempted: set[tuple[str, str]] = set()
    paths = sorted(
        (path for path in log_directory.glob("[0-9]*.log") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("schema") != SCHEMA:
                    continue
                if record.get("record_type") == "candidate_failure":
                    candidate = record.get("candidate") or {}
                    key = (
                        str(candidate.get("workload_id", "")),
                        str(candidate.get("rank", "")),
                    )
                    if key in manifest:
                        attempted.add(key)
                    continue
                if record.get("record_type") != "candidate":
                    continue
                profile = record.get("measurement") or {}
                key = (str(profile.get("workload_id", "")), str(profile.get("rank", "")))
                expected = manifest.get(key)
                if expected is None:
                    continue
                if not (
                    profile.get("measurement_source") == "direct_tiling_buffer"
                    and truthy(profile.get("tiling_applied"))
                    and truthy(profile.get("full_output_validated"))
                    and truthy(profile.get("success"))
                    and profile.get("model_schedule_sha256") == expected["model_schedule_sha256"]
                    and profile.get("actual_tiling_sha256") == expected["tiling_sha256"]
                    and profile.get("actual_tiling_fnv1a64") == expected["tiling_fnv1a64"]
                    and str(profile.get("actual_kernel_suffix")) == expected["kernel_suffix"]
                    and str(profile.get("actual_block_dim")) == expected["used_core_num"]
                ):
                    continue
                raw_samples = record.get("samples_ms") or []
                if len(raw_samples) != int(profile.get("samples", "0")):
                    continue
                numeric_samples = [float(value) for value in raw_samples]
                if any(not math.isfinite(value) or value <= 0 for value in numeric_samples):
                    continue
                completed[key] = {field: str(profile.get(field, "")) for field in PROFILE_COLUMNS}
                samples[key] = numeric_samples
                attempted.add(key)
    return completed, samples, attempted


def candidate_profile(
    candidate: dict[str, str], result: dict, warmup: int, repeat: int, samples: int,
) -> dict[str, str]:
    row = {field: "" for field in PROFILE_COLUMNS}
    row.update({
        "workload_id": candidate["workload_id"], "rank": candidate["rank"],
        "source": "hardware_cost_model_direct_cann81",
        "candidate_role": "searched",
        "m": candidate["m"], "n": candidate["n"], "k": candidate["k"],
        "dtype": candidate["dtype"], "trans_a": candidate["trans_a"],
        "trans_b": candidate["trans_b"],
        "execution_mode": candidate.get("execution_mode", ""),
        "used_core_num": candidate["used_core_num"],
        "hint_single_core_m": candidate.get("single_core_m", ""),
        "hint_single_core_n": candidate.get("single_core_n", ""),
        "hint_single_core_k": candidate.get("single_core_k", ""),
        "hint_base_m": candidate.get("base_m", ""),
        "hint_base_n": candidate.get("base_n", ""),
        "hint_base_k": candidate.get("base_k", ""),
        "proxy_total": candidate.get("new_model_cycles", ""),
        "success": "1", "preflight_passed": "1",
        "preflight_mode": "numeric_signed_axes_full_v3", "error": "",
        "warmup": str(warmup), "repeat": str(repeat), "samples": str(samples),
        "tiling_signature": candidate.get("tiling_signature", ""),
        "tiling_bin": candidate.get("tiling_bin", ""),
        "model_schedule_sha256": candidate["model_schedule_sha256"],
    })
    for field in (
        "min_ms", "mean_ms", "median_ms", "stddev_ms", "p95_ms", "max_ms",
        "tflops", "device_prepare_ms", "numeric_preflight_ms", "workspace_bytes",
        "warmup_wall_ms", "measurement_wall_ms", "runner_total_ms",
        "measurement_source", "tiling_applied", "full_output_validated",
        "actual_tiling_sha256", "actual_tiling_fnv1a64", "actual_kernel_suffix",
        "actual_block_dim",
    ):
        row[field] = str(result.get(field, ""))
    return row


def api_baseline(row: dict[str, str]) -> dict[str, str]:
    output = {field: "" for field in PROFILE_COLUMNS}
    output.update({field: row.get(field, "") for field in PROFILE_COLUMNS})
    output.update({
        "rank": "0", "source": "official_default",
        "candidate_role": "api_auto_baseline",
        "measurement_source": "separate_installed_aclnn_baseline",
    })
    return output


def validate_official(rows: list[dict[str, str]], workloads: list[dict[str, str]]) -> bool:
    expected = {row["workload_id"] for row in workloads}
    actual = {row.get("workload_id", "") for row in rows}
    return (
        actual == expected and len(rows) == len(expected)
        and all(
            row.get("source") == "installed_aclnn_matmul"
            and row.get("candidate_role") == "official_operator_baseline"
            and truthy(row.get("success")) and truthy(row.get("preflight_passed"))
            and row.get("preflight_mode") == "numeric_signed_axes_full_v3"
            and float(row.get("median_ms") or 0) > 0
            for row in rows
        )
    )


def official_samples(
    rows: list[dict[str, str]], workloads: list[dict[str, str]], count: int,
) -> dict[str, list[float]] | None:
    expected = {row["workload_id"] for row in workloads}
    grouped: dict[str, list[tuple[int, float]]] = {}
    try:
        for row in rows:
            workload_id = row["workload_id"]
            sample = int(row["sample"])
            latency = float(row["latency_ms"])
            if workload_id not in expected or not math.isfinite(latency) or latency <= 0:
                return None
            grouped.setdefault(workload_id, []).append((sample, latency))
    except (KeyError, TypeError, ValueError):
        return None
    if set(grouped) != expected:
        return None
    result = {}
    for workload_id, values in grouped.items():
        values.sort()
        if [sample for sample, _ in values] != list(range(count)):
            return None
        result[workload_id] = [latency for _, latency in values]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-builder", type=Path, required=True)
    parser.add_argument("--variant-runner-directory", type=Path, required=True)
    parser.add_argument("--official-runner", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tiling-directory", type=Path, required=True)
    parser.add_argument("--profile-output", type=Path, required=True)
    parser.add_argument("--samples-output", type=Path, required=True)
    parser.add_argument("--official-output", type=Path, required=True)
    parser.add_argument("--official-samples-output", type=Path, required=True)
    parser.add_argument("--log-directory", type=Path, required=True)
    parser.add_argument("--log-max-bytes", type=int, default=50 * 1024 * 1024)
    parser.add_argument("--l2-bytes", type=int, required=True)
    parser.add_argument("--aic-cores", type=int, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=20)
    args = parser.parse_args()

    workloads = read_rows(args.workloads)
    all_candidates = [
        row for row in read_rows(args.candidates)
        if row.get("candidate_role") == "searched"
    ]
    candidates = [row for row in all_candidates if not truthy(row.get("is_reserve"))]
    reserves = [row for row in all_candidates if truthy(row.get("is_reserve"))]
    if (
        len(workloads) != 70 or len(candidates) != 2185
        or len(reserves) != 560 or len(all_candidates) != 2745
    ):
        raise RuntimeError(
            "formal input contract failed: "
            f"shapes={len(workloads)} candidates={len(candidates)} "
            f"reserves={len(reserves)}"
        )
    workload_contract = {
        row["workload_id"]: int(row["required_successful_tilings"])
        for row in workloads
    }
    grouped_candidates: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in all_candidates:
        grouped_candidates[row["workload_id"]].append(row)
    if set(grouped_candidates) != set(workload_contract):
        raise RuntimeError("candidate workload identities do not match the fixed campaign")
    for workload_id, rows in grouped_candidates.items():
        required_count = workload_contract[workload_id]
        roles = [truthy(row.get("is_reserve")) for row in rows]
        if (
            roles.count(False) != required_count
            or roles.count(True) != 8
            or roles != sorted(roles)
            or [int(row["rank"]) for row in rows] != list(range(1, len(rows) + 1))
        ):
            raise RuntimeError(
                f"{workload_id}: formal/reserve ordering contract failed"
            )
    materialized = write_manifest(
        args.candidates, args.tiling_directory, args.manifest,
        l2_bytes=args.l2_bytes, aic_cores=args.aic_cores,
        include_reserves=True,
    )
    if materialized != len(all_candidates):
        raise RuntimeError("not every candidate/reserve produced one exact tiling buffer")
    manifest_rows = read_rows(args.manifest)
    manifest = {(row["workload_id"], row["rank"]): row for row in manifest_rows}
    candidate_map = {
        (row["workload_id"], row["rank"]): row for row in all_candidates
    }
    log = JsonlLog(args.log_directory, args.log_max_bytes)
    log.append_once("campaign:begin", {
        "schema": SCHEMA,
        "record_type": "campaign_begin",
        "status": "running",
        "shapes": 70,
        "required_successful_candidates": 2185,
        "available_reserves": 560,
        "official_baselines": 70,
        "measurement": "one warmup plus three device-event samples",
        "candidate_execution": "compile_one_variant_then_measure_immediately",
    })

    official_rows = read_rows(args.official_output) if args.official_output.is_file() else []
    official_sample_rows = (
        read_rows(args.official_samples_output)
        if args.official_samples_output.is_file() else []
    )
    official_sample_map = official_samples(
        official_sample_rows, workloads, args.samples
    )
    if not validate_official(official_rows, workloads) or official_sample_map is None:
        command = [
            str(args.official_runner), "--candidates", str(args.workloads),
            "--output", str(args.official_output),
            "--samples-output", str(args.official_samples_output),
            "--device", str(args.device), "--warmup", str(args.warmup),
            "--repeat", str(args.repeat), "--samples", str(args.samples),
            "--numeric-preflight-max-mib", "256", "--structured-full-preflight",
            "--validate-after-measurement",
        ]
        environment = dict(os.environ)
        for name in tuple(environment):
            if "RUNTIME_KB" in name or "TUNING_BANK" in name:
                environment.pop(name, None)
        print("OFFICIAL_BASELINE_BEGIN shapes=70", flush=True)
        official_process = subprocess.Popen(
            command, env=environment, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, bufsize=1,
        )
        assert official_process.stdout is not None
        tail_lines: deque[str] = deque(maxlen=20)
        official_done = 0
        for line in official_process.stdout:
            line = line.rstrip("\n")
            tail_lines.append(line)
            if line.startswith("official_done "):
                official_done += 1
                if official_done == 1 or official_done % 20 == 0 or official_done == 70:
                    print(f"OFFICIAL_BASELINE_PROGRESS {official_done}/70", flush=True)
        official_return_code = official_process.wait()
        if official_return_code:
            raise RuntimeError(
                f"official baseline failed rc={official_return_code}: "
                + "\n".join(tail_lines)
            )
        official_rows = read_rows(args.official_output)
        official_sample_rows = read_rows(args.official_samples_output)
        official_sample_map = official_samples(
            official_sample_rows, workloads, args.samples
        )
        if not validate_official(official_rows, workloads) or official_sample_map is None:
            raise RuntimeError("official baseline output contract failed")
        print("OFFICIAL_BASELINE_DONE shapes=70", flush=True)
    else:
        print("OFFICIAL_BASELINE_RESUME shapes=70", flush=True)
    for official in official_rows:
        log.append_once(
            f"official:{official['workload_id']}",
            {"schema": SCHEMA, "record_type": "official_baseline",
             "measurement": official,
             "samples_ms": official_sample_map[official["workload_id"]]},
        )

    completed, sample_map, attempted = load_completed(args.log_directory, manifest)
    required = {
        row["workload_id"]: int(row["required_successful_tilings"])
        for row in workloads
    }

    def completion_counts() -> dict[str, int]:
        counts: dict[str, int] = {}
        for workload_id, _ in completed:
            counts[workload_id] = counts.get(workload_id, 0) + 1
        return counts

    completed_by_workload = completion_counts()
    if any(completed_by_workload.get(key, 0) > value for key, value in required.items()):
        raise RuntimeError("resume log contains more accepted candidates than requested")
    remaining_successes = sum(
        value - completed_by_workload.get(key, 0)
        for key, value in required.items()
    )
    print(
        "DIRECT_MEASUREMENT_PLAN "
        f"success_target=2185 completed={len(completed)} "
        f"remaining_successes={remaining_successes} "
        f"available_pending_pool={len(all_candidates) - len(attempted)} "
        "reserves=only_after_numeric_failure",
        flush=True,
    )

    direct_environment = dict(os.environ)
    for name in tuple(direct_environment):
        if "RUNTIME_KB" in name or "TUNING_BANK" in name:
            direct_environment.pop(name, None)
    built_variants: set[tuple[str, str]] = set()

    def run_variant(dtype: str, suffix: str, rows: list[dict[str, str]], phase: str) -> None:
        if not rows:
            return
        target = f"direct_matmul_kernel_{dtype}_{suffix}"
        runner = args.variant_runner_directory / f"direct_matmul_{dtype}_k{suffix}"
        if (dtype, suffix) not in built_variants:
            build_environment = dict(direct_environment)
            build_environment.update({
                "BUILD_COMPONENTS": "variant",
                "BUILD_JOBS": "1",
                "DIRECT_KERNEL_TARGET": target,
            })
            print(
                f"DIRECT_VARIANT_BUILD_BEGIN variant={dtype}_k{suffix} "
                f"candidate_count={len(rows)}",
                flush=True,
            )
            build_process = subprocess.Popen(
                [str(args.variant_builder)], env=build_environment, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
            )
            assert build_process.stdout is not None
            build_tail: deque[str] = deque(maxlen=30)
            for line in build_process.stdout:
                line = line.rstrip("\n")
                build_tail.append(line)
                if line.startswith(("DIRECT_KERNEL_BUILD ",
                                    "DIRECT_VARIANT_RUNNER_LINK ",
                                    "DIRECT_VARIANT_READY ", "fatal:",
                                    "build_error:")):
                    print(line, flush=True)
            build_return_code = build_process.wait()
            if build_return_code or not runner.is_file() or not os.access(runner, os.X_OK):
                raise RuntimeError(
                    f"variant build failed for {dtype}_k{suffix} "
                    f"rc={build_return_code}: " + "\n".join(build_tail)
                )
            built_variants.add((dtype, suffix))
            print(f"DIRECT_VARIANT_BUILD_DONE variant={dtype}_k{suffix}", flush=True)

        group_sizes: dict[str, int] = {}
        for row in rows:
            group_sizes[row["workload_id"]] = group_sizes.get(row["workload_id"], 0) + 1
        variant_rows: list[dict[str, str]] = []
        for row in rows:
            variant_row = dict(row)
            variant_row["required_successful_tilings"] = str(
                group_sizes[row["workload_id"]]
            )
            variant_rows.append(variant_row)
        manifest_directory = args.manifest.with_name("variant_manifests")
        variant_manifest = manifest_directory / f"{dtype}_k{suffix}_{phase}.csv"
        write_rows(variant_manifest, variant_rows, list(variant_rows[0]))

        preflight = subprocess.run(
            [str(runner), "--manifest", str(variant_manifest), "--validate-input"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=direct_environment, check=False,
        )
        if preflight.returncode or "DIRECT_MATMUL_INPUT status=passed" not in preflight.stdout:
            raise RuntimeError(
                f"variant input preflight failed for {dtype}_k{suffix} "
                f"rc={preflight.returncode}: " + preflight.stdout[-2000:]
            )
        print(
            f"DIRECT_VARIANT_MEASUREMENT_BEGIN variant={dtype}_k{suffix} "
            f"candidates={len(variant_rows)} phase={phase}",
            flush=True,
        )
        command = [
            str(runner), "--manifest", str(variant_manifest),
            "--device", str(args.device), "--warmup", str(args.warmup),
            "--repeat", str(args.repeat), "--samples", str(args.samples),
            "--allow-partial",
        ]
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=direct_environment,
        )
        assert process.stdout is not None
        direct_tail: deque[str] = deque(maxlen=20)
        direct_failure = ""
        returned: set[tuple[str, str]] = set()
        for line in process.stdout:
            line = line.rstrip("\n")
            direct_tail.append(line)
            if not line.startswith("DIRECT_MATMUL_RESULT "):
                continue
            if direct_failure:
                continue
            result = json.loads(line.split(" ", 1)[1])
            key = (str(result.get("workload_id", "")), str(result.get("rank", "")))
            returned.add(key)
            candidate = candidate_map.get(key)
            expected_manifest = manifest.get(key)
            if candidate is None or expected_manifest is None:
                direct_failure = f"direct runner returned an unknown identity: {key}"
                continue
            if result.get("status") != "success":
                log.append_once(f"failure:{key[0]}:{key[1]}", {
                    "schema": SCHEMA, "record_type": "candidate_failure",
                    "candidate": candidate, "runner": result,
                })
                attempted.add(key)
                continue
            profile = candidate_profile(
                candidate, result, args.warmup, args.repeat, args.samples
            )
            if not (
                profile["model_schedule_sha256"] == expected_manifest["model_schedule_sha256"]
                and profile["actual_tiling_sha256"] == expected_manifest["tiling_sha256"]
                and profile["actual_tiling_fnv1a64"] == expected_manifest["tiling_fnv1a64"]
                and profile["actual_kernel_suffix"] == expected_manifest["kernel_suffix"]
                and profile["actual_block_dim"] == expected_manifest["used_core_num"]
                and profile["measurement_source"] == "direct_tiling_buffer"
                and truthy(profile["tiling_applied"])
                and truthy(profile["full_output_validated"])
            ):
                direct_failure = f"direct execution attestation mismatch: {key}"
                continue
            raw_samples = [float(value) for value in result.get("samples_ms", [])]
            if len(raw_samples) != args.samples or any(
                not math.isfinite(value) or value <= 0 for value in raw_samples
            ):
                direct_failure = f"raw sample contract failed: {key}"
                continue
            log.append_once(f"candidate:{key[0]}:{key[1]}", {
                "schema": SCHEMA, "record_type": "candidate",
                "candidate": candidate, "manifest": expected_manifest,
                "measurement": profile, "samples_ms": raw_samples,
            })
            completed[key] = profile
            sample_map[key] = raw_samples
            attempted.add(key)
            count = len(completed)
            if count == 1 or count % args.progress_every == 0 or count == len(candidates):
                print(f"DIRECT_MEASUREMENT_PROGRESS {count}/2185", flush=True)
        return_code = process.wait()
        if direct_failure:
            raise RuntimeError(direct_failure)
        if return_code:
            raise RuntimeError(
                f"direct MatMul runner failed rc={return_code}: "
                + "\n".join(direct_tail)
            )
        expected_returned = {
            (row["workload_id"], row["rank"]) for row in variant_rows
        }
        if returned != expected_returned:
            raise RuntimeError(
                f"variant runner omitted results for {dtype}_k{suffix}: "
                f"returned={len(returned)} expected={len(expected_returned)}"
            )
        print(
            f"DIRECT_VARIANT_MEASUREMENT_DONE variant={dtype}_k{suffix} "
            f"candidates={len(variant_rows)} total_completed={len(completed)}/2185",
            flush=True,
        )

    def run_grouped(rows: list[dict[str, str]], phase: str) -> None:
        grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
        for row in rows:
            key = (row["dtype"], row["kernel_suffix"])
            grouped.setdefault(key, []).append(row)
        for (dtype, suffix), variant_rows in grouped.items():
            run_variant(dtype, suffix, variant_rows, phase)

    completed_by_workload = completion_counts()
    formal_pending = [
        row for row in manifest_rows
        if not truthy(row.get("is_reserve"))
        and (row["workload_id"], row["rank"]) not in attempted
        and completed_by_workload.get(row["workload_id"], 0)
            < required[row["workload_id"]]
    ]
    run_grouped(formal_pending, "formal")

    reserve_round = 0
    while True:
        completed_by_workload = completion_counts()
        deficits = {
            workload_id: target - completed_by_workload.get(workload_id, 0)
            for workload_id, target in required.items()
            if completed_by_workload.get(workload_id, 0) < target
        }
        if not deficits:
            break
        reserve_round += 1
        reserve_pending: list[dict[str, str]] = []
        for workload_id, deficit in deficits.items():
            available = [
                row for row in manifest_rows
                if row["workload_id"] == workload_id
                and truthy(row.get("is_reserve"))
                and (row["workload_id"], row["rank"]) not in attempted
            ]
            reserve_pending.extend(available[:deficit])
        if not reserve_pending:
            raise RuntimeError(
                "legal reserve pool exhausted before every workload reached its target"
            )
        run_grouped(reserve_pending, f"reserve_{reserve_round}")

    successful_by_workload: dict[str, int] = {}
    for workload_id, _ in completed:
        successful_by_workload[workload_id] = successful_by_workload.get(workload_id, 0) + 1
    if (
        len(completed) != len(candidates)
        or any(successful_by_workload.get(key, 0) != value for key, value in required.items())
    ):
        raise RuntimeError(f"formal direct measurements incomplete: {len(completed)}/2185")
    official_by_id = {row["workload_id"]: row for row in official_rows}
    custom_rows: list[dict[str, str]] = []
    for workload in workloads:
        custom_rows.append(api_baseline(official_by_id[workload["workload_id"]]))
    custom_rows.extend(
        completed[(row["workload_id"], row["rank"])]
        for row in all_candidates
        if (row["workload_id"], row["rank"]) in completed
    )
    sample_rows: list[dict[str, str]] = []
    for row in all_candidates:
        key = (row["workload_id"], row["rank"])
        if key not in completed:
            continue
        for index, latency in enumerate(sample_map[key]):
            sample_rows.append({
                "workload_id": key[0], "rank": key[1], "candidate_role": "searched",
                "sample": str(index), "latency_ms": f"{latency:.12g}",
            })
    write_rows(args.profile_output, custom_rows, PROFILE_COLUMNS)
    write_rows(args.samples_output, sample_rows, SAMPLE_COLUMNS)
    log.append_once("campaign:complete", {
        "schema": SCHEMA, "record_type": "campaign_complete",
        "status": "complete", "candidate_records": 2185,
        "official_baselines": 70, "records": 2255,
    })
    log.close()
    print("DIRECT_MEASUREMENT_COMPLETE candidates=2185 baselines=70 records=2255", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            "DIRECT_MEASUREMENT_FATAL "
            + json.dumps({"error": str(error)}, separators=(",", ":")),
            file=sys.stderr, flush=True,
        )
        raise SystemExit(1)
