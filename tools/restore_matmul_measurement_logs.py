#!/usr/bin/env python3
"""Restore resumable measurement CSVs from crash-safe JSONL logs."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path

from profile_official_tilings import PROFILE_COLUMNS, SAMPLE_COLUMNS


SUCCESS_RECORDS = {"official_baseline", "candidate_measurement"}


def atomic_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", delete=False,
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(stream.name)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-directory", type=Path, required=True)
    parser.add_argument("--profile-output", type=Path, required=True)
    parser.add_argument("--samples-output", type=Path, required=True)
    args = parser.parse_args()

    profiles: dict[tuple[str, str, str], dict] = {}
    samples: dict[tuple[str, str, str, int], dict] = {}
    paths = sorted(
        (path for path in args.log_directory.glob("[0-9]*.log") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("record_type") not in SUCCESS_RECORDS:
                    continue
                profile = record.get("measurement")
                if not isinstance(profile, dict) or str(profile.get("success", "")).lower() not in {
                    "1", "true", "yes", "on",
                }:
                    continue
                workload_id = str(profile.get("workload_id", ""))
                role = str(profile.get("candidate_role", ""))
                rank = str(profile.get("rank", ""))
                if not workload_id or not role or not rank:
                    continue
                key = (workload_id, role, rank)
                profiles[key] = {field: profile.get(field, "") for field in PROFILE_COLUMNS}
                for sample_index, latency_ms in enumerate(record.get("samples_ms") or []):
                    try:
                        latency = float(latency_ms)
                    except (TypeError, ValueError):
                        continue
                    samples[(*key, sample_index)] = {
                        "workload_id": workload_id,
                        "rank": rank,
                        "candidate_role": role,
                        "sample": str(sample_index),
                        "latency_ms": f"{latency:.12g}",
                    }

    atomic_csv(args.profile_output, PROFILE_COLUMNS, list(profiles.values()))
    atomic_csv(args.samples_output, SAMPLE_COLUMNS, list(samples.values()))
    print(f"JSONL_RESUME_RESTORED profiles={len(profiles)} samples={len(samples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
