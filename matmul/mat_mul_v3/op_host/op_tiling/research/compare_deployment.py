#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


COLUMNS = [
    "workload_id",
    "m",
    "n",
    "k",
    "dtype",
    "trans_a",
    "trans_b",
    "model_template",
    "model_signature",
    "model_host_history_load_ms",
    "model_host_model_setup_ms",
    "model_host_tiling_ms",
    "model_npu_ms",
    "model_speedup_vs_official",
    "model_speedup_vs_bank",
    "model_status_vs_official",
    "model_status_vs_bank",
    "base_template",
    "base_signature",
    "base_host_history_load_ms",
    "base_host_model_setup_ms",
    "base_host_tiling_ms",
    "base_npu_ms",
    "base_speedup_vs_official",
    "base_speedup_vs_bank",
    "base_status_vs_official",
    "base_status_vs_bank",
    "preferred_strategy",
]


def read_measurements(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as source:
        rows = csv.DictReader(source)
        return {
            row["workload_id"]: row
            for row in rows
            if row.get("candidate_role") == "searched"
        }


def trusted(row: dict[str, str] | None) -> bool:
    return bool(
        row
        and row.get("success") == "1"
        and row.get("pair_validated") == "1"
        and row.get("speedup_vs_official")
        and row.get("speedup_vs_bank")
    )


def worst_ratio(row: dict[str, str]) -> float:
    return max(
        1.0 / float(row["speedup_vs_official"]),
        1.0 / float(row["speedup_vs_bank"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = read_measurements(args.model)
    base = read_measurements(args.base)
    rows = []
    counts = {"model": 0, "base": 0, "tie": 0, "unavailable": 0}
    for workload_id in sorted(set(model) | set(base)):
        model_row = model.get(workload_id)
        base_row = base.get(workload_id)
        source = model_row or base_row or {}
        if trusted(model_row) and trusted(base_row):
            model_ratio = worst_ratio(model_row)
            base_ratio = worst_ratio(base_row)
            if abs(model_ratio - base_ratio) <= 0.005:
                preferred = "tie_within_0.5pct"
                counts["tie"] += 1
            elif model_ratio < base_ratio:
                preferred = "compact_data_driven"
                counts["model"] += 1
            else:
                preferred = "direct_base_policy"
                counts["base"] += 1
        elif trusted(model_row):
            preferred = "compact_data_driven_only_trusted"
            counts["model"] += 1
        elif trusted(base_row):
            preferred = "direct_base_policy_only_trusted"
            counts["base"] += 1
        else:
            preferred = "no_trusted_pair"
            counts["unavailable"] += 1

        row = {column: "" for column in COLUMNS}
        for column in (
            "workload_id",
            "m",
            "n",
            "k",
            "dtype",
            "trans_a",
            "trans_b",
        ):
            row[column] = source.get(column, "")
        for prefix, item in (("model", model_row), ("base", base_row)):
            if item is None:
                continue
            row.update(
                {
                    f"{prefix}_template": item.get(
                        "search_template", ""
                    ),
                    f"{prefix}_signature": item.get(
                        "tiling_signature", ""
                    ),
                    f"{prefix}_host_history_load_ms": item.get(
                        "host_history_load_ms", ""
                    ),
                    f"{prefix}_host_model_setup_ms": item.get(
                        "host_model_setup_ms", ""
                    ),
                    f"{prefix}_host_tiling_ms": item.get(
                        "host_tiling_total_ms", ""
                    ),
                    f"{prefix}_npu_ms": item.get("median_ms", ""),
                    f"{prefix}_speedup_vs_official": item.get(
                        "speedup_vs_official", ""
                    ),
                    f"{prefix}_speedup_vs_bank": item.get(
                        "speedup_vs_bank", ""
                    ),
                    f"{prefix}_status_vs_official": item.get(
                        "status_vs_official", ""
                    ),
                    f"{prefix}_status_vs_bank": item.get(
                        "status_vs_bank", ""
                    ),
                }
            )
        row["preferred_strategy"] = preferred
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(
        "DUAL_COMPARISON "
        f"workloads={len(rows)} "
        f"model_preferred={counts['model']} "
        f"base_preferred={counts['base']} "
        f"ties={counts['tie']} "
        f"unavailable={counts['unavailable']} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
