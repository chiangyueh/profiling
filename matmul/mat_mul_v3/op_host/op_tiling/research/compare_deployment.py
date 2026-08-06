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
    "rule_template",
    "rule_signature",
    "rule_host_history_load_ms",
    "rule_host_model_setup_ms",
    "rule_host_tiling_ms",
    "rule_npu_ms",
    "rule_speedup_vs_official",
    "rule_speedup_vs_bank",
    "rule_status_vs_official",
    "rule_status_vs_bank",
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
    parser.add_argument("--rule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = read_measurements(args.model)
    rule = read_measurements(args.rule)
    rows = []
    counts = {"model": 0, "rule": 0, "tie": 0, "unavailable": 0}
    for workload_id in sorted(set(model) | set(rule)):
        model_row = model.get(workload_id)
        rule_row = rule.get(workload_id)
        source = model_row or rule_row or {}
        if trusted(model_row) and trusted(rule_row):
            model_ratio = worst_ratio(model_row)
            rule_ratio = worst_ratio(rule_row)
            if abs(model_ratio - rule_ratio) <= 0.005:
                preferred = "tie_within_0.5pct"
                counts["tie"] += 1
            elif model_ratio < rule_ratio:
                preferred = "compact_data_driven"
                counts["model"] += 1
            else:
                preferred = "direct_rule_base"
                counts["rule"] += 1
        elif trusted(model_row):
            preferred = "compact_data_driven_only_trusted"
            counts["model"] += 1
        elif trusted(rule_row):
            preferred = "direct_rule_base_only_trusted"
            counts["rule"] += 1
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
        for prefix, item in (("model", model_row), ("rule", rule_row)):
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
        f"rule_preferred={counts['rule']} "
        f"ties={counts['tie']} "
        f"unavailable={counts['unavailable']} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
