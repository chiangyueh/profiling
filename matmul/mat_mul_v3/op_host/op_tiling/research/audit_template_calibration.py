#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from tiling_search.calibration_workloads import decode_template_quotas
from tiling_search.contracts import template_of
from tiling_search.domain import Schedule, Template
from tiling_search.feedback import STRICT_NUMERIC_PREFLIGHT_MODES


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument(
        "--evidence",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--soc", required=True)
    parser.add_argument("--aic", type=int, required=True)
    parser.add_argument("--toolkit", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.workloads.open(newline="", encoding="utf-8") as source:
        workload_rows = list(csv.DictReader(source))
    targets = {
        row.get("workload_id") or row["id"]: decode_template_quotas(
            row.get("template_quotas", "")
        )
        for row in workload_rows
        if row.get("template_quotas")
    }
    workload_identities = {
        row.get("workload_id") or row["id"]: (
            int(row["m"]),
            int(row["n"]),
            int(row["k"]),
            row["dtype"],
            truthy(row.get("trans_a")),
            truthy(row.get("trans_b")),
        )
        for row in workload_rows
        if row.get("template_quotas")
    }
    rows = []
    for path in [*args.evidence, args.resume]:
        if path.is_file():
            with path.open(newline="", encoding="utf-8") as source:
                rows.extend(csv.DictReader(source))

    states: dict[tuple[str, Template, tuple[int, ...]], str] = {}
    priority = {"unpaired": 1, "runtime_rejected": 2, "paired": 3}
    for row in rows:
        if (
            row.get("candidate_role") != "searched"
            or row.get("soc") != args.soc
            or int(row.get("aic") or 0) != args.aic
            or row.get("toolkit") != args.toolkit
        ):
            continue
        try:
            schedule = Schedule.from_signature(row["tiling_signature"])
            template = template_of(schedule)
        except (KeyError, ValueError):
            continue
        workload_id = row.get("workload_id", "")
        identity = workload_identities.get(workload_id)
        try:
            row_identity = (
                int(row.get("m") or -1),
                int(row.get("n") or -1),
                int(row.get("k") or -1),
                row.get("dtype", ""),
                truthy(row.get("trans_a")),
                truthy(row.get("trans_b")),
            )
        except ValueError:
            continue
        if (
            workload_id not in targets
            or template not in targets[workload_id]
            or row_identity != identity
        ):
            continue
        key = (workload_id, template, schedule.signature())
        if (
            truthy(row.get("success"))
            and truthy(row.get("pair_validated"))
            and row.get("preflight_mode")
            in STRICT_NUMERIC_PREFLIGHT_MODES
            and row.get("official_ms")
            and row.get("bank_ms")
        ):
            state = "paired"
        elif (
            not truthy(row.get("success"))
            and row.get("preflight_mode")
            not in {
                "",
                "baseline_drift",
                "provisional",
                "runner_failed",
            }
        ):
            state = "runtime_rejected"
        elif (
            truthy(row.get("success"))
            and row.get("preflight_mode")
            in STRICT_NUMERIC_PREFLIGHT_MODES
        ):
            state = "unpaired"
        else:
            continue
        previous = states.get(key)
        if previous is None or priority[state] > priority[previous]:
            states[key] = state

    paired: Counter[tuple[str, Template]] = Counter()
    runtime_rejected: Counter[tuple[str, Template]] = Counter()
    unpaired: Counter[tuple[str, Template]] = Counter()
    for (workload_id, template, _), state in states.items():
        target = (
            paired
            if state == "paired"
            else runtime_rejected
            if state == "runtime_rejected"
            else unpaired
        )
        target[(workload_id, template)] += 1

    gaps = []
    totals: Counter[Template] = Counter()
    required_totals: Counter[Template] = Counter()
    for workload_id, quotas in targets.items():
        for template, required in sorted(
            quotas.items(), key=lambda item: item[0].value
        ):
            measured = paired[(workload_id, template)]
            rejected = runtime_rejected[(workload_id, template)]
            drifted = unpaired[(workload_id, template)]
            totals[template] += measured
            required_totals[template] += required
            status = "complete" if measured >= required else "gap"
            print(
                "TEMPLATE_CALIBRATION_WORKLOAD "
                f"workload={workload_id} "
                f"template={template.value} "
                f"paired={measured} required={required} "
                f"runtime_rejected={rejected} unpaired={drifted} "
                f"status={status}"
            )
            if measured < required:
                gaps.append(
                    f"{workload_id}:{template.value}:"
                    f"{measured}/{required}"
                )
    for template in sorted(required_totals, key=lambda item: item.value):
        print(
            "TEMPLATE_CALIBRATION_TOTAL "
            f"template={template.value} "
            f"paired={totals[template]} "
            f"required={required_totals[template]}"
        )
    if gaps:
        print(
            "TEMPLATE_CALIBRATION_AUDIT status=retryable "
            f"gaps={len(gaps)} "
            f"runtime_rejected={sum(runtime_rejected.values())} "
            f"unpaired={sum(unpaired.values())}"
        )
        print("TEMPLATE_CALIBRATION_GAPS " + ",".join(gaps))
        raise SystemExit(3)
    print(
        "TEMPLATE_CALIBRATION_AUDIT status=passed "
        f"workloads={len(targets)} "
        f"target_templates={len(required_totals)}"
    )


if __name__ == "__main__":
    main()
