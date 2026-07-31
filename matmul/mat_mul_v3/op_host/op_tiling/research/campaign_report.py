#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

from generate import load_resume_feedback, load_workloads
from tiling_search.contracts import template_of
from tiling_search.domain import MeasuredObservation, Workload
from tiling_search.feedback import load_feedback


COLUMNS = (
    "workload_id",
    "m",
    "n",
    "k",
    "dtype",
    "trans_a",
    "trans_b",
    "campaign_status",
    "measurements",
    "runtime_rejected",
    "winning_measurements",
    "best_template",
    "best_source",
    "best_signature",
    "speedup_vs_official",
    "speedup_vs_bank",
    "record_id",
)


def summarize_campaign(
    workloads: Sequence[Workload],
    observations: Sequence[MeasuredObservation],
) -> list[dict[str, str]]:
    rows = []
    for workload in workloads:
        exact = [
            observation
            for observation in observations
            if observation.workload.identity() == workload.identity()
        ]
        rejected = [
            observation
            for observation in exact
            if observation.source == "runtime_rejected"
        ]
        measured = [
            observation
            for observation in exact
            if observation.source != "runtime_rejected"
        ]
        winners = [
            observation
            for observation in measured
            if observation.is_winner
        ]
        eligible = winners or measured
        best = (
            min(
                eligible,
                key=lambda observation: (
                    observation.measured_ratio,
                    observation.schedule.signature(),
                ),
            )
            if eligible
            else None
        )
        row = {column: "" for column in COLUMNS}
        row.update(
            {
                "workload_id": workload.workload_id,
                "m": str(workload.m),
                "n": str(workload.n),
                "k": str(workload.k),
                "dtype": workload.dtype,
                "trans_a": str(int(workload.trans_a)),
                "trans_b": str(int(workload.trans_b)),
                "campaign_status": "solved" if winners else "open",
                "measurements": str(len(measured)),
                "runtime_rejected": str(len(rejected)),
                "winning_measurements": str(len(winners)),
            }
        )
        if best is not None:
            row.update(
                {
                    "best_template": template_of(best.schedule).value,
                    "best_source": best.source,
                    "best_signature": best.schedule.signature_text(),
                    "speedup_vs_official": (
                        f"{1.0 / best.ratio_vs_official:.12g}"
                    ),
                    "speedup_vs_bank": (
                        f"{1.0 / best.ratio_vs_bank:.12g}"
                    ),
                    "record_id": best.record_id,
                }
            )
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--observations", type=Path, action="append", default=[])
    parser.add_argument("--soc", required=True)
    parser.add_argument("--aic", type=int, required=True)
    parser.add_argument("--toolkit", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workloads = load_workloads(args.workloads, args.aic)
    observations, _ = load_feedback(
        soc=args.soc,
        aic_cores=args.aic,
        observation_paths=args.observations,
    )
    resume_observations, _ = load_resume_feedback(
        args.resume,
        args.soc,
        args.aic,
        args.toolkit,
    )
    observations.extend(resume_observations)
    rows = summarize_campaign(workloads, observations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        writer = csv.DictWriter(destination, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print("CAMPAIGN_RESULT_BEGIN")
    for row in rows:
        print(
            f"CAMPAIGN_WORKLOAD {row['workload_id']} "
            f"status={row['campaign_status']} "
            f"measurements={row['measurements']} "
            f"runtime_rejected={row['runtime_rejected']} "
            f"winners={row['winning_measurements']} "
            f"best_template={row['best_template'] or 'NA'} "
            f"speedup={row['speedup_vs_official'] or 'NA'} "
            f"speedup_vs_bank={row['speedup_vs_bank'] or 'NA'}"
        )
    solved = sum(row["campaign_status"] == "solved" for row in rows)
    print(
        f"CAMPAIGN_TOTAL workloads={len(rows)} solved={solved} "
        f"open={len(rows) - solved} "
        f"measurements={sum(int(row['measurements']) for row in rows)} "
        f"runtime_rejected="
        f"{sum(int(row['runtime_rejected']) for row in rows)}"
    )
    print("CAMPAIGN_RESULT_END")


if __name__ == "__main__":
    main()
