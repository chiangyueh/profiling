from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


SUCCESS_VALUES = {"1", "true", "True"}
SHAPE_FIELDS = ["m", "n", "k", "dtype", "trans_a", "trans_b"]
MAX_RELIABLE_MEASUREMENT_CV = 0.05
MAX_BASELINE_PAIR_RELATIVE_GAP = 0.15


def as_float(row: dict[str, str], key: str, fallback: float = float("inf")) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return fallback


def is_success(row: dict[str, str]) -> bool:
    return row.get("success") in SUCCESS_VALUES


def measurement_is_stable(row: dict[str, str]) -> bool:
    median_ms = as_float(row, "median_ms")
    stddev_ms = as_float(row, "stddev_ms")
    return (
        math.isfinite(median_ms)
        and math.isfinite(stddev_ms)
        and median_ms > 0
        and 0 <= stddev_ms / median_ms <= MAX_RELIABLE_MEASUREMENT_CV
    )


def baseline_pair_relative_gap(
    official: dict[str, str],
    bank: dict[str, str],
) -> float:
    official_ms = as_float(official, "median_ms")
    bank_ms = as_float(bank, "median_ms")
    if (
        not math.isfinite(official_ms)
        or not math.isfinite(bank_ms)
        or official_ms <= 0
        or bank_ms <= 0
    ):
        return float("inf")
    return max(official_ms, bank_ms) / min(official_ms, bank_ms) - 1.0


def baseline_pair_status(
    official: dict[str, str] | None,
    bank: dict[str, str] | None,
) -> str:
    if official is None or bank is None:
        return "unavailable"
    if not measurement_is_stable(official) or not measurement_is_stable(bank):
        return "unstable"
    if baseline_pair_relative_gap(official, bank) > MAX_BASELINE_PAIR_RELATIVE_GAP:
        return "incoherent"
    return "coherent"


def is_api_auto_baseline(row: dict[str, str]) -> bool:
    return (
        row.get("candidate_role") == "api_auto_baseline"
        or row.get("source") == "official_default"
    )


def is_searched(row: dict[str, str]) -> bool:
    return row.get("candidate_role") == "searched" and not is_api_auto_baseline(row)


def is_bank_seed_control(row: dict[str, str]) -> bool:
    return row.get("candidate_role") == "bank_seed_control"


def is_official_operator_baseline(row: dict[str, str]) -> bool:
    return (
        row.get("candidate_role") == "official_operator_baseline"
        and row.get("source") == "installed_aclnn_matmul"
    )


def read_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fields = set(reader.fieldnames or [])
        missing = sorted(required - fields)
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(missing)}")
        rows = list(reader)
    for line, row in enumerate(rows, 2):
        if None in row:
            raise ValueError(f"{path}:{line}: row has more fields than the header")
    return rows


def comparison_metrics(
    reference: dict[str, str], candidate: dict[str, str]
) -> tuple[float, float, float, str]:
    reference_ms = as_float(reference, "median_ms")
    candidate_ms = as_float(candidate, "median_ms")
    if not math.isfinite(reference_ms) or reference_ms <= 0:
        raise ValueError("reference baseline has invalid median_ms")
    if not math.isfinite(candidate_ms) or candidate_ms <= 0:
        raise ValueError("searched candidate has invalid median_ms")

    speedup = reference_ms / candidate_ms
    latency_change_pct = 100.0 * (candidate_ms - reference_ms) / reference_ms
    reference_stddev = max(0.0, as_float(reference, "stddev_ms", 0.0))
    candidate_stddev = max(0.0, as_float(candidate, "stddev_ms", 0.0))
    noise_threshold_pct = max(
        1.0,
        200.0
        * math.sqrt(reference_stddev**2 + candidate_stddev**2)
        / reference_ms,
    )
    if (
        reference_stddev / reference_ms > MAX_RELIABLE_MEASUREMENT_CV
        or candidate_stddev / candidate_ms > MAX_RELIABLE_MEASUREMENT_CV
    ):
        verdict = "unstable_measurement"
    elif latency_change_pct < -noise_threshold_pct:
        verdict = "improved"
    elif latency_change_pct > noise_threshold_pct:
        verdict = "regressed"
    else:
        verdict = "within_noise"
    return speedup, latency_change_pct, noise_threshold_pct, verdict


def optimization_decision(
    official_verdict: str,
    bank_verdict: str,
    baselines_coherent: bool = True,
) -> tuple[str, str, str]:
    if "unstable_measurement" in {official_verdict, bank_verdict}:
        return (
            "inconclusive",
            "measurement_variance_exceeds_reliability_limit",
            "unstable_measurement",
        )
    if not baselines_coherent:
        return (
            "inconclusive",
            "official_and_bank_baselines_diverge",
            "incoherent_baselines",
        )
    comparable = {"improved", "within_noise", "regressed"}
    if official_verdict == "improved" and bank_verdict == "improved":
        return (
            "improved",
            "measured_improvement_beats_official_and_bank_control",
            "improved",
        )
    if official_verdict not in comparable:
        return (
            "baseline_unavailable",
            "best_searched_candidate_without_baseline",
            official_verdict,
        )
    if bank_verdict not in comparable:
        return (
            "control_unavailable",
            "bank_seed_control_not_available",
            bank_verdict,
        )
    return (
        "not_improved",
        "no_searched_tiling_beats_official_and_bank_control_beyond_noise",
        "regressed"
        if "regressed" in {official_verdict, bank_verdict}
        else "within_noise",
    )


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def reference_failure_status(
    rows: list[dict[str, str]], unavailable: str, failed: str, unsupported: str
) -> str:
    if not rows:
        return unavailable
    errors = " ".join(row.get("error", "") for row in rows)
    if "unsupported_by_aclnnMatmul" in errors or "unsupported_by_MatMulV3" in errors:
        return unsupported
    return failed


def fill_ranked_reference(
    enriched: dict[str, str],
    row: dict[str, str],
    reference: dict[str, str] | None,
    prefix: str,
    unavailable_status: str,
    baseline_status: str | None = None,
) -> None:
    enriched[f"{prefix}_median_ms"] = (
        reference.get("median_ms", "") if reference is not None else ""
    )
    if reference is None:
        enriched[f"speedup_vs_{prefix}"] = ""
        enriched[f"latency_change_pct_vs_{prefix}"] = ""
        enriched[f"{prefix}_noise_threshold_pct"] = ""
        enriched[f"{prefix}_verdict"] = unavailable_status
        return
    if baseline_status is not None and row is reference:
        enriched[f"speedup_vs_{prefix}"] = "1"
        enriched[f"latency_change_pct_vs_{prefix}"] = "0"
        enriched[f"{prefix}_noise_threshold_pct"] = ""
        enriched[f"{prefix}_verdict"] = baseline_status
        return
    speedup, change, threshold, verdict = comparison_metrics(reference, row)
    enriched[f"speedup_vs_{prefix}"] = f"{speedup:.12g}"
    enriched[f"latency_change_pct_vs_{prefix}"] = f"{change:.12g}"
    enriched[f"{prefix}_noise_threshold_pct"] = f"{threshold:.12g}"
    enriched[f"{prefix}_verdict"] = verdict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/npu_profile.csv"))
    parser.add_argument("--official-input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/npu_ranked.csv"))
    parser.add_argument("--summary", type=Path, default=Path("results/npu_best_per_workload.csv"))
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path("results/npu_baseline_comparison.csv"),
    )
    args = parser.parse_args()

    required = {
        "workload_id",
        "rank",
        "source",
        "candidate_role",
        "success",
        "error",
        "median_ms",
        "stddev_ms",
        "tflops",
        *SHAPE_FIELDS,
    }
    custom_rows = read_rows(args.input, required)
    official_rows = (
        read_rows(args.official_input, required) if args.official_input is not None else []
    )

    custom_grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    official_grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in custom_rows:
        if row.get("candidate_role") not in {
            "api_auto_baseline", "bank_seed_control", "searched",
        }:
            raise ValueError(
                f"{args.input}: unsupported candidate_role={row.get('candidate_role')!r}"
            )
        custom_grouped[row["workload_id"]].append(row)
    for row in official_rows:
        if not is_official_operator_baseline(row):
            raise ValueError(
                f"{args.official_input}: official baseline identity contract failed"
            )
        official_grouped[row["workload_id"]].append(row)

    for workload_id in set(custom_grouped) & set(official_grouped):
        custom_shapes = {
            tuple(row.get(field, "") for field in SHAPE_FIELDS)
            for row in custom_grouped[workload_id]
        }
        official_shapes = {
            tuple(row.get(field, "") for field in SHAPE_FIELDS)
            for row in official_grouped[workload_id]
        }
        if len(custom_shapes) != 1 or official_shapes != custom_shapes:
            raise ValueError(
                f"{workload_id}: custom/official workload shape mismatch "
                f"(custom={sorted(custom_shapes)!r}, "
                f"official={sorted(official_shapes)!r})"
            )

    successful_custom = [row for row in custom_rows if is_success(row)]
    if not successful_custom:
        raise RuntimeError("no successful NPU profile rows")

    ranked: list[dict[str, str]] = []
    best: list[dict[str, str]] = []
    comparisons: list[dict[str, str]] = []
    workload_ids = sorted(set(custom_grouped) | set(official_grouped))
    for workload_id in workload_ids:
        custom_all = custom_grouped.get(workload_id, [])
        custom_success = sorted(
            (row for row in custom_all if is_success(row)),
            key=lambda row: as_float(row, "median_ms"),
        )
        api_all = [row for row in custom_all if is_api_auto_baseline(row)]
        api_rows = [row for row in api_all if is_success(row)]
        api_auto = (
            min(api_rows, key=lambda row: as_float(row, "median_ms"))
            if api_rows
            else None
        )
        bank_all = [
            row for row in custom_all if is_bank_seed_control(row)
        ]
        bank_rows = [row for row in bank_all if is_success(row)]
        bank_control = (
            min(bank_rows, key=lambda row: as_float(row, "median_ms"))
            if bank_rows
            else None
        )
        searched_rows = [row for row in custom_success if is_searched(row)]
        stable_searched_rows = [
            row for row in searched_rows if measurement_is_stable(row)
        ]
        best_searched = (
            min(
                stable_searched_rows or searched_rows,
                key=lambda row: as_float(row, "median_ms"),
            )
            if stable_searched_rows or searched_rows
            else None
        )
        official_all = official_grouped.get(workload_id, [])
        official_success = [row for row in official_all if is_success(row)]
        official_operator = (
            min(official_success, key=lambda row: as_float(row, "median_ms"))
            if official_success
            else None
        )
        pair_status = baseline_pair_status(official_operator, bank_control)
        pair_gap = (
            baseline_pair_relative_gap(official_operator, bank_control)
            if official_operator is not None and bank_control is not None
            else float("nan")
        )

        if custom_success:
            best_ms = as_float(custom_success[0], "median_ms")
            first_ranked: dict[str, str] | None = None
            official_status = reference_failure_status(
                official_all,
                "official_operator_not_measured",
                "official_operator_failed",
                "official_operator_unsupported",
            )
            for rank_value, row in enumerate(custom_success, 1):
                enriched = dict(row)
                enriched["npu_rank"] = str(rank_value)
                current = as_float(row, "median_ms")
                enriched["relative_to_best"] = (
                    f"{current / best_ms:.12g}" if best_ms > 0 else ""
                )
                fill_ranked_reference(
                    enriched,
                    row,
                    api_auto,
                    "api_auto",
                    "api_auto_failed" if api_all else "api_auto_unavailable",
                    "api_auto_baseline",
                )
                fill_ranked_reference(
                    enriched,
                    row,
                    official_operator,
                    "official_operator",
                    official_status,
                )
                fill_ranked_reference(
                    enriched,
                    row,
                    bank_control,
                    "bank_seed",
                    "bank_seed_failed"
                    if bank_all
                    else "bank_seed_unavailable",
                    "bank_seed_control",
                )
                ranked.append(enriched)
                if first_ranked is None:
                    first_ranked = enriched
            if first_ranked is not None:
                best.append(first_ranked)

        shape_source = (
            custom_all[0]
            if custom_all
            else official_all[0]
            if official_all
            else {}
        )
        comparison = {
            "workload_id": workload_id,
            **{field: shape_source.get(field, "") for field in SHAPE_FIELDS},
            "best_searched_rank": best_searched.get("rank", "") if best_searched else "",
            "best_searched_source": best_searched.get("source", "") if best_searched else "",
            "best_searched_median_ms": (
                best_searched.get("median_ms", "") if best_searched else ""
            ),
            "best_searched_stddev_ms": (
                best_searched.get("stddev_ms", "") if best_searched else ""
            ),
            "best_searched_tflops": (
                best_searched.get("tflops", "") if best_searched else ""
            ),
            "api_auto_rank": api_auto.get("rank", "") if api_auto else "",
            "api_auto_source": (
                api_auto.get("source", "")
                if api_auto
                else api_all[0].get("source", "")
                if api_all
                else ""
            ),
            "api_auto_success": "1" if api_auto else "0",
            "api_auto_error": (
                ""
                if api_auto
                else api_all[0].get("error", "")
                if api_all
                else ""
            ),
            "api_auto_median_ms": api_auto.get("median_ms", "") if api_auto else "",
            "api_auto_stddev_ms": api_auto.get("stddev_ms", "") if api_auto else "",
            "api_auto_tflops": api_auto.get("tflops", "") if api_auto else "",
            "speedup_vs_api_auto": "",
            "latency_change_pct_vs_api_auto": "",
            "api_auto_noise_threshold_pct": "",
            "api_auto_verdict": (
                "api_auto_failed"
                if api_auto is None and api_all
                else "api_auto_unavailable"
                if api_auto is None
                else "no_searched_candidate"
            ),
            "bank_seed_source": (
                bank_control.get("source", "")
                if bank_control
                else bank_all[0].get("source", "")
                if bank_all
                else ""
            ),
            "bank_seed_success": "1" if bank_control else "0",
            "bank_seed_error": (
                ""
                if bank_control
                else bank_all[0].get("error", "")
                if bank_all
                else ""
            ),
            "bank_seed_median_ms": (
                bank_control.get("median_ms", "") if bank_control else ""
            ),
            "bank_seed_stddev_ms": (
                bank_control.get("stddev_ms", "") if bank_control else ""
            ),
            "speedup_vs_bank_seed": "",
            "latency_change_pct_vs_bank_seed": "",
            "bank_seed_noise_threshold_pct": "",
            "bank_seed_verdict": (
                "bank_seed_failed"
                if bank_control is None and bank_all
                else "bank_seed_unavailable"
                if bank_control is None
                else "no_searched_candidate"
            ),
            "official_operator_source": (
                official_operator.get("source", "")
                if official_operator
                else official_all[0].get("source", "")
                if official_all
                else ""
            ),
            "official_operator_success": "1" if official_operator else "0",
            "official_operator_error": (
                ""
                if official_operator
                else official_all[0].get("error", "")
                if official_all
                else ""
            ),
            "official_operator_median_ms": (
                official_operator.get("median_ms", "") if official_operator else ""
            ),
            "official_operator_stddev_ms": (
                official_operator.get("stddev_ms", "") if official_operator else ""
            ),
            "official_operator_tflops": (
                official_operator.get("tflops", "") if official_operator else ""
            ),
            "speedup_vs_official_operator": "",
            "latency_change_pct_vs_official_operator": "",
            "official_operator_noise_threshold_pct": "",
            "official_operator_verdict": reference_failure_status(
                official_all,
                "official_operator_not_measured",
                "official_operator_failed",
                "official_operator_unsupported",
            )
            if official_operator is None
            else "no_searched_candidate",
            "baseline_pair_status": pair_status,
            "official_bank_relative_gap_pct": (
                f"{100 * pair_gap:.12g}" if math.isfinite(pair_gap) else ""
            ),
            "primary_reference": "",
            "primary_verdict": (
                "no_successful_searched_candidate"
                if best_searched is None
                else "baseline_unavailable"
            ),
            "selected_source": "",
            "selected_rank": "",
            "selected_median_ms": "",
            "selection_reason": "",
            "optimization_result": (
                "no_successful_searched_candidate"
                if best_searched is None
                else "baseline_unavailable"
            ),
        }
        if api_auto is not None and best_searched is not None:
            speedup, change, threshold, verdict = comparison_metrics(api_auto, best_searched)
            comparison.update(
                {
                    "speedup_vs_api_auto": f"{speedup:.12g}",
                    "latency_change_pct_vs_api_auto": f"{change:.12g}",
                    "api_auto_noise_threshold_pct": f"{threshold:.12g}",
                    "api_auto_verdict": verdict,
                }
            )
        if bank_control is not None and best_searched is not None:
            speedup, change, threshold, verdict = comparison_metrics(
                bank_control, best_searched
            )
            comparison.update(
                {
                    "speedup_vs_bank_seed": f"{speedup:.12g}",
                    "latency_change_pct_vs_bank_seed": f"{change:.12g}",
                    "bank_seed_noise_threshold_pct": f"{threshold:.12g}",
                    "bank_seed_verdict": verdict,
                }
            )
        if official_operator is not None and best_searched is not None:
            speedup, change, threshold, verdict = comparison_metrics(
                official_operator, best_searched
            )
            comparison.update(
                {
                    "speedup_vs_official_operator": f"{speedup:.12g}",
                    "latency_change_pct_vs_official_operator": f"{change:.12g}",
                    "official_operator_noise_threshold_pct": f"{threshold:.12g}",
                    "official_operator_verdict": verdict,
                    "primary_reference": "official_operator",
                    "primary_verdict": verdict,
                }
            )
        elif api_auto is not None and best_searched is not None:
            comparison["primary_reference"] = "api_auto"
            comparison["primary_verdict"] = comparison["api_auto_verdict"]

        if best_searched is not None:
            official_verdict = comparison["primary_verdict"]
            bank_verdict = comparison["bank_seed_verdict"]
            (
                optimization_result,
                selection_reason,
                combined_verdict,
            ) = optimization_decision(
                official_verdict,
                bank_verdict,
                pair_status != "incoherent",
            )
            if optimization_result != "baseline_unavailable":
                comparison["primary_reference"] = (
                    "official_operator+bank_seed_control"
                )
                comparison["primary_verdict"] = combined_verdict
            comparison.update(
                {
                    "selected_source": (
                        "searched" if optimization_result == "improved" else ""
                    ),
                    "selected_rank": (
                        best_searched.get("rank", "")
                        if optimization_result == "improved"
                        else ""
                    ),
                    "selected_median_ms": (
                        best_searched.get("median_ms", "")
                        if optimization_result == "improved"
                        else ""
                    ),
                    "selection_reason": (
                        selection_reason
                    ),
                    "optimization_result": optimization_result,
                }
            )
        else:
            comparison["selection_reason"] = "no_successful_searched_candidate"
            comparison["optimization_result"] = "no_successful_searched_candidate"
        comparisons.append(comparison)

    added_fields = [
        "npu_rank",
        "relative_to_best",
        "api_auto_median_ms",
        "speedup_vs_api_auto",
        "latency_change_pct_vs_api_auto",
        "api_auto_noise_threshold_pct",
        "api_auto_verdict",
        "official_operator_median_ms",
        "speedup_vs_official_operator",
        "latency_change_pct_vs_official_operator",
        "official_operator_noise_threshold_pct",
        "official_operator_verdict",
        "bank_seed_median_ms",
        "speedup_vs_bank_seed",
        "latency_change_pct_vs_bank_seed",
        "bank_seed_noise_threshold_pct",
        "bank_seed_verdict",
    ]
    added_set = set(added_fields)
    ranked_fields = added_fields + [
        key for key in ranked[0] if key not in added_set
    ]
    write_rows(args.output, ranked, ranked_fields)
    write_rows(args.summary, best, ranked_fields)
    write_rows(args.comparison, comparisons, list(comparisons[0]))

    print(f"ranked={args.output}")
    print(f"best={args.summary}")
    print(f"baseline_comparison={args.comparison}")


if __name__ == "__main__":
    main()
