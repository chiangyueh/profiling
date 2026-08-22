from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter
from pathlib import Path


PRIOR_REGRESSION_WORKLOADS = {
    "llm_full_4096_ffn_up",
    "llm_full_4096_ffn_down",
    "llm_8192_square",
    "attention_score_1024",
    "vision_projection",
    "skinny_m_large_k",
}

DET_SPLIT_K_POSITIVE_RANGE = {
    "large_k_small_mn",
    "det_split_k_aligned_holdout_k16384",
    "det_split_k_aligned_holdout_k49152",
}


def evidence_group(workload_id: str) -> str:
    if workload_id == "int8_projection":
        return "unsupported_control"
    if workload_id in DET_SPLIT_K_POSITIVE_RANGE:
        return "det_split_k_positive_range"
    if workload_id == "det_split_k_aligned_holdout_k65536":
        return "det_split_k_rejected_control"
    if workload_id == "deterministic_split_k_unaligned":
        return "alignment_negative_control"
    if workload_id == "skinny_n_large_k":
        return "known_anchor"
    if workload_id.startswith("skinny_n_k16384_holdout_"):
        return "skinny_n_k16384_holdout"
    if workload_id.startswith("skinny_n_boundary_holdout_"):
        return "skinny_n_boundary_holdout"
    if workload_id.startswith("skinny_n_boundary64_holdout_"):
        return "skinny_n_boundary64_holdout"
    if workload_id.startswith("skinny_n_holdout_"):
        return "skinny_n_initial_holdout"
    if workload_id.startswith("skinny_n_boundary_"):
        return "boundary_control"
    if workload_id in PRIOR_REGRESSION_WORKLOADS:
        return "prior_regression"
    return "broad_validation"


def strict_evidence_status(
    results: Counter[str] | None,
    minimum_workloads: int,
) -> str:
    if not results or sum(results.values()) < minimum_workloads:
        return "insufficient_evidence"
    if results.get("improved", 0) == sum(results.values()):
        return "supported"
    return "not_supported"


def campaign_statuses(
    evidence_results: dict[str, Counter[str]],
) -> dict[str, str]:
    skinny_initial = strict_evidence_status(
        evidence_results.get("skinny_n_initial_holdout"), 3
    )
    skinny_k16384 = strict_evidence_status(
        evidence_results.get("skinny_n_k16384_holdout"), 3
    )
    skinny_boundary = strict_evidence_status(
        evidence_results.get("skinny_n_boundary_holdout"), 3
    )
    skinny_boundary64 = strict_evidence_status(
        evidence_results.get("skinny_n_boundary64_holdout"), 3
    )
    det_split_k = strict_evidence_status(
        evidence_results.get("det_split_k_positive_range"), 3
    )
    prior_failures = strict_evidence_status(
        evidence_results.get("prior_regression"), 6
    )
    broad_validation = strict_evidence_status(
        evidence_results.get("broad_validation"), 26
    )
    wide = (
        "supported"
        if skinny_k16384 == "supported"
        and det_split_k == "supported"
        and prior_failures == "supported"
        and broad_validation == "supported"
        else "not_proven"
    )
    return {
        "skinny_initial": skinny_initial,
        "skinny_k16384": skinny_k16384,
        "skinny_boundary": skinny_boundary,
        "skinny_boundary64": skinny_boundary64,
        "det_split_k": det_split_k,
        "prior_failures": prior_failures,
        "broad_validation": broad_validation,
        "wide": wide,
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
    if not reader.fieldnames:
        raise ValueError(f"{path}: missing CSV header")
    return rows


def compact_number(value: str) -> str:
    if not value:
        return "NA"
    try:
        return f"{float(value):.6g}"
    except ValueError:
        return value


def clean(value: str) -> str:
    return value.replace(" ", "_") if value else "NA"


def finite_number(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print a compact, copyable report from an NPU MatMul run."
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument(
        "--all-workloads",
        action="store_true",
        help="print every workload instead of only analysis-relevant rows",
    )
    parser.add_argument(
        "--direct-comparison-only",
        action="store_true",
        help="print only this campaign's solver-versus-official comparison",
    )
    args = parser.parse_args()

    summaries = read_csv(args.summary)
    candidates = read_csv(args.candidates)
    candidate_by_key = {
        (row.get("workload_id", ""), row.get("rank", "")): row
        for row in candidates
        if row.get("candidate_role") == "searched"
    }

    verdicts: Counter[str] = Counter()
    print("NPU_RESULT_BEGIN")
    print("LEGEND npu_ms=solver/original_matmul_v3; speedup=original/solver (>1 is faster)")
    print("LEGEND tiling_cpu_ms is host wall time: original callback versus the full solver path")
    print("LEGEND solver_total=original callback + RuntimeKb seed + model selection + selected-candidate callback")
    print("LEGEND tpl=MatMulV3 template T=baseMxbaseNxbaseK S=kernel_single_MxN")
    print("LEGEND C=selected cores (cap is the tested solver core limit) G=output_block_grid")
    print("LEGEND L2=tile_count_MxN(blocks_per_tile_MxN)")
    print("LEGEND verdict uses the measured noise threshold; model internals remain in CSV only")
    total = len(summaries)
    width = max(2, len(str(total)))
    optimization_results: Counter[str] = Counter()
    evidence_results: dict[str, Counter[str]] = {}
    printed_workloads = 0
    callback_layout_changed = 0
    callback_layout_unchanged = 0
    tiling_times: dict[str, list[float]] = {
        "original_callback": [],
        "runtime_kb_seed": [],
        "solver_select": [],
        "solver_callback": [],
        "solver_total": [],
    }
    for index, summary in enumerate(summaries, 1):
        workload_id = summary.get("workload_id", "")
        evidence = evidence_group(workload_id)
        searched_rank = summary.get("best_searched_rank", "")
        candidate = candidate_by_key.get((workload_id, searched_rank), {})
        if candidate:
            if candidate.get("callback_derived_diff_vs_default", "").strip():
                callback_layout_changed += 1
            else:
                callback_layout_unchanged += 1
        for key, column in (
            ("original_callback", "tiling_official_callback_ms"),
            ("runtime_kb_seed", "tiling_runtime_kb_seed_ms"),
            ("solver_select", "tiling_solver_select_ms"),
            ("solver_callback", "tiling_solver_callback_ms"),
            ("solver_total", "tiling_solver_total_ms"),
        ):
            value = finite_number(candidate.get(column, ""))
            if value is not None:
                tiling_times[key].append(value)
        verdict = summary.get("primary_verdict", "") or "unknown"
        verdicts[verdict] += 1
        optimization = (
            summary.get("optimization_result", "")
            or "no_successful_searched_candidate"
        )
        optimization_results[optimization] += 1
        evidence_results.setdefault(evidence, Counter())[optimization] += 1

        should_print = (
            args.all_workloads
            or optimization == "improved"
            or verdict == "regressed"
            or candidate.get("search_resume_policy") == "allow_new"
        )
        if not should_print:
            continue
        printed_workloads += 1

        shape = "x".join(
            summary.get(field, "") or "?" for field in ("m", "n", "k")
        )
        tiling = "x".join(
            candidate.get(field, "") or "?"
            for field in ("official_base_m", "official_base_n", "official_base_k")
        )
        core_dims = "x".join(
            candidate.get(field, "") or "?"
            for field in ("m_base_blocks", "n_base_blocks")
        )
        single = "x".join(
            candidate.get(field, "") or "?"
            for field in ("kernel_single_core_m", "kernel_single_core_n")
        )
        l2_count = "x".join(
            candidate.get(field, "") or "?"
            for field in ("l2_m_tile_count", "l2_n_tile_count")
        )
        l2_block = "x".join(
            candidate.get(field, "") or "?"
            for field in ("l2_m_tile_block", "l2_n_tile_block")
        )
        print(
            f"[{index:0{width}d}/{total:0{width}d}] {clean(workload_id)} "
            f"verdict={clean(verdict)} optimization={clean(optimization)}"
        )
        print(
            f"  shape={shape} dtype={clean(summary.get('dtype', ''))}"
            f" trans={clean(summary.get('trans_a', ''))}{clean(summary.get('trans_b', ''))}"
            f" core_cap={clean(candidate.get('max_cores', ''))}"
            f" tpl={clean(candidate.get('kernel_template', ''))}"
            f" T={tiling}"
            f" S={single}"
            f" C={clean(candidate.get('used_core_num', ''))}"
            f" G={core_dims}"
            f" L2={l2_count}({l2_block})"
        )
        print(
            "  tiling_cpu_ms "
            f"original_callback={compact_number(candidate.get('tiling_official_callback_ms', ''))} "
            f"runtime_kb_seed={compact_number(candidate.get('tiling_runtime_kb_seed_ms', ''))} "
            f"solver_select={compact_number(candidate.get('tiling_solver_select_ms', ''))} "
            f"solver_callback={compact_number(candidate.get('tiling_solver_callback_ms', ''))} "
            f"callback_count={clean(candidate.get('tiling_solver_callback_count', ''))} "
            f"solver_total={compact_number(candidate.get('tiling_solver_total_ms', ''))}"
        )
        print(
            "  npu_ms solver="
            f"{compact_number(summary.get('best_searched_median_ms', ''))}/"
            f"{compact_number(summary.get('official_operator_median_ms', ''))}"
            " std="
            f"{compact_number(summary.get('best_searched_stddev_ms', ''))}/"
            f"{compact_number(summary.get('official_operator_stddev_ms', ''))}"
            f" speedup={compact_number(summary.get('speedup_vs_official_operator', ''))}"
            f" delta={compact_number(summary.get('latency_change_pct_vs_official_operator', ''))}%"
            f" noise={compact_number(summary.get('official_operator_noise_threshold_pct', ''))}%"
        )
        cause = "/".join(
            value
            for value in (
                clean(candidate.get("search_bottleneck", "")),
                clean(candidate.get("search_guidance", "")),
            )
            if value != "NA"
        )
        if cause:
            print(f"  selected_by={cause}")

    ordered = ("improved", "within_noise", "regressed")
    totals = " ".join(f"{key}={verdicts.pop(key, 0)}" for key in ordered)
    other = sum(verdicts.values())
    optimized = optimization_results.pop("improved", 0)
    not_improved = optimization_results.pop("not_improved", 0)
    no_candidate = optimization_results.pop(
        "no_successful_searched_candidate", 0
    )
    optimization_other = sum(optimization_results.values())
    if not args.all_workloads:
        print(
            f"PRINTED_WORKLOADS shown={printed_workloads} total={len(summaries)} "
            "policy=improved_or_regressed_or_new_allow_new; "
            "set_PRINT_ALL_RESULTS=1_for_every_workload"
        )
    print(
        f"RESULT_TOTAL workloads={len(summaries)} {totals} other={other} "
        f"optimization_success={optimized} "
        f"optimization_not_improved={not_improved} "
        f"no_searched_candidate={no_candidate} "
        f"optimization_other={optimization_other}"
    )
    print(
        "CALLBACK_LAYOUT_TOTAL "
        f"selected={callback_layout_changed + callback_layout_unchanged} "
        f"derived_layout_changed={callback_layout_changed} "
        f"derived_layout_unchanged={callback_layout_unchanged} "
        f"no_selected_candidate={no_candidate}"
    )
    if tiling_times["solver_total"]:
        mean = {
            key: statistics.fmean(values)
            for key, values in tiling_times.items()
            if values
        }
        median = statistics.median(tiling_times["solver_total"])
        ratio = (
            mean["solver_total"] / mean["original_callback"]
            if mean.get("original_callback", 0.0) > 0.0
            else float("nan")
        )
        print(
            "TILING_TIME_TOTAL "
            f"records={len(tiling_times['solver_total'])} "
            f"original_callback_mean_ms={mean.get('original_callback', float('nan')):.6g} "
            f"runtime_kb_seed_mean_ms={mean.get('runtime_kb_seed', float('nan')):.6g} "
            f"solver_select_mean_ms={mean.get('solver_select', float('nan')):.6g} "
            f"solver_callback_mean_ms={mean.get('solver_callback', float('nan')):.6g} "
            f"solver_total_mean_ms={mean['solver_total']:.6g} "
            f"solver_total_median_ms={median:.6g} "
            f"solver_over_original_callback={ratio:.6g}"
        )
    if args.direct_comparison_only:
        print("NPU_RESULT_END")
        return
    for evidence in (
        "known_anchor",
        "skinny_n_initial_holdout",
        "skinny_n_k16384_holdout",
        "boundary_control",
        "skinny_n_boundary_holdout",
        "skinny_n_boundary64_holdout",
        "det_split_k_positive_range",
        "det_split_k_rejected_control",
        "alignment_negative_control",
        "prior_regression",
        "broad_validation",
        "unsupported_control",
    ):
        results = evidence_results.get(evidence)
        if not results:
            continue
        print(
            f"EVIDENCE_GROUP {evidence} workloads={sum(results.values())} "
            f"improved={results.get('improved', 0)} "
            f"not_improved={results.get('not_improved', 0)} "
            "no_candidate="
            f"{results.get('no_successful_searched_candidate', 0)} "
            f"other={sum(results.values()) - results.get('improved', 0) - results.get('not_improved', 0) - results.get('no_successful_searched_candidate', 0)}"
        )
    campaign = campaign_statuses(evidence_results)
    print(
        "GENERALIZATION_RESULT skinny_n_initial "
        f"status={campaign['skinny_initial']} "
        "criterion=all_three_unseen_shapes_must_improve"
    )
    print(
        "REFINED_SKINNY_N_RESULT k_eq_16384 "
        f"status={campaign['skinny_k16384']} "
        "criterion=all_three_preregistered_k16384_holdouts_must_improve"
    )
    print(
        "SKINNY_N_BOUNDARY_POLICY_RESULT n_33_to_48_k_eq_16384 "
        f"status={campaign['skinny_boundary']} "
        "criterion=n40_n47_baseN48_and_n48_crossover_must_improve"
    )
    print(
        "SKINNY_N_BOUNDARY64_RESULT n_49_to_64_k_eq_16384 "
        f"status={campaign['skinny_boundary64']} "
        "criterion=all_three_preregistered_n49_n56_n64_holdouts_must_improve"
    )
    print(
        "DETERMINISTIC_SPLIT_K_RESULT aligned_mn_k_16k_to_49k "
        f"status={campaign['det_split_k']} "
        "criterion=k16384_k32768_anchor_k49152_must_improve"
    )
    print(
        "PRIOR_FAILURE_RESULT "
        f"status={campaign['prior_failures']} "
        "criterion=all_six_previously_regressed_workloads_must_improve"
    )
    print(
        "BROAD_VALIDATION_RESULT "
        f"status={campaign['broad_validation']} "
        "criterion=all_28_supported_general_workloads_must_improve"
    )
    print(
        "MATMUL_WIDE_RESULT "
        f"status={campaign['wide']} "
        "anchor_success_alone_is_not_counted"
    )
    print("NPU_RESULT_END")


if __name__ == "__main__":
    main()
