#!/usr/bin/env python3
"""Select one callback-fixed MatMul tiling with the hardware simulator."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import refine_matmul_v3_candidates as old
from profile_official_tilings import IncrementalJsonl
from npu_cost_model import (
    MemorySpace,
    Resource,
    ScheduleSpace,
    SearchPolicy,
    ascend_910b3,
    derive_ideal_region,
    plan_from_cann,
    simulate,
)
from npu_cost_model.operators import matmul


SELECTED_WORKLOADS = 200
SEARCHED_CANDIDATES = 1
MEASURED_TILINGS = 1
CUSTOM_COLUMNS = (
    "search_core_cap",
    "pool_sequence",
    "pool_size",
    "pool_selection",
    "new_model_cycles",
    "new_model_ratio_vs_official",
    "new_model_rank",
    "new_model_bottleneck",
    "new_model_breakdown",
    "new_model_score_ns",
    "candidate_generation_ms",
    "static_legality_ms",
    "official_callback_ms",
    "final_callback_ms",
    "final_callback_count",
    "final_callback_rejections",
    "generated_candidate_count",
    "legal_candidate_count",
    "simulator_valid_candidate_count",
    "ideal_anchor_count",
    "ideal_region_count",
    "ideal_discovery_evaluations",
    "execution_graphs_represented",
    "model_input_source",
)


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def truthy(value: object) -> bool:
    return str(value or "").lower() in {"1", "true", "yes", "on"}


def make_base_from_plan(
    workload: old.Workload,
    hardware: old.Hardware,
    plan,
) -> dict[str, int] | None:
    base_m = plan.tiles["m"]
    base_n = plan.tiles["n"]
    base_k = plan.tiles["k"]
    m_parts = old.ceil_div(workload.m, base_m)
    n_parts = old.ceil_div(workload.n, base_n)
    l2_m = max(1, min(m_parts, old.ceil_div(plan.caches["m"], base_m)))
    l2_n = max(1, min(n_parts, old.ceil_div(plan.caches["n"], base_n)))
    traversal = plan.traversal or ("m", "n")
    l2_order = 1 if traversal[-1] == "n" else 2
    iterate_order = 0 if traversal[-1] == "n" else 1
    buffers = plan.buffer_counts
    cores = min(plan.used_cores, m_parts * n_parts, workload.max_cores)
    knowledge = {
        "usedCoreNum": cores,
        "singleCoreM": base_m,
        "singleCoreN": base_n,
        "singleCoreK": workload.k,
        "baseM": base_m,
        "baseN": base_n,
        "baseK": base_k,
        "depthA1": 1,
        "depthB1": 1,
        "stepM": 1,
        "stepN": 1,
        "iterateOrder": iterate_order,
        "stepKa": 1,
        "stepKb": 1,
        "dbL0A": buffers.get(MemorySpace.L0A, 1),
        "dbL0B": buffers.get(MemorySpace.L0B, 1),
        "dbL0C": buffers.get(MemorySpace.L0C, 1),
        "l2MTileCnt": old.ceil_div(old.ceil_div(workload.m, base_m), l2_m),
        "l2NTileCnt": old.ceil_div(old.ceil_div(workload.n, base_n), l2_n),
        "l2MTileBlock": l2_m,
        "l2NTileBlock": l2_n,
        "l2IterateOrder": l2_order,
        "tilingEnable": 0,
    }
    depth_a, depth_b, step_a, step_b = old.official_l1_for_base(
        workload, knowledge, hardware
    )
    knowledge.update(
        depthA1=depth_a,
        depthB1=depth_b,
        stepKa=step_a,
        stepKb=step_b,
    )
    return knowledge if old.hard_legal(workload, knowledge, hardware) else None


def make_parallel_reduction_from_plan(
    workload: old.Workload,
    hardware: old.Hardware,
    plan,
) -> dict[str, int] | None:
    """Encode a numeric parallel-reduction plan in the CANN 8.1 ABI.

    The fixed values are the public deterministic-reduction execution-graph
    contract. They do not decide whether reduction parallelism is searched or
    how it is ranked; the generic IR solver has already made that decision.
    """

    in_bytes = old.INPUT_BYTES[workload.dtype]
    base_k = 256 // in_bytes
    single_k = 3 * base_k
    k_chunks = old.ceil_div(workload.k, single_k)
    if k_chunks < 2:
        return None
    traversal = plan.traversal or ("m", "n")
    if traversal[-1] == "n":
        step_m, step_n, depth_a, depth_b = 3, 1, 9, 6
        iterate_order, l2_order = 1, 0
        single_m, single_n = 384, max(128, workload.n)
    else:
        step_m, step_n, depth_a, depth_b = 1, 3, 6, 9
        iterate_order, l2_order = 0, 1
        single_m, single_n = max(128, workload.m), 384
    m_chunks = old.ceil_div(workload.m, single_m)
    n_chunks = old.ceil_div(workload.n, single_n)
    knowledge = {
        "usedCoreNum": min(
            plan.used_cores, k_chunks, workload.max_cores, hardware.aic_cores
        ),
        "singleCoreM": single_m,
        "singleCoreN": single_n,
        "singleCoreK": single_k,
        "baseM": 128,
        "baseN": 128,
        "baseK": base_k,
        "depthA1": depth_a,
        "depthB1": depth_b,
        "stepM": step_m,
        "stepN": step_n,
        "iterateOrder": iterate_order,
        "stepKa": 3,
        "stepKb": 3,
        "dbL0A": 2,
        "dbL0B": 2,
        "dbL0C": 2,
        "l2MTileCnt": 1,
        "l2NTileCnt": 1,
        "l2MTileBlock": max(1, m_chunks),
        "l2NTileBlock": max(1, n_chunks),
        "l2IterateOrder": l2_order,
        "tilingEnable": 3,
    }
    return knowledge if old.hard_legal(workload, knowledge, hardware) else None


def derive_proposals(
    workload: old.Workload,
    hardware: old.Hardware,
    core_cap: int,
):
    generic = generic_hardware(hardware)
    operator = matmul(
        workload.m, workload.n, workload.k, workload.dtype,
        trans_a=workload.trans_a, trans_b=workload.trans_b,
    )
    region = derive_ideal_region(
        operator,
        generic,
        ScheduleSpace(core_options=tuple(range(1, core_cap + 1))),
        SearchPolicy(top_k=1, max_evaluations=10000),
    )
    result: list[dict[str, int]] = []
    seen: set[tuple[int, ...]] = set()
    for plan in region.plans:
        reduction_parts = math.prod(plan.reductions.values()) if plan.reductions else 1
        knowledge = (
            make_base_from_plan(workload, hardware, plan)
            if reduction_parts == 1
            else make_parallel_reduction_from_plan(workload, hardware, plan)
        )
        if knowledge is None:
            continue
        signature = old.knowledge_signature(knowledge)
        if signature not in seen:
            seen.add(signature)
            result.append(knowledge)
    return result, region


def proposal_space(
    workload: old.Workload,
    hardware: old.Hardware,
    core_cap: int,
) -> list[dict[str, int]]:
    return derive_proposals(workload, hardware, core_cap)[0]


def proposal_space_for(
    workload: old.Workload,
    seed: old.Seed,
    hardware: old.Hardware,
    core_cap: int,
    search_family: str = "hardware_ideal_region",
) -> list[dict[str, int]]:
    del seed, search_family
    return proposal_space(workload, hardware, core_cap)


def generic_hardware(platform: old.Hardware):
    base = ascend_910b3()
    core_counts = dict(base.core_counts)
    core_counts[Resource.CUBE] = platform.aic_cores
    capacities = dict(base.capacities)
    capacities.update({
        MemorySpace.L0A: platform.l0a_bytes,
        MemorySpace.L0B: platform.l0b_bytes,
        MemorySpace.L0C: platform.l0c_bytes,
        MemorySpace.L1: platform.l1_bytes,
        MemorySpace.L2: platform.l2_bytes,
    })
    return replace(
        base,
        core_counts=core_counts,
        capacities=capacities,
        aggregate_hbm_bytes_per_cycle=(
            platform.hbm_bytes_per_cycle_per_core * platform.aic_cores
        ),
        aggregate_l2_bytes_per_cycle=(
            platform.l2_bytes_per_cycle_per_core * platform.aic_cores
        ),
    )


def new_breakdown(result) -> str:
    return json.dumps(
        {
            "critical_core": result.critical_core_cycles,
            "hbm": result.hbm_cycles,
            "l2": result.l2_cycles,
            "shared": result.shared_resource_cycles,
            "active_cores": result.active_cores,
            "gm_read_bytes": result.gm_read_bytes,
            "gm_write_bytes": result.gm_write_bytes,
            "l2_bytes": result.l2_bytes,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def ranked_pool(
    workload: old.Workload,
    proposals: list[dict[str, int]],
    platform: old.Hardware,
) -> tuple[list[dict], int]:
    generic = generic_hardware(platform)
    operator = matmul(
        workload.m,
        workload.n,
        workload.k,
        workload.dtype,
        trans_a=workload.trans_a,
        trans_b=workload.trans_b,
    )
    scored: list[dict] = []
    new_ns = 0
    for sequence, knowledge in enumerate(proposals, 1):
        started = time.perf_counter_ns()
        new_result = simulate(
            operator,
            plan_from_cann(workload.m, workload.n, workload.k, knowledge),
            generic,
        )
        new_ns += time.perf_counter_ns() - started
        if not new_result.valid:
            continue
        scored.append({
            "knowledge": knowledge,
            "pool_sequence": sequence,
            "new": new_result,
        })
    new_order = sorted(scored, key=lambda item: (
        item["new"].total_cycles, old.knowledge_signature(item["knowledge"])
    ))
    for rank, item in enumerate(new_order, 1):
        item["new_rank_all"] = rank
        item["selection"] = "new_hardware_simulator"
    return new_order, new_ns


def attach_models(
    row: dict[str, str],
    new_result,
    *,
    new_rank: int,
    new_ns: int,
    core_cap: int,
    pool_sequence: int,
    pool_size: int,
    selection: str,
) -> None:
    row.update({
        "search_core_cap": str(core_cap),
        "pool_sequence": str(pool_sequence),
        "pool_size": str(pool_size),
        "pool_selection": selection,
        "new_model_cycles": f"{new_result.total_cycles:.12g}",
        "new_model_rank": str(new_rank),
        "new_model_bottleneck": new_result.bottleneck,
        "new_model_breakdown": new_breakdown(new_result),
        "new_model_score_ns": str(new_ns),
        "model_input_source": "parameters_only_no_latency_history_no_cce_table",
    })


def attach_callback(
    row: dict[str, str],
    callback: old.CallbackTiling,
    official: old.CallbackTiling,
) -> None:
    row.update({
        "callback_tiling_sha256": callback.sha256,
        "callback_tiling_bytes": str(len(callback.blob)),
        "callback_tiling_key": str(callback.key),
        "callback_block_dim": str(callback.block_dim),
        "callback_workspace_bytes": str(sum(callback.workspaces)),
        "callback_l2_cache_flag": str(callback.derived.get("l2CacheFlag", "")),
        "callback_base_an": str(callback.derived.get("baseAN", "")),
        "callback_base_ad": str(callback.derived.get("baseAD", "")),
        "callback_base_bn": str(callback.derived.get("baseBN", "")),
        "callback_base_bd": str(callback.derived.get("baseBD", "")),
        "callback_kernel_suffix": str(old.kernel_suffix(callback.key)),
        "callback_kernel_variant": old.kernel_variant(callback.key),
        "callback_kernel_family": old.kernel_family(callback.key),
        "callback_derived_diff_vs_default": old.callback_derived_diff(
            official, callback
        ),
        "callback_derived_diff_vs_bank_seed": old.callback_derived_diff(
            official, callback
        ),
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-candidates", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-output", type=Path, required=True)
    parser.add_argument("--soc", required=True)
    parser.add_argument("--aic-cores", type=int, required=True)
    parser.add_argument("--l0a-bytes", type=int, required=True)
    parser.add_argument("--l0b-bytes", type=int, required=True)
    parser.add_argument("--l0c-bytes", type=int, required=True)
    parser.add_argument("--l1-bytes", type=int, required=True)
    parser.add_argument("--l2-bytes", type=int, required=True)
    parser.add_argument("--l2-bytes-per-cycle-per-core", type=float, required=True)
    parser.add_argument("--hbm-bytes-per-cycle-per-core", type=float, required=True)
    parser.add_argument(
        "--selected-workloads", type=int, default=SELECTED_WORKLOADS
    )
    parser.add_argument(
        "--searched-candidates", type=int, default=SEARCHED_CANDIDATES
    )
    parser.add_argument("--jsonl-log-directory", type=Path)
    args = parser.parse_args()

    if args.selected_workloads <= 0 or args.searched_candidates != 1:
        raise old.SearchError(
            "model validation selects exactly one simulator tiling per shape"
        )

    platform = old.Hardware(
        args.aic_cores, args.l0a_bytes, args.l0b_bytes, args.l0c_bytes,
        args.l1_bytes, args.l2_bytes, args.l2_bytes_per_cycle_per_core,
        args.hbm_bytes_per_cycle_per_core,
    )
    from tbe.common.platform import set_current_compile_soc_info
    from tbe.common.utils import op_tiling
    set_current_compile_soc_info(args.soc)
    op_tiling._RT_BANK_CACHE = {}

    raw_fields, _ = read_rows(args.raw_candidates)
    fields = ["rank", *(field for field in raw_fields if field != "rank")]
    for field in (*old.EXTRA_COLUMNS, *CUSTOM_COLUMNS):
        if field not in fields:
            fields.append(field)
    catalog_fields, catalog_rows = read_rows(args.catalog)
    selected_workloads: list[dict[str, str]] = []
    selected_rows: list[dict[str, str]] = []
    all_rows: list[dict[str, str]] = []
    execution_graph_counts: dict[str, int] = {}
    anchor_ids = {"matmul_rank_000", "matmul_rank_001", "matmul_rank_002"}
    catalog_ids = {row["workload_id"] for row in catalog_rows}
    required_anchors = (
        anchor_ids
        if args.selected_workloads >= 3 and anchor_ids <= catalog_ids
        else set()
    )
    timing_log = IncrementalJsonl(
        args.jsonl_log_directory.resolve()
        if args.jsonl_log_directory is not None else None,
        50 * 1024 * 1024,
    )
    campaign_started = time.perf_counter_ns()
    stage_totals = {
        "official_callback_ns": 0,
        "candidate_generation_ns": 0,
        "static_legality_ns": 0,
        "simulator_scoring_ns": 0,
        "final_callback_ns": 0,
    }

    for catalog_index, metadata in enumerate(catalog_rows, 1):
        if len(selected_workloads) >= args.selected_workloads:
            break
        metadata = dict(metadata)
        search_family = "hardware_ideal_region"
        metadata["search_family"] = search_family
        callback_workload = old.Workload(
            metadata["workload_id"], int(metadata["m"]), int(metadata["n"]),
            int(metadata["k"]), metadata["dtype"], truthy(metadata["trans_a"]),
            truthy(metadata["trans_b"]), args.aic_cores,
        )
        core_cap = min(int(metadata["search_core_cap"]), args.aic_cores)
        search_workload = replace(callback_workload, max_cores=core_cap)
        shape_started = time.perf_counter_ns()

        started = time.perf_counter_ns()
        official = old.invoke_official_callback(callback_workload)
        official_callback_ns = time.perf_counter_ns() - started

        started = time.perf_counter_ns()
        proposals, ideal_region = derive_proposals(
            search_workload, platform, core_cap
        )
        candidate_generation_ns = time.perf_counter_ns() - started
        generated_count = len(proposals)

        started = time.perf_counter_ns()
        official_signature = old.knowledge_signature(official.knowledge)
        legal: list[dict[str, int]] = []
        legal_signatures: set[tuple[int, ...]] = set()
        for knowledge in proposals:
            signature = old.knowledge_signature(knowledge)
            if (
                signature == official_signature
                or signature in legal_signatures
                or not old.hard_legal(search_workload, knowledge, platform)
            ):
                continue
            legal_signatures.add(signature)
            legal.append(knowledge)
        static_legality_ns = time.perf_counter_ns() - started

        ranked, new_score_ns = ranked_pool(search_workload, legal, platform)
        if not ranked:
            continue

        official_operator = matmul(
            callback_workload.m,
            callback_workload.n,
            callback_workload.k,
            callback_workload.dtype,
            trans_a=callback_workload.trans_a,
            trans_b=callback_workload.trans_b,
        )
        started = time.perf_counter_ns()
        official_new = simulate(
            official_operator,
            plan_from_cann(
                callback_workload.m,
                callback_workload.n,
                callback_workload.k,
                official.knowledge,
            ),
            generic_hardware(platform),
        )
        new_score_ns += time.perf_counter_ns() - started
        if not official_new.valid:
            continue

        callback_failures = 0
        callback_duplicates = 0
        callback_ns = 0
        callback_count = 0
        selected: dict | None = None
        for item in ranked:
            ratio = item["new"].total_cycles / max(
                1.0, official_new.total_cycles
            )
            state = old.State(
                row=old.row_from_state(
                    fields, None, callback_workload, item["knowledge"],
                    "generic_hardware_simulator_v1",
                    old.template_name(item["knowledge"]),
                    ratio,
                    item["new"].gm_read_bytes + item["new"].gm_write_bytes,
                    item["new"].l2_bytes,
                    official.key,
                    guidance=item["selection"], estimate=None,
                    bottleneck=item["new"].bottleneck,
                    rationale="lowest predicted cycles from the hardware simulator",
                    resume_policy="allow_new",
                ),
                knowledge=item["knowledge"],
                model_score=item["new"].total_cycles,
                normalized_score=ratio,
                hbm_bytes=(
                    item["new"].gm_read_bytes + item["new"].gm_write_bytes
                ),
                l2_bytes=item["new"].l2_bytes,
                template=old.template_name(item["knowledge"]),
                guidance=item["selection"],
                estimate=None,
            )
            started = time.perf_counter_ns()
            callback_count += 1
            try:
                callback = old.validate_callback(callback_workload, state)
            except Exception:
                callback_ns += time.perf_counter_ns() - started
                callback_failures += 1
                continue
            callback_ns += time.perf_counter_ns() - started
            if callback.sha256 == official.sha256:
                callback_duplicates += 1
                continue
            state.callback = callback
            attach_callback(state.row, callback, official)
            item["state"] = state
            selected = item
            break
        if selected is None:
            print(
                f"MODEL_VALIDATION_SKIP [{catalog_index}/{len(catalog_rows)}] "
                f"{metadata['workload_id']} no_final_callback_fixed_candidate "
                f"callback_rejected={callback_failures} "
                f"callback_duplicates={callback_duplicates}",
                flush=True,
            )
            continue

        row = selected["state"].row
        row["rank"] = "1"
        attach_models(
            row,
            selected["new"],
            new_rank=selected["new_rank_all"],
            new_ns=new_score_ns,
            core_cap=core_cap,
            pool_sequence=selected["pool_sequence"],
            pool_size=len(legal),
            selection=selected["selection"],
        )
        total_ns = time.perf_counter_ns() - shape_started
        timing_values = {
            "candidate_generation_ms": candidate_generation_ns / 1e6,
            "static_legality_ms": static_legality_ns / 1e6,
            "official_callback_ms": official_callback_ns / 1e6,
            "final_callback_ms": callback_ns / 1e6,
        }
        row.update({
            **{name: f"{value:.9g}" for name, value in timing_values.items()},
            "final_callback_count": str(callback_count),
            "final_callback_rejections": str(callback_failures),
            "generated_candidate_count": str(generated_count),
            "legal_candidate_count": str(len(legal)),
            "simulator_valid_candidate_count": str(len(ranked)),
            "ideal_anchor_count": str(len(ideal_region.anchors)),
            "ideal_region_count": str(len(ideal_region.plans)),
            "ideal_discovery_evaluations": str(ideal_region.evaluated),
            "execution_graphs_represented": ";".join(sorted({
                old.template_name(item) for item in proposals
            })),
            "tiling_official_callback_ms": f"{official_callback_ns / 1e6:.9g}",
            "tiling_runtime_kb_seed_ms": "0",
            "tiling_solver_select_ms": f"{new_score_ns / 1e6:.9g}",
            "tiling_solver_callback_ms": f"{callback_ns / 1e6:.9g}",
            "tiling_solver_callback_count": str(callback_count),
            "tiling_solver_extra_ms": f"{(candidate_generation_ns + static_legality_ns) / 1e6:.9g}",
            "tiling_solver_total_ms": f"{total_ns / 1e6:.9g}",
            "search_model_cycles": f"{selected['new'].total_cycles:.12g}",
            "search_model_raw_ratio_vs_bank_seed": f"{selected['state'].normalized_score:.12g}",
            "search_model_ratio_vs_bank_seed": f"{selected['state'].normalized_score:.12g}",
            "new_model_ratio_vs_official": f"{selected['state'].normalized_score:.12g}",
            "search_model_breakdown": new_breakdown(selected["new"]),
        })
        selected_rows.append(dict(row))
        all_rows.append(dict(row))
        selected_workloads.append(metadata)
        selected_graph = old.template_name(selected["knowledge"])
        execution_graph_counts[selected_graph] = (
            execution_graph_counts.get(selected_graph, 0) + 1
        )
        stage_totals["official_callback_ns"] += official_callback_ns
        stage_totals["candidate_generation_ns"] += candidate_generation_ns
        stage_totals["static_legality_ns"] += static_legality_ns
        stage_totals["simulator_scoring_ns"] += new_score_ns
        stage_totals["final_callback_ns"] += callback_ns
        timing_log.write(
            f"selection:{metadata['workload_id']}",
            {
                "schema": "matmul_hardware_simulator_selection_v3",
                "record_type": "tiling_selection_timing",
                "workload": metadata,
                "generated_candidates": generated_count,
                "legal_candidates": len(legal),
                "simulator_valid_candidates": len(ranked),
                "ideal_anchors": len(ideal_region.anchors),
                "ideal_region_plans": len(ideal_region.plans),
                "ideal_discovery_evaluations": ideal_region.evaluated,
                "execution_graphs_represented": sorted({
                    old.template_name(item) for item in proposals
                }),
                "selected_new_model_rank": selected["new_rank_all"],
                "selected_used_core_num": selected["knowledge"]["usedCoreNum"],
                "callback_attempts": callback_count,
                "callback_rejections": callback_failures,
                "timing_ms": {
                    **timing_values,
                    "simulator_scoring_ms": new_score_ns / 1e6,
                    "total_ms": total_ns / 1e6,
                },
            },
        )
        print(
            f"MODEL_VALIDATION_CANDIDATES [{len(selected_workloads)}/{args.selected_workloads}] "
            f"{metadata['workload_id']} shape={metadata['m']}x{metadata['n']}x{metadata['k']} "
            f"dtype={metadata['dtype']} trans={metadata['trans_a']}{metadata['trans_b']} "
            f"core_cap={core_cap} pool={generated_count} legal={len(legal)} "
            f"graph={selected_graph} selected_core={selected['knowledge']['usedCoreNum']} "
            f"anchors={len(ideal_region.anchors)} region={len(ideal_region.plans)} "
            f"generation_ms={timing_values['candidate_generation_ms']:.3f} "
            f"legality_ms={timing_values['static_legality_ms']:.3f} "
            f"model_ms={new_score_ns / 1e6:.3f} "
            f"official_callback_ms={timing_values['official_callback_ms']:.3f} "
            f"final_callback_ms={timing_values['final_callback_ms']:.3f} "
            f"total_ms={total_ns / 1e6:.3f}",
            flush=True,
        )

    if len(selected_workloads) != args.selected_workloads:
        raise old.SearchError(
            f"only {len(selected_workloads)} workloads produced a callback-fixed "
            f"simulator selection; required {args.selected_workloads}"
        )
    if not required_anchors <= {row["workload_id"] for row in selected_workloads}:
        raise old.SearchError("one or more colleague anchor shapes were not admitted")
    args.workloads.parent.mkdir(parents=True, exist_ok=True)
    with args.workloads.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=catalog_fields)
        writer.writeheader()
        writer.writerows(selected_workloads)
    for destination, rows in ((args.output, selected_rows), (args.all_output, all_rows)):
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    print(
        f"MATMUL_MODEL_VALIDATION_CANDIDATES shapes={len(selected_workloads)} "
        "tilings_per_shape=1 "
        f"measured_per_shape={MEASURED_TILINGS} "
        f"selected_execution_graphs={execution_graph_counts} "
        f"wall_ms={(time.perf_counter_ns() - campaign_started) / 1e6:.3f} "
        + " ".join(
            f"{name[:-3]}_ms={value / 1e6:.3f}"
            for name, value in stage_totals.items()
        ),
        flush=True,
    )
    timing_log.write(
        "selection:campaign_complete",
        {
            "schema": "matmul_hardware_simulator_selection_v3",
            "record_type": "tiling_selection_timing_summary",
            "status": "complete",
            "shape_count": len(selected_workloads),
            "selected_execution_graphs": execution_graph_counts,
            "timing_ms": {
                name[:-3]: value / 1e6
                for name, value in stage_totals.items()
            },
            "wall_ms": (time.perf_counter_ns() - campaign_started) / 1e6,
        },
    )
    timing_log.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (old.SearchError, OSError, ValueError, KeyError) as exception:
        print(f"fatal: {exception}", flush=True)
        raise SystemExit(1)
