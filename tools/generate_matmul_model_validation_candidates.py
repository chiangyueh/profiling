#!/usr/bin/env python3
"""Select one MatMul tiling using only the hardware simulator."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
    "hardware_aic_cores",
    "pool_sequence",
    "pool_size",
    "pool_selection",
    "new_model_cycles",
    "new_model_ratio_vs_official",
    "new_model_rank",
    "new_model_bottleneck",
    "new_model_breakdown",
    "new_model_score_ns",
    "model_schedule_sha256",
    "model_kernel_suffix",
    "model_kernel_variant",
    "model_kernel_family",
    "candidate_generation_ms",
    "static_legality_ms",
    "execution_abi_ms",
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
    # CANN 8.1 BASE has no independent task tile below singleCoreM/N:
    # its legal contract requires baseM/N to cover that complete core task.
    # Preserve the generic task geometry at the ABI boundary rather than
    # silently replacing it with the smaller scratchpad inner tile.
    single_m = min(workload.m, plan.tasks["m"])
    single_n = min(workload.n, plan.tasks["n"])
    base_m = old.align_up(single_m, 16)
    base_n = old.align_up(single_n, 16)
    base_k = min(
        plan.tiles["k"],
        old.align_up(workload.k, old.base_k_alignment(workload)),
    )
    m_parts = old.ceil_div(workload.m, single_m)
    n_parts = old.ceil_div(workload.n, single_n)
    l2_m = max(1, min(m_parts, old.ceil_div(plan.caches["m"], single_m)))
    l2_n = max(1, min(n_parts, old.ceil_div(plan.caches["n"], single_n)))
    traversal = plan.traversal or ("m", "n")
    l2_order = 1 if traversal[-1] == "n" else 2
    iterate_order = 0 if traversal[-1] == "n" else 1
    buffers = plan.buffer_counts
    cores = min(plan.used_cores, m_parts * n_parts, workload.max_cores)
    knowledge = {
        "usedCoreNum": cores,
        "singleCoreM": single_m,
        "singleCoreN": single_n,
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
    per_core_l2 = hardware.l2_bytes * 7 // 10 // hardware.aic_cores
    if traversal[-1] == "n":
        fixed_m = min(single_m, workload.m)
        fixed_bytes = single_k * fixed_m * in_bytes
        bytes_per_n = single_k * in_bytes + fixed_m * 4
        if per_core_l2 <= fixed_bytes:
            return None
        n_l2_split = (per_core_l2 - fixed_bytes) // bytes_per_n
        if n_l2_split <= 0:
            return None
        if workload.n > n_l2_split:
            n_l2_split = old.align_up(n_l2_split, 16)
            n_count = old.ceil_div(workload.n, n_l2_split)
            single_n = old.align_up(old.ceil_div(workload.n, n_count), 16)
    else:
        fixed_n = min(single_n, workload.n)
        fixed_bytes = single_k * fixed_n * in_bytes
        bytes_per_m = single_k * in_bytes + fixed_n * 4
        if per_core_l2 <= fixed_bytes:
            return None
        m_l2_split = (per_core_l2 - fixed_bytes) // bytes_per_m
        if m_l2_split <= 0:
            return None
        if workload.m > m_l2_split:
            m_l2_split = old.align_up(m_l2_split, 16)
            m_count = old.ceil_div(workload.m, m_l2_split)
            single_m = old.align_up(old.ceil_div(workload.m, m_count), 16)
    m_chunks = old.ceil_div(workload.m, single_m)
    n_chunks = old.ceil_div(workload.n, single_n)
    used_cores = min(
        plan.used_cores, k_chunks, workload.max_cores, hardware.aic_cores
    )
    # This execution graph is a parallel reduction.  One active core cannot
    # shorten the reduction and is dominated by the serial BASE graph.
    if used_cores < 2:
        return None
    knowledge = {
        "usedCoreNum": used_cores,
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
):
    generic = generic_hardware(hardware)
    operator = matmul(
        workload.m, workload.n, workload.k, workload.dtype,
        trans_a=workload.trans_a, trans_b=workload.trans_b,
    )
    region = derive_ideal_region(
        operator,
        generic,
        ScheduleSpace(
            core_options=tuple(range(1, hardware.aic_cores + 1)),
            # CANN 8.1 BASE exposes one M/N geometry for both the Cube tile
            # and the per-core task. Declare that backend contract so the
            # solver does not score an unrepresentable schedule.
            coupled_task_axes=("m", "n"),
        ),
        SearchPolicy(top_k=1, max_evaluations=10000),
    )
    if not region.exhaustive:
        raise old.SearchError(
            "hardware ideal-region derivation reached its safety ceiling; "
            "refusing to return a truncated search result"
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
) -> list[dict[str, int]]:
    return derive_proposals(workload, hardware)[0]


def proposal_space_for(
    workload: old.Workload,
    seed: old.Seed,
    hardware: old.Hardware,
    search_family: str = "hardware_ideal_region",
) -> list[dict[str, int]]:
    del seed, search_family
    return proposal_space(workload, hardware)


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
    equivalent_scores: dict[tuple[object, ...], object] = {}
    for sequence, knowledge in enumerate(proposals, 1):
        plan = plan_from_cann(workload.m, workload.n, workload.k, knowledge)
        equivalence_key = (
            plan.algorithm,
            plan.axis_tiles,
            plan.task_tiles,
            plan.cache_tiles,
            plan.used_cores,
            plan.reduction_parts,
            plan.buffers,
        )
        new_result = equivalent_scores.get(equivalence_key)
        if new_result is None:
            started = time.perf_counter_ns()
            new_result = simulate(operator, plan, generic)
            new_ns += time.perf_counter_ns() - started
            peak = dict(new_result.peak_memory_bytes)
            if (
                new_result.valid
                and peak.get(MemorySpace.L2, 0)
                <= generic.capacities.get(MemorySpace.L2, 0)
            ):
                # For a fully resident cache group the simulator's hardware
                # graph is traversal-independent.  Reuse that exact result;
                # both ABI schedules remain in the ranked pool.
                equivalent_scores[equivalence_key] = new_result
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
    hardware_aic_cores: int,
    pool_sequence: int,
    pool_size: int,
    selection: str,
) -> None:
    row.update({
        "hardware_aic_cores": str(hardware_aic_cores),
        "pool_sequence": str(pool_sequence),
        "pool_size": str(pool_size),
        "pool_selection": selection,
        "new_model_cycles": f"{new_result.total_cycles:.12g}",
        "new_model_rank": str(new_rank),
        "new_model_bottleneck": new_result.bottleneck,
        "new_model_breakdown": new_breakdown(new_result),
        "new_model_score_ns": str(new_ns),
        "model_input_source": (
            "parameters_only_no_latency_history_no_cce_table_no_official_tiler"
        ),
    })


def kernel_suffix_for(workload: old.Workload, knowledge: dict[str, int]) -> int:
    """Return the CANN 8.1 kernel-family suffix without running its tiler."""

    aligned = (
        workload.m % 16 == 0
        and workload.n % 16 == 0
        and workload.k % old.base_k_alignment(workload) == 0
    )
    low_digit = int(aligned)
    family = old.template_name(knowledge)
    bases = {
        "BASE": 0,
        "SINGLE_CORE_SPLIT_K": 20,
        "DETERMINISTIC_SPLIT_K": 30,
        "AL1_FULL_LOAD": 100,
        "BL1_FULL_LOAD": 200,
        "BL1_FULL_LOAD_FIXPIPE": 10200,
        "BL1_FULL_LOAD_VEC_NZ2ND": 20200,
    }
    suffix = bases[family] + low_digit
    if suffix not in old.CANN81_KERNEL_VARIANTS:
        # AL1 and VEC_NZ2ND have no unaligned CANN 8.1 kernel.
        raise old.SearchError(
            f"CANN 8.1 has no {family} kernel for this alignment"
        )
    return suffix


def attach_execution_identity(
    row: dict[str, str],
    workload: old.Workload,
    knowledge: dict[str, int],
) -> None:
    suffix = kernel_suffix_for(workload, knowledge)
    schedule_payload = json.dumps(
        {
            "shape": [workload.m, workload.n, workload.k],
            "dtype": workload.dtype,
            "trans_a": workload.trans_a,
            "trans_b": workload.trans_b,
            "knowledge": [knowledge[name] for name in old.KNOWLEDGE_FIELDS],
        },
        separators=(",", ":"),
    ).encode()
    schedule_sha = hashlib.sha256(schedule_payload).hexdigest()
    row.update({
        "model_schedule_sha256": schedule_sha,
        "model_kernel_suffix": str(suffix),
        "model_kernel_variant": old.CANN81_KERNEL_VARIANTS[suffix][0],
        "model_kernel_family": old.CANN81_KERNEL_VARIANTS[suffix][1],
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
        "candidate_generation_ns": 0,
        "static_legality_ns": 0,
        "simulator_scoring_ns": 0,
        "execution_abi_ns": 0,
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
        search_workload = callback_workload
        shape_started = time.perf_counter_ns()

        started = time.perf_counter_ns()
        proposals, ideal_region = derive_proposals(
            search_workload, platform
        )
        candidate_generation_ns = time.perf_counter_ns() - started
        generated_count = len(proposals)

        started = time.perf_counter_ns()
        legal: list[dict[str, int]] = []
        legal_signatures: set[tuple[int, ...]] = set()
        for knowledge in proposals:
            signature = old.knowledge_signature(knowledge)
            if (
                signature in legal_signatures
                or not old.hard_legal(search_workload, knowledge, platform)
            ):
                continue
            legal_signatures.add(signature)
            legal.append(knowledge)
        static_legality_ns = time.perf_counter_ns() - started

        ranked, new_score_ns = ranked_pool(search_workload, legal, platform)
        if not ranked:
            continue

        selected = ranked[0]
        started = time.perf_counter_ns()
        suffix = kernel_suffix_for(callback_workload, selected["knowledge"])
        row = old.row_from_state(
            fields, None, callback_workload, selected["knowledge"],
            "generic_hardware_simulator_v1",
            old.template_name(selected["knowledge"]),
            selected["new"].total_cycles,
            selected["new"].gm_read_bytes + selected["new"].gm_write_bytes,
            selected["new"].l2_bytes,
            10_000_000_000_000_000_000 + suffix,
            guidance=selected["selection"], estimate=None,
            bottleneck=selected["new"].bottleneck,
            rationale="lowest predicted cycles from the hardware simulator",
            resume_policy="allow_new",
        )
        attach_execution_identity(
            row, callback_workload, selected["knowledge"]
        )
        execution_abi_ns = time.perf_counter_ns() - started
        # These fields are ratios only in the historical callback-seeded
        # search.  A standalone hardware model has no official denominator.
        row.update({
            "search_model_score": f"{selected['new'].total_cycles:.12g}",
            "search_model_cycles": f"{selected['new'].total_cycles:.12g}",
            "search_model_raw_ratio_vs_bank_seed": "",
            "search_model_ratio_vs_bank_seed": "",
            "new_model_ratio_vs_official": "",
        })
        row["rank"] = "1"
        attach_models(
            row,
            selected["new"],
            new_rank=selected["new_rank_all"],
            new_ns=new_score_ns,
            hardware_aic_cores=args.aic_cores,
            pool_sequence=selected["pool_sequence"],
            pool_size=len(legal),
            selection=selected["selection"],
        )
        total_ns = time.perf_counter_ns() - shape_started
        timing_values = {
            "candidate_generation_ms": candidate_generation_ns / 1e6,
            "static_legality_ms": static_legality_ns / 1e6,
            "execution_abi_ms": execution_abi_ns / 1e6,
        }
        row.update({
            **{name: f"{value:.9g}" for name, value in timing_values.items()},
            "generated_candidate_count": str(generated_count),
            "legal_candidate_count": str(len(legal)),
            "simulator_valid_candidate_count": str(len(ranked)),
            "ideal_anchor_count": str(len(ideal_region.anchors)),
            "ideal_region_count": str(len(ideal_region.plans)),
            "ideal_discovery_evaluations": str(ideal_region.evaluated),
            "execution_graphs_represented": ";".join(sorted({
                old.template_name(item) for item in proposals
            })),
            "tiling_official_callback_ms": "0",
            "tiling_runtime_kb_seed_ms": "0",
            "tiling_solver_select_ms": f"{new_score_ns / 1e6:.9g}",
            "tiling_solver_callback_ms": "0",
            "tiling_solver_callback_count": "0",
            "tiling_solver_extra_ms": f"{(candidate_generation_ns + static_legality_ns) / 1e6:.9g}",
            "tiling_solver_total_ms": f"{total_ns / 1e6:.9g}",
            "search_model_cycles": f"{selected['new'].total_cycles:.12g}",
            "search_model_raw_ratio_vs_bank_seed": "",
            "search_model_ratio_vs_bank_seed": "",
            "new_model_ratio_vs_official": "",
            "search_model_breakdown": new_breakdown(selected["new"]),
        })
        selected_rows.append(dict(row))
        all_rows.append(dict(row))
        selected_workloads.append(metadata)
        selected_graph = old.template_name(selected["knowledge"])
        execution_graph_counts[selected_graph] = (
            execution_graph_counts.get(selected_graph, 0) + 1
        )
        stage_totals["candidate_generation_ns"] += candidate_generation_ns
        stage_totals["static_legality_ns"] += static_legality_ns
        stage_totals["simulator_scoring_ns"] += new_score_ns
        stage_totals["execution_abi_ns"] += execution_abi_ns
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
                "official_tiler_calls": 0,
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
            f"hardware_cores={args.aic_cores} pool={generated_count} legal={len(legal)} "
            f"graph={selected_graph} selected_core={selected['knowledge']['usedCoreNum']} "
            f"anchors={len(ideal_region.anchors)} region={len(ideal_region.plans)} "
            f"generation_ms={timing_values['candidate_generation_ms']:.3f} "
            f"legality_ms={timing_values['static_legality_ms']:.3f} "
            f"model_ms={new_score_ns / 1e6:.3f} "
            f"execution_abi_ms={timing_values['execution_abi_ms']:.3f} "
            f"total_ms={total_ns / 1e6:.3f}",
            flush=True,
        )

    if len(selected_workloads) != args.selected_workloads:
        raise old.SearchError(
            f"only {len(selected_workloads)} workloads produced a legal "
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
