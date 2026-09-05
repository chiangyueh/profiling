#!/usr/bin/env python3
"""Build bounded, hardware-factor MatMul calibration candidates.

The measured set is a union of controlled sweeps around the model optimum.
It never reads measured latency, a CCE table, RuntimeKb, or the official tiler.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from dataclasses import replace
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import refine_matmul_v3_candidates as old
from npu_cost_model import (
    MemorySpace,
    Resource,
    ScheduleSpace,
    SearchPolicy,
    ascend_910b3,
    cann81_matmul_effective_l1_bytes,
    derive_ideal_region,
    execution_mode_name,
    lower_plan_to_cann,
    plan_from_cann,
    simulate,
    source_kernel_suffix,
    validate_cann_tiling,
)
from npu_cost_model.operators import matmul


RESERVE_CANDIDATES = 8
OBSOLETE_EXECUTION_COLUMNS = {
    "search_model_raw_ratio_vs_bank_seed",
    "search_model_ratio_vs_bank_seed",
    "callback_tiling_sha256",
    "callback_tiling_bytes",
    "callback_tiling_key",
    "callback_block_dim",
    "callback_workspace_bytes",
    "callback_l2_cache_flag",
    "callback_base_an",
    "callback_base_ad",
    "callback_base_bn",
    "callback_base_bd",
    "callback_kernel_suffix",
    "callback_kernel_variant",
    "callback_kernel_family",
    "callback_derived_diff_vs_default",
    "callback_derived_diff_vs_bank_seed",
    "tiling_official_callback_ms",
    "tiling_runtime_kb_seed_ms",
    "tiling_solver_callback_ms",
    "tiling_solver_callback_count",
}
FACTOR_FIELDS = {
    "core_parallelism": ("usedCoreNum",),
    "mn_geometry": (
        "singleCoreM", "singleCoreN", "baseM", "baseN", "stepM", "stepN",
    ),
    "k_pipeline": (
        "singleCoreK", "baseK", "depthA1", "depthB1", "stepKa", "stepKb",
    ),
    "l0_buffering": ("dbL0A", "dbL0B", "dbL0C"),
    "cube_traversal": ("iterateOrder",),
    "l2_partition": (
        "l2MTileCnt", "l2NTileCnt", "l2MTileBlock", "l2NTileBlock",
        "l2IterateOrder",
    ),
}
FIELD_TO_FACTOR = {
    field: factor for factor, fields in FACTOR_FIELDS.items() for field in fields
}
CUSTOM_COLUMNS = (
    "hardware_aic_cores", "pool_size", "pool_sequence", "pool_selection",
    "new_model_cycles", "new_model_rank", "new_model_bottleneck",
    "new_model_breakdown", "new_model_score_ns", "model_schedule_sha256",
    "model_kernel_suffix", "model_kernel_variant", "model_kernel_family",
    "model_input_source", "global_model_rank", "controlled_factor",
    "factor_signature", "is_reserve", "coverage_intent",
    "calibration_partition", "required_successful_tilings",
    "candidate_generation_ms", "static_legality_ms", "simulator_scoring_ms",
    "generated_candidate_count", "legal_candidate_count",
)


def truthy(value: object) -> bool:
    return str(value or "").lower() in {"1", "true", "yes", "on"}


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def generic_hardware(platform: old.Hardware):
    base = ascend_910b3()
    cores = dict(base.core_counts)
    cores[Resource.CUBE] = platform.aic_cores
    capacities = dict(base.capacities)
    capacities.update({
        MemorySpace.L0A: platform.l0a_bytes,
        MemorySpace.L0B: platform.l0b_bytes,
        MemorySpace.L0C: platform.l0c_bytes,
        MemorySpace.L1: cann81_matmul_effective_l1_bytes(platform.l1_bytes),
        MemorySpace.L2: platform.l2_bytes,
    })
    return replace(
        base,
        core_counts=cores,
        capacities=capacities,
        aggregate_hbm_bytes_per_cycle=(
            platform.hbm_bytes_per_cycle_per_core * platform.aic_cores
        ),
        aggregate_l2_bytes_per_cycle=(
            platform.l2_bytes_per_cycle_per_core * platform.aic_cores
        ),
    )


def signature(knowledge: dict[str, int]) -> tuple[int, ...]:
    return tuple(int(knowledge[field]) for field in old.KNOWLEDGE_FIELDS)


def selected_partition_values(total: int, current: int) -> tuple[int, ...]:
    values = {1, max(1, total), max(1, min(total, current))}
    for divisor in (2, 3, 4, 8):
        values.add(max(1, old.ceil_div(total, divisor)))
    return tuple(sorted(values))


def raw_hardware_region(
    workload: old.Workload,
    hardware,
):
    operator = matmul(
        workload.m, workload.n, workload.k, workload.dtype,
        trans_a=workload.trans_a, trans_b=workload.trans_b,
    )
    region = derive_ideal_region(
        operator,
        hardware,
        ScheduleSpace(core_options=tuple(range(1, hardware.core_count(Resource.CUBE) + 1))),
        SearchPolicy(top_k=1, max_evaluations=100_000),
    )
    result: list[dict[str, int]] = []
    seen: set[tuple[int, ...]] = set()
    for plan in region.plans:
        try:
            knowledge = lower_plan_to_cann(
                workload.m, workload.n, workload.k, workload.dtype,
                workload.trans_a, workload.trans_b, plan, hardware,
            )
        except ValueError:
            continue
        key = signature(knowledge)
        if key not in seen:
            seen.add(key)
            result.append({field: int(knowledge[field]) for field in old.KNOWLEDGE_FIELDS})
    return operator, region, result


def expanded_knowledge(
    workload: old.Workload,
    seeds: list[dict[str, int]],
    hardware,
    coverage_intent: str,
) -> list[dict[str, int]]:
    """Expand only declared hardware controls around source-lowered seeds."""

    candidates: dict[tuple[int, ...], dict[str, int]] = {}

    def add(value: dict[str, int]) -> None:
        candidates.setdefault(signature(value), value)

    for seed in seeds:
        add(dict(seed))

    # Expand the declared calibration stratum while retaining unexpanded
    # model optima from every other graph for cross-graph ranking.  This is
    # experimental coverage, not a restriction on solver graph selection.
    focused_seeds = [
        seed for seed in seeds
        if execution_mode_name(seed) == coverage_intent
    ][:12]
    for seed in focused_seeds:
        for cores in range(1, hardware.core_count(Resource.CUBE) + 1):
            candidate = dict(seed)
            candidate["usedCoreNum"] = cores
            add(candidate)

        for values in product((1, 2), repeat=3):
            candidate = dict(seed)
            candidate.update(
                dbL0A=values[0], dbL0B=values[1], dbL0C=values[2]
            )
            add(candidate)

        for a_packets, b_packets in product((1, 2), repeat=2):
            candidate = dict(seed)
            candidate["depthA1"] = (
                candidate["stepM"] * candidate["stepKa"] * a_packets
            )
            candidate["depthB1"] = (
                candidate["stepN"] * candidate["stepKb"] * b_packets
            )
            add(candidate)

        if seed["l2MTileBlock"] > 0 and seed["l2NTileBlock"] > 0:
            m_total = old.ceil_div(workload.m, seed["singleCoreM"])
            n_total = old.ceil_div(workload.n, seed["singleCoreN"])
            m_values = selected_partition_values(m_total, seed["l2MTileBlock"])
            n_values = selected_partition_values(n_total, seed["l2NTileBlock"])
            # This is a union of row and column sweeps around the same point,
            # not m_values x n_values.
            partitions = [
                *((value, seed["l2NTileBlock"]) for value in m_values),
                *((seed["l2MTileBlock"], value) for value in n_values),
            ]
            for m_block, n_block in partitions:
                for order in (1, 2):
                    candidate = dict(seed)
                    candidate.update(
                        l2MTileCnt=old.ceil_div(m_total, m_block),
                        l2NTileCnt=old.ceil_div(n_total, n_block),
                        l2MTileBlock=m_block,
                        l2NTileBlock=n_block,
                        l2IterateOrder=order,
                    )
                    add(candidate)
    return list(candidates.values())


def breakdown(result) -> str:
    return json.dumps({
        "critical_core": result.critical_core_cycles,
        "hbm": result.hbm_cycles,
        "l2": result.l2_cycles,
        "shared": result.shared_resource_cycles,
        "active_cores": result.active_cores,
        "gm_read_bytes": result.gm_read_bytes,
        "gm_write_bytes": result.gm_write_bytes,
        "l2_bytes": result.l2_bytes,
        "workspace_bytes": result.workspace_bytes,
        "reduction_cycles": result.reduction_cycles,
        "resources": {resource.value: cycles for resource, cycles in result.resource_cycles},
    }, separators=(",", ":"), sort_keys=True)


def legal_scored_pool(
    workload: old.Workload,
    operator,
    proposed: list[dict[str, int]],
    hardware,
):
    result = []
    legality_ns = 0
    scoring_ns = 0
    for sequence, knowledge in enumerate(proposed, 1):
        started = time.perf_counter_ns()
        violations = validate_cann_tiling(
            workload.m, workload.n, workload.k, workload.dtype,
            workload.trans_a, workload.trans_b, knowledge, hardware,
            aoe_injection=True,
        )
        legality_ns += time.perf_counter_ns() - started
        if violations:
            continue
        try:
            plan = plan_from_cann(
                workload.m, workload.n, workload.k, knowledge,
                dtype=workload.dtype, trans_a=workload.trans_a,
                trans_b=workload.trans_b,
            )
            started = time.perf_counter_ns()
            simulation = simulate(operator, plan, hardware)
            scoring_ns += time.perf_counter_ns() - started
        except (KeyError, ValueError):
            continue
        if simulation.valid:
            result.append({
                "knowledge": knowledge,
                "simulation": simulation,
                "pool_sequence": sequence,
                "family": execution_mode_name(knowledge),
            })
    result.sort(key=lambda item: (
        item["simulation"].total_cycles, signature(item["knowledge"])
    ))
    for rank, item in enumerate(result, 1):
        item["model_rank"] = rank
    return result, legality_ns, scoring_ns


def difference_factors(
    anchor: dict[str, int], candidate: dict[str, int]
) -> tuple[str, ...]:
    factors = {
        FIELD_TO_FACTOR[field]
        for field in old.KNOWLEDGE_FIELDS
        if candidate[field] != anchor[field] and field in FIELD_TO_FACTOR
    }
    return tuple(sorted(factors))


def factor_signature(factor: str, knowledge: dict[str, int]) -> str:
    fields = FACTOR_FIELDS.get(factor, old.KNOWLEDGE_FIELDS)
    return ":".join(f"{field}={knowledge[field]}" for field in fields)


def select_controlled(
    pool: list[dict], count: int, reserves: int, coverage_intent: str,
):
    if not pool:
        raise old.SearchError("empty legal model pool")
    coverage_items = [
        item for item in pool if item["family"] == coverage_intent
    ]
    if not coverage_items:
        raise old.SearchError(
            f"coverage graph {coverage_intent} has no legal hardware schedule"
        )
    anchor = coverage_items[0]["knowledge"]
    by_factor: dict[str, list[dict]] = {factor: [] for factor in FACTOR_FIELDS}
    coupled: list[dict] = []
    for item in coverage_items:
        factors = difference_factors(anchor, item["knowledge"])
        item["factors"] = factors
        if len(factors) == 1:
            by_factor[factors[0]].append(item)
        elif factors:
            coupled.append(item)

    # One representative per factor value first.  Within a value, choose the
    # least coupled and then the best hardware-model score.
    queues: dict[str, list[dict]] = {}
    for factor, items in by_factor.items():
        representatives: dict[str, dict] = {}
        for item in items:
            key = factor_signature(factor, item["knowledge"])
            representatives.setdefault(key, item)
        queues[factor] = sorted(
            representatives.values(), key=lambda item: item["model_rank"]
        )

    selected: list[tuple[dict, str]] = [(pool[0], "model_optimum_anchor")]
    seen = {signature(pool[0]["knowledge"])}
    # pool[0] is the unrestricted hardware-model optimum, so graph selection
    # is tested once without a workload label.  The remaining rows are the
    # controlled factor sweep for this coverage stratum.  Forcing one row
    # from every graph here would promote a low-ranked graph merely because
    # it exists, which is neither model ranking nor controlled calibration.
    coverage_anchor = coverage_items[0]
    coverage_anchor_key = signature(coverage_anchor["knowledge"])
    if coverage_anchor_key not in seen and len(selected) < count:
        seen.add(coverage_anchor_key)
        selected.append((coverage_anchor, "coverage_anchor"))
    factor_order = tuple(FACTOR_FIELDS)
    progress = True
    while len(selected) < count and progress:
        progress = False
        for factor in factor_order:
            queue = queues[factor]
            while queue and signature(queue[0]["knowledge"]) in seen:
                queue.pop(0)
            if queue and len(selected) < count:
                item = queue.pop(0)
                seen.add(signature(item["knowledge"]))
                selected.append((item, factor))
                progress = True

    coupled.sort(key=lambda item: (
        -len(item["factors"]), item["model_rank"]
    ))
    for item in (*coupled, *coverage_items, *pool):
        if len(selected) >= count:
            break
        key = signature(item["knowledge"])
        if key in seen:
            continue
        seen.add(key)
        factor = (
            "+".join(item.get("factors", ()))
            if item.get("factors") else "hardware_frontier_fill"
        )
        selected.append((item, factor))
    if len(selected) < count:
        raise old.SearchError(
            f"only {len(selected)} controlled legal candidates; required {count}"
        )

    reserve_rows: list[tuple[dict, str]] = []
    for item in pool:
        key = signature(item["knowledge"])
        if key in seen:
            continue
        seen.add(key)
        factors = item.get("factors", ())
        reserve_rows.append((
            item,
            "+".join(factors) if factors else "hardware_frontier_fill",
        ))
        if len(reserve_rows) == reserves:
            break
    if len(reserve_rows) < reserves:
        raise old.SearchError(
            f"only {len(reserve_rows)} reserve candidates; required {reserves}"
        )
    return selected, reserve_rows


def execution_identity(
    workload: old.Workload, knowledge: dict[str, int]
) -> tuple[str, int]:
    payload = json.dumps({
        "shape": [workload.m, workload.n, workload.k],
        "dtype": workload.dtype,
        "trans_a": workload.trans_a,
        "trans_b": workload.trans_b,
        "knowledge": [knowledge[field] for field in old.KNOWLEDGE_FIELDS],
    }, separators=(",", ":")).encode()
    suffix = source_kernel_suffix(
        workload.m, workload.n, workload.k, workload.dtype,
        workload.trans_a, workload.trans_b, knowledge,
    )
    return hashlib.sha256(payload).hexdigest(), suffix


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
    parser.add_argument("--reserve-candidates", type=int, default=RESERVE_CANDIDATES)
    args = parser.parse_args()

    platform = old.Hardware(
        args.aic_cores, args.l0a_bytes, args.l0b_bytes, args.l0c_bytes,
        args.l1_bytes, args.l2_bytes, args.l2_bytes_per_cycle_per_core,
        args.hbm_bytes_per_cycle_per_core,
    )
    hardware = generic_hardware(platform)
    raw_fields, _ = read_rows(args.raw_candidates)
    fields = [
        "rank",
        *(
            field for field in raw_fields
            if field != "rank" and field not in OBSOLETE_EXECUTION_COLUMNS
        ),
    ]
    for field in (*old.EXTRA_COLUMNS, *CUSTOM_COLUMNS):
        if field not in fields:
            fields.append(field)
    catalog_fields, catalog = read_rows(args.catalog)
    selected_rows: list[dict[str, str]] = []
    all_rows: list[dict[str, str]] = []
    selected_workloads: list[dict[str, str]] = []
    family_counts: Counter[str] = Counter()
    formal_count = 0

    for workload_index, metadata in enumerate(catalog, 1):
        shape_started = time.perf_counter_ns()
        workload = old.Workload(
            metadata["workload_id"], int(metadata["m"]), int(metadata["n"]),
            int(metadata["k"]), metadata["dtype"], truthy(metadata["trans_a"]),
            truthy(metadata["trans_b"]), args.aic_cores,
        )
        coverage_intent = metadata["coverage_intent"]
        required = int(metadata["required_successful_tilings"])
        generation_started = time.perf_counter_ns()
        operator, region, seeds = raw_hardware_region(workload, hardware)
        proposed = expanded_knowledge(
            workload, seeds, hardware, coverage_intent
        )
        generation_ns = time.perf_counter_ns() - generation_started
        pool, legality_ns, scoring_ns = legal_scored_pool(
            workload, operator, proposed, hardware
        )
        selected, reserves = select_controlled(
            pool, required, args.reserve_candidates, coverage_intent
        )
        chosen = [
            *((item, factor, False) for item, factor in selected),
            *((item, factor, True) for item, factor in reserves),
        ]
        for output_rank, (item, factor, reserve) in enumerate(chosen, 1):
            knowledge = item["knowledge"]
            simulation = item["simulation"]
            family = item["family"]
            schedule_sha, suffix = execution_identity(workload, knowledge)
            row = old.row_from_state(
                fields, None, workload, knowledge,
                "generic_hardware_factor_region_v1", family.upper(),
                simulation.total_cycles,
                simulation.gm_read_bytes + simulation.gm_write_bytes,
                simulation.l2_bytes, 10_000_000_000_000_000_000 + suffix,
                guidance=factor, bottleneck=simulation.bottleneck,
                rationale="bounded hardware-factor sweep around the model optimum",
                resume_policy="allow_new",
            )
            row.update({
                "rank": str(output_rank),
                "execution_mode": family,
                "hardware_aic_cores": str(args.aic_cores),
                "pool_size": str(len(pool)),
                "pool_sequence": str(item["pool_sequence"]),
                "pool_selection": "reserve" if reserve else "formal_factor_sweep",
                "new_model_cycles": f"{simulation.total_cycles:.12g}",
                "new_model_rank": str(item["model_rank"]),
                "global_model_rank": str(item["model_rank"]),
                "new_model_bottleneck": simulation.bottleneck,
                "new_model_breakdown": breakdown(simulation),
                "new_model_score_ns": str(scoring_ns),
                "model_schedule_sha256": schedule_sha,
                "model_kernel_suffix": str(suffix),
                "model_kernel_variant": family.upper(),
                "model_kernel_family": family.upper(),
                "model_input_source": "parameters_only_no_latency_history_no_cce_table_no_official_tiler",
                "controlled_factor": factor,
                "factor_signature": factor_signature(
                    factor if factor in FACTOR_FIELDS else "", knowledge
                ),
                "is_reserve": str(int(reserve)),
                "coverage_intent": coverage_intent,
                "calibration_partition": metadata["calibration_partition"],
                "required_successful_tilings": str(required),
                "candidate_generation_ms": f"{generation_ns / 1e6:.9g}",
                "static_legality_ms": f"{legality_ns / 1e6:.9g}",
                "simulator_scoring_ms": f"{scoring_ns / 1e6:.9g}",
                "generated_candidate_count": str(len(proposed)),
                "legal_candidate_count": str(len(pool)),
                "tiling_solver_select_ms": f"{scoring_ns / 1e6:.9g}",
                "tiling_solver_extra_ms": f"{(generation_ns + legality_ns) / 1e6:.9g}",
                "tiling_solver_total_ms": f"{(time.perf_counter_ns() - shape_started) / 1e6:.9g}",
            })
            selected_rows.append(row)
        for item in pool:
            knowledge = item["knowledge"]
            simulation = item["simulation"]
            family = item["family"]
            schedule_sha, suffix = execution_identity(workload, knowledge)
            row = old.row_from_state(
                fields, None, workload, knowledge,
                "generic_hardware_factor_region_v1", family.upper(),
                simulation.total_cycles,
                simulation.gm_read_bytes + simulation.gm_write_bytes,
                simulation.l2_bytes, 10_000_000_000_000_000_000 + suffix,
                guidance="complete_legal_factor_pool",
                bottleneck=simulation.bottleneck,
                rationale="complete bounded legal factor pool",
                resume_policy="allow_new",
            )
            row.update({
                "rank": str(item["model_rank"]), "execution_mode": family,
                "new_model_cycles": f"{simulation.total_cycles:.12g}",
                "global_model_rank": str(item["model_rank"]),
                "model_schedule_sha256": schedule_sha,
                "model_kernel_suffix": str(suffix),
                "model_kernel_family": family.upper(),
                "coverage_intent": coverage_intent,
                "calibration_partition": metadata["calibration_partition"],
                "required_successful_tilings": str(required),
            })
            all_rows.append(row)
        selected_workloads.append(metadata)
        family_counts.update(item[0]["family"] for item in selected)
        formal_count += required
        print(
            f"HARDWARE_FACTOR_CANDIDATES [{workload_index}/{len(catalog)}] "
            f"{workload.workload_id} coverage={coverage_intent} required={required} "
            f"families={','.join(sorted({item[0]['family'] for item in selected}))} "
            f"reserve={len(reserves)} pool={len(pool)} "
            f"host_ms={(time.perf_counter_ns() - shape_started) / 1e6:.3f}",
            flush=True,
        )

    if len(selected_workloads) != 70 or formal_count != 2185:
        raise old.SearchError(
            f"campaign contract mismatch shapes={len(selected_workloads)} formal={formal_count}"
        )
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
        "MATMUL_HARDWARE_CALIBRATION_CANDIDATES "
        f"shapes=70 scheduled={len(selected_rows)} formal={formal_count} "
        f"reserves={len(selected_rows) - formal_count} baselines=70 records=2255 "
        f"by_family={dict(family_counts)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (old.SearchError, OSError, ValueError, KeyError) as exception:
        print(f"fatal: {exception}", flush=True)
        raise SystemExit(1)
