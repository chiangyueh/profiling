#!/usr/bin/env python3
"""Select one MatMul tiling using only the hardware simulator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import refine_matmul_v3_candidates as old
from profile_official_tilings import IncrementalJsonl
from npu_cost_model import (
    CANN81_MATMUL_FAMILIES,
    CANN81_MATMUL_KERNEL_SUFFIXES,
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


class IncrementalCsv:
    """Write a large audit table incrementally and publish it atomically."""

    def __init__(self, destination: Path, fields: list[str]) -> None:
        self.destination = destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.partial = destination.with_name(f"{destination.name}.partial")
        self.stream = self.partial.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.stream, fieldnames=fields, extrasaction="ignore"
        )
        self.writer.writeheader()
        self.count = 0

    def write(self, row: dict[str, str]) -> None:
        self.writer.writerow(row)
        self.count += 1

    def flush(self) -> None:
        self.stream.flush()

    def publish(self) -> None:
        self.stream.close()
        self.partial.replace(self.destination)


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def truthy(value: object) -> bool:
    return str(value or "").lower() in {"1", "true", "yes", "on"}


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
        ScheduleSpace(core_options=tuple(range(1, hardware.aic_cores + 1))),
        SearchPolicy(top_k=1, max_evaluations=100000),
    )
    if not region.exhaustive:
        raise old.SearchError(
            "hardware ideal-region derivation reached its safety ceiling; "
            "refusing to return a truncated search result"
        )
    result: list[dict[str, int]] = []
    seen: set[tuple[int, ...]] = set()
    for plan in region.plans:
        try:
            knowledge = lower_plan_to_cann(
                workload.m, workload.n, workload.k, workload.dtype,
                workload.trans_a, workload.trans_b, plan, generic,
            )
        except ValueError:
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


@lru_cache(maxsize=8)
def generic_hardware(platform: old.Hardware):
    base = ascend_910b3()
    core_counts = dict(base.core_counts)
    core_counts[Resource.CUBE] = platform.aic_cores
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
        plan = plan_from_cann(
            workload.m, workload.n, workload.k, knowledge,
            dtype=workload.dtype,
            trans_a=workload.trans_a,
            trans_b=workload.trans_b,
        )
        equivalence_key = (
            plan.algorithm,
            plan.axis_tiles,
            plan.task_tiles,
            plan.invocation_tiles,
            plan.transfer_tiles,
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

    return source_kernel_suffix(
        workload.m, workload.n, workload.k, workload.dtype,
        workload.trans_a, workload.trans_b, knowledge,
    )


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
        "model_kernel_variant": execution_mode_name(knowledge).upper(),
        "model_kernel_family": execution_mode_name(knowledge).upper(),
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
    all_rows = IncrementalCsv(args.all_output, fields)
    execution_graph_counts: dict[str, int] = {}
    generated_graph_counts: Counter[str] = Counter()
    generated_kernel_suffixes: set[int] = set()
    legal_graph_counts: Counter[str] = Counter()
    legal_kernel_suffixes: set[int] = set()
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
        for proposal in proposals:
            generated_graph_counts[execution_mode_name(proposal)] += 1
            generated_kernel_suffixes.add(kernel_suffix_for(
                search_workload, proposal
            ))
        candidate_generation_ns = time.perf_counter_ns() - started
        generated_count = len(proposals)

        started = time.perf_counter_ns()
        legal: list[dict[str, int]] = []
        legal_signatures: set[tuple[int, ...]] = set()
        for knowledge in proposals:
            signature = old.knowledge_signature(knowledge)
            if (
                signature in legal_signatures
                or validate_cann_tiling(
                    search_workload.m, search_workload.n, search_workload.k,
                    search_workload.dtype, search_workload.trans_a,
                    search_workload.trans_b, knowledge,
                    generic_hardware(platform),
                )
            ):
                continue
            legal_signatures.add(signature)
            legal.append(knowledge)
        static_legality_ns = time.perf_counter_ns() - started

        ranked, new_score_ns = ranked_pool(search_workload, legal, platform)
        if not ranked:
            continue
        for item in ranked:
            legal_graph_counts[execution_mode_name(item["knowledge"])] += 1
            legal_kernel_suffixes.add(kernel_suffix_for(
                search_workload, item["knowledge"]
            ))

        selected = ranked[0]
        started = time.perf_counter_ns()

        def materialize_model_row(item: dict, model_rank: int) -> dict[str, str]:
            knowledge = item["knowledge"]
            result = item["new"]
            suffix = kernel_suffix_for(callback_workload, knowledge)
            model_row = old.row_from_state(
                fields, None, callback_workload, knowledge,
                "generic_hardware_simulator_v1",
                execution_mode_name(knowledge).upper(),
                result.total_cycles,
                result.gm_read_bytes + result.gm_write_bytes,
                result.l2_bytes,
                10_000_000_000_000_000_000 + suffix,
                guidance=item["selection"], estimate=None,
                bottleneck=result.bottleneck,
                rationale=(
                    "lowest predicted cycles from the hardware simulator"
                    if model_rank == 1 else
                    "hardware-simulator ranked legal alternative"
                ),
                resume_policy="allow_new",
            )
            model_row["execution_mode"] = execution_mode_name(knowledge)
            attach_execution_identity(model_row, callback_workload, knowledge)
            model_row.update({
                "rank": str(model_rank),
                "search_model_score": f"{result.total_cycles:.12g}",
                "search_model_cycles": f"{result.total_cycles:.12g}",
                "search_model_raw_ratio_vs_bank_seed": "",
                "search_model_ratio_vs_bank_seed": "",
                "new_model_ratio_vs_official": "",
                "search_model_breakdown": new_breakdown(result),
            })
            attach_models(
                model_row,
                result,
                new_rank=model_rank,
                new_ns=new_score_ns,
                hardware_aic_cores=args.aic_cores,
                pool_sequence=item["pool_sequence"],
                pool_size=len(legal),
                selection=item["selection"],
            )
            return model_row

        row = materialize_model_row(selected, 1)
        execution_abi_ns = time.perf_counter_ns() - started
        total_ns = time.perf_counter_ns() - shape_started
        timing_values = {
            "candidate_generation_ms": candidate_generation_ns / 1e6,
            "static_legality_ms": static_legality_ns / 1e6,
            "execution_abi_ms": execution_abi_ns / 1e6,
        }
        common_row_values = {
            **{name: f"{value:.9g}" for name, value in timing_values.items()},
            "generated_candidate_count": str(generated_count),
            "legal_candidate_count": str(len(legal)),
            "simulator_valid_candidate_count": str(len(ranked)),
            "ideal_anchor_count": str(len(ideal_region.anchors)),
            "ideal_region_count": str(len(ideal_region.plans)),
            "ideal_discovery_evaluations": str(ideal_region.evaluated),
            "execution_graphs_represented": ";".join(sorted({
                execution_mode_name(item) for item in proposals
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
        }
        row.update(common_row_values)
        selected_rows.append(dict(row))
        all_rows.write(dict(row))
        for model_rank, item in enumerate(ranked[1:], 2):
            alternative = materialize_model_row(item, model_rank)
            alternative.update(common_row_values)
            alternative["search_model_cycles"] = (
                f"{item['new'].total_cycles:.12g}"
            )
            alternative["search_model_breakdown"] = new_breakdown(item["new"])
            all_rows.write(alternative)
        all_rows.flush()
        selected_workloads.append(metadata)
        selected_graph = execution_mode_name(selected["knowledge"])
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
                    execution_mode_name(item) for item in proposals
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
    missing_graphs = set(CANN81_MATMUL_FAMILIES) - set(legal_graph_counts)
    missing_suffixes = (
        set(CANN81_MATMUL_KERNEL_SUFFIXES) - legal_kernel_suffixes
    )
    if missing_graphs or missing_suffixes:
        raise old.SearchError(
            "CANN 8.1 family coverage is incomplete: "
            f"missing_graphs={sorted(missing_graphs)} "
            f"missing_suffixes={sorted(missing_suffixes)}"
        )
    if not required_anchors <= {row["workload_id"] for row in selected_workloads}:
        raise old.SearchError("one or more colleague anchor shapes were not admitted")
    args.workloads.parent.mkdir(parents=True, exist_ok=True)
    with args.workloads.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=catalog_fields)
        writer.writeheader()
        writer.writerows(selected_workloads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected_rows)
    all_rows.publish()
    print(
        f"MATMUL_MODEL_VALIDATION_CANDIDATES shapes={len(selected_workloads)} "
        "tilings_per_shape=1 "
        f"measured_per_shape={MEASURED_TILINGS} "
        f"selected_execution_graphs={execution_graph_counts} "
        f"generated_execution_graphs={dict(generated_graph_counts)} "
        f"generated_kernel_suffixes={sorted(generated_kernel_suffixes)} "
        f"legal_execution_graphs={dict(legal_graph_counts)} "
        f"legal_kernel_suffixes={sorted(legal_kernel_suffixes)} "
        f"full_ranked_rows={all_rows.count} "
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
            "generated_execution_graphs": dict(generated_graph_counts),
            "generated_kernel_suffixes": sorted(generated_kernel_suffixes),
            "legal_execution_graphs": dict(legal_graph_counts),
            "legal_kernel_suffixes": sorted(legal_kernel_suffixes),
            "full_ranked_rows": all_rows.count,
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
