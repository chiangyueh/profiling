#!/usr/bin/env python3
"""Build one callback-fixed pool scored by both MatMul cost models."""

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
import generate_matmul_controlled_candidates as controlled
from npu_cost_model import (
    MemorySpace,
    Resource,
    ascend_910b3,
    plan_from_cann,
    simulate,
)
from npu_cost_model.operators import matmul


SELECTED_WORKLOADS = 200
SEARCHED_CANDIDATES = 31
MEASURED_TILINGS = 24
MODEL_FRONTIER = 6
GEOMETRY_LIMIT = 192
CUSTOM_COLUMNS = (
    "search_core_cap",
    "pool_sequence",
    "pool_size",
    "pool_selection",
    "old_model_cycles",
    "old_model_rank",
    "old_model_breakdown",
    "new_model_cycles",
    "new_model_rank",
    "new_model_bottleneck",
    "new_model_breakdown",
    "old_model_score_ns",
    "new_model_score_ns",
    "model_input_source",
)


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def truthy(value: object) -> bool:
    return str(value or "").lower() in {"1", "true", "yes", "on"}


def axis_values(extent: int, core_cap: int) -> tuple[int, ...]:
    values = {16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256}
    for parts in {1, 2, 3, 4, 5, 8, 10, 12, 16, 20, core_cap}:
        value = old.align_up(old.ceil_div(extent, parts), 16)
        if value <= 256:
            values.add(value)
    return tuple(sorted(value for value in values if value <= max(16, extent)))


def core_values(core_cap: int, tasks: int) -> tuple[int, ...]:
    return tuple(sorted(
        value for value in {1, 2, 3, 4, 5, 8, 10, 12, 16, 20, core_cap, tasks}
        if value <= core_cap and value <= tasks
    ))


def geometry_values(
    workload: old.Workload,
    core_cap: int,
) -> list[tuple[int, int, int]]:
    k0 = old.base_k_alignment(workload)
    k_values = tuple(k0 * factor for factor in (1, 2, 4, 8, 16))
    combinations = [
        (base_m, base_n, base_k)
        for base_m in axis_values(workload.m, core_cap)
        for base_n in axis_values(workload.n, core_cap)
        for base_k in k_values
    ]
    combinations.sort(key=lambda item: (
        (item[0] * 73856093 ^ item[1] * 19349663 ^ item[2] * 83492791)
        % 2147483647,
        item,
    ))
    preferred = [
        item for item in combinations
        if item[0] in (64, 128, 256)
        and item[1] in (64, 128, 256)
        and item[2] in (4 * k0, 8 * k0)
    ]
    result: list[tuple[int, int, int]] = []
    for item in (*preferred, *combinations):
        if item not in result:
            result.append(item)
        if len(result) == GEOMETRY_LIMIT:
            break
    return result


def l2_profiles(m_parts: int, n_parts: int, cores: int) -> tuple[tuple[int, int, int], ...]:
    ideal_m = max(1, min(m_parts, round(math.sqrt(
        cores * m_parts / max(1, n_parts)
    ))))
    profiles = {
        (m_parts, n_parts, 0),
        (ideal_m, max(1, min(n_parts, old.ceil_div(cores, ideal_m))), 0),
        (min(m_parts, cores), 1, 1),
        (1, min(n_parts, cores), 2),
    }
    return tuple(sorted(profiles))


def make_base(
    workload: old.Workload,
    hardware: old.Hardware,
    base_m: int,
    base_n: int,
    base_k: int,
    cores: int,
    l2_m: int,
    l2_n: int,
    l2_order: int,
    iterate_order: int,
    buffers: tuple[int, int, int],
) -> dict[str, int] | None:
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
        "dbL0A": buffers[0],
        "dbL0B": buffers[1],
        "dbL0C": buffers[2],
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


def proposal_space(
    workload: old.Workload,
    hardware: old.Hardware,
    core_cap: int,
) -> list[dict[str, int]]:
    result: list[dict[str, int]] = []
    seen: set[tuple[int, ...]] = set()
    for base_m, base_n, base_k in geometry_values(workload, core_cap):
        m_parts = old.ceil_div(workload.m, base_m)
        n_parts = old.ceil_div(workload.n, base_n)
        maximum_cores = min(core_cap, m_parts * n_parts)
        if maximum_cores <= 0:
            continue

        specs: list[tuple[int, int, int, int, int, tuple[int, int, int]]] = []
        for cores in core_values(core_cap, m_parts * n_parts):
            specs.append((cores, m_parts, n_parts, 0, 0, (2, 2, 1)))
        for l2_m, l2_n, l2_order in l2_profiles(m_parts, n_parts, maximum_cores):
            specs.append((maximum_cores, l2_m, l2_n, l2_order, 0, (2, 2, 1)))
        specs.extend(
            (maximum_cores, m_parts, n_parts, 0, order, buffers)
            for order in (0, 1)
            for buffers in ((1, 1, 1), (2, 2, 1), (2, 2, 2))
        )
        for cores, l2_m, l2_n, l2_order, order, buffers in specs:
            knowledge = make_base(
                workload, hardware, base_m, base_n, base_k, cores,
                l2_m, l2_n, l2_order, order, buffers,
            )
            if knowledge is None:
                continue
            signature = old.knowledge_signature(knowledge)
            if signature in seen:
                continue
            seen.add(signature)
            result.append(knowledge)
    return result


def proposal_space_for(
    workload: old.Workload,
    seed: old.Seed,
    hardware: old.Hardware,
    core_cap: int,
    search_family: str,
) -> list[dict[str, int]]:
    if search_family == "base":
        return proposal_space(workload, hardware, core_cap)
    if search_family == "deterministic_split_k":
        return [
            knowledge
            for knowledge, _, _, _
            in controlled.deterministic_splitk_candidates(workload, hardware)
            if knowledge["usedCoreNum"] <= core_cap
        ]
    raise old.SearchError(f"unknown model-validation search family: {search_family}")


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
) -> tuple[list[dict], int, int]:
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
    old_ns = 0
    new_ns = 0
    for sequence, knowledge in enumerate(proposals, 1):
        started = time.perf_counter_ns()
        old_result = old.analytical_score(workload, knowledge, platform)
        old_ns += time.perf_counter_ns() - started
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
            "old": old_result,
            "new": new_result,
        })
    old_order = sorted(scored, key=lambda item: (
        item["old"].cycles, old.knowledge_signature(item["knowledge"])
    ))
    new_order = sorted(scored, key=lambda item: (
        item["new"].total_cycles, old.knowledge_signature(item["knowledge"])
    ))
    for rank, item in enumerate(old_order, 1):
        item["old_rank_all"] = rank
    for rank, item in enumerate(new_order, 1):
        item["new_rank_all"] = rank

    priority: list[dict] = []
    source: dict[tuple[int, ...], set[str]] = {}
    for rank in range(MODEL_FRONTIER):
        for name, ordered in (("new_frontier", new_order), ("old_frontier", old_order)):
            if rank >= len(ordered):
                continue
            item = ordered[rank]
            signature = old.knowledge_signature(item["knowledge"])
            source.setdefault(signature, set()).add(name)
            if item not in priority:
                priority.append(item)

    diversity = sorted(scored, key=lambda item: (
        item["knowledge"]["usedCoreNum"],
        item["knowledge"]["baseM"] * item["knowledge"]["baseN"],
        item["knowledge"]["baseK"],
        item["knowledge"]["l2IterateOrder"],
        old.knowledge_signature(item["knowledge"]),
    ))
    if diversity:
        for index in range(12):
            item = diversity[round(index * (len(diversity) - 1) / 11)]
            signature = old.knowledge_signature(item["knowledge"])
            source.setdefault(signature, set()).add("hardware_coverage")
            if item not in priority:
                priority.append(item)
    for item in sorted(scored, key=lambda value: (
        min(value["old_rank_all"], value["new_rank_all"]),
        max(value["old_rank_all"], value["new_rank_all"]),
    )):
        if item not in priority:
            priority.append(item)
    for item in priority:
        item["selection"] = "+".join(sorted(
            source.get(old.knowledge_signature(item["knowledge"]), {"backup"})
        ))
    return priority, old_ns, new_ns


def attach_models(
    row: dict[str, str],
    old_result,
    new_result,
    *,
    old_rank: int,
    new_rank: int,
    old_ns: int,
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
        "old_model_cycles": f"{old_result.cycles:.12g}",
        "old_model_rank": str(old_rank),
        "old_model_breakdown": old_result.breakdown(),
        "new_model_cycles": f"{new_result.total_cycles:.12g}",
        "new_model_rank": str(new_rank),
        "new_model_bottleneck": new_result.bottleneck,
        "new_model_breakdown": new_breakdown(new_result),
        "old_model_score_ns": str(old_ns),
        "new_model_score_ns": str(new_ns),
        "model_input_source": "parameters_only_no_latency_history_no_cce_table",
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
    parser.add_argument("--splitk-workloads", type=int)
    args = parser.parse_args()

    if args.selected_workloads <= 0 or args.searched_candidates <= 0:
        raise old.SearchError("selected workload and candidate counts must be positive")

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
    split_target = (
        args.splitk_workloads
        if args.splitk_workloads is not None
        else 40 if args.selected_workloads >= 200 else 0
    )
    if not 0 <= split_target <= args.selected_workloads:
        raise old.SearchError("split-K workload count is outside the selected total")
    family_targets = {
        "base": args.selected_workloads - split_target,
        "deterministic_split_k": split_target,
    }
    family_counts = {name: 0 for name in family_targets}
    anchor_ids = {"matmul_rank_000", "matmul_rank_001", "matmul_rank_002"}
    catalog_ids = {row["workload_id"] for row in catalog_rows}
    required_anchors = (
        anchor_ids
        if args.selected_workloads >= 3 and anchor_ids <= catalog_ids
        else set()
    )

    for catalog_index, metadata in enumerate(catalog_rows, 1):
        if len(selected_workloads) >= args.selected_workloads:
            break
        search_family = metadata.get("search_family", "base")
        if family_counts.get(search_family, 0) >= family_targets.get(
            search_family, 0
        ):
            continue
        callback_workload = old.Workload(
            metadata["workload_id"], int(metadata["m"]), int(metadata["n"]),
            int(metadata["k"]), metadata["dtype"], truthy(metadata["trans_a"]),
            truthy(metadata["trans_b"]), args.aic_cores,
        )
        core_cap = min(int(metadata["search_core_cap"]), args.aic_cores)
        search_workload = replace(callback_workload, max_cores=core_cap)
        seed_started = time.perf_counter_ns()
        seed = old.parse_seed(callback_workload)
        official_family = old.template_name(seed.bank.knowledge)
        if official_family not in {"BASE", "DETERMINISTIC_SPLIT_K"}:
            print(
                f"MODEL_VALIDATION_SKIP [{catalog_index}/{len(catalog_rows)}] "
                f"{metadata['workload_id']} official_family="
                f"{official_family}",
                flush=True,
            )
            continue
        proposals = proposal_space_for(
            search_workload, seed, platform, core_cap, search_family
        )
        seed_signature = old.knowledge_signature(seed.bank.knowledge)
        proposals = [
            knowledge for knowledge in proposals
            if old.knowledge_signature(knowledge) != seed_signature
        ]
        priority, old_score_ns, new_score_ns = ranked_pool(
            search_workload, proposals, platform
        )
        if not priority:
            continue

        seed_old = old.analytical_score(
            callback_workload, seed.bank.knowledge, platform
        )
        seed_operator = matmul(
            callback_workload.m,
            callback_workload.n,
            callback_workload.k,
            callback_workload.dtype,
            trans_a=callback_workload.trans_a,
            trans_b=callback_workload.trans_b,
        )
        seed_new = simulate(
            seed_operator,
            plan_from_cann(
                callback_workload.m,
                callback_workload.n,
                callback_workload.k,
                seed.bank.knowledge,
            ),
            generic_hardware(platform),
        )
        if not seed_new.valid:
            continue
        control = old.bank_seed_control(fields, callback_workload, seed, platform)
        accepted: list[dict] = []
        callback_failures = 0
        callback_duplicates = 0
        callback_ns = 0
        callback_hashes = {seed.bank.sha256}
        for item in priority:
            state = old.State(
                row=old.row_from_state(
                    fields, None, callback_workload, item["knowledge"],
                    "paired_parameter_frontier_v1",
                    old.template_name(item["knowledge"]),
                    item["old"].cycles / max(1.0, seed_old.cycles),
                    item["old"].hbm_bytes, item["old"].l2_bytes, seed.key,
                    guidance=item["selection"], estimate=item["old"],
                    bottleneck="paired_model_validation",
                    rationale="shared legal parameter pool for old/new model ranking",
                    resume_policy="allow_new",
                ),
                knowledge=item["knowledge"],
                model_score=item["old"].cycles,
                normalized_score=item["old"].cycles / max(1.0, seed_old.cycles),
                hbm_bytes=item["old"].hbm_bytes,
                l2_bytes=item["old"].l2_bytes,
                template=old.template_name(item["knowledge"]),
                guidance=item["selection"],
                estimate=item["old"],
            )
            started = time.perf_counter_ns()
            try:
                callback = old.validate_callback(callback_workload, state)
            except Exception:
                callback_ns += time.perf_counter_ns() - started
                callback_failures += 1
                continue
            callback_ns += time.perf_counter_ns() - started
            if callback.sha256 in callback_hashes:
                callback_duplicates += 1
                continue
            callback_hashes.add(callback.sha256)
            old.update_callback_columns(
                state, callback, seed, seed_old, platform, callback_workload
            )
            item["state"] = state
            accepted.append(item)
            if len(accepted) == args.searched_candidates:
                break
        if len(accepted) < args.searched_candidates:
            print(
                f"MODEL_VALIDATION_SKIP [{catalog_index}/{len(catalog_rows)}] "
                f"{metadata['workload_id']} callback_fixed={len(accepted)} "
                f"required={args.searched_candidates} callback_rejected={callback_failures} "
                f"callback_duplicates={callback_duplicates}",
                flush=True,
            )
            continue

        combined = [
            {
                "row": control.row,
                "old": seed_old,
                "new": seed_new,
                "pool_sequence": 0,
                "selection": "official_autotiling_control",
            },
            *(
                {
                    "row": item["state"].row,
                    "old": item["old"],
                    "new": item["new"],
                    "pool_sequence": item["pool_sequence"],
                    "selection": item["selection"],
                }
                for item in accepted
            ),
        ]
        old_order = {
            id(item): rank for rank, item in enumerate(
                sorted(combined, key=lambda value: value["old"].cycles), 1
            )
        }
        new_order = {
            id(item): rank for rank, item in enumerate(
                sorted(combined, key=lambda value: value["new"].total_cycles), 1
            )
        }
        total_score_ns = max(1, old_score_ns + new_score_ns)
        for execution_rank, item in enumerate(combined):
            row = item["row"]
            row["rank"] = str(execution_rank)
            attach_models(
                row, item["old"], item["new"],
                old_rank=old_order[id(item)], new_rank=new_order[id(item)],
                old_ns=old_score_ns, new_ns=new_score_ns,
                core_cap=core_cap, pool_sequence=item["pool_sequence"],
                pool_size=len(proposals), selection=item["selection"],
            )
            row["tiling_solver_select_ms"] = f"{total_score_ns / 1e6:.9g}"
            row["tiling_solver_callback_ms"] = f"{callback_ns / 1e6:.9g}"
            row["tiling_solver_callback_count"] = str(
                len(accepted) + callback_failures
            )
            total_ms = (time.perf_counter_ns() - seed_started) / 1e6
            row["tiling_solver_total_ms"] = f"{total_ms:.9g}"
            selected_rows.append(dict(row))
            all_rows.append(dict(row))
        selected_workloads.append(metadata)
        family_counts[search_family] += 1
        print(
            f"MODEL_VALIDATION_CANDIDATES [{len(selected_workloads)}/{args.selected_workloads}] "
            f"{metadata['workload_id']} shape={metadata['m']}x{metadata['n']}x{metadata['k']} "
            f"dtype={metadata['dtype']} trans={metadata['trans_a']}{metadata['trans_b']} "
            f"core_cap={core_cap} pool={len(proposals)} callback_fixed={len(accepted)} "
            f"family={search_family} callback_rejected={callback_failures} "
            f"callback_duplicates={callback_duplicates}",
            flush=True,
        )

    if len(selected_workloads) != args.selected_workloads:
        raise old.SearchError(
            f"only {len(selected_workloads)} BASE workloads have {args.searched_candidates} "
            f"callback-fixed candidates; required {args.selected_workloads}"
        )
    if not required_anchors <= {row["workload_id"] for row in selected_workloads}:
        raise old.SearchError("one or more colleague anchor shapes were not admitted")
    if family_counts != family_targets:
        raise old.SearchError(
            f"selected family counts {family_counts}, required {family_targets}"
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
        f"MATMUL_MODEL_VALIDATION_CANDIDATES shapes={len(selected_workloads)} "
        f"tilings_per_shape={args.searched_candidates + 1} "
        f"measured_per_shape={MEASURED_TILINGS} "
        f"families={family_counts}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (old.SearchError, OSError, ValueError, KeyError) as exception:
        print(f"fatal: {exception}", flush=True)
        raise SystemExit(1)
