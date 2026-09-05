#!/usr/bin/env python3
"""Create a preregistered, model-independent MatMul hardware frontier.

Candidate identities are selected from CANN-legal parameter space using only
hardware capacities and structural strata.  The cost simulator is invoked
only after the measured set has been frozen, so its score cannot influence
which tilings are sent to the NPU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict, deque
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
    ascend_910b3,
    cann81_matmul_effective_l1_bytes,
    execution_mode_name,
    plan_from_cann,
    simulate,
    source_kernel_suffix,
    validate_cann_tiling,
)
from npu_cost_model.operators import matmul


RESERVES_PER_SHAPE = 32
FORMAL_GEOMETRY = 300
FORMAL_PAIRED = 300
FORMAL_INTERACTION = 60
FORMAL_SERIAL_SPLIT_K = 40
FORMAL_DETERMINISTIC_SPLIT_K = 20
PAIRED_FACTOR_QUOTAS = {
    "core_parallelism": 100,
    "k_pipeline": 95,
    "l0_buffering": 37,
    "l2_partition": 56,
    "cube_traversal": 12,
}
CUSTOM_COLUMNS = (
    "hardware_aic_cores", "new_model_cycles", "new_model_rank",
    "global_model_rank", "new_model_bottleneck", "new_model_breakdown",
    "model_schedule_sha256", "model_kernel_suffix", "model_kernel_variant",
    "model_kernel_family", "model_input_source", "controlled_factor",
    "factor_signature", "pair_id", "design_role", "hardware_stratum",
    "selection_basis", "candidate_set_frozen_before_model_scoring",
    "is_reserve", "coverage_intent", "calibration_partition",
    "required_successful_tilings", "generated_candidate_count",
    "legal_candidate_count", "candidate_generation_ms",
    "static_legality_ms", "simulator_scoring_ms", "tiling_solver_select_ms",
    "tiling_solver_extra_ms", "tiling_solver_total_ms",
)
MANDATORY_COLUMNS = (
    "workload_id", "m", "n", "k", "dtype", "trans_a", "trans_b",
    "max_cores", "source", "candidate_role", "source_iteration", "valid",
    "error", "execution_mode", "candidate_single_core_m",
    "candidate_single_core_n", "candidate_single_core_k", "candidate_base_m",
    "candidate_base_n", "candidate_base_k", "candidate_traverse",
    "candidate_db_a", "candidate_db_b", "candidate_split_k", "used_core_num",
    "official_core_num", "official_m_dim", "official_n_dim", "m_core_parts",
    "n_core_parts", "k_core_parts", "single_core_m", "single_core_n",
    "single_core_k", "base_m", "base_n", "base_k", "depth_a1", "depth_b1",
    "step_m", "step_n", "step_ka", "step_kb", "iterate_order", "db_l0a",
    "db_l0b", "db_l0c", "proxy_total", "official_return",
    "tiling_signature", "tiling_bin",
)
OBSOLETE_EXECUTION_COLUMNS = {
    name for name in old.EXTRA_COLUMNS
    if "callback" in name or "runtime_kb" in name or "bank_seed" in name
}
BMN_VALUES = tuple(range(16, 257, 16)) + (288, 320, 384, 448, 512)
BK_VALUES = (16, 32, 64, 96, 128, 192, 256)


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
        base, core_counts=cores, capacities=capacities,
        aggregate_hbm_bytes_per_cycle=(
            platform.hbm_bytes_per_cycle_per_core * platform.aic_cores
        ),
        aggregate_l2_bytes_per_cycle=(
            platform.l2_bytes_per_cycle_per_core * platform.aic_cores
        ),
    )


def signature(knowledge: dict[str, int]) -> tuple[int, ...]:
    return tuple(int(knowledge[field]) for field in old.KNOWLEDGE_FIELDS)


def l2_schedule(
    workload: old.Workload,
    knowledge: dict[str, int],
    m_block: int,
    n_block: int,
    order: int,
) -> dict[str, int]:
    result = dict(knowledge)
    m_total = old.ceil_div(workload.m, result["singleCoreM"])
    n_total = old.ceil_div(workload.n, result["singleCoreN"])
    m_block = max(1, min(m_total, m_block))
    n_block = max(1, min(n_total, n_block))
    result.update(
        l2MTileCnt=old.ceil_div(m_total, m_block),
        l2NTileCnt=old.ceil_div(n_total, n_block),
        l2MTileBlock=m_block,
        l2NTileBlock=n_block,
        l2IterateOrder=order,
    )
    return result


def base_knowledge(
    workload: old.Workload, base_m: int, base_n: int, base_k: int,
    aic_cores: int,
) -> dict[str, int]:
    m_total = old.ceil_div(workload.m, base_m)
    n_total = old.ceil_div(workload.n, base_n)
    return {
        "usedCoreNum": min(aic_cores, workload.max_cores, m_total * n_total),
        "singleCoreM": base_m, "singleCoreN": base_n,
        "singleCoreK": workload.k,
        "baseM": base_m, "baseN": base_n, "baseK": base_k,
        "depthA1": 1, "depthB1": 1,
        "stepM": 1, "stepN": 1, "iterateOrder": 0,
        "stepKa": 1, "stepKb": 1,
        "dbL0A": 1, "dbL0B": 1, "dbL0C": 1,
        "l2MTileCnt": 1, "l2NTileCnt": 1,
        "l2MTileBlock": m_total, "l2NTileBlock": n_total,
        "l2IterateOrder": 0, "tilingEnable": 0,
    }


def hardware_stratum(
    workload: old.Workload, knowledge: dict[str, int], hardware,
) -> str:
    base_m, base_n, base_k = (
        knowledge["baseM"], knowledge["baseN"], knowledge["baseK"]
    )
    l0c_ratio = (
        base_m * base_n * 4 * knowledge["dbL0C"]
        / hardware.capacities[MemorySpace.L0C]
    )
    aspect = math.log2(base_m / base_n)
    tasks = (
        old.ceil_div(workload.m, knowledge["singleCoreM"])
        * old.ceil_div(workload.n, knowledge["singleCoreN"])
    )
    waves = old.ceil_div(tasks, max(1, knowledge["usedCoreNum"]))
    return (
        f"family={execution_mode_name(knowledge)};bk={base_k};"
        f"aspect={max(-3, min(3, math.floor(aspect)))};"
        f"l0c={min(3, int(l0c_ratio * 4))};waves={min(4, waves)}"
    )


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


def legal(
    workload: old.Workload, knowledge: dict[str, int],
    platform: old.Hardware, hardware,
) -> bool:
    if not old.hard_legal(workload, knowledge, platform):
        return False
    if validate_cann_tiling(
        workload.m, workload.n, workload.k, workload.dtype,
        workload.trans_a, workload.trans_b, knowledge, hardware,
        aoe_injection=True,
    ):
        return False
    try:
        plan_from_cann(
            workload.m, workload.n, workload.k, knowledge,
            dtype=workload.dtype, trans_a=workload.trans_a,
            trans_b=workload.trans_b,
        )
    except (KeyError, ValueError):
        return False
    return True


def partition_values(total: int) -> tuple[int, ...]:
    values = {1, total}
    for divisor in (2, 3, 4, 5, 8, 10, 16, 20):
        values.add(max(1, old.ceil_div(total, divisor)))
    return tuple(sorted(values))


def pick_anchors(
    canonical: list[dict], hardware,
) -> list[dict]:
    targets = (
        (0.18, -3.0, 32), (0.18, 0.0, 64), (0.18, 3.0, 128),
        (0.45, -2.0, 64), (0.45, 0.0, 128), (0.45, 2.0, 32),
        (0.78, -2.0, 128), (0.78, 0.0, 32), (0.78, 2.0, 64),
        (0.95, -1.0, 64), (0.95, 0.0, 128), (0.95, 1.0, 32),
    )
    chosen: list[dict] = []
    seen: set[tuple[int, ...]] = set()
    l0c = hardware.capacities[MemorySpace.L0C]
    for occupancy, aspect, base_k in targets:
        ranked = sorted(canonical, key=lambda item: (
            abs(item["knowledge"]["baseM"] * item["knowledge"]["baseN"] * 4 / l0c - occupancy)
            + 0.12 * abs(math.log2(
                item["knowledge"]["baseM"] / item["knowledge"]["baseN"]
            ) - aspect)
            + 0.003 * abs(item["knowledge"]["baseK"] - base_k),
            signature(item["knowledge"]),
        ))
        selected = next(
            item for item in ranked if signature(item["knowledge"]) not in seen
        )
        seen.add(signature(selected["knowledge"]))
        chosen.append(selected)
    return chosen


def round_robin_strata(items: list[dict], count: int, key) -> list[dict]:
    queues: dict[object, deque[dict]] = defaultdict(deque)
    for item in sorted(items, key=lambda value: signature(value["knowledge"])):
        queues[key(item)].append(item)
    result: list[dict] = []
    keys = sorted(queues, key=str)
    while len(result) < count:
        progressed = False
        for group in keys:
            if queues[group] and len(result) < count:
                result.append(queues[group].popleft())
                progressed = True
        if not progressed:
            break
    return result


def proposed_candidates(
    workload: old.Workload, platform: old.Hardware, hardware,
) -> tuple[list[dict], list[dict], int]:
    items: dict[tuple[int, ...], dict] = {}
    legality_ns = 0

    def add(
        knowledge: dict[str, int], role: str, factor: str,
        pair_id: str = "", factor_value: str = "",
    ) -> None:
        nonlocal legality_ns
        key = signature(knowledge)
        if key in items:
            return
        started = time.perf_counter_ns()
        accepted = legal(workload, knowledge, platform, hardware)
        legality_ns += time.perf_counter_ns() - started
        if not accepted:
            return
        items[key] = {
            "knowledge": dict(knowledge), "design_role": role,
            "controlled_factor": factor, "pair_id": pair_id,
            "factor_signature": factor_value or factor,
            "hardware_stratum": hardware_stratum(workload, knowledge, hardware),
        }

    canonical: list[dict] = []
    for base_m, base_n, base_k in product(BMN_VALUES, BMN_VALUES, BK_VALUES):
        knowledge = base_knowledge(
            workload, base_m, base_n, base_k, platform.aic_cores
        )
        add(
            knowledge, "broad_geometry", "mnk_geometry", "",
            f"baseM={base_m}:baseN={base_n}:baseK={base_k}",
        )
        item = items.get(signature(knowledge))
        if item is not None:
            canonical.append(item)
    anchors = pick_anchors(canonical, hardware)
    anchor_keys = {signature(item["knowledge"]) for item in anchors}
    for index, anchor in enumerate(anchors):
        anchor["pair_id"] = f"anchor_{index:02d}"
        anchor["design_role"] = "paired_anchor"
        anchor["controlled_factor"] = "anchor"
        anchor["factor_signature"] = "reference"
        seed = anchor["knowledge"]
        pair_id = anchor["pair_id"]
        m_total = old.ceil_div(workload.m, seed["singleCoreM"])
        n_total = old.ceil_div(workload.n, seed["singleCoreN"])
        tasks = m_total * n_total

        for cores in range(1, min(platform.aic_cores, tasks) + 1):
            candidate = dict(seed)
            candidate["usedCoreNum"] = cores
            add(candidate, "paired_factor", "core_parallelism", pair_id,
                f"usedCoreNum={cores}")

        step_pairs = (
            (1, 1), (1, 2), (2, 1), (2, 2), (2, 4), (4, 2),
            (4, 4), (4, 8), (8, 4), (8, 8), (8, 16), (16, 8),
            (16, 16), (16, 32), (32, 16), (32, 32),
        )
        legal_k_contexts: list[dict[str, int]] = []
        for step_ka, step_kb in step_pairs:
            for packets_a, packets_b in product((1, 2), repeat=2):
                candidate = dict(seed)
                candidate.update(
                    stepKa=step_ka, stepKb=step_kb,
                    depthA1=step_ka * packets_a,
                    depthB1=step_kb * packets_b,
                )
                before = len(items)
                add(candidate, "paired_factor", "k_pipeline", pair_id,
                    f"stepKa={step_ka}:stepKb={step_kb}:"
                    f"packetsA={packets_a}:packetsB={packets_b}")
                if len(items) > before:
                    legal_k_contexts.append(candidate)

        for db_a, db_b, db_c in product((1, 2), repeat=3):
            candidate = dict(seed)
            candidate.update(dbL0A=db_a, dbL0B=db_b, dbL0C=db_c)
            add(candidate, "paired_factor", "l0_buffering", pair_id,
                f"dbL0A={db_a}:dbL0B={db_b}:dbL0C={db_c}")

        for order in (0, 1):
            candidate = dict(seed)
            candidate["iterateOrder"] = order
            add(candidate, "paired_factor", "cube_traversal", pair_id,
                f"iterateOrder={order}")

        partitions = []
        for m_block in partition_values(m_total):
            partitions.append((m_block, n_total))
        for n_block in partition_values(n_total):
            partitions.append((m_total, n_block))
        for m_block, n_block in partitions:
            for order in (1, 2):
                candidate = l2_schedule(
                    workload, seed, m_block, n_block, order
                )
                add(candidate, "paired_factor", "l2_partition", pair_id,
                    f"mBlock={m_block}:nBlock={n_block}:order={order}")

        # A small balanced interaction design tests whether individual effects
        # compose; it is not selected by the cost score.
        k_contexts = legal_k_contexts[::max(1, len(legal_k_contexts) // 4)][:4]
        db_contexts = ((1, 1, 1), (2, 2, 1), (1, 1, 2), (2, 2, 2))
        core_contexts = tuple(sorted({
            min(platform.aic_cores, tasks),
            max(1, min(platform.aic_cores, tasks) // 2),
        }))
        for interaction_index, (k_context, dbs, cores, order) in enumerate(
            product(k_contexts, db_contexts, core_contexts, (0, 1))
        ):
            candidate = dict(k_context)
            candidate.update(
                dbL0A=dbs[0], dbL0B=dbs[1], dbL0C=dbs[2],
                usedCoreNum=cores, iterateOrder=order,
            )
            add(candidate, "factor_interaction", "interaction", pair_id,
                f"interaction={interaction_index}")

    # Serial split-K: source-visible schedules independent of the model.
    serial_algorithms = (
        (3, 1, 3, 9, 6, 1, 0),
        (1, 3, 3, 6, 9, 0, 1),
        (2, 1, 4, 8, 8, 1, 0),
        (1, 1, 4, 8, 8, 1, 0),
    )
    for algorithm, (step_m, step_n, step_k, depth_a, depth_b, order, l2_order) in enumerate(serial_algorithms):
        base_m = base_n = base_k = 128
        inner_m, inner_n = step_m * base_m, step_n * base_n
        m_values = sorted({
            inner_m, old.align_up(old.ceil_div(workload.m, 20), 16),
            old.align_up(old.ceil_div(workload.m, 10), 16),
            old.align_up(old.ceil_div(workload.m, 5), 16),
        })
        n_values = sorted({
            inner_n, old.align_up(old.ceil_div(workload.n, 20), 128),
            old.align_up(old.ceil_div(workload.n, 10), 128),
            old.align_up(old.ceil_div(workload.n, 5), 128),
        })
        for single_m, single_n in product(m_values, n_values):
            if single_m < inner_m or single_n < inner_n:
                continue
            m_total = old.ceil_div(workload.m, single_m)
            n_total = old.ceil_div(workload.n, single_n)
            for cores in sorted({1, min(5, m_total * n_total), min(10, m_total * n_total), min(20, m_total * n_total)}):
                candidate = {
                    "usedCoreNum": cores,
                    "singleCoreM": single_m, "singleCoreN": single_n,
                    "singleCoreK": step_k * base_k,
                    "baseM": base_m, "baseN": base_n, "baseK": base_k,
                    "depthA1": depth_a, "depthB1": depth_b,
                    "stepM": step_m, "stepN": step_n,
                    "iterateOrder": order, "stepKa": step_k, "stepKb": step_k,
                    "dbL0A": 2, "dbL0B": 2, "dbL0C": 2,
                    "l2MTileCnt": 1, "l2NTileCnt": 1,
                    "l2MTileBlock": m_total, "l2NTileBlock": n_total,
                    "l2IterateOrder": l2_order, "tilingEnable": 2,
                }
                add(candidate, "execution_graph_control", "serial_split_k", f"serial_{algorithm}",
                    f"algorithm={algorithm}:cores={cores}:singleM={single_m}:singleN={single_n}")

    # Deterministic parallel split-K has two exact 3x3 dataflow layouts.
    single_k = 384
    k_chunks = old.ceil_div(workload.k, single_k)
    layouts = (
        (3, 1, 9, 6, 1, 384, max(128, workload.n), 0),
        (1, 3, 6, 9, 0, max(128, workload.m), 384, 1),
    )
    for layout, (step_m, step_n, depth_a, depth_b, order, single_m, single_n, l2_order) in enumerate(layouts):
        m_total = old.ceil_div(workload.m, single_m)
        n_total = old.ceil_div(workload.n, single_n)
        for cores in range(2, min(platform.aic_cores, k_chunks) + 1):
            candidate = {
                "usedCoreNum": cores,
                "singleCoreM": single_m, "singleCoreN": single_n,
                "singleCoreK": single_k,
                "baseM": 128, "baseN": 128, "baseK": 128,
                "depthA1": depth_a, "depthB1": depth_b,
                "stepM": step_m, "stepN": step_n,
                "iterateOrder": order, "stepKa": 3, "stepKb": 3,
                "dbL0A": 2, "dbL0B": 2, "dbL0C": 2,
                "l2MTileCnt": 1, "l2NTileCnt": 1,
                "l2MTileBlock": m_total, "l2NTileBlock": n_total,
                "l2IterateOrder": l2_order, "tilingEnable": 3,
            }
            add(candidate, "execution_graph_control", "deterministic_split_k", f"deterministic_{layout}",
                f"layout={layout}:cores={cores}")

    return list(items.values()), anchors, legality_ns


def select_fixed_design(
    proposed: list[dict], anchors: list[dict], required: int,
) -> tuple[list[dict], list[dict]]:
    selected: list[dict] = []
    seen: set[tuple[int, ...]] = set()

    for item in anchors:
        key = signature(item["knowledge"])
        if key not in seen:
            selected.append(item)
            seen.add(key)

    geometry = [item for item in proposed if item["design_role"] in {"broad_geometry", "paired_anchor"}]
    geometry_pick = round_robin_strata(
        geometry, FORMAL_GEOMETRY,
        lambda item: item["hardware_stratum"],
    )
    for item in geometry_pick:
        if len([value for value in selected if value["design_role"] in {"broad_geometry", "paired_anchor"}]) >= FORMAL_GEOMETRY:
            break
        key = signature(item["knowledge"])
        if key not in seen:
            selected.append(item); seen.add(key)

    paired_added = 0
    for factor, quota in PAIRED_FACTOR_QUOTAS.items():
        pool = [
            item for item in proposed
            if item["design_role"] == "paired_factor"
            and item["controlled_factor"] == factor
        ]
        added = 0
        for item in round_robin_strata(
            pool, len(pool), lambda value: value["factor_signature"]
        ):
            key = signature(item["knowledge"])
            if key in seen:
                continue
            selected.append(item); seen.add(key); added += 1; paired_added += 1
            if added == quota:
                break
    if paired_added != FORMAL_PAIRED:
        raise old.SearchError(
            f"paired hardware design has {paired_added} candidates; "
            f"required {FORMAL_PAIRED}"
        )

    categories = (
        ("factor_interaction", FORMAL_INTERACTION, lambda item: item["pair_id"]),
        ("serial_split_k", FORMAL_SERIAL_SPLIT_K, lambda item: item["pair_id"]),
        ("deterministic_split_k", FORMAL_DETERMINISTIC_SPLIT_K, lambda item: item["pair_id"]),
    )
    for category, count, key_function in categories:
        if category in {"serial_split_k", "deterministic_split_k"}:
            pool = [item for item in proposed if item["controlled_factor"] == category]
        else:
            pool = [item for item in proposed if item["design_role"] == category]
        added = 0
        for item in round_robin_strata(pool, len(pool), key_function):
            key = signature(item["knowledge"])
            if key in seen:
                continue
            selected.append(item); seen.add(key); added += 1
            if added == count:
                break

    if len(selected) < required:
        remaining = [item for item in proposed if signature(item["knowledge"]) not in seen]
        for item in round_robin_strata(
            remaining, required - len(selected),
            lambda value: (value["design_role"], value["hardware_stratum"]),
        ):
            key = signature(item["knowledge"])
            if key not in seen:
                selected.append(item); seen.add(key)
            if len(selected) == required:
                break
    if len(selected) != required:
        raise old.SearchError(
            f"fixed design has {len(selected)} legal candidates; required {required}"
        )
    remaining = [item for item in proposed if signature(item["knowledge"]) not in seen]
    reserves = round_robin_strata(
        remaining, RESERVES_PER_SHAPE,
        lambda value: (value["design_role"], value["hardware_stratum"]),
    )
    if len(reserves) != RESERVES_PER_SHAPE:
        raise old.SearchError("fixed design lacks legal numeric-failure reserves")
    return selected, reserves


def score_frozen_design(workload: old.Workload, items: list[dict], hardware) -> int:
    operator = matmul(
        workload.m, workload.n, workload.k, workload.dtype,
        trans_a=workload.trans_a, trans_b=workload.trans_b,
    )
    scoring_ns = 0
    for item in items:
        plan = plan_from_cann(
            workload.m, workload.n, workload.k, item["knowledge"],
            dtype=workload.dtype, trans_a=workload.trans_a,
            trans_b=workload.trans_b,
        )
        started = time.perf_counter_ns()
        result = simulate(operator, plan, hardware)
        scoring_ns += time.perf_counter_ns() - started
        if not result.valid:
            raise old.SearchError("central simulator rejected a frozen legal schedule")
        item["simulation"] = result
    ranked = sorted(items, key=lambda item: (
        item["simulation"].total_cycles, signature(item["knowledge"])
    ))
    for rank, item in enumerate(ranked, 1):
        item["model_rank"] = rank
    return scoring_ns


def model_breakdown(result) -> str:
    return json.dumps({
        "critical_core": result.critical_core_cycles,
        "hbm": result.hbm_cycles, "l2": result.l2_cycles,
        "shared": result.shared_resource_cycles,
        "active_cores": result.active_cores,
        "gm_read_bytes": result.gm_read_bytes,
        "gm_write_bytes": result.gm_write_bytes,
        "l2_bytes": result.l2_bytes,
        "workspace_bytes": result.workspace_bytes,
        "reduction_cycles": result.reduction_cycles,
        "resources": {resource.value: cycles for resource, cycles in result.resource_cycles},
    }, separators=(",", ":"), sort_keys=True)


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
    args = parser.parse_args()

    platform = old.Hardware(
        args.aic_cores, args.l0a_bytes, args.l0b_bytes, args.l0c_bytes,
        args.l1_bytes, args.l2_bytes, args.l2_bytes_per_cycle_per_core,
        args.hbm_bytes_per_cycle_per_core,
    )
    hardware = generic_hardware(platform)
    raw_fields, _ = read_rows(args.raw_candidates)
    fields = ["rank", *(field for field in raw_fields if field != "rank" and field not in OBSOLETE_EXECUTION_COLUMNS)]
    for field in (*MANDATORY_COLUMNS, *old.EXTRA_COLUMNS, *CUSTOM_COLUMNS):
        if field not in fields and field not in OBSOLETE_EXECUTION_COLUMNS:
            fields.append(field)
    catalog_fields, catalog = read_rows(args.catalog)
    selected_rows: list[dict[str, str]] = []
    all_rows: list[dict[str, str]] = []
    selected_workloads: list[dict[str, str]] = []
    family_counts: Counter[str] = Counter()

    for workload_index, metadata in enumerate(catalog, 1):
        shape_started = time.perf_counter_ns()
        workload = old.Workload(
            metadata["workload_id"], int(metadata["m"]), int(metadata["n"]),
            int(metadata["k"]), metadata["dtype"], truthy(metadata["trans_a"]),
            truthy(metadata["trans_b"]), args.aic_cores,
        )
        required = int(metadata["required_successful_tilings"])
        generation_started = time.perf_counter_ns()
        proposed, anchors, legality_ns = proposed_candidates(
            workload, platform, hardware
        )
        generation_ns = time.perf_counter_ns() - generation_started
        formal, reserves = select_fixed_design(proposed, anchors, required)
        frozen = [*formal, *reserves]
        scoring_ns = score_frozen_design(workload, frozen, hardware)
        chosen = [
            *((item, False) for item in formal),
            *((item, True) for item in reserves),
        ]
        shape_rows: list[dict[str, str]] = []
        for output_rank, (item, reserve) in enumerate(chosen, 1):
            knowledge = item["knowledge"]
            simulation = item["simulation"]
            family = execution_mode_name(knowledge)
            schedule_sha, suffix = execution_identity(workload, knowledge)
            row = old.row_from_state(
                fields, None, workload, knowledge,
                "hardware_stratified_frontier_v1", family.upper(),
                simulation.total_cycles,
                simulation.gm_read_bytes + simulation.gm_write_bytes,
                simulation.l2_bytes, 10_000_000_000_000_000_000 + suffix,
                guidance=item["controlled_factor"],
                bottleneck=simulation.bottleneck,
                rationale="fixed structural hardware design; model scored only after freeze",
                resume_policy="allow_new",
            )
            row.update({
                "rank": str(output_rank), "execution_mode": family,
                "hardware_aic_cores": str(args.aic_cores),
                "new_model_cycles": f"{simulation.total_cycles:.12g}",
                "new_model_rank": str(item["model_rank"]),
                "global_model_rank": str(item["model_rank"]),
                "new_model_bottleneck": simulation.bottleneck,
                "new_model_breakdown": model_breakdown(simulation),
                "model_schedule_sha256": schedule_sha,
                "model_kernel_suffix": str(suffix),
                "model_kernel_variant": family.upper(),
                "model_kernel_family": family.upper(),
                "model_input_source": "parameters_only_after_preregistered_candidate_freeze",
                "controlled_factor": item["controlled_factor"],
                "factor_signature": item["factor_signature"],
                "pair_id": item["pair_id"],
                "design_role": item["design_role"],
                "hardware_stratum": item["hardware_stratum"],
                "selection_basis": "hardware_capacity_and_structural_strata_no_cost_score",
                "candidate_set_frozen_before_model_scoring": "1",
                "is_reserve": str(int(reserve)),
                "coverage_intent": metadata["coverage_intent"],
                "calibration_partition": metadata["calibration_partition"],
                "required_successful_tilings": str(required),
                "generated_candidate_count": str(len(proposed)),
                "legal_candidate_count": str(len(proposed)),
                "candidate_generation_ms": f"{generation_ns / 1e6:.9g}",
                "static_legality_ms": f"{legality_ns / 1e6:.9g}",
                "simulator_scoring_ms": f"{scoring_ns / 1e6:.9g}",
                "tiling_solver_select_ms": f"{scoring_ns / 1e6:.9g}",
                "tiling_solver_extra_ms": f"{generation_ns / 1e6:.9g}",
                "tiling_solver_total_ms": f"{(time.perf_counter_ns() - shape_started) / 1e6:.9g}",
            })
            shape_rows.append(row)
            if not reserve:
                family_counts[family] += 1
        selected_rows.extend(shape_rows)
        all_rows.extend(shape_rows)
        selected_workloads.append(metadata)
        print(
            f"VICTOR_FRONTIER_CANDIDATES [{workload_index}/{len(catalog)}] "
            f"{workload.workload_id} formal={len(formal)} reserves={len(reserves)} "
            f"legal_pool={len(proposed)} families="
            f"{','.join(sorted({execution_mode_name(item['knowledge']) for item in formal}))} "
            f"host_ms={(time.perf_counter_ns() - shape_started) / 1e6:.3f}",
            flush=True,
        )

    formal_total = sum(int(row["required_successful_tilings"]) for row in catalog)
    if len(catalog) != 3 or formal_total != 2160:
        raise old.SearchError("Victor frontier workload contract changed")
    args.workloads.parent.mkdir(parents=True, exist_ok=True)
    with args.workloads.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=catalog_fields)
        writer.writeheader(); writer.writerows(selected_workloads)
    for destination, rows in ((args.output, selected_rows), (args.all_output, all_rows)):
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows)
    print(
        "MATMUL_VICTOR_FRONTIER_CANDIDATES "
        f"shapes=3 scheduled={len(selected_rows)} formal=2160 reserves=96 "
        f"baselines=3 records=2163 by_family={dict(family_counts)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (old.SearchError, OSError, ValueError, KeyError) as exception:
        print(f"fatal: {exception}", flush=True)
        raise SystemExit(1)
