#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import refine_matmul_v3_candidates as contract


CONTROL_COLUMNS = (
    "experiment_block",
    "pair_id",
    "changed_factor",
    "pair_variant",
    "controlled_sequence",
)


def selected_axis_values(total: int, seed_value: int) -> list[int]:
    values = {
        1,
        total,
        seed_value,
        max(1, seed_value - 1),
        min(total, seed_value + 1),
    }
    for divisor in (2, 3, 4, 5, 8, 10, 16, 20):
        values.add(max(1, min(total, contract.ceil_div(total, divisor))))
        values.add(max(1, min(total, divisor)))
    return sorted(value for value in values if 1 <= value <= total)


def l2_candidates(
    workload: contract.Workload,
    seed: contract.Seed,
    hardware: contract.Hardware,
) -> list[tuple[dict[str, int], str, str, str]]:
    anchor = seed.bank.knowledge
    if contract.template_name(anchor) != "BASE":
        return []
    m_total = contract.ceil_div(workload.m, anchor["singleCoreM"])
    n_total = contract.ceil_div(workload.n, anchor["singleCoreN"])
    m_values = selected_axis_values(m_total, anchor["l2MTileBlock"])
    n_values = selected_axis_values(n_total, anchor["l2NTileBlock"])
    result: list[tuple[dict[str, int], str, str, str]] = []
    all_pairs = [(m_value, n_value) for m_value in m_values for n_value in n_values]
    priority = [
        (anchor["l2MTileBlock"], anchor["l2NTileBlock"]),
        (1, 1),
        (m_total, n_total),
        (1, n_total),
        (m_total, 1),
    ]
    pair_order: list[tuple[int, int]] = []
    for pair in [
        *priority,
        *sorted(
            all_pairs,
            key=lambda item: (
                (item[0] * 131 + item[1] * 197) % 1009,
                item,
            ),
        ),
    ]:
        if pair in all_pairs and pair not in pair_order:
            pair_order.append(pair)
    for m_block, n_block in pair_order:
        pair = f"{workload.workload_id}:l2:m{m_block}:n{n_block}"
        for order in (0, 1):
            candidate = dict(anchor)
            candidate.update(
                l2MTileCnt=contract.ceil_div(m_total, m_block),
                l2NTileCnt=contract.ceil_div(n_total, n_block),
                l2MTileBlock=m_block,
                l2NTileBlock=n_block,
                l2IterateOrder=order,
            )
            if contract.hard_legal(workload, candidate, hardware):
                result.append(
                    (candidate, pair, "l2_partition_and_order", f"order_{order}")
                )
    return result


def base_geometry_anchors(
    workload: contract.Workload,
    seed: contract.Seed,
    hardware: contract.Hardware,
) -> list[dict[str, int]]:
    seed_k = seed.bank.knowledge
    anchors: list[dict[str, int]] = []
    if contract.template_name(seed_k) == "BASE":
        anchors.append(dict(seed_k))
    m_values = {
        seed_k["baseM"],
        16,
        32,
        64,
        80,
        96,
        112,
        128,
        160,
        192,
        208,
        224,
        240,
        256,
    }
    n_values = {
        seed_k["baseN"],
        16,
        32,
        64,
        80,
        96,
        112,
        128,
        160,
        192,
        208,
        224,
        240,
        256,
    }
    k0 = contract.base_k_alignment(workload)
    k_values = {
        seed_k["baseK"],
        k0,
        2 * k0,
        4 * k0,
        8 * k0,
    }
    for base_m in sorted(m_values):
        for base_n in sorted(n_values):
            for base_k in sorted(k_values):
                candidate = contract.configure_base_candidate(
                    workload, seed, hardware, base_m, base_n, base_k
                )
                if candidate is not None:
                    anchors.append(candidate)
    return contract.deduplicate_knowledge(anchors)


def concurrency_candidates(
    workload: contract.Workload,
    seed: contract.Seed,
    hardware: contract.Hardware,
) -> list[tuple[dict[str, int], str, str, str]]:
    result: list[tuple[dict[str, int], str, str, str]] = []
    core_limit = min(workload.max_cores, hardware.aic_cores)
    for anchor_index, anchor in enumerate(
        base_geometry_anchors(workload, seed, hardware)[:8]
    ):
        tasks = (
            contract.ceil_div(workload.m, anchor["singleCoreM"])
            * contract.ceil_div(workload.n, anchor["singleCoreN"])
        )
        pair = f"{workload.workload_id}:core:g{anchor_index:02d}"
        for cores in range(1, min(core_limit, tasks) + 1):
            candidate = dict(anchor)
            candidate["usedCoreNum"] = cores
            if contract.hard_legal(workload, candidate, hardware):
                result.append(
                    (candidate, pair, "core_concurrency", f"cores_{cores}")
                )
    return result


def buffer_candidates(
    workload: contract.Workload,
    seed: contract.Seed,
    hardware: contract.Hardware,
) -> list[tuple[dict[str, int], str, str, str]]:
    result: list[tuple[dict[str, int], str, str, str]] = []
    for anchor_index, anchor in enumerate(
        base_geometry_anchors(workload, seed, hardware)[:12]
    ):
        l0_pair = f"{workload.workload_id}:l0:g{anchor_index:02d}"
        for db_a in (1, 2):
            for db_b in (1, 2):
                for db_c in (1, 2):
                    candidate = dict(anchor)
                    candidate.update(dbL0A=db_a, dbL0B=db_b, dbL0C=db_c)
                    if contract.hard_legal(workload, candidate, hardware):
                        result.append(
                            (
                                candidate,
                                l0_pair,
                                "l0_double_buffer",
                                f"db_{db_a}{db_b}{db_c}",
                            )
                        )

        # At fixed geometry, steps and L0 buffering, depth/step is exactly the
        # number of resident L1 packets.  The contract permits one or two.
        l1_anchor = dict(anchor)
        l1_anchor.update(dbL0A=1, dbL0B=1, dbL0C=1)
        l1_pair = f"{workload.workload_id}:l1:g{anchor_index:02d}"
        for a_packets in (1, 2):
            for b_packets in (1, 2):
                candidate = dict(l1_anchor)
                candidate["depthA1"] = (
                    candidate["stepM"] * candidate["stepKa"] * a_packets
                )
                candidate["depthB1"] = (
                    candidate["stepN"] * candidate["stepKb"] * b_packets
                )
                if contract.hard_legal(workload, candidate, hardware):
                    result.append(
                        (
                            candidate,
                            l1_pair,
                            "l1_packet_residency",
                            f"packets_{a_packets}{b_packets}",
                        )
                    )
    return result


def deterministic_splitk_candidates(
    workload: contract.Workload,
    hardware: contract.Hardware,
) -> list[tuple[dict[str, int], str, str, str]]:
    in_bytes = contract.INPUT_BYTES[workload.dtype]
    base_k = 256 // in_bytes
    single_k = 3 * base_k
    k_chunks = contract.ceil_div(workload.k, single_k)
    core_limit = min(workload.max_cores, hardware.aic_cores, k_chunks)
    layouts = (
        ("mk", 3, 1, 9, 6, 1, 384, max(128, workload.n), 0),
        ("nk", 1, 3, 6, 9, 0, max(128, workload.m), 384, 1),
    )
    result: list[tuple[dict[str, int], str, str, str]] = []
    for name, step_m, step_n, depth_a, depth_b, order, single_m, single_n, l2_order in layouts:
        m_chunks = contract.ceil_div(workload.m, single_m)
        n_chunks = contract.ceil_div(workload.n, single_n)
        pair = f"{workload.workload_id}:splitk:{name}"
        for cores in range(1, core_limit + 1):
            candidate = {
                "usedCoreNum": cores,
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
                "iterateOrder": order,
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
            if contract.hard_legal(workload, candidate, hardware):
                result.append(
                    (candidate, pair, "splitk_concurrency", f"cores_{cores}")
                )
    return result


def splitk_candidates(
    workload: contract.Workload,
    seed: contract.Seed,
    hardware: contract.Hardware,
) -> list[tuple[dict[str, int], str, str, str]]:
    result = deterministic_splitk_candidates(workload, hardware)
    for index, candidate in enumerate(
        contract.single_core_split_k_candidate_space(workload, seed, hardware)
    ):
        result.append(
            (
                candidate,
                f"{workload.workload_id}:single_splitk:{index:03d}",
                "splitk_route_geometry",
                "single_core_splitk",
            )
        )
    return result


def proposals_for(
    block: str,
    workload: contract.Workload,
    seed: contract.Seed,
    hardware: contract.Hardware,
) -> list[tuple[dict[str, int], str, str, str]]:
    if block == "l2":
        proposals = l2_candidates(workload, seed, hardware)
    elif block == "concurrency":
        proposals = concurrency_candidates(workload, seed, hardware)
    elif block == "buffer":
        proposals = buffer_candidates(workload, seed, hardware)
    elif block == "splitk":
        proposals = splitk_candidates(workload, seed, hardware)
    else:
        raise contract.SearchError(f"unknown experiment block: {block}")
    unique: list[tuple[dict[str, int], str, str, str]] = []
    seen: set[tuple[int, ...]] = set()
    seed_signature = contract.knowledge_signature(seed.bank.knowledge)
    for proposal in proposals:
        signature = contract.knowledge_signature(proposal[0])
        if signature == seed_signature or signature in seen:
            continue
        seen.add(signature)
        unique.append(proposal)
    return unique


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-candidates", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=36)
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
    if args.top_k < 19:
        raise contract.SearchError("top-k must leave at least 19 searched candidates")

    hardware = contract.Hardware(
        aic_cores=args.aic_cores,
        l0a_bytes=args.l0a_bytes,
        l0b_bytes=args.l0b_bytes,
        l0c_bytes=args.l0c_bytes,
        l1_bytes=args.l1_bytes,
        l2_bytes=args.l2_bytes,
        l2_bytes_per_cycle_per_core=args.l2_bytes_per_cycle_per_core,
        hbm_bytes_per_cycle_per_core=args.hbm_bytes_per_cycle_per_core,
    )
    from tbe.common.platform import set_current_compile_soc_info
    from tbe.common.utils import op_tiling

    set_current_compile_soc_info(args.soc)
    op_tiling._RT_BANK_CACHE = {}

    raw_fields, _ = contract.read_csv(args.raw_candidates)
    output_fields = ["rank", *(field for field in raw_fields if field != "rank")]
    for column in (*contract.EXTRA_COLUMNS, *CONTROL_COLUMNS):
        if column not in output_fields:
            output_fields.append(column)
    workloads = contract.load_workloads(args.workloads)
    _, workload_rows = contract.read_csv(args.workloads)
    metadata = {
        (row.get("workload_id") or row.get("id", "")): row
        for row in workload_rows
    }

    selected: list[dict[str, str]] = []
    all_rows: list[dict[str, str]] = []
    for workload_index, workload in enumerate(workloads, 1):
        started = time.perf_counter_ns()
        seed = contract.parse_seed(workload)
        block = metadata[workload.workload_id]["experiment_block"]
        control = contract.bank_seed_control(output_fields, workload, seed, hardware)
        seed_k = seed.bank.knowledge
        if block == "l2":
            control_pair = (
                f"{workload.workload_id}:l2:"
                f"m{seed_k['l2MTileBlock']}:n{seed_k['l2NTileBlock']}"
            )
            control_factor = "l2_partition_and_order"
            control_variant = f"order_{seed_k['l2IterateOrder']}"
        elif block == "concurrency":
            control_pair = f"{workload.workload_id}:core:g00"
            control_factor = "core_concurrency"
            control_variant = f"cores_{seed_k['usedCoreNum']}"
        elif block == "buffer":
            control_pair = f"{workload.workload_id}:l0:g00"
            control_factor = "l0_double_buffer"
            control_variant = (
                f"db_{seed_k['dbL0A']}{seed_k['dbL0B']}{seed_k['dbL0C']}"
            )
        else:
            control_pair = f"{workload.workload_id}:official_seed"
            control_factor = "official_seed_control"
            control_variant = "control"
        control.row.update(
            experiment_block=block,
            pair_id=control_pair,
            changed_factor=control_factor,
            pair_variant=control_variant,
            controlled_sequence="0",
        )
        selected.append(dict(control.row))
        all_rows.append(dict(control.row))

        bank_estimate = contract.analytical_score(
            workload, seed.bank.knowledge, hardware, seed.bank
        )
        accepted: list[contract.State] = []
        failures = 0
        for sequence, (knowledge, pair_id, factor, variant) in enumerate(
            proposals_for(block, workload, seed, hardware), 1
        ):
            estimate = contract.analytical_score(workload, knowledge, hardware)
            ratio = estimate.cycles / max(1.0, bank_estimate.cycles)
            row = contract.row_from_state(
                output_fields,
                None,
                workload,
                knowledge,
                "cann81_controlled_hardware_frontier_v1",
                contract.template_name(knowledge),
                ratio,
                estimate.hbm_bytes,
                estimate.l2_bytes,
                seed.key,
                guidance=factor,
                estimate=estimate,
                bottleneck=block,
                rationale="fixed hardware-factor comparison",
                resume_policy="allow_new",
            )
            row.update(
                experiment_block=block,
                pair_id=pair_id,
                changed_factor=factor,
                pair_variant=variant,
                controlled_sequence=str(sequence),
            )
            state = contract.State(
                row=row,
                knowledge=knowledge,
                model_score=estimate.cycles,
                normalized_score=ratio,
                hbm_bytes=estimate.hbm_bytes,
                l2_bytes=estimate.l2_bytes,
                template=contract.template_name(knowledge),
                guidance=factor,
                estimate=estimate,
            )
            try:
                callback = contract.validate_callback(workload, state)
            except Exception:
                failures += 1
                continue
            contract.update_callback_columns(
                state, callback, seed, bank_estimate, hardware, workload
            )
            accepted.append(state)
            all_rows.append(dict(state.row))
            if len(accepted) == args.top_k:
                break

        if len(accepted) < args.top_k:
            raise contract.SearchError(
                f"{workload.workload_id}: only {len(accepted)} callback-fixed "
                f"controlled candidates, required {args.top_k}; rejected={failures}"
            )
        for rank, state in enumerate(accepted, 1):
            state.row["rank"] = str(rank)
            selected.append(dict(state.row))
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        print(
            f"CONTROLLED_CANDIDATES [{workload_index}/{len(workloads)}] "
            f"{workload.workload_id} block={block} callback_fixed={len(accepted)} "
            f"callback_rejected={failures} host_ms={elapsed_ms:.3f}",
            flush=True,
        )

    for destination, rows in ((args.output, selected), (args.all_output, all_rows)):
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=output_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    print(
        f"MATMUL_CONTROLLED_CANDIDATES workloads={len(workloads)} "
        f"selected_rows={len(selected)} searched_per_workload={args.top_k}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
