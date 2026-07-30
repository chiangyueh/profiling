#!/usr/bin/env python3
from __future__ import annotations

import csv
import struct
import sys
import tempfile
from collections import Counter
from types import SimpleNamespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import refine_matmul_v3_candidates as refine
import rank_npu_results as rank_results
import profile_official_tilings as profile_tilings
import print_npu_summary as print_summary


HARDWARE = refine.Hardware(
    aic_cores=20,
    l0a_bytes=64 * 1024,
    l0b_bytes=64 * 1024,
    l0c_bytes=128 * 1024,
    l1_bytes=512 * 1024,
    l2_bytes=192 * 1024 * 1024,
    l2_bytes_per_cycle_per_core=110.0,
    hbm_bytes_per_cycle_per_core=32.0,
)


def test_reference_pair_coherence_guard() -> None:
    close_official = {"median_ms": "1.0", "stddev_ms": "0.001"}
    close_bank = {"median_ms": "1.02", "stddev_ms": "0.001"}
    drifted_official = {"median_ms": "2.08892", "stddev_ms": "0.00681975"}
    drifted_bank = {"median_ms": "0.754802", "stddev_ms": "0.00681975"}
    assert profile_tilings.baseline_pair_is_coherent(
        close_official, close_bank
    )
    assert not profile_tilings.baseline_pair_is_coherent(
        drifted_official, drifted_bank
    )


def base_knowledge(
    workload: refine.Workload,
    base_m: int,
    base_n: int,
    base_k: int,
) -> dict[str, int]:
    m_blocks = refine.ceil_div(workload.m, base_m)
    n_blocks = refine.ceil_div(workload.n, base_n)
    return {
        "usedCoreNum": HARDWARE.aic_cores,
        "singleCoreM": base_m,
        "singleCoreN": base_n,
        "singleCoreK": workload.k,
        "baseM": base_m,
        "baseN": base_n,
        "baseK": base_k,
        "depthA1": 2,
        "depthB1": 2,
        "stepM": 1,
        "stepN": 1,
        "iterateOrder": 0,
        "stepKa": 1,
        "stepKb": 1,
        "dbL0A": 2,
        "dbL0B": 2,
        "dbL0C": 1,
        "l2MTileCnt": 1,
        "l2NTileCnt": 1,
        "l2MTileBlock": m_blocks,
        "l2NTileBlock": n_blocks,
        "l2IterateOrder": 0,
        "tilingEnable": 0,
    }


def test_layout_specific_fp32_alignment() -> None:
    nn = refine.Workload("nn", 64, 64, 64, "fp32", False, False, 20)
    nt = refine.Workload("nt", 64, 64, 64, "fp32", False, True, 20)
    tn = refine.Workload("tn", 64, 64, 64, "fp32", True, False, 20)
    assert refine.base_k_alignment(nn) == 16
    assert refine.base_k_alignment(nt) == 8
    assert refine.base_k_alignment(tn) == 16


def test_complete_callback_blob_parsing() -> None:
    words = [0] * 68
    words[0] = 20
    words[5:15] = [128, 256, 4096, 128, 256, 64, 16, 8, 1, 1]
    words[17] = 0
    words[26:28] = [8, 4]
    words[30:33] = [2, 2, 1]
    words[50:55] = [1, 1, 32, 16, 0]
    words[62:67] = [1, 32, 64, 16, 128]
    result = {
        "tiling_key": 11000000000100000,
        "block_dim": 20,
        "workspaces": [1024],
        "tiling_data": struct.pack("<68I", *words),
    }
    callback = refine.parse_callback_result(result)
    assert len(callback.blob) == 272
    assert callback.knowledge["baseM"] == 128
    assert callback.knowledge["baseN"] == 256
    assert callback.derived == {
        "l2CacheFlag": 1,
        "baseAN": 32,
        "baseAD": 64,
        "baseBN": 16,
        "baseBD": 128,
    }
    assert len(callback.sha256) == 64


def test_cann81_kernel_key_contract() -> None:
    prefix = 10_000_000_000_000_000_000
    expected = {
        0: (0, "BASE_UNALIGNED", "BASE"),
        1: (0, "BASE_ALIGNED", "BASE"),
        20: (2, "SINGLE_CORE_SPLIT_K_UNALIGNED", "SINGLE_CORE_SPLIT_K"),
        21: (2, "SINGLE_CORE_SPLIT_K_ALIGNED", "SINGLE_CORE_SPLIT_K"),
        30: (3, "DETERMINISTIC_SPLIT_K_UNALIGNED", "DETERMINISTIC_SPLIT_K"),
        31: (3, "DETERMINISTIC_SPLIT_K_ALIGNED", "DETERMINISTIC_SPLIT_K"),
        101: (10, "AL1_FULL_LOAD_ALIGNED", "AL1_FULL_LOAD"),
        200: (20, "BL1_FULL_LOAD_UNALIGNED", "BL1_FULL_LOAD"),
        201: (20, "BL1_FULL_LOAD_ALIGNED", "BL1_FULL_LOAD"),
        10200: (
            1020,
            "BL1_FULL_LOAD_FIXPIPE_UNALIGNED",
            "BL1_FULL_LOAD_FIXPIPE",
        ),
        10201: (
            1020,
            "BL1_FULL_LOAD_FIXPIPE_ALIGNED",
            "BL1_FULL_LOAD_FIXPIPE",
        ),
        20201: (
            2020,
            "BL1_FULL_LOAD_VEC_NZ2ND",
            "BL1_FULL_LOAD_VEC_NZ2ND",
        ),
    }
    for suffix, (tiling_enable, variant, family) in expected.items():
        key = prefix + suffix
        assert refine.decode_tiling_enable(key) == tiling_enable
        assert refine.kernel_variant(key) == variant
        assert refine.kernel_family(key) == family


def test_twenty_core_slot_balance_is_ranked() -> None:
    workload = refine.Workload(
        "skinny_n", 4096, 17, 16384, "fp16", False, False, 20
    )
    exact_twenty = refine.analytical_score(
        workload, base_knowledge(workload, 208, 32, 64), HARDWARE
    )
    thirteen_blocks = refine.analytical_score(
        workload, base_knowledge(workload, 320, 32, 32), HARDWARE
    )
    assert exact_twenty.balance < thirteen_blocks.balance
    assert exact_twenty.cycles < thirteen_blocks.cycles


def test_focused_skinny_n_ablation_space() -> None:
    workload = refine.Workload(
        "skinny_n_large_k", 4096, 17, 16384,
        "fp16", False, False, 20,
    )
    seed_knowledge = base_knowledge(workload, 128, 32, 64)
    seed_knowledge.update(
        depthA1=16,
        depthB1=64,
        stepKa=8,
        stepKb=32,
        l2MTileCnt=8,
        l2NTileCnt=1,
        l2MTileBlock=4,
        l2NTileBlock=1,
    )
    seed = SimpleNamespace(
        bank=SimpleNamespace(knowledge=seed_knowledge)
    )
    candidates = refine.skinny_n_large_k_candidate_space(
        workload, seed, HARDWARE
    )
    assert len(candidates) == 4
    assert all(refine.hard_legal(workload, row, HARDWARE) for row in candidates)
    assert {
        (row["baseM"], row["baseN"], row["baseK"])
        for row in candidates
    } == {(208, 32, 64)}
    assert (
        candidates[0]["depthA1"],
        candidates[0]["depthB1"],
        candidates[0]["stepKa"],
        candidates[0]["stepKb"],
    ) == (8, 64, 4, 32)
    assert (
        candidates[-1]["depthA1"],
        candidates[-1]["depthB1"],
        candidates[-1]["dbL0C"],
        candidates[-1]["l2IterateOrder"],
    ) == (16, 8, 2, 1)


def test_skinny_n_l1_rebalance_respects_base_m_256_capacity() -> None:
    workload = refine.Workload(
        "skinny_n_k16384_holdout_m5120_n29",
        5120, 29, 16384, "fp16", False, False, 20,
    )
    seed_knowledge = base_knowledge(workload, 128, 32, 64)
    seed = SimpleNamespace(
        bank=SimpleNamespace(knowledge=seed_knowledge)
    )
    candidates = refine.skinny_n_large_k_candidate_space(
        workload, seed, HARDWARE
    )
    assert len(candidates) == 4
    assert all(refine.hard_legal(workload, row, HARDWARE) for row in candidates)
    assert (
        candidates[1]["baseM"],
        candidates[1]["depthA1"],
        candidates[1]["depthB1"],
        candidates[1]["stepKa"],
        candidates[1]["stepKb"],
    ) == (256, 8, 8, 4, 4)
    learned = refine.learned_skinny_n_k16384_schedule(
        workload, seed, HARDWARE
    )
    assert len(learned) == 1
    assert learned[0]["dbL0C"] == 2
    assert learned[0]["l2IterateOrder"] == 1


def test_official_local_search_changes_one_order_field() -> None:
    workload = refine.Workload(
        "llm_full_4096_ffn_up", 4096, 11008, 4096,
        "fp16", False, False, 20,
    )
    seed_knowledge = base_knowledge(workload, 128, 256, 64)
    seed_knowledge.update(
        depthA1=16,
        depthB1=8,
        stepKa=8,
        stepKb=4,
        l2MTileCnt=3,
        l2NTileCnt=2,
        l2MTileBlock=11,
        l2NTileBlock=22,
    )
    seed = SimpleNamespace(
        bank=SimpleNamespace(knowledge=seed_knowledge)
    )
    candidates = refine.official_seed_order_candidate_space(
        workload, seed, HARDWARE
    )
    assert len(candidates) == 2
    changed_fields = []
    for candidate in candidates:
        changed_fields.append(
            {
                field
                for field in refine.KNOWLEDGE_FIELDS
                if candidate[field] != seed_knowledge[field]
            }
        )
        assert refine.hard_legal(workload, candidate, HARDWARE)
    assert changed_fields == [{"iterateOrder"}, {"l2IterateOrder"}]
    states = []
    for index, candidate in enumerate(candidates, 1):
        guidance = refine.official_local_guidance(candidate, seed)
        states.append(
            refine.State(
                row={"search_guidance": guidance},
                knowledge=candidate,
                model_score=float(index),
                normalized_score=float(index),
                hbm_bytes=0.0,
                l2_bytes=0.0,
                template="BASE",
                guidance=guidance,
            )
        )
    retained = refine.constraint_aware_beam(states, 16)
    assert {
        state.row["search_guidance"] for state in retained
    } == {
        "official_seed_iterate_order_ablation",
        "official_seed_l2_order_ablation",
    }


def test_general_search_builds_independent_multistart_sources() -> None:
    workload = refine.Workload(
        "unseen_without_family_name",
        1536,
        2560,
        3072,
        "fp16",
        False,
        False,
        20,
    )
    seed_knowledge = base_knowledge(workload, 128, 256, 64)
    seed_knowledge.update(
        depthA1=16,
        depthB1=8,
        stepKa=8,
        stepKb=4,
        l2MTileCnt=1,
        l2NTileCnt=1,
        l2MTileBlock=12,
        l2NTileBlock=10,
    )
    callback = SimpleNamespace(
        knowledge=seed_knowledge,
        derived={},
    )
    seed = SimpleNamespace(
        knowledge=seed_knowledge,
        bank=callback,
    )
    estimate = refine.analytical_score(
        workload, seed_knowledge, HARDWARE, callback
    )
    bottleneck = refine.diagnose_bottleneck(
        workload, seed_knowledge, estimate, HARDWARE
    )
    transfer = refine.TransferGeometry(
        workload_id="measured_elsewhere",
        m=4096,
        n=17,
        k=16384,
        dtype="fp16",
        trans_a=False,
        trans_b=False,
        template="BASE",
        base_m=208,
        base_n=32,
        base_k=64,
        depth_a1=16,
        depth_b1=8,
        iterate_order=0,
        db_l0a=2,
        db_l0b=2,
        db_l0c=1,
        l2_iterate_order=0,
        speedup=1.50,
    )
    proposals, stop = refine.general_search_candidate_proposals(
        workload,
        seed,
        HARDWARE,
        [],
        estimate,
        bottleneck,
        [transfer],
    )
    assert stop == ""
    assert 4 <= len(proposals) <= 60
    assert all(
        refine.hard_legal(workload, proposal.knowledge, HARDWARE)
        for proposal in proposals
    )
    sources = {proposal.source for proposal in proposals}
    assert {"local", "global", "diverse"} <= sources
    assert any(
        refine.structural_distance(
            proposal.knowledge, seed_knowledge
        ) > 1.0
        for proposal in proposals
        if proposal.source == "diverse"
    )


def test_general_transfer_reconstructs_target_partition_geometry() -> None:
    workload = refine.Workload(
        "unseen_transfer_target",
        4608,
        31,
        16384,
        "fp16",
        False,
        False,
        20,
    )
    seed_knowledge = base_knowledge(workload, 128, 32, 64)
    seed_knowledge.update(
        depthA1=16,
        depthB1=64,
        stepKa=8,
        stepKb=32,
        l2MTileCnt=8,
        l2NTileCnt=1,
        l2MTileBlock=5,
        l2NTileBlock=1,
    )
    seed = SimpleNamespace(
        knowledge=seed_knowledge,
        bank=SimpleNamespace(
            knowledge=seed_knowledge,
            derived={},
        ),
    )
    source = refine.TransferGeometry(
        workload_id="measured_source",
        m=4096,
        n=17,
        k=16384,
        dtype="fp16",
        trans_a=False,
        trans_b=False,
        template="BASE",
        base_m=208,
        base_n=32,
        base_k=64,
        depth_a1=16,
        depth_b1=8,
        iterate_order=0,
        db_l0a=2,
        db_l0b=2,
        db_l0c=1,
        l2_iterate_order=0,
        speedup=1.50,
    )
    transferred = refine.general_transfer_geometry_space(
        workload, seed, HARDWARE, [source]
    )
    assert len(transferred) == 1
    candidate, geometry = transferred[0]
    assert geometry.workload_id == "measured_source"
    assert candidate["baseM"] == 240
    assert candidate["baseN"] == 32
    assert candidate["depthA1"] == 16
    assert candidate["depthB1"] == 8
    assert candidate["dbL0C"] == 1
    assert refine.hard_legal(workload, candidate, HARDWARE)


def test_general_frontier_preserves_each_start_source() -> None:
    workload = refine.Workload(
        "source_quota", 512, 512, 1024,
        "fp16", False, False, 20,
    )
    seed_knowledge = base_knowledge(workload, 128, 128, 64)
    seed = SimpleNamespace(
        bank=SimpleNamespace(knowledge=seed_knowledge)
    )
    states = []
    for index, source in enumerate(
        ("local", "local", "local", "global", "transfer", "diverse")
    ):
        knowledge = dict(seed_knowledge)
        knowledge["baseM"] = 16 * (index + 1)
        states.append(
            refine.State(
                row={"search_candidate_source": source},
                knowledge=knowledge,
                model_score=float(index + 1),
                normalized_score=float(index + 1),
                hbm_bytes=0.0,
                l2_bytes=0.0,
                template="BASE",
            )
        )
    retained = refine.general_source_frontier(states, 4, seed)
    assert {
        state.row["search_candidate_source"] for state in retained
    } == {"local", "global", "transfer", "diverse"}


def test_general_active_frontier_excludes_measured_fingerprints() -> None:
    workload = refine.Workload(
        "active_source_quota", 512, 512, 1024,
        "fp16", False, False, 20,
    )
    seed_knowledge = base_knowledge(workload, 128, 128, 64)
    seed = SimpleNamespace(
        bank=SimpleNamespace(knowledge=seed_knowledge)
    )
    states = []
    for source_index, source in enumerate(
        ("local", "global", "transfer", "diverse")
    ):
        for measured in (True, False):
            knowledge = dict(seed_knowledge)
            knowledge["baseM"] = 16 * (
                2 * source_index + int(measured) + 1
            )
            states.append(
                refine.State(
                    row={
                        "search_candidate_source": source,
                        "search_history_match": (
                            f"measured:{source}" if measured else ""
                        ),
                    },
                    knowledge=knowledge,
                    model_score=1.0,
                    normalized_score=1.0,
                    hbm_bytes=0.0,
                    l2_bytes=0.0,
                    template="BASE",
                )
            )
    retained = refine.general_source_frontier(
        states, 4, seed, active_learning=True
    )
    assert len(retained) == 4
    assert all(
        not state.row["search_history_match"] for state in retained
    )
    assert {
        state.row["search_candidate_source"] for state in retained
    } == {"local", "global", "transfer", "diverse"}


def test_profile_resume_drives_search_history_and_transfer() -> None:
    workload = refine.Workload(
        "profile_feedback", 1537, 2305, 4099,
        "fp16", False, False, 20,
    )
    knowledge = base_knowledge(workload, 224, 144, 64)
    signature = ":".join(
        str(knowledge[field]) for field in refine.KNOWLEDGE_FIELDS
    )
    common = {
        "resume_soc": "Ascend910B3",
        "resume_aic": "20",
        "resume_run": "paired_run",
        "success": "1",
        "workload_id": workload.workload_id,
        "m": str(workload.m),
        "n": str(workload.n),
        "k": str(workload.k),
        "dtype": workload.dtype,
        "trans_a": "0",
        "trans_b": "0",
    }
    rows = [
        {
            **common,
            "candidate_role": "official_operator_baseline",
            "median_ms": "1.0",
            "stddev_ms": "0.001",
        },
        {
            **common,
            "candidate_role": "bank_seed_control",
            "preflight_passed": "1",
            "median_ms": "1.02",
            "stddev_ms": "0.001",
        },
        {
            **common,
            "candidate_role": "searched",
            "preflight_passed": "1",
            "median_ms": "0.80",
            "stddev_ms": "0.001",
            "rank": "5",
            "tiling_signature": signature,
            "resume_record_id": "profile_feedback:rank5",
        },
    ]
    fields = sorted({field for row in rows for field in row})
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "resume.csv"
        with path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        history = refine.load_profile_measurement_history(
            path, "Ascend910B3", 20
        )
        evidence = history[
            refine.state_history_key(workload, knowledge)
        ][0]
        assert abs(evidence.ratio_vs_official - 0.80) < 1.0e-12
        assert evidence.ratio_vs_bank is not None
        assert abs(evidence.ratio_vs_bank - 0.80 / 1.02) < 1.0e-12
        assert evidence.exact_profile
        transfers = refine.load_profile_transfer_geometries(
            path, "Ascend910B3", 20
        )
        assert len(transfers) == 1
        assert transfers[0].base_m == 224
        assert transfers[0].base_n == 144


def test_unstable_profile_is_excluded_without_calibration() -> None:
    workload = refine.Workload(
        "unstable_profile", 1024, 1024, 1024,
        "fp16", False, False, 20,
    )
    knowledge = base_knowledge(workload, 128, 128, 64)
    signature = ":".join(
        str(knowledge[field]) for field in refine.KNOWLEDGE_FIELDS
    )
    common = {
        "resume_soc": "Ascend910B3",
        "resume_aic": "20",
        "resume_run": "unstable_run",
        "success": "1",
        "workload_id": workload.workload_id,
        "m": str(workload.m),
        "n": str(workload.n),
        "k": str(workload.k),
        "dtype": workload.dtype,
        "trans_a": "0",
        "trans_b": "0",
    }
    rows = [
        {
            **common,
            "candidate_role": "official_operator_baseline",
            "median_ms": "1.0",
            "stddev_ms": "0.1",
        },
        {
            **common,
            "candidate_role": "bank_seed_control",
            "preflight_passed": "1",
            "median_ms": "1.0",
            "stddev_ms": "0.001",
        },
        {
            **common,
            "candidate_role": "searched",
            "preflight_passed": "1",
            "median_ms": "0.5",
            "stddev_ms": "0.001",
            "rank": "1",
            "tiling_signature": signature,
        },
    ]
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "resume.csv"
        with path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(
                destination,
                fieldnames=sorted({field for row in rows for field in row}),
            )
            writer.writeheader()
            writer.writerows(rows)
        history = refine.load_profile_measurement_history(
            path, "Ascend910B3", 20
        )
        evidence = history[
            refine.state_history_key(workload, knowledge)
        ][0]
        assert evidence.exact_profile
        assert evidence.ratio_vs_official is None
        assert evidence.ratio_vs_bank is None
        assert not refine.load_profile_transfer_geometries(
            path, "Ascend910B3", 20
        )


def test_campaign_manifest_excludes_exact_fingerprint() -> None:
    workload = refine.Workload(
        "campaign_workload", 512, 768, 1024,
        "fp16", False, False, 20,
    )
    knowledge = base_knowledge(workload, 128, 128, 64)
    signature = ":".join(
        str(knowledge[field]) for field in refine.KNOWLEDGE_FIELDS
    )
    row = {
        "campaign": "round1",
        "soc": "Ascend910B3",
        "aic": "20",
        "workload_id": workload.workload_id,
        "m": str(workload.m),
        "n": str(workload.n),
        "k": str(workload.k),
        "dtype": workload.dtype,
        "trans_a": "0",
        "trans_b": "0",
        "tiling_signature": signature,
    }
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "campaign.csv"
        with path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        history, count = refine.load_campaign_exclusions(
            path, "Ascend910B3", 20
        )
        assert count == 1
        evidence = history[
            refine.state_history_key(workload, knowledge)
        ][0]
        assert evidence.exact_profile
        assert evidence.ratio_vs_official is None

    versioned_history, versioned_count = refine.load_campaign_exclusions(
        ROOT / "config/general_search_v1_round1_fingerprints.csv",
        "Ascend910B3",
        20,
    )
    assert versioned_count == 187
    assert len(versioned_history) == 187
    round2_history, round2_count = refine.load_campaign_exclusions(
        ROOT / "config/general_search_v1_round2_fingerprints.csv",
        "Ascend910B3",
        20,
    )
    assert round2_count == 145
    assert len(round2_history) == 145
    round3_history, round3_count = refine.load_campaign_exclusions(
        ROOT / "config/general_search_v1_round3_partial_fingerprints.csv",
        "Ascend910B3",
        20,
    )
    assert round3_count == 48
    assert len(round3_history) == 48


def test_campaign_observations_calibrate_sources_and_transfer() -> None:
    history, transfers, count = refine.load_campaign_observations(
        ROOT / "config/general_search_v1_round2_observations.csv",
        "Ascend910B3",
        20,
    )
    assert count == 44
    assert len(history) == 44
    assert len(transfers) == 3
    corrections = refine.conservative_source_corrections(history)
    assert 1.09 < corrections["local"] < 1.15
    assert 1.5 < corrections["global"] < 1.6
    assert 1.5 < corrections["diverse"] < 1.6
    assert "transfer" not in corrections

    workload = refine.Workload(
        "campaign_prior", 1024, 1024, 1024,
        "fp16", False, False, 20,
    )
    knowledge = base_knowledge(workload, 192, 160, 64)
    state = refine.State(
        row={
            "search_candidate_source": "global",
            "search_model_confidence": "high",
        },
        knowledge=knowledge,
        model_score=1.0,
        normalized_score=1.0,
        hbm_bytes=0.0,
        l2_bytes=0.0,
        template="BASE",
    )
    refine.calibrate_from_history(
        workload, [state], history, "", corrections
    )
    assert state.normalized_score == corrections["global"]
    assert state.row["search_model_confidence"] == (
        "campaign_source_calibrated"
    )

    partial_history, partial_transfers, partial_count = (
        refine.load_campaign_observations(
            ROOT / "config/general_search_v1_round3_partial_observations.csv",
            "Ascend910B3",
            20,
        )
    )
    assert partial_count == 5
    assert len(partial_history) == 5
    assert len(partial_transfers) == 1
    combined_history = {
        key: list(records) for key, records in history.items()
    }
    for key, records in partial_history.items():
        combined_history.setdefault(key, []).extend(records)
    combined_corrections = refine.conservative_source_corrections(
        combined_history
    )
    assert 1.55 < combined_corrections["global"] < 1.60
    assert 1.55 < combined_corrections["diverse"] < 1.60


def test_unstable_comparison_cannot_claim_improvement() -> None:
    _, _, _, status = profile_tilings.comparison_status(
        {"median_ms": "1.0", "stddev_ms": "0.06"},
        {"median_ms": "0.5", "stddev_ms": "0.001"},
    )
    assert status == "unstable_measurement"


def test_unpaired_exact_profile_is_excluded_without_calibration() -> None:
    workload = refine.Workload(
        "old_profile_exact", 1024, 1024, 1024,
        "fp16", False, False, 20,
    )
    knowledge = base_knowledge(workload, 128, 128, 64)
    state = refine.State(
        row={
            "search_candidate_source": "global",
            "search_model_confidence": "high",
        },
        knowledge=knowledge,
        model_score=1.0,
        normalized_score=1.2,
        hbm_bytes=0.0,
        l2_bytes=0.0,
        template="BASE",
    )
    _, matches = refine.calibrate_from_history(
        workload,
        [state],
        {
            refine.state_history_key(workload, knowledge): [
                refine.MeasurementEvidence(
                    ratio_vs_official=None,
                    ratio_vs_bank=None,
                    record_id="old_profile_exact:rank1",
                    exact_profile=True,
                )
            ]
        },
        "",
    )
    assert matches == 1
    assert state.normalized_score == 1.2
    assert state.row["search_history_match"] == (
        "old_profile_exact:rank1"
    )
    assert state.row["search_model_confidence"] == (
        "measured_history_unpaired"
    )


def test_history_calibration_is_source_specific() -> None:
    workload = refine.Workload(
        "source_calibration", 1024, 1024, 1024,
        "fp16", False, False, 20,
    )
    states = []
    history = {}
    for index in range(3):
        knowledge = base_knowledge(
            workload, 128 + 16 * index, 128, 64
        )
        state = refine.State(
            row={
                "search_candidate_source": "global",
                "search_model_confidence": "high",
            },
            knowledge=knowledge,
            model_score=1.0,
            normalized_score=1.0,
            hbm_bytes=0.0,
            l2_bytes=0.0,
            template="BASE",
        )
        states.append(state)
        if index < 2:
            history[refine.state_history_key(workload, knowledge)] = [
                refine.MeasurementEvidence(
                    ratio_vs_official=1.5,
                    ratio_vs_bank=1.5,
                    record_id=f"global:{index}",
                )
            ]
    refine.calibrate_from_history(
        workload, states, history, ""
    )
    assert states[2].normalized_score == 1.5
    assert states[2].row["search_model_calibration"] == "1.5"
    assert states[2].row["search_model_confidence"] == "source_calibrated"


def test_split_k_order_search_requires_aligned_deterministic_template() -> None:
    aligned = refine.Workload(
        "det_aligned", 128, 128, 32768,
        "fp16", False, False, 20,
    )
    deterministic = base_knowledge(aligned, 128, 128, 128)
    deterministic["tilingEnable"] = 3
    seed = SimpleNamespace(
        bank=SimpleNamespace(knowledge=deterministic)
    )
    assert refine.official_seed_order_fields(
        aligned, seed
    ) == ("iterateOrder",)

    unaligned = refine.Workload(
        "det_unaligned", 127, 127, 32769,
        "fp16", False, False, 20,
    )
    assert refine.official_seed_order_fields(unaligned, seed) == ()

    single_core = dict(deterministic)
    single_core["tilingEnable"] = 2
    single_core_seed = SimpleNamespace(
        bank=SimpleNamespace(knowledge=single_core)
    )
    assert refine.official_seed_order_fields(
        aligned, single_core_seed
    ) == ()


def test_skinny_n_generalization_and_evidence_groups() -> None:
    seed_knowledge = base_knowledge(
        refine.Workload(
            "seed", 4096, 17, 16384,
            "fp16", False, False, 20,
        ),
        128,
        32,
        64,
    )
    seed = SimpleNamespace(
        bank=SimpleNamespace(knowledge=seed_knowledge)
    )
    for workload in (
        refine.Workload(
            "skinny_n_k16384_holdout_m3072_n17",
            3072, 17, 16384, "fp16", False, False, 20,
        ),
        refine.Workload(
            "skinny_n_k16384_holdout_m4096_n24",
            4096, 24, 16384, "fp16", False, False, 20,
        ),
        refine.Workload(
            "skinny_n_k16384_holdout_m5120_n29",
            5120, 29, 16384, "fp16", False, False, 20,
        ),
    ):
        assert refine.skinny_n_large_k_applicable(
            workload, seed, HARDWARE
        )
    boundary = refine.Workload(
        "skinny_n_boundary_n33",
        4096, 33, 16384, "fp16", False, False, 20,
    )
    assert not refine.skinny_n_large_k_applicable(
        boundary, seed, HARDWARE
    )
    assert refine.skinny_n_boundary_k16384_applicable(
        boundary, seed, HARDWARE
    )
    boundary64 = refine.Workload(
        "skinny_n_boundary64_holdout_m4096_n56",
        4096, 56, 16384, "fp16", False, False, 20,
    )
    assert not refine.skinny_n_boundary_k16384_applicable(
        boundary64, seed, HARDWARE
    )
    assert refine.skinny_n_boundary64_k16384_applicable(
        boundary64, seed, HARDWARE
    )
    transition48 = refine.Workload(
        "skinny_n_boundary_holdout_m5120_n48",
        5120, 48, 16384, "fp16", False, False, 20,
    )
    assert not refine.skinny_n_boundary_k16384_applicable(
        transition48, seed, HARDWARE
    )
    assert refine.skinny_n_transition48_k16384_applicable(
        transition48, seed, HARDWARE
    )
    for workload in (
        refine.Workload(
            "skinny_n_holdout_m3072_k8192",
            3072, 17, 8192, "fp16", False, False, 20,
        ),
        refine.Workload(
            "skinny_n_holdout_n24_k12288",
            4096, 24, 12288, "fp16", False, False, 20,
        ),
    ):
        assert not refine.skinny_n_large_k_applicable(
            workload, seed, HARDWARE
        )
        assert not refine.skinny_n_low_k_causal_applicable(
            workload, seed, HARDWARE
        )
        assert refine.skinny_n_low_k_falsified_applicable(
            workload, seed, HARDWARE
        )
    assert print_summary.evidence_group(
        "skinny_n_large_k"
    ) == "known_anchor"
    assert print_summary.evidence_group(
        "skinny_n_holdout_m3072_k8192"
    ) == "skinny_n_initial_holdout"
    assert print_summary.evidence_group(
        "skinny_n_k16384_holdout_m3072_n17"
    ) == "skinny_n_k16384_holdout"
    assert print_summary.evidence_group(
        "skinny_n_boundary_holdout_m3072_n40"
    ) == "skinny_n_boundary_holdout"
    assert print_summary.evidence_group(
        "skinny_n_boundary64_holdout_m3072_n49"
    ) == "skinny_n_boundary64_holdout"
    assert print_summary.evidence_group(
        "large_k_small_mn"
    ) == "det_split_k_positive_range"
    assert print_summary.evidence_group(
        "det_split_k_aligned_holdout_k16384"
    ) == "det_split_k_positive_range"
    assert print_summary.evidence_group(
        "det_split_k_aligned_holdout_k65536"
    ) == "det_split_k_rejected_control"
    assert print_summary.evidence_group(
        "llm_full_4096_ffn_up"
    ) == "prior_regression"
    assert print_summary.evidence_group(
        "int8_projection"
    ) == "unsupported_control"
    assert print_summary.strict_evidence_status(
        Counter({"improved": 1}), 3
    ) == "insufficient_evidence"
    assert print_summary.strict_evidence_status(
        Counter({"improved": 3}), 3
    ) == "supported"
    assert print_summary.strict_evidence_status(
        Counter({"improved": 2, "not_improved": 1}), 3
    ) == "not_supported"


def test_known_anchor_cannot_pass_broad_campaign() -> None:
    anchor_only = print_summary.campaign_statuses(
        {"known_anchor": Counter({"improved": 1})}
    )
    assert anchor_only["skinny_initial"] == "insufficient_evidence"
    assert anchor_only["skinny_k16384"] == "insufficient_evidence"
    assert anchor_only["skinny_boundary"] == "insufficient_evidence"
    assert anchor_only["skinny_boundary64"] == "insufficient_evidence"
    assert anchor_only["det_split_k"] == "insufficient_evidence"
    assert anchor_only["prior_failures"] == "insufficient_evidence"
    assert anchor_only["broad_validation"] == "insufficient_evidence"
    assert anchor_only["wide"] == "not_proven"

    complete = print_summary.campaign_statuses(
        {
            "known_anchor": Counter({"improved": 1}),
            "skinny_n_initial_holdout": Counter(
                {"improved": 1, "not_improved": 2}
            ),
            "skinny_n_k16384_holdout": Counter({"improved": 3}),
            "skinny_n_boundary_holdout": Counter({"improved": 3}),
            "skinny_n_boundary64_holdout": Counter({"improved": 3}),
            "det_split_k_positive_range": Counter({"improved": 3}),
            "prior_regression": Counter({"improved": 6}),
            "broad_validation": Counter({"improved": 26}),
        }
    )
    assert complete["wide"] == "supported"


def test_live_optimization_requires_both_baselines() -> None:
    assert profile_tilings.dual_baseline_optimization_status(
        "improved", "improved"
    ) == "improved"
    assert profile_tilings.dual_baseline_optimization_status(
        "within_noise", "improved"
    ) == "not_improved"
    assert profile_tilings.dual_baseline_optimization_status(
        "improved", "within_noise"
    ) == "not_improved"


def test_exact_resume_guard_refuses_any_prior_remeasurement() -> None:
    workloads = [
        {
            "workload_id": "old_fp16",
            "dtype": "fp16",
        },
        {
            "workload_id": "old_int8",
            "dtype": "int8",
        },
        {
            "workload_id": "new_fp16",
            "dtype": "fp16",
        },
    ]
    candidates = [
        {
            "workload_id": "old_fp16",
            "candidate_role": "bank_seed_control",
            "rank": "0",
        },
        {
            "workload_id": "old_fp16",
            "candidate_role": "searched",
            "rank": "1",
        },
        {
            "workload_id": "old_fp16",
            "candidate_role": "searched",
            "rank": "2",
            "search_resume_policy": "allow_new",
        },
        {
            "workload_id": "new_fp16",
            "candidate_role": "searched",
            "rank": "1",
        },
    ]
    protected, missing_baselines, missing_candidates = (
        profile_tilings.exact_resume_guard_missing(
            workloads,
            candidates,
            {"old_fp16"},
            {
                ("old_fp16", "bank_seed_control", "0"),
                ("old_fp16", "searched", "1"),
            },
            2,
        )
    )
    assert protected == {"old_fp16"}
    assert missing_baselines == []
    assert missing_candidates == []

    _, missing_baselines, missing_candidates = (
        profile_tilings.exact_resume_guard_missing(
            workloads,
            candidates,
            set(),
            {("old_fp16", "bank_seed_control", "0")},
            2,
        )
    )
    assert missing_baselines == ["old_fp16"]
    assert missing_candidates == [
        ("old_fp16", "searched", "1"),
    ]


def test_completed_frontier_requires_history_or_explicit_new_policy() -> None:
    measured = SimpleNamespace(
        row={
            "search_history_match": "candidate:measured",
            "search_resume_policy": "require_existing",
        }
    )
    preregistered = SimpleNamespace(
        row={
            "search_history_match": "",
            "search_resume_policy": "allow_new",
        }
    )
    accidental = SimpleNamespace(
        row={
            "search_history_match": "",
            "search_resume_policy": "require_existing",
        }
    )
    assert refine.completed_frontier_candidate_allowed(measured)
    assert refine.completed_frontier_candidate_allowed(preregistered)
    assert not refine.completed_frontier_candidate_allowed(accidental)


def test_l2_order_two_is_legal_and_follows_operand_reuse() -> None:
    b_larger = refine.Workload(
        "b_larger", 4096, 11008, 4096,
        "fp16", False, False, 20,
    )
    b_knowledge = base_knowledge(b_larger, 128, 256, 64)
    b_knowledge.update(
        depthA1=16,
        depthB1=8,
        stepKa=8,
        stepKb=4,
        l2MTileCnt=3,
        l2NTileCnt=2,
        l2MTileBlock=11,
        l2NTileBlock=22,
    )
    b_seed = SimpleNamespace(
        bank=SimpleNamespace(knowledge=b_knowledge)
    )
    b_proposal = refine.l2_reuse_order_proposal(
        b_larger, b_seed, HARDWARE
    )
    assert b_proposal is not None
    assert b_proposal.knowledge["l2IterateOrder"] == 2
    assert b_proposal.guidance == "bottleneck_l2_reuse_b_col_first"
    assert refine.hard_legal(
        b_larger, b_proposal.knowledge, HARDWARE
    )

    a_larger = refine.Workload(
        "a_larger", 11008, 4096, 4096,
        "fp16", False, False, 20,
    )
    a_knowledge = base_knowledge(a_larger, 256, 128, 64)
    a_knowledge.update(
        depthA1=8,
        depthB1=16,
        stepKa=4,
        stepKb=8,
        l2MTileCnt=2,
        l2NTileCnt=3,
        l2MTileBlock=22,
        l2NTileBlock=11,
    )
    a_seed = SimpleNamespace(
        bank=SimpleNamespace(knowledge=a_knowledge)
    )
    a_proposal = refine.l2_reuse_order_proposal(
        a_larger, a_seed, HARDWARE
    )
    assert a_proposal is not None
    assert a_proposal.knowledge["l2IterateOrder"] == 1
    assert a_proposal.guidance == "bottleneck_l2_reuse_a_row_first"
    assert refine.hard_legal(
        a_larger, a_proposal.knowledge, HARDWARE
    )


def test_broad_l2_frontier_is_closed_after_npu_falsification() -> None:
    workload = refine.Workload(
        "llm_8192_square", 8192, 8192, 8192,
        "fp16", False, False, 20,
    )
    knowledge = base_knowledge(workload, 128, 256, 64)
    knowledge.update(
        depthA1=16,
        depthB1=8,
        stepKa=8,
        stepKb=4,
        l2MTileCnt=5,
        l2NTileCnt=3,
        l2MTileBlock=13,
        l2NTileBlock=11,
    )
    callback = SimpleNamespace(knowledge=knowledge)
    seed = SimpleNamespace(bank=callback)
    estimate = refine.analytical_score(
        workload, knowledge, HARDWARE
    )
    bottleneck = refine.diagnose_bottleneck(
        workload, knowledge, estimate, HARDWARE
    )
    proposals, stop = refine.bottleneck_guided_candidate_proposals(
        workload, seed, HARDWARE, estimate, bottleneck
    )
    assert proposals == []
    assert stop == "broad_base_transitions_rejected_by_npu_evidence"


def test_smoke_keeps_one_searched_custom_tiling_path() -> None:
    workload = refine.Workload(
        "npu_smoke_fp16", 256, 256, 256,
        "fp16", False, False, 20,
    )
    knowledge = base_knowledge(workload, 64, 64, 128)
    knowledge.update(
        depthA1=16,
        depthB1=16,
        stepKa=8,
        stepKb=8,
        l2MTileCnt=1,
        l2NTileCnt=1,
        l2MTileBlock=4,
        l2NTileBlock=4,
    )
    seed = SimpleNamespace(
        bank=SimpleNamespace(knowledge=knowledge)
    )
    estimate = refine.analytical_score(
        workload, knowledge, HARDWARE
    )
    bottleneck = refine.diagnose_bottleneck(
        workload, knowledge, estimate, HARDWARE
    )
    proposals, stop = refine.bottleneck_guided_candidate_proposals(
        workload, seed, HARDWARE, estimate, bottleneck
    )
    assert stop == ""
    assert proposals


def test_guided_skinny_n_uses_the_measured_family_schedule() -> None:
    workload = refine.Workload(
        "skinny_n_large_k", 4096, 17, 16384,
        "fp16", False, False, 20,
    )
    knowledge = base_knowledge(workload, 128, 32, 64)
    knowledge.update(
        depthA1=16,
        depthB1=64,
        stepKa=8,
        stepKb=32,
        l2MTileCnt=8,
        l2NTileCnt=1,
        l2MTileBlock=4,
        l2NTileBlock=1,
    )
    seed = SimpleNamespace(
        bank=SimpleNamespace(knowledge=knowledge)
    )
    estimate = refine.analytical_score(
        workload, knowledge, HARDWARE
    )
    bottleneck = refine.diagnose_bottleneck(
        workload, knowledge, estimate, HARDWARE
    )
    proposals, stop = refine.bottleneck_guided_candidate_proposals(
        workload, seed, HARDWARE, estimate, bottleneck
    )
    assert stop == ""
    assert len(proposals) == 1
    assert proposals[0].guidance == "skinny_n_ablation_l1_rebalance"
    assert proposals[0].knowledge["l2MTileCnt"] == 5
    assert proposals[0].knowledge["l2MTileBlock"] == 4
    assert proposals[0].knowledge["depthA1"] == 16
    assert proposals[0].knowledge["depthB1"] == 8
    assert proposals[0].knowledge["dbL0C"] == 1
    assert proposals[0].knowledge["l2IterateOrder"] == 0
    assert proposals[0].resume_policy == "require_existing"


def test_guided_skinny_n_boundary_uses_measured_schedule() -> None:
    workload = refine.Workload(
        "skinny_n_boundary_n33", 4096, 33, 16384,
        "fp16", False, False, 20,
    )
    knowledge = base_knowledge(workload, 128, 48, 64)
    knowledge.update(
        depthA1=16,
        depthB1=32,
        stepKa=8,
        stepKb=16,
        l2MTileCnt=8,
        l2NTileCnt=1,
        l2MTileBlock=4,
        l2NTileBlock=1,
    )
    seed = SimpleNamespace(
        bank=SimpleNamespace(knowledge=knowledge)
    )
    estimate = refine.analytical_score(
        workload, knowledge, HARDWARE
    )
    bottleneck = refine.diagnose_bottleneck(
        workload, knowledge, estimate, HARDWARE
    )
    proposals, stop = refine.bottleneck_guided_candidate_proposals(
        workload, seed, HARDWARE, estimate, bottleneck
    )
    assert stop == ""
    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.guidance == "skinny_n_boundary_k16384_one_block_per_aic"
    assert proposal.resume_policy == "require_existing"
    assert proposal.knowledge["baseM"] == 208
    assert proposal.knowledge["baseN"] == 48
    assert proposal.knowledge["baseK"] == 64
    assert proposal.knowledge["depthA1"] == 8
    assert proposal.knowledge["depthB1"] == 40
    assert proposal.knowledge["stepKa"] == 4
    assert proposal.knowledge["stepKb"] == 20
    assert proposal.knowledge["l2MTileCnt"] == 1
    assert proposal.knowledge["l2MTileBlock"] == 20
    assert refine.hard_legal(workload, proposal.knowledge, HARDWARE)
    state = refine.State(
        row={
            "search_guidance": proposal.guidance,
        },
        knowledge=proposal.knowledge,
        model_score=1.0,
        normalized_score=1.0,
        hbm_bytes=1.0,
        l2_bytes=1.0,
        template="BASE",
        guidance=proposal.guidance,
    )
    assert refine.constraint_aware_beam([state], 4)[0].guidance == (
        "skinny_n_boundary_k16384_one_block_per_aic"
    )


def test_guided_skinny_n_boundary_holdouts_are_preregistered() -> None:
    for workload, expected_base_m in (
        (
            refine.Workload(
                "skinny_n_boundary_holdout_m3072_n40",
                3072, 40, 16384, "fp16", False, False, 20,
            ),
            160,
        ),
        (
            refine.Workload(
                "skinny_n_boundary_holdout_m4096_n47",
                4096, 47, 16384, "fp16", False, False, 20,
            ),
            208,
        ),
    ):
        knowledge = base_knowledge(workload, 128, 48, 64)
        knowledge.update(
            depthA1=16,
            depthB1=32,
            stepKa=8,
            stepKb=16,
        )
        seed = SimpleNamespace(
            bank=SimpleNamespace(knowledge=knowledge)
        )
        estimate = refine.analytical_score(
            workload, knowledge, HARDWARE
        )
        bottleneck = refine.diagnose_bottleneck(
            workload, knowledge, estimate, HARDWARE
        )
        proposals, stop = refine.bottleneck_guided_candidate_proposals(
            workload, seed, HARDWARE, estimate, bottleneck
        )
        assert stop == ""
        assert len(proposals) == 1
        proposal = proposals[0]
        assert proposal.resume_policy == "allow_new"
        assert proposal.knowledge["baseM"] == expected_base_m
        assert proposal.knowledge["baseN"] == 48
        assert proposal.knowledge["usedCoreNum"] == 20
        assert proposal.knowledge["l2MTileBlock"] == 20
        assert refine.hard_legal(
            workload, proposal.knowledge, HARDWARE
        )


def test_guided_skinny_n_transition48_uses_adjacent_base_n64() -> None:
    workload = refine.Workload(
        "skinny_n_boundary_holdout_m5120_n48",
        5120, 48, 16384, "fp16", False, False, 20,
    )
    knowledge = base_knowledge(workload, 128, 48, 64)
    seed = SimpleNamespace(bank=SimpleNamespace(knowledge=knowledge))
    estimate = refine.analytical_score(workload, knowledge, HARDWARE)
    bottleneck = refine.diagnose_bottleneck(
        workload, knowledge, estimate, HARDWARE
    )
    proposals, stop = refine.bottleneck_guided_candidate_proposals(
        workload, seed, HARDWARE, estimate, bottleneck
    )
    assert stop == ""
    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.resume_policy == "allow_new"
    assert proposal.guidance == (
        "skinny_n_transition48_k16384_base_n64_one_block_per_aic"
    )
    assert proposal.knowledge["baseM"] == 256
    assert proposal.knowledge["baseN"] == 64
    assert proposal.knowledge["depthA1"] == 8
    assert proposal.knowledge["depthB1"] == 32
    assert proposal.knowledge["stepKa"] == 4
    assert proposal.knowledge["stepKb"] == 16
    assert proposal.knowledge["usedCoreNum"] == 20
    assert proposal.knowledge["l2MTileBlock"] == 20
    assert refine.hard_legal(workload, proposal.knowledge, HARDWARE)


def test_transition48_survives_proxy_model_gate() -> None:
    workload = refine.Workload(
        "skinny_n_boundary_holdout_m5120_n48",
        5120, 48, 16384, "fp16", False, False, 20,
    )
    knowledge = base_knowledge(workload, 256, 64, 64)
    seed = SimpleNamespace(
        bank=SimpleNamespace(
            knowledge=base_knowledge(workload, 128, 48, 64)
        )
    )
    state = refine.State(
        row={
            "search_resume_policy": "allow_new",
            "search_model_confidence": "high",
            "search_transition_gain": "0.1",
            "search_history_match": "",
        },
        knowledge=knowledge,
        model_score=1.4,
        normalized_score=1.4,
        hbm_bytes=0.0,
        l2_bytes=0.0,
        template="BASE",
        guidance=(
            "skinny_n_transition48_k16384_base_n64_one_block_per_aic"
        ),
    )
    beam = refine.constraint_aware_beam([state], 64)
    assert beam == [state]
    assert state.guidance == (
        "skinny_n_transition48_k16384_base_n64_one_block_per_aic"
    )
    frontier = refine.guided_action_frontier(
        workload, seed, HARDWARE, beam, 1.03, 3
    )
    assert frontier == [state]
    assert state.row["search_model_confidence"] == "preregistered_new"
    assert state.row["search_stop_reason"] == (
        "source_guided_allow_new_candidate_despite_proxy_rejection"
    )


def test_transition48_campaign_contract_fails_before_npu_if_missing() -> None:
    workload = refine.Workload(
        "skinny_n_boundary_holdout_m5120_n48",
        5120, 48, 16384, "fp16", False, False, 20,
    )
    try:
        refine.assert_preregistered_campaign_candidates(
            [workload],
            [
                {
                    "workload_id": workload.workload_id,
                    "candidate_role": "bank_seed_control",
                }
            ],
        )
    except refine.SearchError as error:
        assert "crossover missing before NPU profiling" in str(error)
    else:
        raise AssertionError("missing N=48 crossover was not rejected")

    refine.assert_preregistered_campaign_candidates(
        [workload],
        [
            {
                "workload_id": workload.workload_id,
                "candidate_role": "searched",
                "base_m": "256",
                "base_n": "64",
                "base_k": "64",
                "depth_a1": "8",
                "depth_b1": "32",
                "search_guidance": (
                    "skinny_n_transition48_k16384_base_n64_"
                    "one_block_per_aic"
                ),
            }
        ],
    )


def test_new_search_forces_same_run_controls_only_for_that_workload() -> None:
    candidates = [
        {
            "workload_id": "new_shape",
            "candidate_role": "searched",
            "rank": "1",
            "search_resume_policy": "allow_new",
        },
        {
            "workload_id": "new_shape",
            "candidate_role": "bank_seed_control",
            "rank": "0",
            "search_resume_policy": "require_existing",
        },
        {
            "workload_id": "completed_shape",
            "candidate_role": "searched",
            "rank": "1",
            "search_resume_policy": "require_existing",
        },
        {
            "workload_id": "paired_shape",
            "candidate_role": "searched",
            "rank": "1",
            "search_resume_policy": "allow_new",
        },
        {
            "workload_id": "paired_shape",
            "candidate_role": "bank_seed_control",
            "rank": "0",
            "search_resume_policy": "require_existing",
        },
        {
            "workload_id": "unpaired_shape",
            "candidate_role": "searched",
            "rank": "1",
            "search_resume_policy": "allow_new",
        },
        {
            "workload_id": "unpaired_shape",
            "candidate_role": "bank_seed_control",
            "rank": "0",
            "search_resume_policy": "require_existing",
        },
    ]
    baselines = {
        "new_shape": {"median_ms": "2"},
        "completed_shape": {"median_ms": "3"},
        "paired_shape": {"median_ms": "4", "run_id": "same"},
        "unpaired_shape": {
            "median_ms": "5", "run_id": "candidate_run",
        },
    }
    assignments = {
        ("new_shape", "bank_seed_control", "0"): {"median_ms": "2.1"},
        ("completed_shape", "searched", "1"): {"median_ms": "2.5"},
        ("paired_shape", "searched", "1"): {
            "median_ms": "3.5", "run_id": "same",
        },
        ("paired_shape", "bank_seed_control", "0"): {
            "median_ms": "4.1", "run_id": "same",
        },
        ("unpaired_shape", "searched", "1"): {
            "median_ms": "4.5", "run_id": "candidate_run",
        },
    }
    workloads, removed_baselines, removed_schedules = (
        profile_tilings.force_paired_measurements_for_new_search(
            candidates, baselines, assignments
        )
    )
    assert workloads == {"new_shape", "unpaired_shape"}
    assert removed_baselines == 2
    assert removed_schedules == 2
    assert "new_shape" not in baselines
    assert baselines["completed_shape"]["median_ms"] == "3"
    assert (
        "completed_shape", "searched", "1"
    ) in assignments
    assert baselines["paired_shape"]["median_ms"] == "4"
    assert ("paired_shape", "searched", "1") in assignments
    assert ("paired_shape", "bank_seed_control", "0") in assignments


def test_guided_skinny_n_boundary64_holdouts_are_preregistered() -> None:
    for workload, expected_base_m in (
        (
            refine.Workload(
                "skinny_n_boundary64_holdout_m3072_n49",
                3072, 49, 16384, "fp16", False, False, 20,
            ),
            160,
        ),
        (
            refine.Workload(
                "skinny_n_boundary64_holdout_m4096_n56",
                4096, 56, 16384, "fp16", False, False, 20,
            ),
            208,
        ),
        (
            refine.Workload(
                "skinny_n_boundary64_holdout_m5120_n64",
                5120, 64, 16384, "fp16", False, False, 20,
            ),
            256,
        ),
    ):
        knowledge = base_knowledge(workload, 128, 64, 64)
        knowledge.update(
            depthA1=16,
            depthB1=32,
            stepKa=8,
            stepKb=16,
        )
        seed = SimpleNamespace(
            bank=SimpleNamespace(knowledge=knowledge)
        )
        estimate = refine.analytical_score(
            workload, knowledge, HARDWARE
        )
        bottleneck = refine.diagnose_bottleneck(
            workload, knowledge, estimate, HARDWARE
        )
        proposals, stop = refine.bottleneck_guided_candidate_proposals(
            workload, seed, HARDWARE, estimate, bottleneck
        )
        assert stop == ""
        assert len(proposals) == 1
        proposal = proposals[0]
        assert proposal.resume_policy == "allow_new"
        assert proposal.guidance == (
            "skinny_n_boundary_k16384_base_n64_one_block_per_aic"
        )
        assert proposal.knowledge["baseM"] == expected_base_m
        assert proposal.knowledge["baseN"] == 64
        assert proposal.knowledge["depthA1"] == 8
        assert proposal.knowledge["depthB1"] == 32
        assert proposal.knowledge["stepKa"] == 4
        assert proposal.knowledge["stepKb"] == 16
        assert proposal.knowledge["usedCoreNum"] == 20
        assert proposal.knowledge["l2MTileBlock"] == 20
        assert refine.hard_legal(
            workload, proposal.knowledge, HARDWARE
        )


def test_attention_score_frontier_is_closed_after_npu_falsification() -> None:
    workload = refine.Workload(
        "attention_score_1024", 1024, 1024, 128,
        "fp16", False, True, 20,
    )
    knowledge = base_knowledge(workload, 128, 256, 64)
    knowledge.update(
        depthA1=16,
        depthB1=8,
        stepKa=8,
        stepKb=4,
        l2MTileCnt=1,
        l2NTileCnt=1,
        l2MTileBlock=8,
        l2NTileBlock=4,
    )
    seed = SimpleNamespace(
        bank=SimpleNamespace(knowledge=knowledge)
    )
    candidates = refine.attention_score_k128_l2_candidate_space(
        workload, seed, HARDWARE
    )
    assert len(candidates) == 2
    assert all(refine.hard_legal(workload, row, HARDWARE) for row in candidates)
    assert {
        (
            row["baseM"],
            row["baseN"],
            row["l2MTileCnt"],
            row["l2NTileCnt"],
            row["l2MTileBlock"],
            row["l2NTileBlock"],
        )
        for row in candidates
    } == {
        (128, 208, 2, 1, 4, 5),
        (112, 256, 3, 1, 4, 4),
    }

    estimate = refine.analytical_score(
        workload, knowledge, HARDWARE
    )
    bottleneck = refine.diagnose_bottleneck(
        workload, knowledge, estimate, HARDWARE
    )
    proposals, stop = refine.bottleneck_guided_candidate_proposals(
        workload, seed, HARDWARE, estimate, bottleneck
    )
    assert proposals == []
    assert stop == "broad_base_transitions_rejected_by_npu_evidence"


def test_bank_advantage_order_is_closed_after_npu_falsification() -> None:
    for workload in (
        refine.Workload(
            "odd_tail_2", 1025, 4097, 3073,
            "fp16", False, False, 20,
        ),
        refine.Workload(
            "trans_ab_case", 1024, 1536, 2048,
            "fp16", True, True, 20,
        ),
    ):
        knowledge = base_knowledge(workload, 128, 256, 64)
        knowledge.update(
            depthA1=16,
            depthB1=8,
            stepKa=8,
            stepKb=4,
        )
        assert refine.hard_legal(workload, knowledge, HARDWARE)
        seed = SimpleNamespace(
            bank=SimpleNamespace(knowledge=knowledge)
        )
        estimate = refine.analytical_score(
            workload, knowledge, HARDWARE
        )
        bottleneck = refine.diagnose_bottleneck(
            workload, knowledge, estimate, HARDWARE
        )
        proposals, stop = refine.bottleneck_guided_candidate_proposals(
            workload, seed, HARDWARE, estimate, bottleneck
        )
        assert proposals == []
        assert stop == "broad_base_transitions_rejected_by_npu_evidence"


def test_completed_frontier_preserves_boundary_holdout_policy() -> None:
    state = SimpleNamespace(
        row={
            "search_history_match": "",
            "search_resume_policy": "allow_new",
        }
    )
    assert refine.completed_frontier_candidate_allowed(state)


def test_guided_low_k_skinny_n_is_closed_after_falsification() -> None:
    workload = refine.Workload(
        "skinny_n_holdout_m3072_k8192", 3072, 17, 8192,
        "fp16", False, False, 20,
    )
    knowledge = base_knowledge(workload, 128, 32, 64)
    knowledge.update(
        depthA1=16,
        depthB1=64,
        stepKa=8,
        stepKb=32,
        l2MTileCnt=1,
        l2NTileCnt=1,
        l2MTileBlock=24,
        l2NTileBlock=1,
    )
    seed = SimpleNamespace(
        bank=SimpleNamespace(knowledge=knowledge)
    )
    estimate = refine.analytical_score(
        workload, knowledge, HARDWARE
    )
    bottleneck = refine.diagnose_bottleneck(
        workload, knowledge, estimate, HARDWARE
    )
    proposals, stop = refine.bottleneck_guided_candidate_proposals(
        workload, seed, HARDWARE, estimate, bottleneck
    )
    assert proposals == []
    assert stop == "low_k_skinny_n_extrapolation_rejected_by_npu_evidence"
    assert refine.skinny_n_low_k_falsified_applicable(
        workload, seed, HARDWARE
    )


def test_guided_split_k_emits_only_source_supported_transition() -> None:
    aligned = refine.Workload(
        "large_k_small_mn", 128, 128, 32768,
        "fp16", False, False, 20,
    )
    knowledge = {
        "usedCoreNum": 20,
        "singleCoreM": 384,
        "singleCoreN": 128,
        "singleCoreK": 384,
        "baseM": 128,
        "baseN": 128,
        "baseK": 128,
        "depthA1": 9,
        "depthB1": 6,
        "stepM": 3,
        "stepN": 1,
        "iterateOrder": 0,
        "stepKa": 3,
        "stepKb": 3,
        "dbL0A": 2,
        "dbL0B": 2,
        "dbL0C": 2,
        "l2MTileCnt": 1,
        "l2NTileCnt": 1,
        "l2MTileBlock": 1,
        "l2NTileBlock": 1,
        "l2IterateOrder": 0,
        "tilingEnable": 3,
    }
    assert refine.hard_legal(aligned, knowledge, HARDWARE)
    seed = SimpleNamespace(
        bank=SimpleNamespace(knowledge=knowledge)
    )
    estimate = refine.analytical_score(
        aligned, knowledge, HARDWARE
    )
    bottleneck = refine.diagnose_bottleneck(
        aligned, knowledge, estimate, HARDWARE
    )
    proposals, stop = refine.bottleneck_guided_candidate_proposals(
        aligned, seed, HARDWARE, estimate, bottleneck
    )
    assert stop == ""
    assert len(proposals) == 1
    assert proposals[0].guidance == (
        "official_seed_iterate_order_ablation"
    )
    assert proposals[0].resume_policy == "require_existing"
    assert proposals[0].knowledge["iterateOrder"] == 1

    unaligned = refine.Workload(
        "unaligned_split_k", 127, 127, 32769,
        "fp16", False, False, 20,
    )
    unaligned_estimate = refine.analytical_score(
        unaligned, knowledge, HARDWARE
    )
    unaligned_bottleneck = refine.diagnose_bottleneck(
        unaligned, knowledge, unaligned_estimate, HARDWARE
    )
    proposals, stop = refine.bottleneck_guided_candidate_proposals(
        unaligned,
        seed,
        HARDWARE,
        unaligned_estimate,
        unaligned_bottleneck,
    )
    assert proposals == []
    assert stop == (
        "unaligned_split_k_order_is_excluded_by_negative_npu_evidence"
    )

    k65536 = refine.Workload(
        "det_split_k_aligned_holdout_k65536", 128, 128, 65536,
        "fp16", False, False, 20,
    )
    assert not refine.deterministic_split_k_supported_range(k65536)
    k65536_estimate = refine.analytical_score(
        k65536, knowledge, HARDWARE
    )
    k65536_bottleneck = refine.diagnose_bottleneck(
        k65536, knowledge, k65536_estimate, HARDWARE
    )
    proposals, stop = refine.bottleneck_guided_candidate_proposals(
        k65536,
        seed,
        HARDWARE,
        k65536_estimate,
        k65536_bottleneck,
    )
    assert proposals == []
    assert stop == "deterministic_split_k_range_rejected_by_npu_evidence"


def test_history_calibration_uses_same_run_bank_control() -> None:
    workload = refine.Workload(
        "history_case", 4096, 17, 16384,
        "fp16", False, False, 20,
    )
    knowledge = base_knowledge(workload, 208, 32, 64)
    state = refine.State(
        row={"search_model_confidence": "high"},
        knowledge=knowledge,
        model_score=1.0,
        normalized_score=0.97,
        hbm_bytes=0.0,
        l2_bytes=0.0,
        template="BASE",
    )
    history = {
        refine.state_history_key(workload, knowledge): [
            refine.MeasurementEvidence(
                ratio_vs_official=0.66,
                ratio_vs_bank=0.64,
                record_id="candidate:history_case:test",
            )
        ]
    }
    correction, matches = refine.calibrate_from_history(
        workload,
        [state],
        history,
        "baseBD:410->1366",
    )
    assert correction == 1.0
    assert matches == 1
    assert state.normalized_score == 0.64
    assert state.row["search_model_confidence"] == "measured_history"
    assert state.row["search_history_match"] == (
        "candidate:history_case:test"
    )


def test_large_official_improvement_survives_missing_bank_pair() -> None:
    workload = refine.Workload(
        "strong_history_case", 4096, 33, 16384,
        "fp16", False, False, 20,
    )
    knowledge = base_knowledge(workload, 208, 48, 64)
    state = refine.State(
        row={"search_model_confidence": "high"},
        knowledge=knowledge,
        model_score=1.0,
        normalized_score=1.01,
        hbm_bytes=0.0,
        l2_bytes=0.0,
        template="BASE",
    )
    key = refine.state_history_key(workload, knowledge)
    correction, matches = refine.calibrate_from_history(
        workload,
        [state],
        {
            key: [
                refine.MeasurementEvidence(
                    ratio_vs_official=0.85,
                    ratio_vs_bank=None,
                    record_id="candidate:strong_history_case:test",
                )
            ]
        },
        "baseBD:410->911",
    )
    assert correction == 1.0
    assert matches == 1
    assert state.normalized_score == 0.85
    assert state.row["search_model_confidence"] == "measured_history"

    weak = refine.State(
        row={"search_model_confidence": "high"},
        knowledge=knowledge,
        model_score=1.0,
        normalized_score=1.01,
        hbm_bytes=0.0,
        l2_bytes=0.0,
        template="BASE",
    )
    _, weak_matches = refine.calibrate_from_history(
        workload,
        [weak],
        {
            key: [
                refine.MeasurementEvidence(
                    ratio_vs_official=0.95,
                    ratio_vs_bank=None,
                    record_id="candidate:weak_history_case:test",
                )
            ]
        },
        "baseBD:410->911",
    )
    assert weak_matches == 0
    assert weak.normalized_score == 1.01


def test_three_measured_anchors_allow_one_proxy_disagreement() -> None:
    workload = refine.Workload(
        "skinny_holdout", 3072, 17, 16384,
        "fp16", False, False, 20,
    )
    knowledge = base_knowledge(workload, 160, 32, 64)
    seed_knowledge = base_knowledge(workload, 128, 32, 64)
    seed = SimpleNamespace(
        bank=SimpleNamespace(knowledge=seed_knowledge)
    )
    states = [
        refine.State(
            row={
                "search_guidance": "skinny_n_ablation_l1_rebalance",
                "search_model_confidence": "high",
                "search_transition_gain": "0.1",
                "search_rationale": "learned family schedule",
                "search_history_match": "",
            },
            knowledge=dict(knowledge),
            model_score=1.38,
            normalized_score=1.38,
            hbm_bytes=0.0,
            l2_bytes=0.0,
            template="BASE",
            guidance="skinny_n_ablation_l1_rebalance",
        )
    ]
    frontier = refine.guided_action_frontier(
        workload, seed, HARDWARE, states, 1.03, 2
    )
    assert frontier == []
    frontier = refine.guided_action_frontier(
        workload, seed, HARDWARE, states, 1.03, 3
    )
    assert len(frontier) == 1
    assert frontier[0].guidance == "skinny_n_ablation_l1_rebalance"
    assert frontier[0].row["search_model_confidence"] == (
        "evidence_transfer"
    )


def test_low_k_evidence_transfer_stays_closed_after_falsification() -> None:
    workload = refine.Workload(
        "skinny_n_holdout_m3072_k8192", 3072, 17, 8192,
        "fp16", False, False, 20,
    )
    knowledge = base_knowledge(workload, 160, 32, 64)
    seed_knowledge = base_knowledge(workload, 128, 32, 64)
    seed = SimpleNamespace(
        bank=SimpleNamespace(knowledge=seed_knowledge)
    )
    state = refine.State(
        row={
            "search_guidance": "skinny_n_low_k_l2_partition",
            "search_model_confidence": "high",
            "search_transition_gain": "0.1",
            "search_rationale": "single-variable L2 test",
            "search_history_match": "0",
        },
        knowledge=dict(knowledge),
        model_score=1.38,
        normalized_score=1.38,
        hbm_bytes=0.0,
        l2_bytes=0.0,
        template="BASE",
        guidance="skinny_n_low_k_l2_partition",
    )
    frontier = refine.guided_action_frontier(
        workload, seed, HARDWARE, [state], 1.03, 3
    )
    assert frontier == []


def test_completed_frontier_coverage_is_exact() -> None:
    fields = [
        "record_type",
        "soc",
        "aic",
        "ocr_complete",
        "status",
        "global_index",
        "workload_index",
        "notes",
    ]
    row = {
        "record_type": "coverage",
        "soc": "Ascend910B3",
        "aic": "20",
        "ocr_complete": "1",
        "status": "success",
        "global_index": "1-47",
        "workload_index": "1-46",
        "notes": "complete_frontier_exact_resume_137",
    }
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "history.csv"
        with path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row)
        assert refine.completed_bottleneck_frontier_coverage(
            path, "Ascend910B3", 20
        )
        assert not refine.completed_bottleneck_frontier_coverage(
            path, "Ascend910B3", 24
        )


def test_base_partition_must_fit_one_base_tile() -> None:
    workload = refine.Workload(
        "large_base_partition", 4096, 11008, 4096,
        "fp16", False, False, 20,
    )
    knowledge = {
        "usedCoreNum": 20,
        "singleCoreM": 512,
        "singleCoreN": 736,
        "singleCoreK": 4096,
        "baseM": 128,
        "baseN": 256,
        "baseK": 64,
        "depthA1": 16,
        "depthB1": 8,
        "stepM": 1,
        "stepN": 1,
        "iterateOrder": 0,
        "stepKa": 8,
        "stepKb": 4,
        "dbL0A": 2,
        "dbL0B": 2,
        "dbL0C": 1,
        "l2MTileCnt": 2,
        "l2NTileCnt": 2,
        "l2MTileBlock": 4,
        "l2NTileBlock": 8,
        "l2IterateOrder": 0,
        "tilingEnable": 0,
    }
    assert knowledge["singleCoreM"] > knowledge["baseM"]
    assert knowledge["singleCoreN"] > knowledge["baseN"]
    assert not refine.hard_legal(workload, knowledge, HARDWARE)

    knowledge.update({
        "singleCoreM": 128,
        "singleCoreN": 256,
        "l2MTileCnt": 1,
        "l2NTileCnt": 1,
        "l2MTileBlock": 32,
        "l2NTileBlock": 43,
    })
    assert refine.hard_legal(workload, knowledge, HARDWARE)


def test_idle_official_cores_are_legal() -> None:
    workload = refine.Workload(
        "skinny_official_seed", 17, 4096, 16384,
        "fp16", False, False, 20,
    )
    knowledge = {
        "usedCoreNum": 20,
        "singleCoreM": 32,
        "singleCoreN": 256,
        "singleCoreK": 16384,
        "baseM": 32,
        "baseN": 256,
        "baseK": 64,
        "depthA1": 64,
        "depthB1": 8,
        "stepM": 1,
        "stepN": 1,
        "iterateOrder": 0,
        "stepKa": 32,
        "stepKb": 4,
        "dbL0A": 2,
        "dbL0B": 2,
        "dbL0C": 1,
        "l2MTileCnt": 1,
        "l2NTileCnt": 4,
        "l2MTileBlock": 1,
        "l2NTileBlock": 5,
        "l2IterateOrder": 0,
        "tilingEnable": 0,
    }
    output_tiles = (
        refine.ceil_div(workload.m, knowledge["singleCoreM"])
        * refine.ceil_div(workload.n, knowledge["singleCoreN"])
    )
    assert output_tiles == 16
    assert knowledge["usedCoreNum"] > output_tiles
    assert refine.hard_legal(workload, knowledge, HARDWARE)


def test_effective_l1_matches_official_allocator_boundary() -> None:
    reported = refine.Hardware(
        aic_cores=20,
        l0a_bytes=64 * 1024,
        l0b_bytes=64 * 1024,
        l0c_bytes=128 * 1024,
        l1_bytes=524032,
        l2_bytes=192 * 1024 * 1024,
        l2_bytes_per_cycle_per_core=110.0,
        hbm_bytes_per_cycle_per_core=32.0,
    )
    assert refine.effective_l1_bytes(reported) == 512 * 1024


def test_bank_control_is_not_api_auto_baseline() -> None:
    control = {
        "rank": "0",
        "source": "official_seed_bank_roundtrip",
        "candidate_role": "bank_seed_control",
    }
    assert not rank_results.is_api_auto_baseline(control)
    assert rank_results.is_bank_seed_control(control)


def test_profile_failure_classification() -> None:
    coverage = profile_tilings.RunnerProfileError(
        "official output coverage failed at C index=2048",
        {},
        [],
        "official preflight synchronize",
        1,
    )
    assert profile_tilings.isolated_candidate_failure(coverage)

    runtime = profile_tilings.RunnerProfileError(
        "aclInit failed, rc=507008",
        {},
        [],
        "",
        1,
    )
    assert not profile_tilings.isolated_candidate_failure(runtime)


def test_failed_candidate_profile_preserves_workload_identity() -> None:
    candidate = {
        "workload_id": "fp32_square",
        "rank": "1",
        "m": "2048",
        "n": "2048",
        "k": "2048",
        "dtype": "fp32",
        "trans_a": "0",
        "trans_b": "0",
    }
    failed = profile_tilings.failed_runner_row(
        "fp32_square", "official MatMulV3 timeout after 60s"
    )
    row = profile_tilings.candidate_profile(
        candidate,
        failed,
        "1",
        "searched",
        "searched",
    )
    assert tuple(row[field] for field in rank_results.SHAPE_FIELDS) == (
        "2048", "2048", "2048", "fp32", "0", "0",
    )
    assert row["success"] == "0"
    assert "timeout" in row["error"]


def test_empty_runner_csv_reports_parse_error() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "empty.csv"
        path.write_text("", encoding="utf-8")
        try:
            profile_tilings.read_csv(path)
        except profile_tilings.ProfileError as exception:
            assert "missing CSV header" in str(exception)
        else:
            raise AssertionError("empty runner CSV was accepted")


def test_history_reuse_does_not_require_candidate_directory() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        candidate_root = Path(temporary) / "candidate" / "history_rank0"
        history = {"run_id": "previous"}
        run_dir = profile_tilings.prepare_measurement_run_dir(
            candidate_root, history
        )
        assert run_dir is None
        assert not candidate_root.exists()

        run_dir = profile_tilings.prepare_measurement_run_dir(
            candidate_root, None
        )
        assert run_dir == candidate_root / "run"
        assert run_dir.is_dir()


def test_historical_official_profile_preserves_implementation_identity() -> None:
    workload = {
        "workload_id": "skinny_n_large_k",
        "m": "4096",
        "n": "17",
        "k": "16384",
        "dtype": "fp16",
        "trans_a": "0",
        "trans_b": "0",
    }
    history = {
        "record_key": "baseline:skinny_n_large_k",
        "run_id": "20260727_074700",
        "rank": "",
        "median_ms": "0.103036",
        "stddev_ms": "0.000200036",
        "tflops": "22.1446",
    }
    row = profile_tilings.historical_official_profile(workload, history)
    assert row["rank"] == "-1"
    assert row["source"] == "installed_aclnn_matmul"
    assert row["candidate_role"] == "official_operator_baseline"
    assert row["measurement_source"].startswith("history:")


def test_legacy_baseline_reuse_requires_complete_workload_identity() -> None:
    workload = {
        "workload_id": "legacy_case",
        "m": "128",
        "n": "256",
        "k": "512",
        "dtype": "fp16",
        "trans_a": "0",
        "trans_b": "1",
    }
    history = {
        "workload_id": "legacy_case",
        "shape": "128x256x512",
        "dtype": "fp16",
        "trans_a": "0",
        "trans_b": "1",
    }
    assert profile_tilings.historical_baseline_matches_workload(
        workload, history
    )
    for field, value in (
        ("shape", "128x256x1024"),
        ("dtype", "bf16"),
        ("trans_a", "1"),
        ("trans_b", "0"),
    ):
        changed = dict(history)
        changed[field] = value
        assert not profile_tilings.historical_baseline_matches_workload(
            workload, changed
        )


def test_exact_profile_resume_uses_full_tiling_fingerprint() -> None:
    knowledge = base_knowledge(
        refine.Workload(
            "resume_case", 4096, 17, 16384,
            "fp16", False, False, 20,
        ),
        208,
        32,
        64,
    )
    signature = ":".join(
        str(knowledge[field]) for field in refine.KNOWLEDGE_FIELDS
    )
    callback_sha = "a" * 64
    profile = {
        column: "" for column in profile_tilings.PROFILE_COLUMNS
    }
    profile.update(
        {
            "workload_id": "resume_case",
            "rank": "4",
            "source": "cann81_official_local_v2",
            "candidate_role": "searched",
            "m": "4096",
            "n": "17",
            "k": "16384",
            "dtype": "fp16",
            "trans_a": "0",
            "trans_b": "0",
            "used_core_num": "20",
            "kernel_template": "MatMulV3_BASE",
            "success": "1",
            "preflight_passed": "1",
            "median_ms": "0.0674828",
            "stddev_ms": "0.000171182",
            "tiling_signature": signature,
            "callback_tiling_sha256": callback_sha,
        }
    )
    current = {
        "workload_id": "resume_case",
        "rank": "1",
        "candidate_role": "searched",
        "m": "4096",
        "n": "17",
        "k": "16384",
        "dtype": "fp16",
        "trans_a": "0",
        "trans_b": "0",
        "search_template": "BASE",
        "tiling_signature": signature,
        "callback_tiling_sha256": callback_sha,
    }
    assert (
        profile_tilings.exact_profile_fingerprint(profile)
        == profile_tilings.exact_profile_fingerprint(current)
    )
    changed = dict(current)
    changed_words = signature.split(":")
    changed_words[17] = str(1 - int(changed_words[17]))
    changed["tiling_signature"] = ":".join(changed_words)
    assert (
        profile_tilings.exact_profile_fingerprint(profile)
        != profile_tilings.exact_profile_fingerprint(changed)
    )

    official = {
        column: "" for column in profile_tilings.PROFILE_COLUMNS
    }
    official.update(
        {
            "workload_id": "resume_case",
            "rank": "-1",
            "source": "installed_aclnn_matmul",
            "candidate_role": "official_operator_baseline",
            "m": "4096",
            "n": "17",
            "k": "16384",
            "dtype": "fp16",
            "trans_a": "0",
            "trans_b": "0",
            "success": "1",
            "preflight_passed": "1",
            "median_ms": "0.103036",
            "stddev_ms": "0.000200036",
        }
    )

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "old_candidates.csv"
        resume = root / "resume.csv"
        with source.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(
                destination,
                fieldnames=profile_tilings.PROFILE_COLUMNS,
            )
            writer.writeheader()
            writer.writerow(profile)
            writer.writerow(official)
        count = profile_tilings.merge_exact_profile_history(
            resume,
            [source],
            "Ascend910B3",
            20,
            "test_run",
        )
        assert count == 2
        history = profile_tilings.load_exact_profile_history(
            [resume], "Ascend910B3", 20
        )
        key = profile_tilings.exact_profile_fingerprint(current)
        assert key in history
        assert history[key][0]["median_ms"] == "0.0674828"
        assert not profile_tilings.load_exact_profile_history(
            [resume], "Ascend910B2", 20
        )
        official_history = profile_tilings.load_exact_official_history(
            [resume], "Ascend910B3", 20
        )
        official_key = profile_tilings.exact_official_fingerprint(official)
        assert official_key in official_history
        assert official_history[official_key][0]["median_ms"] == "0.103036"


def main() -> None:
    test_layout_specific_fp32_alignment()
    test_complete_callback_blob_parsing()
    test_cann81_kernel_key_contract()
    test_twenty_core_slot_balance_is_ranked()
    test_focused_skinny_n_ablation_space()
    test_skinny_n_l1_rebalance_respects_base_m_256_capacity()
    test_official_local_search_changes_one_order_field()
    test_general_search_builds_independent_multistart_sources()
    test_general_transfer_reconstructs_target_partition_geometry()
    test_general_frontier_preserves_each_start_source()
    test_general_active_frontier_excludes_measured_fingerprints()
    test_profile_resume_drives_search_history_and_transfer()
    test_unstable_profile_is_excluded_without_calibration()
    test_campaign_manifest_excludes_exact_fingerprint()
    test_campaign_observations_calibrate_sources_and_transfer()
    test_unstable_comparison_cannot_claim_improvement()
    test_unpaired_exact_profile_is_excluded_without_calibration()
    test_history_calibration_is_source_specific()
    test_split_k_order_search_requires_aligned_deterministic_template()
    test_skinny_n_generalization_and_evidence_groups()
    test_known_anchor_cannot_pass_broad_campaign()
    test_live_optimization_requires_both_baselines()
    test_exact_resume_guard_refuses_any_prior_remeasurement()
    test_completed_frontier_requires_history_or_explicit_new_policy()
    test_l2_order_two_is_legal_and_follows_operand_reuse()
    test_broad_l2_frontier_is_closed_after_npu_falsification()
    test_smoke_keeps_one_searched_custom_tiling_path()
    test_guided_skinny_n_uses_the_measured_family_schedule()
    test_guided_skinny_n_boundary_uses_measured_schedule()
    test_guided_skinny_n_boundary_holdouts_are_preregistered()
    test_guided_skinny_n_transition48_uses_adjacent_base_n64()
    test_transition48_survives_proxy_model_gate()
    test_transition48_campaign_contract_fails_before_npu_if_missing()
    test_new_search_forces_same_run_controls_only_for_that_workload()
    test_guided_skinny_n_boundary64_holdouts_are_preregistered()
    test_attention_score_frontier_is_closed_after_npu_falsification()
    test_bank_advantage_order_is_closed_after_npu_falsification()
    test_completed_frontier_preserves_boundary_holdout_policy()
    test_guided_low_k_skinny_n_is_closed_after_falsification()
    test_guided_split_k_emits_only_source_supported_transition()
    test_history_calibration_uses_same_run_bank_control()
    test_large_official_improvement_survives_missing_bank_pair()
    test_three_measured_anchors_allow_one_proxy_disagreement()
    test_low_k_evidence_transfer_stays_closed_after_falsification()
    test_completed_frontier_coverage_is_exact()
    test_base_partition_must_fit_one_base_tile()
    test_idle_official_cores_are_legal()
    test_effective_l1_matches_official_allocator_boundary()
    test_bank_control_is_not_api_auto_baseline()
    test_profile_failure_classification()
    test_failed_candidate_profile_preserves_workload_identity()
    test_empty_runner_csv_reports_parse_error()
    test_history_reuse_does_not_require_candidate_directory()
    test_historical_official_profile_preserves_implementation_identity()
    test_legacy_baseline_reuse_requires_complete_workload_identity()
    test_exact_profile_resume_uses_full_tiling_fingerprint()
    print("refine_model_test passed")


if __name__ == "__main__":
    main()
