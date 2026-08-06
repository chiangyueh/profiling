from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path


RESEARCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH))

from tiling_search import (
    CandidateEngine,
    GenerationBudget,
    Hardware,
    SearchConfig,
    Template,
    Workload,
    direct_base_candidate,
    fit_direct_base_evidence,
)
from generate import load_resume_feedback
from tiling_search.contracts import template_of, validate_schedule
from tiling_search.solvers.base_policy import upstream_base_l2_policy


class DeploymentStrategiesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.hardware = Hardware(
            aic_cores=20,
            l0a_bytes=65536,
            l0b_bytes=65536,
            l0c_bytes=131072,
            l1_bytes=524032,
            l2_bytes=201326592,
            l2_bytes_per_cycle_per_core=110.0,
            hbm_bytes_per_cycle_per_core=32.0,
            ub_bytes=262144,
        )

    def test_direct_base_is_one_coupled_legal_record(self) -> None:
        workload = Workload(
            workload_id="direct_dense",
            m=3584,
            n=4352,
            k=6656,
            dtype="fp16",
            trans_a=False,
            trans_b=False,
            max_cores=20,
        )
        candidate = direct_base_candidate(workload, self.hardware)

        self.assertEqual(candidate.source, "direct_base_policy")
        self.assertEqual(candidate.template, Template.BASE)
        self.assertEqual(candidate.metrics["model_enabled"], 0.0)
        self.assertEqual(candidate.metrics["history_rows_used"], 0.0)
        self.assertEqual(candidate.metrics["candidate_pool_size"], 1.0)
        self.assertNotIn(
            "predicted_latency_ratio",
            candidate.metrics,
        )
        self.assertTrue(
            validate_schedule(
                workload, candidate.schedule, self.hardware
            ).valid
        )
        self.assertEqual(
            (
                candidate.schedule["baseM"],
                candidate.schedule["baseN"],
                candidate.schedule["baseK"],
                candidate.schedule["depthA1"],
                candidate.schedule["depthB1"],
                candidate.schedule["stepKa"],
                candidate.schedule["stepKb"],
            ),
            (128, 256, 64, 16, 8, 8, 4),
        )
        self.assertEqual(
            (
                candidate.schedule["l2MTileCnt"],
                candidate.schedule["l2NTileCnt"],
                candidate.schedule["l2MTileBlock"],
                candidate.schedule["l2NTileBlock"],
            ),
            (2, 1, 14, 17),
        )
        self.assertEqual(
            candidate.metrics["policy_l2_mode_cache"], 1.0
        )

    def test_direct_base_keeps_small_working_set_resident(self) -> None:
        workload = Workload(
            workload_id="direct_small",
            m=512,
            n=512,
            k=512,
            dtype="fp16",
            trans_a=False,
            trans_b=False,
            max_cores=20,
        )
        candidate = direct_base_candidate(workload, self.hardware)

        self.assertEqual(
            (
                candidate.schedule["l2MTileCnt"],
                candidate.schedule["l2NTileCnt"],
                candidate.schedule["l2MTileBlock"],
                candidate.schedule["l2NTileBlock"],
            ),
            (1, 1, 5, 4),
        )
        self.assertEqual(
            candidate.metrics["policy_l2_mode_whole"], 1.0
        )

    def test_history_distillation_reconstructs_all_base_banks(
        self,
    ) -> None:
        names = (
            "paired_measurements_net_log25_26.csv",
            "paired_measurements_net_log27.csv",
            "paired_measurements_net_log28.csv",
            "paired_measurements_net_log30.csv",
            "paired_measurements_net_log31.csv",
        )
        config = RESEARCH / "config"
        observations = []
        for name in names:
            loaded, _ = load_resume_feedback(
                config / name,
                "Ascend910B3",
                20,
                "8.1.RC1+toolkit-7.7.0.1.225",
            )
            observations.extend(loaded)

        evidence = fit_direct_base_evidence(
            observations, self.hardware
        )
        self.assertEqual(evidence.paired_base_records, 498)
        self.assertEqual(evidence.unique_base_records, 373)
        self.assertEqual(evidence.base_workloads, 78)
        self.assertEqual(evidence.bank_base_workloads, 58)
        self.assertEqual(evidence.geometry_policy_matches, 58)
        self.assertEqual(evidence.l1_policy_matches, 58)
        self.assertEqual(evidence.core_policy_matches, 58)
        self.assertEqual(evidence.whole_l2_bank_workloads, 36)
        self.assertEqual(
            evidence.partitioned_l2_bank_workloads, 22
        )
        self.assertGreater(
            evidence.resident_l2_threshold_bytes,
            94.625 * 1024 * 1024,
        )
        self.assertLess(
            evidence.resident_l2_threshold_bytes,
            101.553 * 1024 * 1024,
        )

        banks = {}
        for observation in observations:
            bank = observation.bank_schedule
            if (
                observation.verified
                and bank is not None
                and template_of(bank) == Template.BASE
            ):
                banks[observation.workload.identity()] = (
                    observation.workload,
                    bank,
                )
        self.assertEqual(len(banks), 58)
        for workload, bank in banks.values():
            reconstructed, _ = upstream_base_l2_policy(
                workload,
                self.hardware,
                base_m=bank["baseM"],
                base_n=bank["baseN"],
                resident_threshold_bytes=(
                    evidence.resident_l2_threshold_bytes
                ),
            )
            expected = tuple(
                bank[field]
                for field in (
                    "l2MTileCnt",
                    "l2NTileCnt",
                    "l2MTileBlock",
                    "l2NTileBlock",
                    "l2IterateOrder",
                )
            )
            self.assertEqual(
                reconstructed,
                expected,
                workload.workload_id,
            )
            candidate = direct_base_candidate(
                workload,
                self.hardware,
                evidence,
            )
            self.assertEqual(
                candidate.schedule.signature(),
                bank.signature(),
                workload.workload_id,
            )

    def test_direct_base_covers_dtype_and_layout_variants(self) -> None:
        cases = (
            ("bf16_nn", 1791, 2433, 4609, "bf16", False, False),
            ("fp32_nt", 144, 768, 4864, "fp32", False, True),
            ("fp32_tn", 1152, 1664, 2304, "fp32", True, False),
            ("fp32_tt", 896, 1280, 1792, "fp32", True, True),
            ("fp16_shallow_k", 160, 255, 31, "fp16", False, False),
            ("fp32_nt_shallow_k", 80, 129, 16, "fp32", False, True),
        )
        for case in cases:
            with self.subTest(case=case[0]):
                workload = Workload(*case, max_cores=20)
                candidate = direct_base_candidate(
                    workload, self.hardware
                )
                self.assertTrue(
                    validate_schedule(
                        workload, candidate.schedule, self.hardware
                    ).valid
                )

    def test_compact_frontier_is_bounded_and_template_aware(self) -> None:
        workload = Workload(
            workload_id="compact_deep_k",
            m=640,
            n=448,
            k=28672,
            dtype="fp16",
            trans_a=False,
            trans_b=False,
            max_cores=20,
        )
        engine = CandidateEngine(
            config=SearchConfig(
                budget=GenerationBudget(
                    raw_attempts=48,
                    legal_candidates=12,
                    behavior_candidates=12,
                    callback_candidates=8,
                    npu_candidates=1,
                ),
                include_exploration=False,
            ),
            observations=(),
            exclusions=set(),
        )
        result = engine.generate(workload, self.hardware)
        templates = Counter(
            candidate.template for candidate in result.callback_candidates
        )

        self.assertLessEqual(
            sum(report.raw_generated for report in result.reports), 48
        )
        self.assertLessEqual(result.legal_candidates, 12)
        self.assertLessEqual(len(result.callback_candidates), 8)
        self.assertIn(Template.BASE, templates)
        self.assertIn(Template.SINGLE_CORE_SPLIT_K, templates)
        self.assertIn(Template.DETERMINISTIC_SPLIT_K, templates)


if __name__ == "__main__":
    unittest.main()
