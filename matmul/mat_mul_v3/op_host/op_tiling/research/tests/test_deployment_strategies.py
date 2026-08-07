from __future__ import annotations

import math
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


RESEARCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH))

from tiling_search import (
    CandidateEngine,
    DIRECT_BASE_AUDIT_BANK_WORKLOADS,
    DIRECT_BASE_AUDIT_PAIRED_RECORDS,
    DIRECT_BASE_AUDIT_UNIQUE_RECORDS,
    DIRECT_BASE_AUDIT_WINNER_WORKLOADS,
    DIRECT_BASE_AUDIT_WORKLOADS,
    DIRECT_BASE_L2_RESIDENT_RATIO,
    GenerationBudget,
    Hardware,
    SearchConfig,
    Template,
    Workload,
    direct_base_candidate,
    direct_rule_candidate,
)
from generate import load_resume_feedback
from tiling_search.contracts import ceil_div, template_of, validate_schedule
from tiling_search.domain import INPUT_BYTES, OUTPUT_BYTES


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

    def test_offline_audit_supports_fixed_direct_rules(
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

        paired = [
            observation
            for observation in observations
            if observation.verified
            and observation.source
            not in {"runtime_rejected", "runtime_verified"}
            and template_of(observation.schedule) == Template.BASE
        ]
        grouped = defaultdict(list)
        for observation in paired:
            grouped[
                (
                    observation.workload.identity(),
                    observation.schedule.signature(),
                )
            ].append(observation.measured_ratio)
        best_by_workload = {}
        for key, ratios in grouped.items():
            workload_identity = key[0]
            best_by_workload[workload_identity] = min(
                median(ratios),
                best_by_workload.get(
                    workload_identity, float("inf")
                ),
            )

        self.assertEqual(
            len(paired), DIRECT_BASE_AUDIT_PAIRED_RECORDS
        )
        self.assertEqual(
            len(grouped), DIRECT_BASE_AUDIT_UNIQUE_RECORDS
        )
        self.assertEqual(
            len(best_by_workload), DIRECT_BASE_AUDIT_WORKLOADS
        )
        self.assertEqual(
            sum(ratio < 0.99 for ratio in best_by_workload.values()),
            DIRECT_BASE_AUDIT_WINNER_WORKLOADS,
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
        self.assertEqual(
            len(banks), DIRECT_BASE_AUDIT_BANK_WORKLOADS
        )

        whole_sizes = []
        partitioned_sizes = []
        for workload, bank in banks.values():
            m_tasks = ceil_div(
                workload.m, bank["singleCoreM"]
            )
            n_tasks = ceil_div(
                workload.n, bank["singleCoreN"]
            )
            total_bytes = (
                workload.m
                * workload.k
                * INPUT_BYTES[workload.dtype]
                + workload.k
                * workload.n
                * INPUT_BYTES[workload.dtype]
                + workload.m
                * workload.n
                * OUTPUT_BYTES[workload.dtype]
            )
            if (
                bank["l2MTileCnt"],
                bank["l2NTileCnt"],
                bank["l2MTileBlock"],
                bank["l2NTileBlock"],
            ) == (1, 1, m_tasks, n_tasks):
                whole_sizes.append(total_bytes)
            else:
                partitioned_sizes.append(total_bytes)

            candidate = direct_base_candidate(
                workload,
                self.hardware,
            )
            self.assertEqual(
                candidate.schedule.signature(),
                bank.signature(),
                workload.workload_id,
            )

        self.assertEqual(len(whole_sizes), 36)
        self.assertEqual(len(partitioned_sizes), 22)
        threshold = math.sqrt(
            max(whole_sizes) * min(partitioned_sizes)
        )
        self.assertAlmostEqual(
            threshold / self.hardware.l2_bytes,
            DIRECT_BASE_L2_RESIDENT_RATIO,
            places=12,
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

    def test_direct_rule_keeps_dense_output_on_base(self) -> None:
        workload = Workload(
            workload_id="dense_deep_k",
            m=2368,
            n=2880,
            k=31744,
            dtype="fp16",
            trans_a=False,
            trans_b=False,
            max_cores=20,
        )
        candidate = direct_rule_candidate(workload, self.hardware)

        self.assertEqual(candidate.source, "direct_rule_policy")
        self.assertEqual(candidate.template, Template.BASE)
        self.assertEqual(candidate.metrics["model_enabled"], 0.0)
        self.assertEqual(candidate.metrics["candidate_pool_size"], 1.0)

    def test_direct_rule_uses_split_k_for_net_log32_counterexample(
        self,
    ) -> None:
        workload = Workload(
            workload_id="oneshot_v5_08",
            m=512,
            n=320,
            k=30720,
            dtype="fp16",
            trans_a=False,
            trans_b=False,
            max_cores=20,
        )
        candidate = direct_rule_candidate(workload, self.hardware)

        self.assertEqual(candidate.source, "direct_rule_policy")
        self.assertEqual(
            candidate.template, Template.DETERMINISTIC_SPLIT_K
        )
        self.assertEqual(
            candidate.schedule.signature_text(),
            "20:384:320:384:128:128:128:9:6:3:1:1:"
            "3:3:2:2:2:1:1:2:1:0:3",
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
