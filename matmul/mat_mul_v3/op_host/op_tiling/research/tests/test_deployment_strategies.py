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
)
from tiling_search.contracts import validate_schedule


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

    def test_direct_base_is_one_analytical_legal_record(self) -> None:
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
            (7, 4, 4, 5),
        )

    def test_direct_base_covers_dtype_and_layout_variants(self) -> None:
        cases = (
            ("bf16_nn", 1791, 2433, 4609, "bf16", False, False),
            ("fp32_nt", 144, 768, 4864, "fp32", False, True),
            ("fp32_tn", 1152, 1664, 2304, "fp32", True, False),
            ("fp32_tt", 896, 1280, 1792, "fp32", True, True),
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
