from __future__ import annotations

import sys
import unittest
from pathlib import Path


RESEARCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH))

from tiling_search.behavior import FeedbackPrediction
from tiling_search.domain import (
    Candidate,
    Hardware,
    MeasuredObservation,
    Schedule,
    Template,
    Workload,
)
from tiling_search.families import classify_workload
from tiling_search.one_shot import select_one_shot_candidate


class _PredictionModel:
    def __init__(self, latency_ratio: float = 0.85) -> None:
        self.calls = []
        self.latency_ratio = latency_ratio

    def predict(
        self,
        workload,
        vector,
        *,
        exclude_workload=None,
        cross_workload_latency_weight=0.15,
        **_,
    ):
        self.calls.append(
            (exclude_workload, cross_workload_latency_weight)
        )
        active_cores = vector.metrics["active_cores"]
        if active_cores == 19:
            return FeedbackPrediction(0.60, 0.20, 0.50, 0.90, 1.0)
        return FeedbackPrediction(
            self.latency_ratio, 0.20, 0.50, 0.10, 1.0
        )


class OneShotSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.hardware = Hardware(
            aic_cores=20,
            l0a_bytes=65536,
            l0b_bytes=65536,
            l0c_bytes=131072,
            l1_bytes=524032,
            l2_bytes=201326592,
            l2_bytes_per_cycle_per_core=64.0,
            hbm_bytes_per_cycle_per_core=16.0,
        )
        self.workload = Workload(
            workload_id="arbitrary_unseen_name",
            m=4096,
            n=128,
            k=16384,
            dtype="fp16",
            trans_a=False,
            trans_b=False,
            max_cores=20,
        )
        self.schedule = Schedule.from_signature(
            "20:208:128:16384:208:128:64:16:8:1:1:0:"
            "8:8:2:2:1:5:1:4:1:0:0"
        )
        self.incumbent = Candidate(
            schedule=self.schedule.replace(iterateOrder=1),
            template=Template.BASE,
            source="bank_incumbent",
            rationale="bank",
        )

    def test_one_shot_retains_incumbent_without_replacement_evidence(
        self,
    ) -> None:
        safe = Candidate(
            schedule=self.schedule,
            template=Template.BASE,
            source="contract_global",
            rationale="independent",
        )
        risky = Candidate(
            schedule=self.schedule.replace(usedCoreNum=19),
            template=Template.BASE,
            source="contract_global",
            rationale="independent",
        )
        model = _PredictionModel()
        decision = select_one_shot_candidate(
            self.workload,
            [risky, safe],
            self.incumbent,
            (),
            self.hardware,
            cost_model=model,
        )
        self.assertEqual(
            decision.candidate.schedule, self.incumbent.schedule
        )
        self.assertEqual(
            decision.candidate.source, "one_shot_bank_fallback"
        )
        self.assertTrue(decision.incumbent_fallback)
        self.assertEqual(decision.evaluated, 2)
        self.assertEqual(decision.direct_base_candidates, 2)
        self.assertEqual(decision.transfer_eligible_candidates, 0)
        self.assertEqual(decision.custom_eligible_candidates, 0)
        self.assertTrue(
            all(
                call
                == (self.workload.identity(), 0.15)
                for call in model.calls
            )
        )

    def test_one_shot_selects_custom_only_with_cross_workload_evidence(
        self,
    ) -> None:
        global_candidate = Candidate(
            schedule=self.schedule,
            template=Template.BASE,
            source="contract_global",
            rationale="independent",
        )
        local_candidate = Candidate(
            schedule=self.schedule,
            template=Template.BASE,
            source="local_bank_anchor",
            rationale="bank mutation",
        )
        observations = []
        for index in range(8):
            evidence_workload = Workload(
                workload_id=f"evidence_{index}",
                m=3968 + index * 16,
                n=128,
                k=16384,
                dtype="fp16",
                trans_a=False,
                trans_b=False,
                max_cores=20,
            )
            observations.append(
                MeasuredObservation(
                    workload=evidence_workload,
                    schedule=self.schedule,
                    ratio_vs_official=0.82,
                    ratio_vs_bank=0.84,
                    source="paired_evidence",
                    record_id=str(index),
                    status_vs_official="improved",
                    status_vs_bank="improved",
                    verified=True,
                    structured_verified=True,
                )
            )
        decision = select_one_shot_candidate(
            self.workload,
            [local_candidate, global_candidate],
            self.incumbent,
            observations,
            self.hardware,
            cost_model=_PredictionModel(latency_ratio=0.90),
        )
        self.assertEqual(decision.candidate.schedule, self.schedule)
        self.assertEqual(decision.candidate.source, "one_shot_model")
        self.assertFalse(decision.incumbent_fallback)
        self.assertEqual(decision.custom_eligible_candidates, 1)
        self.assertGreaterEqual(
            decision.transfer_eligible_candidates, 1
        )

    def test_one_shot_rejects_non_candidate_sources(self) -> None:
        global_candidate = Candidate(
            schedule=self.schedule,
            template=Template.BASE,
            source="contract_global",
            rationale="independent",
        )
        feedback_candidate = Candidate(
            schedule=self.schedule.replace(usedCoreNum=18),
            template=Template.BASE,
            source="feedback_winner_mutation",
            rationale="target feedback",
        )
        decision = select_one_shot_candidate(
            self.workload,
            [feedback_candidate, global_candidate],
            self.incumbent,
            (),
            self.hardware,
            cost_model=_PredictionModel(),
        )
        self.assertEqual(decision.evaluated, 1)
        self.assertTrue(decision.incumbent_fallback)

    def test_shape_strata_are_name_independent_and_orthogonal(self) -> None:
        first = classify_workload(self.workload, 20)
        renamed = classify_workload(
            Workload(
                workload_id="skinny_name_does_not_matter",
                m=self.workload.m,
                n=self.workload.n,
                k=self.workload.k,
                dtype=self.workload.dtype,
                trans_a=self.workload.trans_a,
                trans_b=self.workload.trans_b,
                max_cores=self.workload.max_cores,
            ),
            20,
        )
        self.assertEqual(first, renamed)
        self.assertEqual(first.geometry, "skinny_n")
        self.assertEqual(first.reduction, "deep_k")
        self.assertEqual(first.layout, "NN")


if __name__ == "__main__":
    unittest.main()
