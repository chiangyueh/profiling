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
    Schedule,
    Template,
    Workload,
)
from tiling_search.families import classify_workload
from tiling_search.one_shot import select_one_shot_candidate


class _PredictionModel:
    def __init__(self) -> None:
        self.calls = []

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
        return FeedbackPrediction(0.85, 0.30, 0.50, 0.10, 1.0)


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

    def test_one_shot_uses_leave_target_out_and_rejects_high_risk(self) -> None:
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
            (),
            self.hardware,
            cost_model=model,
        )
        self.assertEqual(decision.candidate.schedule, safe.schedule)
        self.assertEqual(decision.candidate.source, "one_shot_model")
        self.assertEqual(decision.evaluated, 2)
        self.assertEqual(decision.safe_candidates, 1)
        self.assertEqual(decision.direct_base_candidates, 2)
        self.assertEqual(decision.transfer_eligible_candidates, 0)
        self.assertTrue(
            all(
                call
                == (self.workload.identity(), 0.15)
                for call in model.calls
            )
        )

    def test_one_shot_does_not_select_feedback_or_local_candidates(self) -> None:
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
            (),
            self.hardware,
            cost_model=_PredictionModel(),
        )
        self.assertEqual(decision.evaluated, 1)
        self.assertEqual(
            decision.candidate.schedule,
            global_candidate.schedule,
        )

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
