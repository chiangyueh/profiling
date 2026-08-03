from __future__ import annotations

import sys
import unittest
from pathlib import Path


RESEARCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH))

from tiling_search.domain import (
    Candidate,
    Hardware,
    MeasuredObservation,
    Schedule,
    Template,
    Workload,
)
from tiling_search.ranking import PairwiseLatencyRanker


class PairwiseLatencyRankerTest(unittest.TestCase):
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
        self.schedule = Schedule.from_signature(
            "20:128:256:4096:128:256:64:16:8:1:1:0:"
            "8:8:2:2:1:8:4:2:2:0:0"
        )

    def test_cross_workload_pairs_learn_consistent_ordering(self) -> None:
        observations = []
        for index in range(8):
            workload = Workload(
                workload_id=f"train_{index}",
                m=1024 + 128 * index,
                n=2048,
                k=4096,
                dtype="fp16",
                trans_a=False,
                trans_b=False,
                max_cores=20,
            )
            for iterate_order, ratio in ((0, 1.10), (1, 0.90)):
                observations.append(
                    MeasuredObservation(
                        workload=workload,
                        schedule=self.schedule.replace(
                            iterateOrder=iterate_order
                        ),
                        ratio_vs_official=ratio,
                        ratio_vs_bank=ratio,
                        source="contract_global",
                        record_id=f"campaign:{workload.workload_id}",
                        verified=True,
                        structured_verified=True,
                    )
                )
        target = Workload(
            workload_id="unseen",
            m=1792,
            n=2048,
            k=4096,
            dtype="fp16",
            trans_a=False,
            trans_b=False,
            max_cores=20,
        )
        incumbent = Candidate(
            schedule=self.schedule.replace(iterateOrder=0),
            template=Template.BASE,
            source="bank_incumbent",
            rationale="bank",
        )
        candidate = Candidate(
            schedule=self.schedule.replace(iterateOrder=1),
            template=Template.BASE,
            source="local_bank_anchor",
            rationale="mutation",
        )
        prediction = PairwiseLatencyRanker(
            observations, self.hardware
        ).compare(target, incumbent, candidate, self.hardware)
        self.assertLess(prediction.relative_ratio, 1.0)
        self.assertGreater(prediction.support, 0.0)

    def test_runtime_rejections_are_not_latency_training_rows(self) -> None:
        workload = Workload(
            workload_id="rejected",
            m=1024,
            n=2048,
            k=4096,
            dtype="fp16",
            trans_a=False,
            trans_b=False,
            max_cores=20,
        )
        observations = [
            MeasuredObservation(
                workload=workload,
                schedule=self.schedule.replace(iterateOrder=value),
                ratio_vs_official=1.0,
                ratio_vs_bank=1.0,
                source="runtime_rejected",
                record_id="reject",
            )
            for value in (0, 1)
        ]
        model = PairwiseLatencyRanker(observations, self.hardware)
        self.assertEqual(model.samples, 0)


if __name__ == "__main__":
    unittest.main()
