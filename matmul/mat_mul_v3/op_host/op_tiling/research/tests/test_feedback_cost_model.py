from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


RESEARCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH))

from generate import load_resume_feedback, load_workloads
from tiling_search.behavior import FeedbackCostModel, behavior_vector
from tiling_search.domain import (
    Hardware,
    MeasuredObservation,
    Schedule,
    Workload,
)


class FeedbackCostModelTest(unittest.TestCase):
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
            workload_id="feedback_probe",
            m=128,
            n=128,
            k=16384,
            dtype="fp16",
            trans_a=False,
            trans_b=False,
            max_cores=20,
        )
        self.success_schedule = Schedule.from_signature(
            "20:128:128:16384:128:128:128:9:6:1:1:0:"
            "1:1:2:2:2:1:1:8:1:0:0"
        )
        self.reject_schedule = Schedule.from_signature(
            "20:128:128:16384:64:128:128:9:6:1:1:0:"
            "1:1:2:2:2:1:1:8:1:0:0"
        )

    def test_runtime_rejection_does_not_become_latency_sample(self) -> None:
        observations = [
            MeasuredObservation(
                workload=self.workload,
                schedule=self.success_schedule,
                ratio_vs_official=1.2,
                ratio_vs_bank=1.1,
                source="contract_global",
                record_id="success",
                status_vs_official="regressed",
                status_vs_bank="regressed",
            ),
            MeasuredObservation(
                workload=self.workload,
                schedule=self.reject_schedule,
                ratio_vs_official=1.0,
                ratio_vs_bank=1.0,
                source="runtime_rejected",
                record_id="reject",
                status_vs_official="runtime_rejected",
                status_vs_bank="runtime_rejected",
            ),
        ]
        model = FeedbackCostModel(observations, self.hardware)
        prediction = model.predict(
            self.workload,
            behavior_vector(
                self.workload,
                self.reject_schedule,
                self.hardware,
            ),
        )
        self.assertAlmostEqual(prediction.latency_ratio, 1.2)
        self.assertGreater(prediction.runtime_risk_score, 0.5)

    def test_workload_core_limit_is_normalized_to_hardware(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workloads.csv"
            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(
                    output,
                    fieldnames=(
                        "id",
                        "m",
                        "n",
                        "k",
                        "dtype",
                        "trans_a",
                        "trans_b",
                        "max_cores",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "id": "core_probe",
                        "m": 128,
                        "n": 128,
                        "k": 128,
                        "dtype": "fp16",
                        "trans_a": 0,
                        "trans_b": 0,
                        "max_cores": 24,
                    }
                )
            workload = load_workloads(path, aic_cores=20)[0]
        self.assertEqual(workload.max_cores, 20)

    def test_resume_runtime_rejection_becomes_risk_evidence(self) -> None:
        row = {
            "candidate_role": "searched",
            "soc": "Ascend910B3",
            "aic": "20",
            "workload_id": self.workload.workload_id,
            "m": str(self.workload.m),
            "n": str(self.workload.n),
            "k": str(self.workload.k),
            "dtype": self.workload.dtype,
            "trans_a": "0",
            "trans_b": "0",
            "tiling_signature": ":".join(
                str(value) for value in self.reject_schedule.signature()
            ),
            "success": "0",
            "preflight_mode": "runtime_rejected",
            "record_id": "runtime-reject-probe",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.csv"
            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=tuple(row))
                writer.writeheader()
                writer.writerow(row)
            observations, exclusions = load_resume_feedback(
                path,
                soc="Ascend910B3",
                aic_cores=20,
            )
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].source, "runtime_rejected")
        self.assertEqual(len(exclusions), 1)

    def test_deterministic_split_k_models_workspace_reduction(self) -> None:
        workload = Workload(
            workload_id="deterministic_probe",
            m=1024,
            n=1536,
            k=32768,
            dtype="fp16",
            trans_a=False,
            trans_b=False,
            max_cores=20,
        )
        schedule = Schedule.from_signature(
            "20:1024:384:384:128:128:128:6:9:1:3:0:"
            "3:3:2:2:2:1:1:1:4:1:3"
        )
        metrics = behavior_vector(
            workload,
            schedule,
            self.hardware,
        ).metrics
        self.assertEqual(metrics["atomic_output_ratio"], 0.0)
        self.assertGreater(metrics["split_reduction_ratio"], 0.0)
        self.assertGreater(metrics["core_rounds"], 1.0)

    def test_cross_workload_latency_is_only_a_weak_prior(self) -> None:
        other = Workload(
            workload_id="other",
            m=4096,
            n=4096,
            k=4096,
            dtype="fp16",
            trans_a=False,
            trans_b=False,
            max_cores=20,
        )
        observations = [
            MeasuredObservation(
                workload=self.workload,
                schedule=self.success_schedule,
                ratio_vs_official=0.2,
                ratio_vs_bank=0.2,
                source="contract_global",
                record_id="cross-workload",
            )
        ]
        prediction = FeedbackCostModel(
            observations, self.hardware
        ).predict(
            other,
            behavior_vector(other, self.success_schedule, self.hardware),
        )
        self.assertLessEqual(prediction.latency_support, 0.15)
        self.assertGreaterEqual(prediction.latency_uncertainty, 0.80)
        self.assertGreater(prediction.latency_ratio, 0.70)


if __name__ == "__main__":
    unittest.main()
