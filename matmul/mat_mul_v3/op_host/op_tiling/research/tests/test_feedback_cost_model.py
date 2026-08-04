from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


RESEARCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH))

from generate import (
    load_resume_feedback,
    load_workloads,
    merge_candidate_rows,
)
from campaign_report import summarize_campaign
from tiling_search.behavior import (
    BehaviorVector,
    FeedbackCostModel,
    behavior_vector,
    select_behavior_coverage,
)
from tiling_search.feedback import feedback_mutations, feedback_targets
from tiling_search.domain import (
    Candidate,
    Hardware,
    MeasuredObservation,
    Schedule,
    Template,
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
            "toolkit": "8.1.RC1",
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
        incompatible, incompatible_exclusions = load_resume_feedback(
            path,
            soc="Ascend910B3",
            aic_cores=20,
            toolkit="8.5.0",
        )
        self.assertEqual(incompatible, [])
        self.assertEqual(incompatible_exclusions, set())

    def test_output_not_written_becomes_runtime_risk_evidence(
        self,
    ) -> None:
        row = {
            "candidate_role": "searched",
            "soc": "Ascend910B3",
            "aic": "20",
            "toolkit": "8.1.RC1",
            "workload_id": self.workload.workload_id,
            "m": str(self.workload.m),
            "n": str(self.workload.n),
            "k": str(self.workload.k),
            "dtype": self.workload.dtype,
            "trans_a": "0",
            "trans_b": "0",
            "tiling_signature": self.reject_schedule.signature_text(),
            "success": "0",
            "preflight_mode": "output_not_written",
            "record_id": "output-coverage-reject",
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
                toolkit="8.1.RC1",
            )
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].source, "runtime_rejected")
        self.assertEqual(len(exclusions), 1)

    def test_resume_success_requires_verified_measurement_protocol(
        self,
    ) -> None:
        row = {
            "candidate_role": "searched",
            "candidate_source": "contract_global",
            "soc": "Ascend910B3",
            "aic": "20",
            "toolkit": "8.1.RC1",
            "workload_id": self.workload.workload_id,
            "m": str(self.workload.m),
            "n": str(self.workload.n),
            "k": str(self.workload.k),
            "dtype": self.workload.dtype,
            "trans_a": "0",
            "trans_b": "0",
            "tiling_signature": self.success_schedule.signature_text(),
            "success": "1",
            "preflight_mode": "zero_coverage_grid9_v1",
            "pair_validated": "1",
            "median_ms": "0.8",
            "official_ms": "1",
            "bank_ms": "1",
            "status_vs_official": "improved",
            "status_vs_bank": "improved",
            "record_id": "resume-success",
            "run_id": "paired-run",
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
                toolkit="8.1.RC1",
            )
            self.assertEqual(observations, [])
            self.assertEqual(exclusions, set())

            row["preflight_mode"] = "numeric_ones_full_v2"
            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=tuple(row))
                writer.writeheader()
                writer.writerow(row)
            observations, exclusions = load_resume_feedback(
                path,
                soc="Ascend910B3",
                aic_cores=20,
                toolkit="8.1.RC1",
            )
        self.assertEqual(len(observations), 1)
        self.assertTrue(observations[0].verified)
        self.assertFalse(observations[0].structured_verified)
        self.assertEqual(
            observations[0].record_id,
            f"paired-run:{self.workload.workload_id}",
        )
        self.assertEqual(len(exclusions), 1)

        row["preflight_mode"] = "numeric_signed_axes_full_v3"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.csv"
            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=tuple(row))
                writer.writeheader()
                writer.writerow(row)
            observations, _ = load_resume_feedback(
                path,
                soc="Ascend910B3",
                aic_cores=20,
                toolkit="8.1.RC1",
            )
        self.assertTrue(observations[0].structured_verified)

        row["pair_validated"] = "0"
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
                toolkit="8.1.RC1",
            )
        self.assertEqual(observations, [])
        self.assertEqual(len(exclusions), 1)

    def test_structured_measurement_overrides_provisional_duplicate(
        self,
    ) -> None:
        provisional = MeasuredObservation(
            workload=self.workload,
            schedule=self.success_schedule,
            ratio_vs_official=0.2,
            ratio_vs_bank=0.2,
            source="contract_global",
            record_id="provisional",
        )
        structured = MeasuredObservation(
            workload=self.workload,
            schedule=self.success_schedule,
            ratio_vs_official=1.2,
            ratio_vs_bank=1.1,
            source="contract_global",
            record_id="structured",
            verified=True,
            structured_verified=True,
        )
        evidence = FeedbackCostModel(
            [provisional, structured], self.hardware
        ).evidence
        self.assertEqual(len(evidence), 1)
        self.assertAlmostEqual(evidence[0].latency_ratio or 0.0, 1.2)
        self.assertEqual(evidence[0].latency_reliability, 1.0)

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

    def test_same_workload_template_rejections_override_weak_neighbors(
        self,
    ) -> None:
        observations = []
        for index in range(8):
            observations.append(
                MeasuredObservation(
                    workload=self.workload,
                    schedule=self.reject_schedule.replace(
                        l2MTileBlock=8 + index,
                    ),
                    ratio_vs_official=1.0,
                    ratio_vs_bank=1.0,
                    source="runtime_rejected",
                    record_id=f"reject-{index}",
                    status_vs_official="runtime_rejected",
                    status_vs_bank="runtime_rejected",
                )
            )
        prediction = FeedbackCostModel(
            observations, self.hardware
        ).predict(
            self.workload,
            behavior_vector(
                self.workload,
                self.success_schedule,
                self.hardware,
            ),
        )
        self.assertGreater(prediction.runtime_risk_score, 0.75)
        self.assertEqual(prediction.runtime_risk_support, 1.0)

    def test_final_selection_does_not_fill_with_known_severe_losers(
        self,
    ) -> None:
        safe = []
        severe = []
        for index in range(9):
            schedule = self.success_schedule.replace(
                usedCoreNum=20 - index,
                l2MTileBlock=8 + index,
            )
            safe.append(
                Candidate(
                    schedule=schedule,
                    template=Template.BASE,
                    source="contract_global",
                    rationale="safe",
                )
            )
        for index in range(4):
            schedule = self.success_schedule.replace(
                tilingEnable=2,
                singleCoreK=128 * (index + 1),
                usedCoreNum=16 + index,
            )
            severe.append(
                Candidate(
                    schedule=schedule,
                    template=Template.SINGLE_CORE_SPLIT_K,
                    source="contract_global",
                    rationale="known loser",
                )
            )

        scored = []
        for candidate in [*safe, *severe]:
            is_severe = candidate in severe
            is_known_false_positive_range = candidate == safe[0]
            vector = BehaviorVector(
                categories=(
                    candidate.template.value,
                    candidate.schedule["tilingEnable"],
                    "fp16",
                    0,
                    0,
                ),
                values=(
                    candidate.schedule["usedCoreNum"] / 20.0,
                    *(0.0 for _ in range(14)),
                ),
                metrics={
                    "predicted_latency_ratio": (
                        9.0
                        if is_severe
                        else (2.7 if is_known_false_positive_range else 1.0)
                    ),
                    "latency_support": 0.8,
                    "latency_uncertainty": 0.1,
                    "runtime_risk_score": 0.1,
                    "runtime_risk_support": 0.8,
                    "analytical_prior": 1.0,
                },
            )
            scored.append((candidate, vector, 0.0, 0.1))

        with patch(
            "tiling_search.behavior.score_candidates",
            return_value=scored,
        ):
            selected = select_behavior_coverage(
                self.workload,
                [*safe, *severe],
                (),
                self.hardware,
                8,
                probe_templates=False,
            )
        self.assertEqual(len(selected), 8)
        self.assertEqual(
            {candidate.template for candidate in selected},
            {Template.BASE},
        )
        self.assertNotIn(safe[0], selected)

    def test_final_selection_fills_budget_when_only_risky_points_remain(
        self,
    ) -> None:
        candidates = [
            Candidate(
                schedule=self.success_schedule.replace(
                    usedCoreNum=20 - index,
                    l2MTileBlock=8 + index,
                ),
                template=Template.BASE,
                source="contract_global",
                rationale="risk-budget probe",
            )
            for index in range(6)
        ]
        scored = []
        for index, candidate in enumerate(candidates):
            vector = BehaviorVector(
                categories=(Template.BASE.value, 0, "fp16", 0, 0),
                values=(index / 6.0, *(0.0 for _ in range(14))),
                metrics={
                    "predicted_latency_ratio": 2.0,
                    "latency_support": 1.0,
                    "latency_uncertainty": 0.1,
                    "runtime_risk_score": 0.9,
                    "runtime_risk_support": 1.0,
                    "analytical_prior": 1.0,
                },
            )
            scored.append((candidate, vector, float(index), 0.1))
        with patch(
            "tiling_search.behavior.score_candidates",
            return_value=scored,
        ):
            selected = select_behavior_coverage(
                self.workload,
                candidates,
                (),
                self.hardware,
                6,
                probe_templates=False,
            )
        self.assertEqual(len(selected), 6)

    def test_runtime_rejection_is_not_used_as_mutation_centre(
        self,
    ) -> None:
        rejected = MeasuredObservation(
            workload=self.workload,
            schedule=self.success_schedule.replace(
                depthA1=2,
                depthB1=2,
            ),
            ratio_vs_official=1.0,
            ratio_vs_bank=1.0,
            source="runtime_rejected",
            record_id="runtime-reject",
            status_vs_official="runtime_rejected",
            status_vs_bank="runtime_rejected",
        )
        mutations = feedback_mutations(
            self.workload, self.hardware, [rejected]
        )
        self.assertEqual(mutations, [])

    def test_campaign_summary_retains_prior_winner(self) -> None:
        winner = MeasuredObservation(
            workload=self.workload,
            schedule=self.success_schedule,
            ratio_vs_official=0.8,
            ratio_vs_bank=0.9,
            source="contract_global",
            record_id="prior-winner",
            status_vs_official="improved",
            status_vs_bank="improved",
            verified=True,
            structured_verified=True,
        )
        later_regression = MeasuredObservation(
            workload=self.workload,
            schedule=self.success_schedule.replace(usedCoreNum=19),
            ratio_vs_official=1.5,
            ratio_vs_bank=1.4,
            source="feedback_winner_mutation",
            record_id="later-regression",
            status_vs_official="regressed",
            status_vs_bank="regressed",
        )
        rows = summarize_campaign(
            [self.workload],
            [winner, later_regression],
        )
        self.assertEqual(rows[0]["campaign_status"], "solved")
        self.assertEqual(rows[0]["winning_measurements"], "1")
        self.assertEqual(rows[0]["record_id"], "prior-winner")

    def test_campaign_does_not_solve_from_provisional_winner(self) -> None:
        winner = MeasuredObservation(
            workload=self.workload,
            schedule=self.success_schedule,
            ratio_vs_official=0.8,
            ratio_vs_bank=0.8,
            source="contract_global",
            record_id="coverage-only-winner",
            status_vs_official="improved",
            status_vs_bank="improved",
        )
        row = summarize_campaign([self.workload], [winner])[0]
        self.assertEqual(row["campaign_status"], "open")
        self.assertEqual(row["verified_measurements"], "0")
        self.assertEqual(row["provisional_measurements"], "1")

    def test_stage_merge_preserves_old_candidates_and_adds_only_new(
        self,
    ) -> None:
        control = {
            "workload_id": "w",
            "candidate_role": "bank_seed_control",
            "tiling_signature": "control",
        }
        first = {
            "workload_id": "w",
            "candidate_role": "searched",
            "tiling_signature": "first",
            "rank": "1",
        }
        duplicate = dict(first)
        duplicate["rank"] = "7"
        second = {
            "workload_id": "w",
            "candidate_role": "searched",
            "tiling_signature": "second",
            "rank": "1",
        }
        merged = merge_candidate_rows(
            [control, first],
            [control, duplicate, second],
        )
        searched = [
            row for row in merged if row["candidate_role"] == "searched"
        ]
        self.assertEqual(
            [row["tiling_signature"] for row in searched],
            ["first", "second"],
        )
        self.assertEqual([row["rank"] for row in searched], ["1", "2"])

    def test_feedback_targets_do_not_transfer_cross_workload_regressions(
        self,
    ) -> None:
        other = Workload(
            workload_id="other",
            m=256,
            n=256,
            k=16384,
            dtype="fp16",
            trans_a=False,
            trans_b=False,
            max_cores=20,
        )
        same_regression = MeasuredObservation(
            workload=self.workload,
            schedule=self.success_schedule,
            ratio_vs_official=2.0,
            ratio_vs_bank=2.0,
            source="contract_global",
            record_id="same-regression",
            status_vs_official="regressed",
            status_vs_bank="regressed",
        )
        cross_regression = MeasuredObservation(
            workload=other,
            schedule=self.success_schedule,
            ratio_vs_official=2.0,
            ratio_vs_bank=2.0,
            source="contract_global",
            record_id="cross-regression",
            status_vs_official="regressed",
            status_vs_bank="regressed",
        )
        cross_winner = MeasuredObservation(
            workload=other,
            schedule=self.success_schedule,
            ratio_vs_official=0.8,
            ratio_vs_bank=0.8,
            source="contract_global",
            record_id="cross-winner",
            status_vs_official="improved",
            status_vs_bank="improved",
            verified=True,
            structured_verified=True,
        )
        targets = feedback_targets(
            self.workload,
            self.hardware,
            [same_regression, cross_regression, cross_winner],
        )
        origins = [target.origin for target in targets]
        self.assertIn("counterfactual", origins)
        self.assertIn("transfer_winner", origins)
        self.assertEqual(origins.count("counterfactual"), 1)


if __name__ == "__main__":
    unittest.main()
