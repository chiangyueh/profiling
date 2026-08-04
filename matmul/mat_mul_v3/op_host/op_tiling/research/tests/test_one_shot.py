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
from tiling_search.one_shot import (
    BankRelativeEffectModel,
    BankRelativePrediction,
    BankRelativeSafetyPrediction,
    select_calibration_candidates,
    select_one_shot_candidate,
    validate_bank_relative_selector,
)


class _RuntimeModel:
    def __init__(self, risky_signature=None) -> None:
        self.risky_signature = risky_signature

    def predict(self, workload, vector, **_):
        del workload
        risky = (
            self.risky_signature is not None
            and tuple(vector.metrics.get("_signature", ()))
            == self.risky_signature
        )
        return FeedbackPrediction(
            latency_ratio=1.0,
            latency_uncertainty=1.0,
            latency_support=0.0,
            runtime_risk_score=0.90 if risky else 0.05,
            runtime_risk_support=1.0,
        )


class _AlwaysRiskyModel:
    def predict(self, workload, vector, **_):
        del workload, vector
        return FeedbackPrediction(1.0, 1.0, 0.0, 0.90, 1.0)


class _UnsafeRelativeSafetyModel:
    def predict(self, workload, bank, candidate, hardware, **_):
        del workload, bank, candidate, hardware
        return BankRelativeSafetyPrediction(
            samples=4,
            rejected=4,
            risk=1.0,
            nearest_distance=0.1,
            support=1.0,
        )


class _ExpectedRatioModel:
    def predict(self, workload, bank, candidate, hardware, **_):
        del workload, bank, hardware
        if candidate["tilingEnable"] == 3:
            return BankRelativePrediction(
                samples=8,
                robust_ratio=1.80,
                upper_ratio=2.00,
                nearest_distance=0.20,
                support=0.80,
            )
        return BankRelativePrediction(
            samples=4,
            robust_ratio=1.01,
            upper_ratio=3.00,
            nearest_distance=0.50,
            support=0.50,
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
            l2_bytes_per_cycle_per_core=110.0,
            hbm_bytes_per_cycle_per_core=32.0,
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
        self.bank = Schedule.from_signature(
            "20:208:128:16384:208:128:64:16:8:1:1:1:"
            "8:8:2:2:1:5:1:4:1:0:0"
        )
        self.incumbent = Candidate(
            schedule=self.bank,
            template=Template.BASE,
            source="bank_incumbent",
            rationale="bank",
        )
        self.custom = Candidate(
            schedule=self.bank.replace(iterateOrder=0),
            template=Template.BASE,
            source="local_bank_anchor",
            rationale="one-field bank mutation",
        )

    def _paired_effects(
        self,
        candidate: Candidate,
        *,
        ratio: float,
        count: int,
        bank: Schedule | None = None,
    ) -> list[MeasuredObservation]:
        evidence = []
        source_bank = bank or self.bank
        for index in range(count):
            evidence_workload = Workload(
                workload_id=f"paired_{index}",
                m=3968 + index * 16,
                n=128,
                k=16384,
                dtype="fp16",
                trans_a=False,
                trans_b=False,
                max_cores=20,
            )
            evidence.append(
                MeasuredObservation(
                    workload=evidence_workload,
                    schedule=candidate.schedule,
                    ratio_vs_official=ratio,
                    ratio_vs_bank=ratio,
                    source="calibration_local_counterfactual",
                    record_id=f"paired_{index}",
                    status_vs_official=(
                        "improved" if ratio < 0.99 else "regressed"
                    ),
                    status_vs_bank=(
                        "improved" if ratio < 0.99 else "regressed"
                    ),
                    verified=True,
                    structured_verified=True,
                    bank_schedule=source_bank,
                )
            )
        return evidence

    def test_no_paired_effect_evidence_measures_custom_but_deploys_bank(
        self,
    ) -> None:
        decision = select_one_shot_candidate(
            self.workload,
            [self.custom],
            self.incumbent,
            (),
            self.hardware,
            cost_model=_RuntimeModel(),
        )
        self.assertEqual(decision.candidate.schedule, self.custom.schedule)
        self.assertEqual(
            decision.candidate.source, "one_shot_research_candidate"
        )
        self.assertEqual(
            decision.deployment_candidate.schedule, self.bank
        )
        self.assertEqual(
            decision.selection_policy,
            "research_measurement_bank_deployment",
        )

    def test_broad_candidates_cannot_force_a_deployment(self) -> None:
        broad = Candidate(
            schedule=self.custom.schedule,
            template=Template.BASE,
            source="contract_global",
            rationale="research candidate",
        )
        decision = select_one_shot_candidate(
            self.workload,
            [broad],
            self.incumbent,
            (),
            self.hardware,
            cost_model=_RuntimeModel(),
        )
        self.assertEqual(decision.candidate.schedule, self.bank)
        self.assertEqual(decision.deployment_candidate.schedule, self.bank)
        self.assertEqual(decision.evaluated, 0)

    def test_repeated_bank_relative_effect_selects_custom(self) -> None:
        observations = self._paired_effects(
            self.custom, ratio=0.82, count=4
        )
        decision = select_one_shot_candidate(
            self.workload,
            [self.custom],
            self.incumbent,
            observations,
            self.hardware,
            cost_model=_RuntimeModel(),
        )
        self.assertEqual(decision.candidate.schedule, self.custom.schedule)
        self.assertEqual(
            decision.candidate.source, "one_shot_bank_relative"
        )
        self.assertEqual(
            decision.selection_policy, "paired_control_relative_effect"
        )
        self.assertEqual(
            decision.deployment_candidate.schedule, self.custom.schedule
        )
        self.assertGreaterEqual(
            decision.candidate.metrics["bank_relative_samples"], 3
        )

    def test_runtime_risk_overrides_positive_latency_effect(self) -> None:
        observations = self._paired_effects(
            self.custom, ratio=0.82, count=4
        )
        decision = select_one_shot_candidate(
            self.workload,
            [self.custom],
            self.incumbent,
            observations,
            self.hardware,
            cost_model=_AlwaysRiskyModel(),
            safety_model=_UnsafeRelativeSafetyModel(),
        )
        self.assertEqual(decision.candidate.schedule, self.custom.schedule)
        self.assertEqual(decision.deployment_candidate.schedule, self.bank)

    def test_cross_template_requires_stronger_independent_support(self) -> None:
        split = Candidate(
            schedule=self.bank.replace(tilingEnable=3),
            template=Template.DETERMINISTIC_SPLIT_K,
            source="contract_coupled_policy",
            rationale="cross-template candidate",
        )
        weak = self._paired_effects(split, ratio=0.96, count=3)
        weak_decision = select_one_shot_candidate(
            self.workload,
            [split],
            self.incumbent,
            weak,
            self.hardware,
            cost_model=_RuntimeModel(),
        )
        self.assertEqual(weak_decision.candidate.schedule, split.schedule)
        self.assertEqual(
            weak_decision.deployment_candidate.schedule, self.bank
        )

        strong = self._paired_effects(split, ratio=0.80, count=3)
        strong_decision = select_one_shot_candidate(
            self.workload,
            [split],
            self.incumbent,
            strong,
            self.hardware,
            cost_model=_RuntimeModel(),
        )
        self.assertEqual(strong_decision.candidate.schedule, split.schedule)
        self.assertEqual(
            strong_decision.deployment_candidate.schedule, split.schedule
        )

    def test_research_measurement_uses_expected_latency_not_tighter_regression(
        self,
    ) -> None:
        split = Candidate(
            schedule=self.bank.replace(tilingEnable=3),
            template=Template.DETERMINISTIC_SPLIT_K,
            source="contract_coupled_policy",
            rationale="known slow cross-template candidate",
        )
        decision = select_one_shot_candidate(
            self.workload,
            [self.custom, split],
            self.incumbent,
            (),
            self.hardware,
            cost_model=_RuntimeModel(),
            effect_model=_ExpectedRatioModel(),
        )
        self.assertEqual(decision.candidate.schedule, self.custom.schedule)
        self.assertEqual(
            decision.candidate.source, "one_shot_research_candidate"
        )
        self.assertEqual(decision.deployment_candidate.schedule, self.bank)

    def test_calibration_separates_local_coupled_and_template_probes(
        self,
    ) -> None:
        local = self.custom
        coupled = Candidate(
            schedule=self.bank.replace(l2IterateOrder=1),
            template=Template.BASE,
            source="contract_coupled_policy",
            rationale="coupled",
        )
        split = Candidate(
            schedule=self.bank.replace(tilingEnable=3),
            template=Template.DETERMINISTIC_SPLIT_K,
            source="contract_coupled_policy",
            rationale="split",
        )
        selected = select_calibration_candidates(
            self.workload,
            [local, coupled, split],
            self.incumbent,
            (),
            self.hardware,
            budget=3,
        )
        self.assertEqual(len(selected), 3)
        self.assertEqual(
            {candidate.template for candidate in selected},
            {Template.BASE, Template.DETERMINISTIC_SPLIT_K},
        )
        self.assertEqual(
            {candidate.source for candidate in selected},
            {
                "calibration_local_counterfactual",
                "calibration_coupled_counterfactual",
                "calibration_template_probe",
            },
        )

    def test_negative_template_evidence_reduces_probe_budget(self) -> None:
        same_template = [
            Candidate(
                schedule=self.bank.replace(
                    l2MTileCnt=5 + index,
                    l2MTileBlock=4 + index,
                ),
                template=Template.BASE,
                source=(
                    "local_bank_anchor"
                    if index % 2 == 0
                    else "contract_coupled_policy"
                ),
                rationale="same-template counterfactual",
            )
            for index in range(12)
        ]
        split_candidates = [
            Candidate(
                schedule=self.bank.replace(
                    tilingEnable=3,
                    l2MTileCnt=1 + index,
                ),
                template=Template.DETERMINISTIC_SPLIT_K,
                source="contract_coupled_policy",
                rationale="split probe",
            )
            for index in range(4)
        ]
        negative = self._paired_effects(
            split_candidates[0],
            ratio=2.0,
            count=6,
        )
        selected = select_calibration_candidates(
            self.workload,
            [*same_template, *split_candidates],
            self.incumbent,
            negative,
            self.hardware,
            budget=12,
        )
        self.assertEqual(len(selected), 12)
        self.assertEqual(
            sum(
                candidate.template == Template.DETERMINISTIC_SPLIT_K
                for candidate in selected
            ),
            1,
        )

    def test_leave_workload_out_validation_includes_bank_fallback(self) -> None:
        observations = self._paired_effects(
            self.custom, ratio=0.82, count=4
        )
        validation = validate_bank_relative_selector(
            observations, self.hardware
        )
        self.assertEqual(validation.groups, 4)
        self.assertGreaterEqual(validation.custom_winners, 1)
        self.assertEqual(validation.severe_regressions, 0)

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
