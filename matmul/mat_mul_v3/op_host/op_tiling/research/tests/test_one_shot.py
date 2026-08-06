from __future__ import annotations

import sys
import unittest
from pathlib import Path


RESEARCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH))

from tiling_search.behavior import FeedbackPrediction
from tiling_search.calibration_workloads import (
    generate_template_calibration_workloads,
)
from tiling_search.bank_structure import (
    bank_transition,
    schedules_execution_equivalent,
)
from tiling_search.contracts import (
    common_hardware_contract,
    template_kernel_contract,
)
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
    BankRelativeSafetyModel,
    BankRelativeSafetyPrediction,
    select_adaptive_calibration_candidates,
    select_calibration_candidates,
    select_one_shot_candidate,
    validate_bank_relative_selector,
)
from tiling_search.behavior import behavior_vector
from tiling_search.template_competition import compare_templates
from tiling_search.solvers import Al1FullLoadSolver


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


class _UnsupportedDistantRatioModel:
    def predict(self, workload, bank, candidate, hardware, **_):
        del workload, bank, hardware
        if candidate["baseM"] != 208:
            return BankRelativePrediction(
                samples=2,
                robust_ratio=0.80,
                upper_ratio=1.20,
                nearest_distance=1.0,
                support=0.10,
            )
        return BankRelativePrediction(
            samples=2,
            robust_ratio=1.00,
            upper_ratio=1.30,
            nearest_distance=0.5,
            support=0.10,
        )


class _FalseOptimisticCrossTemplateModel:
    def predict(self, workload, bank, candidate, hardware, **_):
        del workload, bank, candidate, hardware
        return BankRelativePrediction(
            samples=8,
            robust_ratio=0.29,
            upper_ratio=0.49,
            nearest_distance=0.20,
            support=0.63,
        )


class _AdaptiveEffectModel:
    def predict(self, workload, bank, candidate, hardware, **_):
        del workload, bank, candidate, hardware
        return BankRelativePrediction(
            samples=6,
            robust_ratio=0.97,
            upper_ratio=1.03,
            nearest_distance=0.25,
            support=0.70,
            behavior_samples=18,
            uncertainty=0.08,
        )


class _AdaptiveSafetyModel:
    def __init__(self, unsafe_signature=None) -> None:
        self.unsafe_signature = unsafe_signature

    def predict(self, workload, bank, candidate, hardware, **_):
        del workload, bank, hardware
        unsafe = candidate.signature() == self.unsafe_signature
        return BankRelativeSafetyPrediction(
            samples=6,
            rejected=6 if unsafe else 0,
            risk=0.90 if unsafe else 0.05,
            nearest_distance=0.20,
            support=0.80,
            behavior_samples=18,
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
            source="contract_coupled_policy",
            rationale="independently generated coupled schedule",
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

    def test_no_paired_effect_evidence_selects_research_challenger(
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
            "paired_feedback_active_challenger",
        )
        self.assertEqual(
            decision.candidate.metrics["deployment_recommended_custom"],
            0.0,
        )
        self.assertEqual(decision.custom_eligible_candidates, 1)

    def test_independent_contract_candidate_can_be_measured(
        self,
    ) -> None:
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
        self.assertEqual(decision.candidate.schedule, broad.schedule)
        self.assertEqual(
            decision.deployment_candidate.schedule, self.bank
        )

    def test_feedback_transfer_is_preferred_for_active_measurement(
        self,
    ) -> None:
        transferred = Candidate(
            schedule=self.custom.schedule,
            template=Template.BASE,
            source="feedback_winner_transfer",
            rationale="projected paired winner transition",
        )
        generic = Candidate(
            schedule=self.bank.replace(l2IterateOrder=1),
            template=Template.BASE,
            source="contract_global",
            rationale="generic independent candidate",
        )
        decision = select_one_shot_candidate(
            self.workload,
            [generic, transferred],
            self.incumbent,
            (),
            self.hardware,
            cost_model=_RuntimeModel(),
        )
        self.assertEqual(decision.candidate.schedule, transferred.schedule)
        self.assertEqual(
            decision.generator_source, "feedback_winner_transfer"
        )
        self.assertEqual(
            decision.candidate.source, "one_shot_research_candidate"
        )
        self.assertEqual(
            decision.selection_policy,
            "paired_feedback_active_challenger",
        )

    def test_independent_bank_reconstruction_is_not_discarded(
        self,
    ) -> None:
        reconstructed = Candidate(
            schedule=self.bank,
            template=Template.BASE,
            source="contract_coupled_policy",
            rationale="independently reconstructed schedule",
        )
        decision = select_one_shot_candidate(
            self.workload,
            [reconstructed],
            self.incumbent,
            (),
            self.hardware,
            cost_model=_RuntimeModel(),
        )
        self.assertEqual(decision.candidate.schedule, self.bank)
        self.assertEqual(
            decision.selection_policy,
            "independent_bank_reconstruction",
        )
        self.assertEqual(decision.custom_eligible_candidates, 0)
        self.assertEqual(decision.bank_equivalent_candidates, 1)
        self.assertEqual(
            decision.candidate.metrics["bank_execution_equivalent"], 1.0
        )
        self.assertEqual(
            decision.candidate.metrics["bank_signature_exact"], 1.0
        )
        self.assertEqual(
            decision.candidate.metrics["one_shot_incumbent_fallback"], 0.0
        )

    def test_bank_transition_groups_23_fields_by_execution_subsystem(
        self,
    ) -> None:
        l2 = bank_transition(
            self.bank,
            self.bank.replace(
                l2MTileCnt=2,
                l2MTileBlock=2,
                l2IterateOrder=1,
            ),
        )
        self.assertEqual(l2.changed_subsystems, frozenset({"l2"}))
        self.assertTrue(l2.preserves_execution_structure)
        self.assertEqual(l2.risk_tier, 0)

        broad = bank_transition(
            self.bank,
            self.bank.replace(
                baseM=128,
                depthA1=8,
                tilingEnable=3,
            ),
        )
        self.assertEqual(
            broad.changed_subsystems,
            frozenset({"template", "l0", "l1"}),
        )
        self.assertFalse(broad.preserves_execution_structure)
        self.assertEqual(broad.risk_tier, 3)

    def test_split_k_ignores_l2_fields_not_read_by_kernel(self) -> None:
        split = self.bank.replace(tilingEnable=2)
        same_execution = split.replace(
            l2MTileCnt=7,
            l2NTileCnt=9,
            l2MTileBlock=11,
            l2NTileBlock=13,
        )
        self.assertTrue(
            schedules_execution_equivalent(split, same_execution)
        )
        self.assertEqual(
            bank_transition(split, same_execution).changed_fields,
            frozenset(),
        )
        changed_order = same_execution.replace(l2IterateOrder=1)
        self.assertFalse(
            schedules_execution_equivalent(split, changed_order)
        )

    def test_unsupported_distant_prediction_does_not_beat_bank_structure(
        self,
    ) -> None:
        distant = Candidate(
            schedule=self.bank.replace(
                singleCoreM=128,
                baseM=128,
                depthA1=8,
                stepKa=4,
            ),
            template=Template.BASE,
            source="contract_global",
            rationale="unsupported multi-subsystem change",
        )
        decision = select_one_shot_candidate(
            self.workload,
            [distant, self.custom],
            self.incumbent,
            (),
            self.hardware,
            cost_model=_RuntimeModel(),
            effect_model=_UnsupportedDistantRatioModel(),
        )
        self.assertEqual(decision.candidate.schedule, self.custom.schedule)
        self.assertEqual(
            decision.candidate.metrics["bank_structure_preserved"], 1.0
        )
        self.assertEqual(
            decision.candidate.metrics["bank_transition_risk_tier"], 0.0
        )

    def test_empty_custom_pool_fails_instead_of_using_bank(self) -> None:
        unsupported = Candidate(
            schedule=self.custom.schedule,
            template=Template.BASE,
            source="unsupported_test_source",
            rationale="not part of the one-shot candidate layer",
        )
        with self.assertRaisesRegex(
            ValueError, "no independently generated candidate"
        ):
            select_one_shot_candidate(
                self.workload,
                [unsupported],
                self.incumbent,
                (),
                self.hardware,
                cost_model=_RuntimeModel(),
            )

    def test_bank_anchor_mutation_is_not_a_one_shot_deployment(
        self,
    ) -> None:
        local = Candidate(
            schedule=self.custom.schedule,
            template=Template.BASE,
            source="local_bank_anchor",
            rationale="calibration-only bank mutation",
        )
        with self.assertRaisesRegex(
            ValueError, "no independently generated candidate"
        ):
            select_one_shot_candidate(
                self.workload,
                [local],
                self.incumbent,
                (),
                self.hardware,
                cost_model=_RuntimeModel(),
            )

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

    def test_relative_models_pool_neighbor_layouts_but_not_templates(
        self,
    ) -> None:
        evidence = []
        layouts = ((False, True), (True, False), (True, True))
        for index, (trans_a, trans_b) in enumerate(layouts):
            workload = Workload(
                workload_id=f"neighbor_layout_{index}",
                m=3968 + index * 16,
                n=128,
                k=16384,
                dtype="fp16",
                trans_a=trans_a,
                trans_b=trans_b,
                max_cores=20,
            )
            evidence.append(
                MeasuredObservation(
                    workload=workload,
                    schedule=self.custom.schedule,
                    ratio_vs_official=0.80,
                    ratio_vs_bank=0.80,
                    source="calibration_coupled_counterfactual",
                    record_id=f"neighbor_layout_{index}",
                    verified=True,
                    structured_verified=True,
                    bank_schedule=self.bank,
                )
            )
        effect = BankRelativeEffectModel(evidence, self.hardware).predict(
            self.workload,
            self.bank,
            self.custom.schedule,
            self.hardware,
        )
        self.assertEqual(effect.samples, 3)
        self.assertGreaterEqual(effect.support, 0.20)
        self.assertAlmostEqual(effect.robust_ratio, 0.80)

        rejected = [
            MeasuredObservation(
                workload=observation.workload,
                schedule=observation.schedule,
                ratio_vs_official=1.0,
                ratio_vs_bank=1.0,
                source="runtime_rejected",
                record_id=observation.record_id,
                bank_schedule=observation.bank_schedule,
            )
            for observation in evidence
        ]
        safety = BankRelativeSafetyModel(
            rejected, self.hardware
        ).predict(
            self.workload,
            self.bank,
            self.custom.schedule,
            self.hardware,
        )
        self.assertEqual(safety.samples, 3)
        self.assertEqual(safety.rejected, 3)
        self.assertGreaterEqual(safety.support, 0.20)

        split = self.custom.schedule.replace(tilingEnable=3)
        isolated = BankRelativeEffectModel(
            [
                MeasuredObservation(
                    workload=observation.workload,
                    schedule=split,
                    ratio_vs_official=0.50,
                    ratio_vs_bank=0.50,
                    source="calibration_template_probe",
                    record_id=observation.record_id,
                    verified=True,
                    structured_verified=True,
                    bank_schedule=self.bank,
                )
                for observation in evidence
            ],
            self.hardware,
        ).predict(
            self.workload,
            self.bank,
            self.custom.schedule,
            self.hardware,
        )
        self.assertEqual(isolated.samples, 0)

    def test_relative_models_use_multiple_bins_without_double_counting_workloads(
        self,
    ) -> None:
        evidence = []
        for workload_index in range(2):
            workload = Workload(
                workload_id=f"hierarchical_{workload_index}",
                m=3968 + workload_index * 64,
                n=128,
                k=16384,
                dtype="fp16",
                trans_a=False,
                trans_b=False,
                max_cores=20,
            )
            for variant in range(3):
                schedule = self.custom.schedule.replace(
                    l2MTileCnt=5 + variant,
                    l2MTileBlock=3 + variant,
                )
                evidence.append(
                    MeasuredObservation(
                        workload=workload,
                        schedule=schedule,
                        ratio_vs_official=0.80 + 0.02 * variant,
                        ratio_vs_bank=0.80 + 0.02 * variant,
                        source="calibration_coupled_counterfactual",
                        record_id=f"hierarchical_{workload_index}_{variant}",
                        verified=True,
                        structured_verified=True,
                        bank_schedule=self.bank,
                    )
                )
        effect = BankRelativeEffectModel(
            evidence, self.hardware
        ).predict(
            self.workload,
            self.bank,
            self.custom.schedule,
            self.hardware,
        )
        self.assertEqual(effect.samples, 2)
        self.assertGreater(effect.behavior_samples, effect.samples)
        self.assertTrue(effect.uncertainty < float("inf"))

        rejected = [
            MeasuredObservation(
                workload=observation.workload,
                schedule=observation.schedule,
                ratio_vs_official=1.0,
                ratio_vs_bank=1.0,
                source="runtime_rejected",
                record_id=observation.record_id,
                bank_schedule=observation.bank_schedule,
            )
            for observation in evidence
        ]
        safety = BankRelativeSafetyModel(
            rejected, self.hardware
        ).predict(
            self.workload,
            self.bank,
            self.custom.schedule,
            self.hardware,
        )
        self.assertEqual(safety.samples, 2)
        self.assertGreater(safety.behavior_samples, safety.samples)
        self.assertGreater(safety.risk, 0.50)

    def test_adaptive_calibration_uses_feedback_and_behavior_quotas(
        self,
    ) -> None:
        candidates = []
        origins = (
            ("feedback_winner_transfer", "iterateOrder"),
            ("feedback_winner_mutation", "l2IterateOrder"),
            ("contract_coupled_policy", "l2MTileCnt"),
            ("contract_coupled_policy", "l2NTileCnt"),
            ("feedback_regression_counterfactual", "baseM"),
            ("feedback_regression_counterfactual", "baseN"),
            ("contract_coupled_policy", "depthA1"),
            ("contract_coupled_policy", "depthB1"),
            ("contract_coupled_policy", "stepM"),
            ("contract_coupled_policy", "stepN"),
            ("contract_coupled_policy", "stepKa"),
            ("contract_coupled_policy", "stepKb"),
        )
        for index, (source, field) in enumerate(origins, 1):
            schedule = self.bank.replace(
                **{field: self.bank[field] + index}
            )
            candidates.append(
                Candidate(
                    schedule=schedule,
                    template=Template.BASE,
                    source=source,
                    rationale=f"adaptive candidate {index}",
                )
            )
        selected = select_adaptive_calibration_candidates(
            self.workload,
            [self.incumbent, *candidates],
            self.incumbent,
            (),
            self.hardware,
            budget=10,
            cost_model=_RuntimeModel(),
            effect_model=_AdaptiveEffectModel(),
            safety_model=_AdaptiveSafetyModel(),
            observed_bins=frozenset(),
        )
        self.assertEqual(len(selected), 10)
        self.assertNotIn(
            self.bank.signature(),
            {candidate.schedule.signature() for candidate in selected},
        )
        sources = {candidate.source for candidate in selected}
        self.assertIn("adaptive_winner_transfer", sources)
        self.assertIn("adaptive_one_subsystem", sources)
        self.assertIn("adaptive_regression_boundary", sources)
        self.assertIn("adaptive_unexplored_behavior", sources)

    def test_safety_model_aggregates_conflicting_bin_evidence(
        self,
    ) -> None:
        success = MeasuredObservation(
            workload=self.workload,
            schedule=self.custom.schedule,
            ratio_vs_official=1.0,
            ratio_vs_bank=1.0,
            source="calibration_coupled_counterfactual",
            record_id="success",
            verified=True,
            structured_verified=True,
            bank_schedule=self.bank,
        )
        rejection = MeasuredObservation(
            workload=self.workload,
            schedule=self.custom.schedule,
            ratio_vs_official=1.0,
            ratio_vs_bank=1.0,
            source="runtime_rejected",
            record_id="rejection",
            bank_schedule=self.bank,
        )
        risks = []
        for evidence in ([success, rejection], [rejection, success]):
            prediction = BankRelativeSafetyModel(
                evidence, self.hardware
            ).predict(
                self.workload,
                self.bank,
                self.custom.schedule,
                self.hardware,
            )
            risks.append(prediction.risk)
        self.assertEqual(risks[0], risks[1])
        self.assertAlmostEqual(risks[0], 0.5)

    def test_adaptive_calibration_does_not_spend_budget_on_known_unsafe(
        self,
    ) -> None:
        safe = self.custom
        unsafe = Candidate(
            schedule=self.bank.replace(baseM=192),
            template=Template.BASE,
            source="feedback_winner_mutation",
            rationale="known unsafe transition",
        )
        selected = select_adaptive_calibration_candidates(
            self.workload,
            [safe, unsafe],
            self.incumbent,
            (),
            self.hardware,
            budget=2,
            cost_model=_RuntimeModel(),
            effect_model=_AdaptiveEffectModel(),
            safety_model=_AdaptiveSafetyModel(
                unsafe.schedule.signature()
            ),
            observed_bins=frozenset(),
        )
        self.assertEqual(
            [candidate.schedule for candidate in selected],
            [safe.schedule],
        )

    def test_adaptive_calibration_skips_saturated_target_template(
        self,
    ) -> None:
        observations = [
            MeasuredObservation(
                workload=self.workload,
                schedule=self.custom.schedule,
                ratio_vs_official=1.04,
                ratio_vs_bank=1.03,
                source="calibration_template_probe",
                record_id=f"negative_{index}",
                status_vs_official="regressed",
                status_vs_bank="regressed",
                verified=True,
                structured_verified=True,
                bank_schedule=self.bank,
            )
            for index in range(12)
        ]
        selected = select_adaptive_calibration_candidates(
            self.workload,
            [self.custom],
            self.incumbent,
            observations,
            self.hardware,
            budget=8,
            cost_model=_RuntimeModel(),
            effect_model=_AdaptiveEffectModel(),
            safety_model=_AdaptiveSafetyModel(),
            observed_bins=frozenset(),
            target_templates=frozenset({Template.BASE}),
        )
        self.assertEqual(selected, [])

    def test_adaptive_calibration_refines_numerical_opportunity(
        self,
    ) -> None:
        observations = [
            MeasuredObservation(
                workload=self.workload,
                schedule=self.custom.schedule,
                ratio_vs_official=0.988,
                ratio_vs_bank=0.987,
                source="calibration_template_probe",
                record_id=f"promising_{index}",
                status_vs_official="within_noise",
                status_vs_bank="within_noise",
                verified=True,
                structured_verified=True,
                bank_schedule=self.bank,
            )
            for index in range(12)
        ]
        selected = select_adaptive_calibration_candidates(
            self.workload,
            [self.custom],
            self.incumbent,
            observations,
            self.hardware,
            budget=8,
            cost_model=_RuntimeModel(),
            effect_model=_AdaptiveEffectModel(),
            safety_model=_AdaptiveSafetyModel(),
            observed_bins=frozenset(),
            target_templates=frozenset({Template.BASE}),
        )
        self.assertEqual(
            [candidate.schedule for candidate in selected],
            [self.custom.schedule],
        )

    def test_paired_full_load_evidence_can_change_one_shot_template(
        self,
    ) -> None:
        workload = generate_template_calibration_workloads(
            self.hardware
        )[0].workload
        bank = Schedule.from_signature(
            "20:96:128:1024:96:128:128:8:8:1:1:0:"
            "4:4:2:2:1:1:1:1:41:0:0"
        )
        incumbent = Candidate(
            bank, Template.BASE, "bank_incumbent", "bank"
        )
        schedule = next(
            schedule
            for schedule in Al1FullLoadSolver().generate(
                workload, self.hardware, ()
            )
            if (
                schedule["usedCoreNum"] == 20
                and common_hardware_contract(
                    workload, schedule, self.hardware
                ).valid
                and template_kernel_contract(
                    workload, schedule, self.hardware
                ).valid
            )
        )
        candidate = Candidate(
            schedule,
            Template.AL1_FULL_LOAD,
            "contract_global",
            "independent AL1 solver",
        )
        evidence = []
        for index in range(4):
            evidence_workload = Workload(
                workload_id=f"al1_evidence_{index}",
                m=workload.m,
                n=workload.n + index * 16,
                k=workload.k,
                dtype=workload.dtype,
                trans_a=workload.trans_a,
                trans_b=workload.trans_b,
                max_cores=20,
            )
            evidence.append(
                MeasuredObservation(
                    workload=evidence_workload,
                    schedule=schedule,
                    ratio_vs_official=0.80,
                    ratio_vs_bank=0.80,
                    source="calibration_template_probe",
                    record_id=f"al1_evidence_{index}",
                    verified=True,
                    structured_verified=True,
                    bank_schedule=bank,
                )
            )
        decision = select_one_shot_candidate(
            workload,
            [candidate],
            incumbent,
            evidence,
            self.hardware,
            cost_model=_RuntimeModel(),
        )
        self.assertEqual(
            decision.selection_policy,
            "paired_control_relative_effect",
        )
        self.assertEqual(
            decision.deployment_candidate.template,
            Template.AL1_FULL_LOAD,
        )
        self.assertGreaterEqual(
            decision.candidate.metrics["bank_relative_samples"], 3
        )

    def test_no_runtime_safe_candidate_fails_instead_of_using_bank(
        self,
    ) -> None:
        observations = self._paired_effects(
            self.custom, ratio=0.82, count=4
        )
        with self.assertRaisesRegex(
            ValueError, "no runtime-safe independent candidate"
        ):
            select_one_shot_candidate(
                self.workload,
                [self.custom],
                self.incumbent,
                observations,
                self.hardware,
                cost_model=_AlwaysRiskyModel(),
                safety_model=_UnsafeRelativeSafetyModel(),
            )

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
        self.assertEqual(
            weak_decision.selection_policy,
            "paired_feedback_active_challenger",
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
            strong_decision.deployment_candidate.schedule, self.bank
        )
        self.assertEqual(
            strong_decision.candidate.metrics["template_competitive"], 0.0
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
        self.assertEqual(
            decision.deployment_candidate.schedule, self.bank
        )

    def test_low_core_split_k_has_no_template_opportunity(self) -> None:
        workload = Workload(
            workload_id="net_log27_counterexample",
            m=2911,
            n=3809,
            k=6273,
            dtype="fp16",
            trans_a=False,
            trans_b=False,
            max_cores=20,
        )
        bank = Schedule.from_signature(
            "20:128:256:6273:128:256:64:16:8:1:1:0:"
            "8:4:2:2:1:2:1:12:15:0:0"
        )
        split = Schedule.from_signature(
            "2:384:128:384:128:128:128:9:6:3:1:1:"
            "3:3:2:2:2:1:3:5:1:0:2"
        )
        competition = compare_templates(
            workload,
            bank,
            split,
            behavior_vector(workload, bank, self.hardware),
            behavior_vector(workload, split, self.hardware),
            self.hardware,
            effect_samples=8,
            effect_support=0.57,
            effect_upper_ratio=4.23,
        )
        self.assertFalse(competition.competitive)
        self.assertGreater(competition.compute_floor_ratio, 9.0)
        self.assertEqual(
            competition.reason, "no_cross_template_advantage"
        )

    def test_underfilled_bank_allows_split_k_parallelism_probe(self) -> None:
        workload = Workload(
            workload_id="underfilled_output",
            m=128,
            n=128,
            k=32768,
            dtype="fp16",
            trans_a=False,
            trans_b=False,
            max_cores=20,
        )
        bank = Schedule.from_signature(
            "1:128:128:32768:128:128:64:16:8:1:1:0:"
            "8:4:2:2:1:1:1:1:1:0:0"
        )
        split = Schedule.from_signature(
            "20:384:128:384:128:128:128:9:6:3:1:1:"
            "3:3:2:2:2:1:1:1:1:0:3"
        )
        competition = compare_templates(
            workload,
            bank,
            split,
            behavior_vector(workload, bank, self.hardware),
            behavior_vector(workload, split, self.hardware),
            self.hardware,
            effect_samples=0,
            effect_support=0.0,
            effect_upper_ratio=float("inf"),
        )
        self.assertTrue(competition.hardware_opportunity)
        self.assertTrue(competition.competitive)
        self.assertGreater(competition.active_core_gain, 10.0)

    def test_same_template_candidate_beats_noncompetitive_split_probe(
        self,
    ) -> None:
        workload = Workload(
            workload_id="net_log27_selector_counterexample",
            m=2911,
            n=3809,
            k=6273,
            dtype="fp16",
            trans_a=False,
            trans_b=False,
            max_cores=20,
        )
        bank = Schedule.from_signature(
            "20:128:256:6273:128:256:64:16:8:1:1:0:"
            "8:4:2:2:1:2:1:12:15:0:0"
        )
        incumbent = Candidate(
            bank, Template.BASE, "bank_incumbent", "bank"
        )
        base = Candidate(
            bank.replace(
                l2MTileCnt=2,
                l2NTileCnt=2,
                l2MTileBlock=6,
                l2NTileBlock=8,
            ),
            Template.BASE,
            "feedback_winner_transfer",
            "same-template L2 hypothesis",
        )
        split = Candidate(
            Schedule.from_signature(
                "2:384:128:384:128:128:128:9:6:3:1:1:"
                "3:3:2:2:2:1:3:5:1:0:2"
            ),
            Template.SINGLE_CORE_SPLIT_K,
            "feedback_regression_counterfactual",
            "v9 failure structure",
        )
        decision = select_one_shot_candidate(
            workload,
            [split, base],
            incumbent,
            (),
            self.hardware,
            cost_model=_RuntimeModel(),
        )
        self.assertEqual(decision.candidate.schedule, base.schedule)
        self.assertEqual(
            decision.candidate.metrics["template_competitive"], 1.0
        )

    def test_cross_template_history_cannot_override_work_floor(
        self,
    ) -> None:
        workload = Workload(
            workload_id="unseen_dense_false_split_prediction",
            m=3840,
            n=4608,
            k=6912,
            dtype="fp16",
            trans_a=False,
            trans_b=False,
            max_cores=20,
        )
        bank = Schedule.from_signature(
            "20:128:256:6912:128:256:64:16:8:1:1:0:"
            "8:4:2:2:1:2:1:15:18:0:0"
        )
        incumbent = Candidate(
            bank, Template.BASE, "bank_incumbent", "bank"
        )
        base = Candidate(
            bank.replace(l2IterateOrder=1),
            Template.BASE,
            "feedback_winner_transfer",
            "same-template control",
        )
        split = Candidate(
            Schedule.from_signature(
                "18:384:4608:384:128:128:128:9:6:3:1:1:"
                "3:3:2:2:2:1:1:1:12:1:3"
            ),
            Template.DETERMINISTIC_SPLIT_K,
            "feedback_regression_counterfactual",
            "false optimistic cross-template prediction",
        )
        decision = select_one_shot_candidate(
            workload,
            [split, base],
            incumbent,
            (),
            self.hardware,
            cost_model=_RuntimeModel(),
            effect_model=_FalseOptimisticCrossTemplateModel(),
        )
        self.assertEqual(decision.candidate.schedule, base.schedule)
        self.assertEqual(
            decision.candidate.metrics["template_competitive"], 1.0
        )

    def test_calibration_separates_local_coupled_and_template_probes(
        self,
    ) -> None:
        local = Candidate(
            schedule=self.custom.schedule,
            template=Template.BASE,
            source="local_bank_anchor",
            rationale="calibration local mutation",
        )
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
