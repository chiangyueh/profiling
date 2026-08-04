from __future__ import annotations

import csv
import re
import sys
import unittest
from collections import Counter
from pathlib import Path


RESEARCH = Path(__file__).resolve().parents[1]
SOURCE_ROOT = RESEARCH.parents[2]
sys.path.insert(0, str(RESEARCH))

from tiling_search import (
    CandidateEngine,
    GenerationBudget,
    Hardware,
    MeasuredObservation,
    SearchConfig,
    Workload,
)
from tiling_search.domain import KNOWLEDGE_FIELDS, Schedule, Template
from tiling_search.behavior import select_behavior_coverage
from tiling_search.behavior import behavior_vector
from tiling_search.contracts import validate_schedule
from tiling_search.feedback import fingerprint
from tiling_search.racing import plan_template_race
from tiling_search.solvers import BaseSolver, SingleCoreSplitKSolver


class ContractSearchTest(unittest.TestCase):
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

    def test_schema_is_read_from_official_85_header(self) -> None:
        text = (
            SOURCE_ROOT / "op_host/op_tiling/matmul_v3_tuning.h"
        ).read_text(encoding="utf-8")
        fields = tuple(
            re.findall(
                r"TUNING_TILING_DATA_FIELD_DEF\(uint32_t,\s*([A-Za-z0-9_]+)\)",
                text,
            )
        )
        self.assertEqual(fields, KNOWLEDGE_FIELDS)

    def test_base_solver_leads_with_upstream_coupled_l1_policy(
        self,
    ) -> None:
        workload = Workload(
            workload_id="base_policy_probe",
            m=4096,
            n=6144,
            k=4096,
            dtype="fp16",
            trans_a=False,
            trans_b=False,
            max_cores=20,
        )
        schedule = next(BaseSolver().generate(workload, self.hardware))
        self.assertEqual(
            (
                schedule["baseM"],
                schedule["baseN"],
                schedule["baseK"],
                schedule["depthA1"],
                schedule["depthB1"],
                schedule["stepKa"],
                schedule["stepKb"],
                schedule["dbL0A"],
                schedule["dbL0B"],
                schedule["dbL0C"],
            ),
            (128, 256, 64, 16, 8, 8, 4, 2, 2, 1),
        )
        self.assertTrue(
            validate_schedule(workload, schedule, self.hardware).valid
        )
        vector = behavior_vector(workload, schedule, self.hardware)
        self.assertGreater(
            vector.metrics["l1_pipeline_efficiency"], 0.90
        )
        self.assertLessEqual(
            vector.metrics["l2_capacity_pressure"], 1.0
        )
        self.assertGreater(
            vector.metrics["l2_wave_efficiency"], 0.90
        )
        underfilled = schedule.replace(
            depthA1=1,
            depthB1=1,
            stepKa=1,
            stepKb=1,
        )
        underfilled_vector = behavior_vector(
            workload, underfilled, self.hardware
        )
        self.assertLess(
            underfilled_vector.metrics["l1_pipeline_efficiency"],
            0.30,
        )
        self.assertGreater(
            underfilled_vector.metrics["analytical_prior"],
            vector.metrics["analytical_prior"],
        )

    def test_single_core_split_k_uses_coupled_upstream_algorithms(
        self,
    ) -> None:
        workload = Workload(
            workload_id="single_split_policy_probe",
            m=1792,
            n=2816,
            k=32768,
            dtype="fp16",
            trans_a=False,
            trans_b=False,
            max_cores=20,
        )
        schedules = list(
            SingleCoreSplitKSolver().generate(
                workload, self.hardware
            )
        )
        self.assertGreaterEqual(len(schedules), 32)
        allowed = {
            (3, 1, 3, 9, 6),
            (1, 3, 3, 6, 9),
            (2, 1, 4, 8, 8),
            (1, 1, 4, 8, 8),
        }
        self.assertEqual(
            {
                (
                    schedule["stepM"],
                    schedule["stepN"],
                    schedule["stepKa"],
                    schedule["depthA1"],
                    schedule["depthB1"],
                )
                for schedule in schedules
            },
            allowed,
        )
        self.assertTrue(
            all(
                (
                    schedule["baseM"],
                    schedule["baseN"],
                    schedule["baseK"],
                    schedule["stepKa"],
                    schedule["stepKb"],
                )
                in {
                    (128, 128, 128, 3, 3),
                    (128, 128, 128, 4, 4),
                }
                and validate_schedule(
                    workload, schedule, self.hardware
                ).valid
                for schedule in schedules
            )
        )

    def test_single_split_k_contract_uses_kernel_l2_semantics(
        self,
    ) -> None:
        workload = Workload(
            workload_id="saved_bank_single_split_k",
            m=2560,
            n=3072,
            k=30720,
            dtype="fp16",
            trans_a=False,
            trans_b=False,
            max_cores=20,
        )
        bank = Schedule.from_signature(
            "20:256:1536:512:128:128:128:8:8:2:1:1:"
            "4:4:2:2:2:4:3:5:4:0:2"
        )
        self.assertTrue(
            validate_schedule(workload, bank, self.hardware).valid
        )

    def test_coupled_solver_reconstructs_bank_without_local_source(
        self,
    ) -> None:
        workload = Workload(
            workload_id="unseen_fp32_nt_reconstruction",
            m=112,
            n=640,
            k=4352,
            dtype="fp32",
            trans_a=False,
            trans_b=True,
            max_cores=20,
        )
        bank = Schedule.from_signature(
            "20:64:64:4352:64:64:128:8:8:1:1:0:"
            "4:4:2:2:1:1:1:2:10:0:0"
        )
        engine = CandidateEngine(
            config=SearchConfig(include_exploration=False)
        )
        result = engine.generate(
            workload,
            self.hardware,
            local_anchor=bank,
        )
        reconstructed = [
            candidate
            for candidate in result.callback_candidates
            if candidate.schedule == bank
        ]
        self.assertEqual(len(reconstructed), 1)
        self.assertEqual(
            reconstructed[0].source, "contract_coupled_policy"
        )

    def test_unseen_name_generates_multiple_templates(self) -> None:
        workload = Workload(
            workload_id="name_has_no_search_semantics",
            m=777,
            n=1333,
            k=8192,
            dtype="fp16",
            trans_a=False,
            trans_b=False,
            max_cores=20,
        )
        engine = CandidateEngine(
            config=SearchConfig(
                GenerationBudget(
                    raw_attempts=4000,
                    legal_candidates=1500,
                    behavior_candidates=96,
                    callback_candidates=64,
                    npu_candidates=40,
                )
            )
        )
        result = engine.generate(workload, self.hardware)
        self.assertGreaterEqual(len(result.callback_candidates), 40)
        self.assertGreater(result.legal_candidates, result.draft_candidates)
        self.assertLessEqual(result.draft_candidates, 512)
        self.assertGreaterEqual(
            len({candidate.template for candidate in result.candidates}), 2
        )
        self.assertLessEqual(
            {candidate.source for candidate in result.candidates},
            {"contract_global", "contract_coupled_policy"},
        )
        self.assertTrue(
            any(
                candidate.source == "contract_coupled_policy"
                for candidate in result.callback_candidates
            )
        )
        self.assertTrue(
            any(
                candidate.template == Template.BASE
                and candidate.schedule["singleCoreM"]
                == candidate.schedule["baseM"]
                and candidate.schedule["singleCoreN"]
                == candidate.schedule["baseN"]
                for candidate in result.callback_candidates
            )
        )
        templates = Counter(
            candidate.template for candidate in result.callback_candidates
        )
        probe_budget = max(
            2, len(result.callback_candidates) // 2
        )
        for template, count in templates.items():
            if template != Template.BASE:
                self.assertLessEqual(count, probe_budget)
        npu_candidates = select_behavior_coverage(
            workload,
            result.callback_candidates,
            (),
            self.hardware,
            40,
            probe_templates=False,
        )
        self.assertGreaterEqual(
            len({candidate.template for candidate in npu_candidates}),
            2,
        )

    def test_deployment_engine_excludes_broad_and_full_load_solvers(
        self,
    ) -> None:
        workload = Workload(
            workload_id="deployment_path_has_no_family_name",
            m=1792,
            n=2816,
            k=3584,
            dtype="fp16",
            trans_a=False,
            trans_b=False,
            max_cores=20,
        )
        engine = CandidateEngine(
            config=SearchConfig(
                GenerationBudget(
                    raw_attempts=4000,
                    legal_candidates=1500,
                    behavior_candidates=96,
                    callback_candidates=48,
                    npu_candidates=1,
                ),
                include_exploration=False,
            )
        )
        result = engine.generate(workload, self.hardware)
        allowed_templates = {
            Template.BASE,
            Template.SINGLE_CORE_SPLIT_K,
            Template.DETERMINISTIC_SPLIT_K,
        }
        self.assertTrue(result.callback_candidates)
        self.assertLessEqual(
            {report.template for report in result.reports},
            allowed_templates,
        )
        self.assertLessEqual(
            {
                candidate.template
                for candidate in result.callback_candidates
            },
            allowed_templates,
        )
        self.assertEqual(
            {
                candidate.source
                for candidate in result.callback_candidates
            },
            {"contract_coupled_policy"},
        )

    def test_solver_without_explicit_source_is_rejected(self) -> None:
        class UnscopedSolver:
            template = Template.BASE

            def generate(self, workload, hardware, targets=()):
                del targets
                yield from BaseSolver().generate(workload, hardware)

        workload = Workload(
            workload_id="unscoped_solver_probe",
            m=512,
            n=768,
            k=1024,
            dtype="fp16",
            trans_a=False,
            trans_b=False,
            max_cores=20,
        )
        engine = CandidateEngine(solvers=(UnscopedSolver(),))
        with self.assertRaisesRegex(
            ValueError, "must declare candidate source"
        ):
            engine.generate(workload, self.hardware)

    def test_one_shot_shapes_are_not_in_feedback(self) -> None:
        config = RESEARCH / "config"
        with (config / "workloads.csv").open(
            newline="", encoding="utf-8"
        ) as source:
            workloads = {
                (
                    row["m"],
                    row["n"],
                    row["k"],
                    row["dtype"],
                    row["trans_a"],
                    row["trans_b"],
                )
                for row in csv.DictReader(source)
            }
        observed = set()
        for path in config.glob("measured_observations*.csv"):
            with path.open(newline="", encoding="utf-8") as source:
                observed.update(
                    (
                        row["m"],
                        row["n"],
                        row["k"],
                        row["dtype"],
                        row["trans_a"],
                        row["trans_b"],
                    )
                    for row in csv.DictReader(source)
                )
        self.assertEqual(len(workloads), 24)
        self.assertFalse(workloads.intersection(observed))

    def test_bank_local_mutations_reach_callback_frontier(self) -> None:
        workload = Workload(
            workload_id="local_anchor_probe",
            m=1024,
            n=1536,
            k=8192,
            dtype="fp16",
            trans_a=False,
            trans_b=False,
            max_cores=20,
        )
        engine = CandidateEngine(
            config=SearchConfig(
                GenerationBudget(
                    raw_attempts=2000,
                    legal_candidates=800,
                    behavior_candidates=48,
                    callback_candidates=24,
                    npu_candidates=1,
                )
            )
        )
        initial = engine.generate(workload, self.hardware)
        anchor = next(
            candidate.schedule
            for candidate in initial.callback_candidates
            if candidate.template == Template.BASE
        )
        anchored = engine.generate(
            workload,
            self.hardware,
            local_anchor=anchor,
        )
        self.assertTrue(
            any(
                candidate.source == "local_bank_anchor"
                for candidate in anchored.callback_candidates
            )
        )

    def test_search_does_not_import_retired_candidate_paths(self) -> None:
        forbidden = (
            "refine_matmul",
            "bottleneck_guided",
            "beam_lns",
            "tabu_lns",
        )
        for path in (RESEARCH / "tiling_search").rglob("*.py"):
            text = path.read_text(encoding="utf-8").lower()
            for token in forbidden:
                self.assertNotIn(token, text, f"{path}: {token}")

    def test_feedback_stage_expands_beyond_measured_fingerprints(
        self,
    ) -> None:
        workload = Workload(
            workload_id="feedback_stage_probe",
            m=1024,
            n=1536,
            k=32768,
            dtype="fp16",
            trans_a=False,
            trans_b=False,
            max_cores=20,
        )
        budget = GenerationBudget(
            raw_attempts=3000,
            legal_candidates=1200,
            behavior_candidates=64,
            callback_candidates=16,
            npu_candidates=8,
        )
        first = CandidateEngine(
            config=SearchConfig(budget)
        ).generate(workload, self.hardware)
        measured = list(first.callback_candidates[:8])
        observations = [
            MeasuredObservation(
                workload=workload,
                schedule=candidate.schedule,
                ratio_vs_official=0.8 if index == 0 else 2.0,
                ratio_vs_bank=0.8 if index == 0 else 2.0,
                source=candidate.source,
                record_id=f"stage1-{index}",
                status_vs_official=(
                    "improved" if index == 0 else "regressed"
                ),
                status_vs_bank=(
                    "improved" if index == 0 else "regressed"
                ),
                verified=True,
                structured_verified=True,
            )
            for index, candidate in enumerate(measured)
        ]
        exclusions = {
            fingerprint(workload, candidate.schedule)
            for candidate in measured
        }
        second = CandidateEngine(
            config=SearchConfig(budget),
            observations=observations,
            exclusions=exclusions,
        ).generate(workload, self.hardware)
        self.assertTrue(
            all(
                fingerprint(workload, candidate.schedule)
                not in exclusions
                for candidate in second.callback_candidates
            )
        )
        self.assertTrue(
            any(
                candidate.source.startswith("feedback_")
                for candidate in second.candidates
            )
        )

    def test_provisional_winner_fingerprint_is_not_remeasured(
        self,
    ) -> None:
        workload = Workload(
            workload_id="revalidation_probe",
            m=128,
            n=128,
            k=16384,
            dtype="fp16",
            trans_a=False,
            trans_b=False,
            max_cores=20,
        )
        budget = GenerationBudget(
            raw_attempts=3000,
            legal_candidates=1200,
            behavior_candidates=64,
            callback_candidates=16,
            npu_candidates=8,
        )
        schedule = CandidateEngine(
            config=SearchConfig(budget)
        ).generate(workload, self.hardware).callback_candidates[0].schedule
        observation = MeasuredObservation(
            workload=workload,
            schedule=schedule,
            ratio_vs_official=0.8,
            ratio_vs_bank=0.8,
            source="contract_global",
            record_id="coverage-only-winner",
            status_vs_official="improved",
            status_vs_bank="improved",
        )
        result = CandidateEngine(
            config=SearchConfig(
                budget
            ),
            observations=[observation],
            exclusions={fingerprint(workload, schedule)},
        ).generate(workload, self.hardware)
        self.assertNotIn(
            schedule.signature(),
            {
                candidate.schedule.signature()
                for candidate in result.callback_candidates
            },
        )

    def test_template_race_reduces_only_with_repeated_paired_winner(
        self,
    ) -> None:
        workload = Workload(
            workload_id="unseen_racing_probe",
            m=1792,
            n=2816,
            k=3584,
            dtype="fp16",
            trans_a=False,
            trans_b=False,
            max_cores=20,
        )
        result = CandidateEngine(
            config=SearchConfig(
                GenerationBudget(
                    raw_attempts=5000,
                    legal_candidates=2000,
                    behavior_candidates=128,
                    callback_candidates=96,
                    npu_candidates=40,
                )
            )
        ).generate(workload, self.hardware)
        by_template = {}
        for candidate in result.candidates:
            by_template.setdefault(candidate.template, []).append(candidate)
        by_template = {
            template: candidates
            for template, candidates in by_template.items()
            if len(candidates) >= 2
        }
        self.assertGreaterEqual(len(by_template), 2)
        leader = sorted(by_template, key=lambda item: item.value)[0]
        observations = []
        for template, candidates in by_template.items():
            for index, candidate in enumerate(candidates[:2]):
                winner = template == leader
                observations.append(
                    MeasuredObservation(
                        workload=workload,
                        schedule=candidate.schedule,
                        ratio_vs_official=0.80 if winner else 1.20,
                        ratio_vs_bank=0.82 if winner else 1.18,
                        source=candidate.source,
                        record_id=f"{template.value}-{index}",
                        status_vs_official=(
                            "improved" if winner else "regressed"
                        ),
                        status_vs_bank=(
                            "improved" if winner else "regressed"
                        ),
                        verified=True,
                        structured_verified=True,
                    )
                )
        candidates = [
            candidate
            for members in by_template.values()
            for candidate in members[:16]
        ]
        plan = plan_template_race(
            workload, candidates, observations, 24
        )
        self.assertEqual(plan.state, "clear_template_leader")
        self.assertEqual(plan.budget, 16)
        self.assertGreater(
            plan.template_quotas[leader],
            min(
                quota
                for template, quota in plan.template_quotas.items()
                if template != leader
            ),
        )
        self.assertTrue(
            all(quota >= 1 for quota in plan.template_quotas.values())
        )
        selected = select_behavior_coverage(
            workload,
            candidates,
            observations,
            self.hardware,
            plan.budget,
            probe_templates=True,
            template_probe_floor=1,
            template_quotas=plan.template_quotas,
        )
        self.assertEqual(len(selected), plan.budget)
        self.assertEqual(
            Counter(candidate.template for candidate in selected),
            Counter(plan.template_quotas),
        )

    def test_template_race_keeps_full_budget_when_evidence_is_weak(
        self,
    ) -> None:
        workload = Workload(
            workload_id="weak_evidence_probe",
            m=1023,
            n=1537,
            k=3073,
            dtype="bf16",
            trans_a=False,
            trans_b=False,
            max_cores=20,
        )
        candidates = list(
            CandidateEngine(
                config=SearchConfig(
                    GenerationBudget(
                        raw_attempts=3000,
                        legal_candidates=1200,
                        behavior_candidates=96,
                        callback_candidates=48,
                        npu_candidates=24,
                    )
                )
            ).generate(workload, self.hardware).callback_candidates
        )
        observation = MeasuredObservation(
            workload=workload,
            schedule=candidates[0].schedule,
            ratio_vs_official=0.8,
            ratio_vs_bank=0.8,
            source=candidates[0].source,
            record_id="single-fast-sample",
            status_vs_official="improved",
            status_vs_bank="improved",
            verified=True,
            structured_verified=True,
        )
        plan = plan_template_race(
            workload, candidates, [observation], 24
        )
        self.assertEqual(plan.state, "templates_competitive")
        self.assertEqual(plan.budget, 24)

    def test_cold_start_balances_templates_and_caps_known_risk(
        self,
    ) -> None:
        workload = Workload(
            workload_id="cold_start_risk_probe",
            m=1792,
            n=2816,
            k=3584,
            dtype="fp16",
            trans_a=False,
            trans_b=False,
            max_cores=20,
        )
        result = CandidateEngine(
            config=SearchConfig(
                GenerationBudget(
                    raw_attempts=5000,
                    legal_candidates=2000,
                    behavior_candidates=128,
                    callback_candidates=96,
                    npu_candidates=40,
                )
            )
        ).generate(workload, self.hardware)
        by_template = {}
        for candidate in result.candidates:
            by_template.setdefault(candidate.template, []).append(candidate)
        eligible = {
            template: candidates[:16]
            for template, candidates in by_template.items()
            if len(candidates) >= 16
        }
        self.assertGreaterEqual(len(eligible), 2)
        selected_templates = sorted(
            eligible, key=lambda item: item.value
        )[:2]
        candidates = [
            candidate.with_selection(
                acquisition=0.0,
                behavior_key=(),
                metrics={
                    "runtime_risk_score": 0.1,
                    "runtime_risk_support": 1.0,
                },
            )
            for template in selected_templates
            for candidate in eligible[template]
        ]
        cold_plan = plan_template_race(
            workload, candidates, (), 16
        )
        self.assertEqual(cold_plan.state, "balanced_cold_start")
        self.assertEqual(cold_plan.budget, 16)
        self.assertLessEqual(
            max(cold_plan.template_quotas.values())
            - min(cold_plan.template_quotas.values()),
            1,
        )

        risky_template = selected_templates[0]
        risk_annotated = [
            candidate.with_selection(
                acquisition=0.0,
                behavior_key=(),
                metrics={
                    "runtime_risk_score": (
                        0.9
                        if candidate.template == risky_template
                        else 0.1
                    ),
                    "runtime_risk_support": 1.0,
                },
            )
            for candidate in candidates
        ]
        risk_plan = plan_template_race(
            workload, risk_annotated, (), 12
        )
        self.assertEqual(risk_plan.budget, 12)
        self.assertEqual(
            risk_plan.template_quotas[risky_template],
            1,
        )


if __name__ == "__main__":
    unittest.main()
