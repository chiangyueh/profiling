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
from tiling_search.feedback import fingerprint
from tiling_search.racing import plan_template_race


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
        self.assertEqual(
            {candidate.source for candidate in result.candidates},
            {"contract_global"},
        )
        templates = Counter(
            candidate.template for candidate in result.callback_candidates
        )
        probe_budget = max(
            2, len(result.callback_candidates) // 8
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

    def test_external_v4_campaign_shapes_are_not_in_feedback(self) -> None:
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
        self.assertEqual(len(workloads), 22)
        self.assertFalse(workloads.intersection(observed))

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
