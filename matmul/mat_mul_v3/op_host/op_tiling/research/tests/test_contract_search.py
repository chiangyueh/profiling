from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
