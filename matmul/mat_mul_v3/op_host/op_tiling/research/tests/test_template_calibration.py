from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


RESEARCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH))

from tiling_search.calibration_workloads import (
    decode_template_quotas,
    encode_template_quotas,
    generate_template_calibration_workloads,
)
from tiling_search.contracts import (
    common_hardware_contract,
    template_kernel_contract,
    template_of,
)
from tiling_search.domain import Candidate, Hardware, Template
from tiling_search.one_shot import select_calibration_candidates
from tiling_search.orchestrator import CandidateEngine, SearchConfig
from tiling_search.solvers import (
    Al1FullLoadSolver,
    BaseSolver,
    Bl1FullLoadSolver,
    DeterministicSplitKSolver,
    SingleCoreSplitKSolver,
)


class TemplateCalibrationTest(unittest.TestCase):
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

    def _target_solvers(self):
        bl1 = Bl1FullLoadSolver()
        return {
            Template.AL1_FULL_LOAD: Al1FullLoadSolver(),
            Template.BL1_FULL_LOAD: bl1,
            Template.BL1_FULL_LOAD_FIXPIPE: bl1,
            Template.BL1_FULL_LOAD_VEC_NZ2ND: bl1,
            Template.SINGLE_CORE_SPLIT_K: SingleCoreSplitKSolver(),
            Template.DETERMINISTIC_SPLIT_K: DeterministicSplitKSolver(),
        }

    def _legal_target_candidates(self, spec, template, limit):
        result = []
        seen = set()
        solver = self._target_solvers()[template]
        for schedule in solver.generate(spec.workload, self.hardware, ()):
            if (
                template_of(schedule) != template
                or schedule.signature() in seen
            ):
                continue
            seen.add(schedule.signature())
            if not common_hardware_contract(
                spec.workload, schedule, self.hardware
            ).valid:
                continue
            if not template_kernel_contract(
                spec.workload, schedule, self.hardware
            ).valid:
                continue
            result.append(schedule)
            if len(result) == limit:
                break
        return result

    def test_design_covers_every_supported_non_base_template(self) -> None:
        specs = generate_template_calibration_workloads(self.hardware)
        totals = Counter()
        for spec in specs:
            totals.update(spec.template_quotas)
            self.assertLess(spec.resident_ratio, 1.0)
            encoded = encode_template_quotas(spec.template_quotas)
            self.assertEqual(
                decode_template_quotas(encoded), spec.template_quotas
            )
        self.assertEqual(len(specs), 36)
        self.assertEqual(
            totals,
            Counter(
                {
                    Template.AL1_FULL_LOAD: 48,
                    Template.BL1_FULL_LOAD: 120,
                    Template.BL1_FULL_LOAD_FIXPIPE: 12,
                    Template.BL1_FULL_LOAD_VEC_NZ2ND: 12,
                    Template.SINGLE_CORE_SPLIT_K: 30,
                    Template.DETERMINISTIC_SPLIT_K: 12,
                }
            ),
        )

    def test_every_target_quota_exists_without_bank_geometry(self) -> None:
        for spec in generate_template_calibration_workloads(self.hardware):
            for template, quota in spec.template_quotas.items():
                with self.subTest(
                    workload=spec.workload.workload_id,
                    template=template.value,
                ):
                    candidates = self._legal_target_candidates(
                        spec, template, quota
                    )
                    self.assertEqual(len(candidates), quota)

    def test_full_load_solvers_are_not_exploration_only(self) -> None:
        engine = CandidateEngine(
            config=SearchConfig(include_exploration=False)
        )
        solver_types = {type(solver) for solver in engine.solvers}
        self.assertIn(Al1FullLoadSolver, solver_types)
        self.assertIn(Bl1FullLoadSolver, solver_types)
        self.assertIn(SingleCoreSplitKSolver, solver_types)
        self.assertIn(DeterministicSplitKSolver, solver_types)

    def test_quota_selection_accepts_independent_full_load_source(self) -> None:
        spec = generate_template_calibration_workloads(
            self.hardware
        )[0]
        schedules = self._legal_target_candidates(
            spec, Template.AL1_FULL_LOAD, 8
        )
        base = next(
            BaseSolver().generate(spec.workload, self.hardware, ())
        )
        incumbent = Candidate(
            schedule=base,
            template=Template.BASE,
            source="bank_incumbent",
            rationale="test control",
        )
        candidates = [
            Candidate(
                schedule=schedule,
                template=Template.AL1_FULL_LOAD,
                source="contract_global",
                rationale="independent solver",
            )
            for schedule in schedules
        ]
        selected = select_calibration_candidates(
            spec.workload,
            candidates,
            incumbent,
            (),
            self.hardware,
            budget=8,
            template_quotas={Template.AL1_FULL_LOAD: 8},
        )
        self.assertEqual(len(selected), 8)
        self.assertEqual(
            {candidate.template for candidate in selected},
            {Template.AL1_FULL_LOAD},
        )
        self.assertEqual(
            {candidate.source for candidate in selected},
            {"calibration_template_probe"},
        )

    def test_audit_executes_and_counts_unique_paired_signatures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workloads = root / "workloads.csv"
            resume = root / "resume.csv"
            workload_row = {
                "id": "audit_probe",
                "template_quotas": "BASE:1",
            }
            with workloads.open(
                "w", newline="", encoding="utf-8"
            ) as target:
                writer = csv.DictWriter(
                    target, fieldnames=tuple(workload_row)
                )
                writer.writeheader()
                writer.writerow(workload_row)
            resume_row = {
                "candidate_role": "searched",
                "soc": "Ascend910B3",
                "aic": "20",
                "toolkit": "8.1.RC1",
                "workload_id": "audit_probe",
                "tiling_signature": (
                    "20:128:128:128:128:128:64:8:8:1:1:0:"
                    "4:4:2:2:1:1:1:1:1:0:0"
                ),
                "success": "1",
                "pair_validated": "1",
                "preflight_mode": "numeric_signed_axes_full_v3",
                "official_ms": "1",
                "bank_ms": "1",
                "record_id": "paired",
            }
            with resume.open(
                "w", newline="", encoding="utf-8"
            ) as target:
                writer = csv.DictWriter(
                    target, fieldnames=tuple(resume_row)
                )
                writer.writeheader()
                writer.writerow(resume_row)
                resume_row["record_id"] = "duplicate"
                writer.writerow(resume_row)
            command = [
                sys.executable,
                str(RESEARCH / "audit_template_calibration.py"),
                "--workloads",
                str(workloads),
                "--resume",
                str(resume),
                "--soc",
                "Ascend910B3",
                "--aic",
                "20",
                "--toolkit",
                "8.1.RC1",
            ]
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(
                "TEMPLATE_CALIBRATION_AUDIT status=passed",
                completed.stdout,
            )

            workload_row["template_quotas"] = "BASE:2"
            with workloads.open(
                "w", newline="", encoding="utf-8"
            ) as target:
                writer = csv.DictWriter(
                    target, fieldnames=tuple(workload_row)
                )
                writer.writeheader()
                writer.writerow(workload_row)
            incomplete = subprocess.run(
                command,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(incomplete.returncode, 0)
            self.assertIn("paired=1 required=2", incomplete.stdout)


if __name__ == "__main__":
    unittest.main()
