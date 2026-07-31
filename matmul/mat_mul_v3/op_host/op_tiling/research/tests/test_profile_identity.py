from __future__ import annotations

import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path


RESEARCH = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "matmul_v3_research_profile", RESEARCH / "profile.py"
)
assert SPEC is not None and SPEC.loader is not None
PROFILE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROFILE
SPEC.loader.exec_module(PROFILE)


class ProfileIdentityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.row = {
            "workload_id": "identity_probe",
            "m": "777",
            "n": "1333",
            "k": "8192",
            "dtype": "fp16",
            "trans_a": "0",
            "trans_b": "1",
            "candidate_role": "searched",
            "tiling_signature": ",".join(str(value) for value in range(1, 24)),
        }

    def test_runtime_kb_input_is_exactly_183_bytes(self) -> None:
        info = PROFILE.make_info(self.row)
        self.assertEqual(len(PROFILE.pack_info(info)), 183)

    def test_resume_identity_covers_runtime_shape_role_and_schedule(self) -> None:
        base = PROFILE.measurement_key(
            "Ascend910B3", 20, "8.1.RC1", self.row
        )
        variants = []
        for field, value in (
            ("m", "778"),
            ("candidate_role", "bank_seed_control"),
            ("tiling_signature", ",".join("2" for _ in range(23))),
        ):
            changed = dict(self.row)
            changed[field] = value
            variants.append(
                PROFILE.measurement_key(
                    "Ascend910B3", 20, "8.1.RC1", changed
                )
            )
        variants.extend(
            (
                PROFILE.measurement_key(
                    "Ascend910B4", 20, "8.1.RC1", self.row
                ),
                PROFILE.measurement_key(
                    "Ascend910B3", 24, "8.1.RC1", self.row
                ),
                PROFILE.measurement_key(
                    "Ascend910B3", 20, "8.5.0", self.row
                ),
            )
        )
        self.assertTrue(all(value != base for value in variants))
        self.assertEqual(len(set(variants)), len(variants))

    def test_summary_ranks_by_paired_ratio_not_cross_run_latency(
        self,
    ) -> None:
        first = dict(self.row)
        first.update(
            {
                "rank": "1",
                "search_template": "BASE",
                "candidate_source": "contract_global",
                "tiling_signature": "first",
            }
        )
        second = dict(self.row)
        second.update(
            {
                "rank": "2",
                "search_template": "BASE",
                "candidate_source": "feedback_winner_mutation",
                "tiling_signature": "second",
            }
        )
        records = {}
        for row, candidate_ms, baseline_ms in (
            (first, 1.0, 1.0),
            (second, 1.1, 2.0),
        ):
            key = PROFILE.measurement_key(
                "Ascend910B3", 20, "8.1.RC1", row
            )
            records[key] = {
                **row,
                "success": "1",
                "median_ms": str(candidate_ms),
                "official_ms": str(baseline_ms),
                "bank_ms": str(baseline_ms),
                "speedup_vs_official": str(
                    baseline_ms / candidate_ms
                ),
                "speedup_vs_bank": str(baseline_ms / candidate_ms),
                "status_vs_official": (
                    "improved" if candidate_ms < baseline_ms else "within_noise"
                ),
                "status_vs_bank": (
                    "improved" if candidate_ms < baseline_ms else "within_noise"
                ),
            }
        with redirect_stdout(io.StringIO()):
            summary = PROFILE.summarize(
                [first, second],
                records,
                "Ascend910B3",
                20,
                "8.1.RC1",
            )[0]
        self.assertEqual(summary["best_rank"], "2")
        self.assertEqual(
            summary["best_source"], "feedback_winner_mutation"
        )


if __name__ == "__main__":
    unittest.main()
