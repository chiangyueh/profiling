from __future__ import annotations

import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path


RESEARCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH))
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

    def test_profile_record_preserves_paired_bank_signature(self) -> None:
        source = dict(self.row)
        source.update(
            {
                "rank": "1",
                "candidate_source": "calibration_local_counterfactual",
                "search_template": "BASE",
                "bank_tiling_signature": ":".join(
                    "2" for _ in range(23)
                ),
            }
        )
        record = PROFILE.profile_record(
            source,
            {
                "success": "1",
                "preflight_passed": "1",
                "preflight_mode": "numeric_signed_axes_full_v3",
                "median_ms": "0.25",
                "stddev_ms": "0.001",
            },
            record_id="paired-record",
            run_id="paired-run",
            soc="Ascend910B3",
            aic=20,
            toolkit="8.1.RC1",
        )
        self.assertEqual(
            record["bank_tiling_signature"],
            source["bank_tiling_signature"],
        )

    def test_research_measurement_does_not_report_custom_deployment(
        self,
    ) -> None:
        self.assertEqual(
            PROFILE.deployment_decision(
                "one_shot_research_candidate",
                "improved",
                "improved",
            ),
            "retain_bank_research_candidate_faster",
        )
        self.assertEqual(
            PROFILE.deployment_decision(
                "one_shot_research_candidate"
            ),
            "retain_bank_research_measurement_failed",
        )

    def test_custom_policy_reports_real_custom_deployment(self) -> None:
        self.assertEqual(
            PROFILE.deployment_decision(
                "one_shot_custom_policy",
                "improved",
                "improved",
            ),
            "custom_faster",
        )
        self.assertEqual(
            PROFILE.deployment_decision("one_shot_custom_policy"),
            "custom_measurement_failed",
        )

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
                "preflight_mode": "numeric_ones_full_v2",
                "pair_validated": "1",
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
        self.assertAlmostEqual(
            float(summary["delta_ms_vs_official"]), -0.9
        )
        self.assertAlmostEqual(
            float(summary["delta_pct_vs_official"]), -45.0
        )
        self.assertEqual(
            summary["paired_outcome"],
            "faster_than_official_and_bank",
        )
        self.assertEqual(summary["deployment_decision"], "custom_faster")

    def test_family_summary_reports_one_shot_speedups(self) -> None:
        rows = [
            {
                "geometry_family": "balanced",
                "reduction_family": "regular_k",
                "parallelism_family": "multi_wave",
                "alignment_family": "macro_aligned",
                "layout_family": "NN",
                "dtype": "fp16",
                "optimization_result": "improved",
                "runtime_rejected": "0",
                "speedup_vs_official": "1.10",
                "speedup_vs_bank": "1.05",
            },
            {
                "geometry_family": "balanced",
                "reduction_family": "regular_k",
                "parallelism_family": "multi_wave",
                "alignment_family": "tail",
                "layout_family": "NN",
                "dtype": "fp16",
                "optimization_result": "not_improved",
                "runtime_rejected": "0",
                "speedup_vs_official": "0.90",
                "speedup_vs_bank": "0.95",
            },
        ]
        families = PROFILE.summarize_families(rows)
        balanced = next(
            row
            for row in families
            if row["axis"] == "geometry"
            and row["family"] == "balanced"
        )
        self.assertEqual(balanced["workloads"], "2")
        self.assertEqual(balanced["improved"], "1")
        self.assertEqual(balanced["not_improved"], "1")
        self.assertAlmostEqual(
            float(balanced["geomean_speedup_vs_official"]),
            (1.10 * 0.90) ** 0.5,
        )

    def test_resume_requires_full_numeric_and_paired_validation(self) -> None:
        legacy = {
            "success": "1",
            "preflight_mode": "zero_coverage_grid9_v1",
            "pair_validated": "1",
            "official_ms": "1",
            "bank_ms": "1",
        }
        self.assertFalse(PROFILE.measurement_reusable(legacy))
        verified = dict(legacy)
        verified["preflight_mode"] = "numeric_ones_full_v2"
        self.assertTrue(PROFILE.measurement_reusable(verified))
        structured = dict(legacy)
        structured["preflight_mode"] = "numeric_signed_axes_full_v3"
        self.assertTrue(PROFILE.measurement_reusable(structured))

    def test_unpaired_numeric_result_is_completed_but_not_rankable(
        self,
    ) -> None:
        unpaired = {
            "success": "1",
            "preflight_passed": "1",
            "preflight_mode": "numeric_signed_axes_full_v3",
            "pair_validated": "0",
            "median_ms": "0.25",
            "official_ms": "0.2",
            "bank_ms": "0.2",
        }
        self.assertTrue(PROFILE.measurement_completed(unpaired))
        self.assertFalse(PROFILE.measurement_reusable(unpaired))

    def test_interrupted_provisional_result_can_be_resumed(self) -> None:
        provisional = {
            "success": "1",
            "preflight_passed": "1",
            "preflight_mode": "provisional",
            "pair_validated": "0",
            "median_ms": "0.25",
        }
        self.assertFalse(PROFILE.measurement_completed(provisional))

    def test_baseline_drift_and_conservative_reference(self) -> None:
        before = {"median_ms": "1.0", "stddev_ms": "0.01"}
        after = {"median_ms": "2.0", "stddev_ms": "0.02"}
        self.assertEqual(PROFILE.baseline_drift_pct(before, after), 100.0)
        reference = PROFILE.conservative_reference(before, after)
        self.assertEqual(reference["median_ms"], "1")
        self.assertEqual(reference["stddev_ms"], "0.02")


if __name__ == "__main__":
    unittest.main()
