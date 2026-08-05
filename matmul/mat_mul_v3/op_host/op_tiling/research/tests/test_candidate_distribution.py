from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


RESEARCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH))

from audit_candidate_distribution import (
    audit_rows,
    coverage_violations,
    group_coverage,
)


def candidate_row(index: int, *, clustered: bool = False) -> dict[str, str]:
    spread = 0 if clustered else index
    metrics = {
        "active_cores": float(4 + index),
        "bank_behavior_distance": 1.0 + index / 10.0,
        "bank_changed_fields": 10.0,
        "calibration_target_quota": 8.0,
    }
    return {
        "workload_id": "distribution_probe",
        "candidate_role": "searched",
        "search_template": "AL1_FULL_LOAD",
        "search_behavior_key": json.dumps(["AL1", spread % 4]),
        "search_behavior_metrics": json.dumps(metrics),
        "tiling_signature": ":".join(
            str(value)
            for value in (
                index + 1,
                64 + 16 * spread,
                64 + 16 * (spread % 4),
                1024,
                16 + 16 * (spread % 4),
                16 + 16 * (spread % 4),
                16,
                1 + spread,
                1 + spread,
                1,
                1,
                0,
                1,
                1,
                2,
                2,
                1,
                1,
                1,
                1,
                1 + spread,
                spread % 2,
                10,
            )
        ),
        "base_m": str(16 + 16 * (spread % 4)),
        "base_n": str(16 + 16 * (spread % 4)),
        "base_k": "16",
        "depth_a1": str(1 + spread),
        "depth_b1": str(1 + spread),
        "step_m": "1",
        "step_n": "1",
        "step_ka": "1",
        "step_kb": "1",
        "db_l0a": "2",
        "db_l0b": "2",
        "db_l0c": "1",
        "iterate_order": str(spread % 2),
        "used_core_num": str(index + 1),
        "single_core_m": str(64 + 16 * spread),
        "single_core_n": str(64 + 16 * (spread % 4)),
        "single_core_k": "1024",
        "l2_m_tile_count": "1",
        "l2_n_tile_count": "1",
        "l2_m_tile_block": "1",
        "l2_n_tile_block": str(1 + spread),
        "l2_iterate_order": str(spread % 2),
    }


class CandidateDistributionTest(unittest.TestCase):
    def test_broad_hardware_behavior_distribution_passes(self) -> None:
        rows = [candidate_row(index) for index in range(8)]
        coverage = group_coverage(rows)
        self.assertEqual(coverage_violations(coverage), [])
        _, violations = audit_rows(rows)
        self.assertEqual(violations, [])

    def test_clustered_candidates_are_rejected(self) -> None:
        rows = [
            candidate_row(index, clustered=True) for index in range(8)
        ]
        violations = coverage_violations(group_coverage(rows))
        self.assertTrue(
            any("behavior_bins" in violation for violation in violations)
        )

    def test_duplicate_signature_is_rejected(self) -> None:
        rows = [candidate_row(index) for index in range(8)]
        rows[1]["tiling_signature"] = rows[0]["tiling_signature"]
        _, violations = audit_rows(rows)
        self.assertIn("duplicate_signatures=1", violations)


if __name__ == "__main__":
    unittest.main()
