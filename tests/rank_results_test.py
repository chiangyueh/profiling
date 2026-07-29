#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from rank_npu_results import (
    is_official_operator_baseline,
    optimization_decision,
)


def main() -> None:
    result, reason, verdict = optimization_decision("improved", "improved")
    assert result == "improved"
    assert verdict == "improved"
    assert "official_and_bank_control" in reason

    result, _, verdict = optimization_decision("improved", "within_noise")
    assert result == "not_improved"
    assert verdict == "within_noise"

    result, _, verdict = optimization_decision("improved", "regressed")
    assert result == "not_improved"
    assert verdict == "regressed"

    result, _, verdict = optimization_decision(
        "improved", "bank_seed_unavailable"
    )
    assert result == "control_unavailable"
    assert verdict == "bank_seed_unavailable"

    assert is_official_operator_baseline(
        {
            "source": "installed_aclnn_matmul",
            "candidate_role": "official_operator_baseline",
        }
    )
    assert not is_official_operator_baseline(
        {
            "source": "historical_installed_aclnn_matmul",
            "candidate_role": "official_operator_baseline",
        }
    )

    print("rank_results_test passed")


if __name__ == "__main__":
    main()
