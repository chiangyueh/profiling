#!/usr/bin/env python3
from __future__ import annotations

import inspect
import sys
from itertools import islice
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import refine_matmul_v3_candidates as legacy
from tiling_search import (
    CandidateEngine,
    GenerationBudget,
    Hardware,
    MeasuredObservation,
    SearchConfig,
    Template,
    Workload,
)
from tiling_search.contracts import validate_schedule
from tiling_search.contracts import template_of
from tiling_search.feedback import (
    feedback_mutations,
    feedback_targets,
    fingerprint,
)
from tiling_search.solvers import (
    Al1FullLoadSolver,
    BaseSolver,
    Bl1FullLoadSolver,
    DeterministicSplitKSolver,
    SingleCoreSplitKSolver,
)


HARDWARE = Hardware(
    aic_cores=20,
    l0a_bytes=64 * 1024,
    l0b_bytes=64 * 1024,
    l0c_bytes=128 * 1024,
    l1_bytes=512 * 1024 - 256,
    l2_bytes=192 * 1024 * 1024,
    l2_bytes_per_cycle_per_core=110.0,
    hbm_bytes_per_cycle_per_core=32.0,
)

SMALL_BUDGET = GenerationBudget(
    raw_attempts=300,
    legal_candidates=180,
    behavior_candidates=40,
    callback_candidates=24,
    npu_candidates=8,
)


def first_valid(
    solver: object,
    workload: Workload,
    expected_template: Template | None = None,
):
    for schedule in islice(
        solver.generate(workload, HARDWARE, ()), 400
    ):
        if (
            validate_schedule(workload, schedule, HARDWARE).valid
            and (
                expected_template is None
                or template_of(schedule) == expected_template
            )
        ):
            return schedule
    raise AssertionError(
        f"{solver.__class__.__name__} generated no valid schedule"
    )


def test_all_solvers_generate_without_old_profitability_gates() -> None:
    cases = (
        (
            BaseSolver(),
            Workload("base", 257, 1009, 4097, "fp16", False, False, 20),
            Template.BASE,
        ),
        (
            SingleCoreSplitKSolver(),
            Workload("split", 64, 96, 512, "fp16", False, False, 20),
            Template.SINGLE_CORE_SPLIT_K,
        ),
        (
            DeterministicSplitKSolver(),
            Workload("det", 64, 80, 1024, "fp16", False, False, 20),
            Template.DETERMINISTIC_SPLIT_K,
        ),
        (
            Al1FullLoadSolver(),
            Workload("al1", 16, 304, 5120, "fp32", False, True, 20),
            Template.AL1_FULL_LOAD,
        ),
        (
            Bl1FullLoadSolver(),
            Workload("bl1", 49152, 160, 128, "fp16", False, False, 20),
            Template.BL1_FULL_LOAD,
        ),
        (
            Bl1FullLoadSolver(),
            Workload("fix", 65536, 7, 16, "fp16", False, False, 20),
            Template.BL1_FULL_LOAD_FIXPIPE,
        ),
        (
            Bl1FullLoadSolver(),
            Workload("vec", 65536, 7, 8, "fp32", False, False, 20),
            Template.BL1_FULL_LOAD_VEC_NZ2ND,
        ),
    )
    for solver, workload, expected in cases:
        schedule = first_valid(solver, workload, expected)
        assert template_of(schedule) == expected
        assert len(schedule.values) == 23


def test_generation_is_name_independent() -> None:
    left = Workload(
        "skinny_n_family_name", 333, 777, 2051, "fp16", False, False, 20
    )
    right = Workload(
        "completely_anonymous", 333, 777, 2051, "fp16", False, False, 20
    )
    solver = BaseSolver()
    left_signatures = {
        schedule.signature()
        for schedule in islice(solver.generate(left, HARDWARE, ()), 80)
    }
    right_signatures = {
        schedule.signature()
        for schedule in islice(solver.generate(right, HARDWARE, ()), 80)
    }
    assert left_signatures == right_signatures


def test_hardware_limits_expose_tiles_beyond_legacy_caps() -> None:
    workload = Workload(
        "large_anonymous", 8192, 8192, 8192, "fp16", False, False, 20
    )
    schedules = islice(
        BaseSolver().generate(workload, HARDWARE, ()), 1000
    )
    assert any(
        max(
            schedule["baseM"],
            schedule["baseN"],
            schedule["baseK"],
        )
        > 512
        and validate_schedule(workload, schedule, HARDWARE).valid
        for schedule in schedules
    )


def test_new_engine_does_not_call_legacy_constructors() -> None:
    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("legacy constructor was called")

    names = (
        "all_template_candidate_spaces",
        "bottleneck_guided_candidate_proposals",
        "general_search_candidate_proposals",
        "split_template_proposals",
    )
    originals = {name: getattr(legacy, name) for name in names}
    try:
        for name in names:
            setattr(legacy, name, forbidden)
        workload = Workload(
            "unknown", 333, 777, 2051, "fp16", False, False, 20
        )
        result = CandidateEngine(
            config=SearchConfig(SMALL_BUDGET)
        ).generate(workload, HARDWARE)
        assert result.candidates
        assert all(
            candidate.source == "contract_global"
            for candidate in result.candidates
        )
    finally:
        for name, value in originals.items():
            setattr(legacy, name, value)


def test_feedback_creates_new_legal_fingerprints() -> None:
    workload = Workload(
        "feedback", 257, 1009, 4097, "fp16", False, False, 20
    )
    parent = first_valid(BaseSolver(), workload)
    observation = MeasuredObservation(
        workload=workload,
        schedule=parent,
        ratio_vs_official=0.90,
        ratio_vs_bank=0.91,
        source="measured",
        record_id="paired-run",
    )
    mutations = feedback_mutations(
        workload, HARDWARE, (observation,)
    )
    assert mutations
    assert all(
        candidate.schedule.signature() != parent.signature()
        for candidate in mutations
    )
    assert all(
        validate_schedule(
            workload, candidate.schedule, HARDWARE
        ).valid
        for candidate in mutations
    )
    assert {
        candidate.source for candidate in mutations
    } == {"feedback_winner_mutation"}
    result = CandidateEngine(
        config=SearchConfig(SMALL_BUDGET),
        observations=(observation,),
        exclusions={fingerprint(workload, parent)},
    ).generate(workload, HARDWARE)
    assert any(
        candidate.source == "feedback_winner_mutation"
        for candidate in result.candidates
    )


def test_budget_and_behavior_coverage_are_bounded() -> None:
    workload = Workload(
        "distant", 333, 777, 2051, "bf16", True, True, 20
    )
    result = CandidateEngine(
        config=SearchConfig(SMALL_BUDGET)
    ).generate(workload, HARDWARE)
    assert 1 <= len(result.candidates) <= SMALL_BUDGET.behavior_candidates
    assert (
        1
        <= len(result.callback_candidates)
        <= SMALL_BUDGET.callback_candidates
    )
    pool_signatures = {
        candidate.schedule.signature()
        for candidate in result.candidates
    }
    assert all(
        candidate.schedule.signature() in pool_signatures
        for candidate in result.callback_candidates
    )
    assert (
        sum(report.raw_generated for report in result.reports)
        <= SMALL_BUDGET.raw_attempts
    )
    assert (
        sum(report.emitted for report in result.reports)
        <= SMALL_BUDGET.legal_candidates
    )
    assert result.behavior_bins > 1
    varied_fields = {
        index
        for index in range(23)
        if len(
            {
                candidate.schedule.values[index]
                for candidate in result.candidates
            }
        )
        > 1
    }
    assert len(varied_fields) >= 10


def test_cross_workload_feedback_changes_generated_frontier() -> None:
    source_workload = Workload(
        "source", 128, 128, 4096, "fp16", False, False, 20
    )
    source_schedule = first_valid(BaseSolver(), source_workload)
    observation = MeasuredObservation(
        workload=source_workload,
        schedule=source_schedule,
        ratio_vs_official=0.85,
        ratio_vs_bank=0.86,
        source="measured",
        record_id="source-paired-run",
    )
    target = Workload(
        "unseen-target", 333, 777, 2051, "fp16", False, False, 20
    )
    targets = feedback_targets(target, HARDWARE, (observation,))
    baseline_generated = {
        schedule.signature()
        for schedule in islice(
            BaseSolver().generate(target, HARDWARE, ()), 100
        )
    }
    informed_generated = {
        schedule.signature()
        for schedule in islice(
            BaseSolver().generate(target, HARDWARE, targets), 100
        )
    }
    assert informed_generated.difference(baseline_generated)
    baseline = CandidateEngine(
        config=SearchConfig(SMALL_BUDGET)
    ).generate(target, HARDWARE)
    informed = CandidateEngine(
        config=SearchConfig(SMALL_BUDGET),
        observations=(observation,),
    ).generate(target, HARDWARE)
    baseline_signatures = {
        candidate.schedule.signature()
        for candidate in baseline.candidates
    }
    informed_signatures = {
        candidate.schedule.signature()
        for candidate in informed.candidates
    }
    assert baseline_signatures != informed_signatures


def test_new_package_has_no_legacy_import() -> None:
    import tiling_search
    from tiling_search import orchestrator

    source = inspect.getsource(orchestrator)
    assert "refine_matmul_v3_candidates" not in source
    assert "all_template_candidate_spaces" not in source
    assert Path(tiling_search.__file__).parent.name == "tiling_search"


def main() -> None:
    test_all_solvers_generate_without_old_profitability_gates()
    test_generation_is_name_independent()
    test_hardware_limits_expose_tiles_beyond_legacy_caps()
    test_new_engine_does_not_call_legacy_constructors()
    test_feedback_creates_new_legal_fingerprints()
    test_budget_and_behavior_coverage_are_bounded()
    test_cross_workload_feedback_changes_generated_frontier()
    test_new_package_has_no_legacy_import()
    print("contract_behavior_test passed")


if __name__ == "__main__":
    main()
