from __future__ import annotations

import json
import math
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


RESEARCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH))

from tiling_search import (
    CandidateEngine,
    DIRECT_BASE_AUDIT_BANK_WORKLOADS,
    DIRECT_BASE_AUDIT_PAIRED_RECORDS,
    DIRECT_BASE_AUDIT_UNIQUE_RECORDS,
    DIRECT_BASE_AUDIT_WINNER_WORKLOADS,
    DIRECT_BASE_AUDIT_WORKLOADS,
    DIRECT_BASE_L2_RESIDENT_RATIO,
    DIRECT_RULE_AUDIT_RECORDS,
    DIRECT_RULE_AUDIT_UNIQUE_RECORDS,
    DIRECT_RULE_AUDIT_WORKLOADS,
    DIRECT_RULE_TEMPLATE_AUDIT,
    DIRECT_RULE_TRUSTED_WINNER_EXECUTION_EQUIVALENT,
    DIRECT_RULE_TRUSTED_WINNER_STRUCTURAL_NEAR,
    DIRECT_RULE_TRUSTED_WINNER_TEMPLATE_MATCHES,
    DIRECT_RULE_TRUSTED_WINNER_WORKLOADS,
    GenerationBudget,
    Hardware,
    KNOWLEDGE_FIELDS,
    SearchConfig,
    Template,
    Workload,
    direct_base_candidate,
    direct_rule_candidate,
)
from generate import load_resume_feedback
from tiling_search.contracts import ceil_div, template_of, validate_schedule
from tiling_search.domain import INPUT_BYTES, OUTPUT_BYTES
from tiling_search.bank_structure import (
    bank_transition,
    schedules_execution_equivalent,
)
from tiling_search.feedback import load_feedback


class DeploymentStrategiesTest(unittest.TestCase):
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
            ub_bytes=262144,
        )

    def test_direct_base_is_one_coupled_legal_record(self) -> None:
        workload = Workload(
            workload_id="direct_dense",
            m=3584,
            n=4352,
            k=6656,
            dtype="fp16",
            trans_a=False,
            trans_b=False,
            max_cores=20,
        )
        candidate = direct_base_candidate(workload, self.hardware)

        self.assertEqual(candidate.source, "direct_base_policy")
        self.assertEqual(candidate.template, Template.BASE)
        self.assertEqual(candidate.metrics["model_enabled"], 0.0)
        self.assertEqual(candidate.metrics["history_rows_used"], 0.0)
        self.assertEqual(candidate.metrics["candidate_pool_size"], 1.0)
        self.assertNotIn(
            "predicted_latency_ratio",
            candidate.metrics,
        )
        self.assertTrue(
            validate_schedule(
                workload, candidate.schedule, self.hardware
            ).valid
        )
        self.assertEqual(
            (
                candidate.schedule["baseM"],
                candidate.schedule["baseN"],
                candidate.schedule["baseK"],
                candidate.schedule["depthA1"],
                candidate.schedule["depthB1"],
                candidate.schedule["stepKa"],
                candidate.schedule["stepKb"],
            ),
            (128, 256, 64, 16, 8, 8, 4),
        )
        self.assertEqual(
            (
                candidate.schedule["l2MTileCnt"],
                candidate.schedule["l2NTileCnt"],
                candidate.schedule["l2MTileBlock"],
                candidate.schedule["l2NTileBlock"],
            ),
            (2, 1, 14, 17),
        )
        self.assertEqual(
            candidate.metrics["policy_l2_mode_cache"], 1.0
        )

    def test_direct_base_keeps_small_working_set_resident(self) -> None:
        workload = Workload(
            workload_id="direct_small",
            m=512,
            n=512,
            k=512,
            dtype="fp16",
            trans_a=False,
            trans_b=False,
            max_cores=20,
        )
        candidate = direct_base_candidate(workload, self.hardware)

        self.assertEqual(
            (
                candidate.schedule["l2MTileCnt"],
                candidate.schedule["l2NTileCnt"],
                candidate.schedule["l2MTileBlock"],
                candidate.schedule["l2NTileBlock"],
            ),
            (1, 1, 5, 4),
        )
        self.assertEqual(
            candidate.metrics["policy_l2_mode_whole"], 1.0
        )

    def test_offline_audit_supports_fixed_direct_rules(
        self,
    ) -> None:
        names = (
            "paired_measurements_net_log25_26.csv",
            "paired_measurements_net_log27.csv",
            "paired_measurements_net_log28.csv",
            "paired_measurements_net_log30.csv",
            "paired_measurements_net_log31.csv",
            "paired_measurements_net_log33.csv",
        )
        config = RESEARCH / "config"
        observations = []
        for name in names:
            loaded, _ = load_resume_feedback(
                config / name,
                "Ascend910B3",
                20,
                "8.1.RC1+toolkit-7.7.0.1.225",
            )
            observations.extend(loaded)

        paired = [
            observation
            for observation in observations
            if observation.verified
            and observation.source
            not in {"runtime_rejected", "runtime_verified"}
            and template_of(observation.schedule) == Template.BASE
        ]
        grouped = defaultdict(list)
        for observation in paired:
            grouped[
                (
                    observation.workload.identity(),
                    observation.schedule.signature(),
                )
            ].append(observation.measured_ratio)
        best_by_workload = {}
        for key, ratios in grouped.items():
            workload_identity = key[0]
            best_by_workload[workload_identity] = min(
                median(ratios),
                best_by_workload.get(
                    workload_identity, float("inf")
                ),
            )

        self.assertEqual(
            len(paired), DIRECT_BASE_AUDIT_PAIRED_RECORDS
        )
        self.assertEqual(
            len(grouped), DIRECT_BASE_AUDIT_UNIQUE_RECORDS
        )
        self.assertEqual(
            len(best_by_workload), DIRECT_BASE_AUDIT_WORKLOADS
        )
        self.assertEqual(
            sum(ratio < 0.99 for ratio in best_by_workload.values()),
            DIRECT_BASE_AUDIT_WINNER_WORKLOADS,
        )

        banks = {}
        for observation in observations:
            bank = observation.bank_schedule
            if (
                observation.verified
                and bank is not None
                and template_of(bank) == Template.BASE
            ):
                banks[observation.workload.identity()] = (
                    observation.workload,
                    bank,
                )
        self.assertEqual(
            len(banks), DIRECT_BASE_AUDIT_BANK_WORKLOADS
        )

        whole_sizes = []
        partitioned_sizes = []
        for workload, bank in banks.values():
            m_tasks = ceil_div(
                workload.m, bank["singleCoreM"]
            )
            n_tasks = ceil_div(
                workload.n, bank["singleCoreN"]
            )
            total_bytes = (
                workload.m
                * workload.k
                * INPUT_BYTES[workload.dtype]
                + workload.k
                * workload.n
                * INPUT_BYTES[workload.dtype]
                + workload.m
                * workload.n
                * OUTPUT_BYTES[workload.dtype]
            )
            if (
                bank["l2MTileCnt"],
                bank["l2NTileCnt"],
                bank["l2MTileBlock"],
                bank["l2NTileBlock"],
            ) == (1, 1, m_tasks, n_tasks):
                whole_sizes.append(total_bytes)
            else:
                partitioned_sizes.append(total_bytes)

            candidate = direct_base_candidate(
                workload,
                self.hardware,
            )
            self.assertEqual(
                candidate.schedule.signature(),
                bank.signature(),
                workload.workload_id,
            )

        self.assertEqual(len(whole_sizes), 32)
        self.assertEqual(len(partitioned_sizes), 16)
        threshold = math.sqrt(
            max(whole_sizes) * min(partitioned_sizes)
        )
        self.assertAlmostEqual(
            threshold / self.hardware.l2_bytes,
            DIRECT_BASE_L2_RESIDENT_RATIO,
            places=12,
        )

    def test_complete_saved_history_matches_direct_rule_audit(
        self,
    ) -> None:
        config = RESEARCH / "config"
        observations, _ = load_feedback(
            soc="Ascend910B3",
            aic_cores=20,
            observation_paths=sorted(
                config.glob("measured_observations*.csv")
            ),
            exclusion_paths=sorted(
                config.glob("measured_fingerprints*.csv")
            ),
        )
        for path in sorted(config.glob("paired_measurements*.csv")):
            loaded, _ = load_resume_feedback(
                path,
                "Ascend910B3",
                20,
                "8.1.RC1+toolkit-7.7.0.1.225",
            )
            observations.extend(loaded)

        self.assertEqual(len(observations), DIRECT_RULE_AUDIT_RECORDS)
        self.assertEqual(
            len(
                {
                    (
                        observation.workload.identity(),
                        observation.schedule.signature(),
                    )
                    for observation in observations
                }
            ),
            DIRECT_RULE_AUDIT_UNIQUE_RECORDS,
        )
        self.assertEqual(
            len(
                {
                    observation.workload.identity()
                    for observation in observations
                }
            ),
            DIRECT_RULE_AUDIT_WORKLOADS,
        )

        preferred = {}
        for observation in observations:
            key = (
                observation.workload.identity(),
                observation.schedule.signature(),
            )
            quality = (
                int(observation.structured_verified),
                int(observation.verified),
                int(
                    observation.source
                    not in {"runtime_rejected", "runtime_verified"}
                ),
                observation.record_id,
            )
            if key not in preferred or quality > preferred[key][0]:
                preferred[key] = (quality, observation)

        by_workload_template = defaultdict(list)
        for _, observation in preferred.values():
            if (
                not observation.verified
                or observation.source
                in {"runtime_rejected", "runtime_verified"}
            ):
                continue
            by_workload_template[
                (
                    observation.workload.identity(),
                    template_of(observation.schedule),
                )
            ].append(observation.measured_ratio)

        actual = Counter()
        for (_, template), ratios in by_workload_template.items():
            best = min(ratios)
            if best < 0.99:
                outcome = "winner"
            elif best > 1.01:
                outcome = "regression"
            else:
                outcome = "within_noise"
            actual[(template, outcome)] += 1

        for template, expected in DIRECT_RULE_TEMPLATE_AUDIT.items():
            workloads, winners, within_noise, regressions = expected
            self.assertEqual(
                (
                    sum(
                        actual[(template, outcome)]
                        for outcome in (
                            "winner",
                            "within_noise",
                            "regression",
                        )
                    ),
                    actual[(template, "winner")],
                    actual[(template, "within_noise")],
                    actual[(template, "regression")],
                ),
                (workloads, winners, within_noise, regressions),
                template.value,
            )

    def test_direct_rule_reconstructs_trusted_historical_winners(
        self,
    ) -> None:
        config = RESEARCH / "config"
        observations, _ = load_feedback(
            soc="Ascend910B3",
            aic_cores=20,
            observation_paths=sorted(
                config.glob("measured_observations*.csv")
            ),
            exclusion_paths=(),
        )
        for path in sorted(config.glob("paired_measurements*.csv")):
            loaded, _ = load_resume_feedback(
                path,
                "Ascend910B3",
                20,
                "8.1.RC1+toolkit-7.7.0.1.225",
            )
            observations.extend(loaded)

        winners = defaultdict(list)
        for observation in observations:
            if (
                observation.is_verified_winner
                and observation.measured_ratio < 0.99
            ):
                winners[observation.workload.identity()].append(
                    observation
                )

        template_matches = 0
        execution_equivalent = 0
        structural_near = 0
        for workload_winners in winners.values():
            winner = min(
                workload_winners,
                key=lambda observation: observation.measured_ratio,
            )
            candidate = direct_rule_candidate(
                winner.workload, self.hardware
            )
            template_matches += (
                candidate.template == template_of(winner.schedule)
            )
            execution_equivalent += schedules_execution_equivalent(
                candidate.schedule, winner.schedule
            )
            transition = bank_transition(
                winner.schedule, candidate.schedule
            )
            structural_near += (
                candidate.template == template_of(winner.schedule)
                and len(transition.changed_subsystems) <= 2
            )

        self.assertEqual(
            len(winners), DIRECT_RULE_TRUSTED_WINNER_WORKLOADS
        )
        self.assertEqual(
            template_matches,
            DIRECT_RULE_TRUSTED_WINNER_TEMPLATE_MATCHES,
        )
        self.assertEqual(
            execution_equivalent,
            DIRECT_RULE_TRUSTED_WINNER_EXECUTION_EQUIVALENT,
        )
        self.assertEqual(
            structural_near,
            DIRECT_RULE_TRUSTED_WINNER_STRUCTURAL_NEAR,
        )

    def test_runtime_kb_contains_exact_records_not_rule_ranges(
        self,
    ) -> None:
        runtime_kb = (
            RESEARCH.parents[2]
            / "op_host/config/ascend910b"
            / "Ascend910B3_20_AiCore_MatMulV3_runtime_kb.json"
        )
        records = [
            json.loads(line)
            for line in runtime_kb.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(records), 13)
        self.assertEqual(
            Counter(
                record["knowledge"]["tilingEnable"]
                for record in records
            ),
            Counter({2: 7, 3: 5, 0: 1}),
        )
        for record in records:
            self.assertEqual(
                tuple(record["knowledge"]),
                KNOWLEDGE_FIELDS,
            )
            info = record["info_dict"]
            for field in (
                "m",
                "n",
                "k",
                "a_dtype",
                "b_dtype",
                "a_format",
                "b_format",
                "trans_a_flag",
                "trans_b_flag",
            ):
                self.assertIn(field, info)
                self.assertNotIsInstance(info[field], (list, dict))

    def test_upstream_template_order_is_audited_from_source(
        self,
    ) -> None:
        source = (
            RESEARCH.parents[2]
            / "op_host/op_tiling/matmul_v3_base_tiling.cpp"
        ).read_text(encoding="utf-8")
        begin = source.index("void MatmulV3BaseTiling::DoSelectTiling()")
        end = source.index(
            "void MatmulV3BaseTiling::FormulateBasicBlockDavid()",
            begin,
        )
        body = source[begin:end]
        ordered = (
            "DoBL1FullloadWithFixpipeTiling()",
            "DoAL1FullLoadTiling()",
            "DoBL1FullLoadTiling()",
            "DoL2CacheTiling()",
            "DoSingleCoreSplitKTiling()",
            "DoDeterministicMultiCoreSplitKTiling()",
        )
        positions = tuple(body.index(call) for call in ordered)
        self.assertEqual(positions, tuple(sorted(positions)))

    def test_direct_base_covers_dtype_and_layout_variants(self) -> None:
        cases = (
            ("bf16_nn", 1791, 2433, 4609, "bf16", False, False),
            ("fp32_nt", 144, 768, 4864, "fp32", False, True),
            ("fp32_tn", 1152, 1664, 2304, "fp32", True, False),
            ("fp32_tt", 896, 1280, 1792, "fp32", True, True),
            ("fp16_shallow_k", 160, 255, 31, "fp16", False, False),
            ("fp32_nt_shallow_k", 80, 129, 16, "fp32", False, True),
        )
        for case in cases:
            with self.subTest(case=case[0]):
                workload = Workload(*case, max_cores=20)
                candidate = direct_base_candidate(
                    workload, self.hardware
                )
                self.assertTrue(
                    validate_schedule(
                        workload, candidate.schedule, self.hardware
                    ).valid
                )

    def test_direct_rule_uses_single_split_for_dense_high_k(self) -> None:
        workload = Workload(
            workload_id="dense_deep_k",
            m=2368,
            n=2880,
            k=31744,
            dtype="fp16",
            trans_a=False,
            trans_b=False,
            max_cores=20,
        )
        candidate = direct_rule_candidate(workload, self.hardware)

        self.assertEqual(candidate.source, "direct_rule_policy")
        self.assertEqual(
            candidate.template, Template.SINGLE_CORE_SPLIT_K
        )
        self.assertEqual(candidate.metrics["model_enabled"], 0.0)
        self.assertEqual(candidate.metrics["candidate_pool_size"], 1.0)

    def test_direct_rule_rejects_reverse_k1536_split_after_net_log33(
        self,
    ) -> None:
        workload = Workload(
            workload_id="single_k_reverse",
            m=53248,
            n=384,
            k=1536,
            dtype="fp16",
            trans_a=False,
            trans_b=False,
            max_cores=20,
        )
        candidate = direct_rule_candidate(workload, self.hardware)

        self.assertEqual(candidate.template, Template.BASE)

    def test_direct_rule_ports_bl1_n512_core_split_variant(
        self,
    ) -> None:
        workload = Workload(
            workload_id="bl1_core_split",
            m=49408,
            n=512,
            k=512,
            dtype="fp16",
            trans_a=False,
            trans_b=True,
            max_cores=20,
        )
        candidate = direct_rule_candidate(workload, self.hardware)

        self.assertEqual(candidate.template, Template.BL1_FULL_LOAD)
        self.assertIn("bl1_n512_core_split", candidate.rationale)
        self.assertEqual(
            (
                candidate.schedule["usedCoreNum"],
                candidate.schedule["singleCoreM"],
                candidate.schedule["singleCoreN"],
                candidate.schedule["baseM"],
                candidate.schedule["baseN"],
            ),
            (20, 4224, 256, 128, 256),
        )
        self.assertTrue(
            validate_schedule(
                workload, candidate.schedule, self.hardware
            ).valid
        )

    def test_direct_rule_ports_fp32_bl1_vnchw_variant(
        self,
    ) -> None:
        workload = Workload(
            workload_id="bl1_vnchw",
            m=73728,
            n=16,
            k=8,
            dtype="fp32",
            trans_a=False,
            trans_b=False,
            max_cores=20,
        )
        candidate = direct_rule_candidate(workload, self.hardware)

        self.assertEqual(candidate.template, Template.BL1_FULL_LOAD)
        self.assertEqual(
            candidate.schedule.signature_text(),
            "20:256:16:8:128:16:32:2:1:1:1:0:"
            "1:1:2:2:2:1:1:288:1:1:20",
        )
        self.assertTrue(
            validate_schedule(
                workload, candidate.schedule, self.hardware
            ).valid
        )

    def test_direct_rule_matches_callback_split_for_net_log33(
        self,
    ) -> None:
        workload = Workload(
            workload_id="oneshot_v5_08",
            m=512,
            n=320,
            k=30720,
            dtype="fp16",
            trans_a=False,
            trans_b=False,
            max_cores=20,
        )
        candidate = direct_rule_candidate(workload, self.hardware)

        self.assertEqual(candidate.source, "direct_rule_policy")
        self.assertEqual(
            candidate.template, Template.DETERMINISTIC_SPLIT_K
        )
        self.assertEqual(
            candidate.schedule.signature_text(),
            "20:512:384:384:128:128:128:6:9:1:3:0:"
            "3:3:2:2:2:1:1:1:1:0:3",
        )
        self.assertTrue(
            validate_schedule(
                workload, candidate.schedule, self.hardware
            ).valid
        )

    def test_direct_rule_covers_every_upstream_template_regime(
        self,
    ) -> None:
        cases = (
            (
                Template.AL1_FULL_LOAD,
                Workload(
                    "al1",
                    16,
                    272,
                    6144,
                    "fp32",
                    False,
                    True,
                    20,
                ),
            ),
            (
                Template.BL1_FULL_LOAD_FIXPIPE,
                Workload(
                    "bl1_fix",
                    53248,
                    176,
                    192,
                    "fp16",
                    False,
                    False,
                    20,
                ),
            ),
            (
                Template.BL1_FULL_LOAD_VEC_NZ2ND,
                Workload(
                    "bl1_fix_fp32",
                    16384,
                    48,
                    128,
                    "fp32",
                    False,
                    False,
                    20,
                ),
            ),
            (
                Template.BL1_FULL_LOAD,
                Workload(
                    "bl1",
                    53248,
                    128,
                    192,
                    "fp16",
                    False,
                    False,
                    20,
                ),
            ),
            (
                Template.SINGLE_CORE_SPLIT_K,
                Workload(
                    "single_k",
                    2368,
                    2880,
                    31744,
                    "fp16",
                    False,
                    False,
                    20,
                ),
            ),
            (
                Template.DETERMINISTIC_SPLIT_K,
                Workload(
                    "det_k",
                    192,
                    128,
                    24576,
                    "fp16",
                    False,
                    False,
                    20,
                ),
            ),
            (
                Template.BASE,
                Workload(
                    "base",
                    2048,
                    2560,
                    4096,
                    "fp16",
                    False,
                    False,
                    20,
                ),
            ),
        )
        for expected, workload in cases:
            with self.subTest(template=expected.value):
                candidate = direct_rule_candidate(
                    workload, self.hardware
                )
                self.assertEqual(candidate.template, expected)
                self.assertTrue(
                    validate_schedule(
                        workload,
                        candidate.schedule,
                        self.hardware,
                    ).valid
                )

    def test_direct_rule_uses_measured_vec_geometry(self) -> None:
        workload = Workload(
            "vec_net_log33",
            16384,
            48,
            128,
            "fp32",
            False,
            False,
            20,
        )
        candidate = direct_rule_candidate(workload, self.hardware)

        self.assertEqual(
            candidate.template, Template.BL1_FULL_LOAD_VEC_NZ2ND
        )
        self.assertEqual(
            candidate.schedule.signature_text(),
            "20:256:48:128:256:48:32:4:4:1:1:0:"
            "4:4:2:2:1:1:1:0:0:0:2020",
        )

    def test_direct_rule_reconstructs_reduction_dominated_winner(self) -> None:
        workload = Workload(
            "unseen_name",
            1024,
            1536,
            32768,
            "fp16",
            False,
            False,
            20,
        )
        candidate = direct_rule_candidate(workload, self.hardware)

        self.assertEqual(
            candidate.template, Template.DETERMINISTIC_SPLIT_K
        )
        self.assertEqual(
            candidate.schedule.signature_text(),
            "20:1024:384:384:128:128:128:6:9:1:3:0:"
            "3:3:2:2:2:1:1:1:4:0:3",
        )

    def test_direct_rule_reconstructs_low_output_split_winners(self) -> None:
        cases = (
            (
                Workload(
                    "first",
                    320,
                    256,
                    28672,
                    "fp16",
                    False,
                    False,
                    20,
                ),
                20,
                320,
            ),
            (
                Workload(
                    "second",
                    192,
                    128,
                    16384,
                    "fp16",
                    False,
                    False,
                    20,
                ),
                8,
                192,
            ),
        )
        for workload, cores, single_m in cases:
            with self.subTest(workload=workload.workload_id):
                candidate = direct_rule_candidate(
                    workload, self.hardware
                )
                self.assertEqual(
                    candidate.template,
                    Template.DETERMINISTIC_SPLIT_K,
                )
                self.assertEqual(
                    candidate.schedule["usedCoreNum"], cores
                )
                self.assertEqual(
                    candidate.schedule["singleCoreM"], single_m
                )
                self.assertEqual(
                    candidate.schedule["singleCoreN"], 384
                )
                self.assertEqual(candidate.schedule["stepM"], 1)
                self.assertEqual(candidate.schedule["stepN"], 3)

    def test_direct_rule_uses_base_when_k_cannot_be_split(self) -> None:
        for k in (384, 448, 512):
            with self.subTest(k=k):
                workload = Workload(
                    "low_k",
                    2688,
                    2944,
                    k,
                    "fp16",
                    False,
                    False,
                    20,
                )
                candidate = direct_rule_candidate(
                    workload, self.hardware
                )
                self.assertEqual(candidate.template, Template.BASE)

    def test_direct_rule_rejects_single_split_for_tt_layout(self) -> None:
        workload = Workload(
            "tt_net_log33",
            2688,
            3328,
            5376,
            "fp16",
            True,
            True,
            20,
        )
        candidate = direct_rule_candidate(workload, self.hardware)

        self.assertEqual(candidate.template, Template.BASE)

    def test_compact_frontier_is_bounded_and_template_aware(self) -> None:
        workload = Workload(
            workload_id="compact_deep_k",
            m=640,
            n=448,
            k=28672,
            dtype="fp16",
            trans_a=False,
            trans_b=False,
            max_cores=20,
        )
        engine = CandidateEngine(
            config=SearchConfig(
                budget=GenerationBudget(
                    raw_attempts=48,
                    legal_candidates=12,
                    behavior_candidates=12,
                    callback_candidates=8,
                    npu_candidates=1,
                ),
                include_exploration=False,
            ),
            observations=(),
            exclusions=set(),
        )
        result = engine.generate(workload, self.hardware)
        templates = Counter(
            candidate.template for candidate in result.callback_candidates
        )

        self.assertLessEqual(
            sum(report.raw_generated for report in result.reports), 48
        )
        self.assertLessEqual(result.legal_candidates, 12)
        self.assertLessEqual(len(result.callback_candidates), 8)
        self.assertIn(Template.BASE, templates)
        self.assertIn(Template.SINGLE_CORE_SPLIT_K, templates)
        self.assertIn(Template.DETERMINISTIC_SPLIT_K, templates)


if __name__ == "__main__":
    unittest.main()
