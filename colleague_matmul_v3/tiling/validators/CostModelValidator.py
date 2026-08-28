from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from tiling.base.Base import BaseParam


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _align_up(value: int, alignment: int) -> int:
    return _ceil_div(value, alignment) * alignment


@dataclass(frozen=True)
class FilterVerdict:
    predicted: str
    rules: tuple[str, ...]
    tiling: dict[str, int]

    @property
    def valid(self) -> bool:
        return self.predicted == "yes"


class CostModelValidator:
    """The BASE-template hard gate used by the hardware cost model.

    This validator is deliberately independent of measured latency and output.
    It predicts legality before the candidate is launched, so its prediction can
    subsequently be audited against the full numeric result.
    """

    def __init__(
        self,
        *,
        aic_cores: int = 20,
        l0a_bytes: int = 64 * 1024,
        l0b_bytes: int = 64 * 1024,
        l0c_bytes: int = 128 * 1024,
        l1_bytes: int = 512 * 1024,
        input_bytes: int = 2,
    ) -> None:
        self.aic_cores = aic_cores
        self.l0a_bytes = l0a_bytes
        self.l0b_bytes = l0b_bytes
        self.l0c_bytes = l0c_bytes
        self.l1_bytes = l1_bytes
        self.input_bytes = input_bytes

    @staticmethod
    def _params_dict(params: Iterable[BaseParam]) -> dict[str, int]:
        return {param.name: int(param.value) for param in params}

    def materialize_base_tiling(
        self,
        params: Iterable[BaseParam],
    ) -> dict[str, int]:
        p = self._params_dict(params)
        m = p["MM_M"]
        n = p["MM_N"]
        k = p["MM_K"]
        single_m = p["MM_SINGLE_M"]
        single_n = p["MM_SINGLE_N"]
        m_total = _ceil_div(m, single_m)
        n_total = _ceil_div(n, single_n)
        logical_blocks = m_total * n_total
        l2_m_block = p.get("MM_L2_M_TILE_BLOCK", m_total)
        l2_n_block = p.get("MM_L2_N_TILE_BLOCK", n_total)
        return {
            "M": m,
            "N": n,
            "K": k,
            "usedCoreNum": int(p.get("MM_USED_CORE_NUM", min(logical_blocks, self.aic_cores))),
            "singleCoreM": single_m,
            "singleCoreN": single_n,
            "singleCoreK": k,
            "baseM": p["MM_BASE_M"],
            "baseN": p["MM_BASE_N"],
            "baseK": p["MM_BASE_K"],
            "depthA1": p["MM_STEP_M"] * p["MM_STEP_Ka"] * 2,
            "depthB1": p["MM_STEP_N"] * p["MM_STEP_Kb"] * 2,
            "stepM": p["MM_STEP_M"],
            "stepN": p["MM_STEP_N"],
            "stepKa": p["MM_STEP_Ka"],
            "stepKb": p["MM_STEP_Kb"],
            "iterateOrder": p.get("MM_ITER_ORDER", 0),
            "dbL0A": p.get("MM_DB_L0A", 2),
            "dbL0B": p.get("MM_DB_L0B", 2),
            "dbL0C": p.get("MM_DB_L0C", 2),
            # These defaults exactly match the direct-injection runner.
            "l2MTileCnt": p.get(
                "MM_L2_M_TILE_CNT",
                _ceil_div(m_total, l2_m_block) if l2_m_block else 1,
            ),
            "l2NTileCnt": p.get(
                "MM_L2_N_TILE_CNT",
                _ceil_div(n_total, l2_n_block) if l2_n_block else 1,
            ),
            "l2MTileBlock": l2_m_block,
            "l2NTileBlock": l2_n_block,
            "l2IterateOrder": p.get("MM_L2_ITERATE_ORDER", 0),
            "tilingEnable": 0,
        }

    def classify(self, params: Iterable[BaseParam]) -> FilterVerdict:
        t = self.materialize_base_tiling(params)
        rules: list[str] = []

        positive = (
            "M", "N", "K", "usedCoreNum", "singleCoreM", "singleCoreN",
            "singleCoreK", "baseM", "baseN", "baseK", "depthA1",
            "depthB1", "stepM", "stepN", "stepKa", "stepKb", "dbL0A",
            "dbL0B", "dbL0C", "l2MTileCnt", "l2NTileCnt",
        )
        if any(t[name] <= 0 for name in positive):
            rules.append("NON_POSITIVE_REQUIRED_FIELD")
        if t["l2MTileBlock"] < 0 or t["l2NTileBlock"] < 0:
            rules.append("NEGATIVE_L2_TILE_BLOCK")
        if t["usedCoreNum"] > self.aic_cores:
            rules.append("USED_CORE_NUM_EXCEEDS_AIC_COUNT")
        if t["iterateOrder"] not in (0, 1):
            rules.append("ITERATE_ORDER_UNSUPPORTED")
        if t["l2IterateOrder"] not in (0, 1, 2):
            rules.append("L2_ITERATE_ORDER_UNSUPPORTED")
        for name in ("dbL0A", "dbL0B", "dbL0C"):
            if t[name] not in (1, 2):
                rules.append(f"{name.upper()}_UNSUPPORTED")

        if t["baseM"] % 16:
            rules.append("BASE_M_NOT_16_ALIGNED")
        if t["baseN"] % 16:
            rules.append("BASE_N_NOT_16_ALIGNED")
        if t["baseK"] % 16:
            rules.append("BASE_K_NOT_16_ALIGNED")

        if t["baseM"] * t["baseK"] * self.input_bytes * t["dbL0A"] > self.l0a_bytes:
            rules.append("L0A_CAPACITY_EXCEEDED")
        if t["baseN"] * t["baseK"] * self.input_bytes * t["dbL0B"] > self.l0b_bytes:
            rules.append("L0B_CAPACITY_EXCEEDED")
        if t["baseM"] * t["baseN"] * 4 * t["dbL0C"] > self.l0c_bytes:
            rules.append("L0C_CAPACITY_EXCEEDED")

        one_a = t["stepM"] * t["stepKa"]
        one_b = t["stepN"] * t["stepKb"]
        if one_a <= 0 or t["depthA1"] % one_a:
            rules.append("DEPTH_A1_PACKET_MISMATCH")
        elif t["depthA1"] // one_a not in (1, 2):
            rules.append("DEPTH_A1_BUFFERING_UNSUPPORTED")
        if one_b <= 0 or t["depthB1"] % one_b:
            rules.append("DEPTH_B1_PACKET_MISMATCH")
        elif t["depthB1"] // one_b not in (1, 2):
            rules.append("DEPTH_B1_BUFFERING_UNSUPPORTED")

        a1 = t["baseM"] * t["baseK"] * t["depthA1"] * self.input_bytes
        b1 = t["baseN"] * t["baseK"] * t["depthB1"] * self.input_bytes
        if a1 + b1 > self.l1_bytes:
            rules.append("L1_CAPACITY_EXCEEDED")
        if t["stepKa"] % t["stepKb"] and t["stepKb"] % t["stepKa"]:
            rules.append("STEP_KA_KB_INCOMMENSURATE")

        # Exact BASE/no-split/no-full-load contract from the cost model.
        if t["singleCoreK"] != t["K"]:
            rules.append("BASE_SINGLE_CORE_K_MUST_EQUAL_K")
        if t["singleCoreM"] > t["baseM"]:
            rules.append("BASE_SINGLE_CORE_M_EXCEEDS_BASE_M")
        if t["singleCoreN"] > t["baseN"]:
            rules.append("BASE_SINGLE_CORE_N_EXCEEDS_BASE_N")
        if t["baseM"] > _align_up(t["singleCoreM"], 16):
            rules.append("BASE_M_EXCEEDS_ALIGNED_SINGLE_CORE_M")
        if t["baseN"] > _align_up(t["singleCoreN"], 16):
            rules.append("BASE_N_EXCEEDS_ALIGNED_SINGLE_CORE_N")
        if t["baseK"] > _align_up(t["singleCoreK"], 16):
            rules.append("BASE_K_EXCEEDS_ALIGNED_SINGLE_CORE_K")

        m_total = _ceil_div(t["M"], t["singleCoreM"])
        n_total = _ceil_div(t["N"], t["singleCoreN"])
        m_block = t["l2MTileBlock"]
        n_block = t["l2NTileBlock"]
        if m_block == 0 or n_block == 0:
            if not (
                m_block == 0
                and n_block == 0
                and t["l2MTileCnt"] == 1
                and t["l2NTileCnt"] == 1
            ):
                rules.append("L2_DISABLED_ENCODING_MISMATCH")
        else:
            expected_m_count = _ceil_div(m_total, m_block)
            expected_n_count = _ceil_div(n_total, n_block)
            if t["l2MTileCnt"] != expected_m_count:
                rules.append("L2_M_TILE_COUNT_MISMATCH")
            if t["l2NTileCnt"] != expected_n_count:
                rules.append("L2_N_TILE_COUNT_MISMATCH")
            m_tail = m_total - (t["l2MTileCnt"] - 1) * m_block
            n_tail = n_total - (t["l2NTileCnt"] - 1) * n_block
            if not 1 <= m_tail <= m_block:
                rules.append("L2_M_TAIL_OUT_OF_RANGE")
            if not 1 <= n_tail <= n_block:
                rules.append("L2_N_TAIL_OUT_OF_RANGE")

        unique_rules = tuple(dict.fromkeys(rules))
        return FilterVerdict(
            predicted="no" if unique_rules else "yes",
            rules=unique_rules,
            tiling=t,
        )
