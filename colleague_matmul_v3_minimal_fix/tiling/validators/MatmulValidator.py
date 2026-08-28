from tiling.base import BaseValidator, BaseParam
from tiling.limits import MatmulLimits
import random
import math
import copy


class MatmulValidator(BaseValidator):
    def __init__(self,
                 limits: MatmulLimits
                 ) -> None:
        self.limits = limits
        super().__init__()

    def is_valid(self, params: list[BaseParam]) -> bool:
        p = {p.name: p.value for p in params}
        return all(self._base_tiles_is_valid(p['MM_BASE_M'], p['MM_BASE_N'], p['MM_BASE_K'],
                                             p['MM_DB_L0A'], p['MM_DB_L0B'], p['MM_DB_L0C'])) and \
            self._single_core_is_valid(p['MM_M'], p['MM_N'], p['MM_SINGLE_M'], p['MM_SINGLE_N']) and \
            self._l1_size_is_valid(p['MM_BASE_M'], p['MM_BASE_N'], p['MM_BASE_K'],
                                   p['MM_STEP_M'], p['MM_STEP_N'], p['MM_STEP_Ka'], p['MM_STEP_Kb'])

    def repair(self, params: list[BaseParam]) -> list[BaseParam]:
        ps_dict = {p.name: p for p in params}
        if not self._single_core_is_valid(
                ps_dict['MM_M'].value,
                ps_dict['MM_N'].value,
                ps_dict['MM_SINGLE_M'].value,
                ps_dict['MM_SINGLE_N'].value):
            singleM, singleN = self._repair_single_core(ps_dict)
            ps_dict.update({'MM_SINGLE_M': singleM, 'MM_SINGLE_N': singleN})
        if not all(self._base_tiles_is_valid(
                ps_dict['MM_BASE_M'].value, ps_dict['MM_BASE_N'].value, ps_dict['MM_BASE_K'].value,
                ps_dict['MM_DB_L0A'].value, ps_dict['MM_DB_L0B'].value, ps_dict['MM_DB_L0C'].value)):
            baseM, baseN, baseK = self._repair_base_tiles(ps_dict)
            ps_dict.update({'MM_BASE_M': baseM, 'MM_BASE_N': baseN, 'MM_BASE_K': baseK})
        if not self._l1_size_is_valid(
                ps_dict['MM_BASE_M'].value, ps_dict['MM_BASE_N'].value, ps_dict['MM_BASE_K'].value,
                ps_dict['MM_STEP_M'].value, ps_dict['MM_STEP_N'].value,
                ps_dict['MM_STEP_Ka'].value, ps_dict['MM_STEP_Kb'].value):
            stepM, stepN, stepKa, stepKb = self._repair_l1_size(ps_dict)
            ps_dict.update({'MM_STEP_M': stepM, 'MM_STEP_N': stepN,
                            'MM_STEP_Ka': stepKa, 'MM_STEP_Kb': stepKb})
        return list(ps_dict.values())

    def get_combinations(self, num: int, const_params: list) -> list[list[BaseParam]]:
        combs = []
        while len(combs) < num:
            params = [self._make_param(p.name, p.value, True) for p in const_params]
            for name, domain in self.limits.domains.items():
                params.append(self._make_param(name, random.choice(domain), False, domain))
            if self.is_valid(params):
                combs.append(params)
        return combs
    
    def _make_param(self, name: str, value: int, is_const: bool, domain: list[int] | None = None) -> BaseParam:
        return BaseParam(name=name, value=value, is_const=is_const,domain=domain or [value])

    def _repair_single_core(self, params: dict[str, BaseParam]) -> tuple[BaseParam]:
        M = params['MM_M'].value
        N = params['MM_N'].value
        singleM = copy.copy(params['MM_SINGLE_M'])
        singleN = copy.copy(params['MM_SINGLE_N'])

        is_repaired = lambda singleM, singleN: \
            self._single_core_is_valid(M, N, singleM.value, singleN.value)
        while not is_repaired(singleM, singleN):
            before = (singleM.index, singleN.index)
            if singleM.index < len(singleM.domain) - 1:
                singleM.update(singleM.index + 1)
            if singleN.index < len(singleN.domain) - 1:
                singleN.update(singleN.index + 1)
            after = (singleM.index, singleN.index)
            if before == after:
                break
        return singleM, singleN

    def _repair_base_tiles(self, params: dict[str, BaseParam]) -> tuple[BaseParam]:
        baseM = copy.copy(params['MM_BASE_M'])
        baseN = copy.copy(params['MM_BASE_N'])
        baseK = copy.copy(params['MM_BASE_K'])
        dbL0A = params['MM_DB_L0A'].value
        dbL0B = params['MM_DB_L0B'].value
        dbL0C = params['MM_DB_L0C'].value

        is_repaired = lambda baseM, baseN, baseK: \
            all(self._base_tiles_is_valid(baseM.value, baseN.value, baseK.value, dbL0A, dbL0B, dbL0C))
        while not is_repaired(baseM, baseN, baseK):
            before = (baseM.index, baseN.index, baseK.index)
            is_valid = self._base_tiles_is_valid(baseM.value, baseN.value, baseK.value, dbL0A, dbL0B, dbL0C)
            if not is_valid[0]:
                if baseM.index > 0:
                    baseM.update(baseM.index - 1)
                if baseK.index > 0:
                    baseK.update(baseK.index - 1)
            elif not is_valid[1]:
                if baseN.index > 0:
                    baseN.update(baseN.index - 1)
                if baseK.index > 0:
                    baseK.update(baseK.index - 1)
            else:
                if baseM.index > 0:
                    baseM.update(baseM.index - 1)
                if baseN.index > 0:
                    baseN.update(baseN.index - 1)
            after = (baseM.index, baseN.index, baseK.index)
            if before == after:
                break
        return baseM, baseN, baseK

    def _repair_l1_size(self, params: dict[str, BaseParam]) -> tuple[BaseParam]:
        baseM = params['MM_BASE_M'].value
        baseN = params['MM_BASE_N'].value
        baseK = params['MM_BASE_K'].value
        stepM = copy.copy(params['MM_STEP_M'])
        stepN = copy.copy(params['MM_STEP_N'])
        stepKa = copy.copy(params['MM_STEP_Ka'])
        stepKb = copy.copy(params['MM_STEP_Kb'])

        is_repaired = lambda stepM, stepN, stepKa, stepKb: \
            self._l1_size_is_valid(baseM, baseN, baseK,
                                   stepM.value, stepN.value,
                                   stepKa.value, stepKb.value)

        while not is_repaired(stepM, stepN, stepKa, stepKb):
            before = (stepM.index, stepN.index,
                      stepKa.index, stepKb.index)

            depthA1 = stepM.value * stepKa.value * 2
            depthB1 = stepN.value * stepKb.value * 2

            if baseM * depthA1 >= baseN * depthB1:
                if stepM.index > 0:
                    stepM.update(stepM.index - 1)
                if stepKa.index > 0:
                    stepKa.update(stepKa.index - 1)
            else:
                if stepN.index > 0:
                    stepN.update(stepN.index - 1)
                if stepKb.index > 0:
                    stepKb.update(stepKb.index - 1)

            after = (stepM.index, stepN.index,
                     stepKa.index, stepKb.index)
            if before == after:
                break

        return stepM, stepN, stepKa, stepKb

    def _single_core_is_valid(self, M: int, N: int, singleM: int, singleN: int) -> bool:
        return (math.ceil(M / singleM) * math.ceil(N / singleN)) <= self.limits.max_cores

    def _base_tiles_is_valid(self,
                             baseM: int, baseN: int, baseK: int,
                             dbL0A: int, dbL0B: int, dbL0C: int) -> tuple[bool]:
        is_valid_L0 = (baseM * baseK * self.limits.dtype_size * dbL0A <= self.limits.L0A_size,
                       baseN * baseK * self.limits.dtype_size * dbL0B <= self.limits.L0B_size,
                       baseM * baseN * 4 * dbL0C <= self.limits.L0C_size)
        return is_valid_L0

    def _l1_size_is_valid(self, baseM: int, baseN: int, baseK: int,
                          stepM: int, stepN: int, stepKa: int, stepKb: int) -> bool:
        depthA1 = stepM * stepKa * 2
        depthB1 = stepN * stepKb * 2
        return (baseM * depthA1 + baseN * depthB1) * baseK * self.limits.dtype_size <= self.limits.L1_size