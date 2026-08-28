from tiling import limits, valids, pso, base
from scripts.gen_data import gen_golden_data

SIZES = [
         (4096, 18432, 7168),
         (4096, 1536, 7168),
         (4096, 24576, 1536)
        ]

class PsoAlgo(base.BaseAlgoReal, pso.PsoAlgo): pass

def get_domains(m: int, n: int, k: int) -> dict:
    return {"MM_BASE_M": [16, 32, 64, 128] + [x for x in range(256, m + 1, 128) if m % x == 0],
            "MM_BASE_N": [16, 32, 64, 128] + [x for x in range(256, m + 1, 128) if n % x == 0],
            "MM_BASE_K": [16, 32, 64, 128] + [x for x in range(256, m + 1, 128) if k % x == 0],
            'MM_SINGLE_M': [16, 32, 64, 128] + [x for x in range(256, m + 1, 128) if m % x == 0],
            'MM_SINGLE_N': [16, 32, 64, 128] + [x for x in range(256, m + 1, 128) if n % x == 0],
            'MM_STEP_Ka': [1, 2, 4, 8, 12, 16],
            'MM_STEP_Kb': [1, 2, 4, 8, 12, 16]}
    

def get_validator(domains: dict) -> valids.MatmulValidator:
    lims = limits.MatmulLimits(
        max_cores=24,
        L0A_size=64 * 1024,
        L0B_size=64 * 1024,
        L0C_size=128 * 1024,
        L1_size=512 * 1024,               
        domains=domains,       
        dtype_size=2,  
    )
    return valids.MatmulValidator(lims)
     

for M, K, N in SIZES:       
    input_params = [
        base.BaseParam(name="MM_M", value=M, is_const=True),
        base.BaseParam(name="MM_N", value=N, is_const=True),
        base.BaseParam(name="MM_K", value=K, is_const=True),
        base.BaseParam(name="MM_STEP_M", value=1, is_const=True),
        base.BaseParam(name="MM_STEP_N", value=1, is_const=True),
        base.BaseParam(name="MM_DB_L0A", value=2, is_const=True),
        base.BaseParam(name="MM_DB_L0B", value=2, is_const=True),
        base.BaseParam(name="MM_DB_L0C", value=2, is_const=True),
        base.BaseParam(name="MM_ITER_ORDER", value=0, is_const=True)
    ]
    domains = get_domains(M, N, K)
    print(f"DOMAINS: {domains}")
    gen_golden_data(M, N, K)
    algo = PsoAlgo(
        is_stop=lambda res: len(res) >= 16,
        swarm_size=32,
        validator=get_validator(domains),
        input_params=input_params,
        cache_path=f"msprof_cache_{M}_{N}_{K}_real.json",
        verbose=True,
    )
    print(f"START: M={M}, N={N}, K={K}")
    results = algo()
    print(f"END: M={M}, N={N}, K={K}")