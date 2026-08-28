from tiling import limits, valids, pso, base

for idx, (M, N, K) in enumerate([(512, 512, 512), (1024, 1024, 1024), (1536, 1536, 1536)]):
    domains = [
        {
        "MM_BASE_M": [16, 32, 64],
        "MM_BASE_N": [16, 32, 64],
        "MM_BASE_K": list(range(128, 513, 128)),
        'MM_SINGLE_M': list(range(64, 257, 64)),
        'MM_STEP_Ka': [1, 2, 4, 8, 12, 16],
        'MM_STEP_Kb': [1, 2, 4, 8, 12, 16]
        },
        {
        "MM_BASE_M": [16, 32, 64],
        "MM_BASE_N": [16, 32, 64],
        "MM_BASE_K": list(range(512, 1025, 128)),
        'MM_SINGLE_M': list(range(64, 256, 64)) + list(range(256, 513, 128)) + [1024],
        'MM_STEP_Ka': [1, 2, 4, 8, 12, 16],
        'MM_STEP_Kb': [1, 2, 4, 8, 12, 16]
        },
        {
        "MM_BASE_M": [16, 32, 64, 128],
        "MM_BASE_N": [16, 32, 64, 128],
        "MM_BASE_K": list(range(512, 2049, 128)),
        'MM_SINGLE_M': list(range(64, 256, 64)) + list(range(256, 769, 128)) + [2048],
        'MM_STEP_Ka': [1, 2, 4, 8, 12, 16],
        'MM_STEP_Kb': [1, 2, 4, 8, 12, 16]
        },
        
        ]

    lims = limits.MatmulLimits(
        max_cores=24,
        L0A_size=64 * 1024,
        L0B_size=64 * 1024,
        L0C_size=128 * 1024,
        L1_size=512 * 1024,               
        domains=domains[idx],       
        dtype_size=2,         
    )

    validator = valids.MatmulValidator(lims)

    input_params = [
        base.BaseParam(name="MM_M", value=M, is_const=True),
        base.BaseParam(name="MM_N", value=N, is_const=True),
        base.BaseParam(name="MM_K", value=K, is_const=True),
        base.BaseParam(name="MM_SINGLE_N", value=N, is_const=True),
        base.BaseParam(name="MM_STEP_M", value=1, is_const=True),
        base.BaseParam(name="MM_STEP_N", value=1, is_const=True),
        base.BaseParam(name="MM_DB_L0A", value=2, is_const=True),
        base.BaseParam(name="MM_DB_L0B", value=2, is_const=True),
        base.BaseParam(name="MM_DB_L0C", value=2, is_const=True),
        base.BaseParam(name="MM_ITER_ORDER", value=0, is_const=True)
    ]

    algo = pso.PsoAlgo(
        is_stop=lambda res: len(res) >= 16,
        swarm_size=32,
        validator=validator,
        input_params=input_params,
        cache_path=f"msprof_cache_{M}_{N}_{K}_sim.json",
        verbose=True,
    )
    print("START PSO")
    results = algo()

    print("END")