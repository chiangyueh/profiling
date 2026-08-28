#!/bin/bash
# прогон всех конфигов из CSV через msprof (runner в режиме -r sim/npu)
set -u

CSV="${1:-results.csv}"
RUNNER="${RUNNER:-./run.sh}"
RUN_MODE="${RUN_MODE:-sim}"
SOC="${SOC:-Ascend910B3}"

tail -n +2 "$CSV" | while IFS=',' read -r idx form base_k single_m single_n step_ka step_kb step_m step_n duration base_m base_n cores; do
    [ -z "$idx" ] && continue

    export MM_M="$form" MM_N="$form" MM_K="$form"
    export MM_BASE_M="$base_m" MM_BASE_N="$base_n" MM_BASE_K="$base_k"
    export MM_SINGLE_M="$single_m" MM_SINGLE_N="$single_n"
    export MM_STEP_M="$step_m" MM_STEP_N="$step_n"
    export MM_STEP_Ka="$step_ka" MM_STEP_Kb="$step_kb"
    export MM_ITER_ORDER=0 MM_OP_TILING=0

    echo "============================================================"
    echo "[$idx] form=${form}^3 cores=${cores} | baseK=${base_k} singleM=${single_m} singleN=${single_n} stepKa=${step_ka} stepKb=${step_kb} | cache_duration=${duration}"
    echo "------------------------------------------------------------"
    bash "$RUNNER" -r "$RUN_MODE" -v "$SOC"
    echo ""
done