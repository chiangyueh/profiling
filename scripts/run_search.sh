#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/env.sh"
cd "$ROOT"

WORKLOADS="${1:-config/workloads.csv}"
BEAM_WIDTH="${BEAM_WIDTH:-64}"
TABU_ITERS="${TABU_ITERS:-64}"
LNS_ROUNDS="${LNS_ROUNDS:-8}"
TOP_K="${TOP_K:-20}"
MODEL_RATIO_LIMIT="${MODEL_RATIO_LIMIT:-1.03}"
SEARCH_SCOPE="${SEARCH_SCOPE:-bottleneck_guided_v1}"
MAX_CORE_ROUNDS="${MAX_CORE_ROUNDS:-0}"
SEARCH_OUTPUT="${SEARCH_OUTPUT:-results/candidates.csv}"
SEARCH_ALL_OUTPUT="${SEARCH_ALL_OUTPUT:-results/all_evaluated.csv}"
SEARCH_TILING_DIR="${SEARCH_TILING_DIR:-results/tilings}"
RAW_OUTPUT="${SEARCH_OUTPUT}.generic.csv"
RAW_ALL_OUTPUT="${SEARCH_ALL_OUTPUT}.generic.csv"
RAW_LOG="${SEARCH_ALL_OUTPUT}.generic.log"
HISTORY_CSV="${MEASUREMENT_HISTORY:-results/npu_full_ocr_measurements.csv}"
HISTORY_ARGS=()
if [[ -f "${HISTORY_CSV}" ]]; then
    HISTORY_ARGS=(--history "${HISTORY_CSV}")
fi
PROFILE_HISTORY_CSV="${SEARCH_PROFILE_HISTORY:-}"
PROFILE_HISTORY_ARGS=()
if [[ -n "${PROFILE_HISTORY_CSV}" && -f "${PROFILE_HISTORY_CSV}" ]]; then
    PROFILE_HISTORY_ARGS=(--profile-history "${PROFILE_HISTORY_CSV}")
fi
CAMPAIGN_EXCLUSION_ARGS=()
CAMPAIGN_OBSERVATION_ARGS=()
if [[ "${SEARCH_SCOPE}" == "general_search_v1" ]]; then
    CAMPAIGN_EXCLUSIONS_SPEC="${SEARCH_CAMPAIGN_EXCLUSIONS:-config/general_search_v1_round1_fingerprints.csv:config/general_search_v1_round2_fingerprints.csv:config/general_search_v1_round3_partial_fingerprints.csv:config/general_search_v1_round4_fingerprints.csv}"
    IFS=: read -r -a CAMPAIGN_EXCLUSIONS <<<"${CAMPAIGN_EXCLUSIONS_SPEC}"
    for campaign_path in "${CAMPAIGN_EXCLUSIONS[@]}"; do
        if [[ -f "${campaign_path}" ]]; then
            CAMPAIGN_EXCLUSION_ARGS+=(
                --campaign-exclusions "${campaign_path}"
            )
        fi
    done
    CAMPAIGN_OBSERVATIONS_SPEC="${SEARCH_CAMPAIGN_OBSERVATIONS:-config/general_search_v1_round2_observations.csv:config/general_search_v1_round3_partial_observations.csv:config/general_search_v1_round4_observations.csv}"
    IFS=: read -r -a CAMPAIGN_OBSERVATIONS <<<"${CAMPAIGN_OBSERVATIONS_SPEC}"
    for campaign_path in "${CAMPAIGN_OBSERVATIONS[@]}"; do
        if [[ -f "${campaign_path}" ]]; then
            CAMPAIGN_OBSERVATION_ARGS+=(
                --campaign-observations "${campaign_path}"
            )
        fi
    done
fi

if [[ "${SEARCH_SCOPE}" == "all_templates_validation" ]]; then
    ./build/matmul_tiling_search \
        --workloads "$WORKLOADS" \
        --beam-width "$BEAM_WIDTH" \
        --tabu-iters "$TABU_ITERS" \
        --lns-rounds "$LNS_ROUNDS" \
        --top-k "$TOP_K" \
        --max-core-rounds "$MAX_CORE_ROUNDS" \
        --max-base-m 512 \
        --max-base-n 512 \
        --max-base-k 1024 \
        --soc "${ASCENDC_SOC_VERSION:-${SOC_VERSION:-Ascend910B}}" \
        --output "${RAW_OUTPUT}" \
        --all-output "${RAW_ALL_OUTPUT}" \
        --tiling-dir "${SEARCH_TILING_DIR}" | tee "${RAW_LOG}"
else
    # The active search is constructed from the official RuntimeKb seed.
    # The C++ host is needed only for authoritative platform capacities; do
    # not spend time enumerating a generic space that Python will discard.
    ./build/matmul_tiling_search \
        --platform-only \
        --soc "${ASCENDC_SOC_VERSION:-${SOC_VERSION:-Ascend910B}}" \
        --output "${RAW_OUTPUT}" \
        --all-output "${RAW_ALL_OUTPUT}" \
        --tiling-dir "${SEARCH_TILING_DIR}" | tee "${RAW_LOG}"
fi

PLATFORM_LINE="$(sed -n '/^CANN platform=/{p;q;}' "${RAW_LOG}")"
platform_field() {
    printf '%s\n' "${PLATFORM_LINE}" |
        sed -n "s/.*[[:space:]]$1=\\([0-9][0-9.]*\\).*/\\1/p"
}

PLATFORM_AIC_CORES="$(platform_field cores)"
PLATFORM_L0A_BYTES="$(platform_field L0A)"
PLATFORM_L0B_BYTES="$(platform_field L0B)"
PLATFORM_L0C_BYTES="$(platform_field L0C)"
PLATFORM_L1_BYTES="$(platform_field L1)"
PLATFORM_L2_BYTES="$(platform_field L2)"
PLATFORM_L2_BPC="$(platform_field L2_Bpc_per_core)"
PLATFORM_HBM_BPC="$(platform_field HBM_Bpc_per_core)"
for value in \
    "${PLATFORM_AIC_CORES}" "${PLATFORM_L0A_BYTES}" \
    "${PLATFORM_L0B_BYTES}" "${PLATFORM_L0C_BYTES}" \
    "${PLATFORM_L1_BYTES}" "${PLATFORM_L2_BYTES}" \
    "${PLATFORM_L2_BPC}" "${PLATFORM_HBM_BPC}"; do
    if [[ -z "${value}" ]]; then
        echo "fatal: cannot read complete platform capacities from the tiling host" >&2
        exit 1
    fi
done
if [[ "${PLATFORM_AIC_CORES}" -le 0 ]]; then
    echo "fatal: platform AIC count is not positive" >&2
    exit 1
fi

python3 tools/refine_matmul_v3_candidates.py \
    --raw-candidates "${RAW_ALL_OUTPUT}" \
    --workloads "${WORKLOADS}" \
    --output "${SEARCH_OUTPUT}" \
    --all-output "${SEARCH_ALL_OUTPUT}" \
    "${HISTORY_ARGS[@]}" \
    "${PROFILE_HISTORY_ARGS[@]}" \
    "${CAMPAIGN_EXCLUSION_ARGS[@]}" \
    "${CAMPAIGN_OBSERVATION_ARGS[@]}" \
    --top-k "${TOP_K}" \
    --beam-width "${BEAM_WIDTH}" \
    --tabu-iters "${TABU_ITERS}" \
    --lns-rounds "${LNS_ROUNDS}" \
    --model-ratio-limit "${MODEL_RATIO_LIMIT}" \
    --optimization-scope "${SEARCH_SCOPE}" \
    --soc "${ASCENDC_SOC_VERSION:-${SOC_VERSION:-Ascend910B}}" \
    --aic-cores "${PLATFORM_AIC_CORES}" \
    --l0a-bytes "${PLATFORM_L0A_BYTES}" \
    --l0b-bytes "${PLATFORM_L0B_BYTES}" \
    --l0c-bytes "${PLATFORM_L0C_BYTES}" \
    --l1-bytes "${PLATFORM_L1_BYTES}" \
    --l2-bytes "${PLATFORM_L2_BYTES}" \
    --l2-bytes-per-cycle-per-core "${PLATFORM_L2_BPC}" \
    --hbm-bytes-per-cycle-per-core "${PLATFORM_HBM_BPC}"

find "${RAW_OUTPUT}" "${RAW_ALL_OUTPUT}" "${RAW_LOG}" -type f -delete
