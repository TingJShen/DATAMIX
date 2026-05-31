#!/bin/bash

set -u

BASE_DIR=/zhdd/home/tjshen/260415_ArcherA100
WORK_DIR="${BASE_DIR}/v13"
MONITOR_TS=${MONITOR_TS:-$(date +%Y%m%d_%H%M%S)}
DIAG_ROOT=${DIAG_ROOT:-"${BASE_DIR}/diag_5A100_v13_qwen25_1_5b_1gpu_monitor_${MONITOR_TS}"}
MONITOR_LOG="${DIAG_ROOT}/monitor.log"

CANDIDATE_GPUS=${CANDIDATE_GPUS:-0,1,2,3,4}
MIN_FREE_MB=${MIN_FREE_MB:-70000}
SLEEP_SECONDS=${SLEEP_SECONDS:-60}
# Ordered from "fill the card" to conservative. The monitor only falls back if
# the previous launch exits, typically from OOM.
PROFILES=${PROFILES:-"32:16:0.70 24:12:0.62 16:8:0.55"}

mkdir -p "${DIAG_ROOT}"

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${MONITOR_LOG}"
}

gpu_is_candidate() {
    case ",${CANDIDATE_GPUS}," in
        *",$1,"*) return 0 ;;
        *) return 1 ;;
    esac
}

pick_gpu() {
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
        awk -F, -v min_free="${MIN_FREE_MB}" '
            {
                gsub(/ /, "", $1);
                gsub(/ /, "", $2);
                if ($2 + 0 >= min_free) {
                    print $1, $2;
                }
            }
        ' |
        while read -r gpu free_mb; do
            if gpu_is_candidate "${gpu}"; then
                printf '%s %s\n' "${gpu}" "${free_mb}"
            fi
        done |
        sort -k2,2nr |
        awk 'NR == 1 {print $1}'
}

own_training_running() {
    pgrep -af "[d]apo.main_dapo.*ArcherCodeR-V13-Qwen25-1_5B-5A100.*train_5a100_v13_qwen25_1_5b_1gpu" >/dev/null 2>&1
}

log "monitor started: candidate_gpus=${CANDIDATE_GPUS}, min_free_mb=${MIN_FREE_MB}, profiles=${PROFILES}"

while true; do
    if own_training_running; then
        log "an own 1-GPU V13 training process is already running; monitor exits"
        exit 0
    fi

    gpu=$(pick_gpu || true)
    if [ -z "${gpu}" ]; then
        snapshot=$(nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits | tr '\n' ';')
        log "no GPU has >= ${MIN_FREE_MB} MB free yet; gpu_snapshot=${snapshot}"
        sleep "${SLEEP_SECONDS}"
        continue
    fi

    for profile in ${PROFILES}; do
        IFS=: read -r bsz mini_bsz gpu_util <<< "${profile}"
        RUN_TS=$(date +%Y%m%d_%H%M%S)
        profile_tag="gpu${gpu}_bsz${bsz}_util${gpu_util}"
        launch_log="${DIAG_ROOT}/launch_${RUN_TS}_${profile_tag}.log"
        output_dir="${WORK_DIR}/output_5A100_single/ArcherCodeR-V13-Qwen25-1_5B-5A100/train_5a100_v13_qwen25_1_5b_1gpu_gpu${gpu}_bsz${bsz}_save100_${RUN_TS}"

        log "launching profile=${profile_tag}; output=${output_dir}; launch_log=${launch_log}"
        (
            cd "${WORK_DIR}" &&
            RUN_TS="${RUN_TS}" \
            CUDA_VISIBLE_DEVICES="${gpu}" \
            CONDA_ENV_NAME=llama2_vllm_copy \
            TRAIN_PROMPT_BSZ="${bsz}" \
            TRAIN_PROMPT_MINI_BSZ="${mini_bsz}" \
            ROLLOUT_GPU_MEMORY_UTILIZATION="${gpu_util}" \
            bash ./dynamic_train_v13_5a100_qwen25_1_5b_1gpu.sh
        ) > "${launch_log}" 2>&1
        status=$?

        if [ "${status}" -eq 0 ]; then
            log "training finished successfully for profile=${profile_tag}"
            exit 0
        fi

        if grep -Eiq 'out of memory|CUDA error|CUBLAS|NCCL|RuntimeError|Traceback|Error executing job' "${launch_log}"; then
            log "profile=${profile_tag} exited with status=${status}; trying next fallback profile"
            sleep 10
            continue
        fi

        log "profile=${profile_tag} exited with status=${status}; non-OOM failure, monitor will keep watching after sleep"
        sleep "${SLEEP_SECONDS}"
        break
    done
done
