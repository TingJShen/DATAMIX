#!/bin/bash

set -euo pipefail

BASE_DIR=/zhdd/home/tjshen/260415_ArcherA100
RUN_TAG=${RUN_TAG:-grad1}
DIAG_DIR="${BASE_DIR}/diag_5090_Lian_v13_grad_qwen3_2b_4gpu_${RUN_TAG}_$(date +%Y%m%d_%H%M%S)"

mkdir -p "${DIAG_DIR}"
cd "${BASE_DIR}/v13_grad"

nohup env \
    CONDA_ENV_NAME=llama2_vllm_copy \
    CUDA_VISIBLE_DEVICES=0,3,4,5 \
    WORK_DIR="${BASE_DIR}/v13_grad" \
    RUNTIME_DIR="${BASE_DIR}/runtime_v13_grad" \
    RAY_TMPDIR="/tmp/ray_v13g" \
    MODEL_PATH="${BASE_DIR}/model_overlays/Qwen3-2B_with_generation_config" \
    REWARD_MODEL_PATH=/zhdd/home/tjshen/260413_Backup_model_resume_20260323_031320/Skywork-Reward-Llama-3.1-8B-v0.2 \
    PROJECT_NAME=ArcherCodeR-V13grad-Qwen3-2B-5090Lian \
    EXP_NAME="train_5090_v13_grad_qwen3_2b_4gpu_0_3_4_5_bsz32_save10_100_${RUN_TAG}" \
    OUTPUT_ROOT=./output_5090_Lian_v13_grad \
    DYNAMIC_METHOD=sample_taylor_v13 \
    TOTAL_TRAINING_STEPS=2000 \
    SAVE_FREQ=100 \
    SAVE_STEPS=10 \
    TRAIN_PROMPT_BSZ=32 \
    TRAIN_PROMPT_MINI_BSZ=8 \
    ROLLOUT_GPU_MEMORY_UTILIZATION=0.25 \
    SHADOW_ANCHOR_SIZE_PER_DOMAIN=128 \
    GRAD_PROJECTION_DIM=256 \
    CURVATURE_REFRESH_FREQ=50 \
    DISABLE_PYTORCH_CUDA_ALLOC_CONF=1 \
    RAY_memory_usage_threshold=0.99 \
    ./dynamic_train_v13_5090_lian_qwen3_2b_4gpu.sh \
    > "${DIAG_DIR}/launch.log" 2>&1 &

echo "PID=$!"
echo "DIAG_DIR=${DIAG_DIR}"
echo "TRAIN_DIR=${BASE_DIR}/v13_grad/output_5090_Lian_v13_grad/ArcherCodeR-V13grad-Qwen3-2B-5090Lian/train_5090_v13_grad_qwen3_2b_4gpu_0_3_4_5_bsz32_save10_100_${RUN_TAG}"
