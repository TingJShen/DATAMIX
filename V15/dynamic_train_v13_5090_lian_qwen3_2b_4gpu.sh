#!/bin/bash

set -euo pipefail

export CONDA_ENV_NAME=${CONDA_ENV_NAME:-llama2_vllm_copy}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export MODEL_PATH=${MODEL_PATH:-/zhdd/home/tjshen/260415_ArcherA100/model_overlays/Qwen3-2B_with_generation_config}
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-/zhdd/home/tjshen/260413_Backup_model_resume_20260323_031320/Skywork-Reward-Llama-3.1-8B-v0.2}

export PROJECT_NAME=${PROJECT_NAME:-ArcherCodeR-V13-Qwen3-2B-5090Lian}
export EXP_NAME=${EXP_NAME:-train_5090_v13_qwen3_2b_4gpu_bsz32_save10_100}
export OUTPUT_ROOT=${OUTPUT_ROOT:-./output_5090_Lian_v15}
export DYNAMIC_METHOD=${DYNAMIC_METHOD:-sample_taylor_v13}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-2000}
export SAVE_FREQ=${SAVE_FREQ:-100}
export SAVE_STEPS=${SAVE_STEPS:-10}

export TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-32}
export TRAIN_PROMPT_MINI_BSZ=${TRAIN_PROMPT_MINI_BSZ:-8}
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}

export SHADOW_ANCHOR_SIZE_PER_DOMAIN=${SHADOW_ANCHOR_SIZE_PER_DOMAIN:-128}
export GRAD_PROJECTION_DIM=${GRAD_PROJECTION_DIM:-256}
export CURVATURE_REFRESH_FREQ=${CURVATURE_REFRESH_FREQ:-50}
export CANDIDATE_MULTIPLIER=${CANDIDATE_MULTIPLIER:-2}

export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}
export ARCHER_HIDE_FLASH_ATTN_FROM_VLLM=${ARCHER_HIDE_FLASH_ATTN_FROM_VLLM:-1}
export DISABLE_PYTORCH_CUDA_ALLOC_CONF=${DISABLE_PYTORCH_CUDA_ALLOC_CONF:-1}

BASE_DIR=/zhdd/home/tjshen/260415_ArcherA100
DIAG_SRC="${BASE_DIR}/diag_5090_Hao_v12_qwen3_2b_20260421"
QWEN35_SHIM="${DIAG_SRC}/qwen35_flashattn_shim"
QWEN35_TRANSFORMERS="${BASE_DIR}/qwen3_pydeps/transformers_qwen3_20260420_min2"
QWEN35_SOURCE_MODELS_DIR="${QWEN35_TRANSFORMERS}/transformers/models"
QWEN35_HUB_WHL="${BASE_DIR}/diag_8A100_qwen3_20260420/pip_tmp_t55/pip-unpack-f2_o3xti/huggingface_hub-1.11.0-py3-none-any.whl"
QWEN35_OVERLAY="${BASE_DIR}/envs/archer_qwen3_vllm_torch210_overlay/python_packages"
RAY_MIN_DASH="${BASE_DIR}/diag_5090_Hao_qwen3_8b_20260420/sitecustomize_ray_minimal_dashboard"
QWEN_VL_TARGET="${BASE_DIR}/python_targets/qwen_vl_utils_20260422"
QWEN35_MASK_COMPAT="${BASE_DIR}/auto_watch_5090_Lian_v12_qwen3_2b_2gpu_3g_fixenv_persistent4_pairsafe/qwen35_mask_compat_shim"
NATIVE_SITE="/home/tjshen/miniconda3/envs/${CONDA_ENV_NAME}/lib/python3.10/site-packages"
NATIVE_NUMPY_SHIM="${BASE_DIR}/v13_grad/runtime_5090_lian/native_numpy_compat"

mkdir -p "${NATIVE_NUMPY_SHIM}"
if [ ! -e "${NATIVE_NUMPY_SHIM}/numpy" ]; then
    ln -s "${NATIVE_SITE}/numpy" "${NATIVE_NUMPY_SHIM}/numpy"
fi
if [ ! -e "${NATIVE_NUMPY_SHIM}/numpy-1.26.4.dist-info" ]; then
    ln -s "${NATIVE_SITE}/numpy-1.26.4.dist-info" "${NATIVE_NUMPY_SHIM}/numpy-1.26.4.dist-info"
fi

export ARCHER_QWEN35_SOURCE_MODELS_DIR=${ARCHER_QWEN35_SOURCE_MODELS_DIR:-${QWEN35_SOURCE_MODELS_DIR}}
export ARCHER_BASE_SITECUSTOMIZE="${QWEN35_SHIM}/sitecustomize.py"
export PYTHONPATH="${QWEN35_MASK_COMPAT}:${QWEN35_SHIM}:${NATIVE_NUMPY_SHIM}:${RAY_MIN_DASH}:${QWEN_VL_TARGET}:${QWEN35_OVERLAY}:${QWEN35_TRANSFORMERS}:${QWEN35_HUB_WHL}${PYTHONPATH:+:${PYTHONPATH}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/dynamic_train_v13_a100_formal.sh"
