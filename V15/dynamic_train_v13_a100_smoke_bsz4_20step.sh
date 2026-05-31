#!/bin/bash

set -xeuo pipefail

BASE_DIR=/zhdd/home/tjshen/260415_ArcherA100
WORK_DIR=${WORK_DIR:-"${BASE_DIR}/v13"}
RUNTIME_DIR=${RUNTIME_DIR:-"${BASE_DIR}/runtime_v13"}

cd "${WORK_DIR}"
CONDA_ENV_NAME=${CONDA_ENV_NAME:-llama2_vllm}
source /home/tjshen/miniconda3/bin/activate "${CONDA_ENV_NAME}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,3,4}
export TMPDIR="${RUNTIME_DIR}/tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export RAY_TMPDIR=${RAY_TMPDIR:-"${BASE_DIR}/r_v13"}
export XDG_CACHE_HOME="${RUNTIME_DIR}/cache/xdg"
export HF_HOME="${RUNTIME_DIR}/cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_HOME="${RUNTIME_DIR}/cache/torch"
export TRITON_CACHE_DIR="${RUNTIME_DIR}/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="${RUNTIME_DIR}/cache/torchinductor"
export VLLM_CACHE_ROOT="${RUNTIME_DIR}/cache/vllm"
export PYTHONPYCACHEPREFIX="${RUNTIME_DIR}/cache/pycache"
export WANDB_DIR="${RUNTIME_DIR}/wandb"
export WANDB_CACHE_DIR="${RUNTIME_DIR}/wandb/cache"
export WANDB_CONFIG_DIR="${RUNTIME_DIR}/wandb/config"
export TORCH_NCCL_ENABLE_MONITORING=0
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export HYDRA_FULL_ERROR=1
export WANDB_MODE=${WANDB_MODE:-offline}
export TOKENIZERS_PARALLELISM=true
if [ "${DISABLE_PYTORCH_CUDA_ALLOC_CONF:-0}" = "1" ]; then
    unset PYTORCH_CUDA_ALLOC_CONF
else
    export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
fi

mkdir -p \
    "${TMPDIR}" \
    "${RAY_TMPDIR}" \
    "${XDG_CACHE_HOME}" \
    "${HF_HOME}" \
    "${TRANSFORMERS_CACHE}" \
    "${HF_DATASETS_CACHE}" \
    "${TORCH_HOME}" \
    "${TRITON_CACHE_DIR}" \
    "${TORCHINDUCTOR_CACHE_DIR}" \
    "${VLLM_CACHE_ROOT}" \
    "${PYTHONPYCACHEPREFIX}" \
    "${WANDB_DIR}" \
    "${WANDB_CACHE_DIR}" \
    "${WANDB_CONFIG_DIR}"

GPU_COUNT=0
IFS=',' read -ra VISIBLE_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
for gpu_id in "${VISIBLE_GPU_IDS[@]}"; do
    if [ -n "${gpu_id//[[:space:]]/}" ]; then
        GPU_COUNT=$((GPU_COUNT + 1))
    fi
done

RAY_PORT=${RAY_PORT:-6381}
RAY_RUNTIME_ENV_AGENT_PORT=${RAY_RUNTIME_ENV_AGENT_PORT:-30001}
RAY_DASHBOARD_AGENT_LISTEN_PORT=${RAY_DASHBOARD_AGENT_LISTEN_PORT:-30002}
RAY_DASHBOARD_AGENT_GRPC_PORT=${RAY_DASHBOARD_AGENT_GRPC_PORT:-30003}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
RAY_METRICS_EXPORT_PORT=${RAY_METRICS_EXPORT_PORT:-30005}
RAY_OBJECT_MANAGER_PORT=${RAY_OBJECT_MANAGER_PORT:-30006}
RAY_NODE_MANAGER_PORT=${RAY_NODE_MANAGER_PORT:-30007}
RAY_MIN_WORKER_PORT=${RAY_MIN_WORKER_PORT:-31000}
RAY_MAX_WORKER_PORT=${RAY_MAX_WORKER_PORT:-39999}

if [ "${SKIP_RAY_START:-0}" = "1" ]; then
    echo "SKIP_RAY_START=1: reuse existing Ray runtime at 127.0.0.1:${RAY_PORT}"
else
    if [ "${SKIP_RAY_STOP:-0}" = "1" ]; then
        echo "SKIP_RAY_STOP=1: skip ray stop before starting this isolated Ray runtime"
    else
        timeout 60s ray stop --force || true
    fi
    ray start --head --port="${RAY_PORT}" --dashboard-port="${RAY_DASHBOARD_PORT}" --include-dashboard=False \
        --temp-dir="${RAY_TMPDIR}" \
        --num-gpus="${GPU_COUNT}" \
        --num-cpus="${RAY_NUM_CPUS:-32}" \
        --runtime-env-agent-port="${RAY_RUNTIME_ENV_AGENT_PORT}" \
        --dashboard-agent-listen-port="${RAY_DASHBOARD_AGENT_LISTEN_PORT}" \
        --dashboard-agent-grpc-port="${RAY_DASHBOARD_AGENT_GRPC_PORT}" \
        --metrics-export-port="${RAY_METRICS_EXPORT_PORT}" \
        --object-manager-port="${RAY_OBJECT_MANAGER_PORT}" \
        --node-manager-port="${RAY_NODE_MANAGER_PORT}" \
        --min-worker-port="${RAY_MIN_WORKER_PORT}" \
        --max-worker-port="${RAY_MAX_WORKER_PORT}"
fi
export RAY_ADDRESS=127.0.0.1:${RAY_PORT}

nnodes=1

project_name=${PROJECT_NAME:-ArcherCodeR-V13-A100}
exp_name=${EXP_NAME:-smoke_v13_bsz4_20step_qwen15b}

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001
kl_loss_type=low_var_kl

clip_ratio_low=0.2
clip_ratio_high=0.2
loss_agg_mode=token-mean

max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-1024}
model_context_length=$((max_prompt_length + max_response_length))
enable_overlong_buffer=False
overlong_buffer_len=16
overlong_penalty_factor=1.0
v_max_response_length=${max_response_length}

train_prompt_bsz=${TRAIN_PROMPT_BSZ:-4}
gen_prompt_bsz=${train_prompt_bsz}
train_prompt_mini_bsz=${TRAIN_PROMPT_MINI_BSZ:-${train_prompt_bsz}}

MODEL_PATH=${MODEL_PATH:-/zhdd/models/Qwen2.5-1.5B-Instruct}
REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-/zhdd/home/tjshen/260413_Backup_model_resume_20260323_031320/Skywork-Reward-Llama-3.1-8B-v0.2}
OUTPUT_ROOT=${OUTPUT_ROOT:-./output_v13}
CKPTS_DIR=${OUTPUT_ROOT}/${project_name}/${exp_name}
TRAIN_FILE=${TRAIN_FILE:-./data/train/Mix_AirRep_nonrepeated_sysprom.json}
TEST_FILE=${TEST_FILE:-./data/test/livecodebench_v5.json}
TARGET_MATH_FILE=${TARGET_MATH_FILE:-./data/test/AIME2025.json}
TARGET_CODE_FILE=${TARGET_CODE_FILE:-./data/test/LCB.json}
TARGET_GENERAL_FILE=${TARGET_GENERAL_FILE:-./data/test/Arena_question.json}

MIN_WEIGHT=0.05
UPDATE_FREQ=10
USE_INVERSE_IMPROVEMENT=True
TARGET_MAX_TOKENS=${model_context_length}
FULL_DATASET_EMBEDDING_BATCH_SIZE=${FULL_DATASET_EMBEDDING_BATCH_SIZE:-1}
DYNAMIC_METHOD=${DYNAMIC_METHOD:-sample_taylor_v13}
SHADOW_ANCHOR_SIZE_PER_DOMAIN=${SHADOW_ANCHOR_SIZE_PER_DOMAIN:-8}
CANDIDATE_MULTIPLIER=${CANDIDATE_MULTIPLIER:-4}
GRAD_PROJECTION_DIM=${GRAD_PROJECTION_DIM:-64}
GRAD_PROJECTION_SEED=${GRAD_PROJECTION_SEED:-20260506}
SAMPLE_SOFTMAX_TEMPERATURE=${SAMPLE_SOFTMAX_TEMPERATURE:-0.7}
DOMAIN_SOFTMAX_TEMPERATURE=${DOMAIN_SOFTMAX_TEMPERATURE:-1.0}
DOMAIN_MIN_WEIGHT=${DOMAIN_MIN_WEIGHT:-0.15}
LEARN_EMA_DECAY=${LEARN_EMA_DECAY:-0.2}
CURVATURE_REFRESH_FREQ=${CURVATURE_REFRESH_FREQ:-10}

n_resp_per_prompt=2
temperature=1.0
top_p=1.0
top_k=-1
rollout_gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}
v_n=1
v_temperature=0.8
v_top_p=1.0
v_top_k=-1

sp_size=1
gen_tp=1
use_dynamic_bsz=False
micro_batch_size_per_gpu=1
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=${OFFLOAD:-False}
reward_param_offload=${REWARD_PARAM_OFFLOAD:-${offload}}

use_token_entropy_separate=True
high_entropy_kl_loss_scale_coef=0.0
low_entropy_clip_ratio_low=0.2
low_entropy_clip_ratio_high=0.2
high_entropy_clip_ratio_low=0.5
high_entropy_clip_ratio_high=0.5

use_overlong_filter=False
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-20}
save_freq=${SAVE_FREQ:-10}
save_steps=${SAVE_STEPS:-}

mkdir -p "${CKPTS_DIR}"

python -m dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    +actor_rollout_ref.actor.use_token_entropy_separate=${use_token_entropy_separate} \
    +actor_rollout_ref.actor.high_entropy_kl_loss_scale_coef=${high_entropy_kl_loss_scale_coef} \
    +actor_rollout_ref.actor.low_entropy_clip_ratio_low=${low_entropy_clip_ratio_low} \
    +actor_rollout_ref.actor.low_entropy_clip_ratio_high=${low_entropy_clip_ratio_high} \
    +actor_rollout_ref.actor.high_entropy_clip_ratio_low=${high_entropy_clip_ratio_low} \
    +actor_rollout_ref.actor.high_entropy_clip_ratio_high=${high_entropy_clip_ratio_high} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${v_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${v_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k="${v_top_k}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=${v_n} \
    +actor_rollout_ref.rollout.val_kwargs.response_length=${v_max_response_length} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.enable=True \
    reward_model.model.path="${REWARD_MODEL_PATH}" \
    reward_model.model.fsdp_config.param_offload=${reward_param_offload} \
    reward_model.micro_batch_size_per_gpu=1 \
    reward_model.reward_manager=wizard \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger=['console'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${GPU_COUNT}" \
    trainer.nnodes="${nnodes}" \
    trainer.balance_batch=False \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=${save_freq} \
    trainer.save_steps="[${save_steps}]" \
    trainer.total_epochs=1 \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=disable \
    +trainer.validation_data_dir=${CKPTS_DIR}/eval \
    +trainer.enable_overlong_filter=${use_overlong_filter} \
    +trainer.rejection_sample=True \
    +trainer.dynamic_sampling.enable=True \
    +trainer.dynamic_sampling.method=${DYNAMIC_METHOD} \
    +trainer.dynamic_sampling.update_freq=${UPDATE_FREQ} \
    +trainer.dynamic_sampling.min_weight=${MIN_WEIGHT} \
    +trainer.dynamic_sampling.use_inverse_improvement=${USE_INVERSE_IMPROVEMENT} \
    +trainer.dynamic_sampling.target_test_files.math_file="${TARGET_MATH_FILE}" \
    +trainer.dynamic_sampling.target_test_files.code_file="${TARGET_CODE_FILE}" \
    +trainer.dynamic_sampling.target_test_files.general_file="${TARGET_GENERAL_FILE}" \
    +trainer.dynamic_sampling.full_dataset_embedding_batch_size=${FULL_DATASET_EMBEDDING_BATCH_SIZE} \
    +trainer.dynamic_sampling.full_train_max_samples_per_category=${FULL_TRAIN_MAX_SAMPLES_PER_CATEGORY:-4} \
    +trainer.dynamic_sampling.full_target_max_samples_per_category=${FULL_TARGET_MAX_SAMPLES_PER_CATEGORY:-4} \
    +trainer.dynamic_sampling.full_train_max_tokens=${max_prompt_length} \
    +trainer.dynamic_sampling.full_target_max_tokens=${TARGET_MAX_TOKENS} \
    +trainer.dynamic_sampling.shadow_anchor_size_per_domain=${SHADOW_ANCHOR_SIZE_PER_DOMAIN} \
    +trainer.dynamic_sampling.candidate_multiplier=${CANDIDATE_MULTIPLIER} \
    +trainer.dynamic_sampling.grad_projection_dim=${GRAD_PROJECTION_DIM} \
    +trainer.dynamic_sampling.grad_projection_seed=${GRAD_PROJECTION_SEED} \
    +trainer.dynamic_sampling.sample_softmax_temperature=${SAMPLE_SOFTMAX_TEMPERATURE} \
    +trainer.dynamic_sampling.domain_softmax_temperature=${DOMAIN_SOFTMAX_TEMPERATURE} \
    +trainer.dynamic_sampling.domain_min_weight=${DOMAIN_MIN_WEIGHT} \
    +trainer.dynamic_sampling.learn_ema_decay=${LEARN_EMA_DECAY} \
    +trainer.dynamic_sampling.curvature_refresh_freq=${CURVATURE_REFRESH_FREQ} \
    +trainer.dynamic_sampling.sample_score_weights.target_rel=${SAMPLE_SCORE_TARGET_REL:-1.0} \
    +trainer.dynamic_sampling.sample_score_weights.align=${SAMPLE_SCORE_ALIGN:-1.0} \
    +trainer.dynamic_sampling.sample_score_weights.learn=${SAMPLE_SCORE_LEARN:-0.5} \
    +trainer.dynamic_sampling.sample_score_weights.curv=${SAMPLE_SCORE_CURV:-0.5} \
    +trainer.dynamic_sampling.sample_score_weights.age=${SAMPLE_SCORE_AGE:-0.05} \
    2>&1 | tee "${CKPTS_DIR}/train.log"
