# V15 Code Snapshot

This directory is preserved in DATAMIX as the V15 server code snapshot.

The server-side V15 training had three numbered attempts. See `ATTEMPTS.md`
for the mapping of `attempt 1`, `attempt 2`, and `attempt 3`.

Original server path:

```text
5090_Lian:/zhdd/home/tjshen/260415_ArcherA100/v15
```

## Original Manifest

# V13 Lightweight Code Manifest

## Provenance

Base synced copy:

```text
D:\Codex_Sandbox\Huawei_Hard\remote_v12_work\v13
```

Overlay patch copy:

```text
D:\Codex_Sandbox\Huawei_Hard\remote_patch_v13_save_steps
```

The overlay updated these files or paths:

```text
dynamic_train_v13_5090_lian_qwen3_2b_4gpu.sh
dynamic_train_v13_a100_formal.sh
dynamic_train_v13_a100_smoke_bsz4_20step.sh
launch_v13_5090_lian_qwen3_2b_4gpu_fixpos.sh
dapo/dapo_ray_trainer_v8.py
verl/trainer/config/ppo_trainer.yaml
```

## Included Files

```text
dapo/dapo_ray_trainer_v11.py
dapo/dapo_ray_trainer_v13.py
dapo/dapo_ray_trainer_v8.py
dapo/dynamic_category_sampler.py
dapo/main_dapo.py
dapo/sample_taylor_sampler.py
docs/v11_to_v13_migration_plan_for_inner_ai.md
docs/v13_code_correspondence_report.md
docs/v13_implementation_audit.md
docs/v13_technical_report.md
dynamic_train_v13_5090_lian_qwen3_2b_4gpu.sh
dynamic_train_v13_5090_lian_qwen3_2b_5gpu.sh
dynamic_train_v13_5a100_qwen25_1_5b_1gpu.sh
dynamic_train_v13_5a100_qwen25_1_5b_2gpu.sh
dynamic_train_v13_a100_formal.sh
dynamic_train_v13_a100_smoke_bsz4_20step.sh
launch_v13_5090_lian_qwen3_2b_4gpu_fixpos.sh
monitor_launch_v13_5a100_qwen25_1_5b_1gpu.sh
tests/test_v13_isolation_and_dispatch.py
tests/test_v13_sample_taylor_sampler.py
tools/model_merge.py
tools/model_merge.sh
verl/trainer/config/ppo_trainer.yaml
verl/workers/fsdp_workers.py
worker_patch/fsdp_workers.py
```

## Excluded Content

This is a code-only archive. It excludes datasets, model outputs, checkpoints, Python caches, Ray runtime state, W&B files, and model weight formats.

Total included code snapshot at archive time:

```text
25 files
432263 bytes
```
