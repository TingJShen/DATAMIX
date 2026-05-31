# V15 Training Attempts

This V15 code snapshot corresponds to three server-side training attempts under:

```text
5090_Lian:/zhdd/home/tjshen/260415_ArcherA100/v15/output_5090_Lian_v15/ArcherCodeR-V15-Qwen3-2B-5090Lian
```

## Attempt 1

```text
train_v15_qwen3_2b_3gpu_1_3_5_bsz30_save10_100_v15_1
```

- Label: `attempt 1`
- GPU set: `1,3,5`
- Observed size on server: `19K`
- Directory timestamp: `2026-05-15 04:22`

## Attempt 2

```text
train_v15_qwen3_2b_3gpu_2_3_5_bsz30_save10_100_v15_2
```

- Label: `attempt 2`
- GPU set: `2,3,5`
- Observed size on server: `19K`
- Directory timestamp: `2026-05-15 04:59`

## Attempt 3

```text
train_v15_qwen3_2b_3gpu_2_3_5_bsz30_save10_100_v15_3
```

- Label: `attempt 3`
- GPU set: `2,3,5`
- Observed size on server: `169G`
- Directory timestamp: `2026-05-22 08:43`
- Contains checkpoints/merged checkpoints observed through `global_step_10`, `100`, `200`, `300`, `400`, and `500`.
