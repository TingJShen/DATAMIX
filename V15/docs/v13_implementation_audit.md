# V13 Implementation Audit

## Verification Date

2026-05-06

## Plan Item Checklist

| Plan item | Status | Evidence |
|---|---|---|
| Create isolated `v13` directory | Implemented | `v13/` contains its own `dapo`, `tests`, `docs`, `tools`, `worker_patch`, and scripts. |
| Keep V11 unmodified | Implemented | Static search found no `sample_taylor_v13`, `RayDAPOTrainerV13`, or `SampleTaylorBatchSampler` in the original `remote_v12_work/dapo/*.py`. |
| Preserve previous technical report | Implemented | `docs/v13_technical_report.md`. |
| Add `sample_taylor_v13` dispatch | Implemented | `dapo/main_dapo.py` dispatches to `RayDAPOTrainerV13`. |
| Add `RayDAPOTrainerV13` | Implemented | `dapo/dapo_ray_trainer_v13.py`. |
| Add `SampleTaylorBatchSampler` | Implemented | `dapo/sample_taylor_sampler.py`; tested candidate windows, no-replacement draw, state round trip, domain floor, and excluded shadow anchors. |
| Preserve V11 target prompt-only representation path | Implemented | V13 inherits V11 target text extraction and calls `_build_target_representations_with_vllm`. |
| Add fixed shadow anchors | Implemented | `RayDAPOTrainerV13._v13_init_shadow_anchor_indices`; saved/restored in dynamic checkpoint. |
| Keep shadow anchors out of main training sampler | Implemented | `SampleTaylorBatchSampler(exclude_indices=...)`; test verifies excluded indices never enter the training pool. |
| Sample-level scoring terms | Implemented | `target_rel`, `align`, `learn`, `curv`, and `age` are computed and normalized inside `_v13_compute_candidate_scores`. |
| Domain budget as guardrail | Implemented | `_v13_budget_weights` blends V11 external weights with top-mean candidate scores and applies a minimum floor. |
| Worker RPC `compute_v13_repr_and_grad_sketch` | Partially implemented | V13 worker patch exposes the RPC and returns embedding plus deterministic residual projection sketch. It does not yet compute exact per-sample gradients over `lm_head + last layer`. |
| PSD curvature proxy | Implemented | Shadow sketches build a symmetric `C = Z^T Z / n` EMA; `curv_i = z_i^T C z_i` is non-negative for the PSD proxy. |
| V13 checkpoint state | Implemented | Saves shadow anchors, learn EMA, last-seen steps, projection seed, curvature matrix, anchor mean sketch, domain budget, and inherited sampler state. |
| V13 logs | Implemented | Adds `dynamic/v13_*` metrics for weights, candidate counts, score components, pool state, and refresh flags. |
| V13 smoke script | Implemented | `dynamic_train_v13_a100_smoke_bsz4_20step.sh`. |
| V13 formal train script | Implemented | `dynamic_train_v13_a100_formal.sh`. |
| Merge/export script | Implemented | `tools/model_merge.sh` plus `tools/model_merge.py`. |

## Extra Module Check

Expected source modules kept in `v13`:

- `dapo/dapo_ray_trainer_v8.py`
- `dapo/dapo_ray_trainer_v11.py`
- `dapo/dapo_ray_trainer_v13.py`
- `dapo/dynamic_category_sampler.py`
- `dapo/main_dapo.py`
- `dapo/sample_taylor_sampler.py`
- `worker_patch/fsdp_workers.py`
- `tools/model_merge.py`
- `tools/model_merge.sh`
- V13 docs, scripts, and tests

Removed from the V13 source tree as extra for this method:

- `dapo/dapo_ray_trainer_v12.py`
- `dapo/curriculum_sampler.py`
- V12 curriculum sampler test
- Unrelated copied tool scripts such as transfer/eval/start-ray helpers

Generated `__pycache__` directories may exist from local `compileall`; they are not source modules.

## Verification Commands Run

```bash
python -m compileall dapo tests worker_patch tools
```

```bash
python manual import runner for:
tests/test_v13_sample_taylor_sampler.py
tests/test_v13_isolation_and_dispatch.py
```

Both commands completed successfully in the local workspace. `pytest` was not available in the local Python environment, so tests were executed through a small in-memory runner that imports the test files and calls each `test_` function.

## Remaining Technical Caveat

The only partial implementation is the exact worker-side gradient sketch. The trainer and RPC interface are in place, and the default sketch is deterministic and target-conditioned through the representation residual. If strict per-parameter `lm_head + last layer` gradients are required, the next step is to replace the worker RPC internals while keeping the same trainer interface.
