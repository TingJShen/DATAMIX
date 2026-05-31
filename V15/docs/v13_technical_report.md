# V13 Technical Report Archive

## Goal

V13 upgrades V11 from domain-level dynamic mixing to sample-level second-order reweighting, while keeping V11 as a read-only baseline. The new code lives in an isolated `v13` directory whose internal package name remains `dapo`, so existing launch style stays `python -m dapo.main_dapo`.

## Method Summary

V11 controls only the domain mixture weights:

```text
math / code / general -> batch quota -> global pool sampling
```

V13 keeps that domain mixture as a budget guardrail, but moves the main decision to the sample level:

```text
domain budget -> candidate samples -> sample Taylor score -> softmax sampling
```

The sample score is:

```text
S_i = target_rel_i + align_i + learn_i - curv_i + age_i
```

Each term is normalized inside its domain before weighting.

## Taylor Link

For a local anchor objective `F_t(theta)`, a sampling distribution `q` induces update direction:

```text
u_t(q) = sum_i q(i) g_i
```

A second-order expansion gives:

```text
F_t(theta - eta u)
~= F_t(theta)
 - eta grad(F_t)^T u
 + eta^2 / 2 * u^T H_t u
```

So the local improvement is approximated by:

```text
eta grad(F_t)^T u - eta^2 / 2 * u^T H_t u
```

The marginal sample value contains:

```text
first-order:  grad(F_t)^T g_i
second-order: g_i^T H_t u_t
```

V13 approximates these terms in a low-dimensional sketch space:

```text
align_i = z_i^T mean(z_anchor)
curv_i  = z_i^T C_t z_i
```

where `z_i` is the gradient sketch for sample `i`, and `C_t` is a PSD curvature proxy estimated from fixed shadow anchors.

## Anchor And Leakage Policy

V13 uses two data sources for control:

- Prompt-only target files inherited from V11. Only prompt fields are read.
- Fixed shadow anchors sampled from the training set by domain.

No test answers or benchmark scores enter the controller.

## Implementation Boundaries

- V11 source code is not modified.
- V13 starts as a copied V11 workspace.
- V12 curriculum phases are not part of the V13 method.
- Sparse instability signals such as format failure, KL spikes, and overlong counts are monitoring-only, not core scoring terms.

## Verification Checklist

After implementation, verify each item explicitly:

- Isolated `v13` directory exists and can run with package name `dapo`.
- `sample_taylor_v13` dispatches to `RayDAPOTrainerV13`.
- Shadow anchor indices are fixed and checkpointed.
- Candidate sampling is sample-level, no-replacement, and keeps unselected candidates available.
- Sample state tracks learn EMA and age by dataset index.
- Curvature proxy is symmetric PSD and produces non-negative `curv_i`.
- V13 logs its own metrics without removing V11 metrics.
- No V11 files were modified.
- No unused extra modules were added beyond the sampler, trainer, worker patch, docs, scripts, and tests needed by V13.
