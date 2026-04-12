# Research Progress

## 2026-04-12

### Change

Added a family-robust analysis and comparison workflow to separate benchmark-maxing behavior from honest cross-family generalization:

- `experiments/split_diagnostics.py`
- `experiments/score_max_models.py`
- `experiments/generalization_workflow.py`
- `retroviral_wall/analysis/gate_audit.py`
- `retroviral_wall/tests/test_generalization_workflow.py`

Also exported the new analysis helper via:

- `retroviral_wall/analysis/__init__.py`

### Rationale

The public leaderboard behavior appears inconsistent with the challenge's scientific objective. The repo's existing gate pipeline was optimized around mechanistic scoring, but we needed to:

1. measure the gap between honest LOFO generalization and easier score-maxing regimes
2. identify leakage-prone feature families
3. compare three model families directly:
   - gates only
   - leakage-controlled handcrafted model
   - hybrid model combining gate scores with vetted residual features

### What Was Implemented

#### 1. Split diagnostics

Measured how performance changes under:

- LOFO
- LOO
- random 5-fold CV
- train-on-train memorization

Key conclusion: near-perfect scores only appear in memorization-style settings, not in honest held-out evaluation.

#### 2. Gate audit

Added a richer audit for the gate branch:

- per-gate AUC / F1
- per-family behavior
- family dependence via eta-squared
- confidence summaries
- failure-rate summaries
- pairwise gate correlation
- weakest-gate counts

#### 3. Leakage-controlled handcrafted branch

Built a feature-selection workflow that ranks handcrafted features by:

- activity signal
- family information leakage
- within-family stability
- leakage-adjusted score

This intentionally de-emphasizes family-proxy features, especially FoldSeek-derived family cues.

#### 4. Hybrid branch

Combined gate scores with the selected robust handcrafted residual features and evaluated the result under strict LOFO.

### Metrics

#### Previous saved pipeline

- Best saved LOFO informative macro-F1: `0.6382`

From:

- `retroviral_wall/outputs/predictions/evaluation_results.json`

#### New comparison workflow

From:

- `outputs/generalization/summary.csv`

Best current honest branch:

- `hybrid_lr`: `0.7217` LOFO informative macro-F1

Other notable branches:

- `handcrafted_no_foldseek_lr`: `0.6881`
- `robust_handcrafted_et`: `0.6339`
- `robust_handcrafted_lr`: `0.6208`
- `gates_lr`: `0.4869`

### Interpretation

- The current mechanistic gates are useful, but not strong enough alone.
- A simple hybrid of gate scores plus carefully chosen non-leaky residual features works substantially better.
- Removing the most suspicious FoldSeek family proxies does not destroy performance, which is encouraging for generalization.
- The benchmark likely rewards distribution matching more than real extrapolation, so the public leaderboard should not be treated as the primary scientific target.

### Files Touched

- `experiments/split_diagnostics.py`
- `experiments/score_max_models.py`
- `experiments/generalization_workflow.py`
- `retroviral_wall/analysis/gate_audit.py`
- `retroviral_wall/analysis/__init__.py`
- `retroviral_wall/tests/test_generalization_workflow.py`

### Validation

Executed:

```bash
venv/bin/pytest retroviral_wall/tests/test_generalization_workflow.py retroviral_wall/tests/test_known_rts.py
```

Result:

- `4 passed`

### Open Questions

1. Is the current hybrid gain coming from a small number of robust biological features or from mild residual family structure?
2. Can the gate definitions be simplified and recalibrated so that the mechanistic branch alone recovers more of the hybrid gain?
3. Would a simpler residual model outperform the current hybrid if we use only a very small biologically coherent feature set?
4. Which features remain predictive when each informative family is removed in turn during feature selection itself, not only during final evaluation?

### Next Hypotheses

1. A smaller, more interpretable hybrid using 8-15 biologically coherent features may match or beat the current 33-feature hybrid.
2. The current gate branch may be underperforming because the gate outputs are poorly calibrated rather than biologically wrong.
3. A sparse logistic model on a manually curated family-robust feature subset may outperform tree models and be easier to publish.

## 2026-04-12 - Simplified Search Pass

### Change

Implemented a simpler, more defensible search branch aimed at testing whether the current hybrid could be replaced by a compact mechanistic model:

- added configurable leakage ablations to:
  - `retroviral_wall/gates/gate2_fusion.py`
  - `retroviral_wall/gates/gate5_processivity.py`
- added:
  - `experiments/simplified_search.py`

This branch evaluates:

- `biology_lr`
- `gates_default_lr`
- `gates_clean_lr`
- `sparse_hybrid_clean_lr`

under nested LOFO.

### Rationale

Both biological and statistical review suggested that the search space should be simplified:

- remove direct similarity support where possible
- test a compact biology-first model
- compare that against a small gate-plus-linear hybrid
- prefer the simplest model that survives family holdout

### What Was Implemented

#### 1. Leakage-ablated gates

Added switches so that:

- fusion compatibility can run without FoldSeek similarity fallback
- processivity can run without MMLV/HIV similarity support and without literature prior support

This makes direct ablation possible without changing default behavior.

#### 2. Compact biology-first model

Tested a handpicked biology-grounded feature set focused on:

- catalytic motif geometry
- active-site exposure
- thumb/processivity electrostatics
- developability
- size burden

#### 3. Nested LOFO simplification search

For each outer held-out family:

- tune compact model settings on the training families only
- choose regularization / feature subset in the inner loop
- evaluate only on the held-out family

This is a more defensible test than a single globally chosen compact configuration.

### Metrics

From:

- `outputs/simplified_search/summary.csv`

Results:

- `sparse_hybrid_clean_lr`: `0.570` LOFO informative macro-F1
- `gates_default_lr`: `0.467`
- `gates_clean_lr`: `0.467`
- `biology_lr`: `0.460`

### Interpretation

This simplification pass did **not** beat the current best generalization branch (`hybrid_lr = 0.7217`).

That negative result matters:

- simplicity alone is not enough
- the current handpicked biology feature set is too weak or too incomplete
- removing similarity support from gates 2 and 5 did not materially change gate-only performance, which suggests the current gate weakness is not caused only by those similarity priors
- the best next move is likely a **smaller but better-chosen hybrid**, not a purely handpicked small model

### Files Touched

- `retroviral_wall/gates/gate2_fusion.py`
- `retroviral_wall/gates/gate5_processivity.py`
- `experiments/simplified_search.py`

### Validation

Executed:

```bash
venv/bin/pytest retroviral_wall/tests/test_generalization_workflow.py retroviral_wall/tests/test_known_rts.py
```

Result:

- `4 passed`

### Next Hypotheses

1. The right simplification target is a sparse hybrid, not a pure biology-only classifier.
2. The biology-first feature set likely needs better processivity / substrate features rather than fewer features.
3. Gate recalibration may matter more than gate leakage ablation.
4. The next search iteration should focus on nested stability selection over a biologically grouped residual set rather than manual minimalism.

## 2026-04-12 - Biology-Intensive Branch

### Change

Implemented a biology-intensive feature and gate redesign branch aimed at improving the frozen `0.7217` benchmark:

- added shared structural feature helpers in:
  - `retroviral_wall/utils/structural_features.py`
- updated:
  - `retroviral_wall/gates/gate2_fusion.py`
  - `retroviral_wall/gates/gate3_substrate_binding.py`
  - `retroviral_wall/gates/gate5_processivity.py`
- added:
  - `experiments/bio_intensive_search.py`

This branch introduced:

- substrate-path geometry features
- continuous surface-track/processivity features
- generic fusion-context features such as terminal accessibility and core compactness
- an automatic Retroviral false-negative audit for the best biology-intensive model

### Rationale

Biological review suggested the most likely path beyond the frozen benchmark was better measurement of:

- substrate-path geometry near the catalytic center
- duplex/processivity support
- prime-editor fusion-context compatibility

### Metrics

From:

- `outputs/bio_intensive/summary.csv`

Results:

- `bio_augmented_default_et`: `0.6257`
- `bio_augmented_default_lr`: `0.5444`
- `bio_augmented_clean_lr`: `0.5444`

Frozen benchmark remains:

- `hybrid_lr`: `0.7217`

### Interpretation

This branch did **not** beat the frozen benchmark.

Important takeaways:

- the new biology-heavy structural features were not strong enough yet to displace the benchmark hybrid
- Retroviral holdout remains the main failure mode
- the best biology-intensive model still leaned heavily on broad structure-quality and pocket proxies
- the new substrate/processivity features showed some directional signal but not enough to dominate model performance

### Retroviral Audit

From:

- `outputs/bio_intensive/retroviral_audit.json`

Best model:

- `bio_augmented_default_et`

Retroviral performance:

- `3` true positives
- `9` false negatives

Notable observation:

- several of the strongest TP/FN deltas were still generic features like Ramachandran summaries and pocket H-bond counts
- the newly added substrate/processivity path features showed only modest separation

### Files Touched

- `retroviral_wall/utils/structural_features.py`
- `retroviral_wall/gates/gate2_fusion.py`
- `retroviral_wall/gates/gate3_substrate_binding.py`
- `retroviral_wall/gates/gate5_processivity.py`
- `experiments/bio_intensive_search.py`

### Validation

Executed:

```bash
venv/bin/pytest retroviral_wall/tests/test_generalization_workflow.py retroviral_wall/tests/test_known_rts.py
```

Result:

- `4 passed`

### Next Hypotheses

1. The next gain is more likely to come from integrating better PE-specific biology into the benchmark hybrid than from replacing the benchmark with a fully new biology-heavy branch.
2. The new substrate/processivity features need better localization and stronger integration, not just presence.
3. Fusion-context modeling likely remains undermeasured.
4. Retroviral false negatives should be used as the main mechanistic debugging set for the next round.

## 2026-04-12 - Frozen Benchmark Augmentation With Corridor Features

### Change

Implemented a narrower augmentation path on top of the frozen `0.7217` winner instead of replacing it:

- extended `retroviral_wall/utils/structural_features.py` with localized catalytic-to-thumb corridor features
- added `experiments/augment_frozen_benchmark.py`

The augmentation workflow compares:

- frozen benchmark as-is
- frozen benchmark + full corridor block
- frozen benchmark + all 1/2/3-feature corridor subsets
- corridor-only control

### Rationale

The previous biology-heavy branch was too broad and underperformed. The next hypothesis was that the benchmark could improve if we added only a small number of tightly localized corridor features on top of the existing winning feature set.

### Metrics

From:

- `outputs/augment_frozen_benchmark/summary.csv`

Best result remains:

- `frozen_hybrid_lr`: `0.7217`

Best corridor-augmented ties:

- `frozen_plus_corridor_hydrophobic_tolerance_lr`: `0.7217`
- `frozen_plus_corridor_order_lr`: `0.7217`
- `frozen_plus_corridor_hydrophobic_tolerance_corridor_order_lr`: `0.7217`

No corridor variant exceeded the frozen benchmark on the primary metric.

### Interpretation

This is another informative negative result:

- the corridor idea is not obviously wrong, but the current implementation is not strong enough to raise LOFO informative macro-F1
- some corridor features can be added without harming the benchmark, but they do not provide measurable primary-metric gain
- the main bottleneck is still not solved, especially on Retroviral holdout

### Files Touched

- `retroviral_wall/utils/structural_features.py`
- `experiments/augment_frozen_benchmark.py`

### Next Hypotheses

1. The current structural abstractions are still too coarse to capture the real PE-specific transferable signal.
2. Further progress likely requires either:
  - sharper biological localization around substrate handling and fusion context, or
   - a different source of biological information than the current handcrafted + coarse structural abstractions.
3. The current frozen hybrid remains the practical benchmark and should stay in place until a materially better biology-driven feature block is found.

## 2026-04-12 - Two Literature-Backed Features

### Change

Implemented two literature-backed structural features and tested them only as augmentations to the frozen benchmark:

- `priming_shell_readiness`
  - motivated by the NAR 2023 RT priming paper
- `template_grip_positive_groove`
  - motivated by the Nature 2024 prime-editor structure paper

Files touched:

- `retroviral_wall/utils/structural_features.py`
- `experiments/augment_frozen_benchmark.py`

### Rationale

The 80/20 next step after exhausting broad in-repo feature engineering was to add a very small number of literature-backed mechanistic features on top of the frozen `0.7217` benchmark instead of broadening the feature space again.

### Metrics

From:

- `outputs/augment_frozen_benchmark/summary.csv`

Results:

- `frozen_hybrid_lr`: `0.7217` LOFO informative macro-F1
- `frozen_plus_template_grip_positive_groove_lr`: `0.7217`
- `frozen_plus_priming_shell_readiness_lr`: `0.6366`

Notable secondary effect:

- `frozen_plus_template_grip_positive_groove_lr` improved overall F1 from `0.7027` to `0.7220` while tying the primary metric.

### Interpretation

- `template_grip_positive_groove` appears to capture a weak but real useful signal.
- `priming_shell_readiness` in its current form is not strong enough and degrades the primary metric.
- The current feature regime remains difficult to improve on the primary metric, but literature-backed narrow augmentations can still produce non-harmful or secondary-metric-positive effects.

### Next Hypotheses

1. `template_grip_positive_groove` is worth keeping as a candidate auxiliary feature because it ties the best primary score and slightly improves overall F1.
2. `priming_shell_readiness` needs sharper localization or better calibration before it is useful.
3. Further literature-driven progress should come from one or two narrowly targeted PE-specific geometric features at a time, not broad feature blocks.
