# Final Research Summary

## Scope

This document consolidates the work completed so far on the Retroviral Wall challenge:

- what was tried
- what failed
- what partially worked
- what currently works best
- where the code for the best methodology lives

The focus is the scientific objective:

- honest cross-family generalization under leave-one-family-out evaluation

Not the public leaderboard.

## Current Best Result

The current best honest result in the repo is:

- model: `hybrid_lr`
- workflow: `experiments/generalization_workflow.py`
- objective: LOFO informative macro-F1
- score: `0.7217`

Secondary metrics:

- overall F1: `0.7027`
- overall AUC: `0.7751`
- Retroviral TP of 12: `7`
- feature count: `33`

Canonical benchmark files:

- [benchmarks/current_winner.json](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/benchmarks/current_winner.json:1)
- [docs/CURRENT_BENCHMARK.md](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/docs/CURRENT_BENCHMARK.md:1)

The score table that established this winner is:

- [outputs/generalization/summary.csv](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/outputs/generalization/summary.csv:1)

## Highest-Scoring Methodology

### Summary

The best method is not a pure mechanistic gates model and not a pure handcrafted model.

It is a hybrid:

1. compute the five mechanistic gate scores
2. select a leakage-controlled residual handcrafted feature set
3. concatenate gate scores + selected residual features
4. fit a regularized logistic regression model
5. evaluate under LOFO

This hybrid outperformed:

- gates only
- handcrafted only
- no-FoldSeek handcrafted baseline
- simplified biology-only models
- later biology-heavy replacement branches

### Why it won

The hybrid appears to benefit from two things simultaneously:

- gate summaries carry real mechanistic signal
- residual handcrafted features recover additional discriminative information the gates are not capturing

The best residual features in the frozen winner were not purely mechanistic. Many are broad structure-quality or local pocket-chemistry proxies. That makes the benchmark stronger empirically, even if it is not the cleanest biological story.

### Code references for the winning methodology

#### 1. Main workflow

The winning workflow is implemented in:

- [experiments/generalization_workflow.py](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/experiments/generalization_workflow.py:1)

Important functions:

- `select_low_leakage_features(...)`
- `build_model(...)`
- `evaluate_lofo(...)`
- `run_generalization_workflow(...)`

#### 2. Mechanistic gates used by the winner

Gate classes:

- [gate1_foldability.py](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/retroviral_wall/gates/gate1_foldability.py:1)
- [gate2_fusion.py](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/retroviral_wall/gates/gate2_fusion.py:1)
- [gate3_substrate_binding.py](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/retroviral_wall/gates/gate3_substrate_binding.py:1)
- [gate4_catalytic.py](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/retroviral_wall/gates/gate4_catalytic.py:1)
- [gate5_processivity.py](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/retroviral_wall/gates/gate5_processivity.py:1)

Shared gate interface:

- [base.py](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/retroviral_wall/gates/base.py:1)

#### 3. Gate diagnostics / audit helpers

- [gate_diagnostics.py](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/retroviral_wall/analysis/gate_diagnostics.py:1)
- [gate_audit.py](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/retroviral_wall/analysis/gate_audit.py:1)

#### 4. Winning output artifacts

- [summary.csv](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/outputs/generalization/summary.csv:1)
- [details.json](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/outputs/generalization/details.json:1)
- [hybrid_lr_lofo_predictions.csv](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/outputs/generalization/hybrid_lr_lofo_predictions.csv:1)

### Exact winning feature structure

The frozen benchmark uses:

- 6 gate score features
- 27 selected residual handcrafted features

Gate score features:

- `foldability_score`
- `fusion_compat_score`
- `fusion_compat_clash_score`
- `substrate_binding_score`
- `catalytic_score`
- `processivity_score`

Selected residual features:

- `ramachandran_outliers`
- `ramachandran_allowed`
- `pocket_hbonds_per_res`
- `ramachandran_favoured`
- `salt_per_res`
- `hydrophobic_per_res`
- `pocket_hbonds`
- `sasa_low_pct`
- `n_salt_bridges`
- `sasa_avg`
- `sasa_total`
- `hbonds_per_res`
- `thumb_charge_class_num`
- `perplexity`
- `avg_log_likelihood`
- `thumb_fident`
- `triad_found_bin`
- `triad_best_rmsd`
- `thumb_total_residues`
- `pct_E`
- `aromaticity`
- `pct_H`
- `D1_D2_dist`
- `best_s1_len`
- `molecular_weight`
- `lysine_k_charge`
- `aspartate_d_charge`

These are recorded in:

- [outputs/generalization/details.json](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/outputs/generalization/details.json:1)

### Method details

#### Step 1: Build gate scores

The gate pipeline computes interpretable mechanistic scores for:

- foldability
- fusion compatibility
- substrate binding
- catalytic competence
- processivity

These are combined with the precomputed handcrafted features and sequence metadata.

#### Step 2: Rank residual handcrafted features

Residual features are scored using a leakage-aware ranking that balances:

- activity signal
- family information content
- within-family stability

This is handled by `select_low_leakage_features(...)` in:

- [generalization_workflow.py](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/experiments/generalization_workflow.py:39)

The ranking report is:

- [feature_leakage_report.csv](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/outputs/generalization/feature_leakage_report.csv:1)

#### Step 3: Fit the hybrid logistic model

The winning model is logistic regression with:

- median imputation
- standard scaling
- class balancing
- regularization

This is implemented in:

- [generalization_workflow.py](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/experiments/generalization_workflow.py:82)

#### Step 4: Evaluate with LOFO

For each held-out family:

1. train on the remaining families
2. optimize the threshold on the training portion only
3. predict on the held-out family
4. aggregate all fold predictions

This is implemented in:

- [generalization_workflow.py](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/experiments/generalization_workflow.py:116)

## What We Tried

### 1. Existing repo gate pipeline

Result:

- best saved pipeline before new work: `0.6382`

Artifact:

- [evaluation_results.json](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/retroviral_wall/outputs/predictions/evaluation_results.json:1)

Outcome:

- decent baseline
- real mechanistic signal
- not strong enough by itself

### 2. Split diagnostics and score-max analysis

Implemented:

- [split_diagnostics.py](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/experiments/split_diagnostics.py:1)
- [score_max_models.py](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/experiments/score_max_models.py:1)

Purpose:

- understand why leaderboard-like scores could be very high
- separate honest LOFO from easier in-distribution or memorization regimes

Outcome:

- near-perfect scores appear only in memorization-like regimes
- true LOFO is much harder
- leaderboard-oriented optimization and hidden-data generalization are different problems here

### 3. Family-robust generalization workflow

Implemented:

- [generalization_workflow.py](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/experiments/generalization_workflow.py:1)

Outcome:

- produced the current winner `hybrid_lr = 0.7217`
- this is the best successful methodology so far

### 4. Simplified search

Implemented:

- [simplified_search.py](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/experiments/simplified_search.py:1)

Models tested:

- `biology_lr`
- `gates_default_lr`
- `gates_clean_lr`
- `sparse_hybrid_clean_lr`

Result:

- best simplified result: `0.570`

Outcome:

- failed to beat benchmark
- showed that ultra-minimal biology-only or sparse clean variants were too weak

### 5. Biology-heavy replacement branch

Implemented:

- [bio_intensive_search.py](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/experiments/bio_intensive_search.py:1)
- [structural_features.py](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/retroviral_wall/utils/structural_features.py:1)

Purpose:

- add substrate-path, processivity, and fusion-context structural features
- redesign the biology-critical gates

Best result:

- `bio_augmented_default_et = 0.6257`

Outcome:

- failed to beat benchmark
- improved biological intent more than predictive power
- Retroviral holdout remained weak

Artifacts:

- [outputs/bio_intensive/summary.csv](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/outputs/bio_intensive/summary.csv:1)
- [outputs/bio_intensive/retroviral_audit.json](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/outputs/bio_intensive/retroviral_audit.json:1)

### 6. Frozen benchmark augmentation with corridor features

Implemented:

- [augment_frozen_benchmark.py](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/experiments/augment_frozen_benchmark.py:1)

Purpose:

- add localized catalytic-to-thumb corridor features only on top of the winner

Outcome:

- no corridor variant beat the frozen benchmark
- best variants only tied it

Artifacts:

- [outputs/augment_frozen_benchmark/summary.csv](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/outputs/augment_frozen_benchmark/summary.csv:1)

### 7. Two literature-backed features

Added:

- `priming_shell_readiness`
- `template_grip_positive_groove`

Motivated by:

- RT priming literature
- prime-editor structure literature

Outcome:

- `template_grip_positive_groove` tied the primary benchmark and slightly improved overall F1
- `priming_shell_readiness` degraded performance

Interpretation:

- very small literature-backed additions can be non-harmful and sometimes useful
- but still no primary-metric improvement beyond `0.7217`

## What Failed

These approaches failed to beat the benchmark:

- gates-only variants
- pure biology-first compact model
- simplified sparse hybrid
- broad biology-heavy replacement branch
- full corridor feature augmentation
- literature-inspired priming feature in its current implementation

What these failures suggest:

- the search space based on coarse internal structural abstractions is close to saturation
- broad new feature blocks tend to add noise faster than signal
- the benchmark gains come from a mix of mechanistic summaries and broad scaffold-quality proxies that are hard to replace cleanly

## What Succeeded

The successful moves were:

- introducing a leakage-aware family-robust evaluation workflow
- combining mechanistic gate scores with residual handcrafted features
- using regularized logistic regression as the final integrator
- freezing the current winner and evaluating all later ideas against it

Partial successes:

- `template_grip_positive_groove` tied the best primary score and improved overall F1
- some corridor features could be added without catastrophic harm

## External Research Used For Inspiration

The main literature-to-feature mapping is documented in:

- [LITERATURE_FEATURE_MAP.md](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/docs/LITERATURE_FEATURE_MAP.md:1)

Key directions taken from literature:

- PE extension-path geometry
- RT priming/initiation readiness
- substrate/edit-path tolerance

## Current Judgment

The current best methodology is the frozen `hybrid_lr` benchmark.

At this point:

- it has survived multiple challenger branches
- no replacement or augmentation has exceeded it on the primary scientific metric
- the remaining path forward inside the current regime is narrow

That likely means:

- the current feature regime is close to exhausted
- further gains will probably require either:
  - sharper PE-specific structural measurements
  - new external biological information
  - or more/better data

## Where To Start If We Resume

If work resumes later, start here:

1. read [benchmarks/current_winner.json](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/benchmarks/current_winner.json:1)
2. inspect [outputs/generalization/summary.csv](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/outputs/generalization/summary.csv:1)
3. inspect [outputs/generalization/details.json](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/outputs/generalization/details.json:1)
4. treat `hybrid_lr = 0.7217` as the baseline to beat
5. avoid re-running broad failed branches before reading:
   - [RESEARCH_PROGRESS.md](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/docs/RESEARCH_PROGRESS.md:1)
   - [LITERATURE_FEATURE_MAP.md](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/docs/LITERATURE_FEATURE_MAP.md:1)

## Related Docs

- [RESEARCH_PROGRESS.md](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/docs/RESEARCH_PROGRESS.md:1)
- [SEARCH_SPACE.md](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/docs/SEARCH_SPACE.md:1)
- [CURRENT_BENCHMARK.md](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/docs/CURRENT_BENCHMARK.md:1)
- [LITERATURE_FEATURE_MAP.md](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/docs/LITERATURE_FEATURE_MAP.md:1)
