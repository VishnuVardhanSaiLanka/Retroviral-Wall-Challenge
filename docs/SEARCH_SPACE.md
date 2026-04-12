# Search Space

## Objective

Find a model that generalizes to unseen RT families for prime-editing activity prediction, with strong scientific interpretability and defensible validation.

Primary target:

- LOFO informative macro-F1

Secondary targets:

- Retroviral holdout TP
- overall F1 / AUC
- ranking quality among active RTs
- simplicity and publishability

## Constraints

- only 57 labeled RTs
- strong family confounding
- public leaderboard likely benchmark-maxable
- hidden evaluation likely rewards transfer beyond family identity
- model should remain scientifically explainable

## Search Space

### Feature Tiers

#### Tier 1: family-robust mechanistic features

- catalytic motif / acidic-cluster geometry
- local active-site exposure and contact features
- processivity / thumb electrostatics
- global foldability / solubility / mild thermostability
- size / burden

These are the main scientific feature candidates.

#### Tier 2: mixed but biologically plausible features

- hairpin / thumb architecture features
- structure-confidence summaries
- fusion-context geometry features that do not directly depend on MMLV/HIV similarity

These require ablation and caution.

#### Tier 3: leakage-prone or family-identity features

- FoldSeek family-reference TM scores
- raw ESM embeddings
- explicit family labels
- gate fallbacks that directly reward similarity to known retroviral winners

These should be treated as diagnostic controls, not primary scientific features.

### Model Families

#### Simple statistical

- ridge logistic regression
- elastic-net logistic regression
- sparse logistic regression

#### Small mechanistic

- gate scorecards
- monotonic additive models
- two-stage filter + linear score

#### Compact hybrid

- gate outputs + 8-15 vetted residual features
- logistic regression final layer

#### Higher-risk models

- tree ensembles
- boosted trees
- large feature stacks

Use only if they win repeatedly under strict nested LOFO.

## Current Best Hypothesis

The best publishable solution is probably one of:

1. a small mechanistic classifier on 8-15 biology-grounded features
2. a compact hybrid of gate summaries plus a sparse residual logistic model

The current 33-feature `hybrid_lr` is a useful stepping stone, not necessarily the final scientific model.

## Most Promising Simpler Directions

### 1. Small biology-first classifier

Try a manually curated feature set centered on:

- `triad_found_bin`
- `triad_best_rmsd`
- `D1_D2_dist`
- `D2_D3_dist`
- `sasa_avg`
- `sasa_high_pct`
- one pocket H-bond / salt metric
- `thumb_surface_net_charge`
- `thumb_charge_ratio`
- `instability_index`
- `camsol_score`
- `t40_raw` / `t45_raw`
- `protein_length_aa`

Model:

- logistic regression with strong regularization

Why:

- simple
- biology-grounded
- easier to defend in a paper

### 2. Recalibrated gate model

Keep the gate abstraction, but:

- strip MMLV/HIV similarity fallbacks from gate 2 and gate 5
- reduce gate count if some are redundant
- recalibrate outputs using only inner-fold training data
- test whether a smaller gate set performs as well as five gates

Why:

- keeps mechanistic story
- may recover generalization if leakage-like gate supports are removed

### 3. Sparse hybrid

Use:

- 4-6 gate summaries
- 4-8 residual features selected by nested stability
- linear final model

Why:

- likely keeps most of the current hybrid gain
- much simpler than a broad feature stack

## Experiments That Matter Most

### Leakage controls

- remove all FoldSeek reference-family features
- remove direct retroviral-reference gate supports
- compare to family-only and FoldSeek-only negative controls

### Mechanism-block ablations

Compare:

- catalytic only
- catalytic + developability
- catalytic + substrate
- catalytic + substrate + processivity
- all biology features

### Stability

- nested feature selection
- bootstrap coefficient stability
- per-family error analysis
- missingness audit

## Search Strategy

1. establish one strong simple baseline
2. evaluate one mechanistic model
3. evaluate one sparse hybrid
4. run leakage and mechanism-block ablations
5. only expand complexity if a more complex model wins repeatedly under nested LOFO

## Stop Conditions

Do not add complexity unless it provides:

- repeated LOFO improvement
- better per-family robustness
- a clearer scientific explanation

If a simpler model matches the larger one within noise, prefer the simpler model.
