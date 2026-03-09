# Gate Architecture Reference

This document explains what the current pipeline actually does in code.

It focuses on three things:

1. The overall architecture.
2. What each gate computes.
3. What the gate output files mean.

## 1. Overall Architecture

The pipeline in [main.py](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/retroviral_wall/main.py) runs in this order:

1. Load sequence metadata from `data/rt_sequences.csv`.
2. Load precomputed handcrafted features from `data/handcrafted_features.csv`.
3. Build five gate scores for every RT.
4. Save all gate outputs to `retroviral_wall/outputs/gate_scores/all_gate_scores.csv`.
5. Run diagnostics on the gate scores.
6. Combine gate scores with one of several integrators:
   - `strict_veto`
   - `harmonic_mean`
   - `weaklink_support`
   - `two_stage_triage`
   - `majority_support`
   - `bayesian_lr`
   - `bart`
7. Evaluate each integrator with leave-one-family-out (LOFO) validation.
8. Save the best model's predictions to `retroviral_wall/outputs/predictions/submission.csv`.

Each gate returns one main score plus several sub-scores.

All gate scores are on a `0` to `1` scale:

- Near `1`: the gate thinks the RT looks favorable for that property.
- Near `0`: the gate thinks the RT looks unfavorable.
- Around `0.5`: weak evidence, fallback value, or missing-information default in many places.

Important caveat: these are not calibrated probabilities. They are heuristic mechanistic scores built from weighted proxies.

## 2. Shared Gate Output Format

Every gate inherits from [base.py](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/retroviral_wall/gates/base.py) and emits:

- `*_score`: the gate's main score.
- `*_confidence`: how trustworthy that gate's estimate is, based on available inputs.
- `*_failure_reason`: a short explanation when the gate detects an obvious problem.
- `*_<subscore>`: the internal components used to build the main score.

The main score for every gate is a weighted geometric mean of its sub-scores. In plain terms:

- every sub-score is first mapped to `0..1`
- each sub-score gets a weight
- weak sub-scores pull the overall gate score down strongly

That design matches the idea that activity can fail if one critical requirement fails badly.

## 3. Gate 1: Foldability

Implementation: [gate1_foldability.py](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/retroviral_wall/gates/gate1_foldability.py)

Question this gate asks:

"Does this RT look like a protein that can fold into a stable structure?"

Current implementation note:

This gate was expanded from mean/core pLDDT plus sequence proxies into a richer structure-confidence and compactness gate. It now uses multiple pLDDT-derived confidence features and a contact-density compactness feature from the current predicted structure.

### Inputs

- Structure-derived pLDDT from the RT PDB file.
- Structure-derived compactness from CA contact density.
- Handcrafted stability features.

### Sub-scores

- `foldability_plddt_mean`
  Mean CA-atom pLDDT across the whole structure, normalized to `0..1`.
- `foldability_plddt_core`
  Mean CA-atom pLDDT after trimming 20 residues from each end when possible. This tries to measure confidence in the structural core rather than flexible termini.
- `foldability_high_conf_frac`
  Fraction of residues with pLDDT at or above `0.70`.
- `foldability_low_conf_frac`
  In the main gate this is inverted so that fewer low-confidence residues helps the score.
- `foldability_longest_conf_segment`
  Fraction of the sequence occupied by the longest contiguous high-confidence segment.
- `foldability_contact_density`
  Contact-density compactness proxy derived from CA-CA neighborhood density in the predicted structure.
- `foldability_thermo_37`
  Average of `t40_raw` and `t45_raw`. If missing, defaults to `0.5`.
- `foldability_solubility`
  `sigmoid(camsol_score)`. Higher means more favorable predicted solubility.
- `foldability_instability`
  `sigmoid(-(instability_index - 40) / 10)`. Lower instability index gives a higher score.

### Main score

Weights:

- `plddt_mean`: `0.18`
- `plddt_core`: `0.12`
- `high_conf_frac`: `0.18`
- `low_conf_frac`: `0.10`
- `longest_conf_segment`: `0.12`
- `contact_density`: `0.10`
- `thermo_37`: `0.10`
- `solubility`: `0.06`
- `instability`: `0.04`

### Confidence and failure

- Confidence starts high and drops when fallback values like `0.5` had to be used.
- Failure reason:
  - `Low global pLDDT` if `plddt_mean < 0.55`
  - `Insufficient high-confidence folded core` if the confident folded region is too small
  - `Low thermostability proxy` if `thermo_37 < 0.35`

### How to read the score

- High score: the RT looks compact and contains a stronger high-confidence folded core.
- Low score: the RT appears fragmented, poorly compacted, or weakly supported by structural confidence.

## 4. Gate 2: Fusion Compatibility

Implementation: [gate2_fusion.py](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/retroviral_wall/gates/gate2_fusion.py)

Question this gate asks:

"Would this RT plausibly fit and function in the prime-editing fusion context?"

This gate is a proxy. It does not run a full physical docking simulation.

Current implementation note:

This gate was updated from a global heuristic to a coarse structural placement model in the prime-editor context. It now uses the actual PE cryo-EM reference structure (`8WUV`) and scores whether a candidate RT can be aligned into the RT position with tolerable clashes, plausible linker geometry, and a reasonably positioned active site.

### Inputs

- Prime-editor reference structure.
- Candidate RT PDB coordinates.
- Coarse structural alignment to the PE RT position.
- Sequence length.
- Catalytic motif/triad features as a weak fallback for active-site plausibility.

### Sub-scores

- `fusion_compat_clash_score`
  Explicit coarse clash score after placing the candidate RT into the PE context and counting steric clashes against Cas9 and the nucleic-acid environment.
- `fusion_compat_alignment_quality`
  Alignment quality of the candidate RT to the PE RT reference position, combined with the existing Foldseek similarity fallback.
- `fusion_compat_active_site_access`
  Measures whether the transformed candidate active-site-like region lands near the reference RT active site and remains reasonably positioned relative to the nucleic-acid path.
- `fusion_compat_linker_feasibility`
  Scores whether one terminus of the placed RT sits at a plausible distance from the inferred Cas9-RT anchor point in the PE structure.
- `fusion_compat_size_penalty`
  `1.0` up to length `800`, then exponentially decays for larger proteins.

### Main score

Weights:

- `clash_score`: `0.35`
- `alignment_quality`: `0.15`
- `active_site_access`: `0.25`
- `linker_feasibility`: `0.15`
- `size_penalty`: `0.10`

### Confidence and failure

- Confidence is higher when both structural alignment and active-site placement look credible.
- Failure reason:
  - `Likely steric incompatibility with Cas9 context`
  - `Poor linker geometry to Cas9 anchor`
  - `Active site poorly positioned in PE context`

### How to read the score

- High score: the RT can be placed into the PE geometry with fewer clashes and a more plausible anchor/access arrangement.
- Low score: the RT likely clashes with the PE environment, fits poorly into the RT position, or has poor anchor/access geometry.

## 5. Gate 3: Substrate Binding

Implementation: [gate3_substrate_binding.py](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/retroviral_wall/gates/gate3_substrate_binding.py)

Question this gate asks:

"Does this RT appear able to bind the nucleic-acid substrate in a productive way?"

This is also proxy-based. It uses feature-engineered signals rather than explicit substrate docking.

Current implementation note:

This gate was updated from a handcrafted pocket heuristic to a structure-based electropositive surface patch detector with a local catalytic-neighborhood constraint. It now follows the simplest common class of nucleic-acid binding predictors: identify exposed, contiguous, positively charged surface regions that could plausibly bind a nucleic-acid backbone, then prefer patches that sit near an acidic active-site-like region.

### Inputs

- RT PDB coordinates.
- Residue identity from the structure.
- Local CA-neighbor density as a surface-exposure proxy.
- Handcrafted catalytic motif/triad features for a weak active-site context prior.

### Sub-scores

- `substrate_binding_patch_detected`
  High when the structure contains an exposed contiguous patch of at least a few polar/positively charged residues with nontrivial positive charge density.
- `substrate_binding_patch_positive_density`
  Measures how enriched the best patch is for Lys/Arg/His-like positive charge.
- `substrate_binding_patch_exposure`
  Mean surface exposure proxy of the best patch. Exposure is estimated from low local CA contact count.
- `substrate_binding_patch_size`
  Rewards patches with enough residues to plausibly support nucleic-acid binding.
- `substrate_binding_patch_contiguity`
  Rewards dense local connectivity among residues in the patch.
- `substrate_binding_active_site_proximity`
  Rewards electropositive patches that sit near a candidate acidic catalytic neighborhood in the same structure, with a fallback to motif/triad-derived catalytic context if no structural acidic cluster is found.

### Main score

Weights:

- `patch_detected`: `0.15`
- `patch_positive_density`: `0.35`
- `patch_exposure`: `0.20`
- `patch_size`: `0.05`
- `patch_contiguity`: `0.05`
- `active_site_proximity`: `0.20`

### Confidence and failure

- Confidence is higher when the method finds an exposed patch confidently from the structure.
- Failure reason:
  - `No confident electropositive surface patch` when no plausible binding patch is found

### How to read the score

- High score: the RT has an exposed, contiguous, electropositive surface patch near a plausible catalytic neighborhood.
- Low score: the structure lacks a convincing active-site-adjacent positive patch or the candidate patch looks poorly positioned.

## 6. Gate 4: Catalytic Competence

Implementation: [gate4_catalytic.py](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/retroviral_wall/gates/gate4_catalytic.py)

Question this gate asks:

"Does the active site look capable of catalysis?"

Current implementation note:

This gate was updated from a simple motif-distance heuristic to a structure-based acidic-cluster detector. It now looks for an exposed active-site-like acidic cluster in the RT structure, scores its compactness and local catalytic neighborhood, and then blends that with motif-context and sequence-model quality.

### Inputs

- RT PDB coordinates.
- Local acidic-cluster geometry from the structure.
- Motif and motif-secondary-structure indicators from handcrafted features.
- Sequence-model perplexity proxy.

### Sub-scores

- `catalytic_acidic_cluster_detected`
  High when the structure contains a plausible local acidic cluster that could correspond to the catalytic carboxylates.
- `catalytic_acidic_cluster_compactness`
  Scores how tightly grouped the candidate acidic cluster is.
- `catalytic_acidic_cluster_exposure`
  Scores whether the acidic cluster is exposed enough to plausibly participate in catalysis.
- `catalytic_catalytic_neighborhood`
  Scores whether the acidic cluster sits in a local neighborhood enriched for catalytic-context residues such as aromatic and basic side chains.
- `catalytic_motif_context`
  Combines the precomputed catalytic Asp distance heuristics and local YXDD secondary-structure context.
- `catalytic_esm_if_quality`
  `exp(-max(perplexity - 5, 0) / 10)`. Lower perplexity gives a higher score.

### Main score

Weights:

- `acidic_cluster_detected`: `0.15`
- `acidic_cluster_compactness`: `0.25`
- `acidic_cluster_exposure`: `0.20`
- `catalytic_neighborhood`: `0.20`
- `motif_context`: `0.10`
- `esm_if_quality`: `0.10`

### Confidence and failure

- Confidence is higher when the structure contains a convincing acidic cluster with reasonable compactness.
- Failure reason:
  - `No confident catalytic acidic cluster`
  - `Catalytic cluster geometry not convincing`

### How to read the score

- High score: the RT contains an active-site-like acidic cluster with plausible catalytic geometry and neighborhood context.
- Low score: the structure lacks a convincing acidic catalytic cluster or the cluster looks poorly arranged.

## 7. Gate 5: Processivity

Implementation: [gate5_processivity.py](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/retroviral_wall/gates/gate5_processivity.py)

Question this gate asks:

"Does this RT look like it can remain productively engaged with substrate long enough to work well?"

Current implementation note:

This gate was updated from a thumb/hairpin heuristic to a structure-based processive-contact-path model. It now scores whether the RT contains an extended exposed positive surface path near the catalytic region, then combines that with structural similarity to known processive RT architectures.

### Inputs

- RT PDB coordinates.
- Extended positive surface path geometry from the structure.
- Structural similarity to MMLV or HIV-1 RT.
- Thumb-region charge as a weak supporting context feature.
- Optional literature-based processivity values for known RTs.

### Sub-scores

- `processivity_path_detected`
  High when the structure contains an exposed positive contact path long enough to plausibly support sustained nucleic-acid engagement.
- `processivity_path_positive_density`
  Measures positive-charge enrichment along the best candidate path.
- `processivity_path_extension`
  Measures how extended the path is along its principal axis.
- `processivity_path_catalytic_proximity`
  Rewards paths that sit near the catalytic region rather than elsewhere on the surface.
- `processivity_thumb_support`
  Weakly rewards thumb-region charge support when available.
- `processivity_structural_sim`
  `max(foldseek_TM_MMLV, foldseek_TM_HIV1)` with fallbacks. If literature processivity exists for that RT, the score becomes the average of structural similarity and a literature-derived score capped at `1.0`.

### Main score

Weights:

- `path_detected`: `0.15`
- `path_positive_density`: `0.20`
- `path_extension`: `0.20`
- `path_catalytic_proximity`: `0.15`
- `structural_sim`: `0.20`
- `thumb_support`: `0.10`

### Confidence and failure

- Confidence is higher when the structure contains a convincing extended contact path.
- Failure reason:
  - `No confident processive contact path`

### How to read the score

- High score: the RT has a plausible extended substrate-contact path and processive RT-like architecture.
- Low score: the structure lacks a convincing processive contact path or the path is poorly placed relative to the catalytic region.

## 8. What `all_gate_scores.csv` Contains

File: [all_gate_scores.csv](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/retroviral_wall/outputs/gate_scores/all_gate_scores.csv)

Each row is one RT.

The columns are:

- `rt_name`
- all foldability outputs
- all fusion compatibility outputs
- all substrate binding outputs
- all catalytic outputs
- all processivity outputs

You can read it at three levels:

1. Main gate scores
   These are the columns ending in `_score`. They are the features passed into the model integrator.
2. Confidence columns
   These show whether a gate relied on stronger evidence or on weak/missing-data fallback behavior.
3. Sub-score columns
   These explain why the main gate score was high or low.

## 9. How the Gate Outputs Become Final Predictions

The integrator in [integrator.py](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/retroviral_wall/calibration/integrator.py) only uses the main gate score columns:

- `foldability_score`
- `fusion_compat_score`
- `substrate_binding_score`
- `catalytic_score`
- `processivity_score`

Optional extra handcrafted residual features may also be appended if they are not too correlated with the gate scores.

Interpretable heuristic strategies tested:

- `strict_veto`
  Strong weak-link penalty that prioritizes filtering likely inactives.
- `harmonic_mean`
  Harmonic mean of the five gate scores with a soft veto on very weak gates.
- `weaklink_support`
  Blends the weakest gate, overall mean, number of supportive gates, and a catastrophe penalty.
- `two_stage_triage`
  Applies a mechanistic filter-first score, then ranks likely passers by stronger supporting gates.
- `majority_support`
  Counts how many gates support activity, then downweights catastrophic weak-link failures.

Learned fallback strategies:

- `bayesian_lr`
  In the current code this is a regularized logistic regression fallback, not a full Bayesian model.
- `bart`
  In the current code this is a random forest fallback, not a true BART implementation.

For each held-out family, the pipeline:

1. trains on the other families
2. predicts a continuous `predicted_score`
3. chooses a threshold on the training folds to maximize F1
4. converts the score to `predicted_active`

## 10. Output Files After Integration

### `submission.csv`

File: [submission.csv](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/retroviral_wall/outputs/predictions/submission.csv)

Columns:

- `rt_name`
- `predicted_active`
- `predicted_score`

Meaning:

- `predicted_score` is the final integrator output for that RT.
- `predicted_active` is the binary call after thresholding.

### `triage_report.csv`

File: [triage_report.csv](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/retroviral_wall/outputs/predictions/triage_report.csv)

This is the wet-lab-facing companion to `submission.csv`.

Columns include:

- `rt_name`
- `predicted_score`
- `predicted_active`
- `threshold_used`
- `weakest_gate`
- `weakest_gate_score`
- `low_gate_count`

Use it to identify both likely actives and the gate that looks most limiting for each RT.

### `strategy_comparison.json`

File: [strategy_comparison.json](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/retroviral_wall/outputs/predictions/strategy_comparison.json)

This stores the LOFO results for every tested integration strategy so you can compare conservative filter-first heuristics against the learned fallbacks.

### `evaluation_results.json`

File: [evaluation_results.json](/Users/vishnu_lanka/projects/Retroviral-Wall-Challenge/retroviral_wall/outputs/predictions/evaluation_results.json)

This contains:

- `primary_metric`
  The main challenge-style metric, LOFO macro-F1 over the four informative families.
- `secondary_metrics`
  Overall F1, AUC, confusion counts, and retroviral true positives.
- `ranking_quality`
  Rank correlation between predicted score and measured PE efficiency among active RTs.
- `per_family`
  Family-by-family LOFO performance.

## 11. Practical Reading Guide

If you want to inspect one RT, read the outputs in this order:

1. Look at the five main gate scores.
2. Check any low-scoring gate's `failure_reason`.
3. Look at that gate's sub-scores to see what specifically pulled it down.
4. Then compare the final `predicted_score` and `predicted_active`.

If you want to debug the model globally:

1. Compare the five main gate scores across known active vs inactive RTs.
2. Use `gate_diagnostics.json` to see which gates are predictive or misleading.
3. Improve gates before spending too much time on integrator tuning.
