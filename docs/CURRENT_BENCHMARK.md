# Current Benchmark

As of `2026-04-12`, the benchmark for honest cross-family generalization is:

- workflow: `experiments/new_breakthrough.py`
- model: `four_model_blend_w40_25_05_30`
- objective: LOFO informative macro-F1
- score: **`0.8231`**
- previous score: `0.7884` (3-model blend)
- original score: `0.7217` (hybrid_lr)

Per-family F1:

| Family | F1 |
|--------|:--:|
| Group_II_Intron | 1.000 |
| Retroviral | 0.737 |
| Retron | 0.889 |
| LTR_Retrotransposon | 0.667 |

Secondary metrics:

- Retroviral TP of 12: `7`
- total features used: 33 (Model A, C) + 88 (Model B) + 94 (Model D) across 4 blended models

## How to Reproduce

```bash
source venv/bin/activate
python experiments/new_breakthrough.py
```

The `four_model_blend_w40_25_05_30` row in `outputs/new_breakthrough/summary.csv` should show
`lofo_macro_f1_informative = 0.8231`.

## Method

Four-model Generative-Discriminative Multi-Paradigm Blend:

- 40% — Discriminative LR(C=0.3, balanced) on 33 features (6 gates + 27 handcrafted)
- 25% — Discriminative LR(C=0.3, balanced) on 88 non-foldseek handcrafted features
- 5%  — Non-parametric KNN(k=3, distance-weighted) on 33 features
- 30% — Generative LinearDiscriminantAnalysis(solver="lsqr") on all 94 features

Threshold: F1-optimal on blended training probabilities.

An alternative simpler formulation `0.15 * prob_hybrid_lr + 0.85 * prob_lda` achieves the exact same maximum macro-F1 score.

## Interpretation

- This benchmark supersedes the previous `3-model blend = 0.7884`
- It resolves the major weakness in predicting the Retron family by exploiting the multivariate structural relationships globally via LDA.
- It is the new baseline all future improvements should beat
- It should not be mutated implicitly by unrelated experiments

## Canonical Artifacts

- `benchmarks/current_winner.json`
- `outputs/new_breakthrough/summary.csv`
- `outputs/new_breakthrough/four_model_blend_w40_25_05_30_predictions.csv`