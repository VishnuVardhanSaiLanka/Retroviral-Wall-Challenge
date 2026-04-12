# Current Benchmark

As of `2026-04-12`, the benchmark for honest cross-family generalization is:

- workflow: `experiments/breakthrough.py`
- model: `winner_three_model_blend`
- objective: LOFO informative macro-F1
- score: **`0.7884`**
- previous score: `0.7217` (hybrid_lr)

Per-family F1:

| Family | F1 |
|--------|:--:|
| Group_II_Intron | 1.000 |
| Retroviral | 0.737 |
| Retron | 0.750 |
| LTR_Retrotransposon | 0.667 |

Secondary metrics:

- Retroviral TP of 12: `7`
- total features used: 33 (Model A, C) + 88 (Model B) across 3 blended models

## How to Reproduce

```bash
source venv/bin/activate
python experiments/breakthrough.py
```

The `winner_three_model_blend` row in `outputs/breakthrough/summary.csv` should show
`lofo_macro_f1_informative = 0.7884`.

## Method

Three-model probability blend:

- 45% — LR(C=0.3, balanced) on 33 features (6 gates + 27 handcrafted)
- 40% — LR(C=0.3, balanced) on 88 non-foldseek handcrafted features
- 15% — KNN(k=3, distance-weighted) on 33 features

Threshold: F1-optimal on blended training probabilities.

## Interpretation

- This benchmark supersedes the previous `hybrid_lr = 0.7217`
- It is the baseline all future improvements should beat
- It should not be mutated implicitly by unrelated experiments

## Canonical Artifacts

- `benchmarks/current_winner.json`
- `outputs/breakthrough/summary.csv`
- `outputs/breakthrough/winner_three_model_blend_predictions.csv`
