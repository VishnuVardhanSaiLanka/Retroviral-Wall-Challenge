# Current Benchmark

As of `2026-04-12`, the frozen benchmark for honest cross-family generalization is:

- workflow: `experiments/generalization_workflow.py`
- model: `hybrid_lr`
- objective: LOFO informative macro-F1
- score: `0.7217`

Secondary metrics:

- overall F1: `0.7027`
- overall AUC: `0.7751`
- Retroviral TP of 12: `7`
- feature count: `33`

This benchmark is the current winner in the repo for the scientific objective.

Interpretation:

- it is the baseline all future biology-driven improvements should beat
- it is preferred over public leaderboard-oriented models
- it should not be mutated implicitly by unrelated experiments

Canonical artifact:

- `benchmarks/current_winner.json`

Source summary:

- `outputs/generalization/summary.csv`
