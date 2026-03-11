# Basic Experiment Setup

Run baseline models with both challenge-required validation protocols:

- Leave-One-Family-Out (LOFO)
- Leave-One-Out (LOO)

## Run

```bash
python experiments/basic_experiments.py
```

Optional arguments:

```bash
python experiments/basic_experiments.py --data-dir data --output-dir outputs/basic_experiments
```

## Output artifacts

- `outputs/basic_experiments/summary_metrics.csv`
- `outputs/basic_experiments/<model>_lofo_predictions.csv`
- `outputs/basic_experiments/<model>_loo_predictions.csv`

## Included baseline models

- `all_inactive`
- `handcrafted_logreg`
- `handcrafted_rf`
- `esm_ridge`
