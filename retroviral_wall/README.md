# Mechanistic Gate Simulation Pipeline

This package implements a working version of the mechanistic multi-gate plan for RT activity prediction.

## What is implemented

- 5 interpretable gates in `retroviral_wall/gates/`:
  - foldability
  - fusion compatibility
  - substrate binding
  - catalytic competence
  - processivity
- External resource bootstrap in `download_external.py`.
- Calibration/integration strategies in `calibration/integrator.py`:
  - multiplicative
  - bayesian_lr (regularized logistic fallback)
  - bart (tree-based fallback)
- LOFO evaluation + ranking metrics in `calibration/evaluation.py`.
- Gate diagnostics in `analysis/gate_diagnostics.py`.
- End-to-end orchestration in `main.py`.

## Run

```bash
python -m retroviral_wall.main
```

Outputs are written to:

- `retroviral_wall/outputs/gate_scores/all_gate_scores.csv`
- `retroviral_wall/outputs/gate_scores/gate_diagnostics.json`
- `retroviral_wall/outputs/predictions/submission.csv`
- `retroviral_wall/outputs/predictions/evaluation_results.json`

## Notes

- This implementation is intentionally robust to missing external binaries (TM-align) and missing heavy Bayesian deps.
- Where full structural simulation is unavailable, biologically motivated proxies are used to keep the pipeline runnable and extensible.
