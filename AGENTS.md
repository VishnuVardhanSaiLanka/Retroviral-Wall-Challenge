# Project Working Rules

## Research Logging

- Every implementation change that improves the measured accuracy or target evaluation metric must be documented in `docs/`.
- Each such doc entry should record:
  - date
  - change made
  - rationale
  - metrics before and after
  - files touched
  - open questions / next hypotheses
- Prefer appending to an existing research progress log when the work is part of the same line of investigation.

## Research Framing

- Treat the research problem as searching for the optimal solution inside a relevant search space.
- Before deep optimization, explicitly define:
  - the search space
  - the objective
  - the constraints
  - the search strategy
- Prefer simpler explanations and simpler models when they achieve the objective competitively.
- Apply an 80/20 rule to research execution: do the 20% of work most likely to yield 80% of the result before broader or more expensive exploration.
