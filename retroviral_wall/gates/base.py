from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class GateResult:
    rt_name: str
    gate_name: str
    score: float
    sub_scores: dict
    confidence: float
    failure_reason: str


class AbstractGate(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute_scores(self, sequences: pd.DataFrame, structures_dir: str, handcrafted: pd.DataFrame, external_data: dict) -> list[GateResult]:
        raise NotImplementedError

    def score_matrix(self, results: list[GateResult]) -> pd.DataFrame:
        rows = []
        for r in results:
            row = {
                "rt_name": r.rt_name,
                f"{self.name}_score": r.score,
                f"{self.name}_confidence": r.confidence,
                f"{self.name}_failure_reason": r.failure_reason,
            }
            for k, v in r.sub_scores.items():
                row[f"{self.name}_{k}"] = v
            rows.append(row)
        return pd.DataFrame(rows)
