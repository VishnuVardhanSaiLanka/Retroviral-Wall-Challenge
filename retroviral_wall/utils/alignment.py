from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AlignmentResult:
    tm_score: float
    rmsd: float
    rotation: np.ndarray
    translation: np.ndarray
    aligned_atoms: np.ndarray
    active_site_center_transformed: np.ndarray | None


def structural_align(mobile_pdb: str, target_pdb: str | None = None, target_atoms: list | None = None, method: str = "proxy") -> AlignmentResult:
    # Lightweight proxy alignment. Uses existing FoldSeek TM features in gate logic.
    return AlignmentResult(
        tm_score=0.5,
        rmsd=5.0,
        rotation=np.eye(3),
        translation=np.zeros(3),
        aligned_atoms=np.zeros((0, 3)),
        active_site_center_transformed=None,
    )
