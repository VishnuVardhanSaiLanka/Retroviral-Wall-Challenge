from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def compute_steric_clashes(query_coords: np.ndarray, env_coords: np.ndarray, clash_dist: float = 2.0, contact_dist: float = 4.0) -> tuple[int, int]:
    if query_coords.size == 0 or env_coords.size == 0:
        return 0, 0
    tree = cKDTree(env_coords)
    clashes = 0
    contacts = 0
    for coord in query_coords:
        contacts += len(tree.query_ball_point(coord, contact_dist))
        clashes += len(tree.query_ball_point(coord, clash_dist))
    return clashes, contacts


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))
