"""
cka.py — CKA fidelity and L2 divergence utilities.

Linear CKA measures the structural alignment between teacher and student
probability output matrices. A drop in CKA signals that the student's
decision surface is diverging from the teacher under attack.
"""

import numpy as np


def compute_cka(X1: np.ndarray, X2: np.ndarray) -> float:
    """
    Linear Centered Kernel Alignment between two (n, c) matrices.

    Returns a scalar in [0, 1] where 1 = perfect alignment.
    Typical baseline on clean data: 0.75–0.92.
    Detection threshold θ = 0.10 drop from baseline.
    """
    X1 = X1 - X1.mean(0)
    X2 = X2 - X2.mean(0)
    dot  = np.linalg.norm(X1.T @ X2, 'fro') ** 2
    norm = np.linalg.norm(X1.T @ X1, 'fro') * np.linalg.norm(X2.T @ X2, 'fro')
    return float(dot / (norm + 1e-8))


def compute_l2_divergence(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Mean L2 norm between two probability matrices (n, c).
    Used by B3 baseline as an alternative fidelity trigger.
    """
    return float(np.mean(np.linalg.norm(p1 - p2, axis=1)))
