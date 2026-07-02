"""Evaluation metrics: Precision@K, CKA, L2 divergence."""
import numpy as np


def precision_at_k(df, scores: np.ndarray, k_frac: float = 0.05,
                   label_col: str = 'priority_label',
                   pos_label: str = 'High') -> float:
    """
    Precision at top-K fraction of scored instances.

    Parameters
    ----------
    df      : DataFrame with ground-truth label_col
    scores  : (n, n_classes) probability array
    k_frac  : fraction of dataset to treat as top-K
    Returns
    -------
    float in [0, 1]
    """
    classes = sorted(df[label_col].unique())
    high_idx = classes.index(pos_label)
    k = max(1, int(len(df) * k_frac))
    rank = np.argsort(scores[:, high_idx])[::-1][:k]
    return float((df[label_col].values[rank] == pos_label).mean())


def compute_cka(X1: np.ndarray, X2: np.ndarray) -> float:
    """
    Linear Centred Kernel Alignment between two probability matrices.

    CKA(X1, X2) = ||X1^T X2||_F^2 / (||X1^T X1||_F * ||X2^T X2||_F)

    Both matrices are mean-centred before computation.
    Returns value in [0, 1]; 1 = identical representations.
    """
    X1 = X1 - X1.mean(axis=0)
    X2 = X2 - X2.mean(axis=0)
    dot  = np.linalg.norm(X1.T @ X2, 'fro') ** 2
    norm = (np.linalg.norm(X1.T @ X1, 'fro') *
            np.linalg.norm(X2.T @ X2, 'fro'))
    return float(dot / (norm + 1e-8))


def compute_l2_divergence(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Mean row-wise L2 distance between two probability matrices.
    A positive value means p2 is further from p1 than the baseline.
    """
    return float(np.mean(np.linalg.norm(p1 - p2, axis=1)))
