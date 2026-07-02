"""
retriage.py — KARAT re-triage engine and baseline methods.

Four methods compared in E2:
    B1  Random        — random ranking (lower bound)
    B2  Static Rule   — az_risk + service_criticality, scored once at t=0
    B3  L2-Trigger    — re-triage triggered by L2 divergence
    KARAT CKA-Trigger — re-triage triggered by CKA fidelity drop (proposed)
"""

import numpy as np
import pandas as pd


# ── KARAT: CKA-triggered re-triage ────────────────────────────────────────────

def karat_retriage(
    df_t: pd.DataFrame,
    scores: np.ndarray,
    cka_drop: float,
    theta: float = 0.10,
) -> np.ndarray:
    """
    Adaptive re-triage: elevate High-risk services when CKA drop exceeds θ.

    Flagging criterion:
        az_risk_score > 70th percentile
        AND downstream_critical == 1
        AND service_criticality > 60th percentile

    Alpha schedule:  max(1.1, 1.4 - (drop - θ) × 1.5)
    Gentle amplification to avoid over-promotion at extreme drops.
    """
    revised = scores.copy()
    if cka_drop <= theta:
        return revised

    alpha = max(1.1, 1.4 - (cka_drop - theta) * 1.5)

    az_thresh  = df_t['az_risk_score'].values > df_t['az_risk_score'].quantile(0.70)
    downstream = df_t['downstream_critical'].values == 1
    high_sc    = df_t['service_criticality'].values > df_t['service_criticality'].quantile(0.60)

    for pos in np.where(az_thresh & downstream & high_sc)[0]:
        revised[pos, 2] = min(revised[pos, 2] * alpha, 1.0)

    return revised


# ── CKA blend correction (used in final clean-architecture variant) ────────────

def cka_correct(
    df_t: pd.DataFrame,
    s_scores: np.ndarray,
    t_scores: np.ndarray,
    cka_drop: float,
    theta: float = 0.10,
    high_col: int = 0,
) -> np.ndarray:
    """
    Blend student scores toward teacher, proportional to CKA drop severity.
    Additionally elevates flagged high-risk services.
    """
    if cka_drop <= theta:
        return s_scores.copy()

    blend     = min((cka_drop - theta) / 0.15, 0.9)
    corrected = (1 - blend) * s_scores + blend * t_scores

    az    = df_t['az_risk_score'].values > df_t['az_risk_score'].quantile(0.70)
    ds    = df_t['downstream_critical'].values == 1
    hsc   = df_t['service_criticality'].values > df_t['service_criticality'].quantile(0.60)
    alpha = max(1.05, 1.2 - cka_drop * 0.5)

    for pos in np.where(az & ds & hsc)[0]:
        corrected[pos, high_col] = min(corrected[pos, high_col] * alpha, 1.0)

    return corrected


# ── Baselines ──────────────────────────────────────────────────────────────────

def baseline_random(df_t: pd.DataFrame, seed: int = 42) -> np.ndarray:
    """B1: purely random service ranking."""
    rng = np.random.RandomState(seed)
    return rng.dirichlet(np.ones(3), size=len(df_t))


def baseline_static_rule(df_t: pd.DataFrame) -> np.ndarray:
    """
    B2: static rule using az_risk_score + service_criticality.
    Score at t=0 on clean data, never updated.
    """
    n     = len(df_t)
    az    = df_t['az_risk_score'].values
    sc    = df_t['service_criticality'].values
    rule  = (az / (az.max() + 1e-8)) * 0.6 + (sc / (sc.max() + 1e-8)) * 0.4
    scores         = np.zeros((n, 3))
    scores[:, 2]   = rule
    scores[:, 1]   = (1 - rule) * 0.8
    scores[:, 0]   = np.clip(1 - scores[:, 1] - scores[:, 2], 0, 1)
    row_sums       = scores.sum(axis=1, keepdims=True)
    return scores / (row_sums + 1e-8)


def baseline_l2_trigger(
    df_t: pd.DataFrame,
    teacher_scores: np.ndarray,
    l2_drop: float,
    theta_l2: float = 0.01,
) -> np.ndarray:
    """B3: same re-triage logic as KARAT but triggered by L2 divergence."""
    revised = teacher_scores.copy()
    if l2_drop <= theta_l2:
        return revised

    alpha      = max(1.1, 1.4 - (l2_drop - theta_l2) * 15)
    az_thresh  = df_t['az_risk_score'].values > df_t['az_risk_score'].quantile(0.70)
    downstream = df_t['downstream_critical'].values == 1
    high_sc    = df_t['service_criticality'].values > df_t['service_criticality'].quantile(0.60)

    for pos in np.where(az_thresh & downstream & high_sc)[0]:
        revised[pos, 2] = min(revised[pos, 2] * alpha, 1.0)

    return revised
