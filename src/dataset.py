"""Synthetic KATS (Knowledge-Aware Triage Service) dataset generator."""
import numpy as np
import pandas as pd

KATS_FEATURES = [
    'az_risk_score', 'migration_complexity', 'service_criticality',
    'bandwidth_required_mbps', 'downstream_critical', 'regulatory_flag',
    'dependency_count', 'latency_sensitivity'
]


def generate_kats_syn(n: int = 75000, seed: int = 42,
                      high_frac: float = 0.30,
                      medium_frac: float = 0.40) -> pd.DataFrame:
    """
    Generate a synthetic cloud-service triage dataset.

    Parameters
    ----------
    n          : total number of services
    seed       : random seed for reproducibility
    high_frac  : fraction of High-priority services
    medium_frac: fraction of Medium-priority services

    Returns
    -------
    pd.DataFrame with KATS_FEATURES columns and 'priority_label'
    """
    rng = np.random.RandomState(seed)
    n_high = int(n * high_frac)
    n_med  = int(n * medium_frac)
    n_low  = n - n_high - n_med

    def _block(size, risk_mu, crit_mu, noise=0.12):
        return {
            'az_risk_score':         np.clip(rng.normal(risk_mu, noise, size), 0, 1),
            'migration_complexity':  np.clip(rng.normal(crit_mu * 0.9, noise, size), 0, 1),
            'service_criticality':   np.clip(rng.normal(crit_mu, noise, size), 0, 1),
            'bandwidth_required_mbps': np.clip(rng.normal(crit_mu * 0.8, 0.15, size), 0, 1),
            'downstream_critical':   (rng.random(size) < crit_mu * 0.85).astype(int),
            'regulatory_flag':       (rng.random(size) < crit_mu * 0.60).astype(int),
            'dependency_count':      np.clip(rng.normal(crit_mu * 10, 1.5, size), 0, 12).astype(int),
            'latency_sensitivity':   np.clip(rng.normal(crit_mu, 0.10, size), 0, 1),
        }

    high_d = _block(n_high, risk_mu=0.78, crit_mu=0.82)
    med_d  = _block(n_med,  risk_mu=0.50, crit_mu=0.50)
    low_d  = _block(n_low,  risk_mu=0.22, crit_mu=0.20)

    rows = []
    for d, label in [(high_d, 'High'), (med_d, 'Medium'), (low_d, 'Low')]:
        size = len(d['az_risk_score'])
        df_b = pd.DataFrame(d)
        df_b['priority_label'] = label
        rows.append(df_b)

    df = pd.concat(rows, ignore_index=True)
    shuffle_idx = rng.permutation(len(df))
    return df.iloc[shuffle_idx].reset_index(drop=True)
