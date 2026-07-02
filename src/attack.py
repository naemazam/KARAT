"""Adversarial injection functions for S1/S2/S3 scenarios."""
import numpy as np
import pandas as pd
from typing import Literal

Scenario = Literal['S1_targeted', 'S2_coordinated', 'S3_cascading']

_NOISE_FEATURES = [
    'az_risk_score', 'migration_complexity',
    'service_criticality', 'bandwidth_required_mbps'
]
_FLIP_FEATURES = ['downstream_critical', 'regulatory_flag']


def inject_adversarial(
    df_clean: pd.DataFrame,
    timestep: int,
    scenario: Scenario = 'S1_targeted',
    seed: int = 42,
    label_col: str = 'priority_label',
) -> pd.DataFrame:
    """
    Inject adversarial label-drift into a clean snapshot at a given timestep.

    Parameters
    ----------
    df_clean  : clean DataFrame (unmodified)
    timestep  : attack intensity step (0 = no attack, 10 = maximum)
    scenario  : one of S1_targeted / S2_coordinated / S3_cascading
    seed      : random seed (vary across experimental seeds)
    label_col : name of the ground-truth priority column

    Returns
    -------
    New DataFrame with corrupted features; labels are preserved.
    """
    rng = np.random.RandomState(seed + timestep * 17)
    df_t = df_clean.copy()
    n = len(df_clean)
    intensity = timestep / 10.0

    if intensity == 0:
        return df_t

    if scenario == 'S1_targeted':
        high_mask = (df_clean[label_col] == 'High').values
        n_att_high = int(high_mask.sum() * 0.75 * intensity)
        n_att_low  = int(n * 0.05 * intensity)
        att_high   = rng.choice(np.where(high_mask)[0],
                                 n_att_high, replace=False)
        att_other  = rng.choice(np.where(~high_mask)[0],
                                 min(n_att_low, (~high_mask).sum()),
                                 replace=False)
        attacked   = np.concatenate([att_high, att_other])
        noise_scale = 0.40

    elif scenario == 'S2_coordinated':
        n_att    = int(n * 0.70 * intensity)
        attacked = rng.choice(n, n_att, replace=False)
        noise_scale = 0.55

    else:  # S3_cascading
        dep      = df_clean['dependency_count'].values
        seed_att = np.argsort(dep)[::-1][:int(n * 0.15)]
        cascade  = np.where(df_clean['downstream_critical'].values == 1)[0]
        n_casc   = int(len(cascade) * 0.70 * intensity)
        casc_sel = (rng.choice(cascade, min(n_casc, len(cascade)), replace=False)
                    if n_casc > 0 and len(cascade) > 0
                    else np.array([], dtype=int))
        boundary = rng.choice(n, int(n * 0.10 * intensity), replace=False)
        attacked = np.unique(np.concatenate([seed_att, casc_sel, boundary]))
        noise_scale = 0.50

    attacked = attacked[attacked < n]
    n_att    = len(attacked)
    df_t['under_attack'] = 0
    df_t.loc[attacked, 'under_attack'] = 1

    for feat in _NOISE_FEATURES:
        noise = rng.uniform(0.15, noise_scale, n_att) * intensity
        df_t.loc[attacked, feat] = np.clip(
            df_t.loc[attacked, feat].values - noise, 0, 1)

    for feat in _FLIP_FEATURES:
        flip = rng.random(n_att) < (0.60 * intensity)
        vals = df_t.loc[attacked, feat].values.copy().astype(float)
        vals[flip] = 1.0 - vals[flip]
        df_t.loc[attacked, feat] = vals

    df_t[label_col] = df_clean[label_col].values
    return df_t
