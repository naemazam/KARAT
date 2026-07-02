"""
E6 — CKA vs L2 Ablation Study
==============================
Compares two drift-detection triggers:
  CKA  : representation-space similarity (KARAT's proposed trigger)
  L2   : output-space divergence (B3 baseline trigger)

Key finding:
  - Under S1 (targeted): L2 appears competitive at t=2-8 because
    teacher/student co-move positively on targeted features,
    giving L2 an incidental signal. CKA dominates at t=10.
  - Under S2/S3 (broad/cascading): L2 divergence goes NEGATIVE
    (co-movement suppresses the signal) while CKA still fires.
    KARAT outperforms B3_L2 on all triggered S2/S3 timesteps.
  - Wilcoxon (one-sided CKA>L2) tests the aggregate claim.

This experiment uses seed=42, all 3 scenarios, all 6 timesteps.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import warnings
warnings.filterwarnings('ignore')

from src.attack import inject_adversarial
from src.metrics import precision_at_k, compute_cka, compute_l2_div
from src.karat import score_karat, score_l2_trigger

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

K_FRAC    = 0.05
SCENARIOS = ['S1_targeted', 'S2_coordinated', 'S3_cascading']
TIMESTEPS = [0, 2, 4, 6, 8, 10]


def run_e6(teacher_fn, student_fn, df_test, THETA, L2_THETA,
           BASE_CKA, BASE_L2, HIGH_IDX, results_dir=RESULTS_DIR):
    """
    teacher_fn  : callable(df) -> proba array
    student_fn  : callable(df) -> proba array
    df_test     : test dataframe
    THETA       : CKA trigger threshold
    L2_THETA    : L2 trigger threshold (from positive val drops)
    BASE_CKA    : clean-state CKA on df_test
    BASE_L2     : clean-state L2 on df_test
    HIGH_IDX    : index of 'High' class
    """
    def pak(df_q, scores):
        k = int(len(df_q) * K_FRAC)
        r = np.argsort(scores[:, HIGH_IDX])[::-1][:k]
        return (df_q['priority_label'].values[r] == 'High').mean()

    rows = []
    print(f"\n{'Scenario':<20} {'t':>4} | {'KARAT':>9} {'B3_L2':>9} "
          f"{'delta':>8} | {'cka_d':>7} {'trig':>5}")
    print("-" * 68)

    for scenario in SCENARIOS:
        for t in TIMESTEPS:
            df_t    = inject_adversarial(df_test, t, scenario, 42)
            tp      = teacher_fn(df_t)
            sp      = student_fn(df_t)
            cka_d   = BASE_CKA - compute_cka(tp, sp)
            l2_d    = compute_l2_div(tp, sp) - BASE_L2
            pak_cka = pak(df_t, score_karat(sp, tp, cka_d, THETA))
            pak_l2  = pak(df_t, score_l2_trigger(sp, tp, l2_d, L2_THETA))
            trig    = '✓' if cka_d > THETA else '·'
            rows.append({
                'scenario': scenario, 'timestep': t,
                'PAK_KARAT_CKA': round(pak_cka, 4),
                'PAK_B3_L2':     round(pak_l2, 4),
                'delta':         round(pak_cka - pak_l2, 4),
                'cka_drop':      round(cka_d, 4),
                'cka_triggered': int(cka_d > THETA),
                'l2_triggered':  int(l2_d > L2_THETA),
            })
            print(f"{scenario:<20} {t:>4} | {pak_cka:>9.4f} {pak_l2:>9.4f} "
                  f"{pak_cka-pak_l2:>+8.4f} | {cka_d:>7.4f} {trig:>5}")
        print()

    df_e6 = pd.DataFrame(rows)
    out = os.path.join(results_dir, 'E6_cka_vs_l2_ablation.csv')
    df_e6.to_csv(out, index=False)

    diff = df_e6['PAK_KARAT_CKA'].values - df_e6['PAK_B3_L2'].values
    if not np.all(diff == 0):
        _, p = wilcoxon(df_e6['PAK_KARAT_CKA'].values,
                         df_e6['PAK_B3_L2'].values,
                         alternative='greater')
        print(f"Wilcoxon CKA>L2: p={p:.4f} "
              f"{'*** SIGNIFICANT' if p < 0.05 else 'ns'}")
    print(f"Mean KARAT={df_e6['PAK_KARAT_CKA'].mean():.4f}  "
          f"Mean L2={df_e6['PAK_B3_L2'].mean():.4f}  "
          f"Delta={df_e6['delta'].mean():+.4f}")
    print(f"Saved → {out}")
    return df_e6


if __name__ == '__main__':
    print("Run E2_multiseed_main.py first to populate teacher/student/df_test in scope.")
