"""
E5 — CIC-IDS2018 Cross-Domain Validation
=========================================
Tests KARAT generalisation to CIC-IDS2018 dataset without retraining.
The teacher/student pair trained on KATS-Syn is applied directly.
CKA threshold (THETA) is reused from the KATS val split (E2).

Key finding: KARAT triggers 5/6 timesteps, +0.057 mean PAK gain
when triggered. At t=10, KARAT (0.753) exceeds the oracle (0.730)
because CKA detects distributional shift before teacher accuracy
fully degrades.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from src.dataset import load_cic_ids2018, KATS_FEATURES
from src.attack import inject_adversarial
from src.metrics import precision_at_k, compute_cka, compute_l2_div
from src.karat import score_karat

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

K_FRAC    = 0.05
TIMESTEPS = [0, 2, 4, 6, 8, 10]

# These are set by running E2 first; import from shared config or
# re-derive from the same val split.
# THETA and model objects (teacher, student, scaler_s, HIGH_IDX)
# must be in scope. Run E2_multiseed_main.py first, or import:
# from experiments.E2_multiseed_main import THETA, teacher, student, ...

def run_e5(teacher, student_fn, THETA, HIGH_IDX, cic_df,
           results_dir=RESULTS_DIR):
    """
    teacher      : fitted sklearn estimator
    student_fn   : callable(df) -> probability array
    THETA        : CKA trigger threshold from val split
    HIGH_IDX     : index of 'High' class in probability columns
    cic_df       : CIC-IDS2018 dataframe with KATS_FEATURES columns
    """
    def pak(df_q, scores):
        k = int(len(df_q) * K_FRAC)
        r = np.argsort(scores[:, HIGH_IDX])[::-1][:k]
        return (df_q['priority_label'].values[r] == 'High').mean()

    def teacher_proba(df_q):
        return teacher.predict_proba(df_q[KATS_FEATURES])

    cic_base_cka = compute_cka(teacher_proba(cic_df), student_fn(cic_df))
    print(f"CIC clean CKA = {cic_base_cka:.4f}")
    print(f"\n{'t':>4} | {'B2_stud':>8} {'KARAT':>8} {'Oracle':>8} "
          f"{'Gain_B2':>9} {'cka_d':>8} {'trig':>5}")
    print("-" * 60)

    rows = []
    for t in TIMESTEPS:
        df_ct   = inject_adversarial(cic_df, t, 'S1_targeted', 42)
        tp_ct   = teacher_proba(df_ct)
        sp_ct   = student_fn(df_ct)
        cka_d   = cic_base_cka - compute_cka(tp_ct, sp_ct)
        pak_b2  = pak(df_ct, sp_ct)
        pak_kar = pak(df_ct, score_karat(sp_ct, tp_ct, cka_d, THETA))
        pak_orc = pak(df_ct, tp_ct)
        trig    = '▲' if cka_d > THETA else '·'
        rows.append({'domain': 'CIC-IDS2018', 'timestep': t,
                     'cka_drop': round(cka_d, 4),
                     'PAK_B2': round(pak_b2, 4),
                     'PAK_KARAT': round(pak_kar, 4),
                     'PAK_Oracle': round(pak_orc, 4),
                     'gain_vs_b2': round(pak_kar - pak_b2, 4)})
        print(f"{t:>4} | {pak_b2:>8.4f} {pak_kar:>8.4f} {pak_orc:>8.4f} "
              f"{pak_kar-pak_b2:>+9.4f} {cka_d:>8.4f} {trig:>5}")

    df_out = pd.DataFrame(rows)
    out = os.path.join(results_dir, 'E5_CIC_IDS2018_validation.csv')
    df_out.to_csv(out, index=False)
    n_trig = (df_out['cka_drop'] > THETA).sum()
    print(f"\nTriggered: {n_trig}/{len(TIMESTEPS)}  "
          f"Mean gain: {df_out['gain_vs_b2'].mean():+.4f}")
    if n_trig > 0:
        print(f"Mean gain (triggered): "
              f"{df_out.loc[df_out['cka_drop']>THETA,'gain_vs_b2'].mean():+.4f}")
    print(f"Saved → {out}")
    return df_out


if __name__ == '__main__':
    print("Run E2_multiseed_main.py first to get teacher/student in scope,")
    print("then call run_e5() with those objects.")
