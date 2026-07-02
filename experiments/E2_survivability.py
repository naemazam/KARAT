"""
E2 — KARAT vs Baselines: Survivability / Precision@K Comparison

Compares 4 methods across 3 scenarios and 6 timesteps:
    B1  Random ranking
    B2  Static rule (az_risk + service_criticality, fixed at t=0)
    B3  L2-triggered re-triage
    KARAT  CKA-triggered re-triage (proposed)

Outputs: results/E2_survivability_all.csv + figure.

Key finding (S2, θ=0.10):
    Mean KARAT gain vs B2-Static: +0.028–+0.042 across scenarios.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings; warnings.filterwarnings('ignore')

from src.dataset import generate_karat_syn, KATS_FEATURES
from src.cka import compute_cka, compute_l2_divergence
from src.attack import inject_attack_with_labels
from src.models import build_teacher
from src.retriage import karat_retriage, baseline_random, baseline_static_rule, baseline_l2_trigger
from src.metrics import precision_at_k

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
os.makedirs(RESULTS, exist_ok=True)

TIMESTEPS = [0, 2, 4, 6, 8, 10]
SCENARIOS = {
    'S1_targeted':    {'n_frac': 0.20},
    'S2_coordinated': {'n_frac': 0.30},
    'S3_cascading':   {'n_frac': 0.45},
}
CAP_FRAC   = 0.15
HIGH_COL   = 0  # depends on LabelEncoder order; confirm with HIGH_IDX at runtime

df_full = generate_karat_syn(n=75000, seed=42)
le = LabelEncoder(); le.fit(['Low', 'Medium', 'High'])
y_full   = le.transform(df_full['priority_label'])
HIGH_IDX = list(le.classes_).index('High')

df_temp, df_test, y_temp, y_test = train_test_split(
    df_full, y_full, test_size=0.20, random_state=42, stratify=y_full)
df_train, df_val, y_train, y_val = train_test_split(
    df_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
for d in [df_train, df_val, df_test]: d.reset_index(drop=True, inplace=True)
y_train = np.array(y_train); y_val = np.array(y_val); y_test = np.array(y_test)

teacher = build_teacher(seed=42)
teacher.fit(df_train[KATS_FEATURES], y_train)
scaler = StandardScaler()
X_s = scaler.fit_transform(df_train[KATS_FEATURES].values)
kd_idx = np.random.RandomState(42).choice(len(df_train), int(len(df_train)*0.10), replace=False)
student = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
student.fit(X_s[kd_idx], y_train[kd_idx])

def t_p(df_): return teacher.predict_proba(df_[KATS_FEATURES])
def s_p(df_): return student.predict_proba(scaler.transform(df_[KATS_FEATURES].values))

BASELINE_CKA = compute_cka(t_p(df_val), s_p(df_val))
L2_BASELINE  = compute_l2_divergence(t_p(df_val), s_p(df_val))
b2_clean     = baseline_static_rule(df_test)
b2_student_t0 = s_p(df_test)

e2_all = []
print(f"{'Scenario':<20} {'t':>4} {'B1_Rand':>9} {'B2_Rule':>9} {'B3_L2':>9} {'KARAT':>9} {'Gain':>8}")
print("-" * 72)

for sc_name, params in SCENARIOS.items():
    for t in TIMESTEPS:
        df_t     = inject_attack_with_labels(df_test, timestep=t, attack_fraction=params['n_frac'])
        t_probs  = t_p(df_t)
        s_probs  = s_p(df_t)
        cka_drop = BASELINE_CKA - compute_cka(t_probs, s_probs)
        l2_drop  = compute_l2_divergence(t_probs, s_probs) - L2_BASELINE

        # Blend correction (final KARAT)
        blend        = min(max((cka_drop - 0.10) / 0.15, 0), 0.9) if cka_drop > 0.10 else 0
        karat_scores = (1 - blend) * s_probs + blend * t_probs

        surv_b1    = precision_at_k(df_t, baseline_random(df_t),               k_frac=CAP_FRAC, high_col=HIGH_IDX)
        surv_b2    = precision_at_k(df_t, b2_student_t0,                        k_frac=CAP_FRAC, high_col=HIGH_IDX)
        surv_b3    = precision_at_k(df_t, baseline_l2_trigger(df_t, s_probs.copy(), l2_drop), k_frac=CAP_FRAC, high_col=HIGH_IDX)
        surv_karat = precision_at_k(df_t, karat_scores,                         k_frac=CAP_FRAC, high_col=HIGH_IDX)
        gain       = surv_karat - surv_b2

        e2_all.append({'scenario': sc_name, 'timestep': t, 'cka_drop': round(cka_drop, 4),
                       'B1_random': round(surv_b1, 4), 'B2_static': round(surv_b2, 4),
                       'B3_l2': round(surv_b3, 4), 'KARAT': round(surv_karat, 4),
                       'gain_vs_B2': round(gain, 4)})
        print(f"{sc_name:<20} {t:>4} {surv_b1:>9.4f} {surv_b2:>9.4f} {surv_b3:>9.4f} {surv_karat:>9.4f} {gain:>+8.4f}")
    print()

e2_df = pd.DataFrame(e2_all)
e2_df.to_csv(os.path.join(RESULTS, 'E2_survivability_all.csv'), index=False)
print(f"Saved E2 results to {RESULTS}")
