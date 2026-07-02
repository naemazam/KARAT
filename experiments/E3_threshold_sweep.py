"""
E3 — Detection Threshold (θ) Sensitivity Sweep

Evaluates KARAT re-triage precision, recall, and survivability gain
across θ ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50} at t=6.

Key finding: θ=0.10 balances precision (0.935) and recall (0.237) with
maximum survivability gain (+0.035).
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings; warnings.filterwarnings('ignore')

from src.dataset import generate_karat_syn, KATS_FEATURES
from src.cka import compute_cka
from src.attack import inject_attack_with_labels
from src.models import build_teacher
from src.metrics import precision_at_k

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
os.makedirs(RESULTS, exist_ok=True)

THETA_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
CAP_FRAC     = 0.15

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

df_t6  = inject_attack_with_labels(df_test, timestep=6, attack_fraction=0.30)
t_p6   = t_p(df_t6)
s_p6   = s_p(df_t6)
drop6  = BASELINE_CKA - compute_cka(t_p6, s_p6)

az_thresh  = df_t6['az_risk_score'].values > df_t6['az_risk_score'].quantile(0.70)
downstream = df_t6['downstream_critical'].values == 1
high_sc    = df_t6['service_criticality'].values > df_t6['service_criticality'].quantile(0.60)
base_flagged = np.where(az_thresh & downstream & high_sc)[0]
true_high_pos = set(np.where(df_t6['priority_label'].values == 'High')[0])

print(f"CKA drop at t=6: {drop6:.4f}")
print(f"\n{'theta':>6} {'flagged':>8} {'prec':>8} {'recall':>8} {'PAK':>8} {'gain':>8}")
print("-" * 52)

e3_results = []
for theta in THETA_VALUES:
    flagged = base_flagged if drop6 > theta else np.array([], dtype=int)
    dyn_probs = s_p6.copy()
    if len(flagged) > 0:
        alpha = max(1.1, 1.4 - (drop6 - theta) * 1.5)
        for pos in flagged:
            dyn_probs[pos, HIGH_IDX] = min(dyn_probs[pos, HIGH_IDX] * alpha, 1.0)

    pak_dyn  = precision_at_k(df_t6, dyn_probs, k_frac=CAP_FRAC, high_col=HIGH_IDX)
    pak_base = precision_at_k(df_t6, s_p6,      k_frac=CAP_FRAC, high_col=HIGH_IDX)
    gain     = pak_dyn - pak_base

    correct   = len(set(flagged) & true_high_pos) if len(flagged) > 0 else 0
    precision = correct / len(flagged) if len(flagged) > 0 else 1.0
    recall    = correct / len(true_high_pos) if len(flagged) > 0 else 0.0

    e3_results.append({'theta': theta, 'flagged': len(flagged),
                       'precision': round(precision, 4), 'recall': round(recall, 4),
                       'pak': round(pak_dyn, 4), 'gain': round(gain, 4)})
    print(f"{theta:>6.2f} {len(flagged):>8} {precision:>8.4f} {recall:>8.4f} {pak_dyn:>8.4f} {gain:>+8.4f}")

e3_df = pd.DataFrame(e3_results)
e3_df.to_csv(os.path.join(RESULTS, 'E3_threshold_sweep.csv'), index=False)
print(f"\nSaved E3 results to {RESULTS}")
