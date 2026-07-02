"""
E1 — KD Fidelity Collapse Under Cyberattack

Measures CKA drop across 3 attack scenarios and 6 timesteps.
Outputs: results/E1_fidelity_collapse_all_scenarios.csv + figure.

Key finding: CKA drops monotonically from ~0.984 (t=0) to ~0.787 (t=10)
for S2_coordinated (30% services attacked).
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import warnings; warnings.filterwarnings('ignore')

from src.dataset import generate_karat_syn, KATS_FEATURES
from src.cka import compute_cka
from src.attack import inject_attack_with_labels
from src.models import build_teacher

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
os.makedirs(RESULTS, exist_ok=True)

TIMESTEPS = [0, 2, 4, 6, 8, 10]
SCENARIOS = {
    'S1_targeted':    {'n_frac': 0.20},
    'S2_coordinated': {'n_frac': 0.30},
    'S3_cascading':   {'n_frac': 0.45},
}

# ── Data + models ──────────────────────────────────────────────────────────────
df_full = generate_karat_syn(n=75000, seed=42)
le = LabelEncoder(); le.fit(['Low', 'Medium', 'High'])
y_full = le.transform(df_full['priority_label'])

df_temp, df_test, y_temp, y_test = train_test_split(
    df_full, y_full, test_size=0.20, random_state=42, stratify=y_full)
df_train, df_val, y_train, y_val = train_test_split(
    df_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
for d in [df_train, df_val, df_test]: d.reset_index(drop=True, inplace=True)
y_train = np.array(y_train); y_val = np.array(y_val); y_test = np.array(y_test)

print("Training teacher...")
teacher = build_teacher(seed=42)
teacher.fit(df_train[KATS_FEATURES], y_train)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(df_train[KATS_FEATURES].values)
kd_idx = np.random.RandomState(42).choice(len(df_train), int(len(df_train)*0.10), replace=False)
student = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
student.fit(X_train_s[kd_idx], y_train[kd_idx])

def t_proba(df_): return teacher.predict_proba(df_[KATS_FEATURES])
def s_proba(df_): return student.predict_proba(scaler.transform(df_[KATS_FEATURES].values))

BASELINE_CKA = compute_cka(t_proba(df_val), s_proba(df_val))
print(f"Baseline CKA: {BASELINE_CKA:.4f}")

# ── Experiment ─────────────────────────────────────────────────────────────────
e1_all = []
print(f"\n{'Scenario':<20} {'t':>4} {'CKA':>8} {'Drop':>8}")
print("-" * 45)

for sc_name, params in SCENARIOS.items():
    for t in TIMESTEPS:
        df_t  = inject_attack_with_labels(df_test, timestep=t, attack_fraction=params['n_frac'])
        cka_t = compute_cka(t_proba(df_t), s_proba(df_t))
        drop  = BASELINE_CKA - cka_t
        e1_all.append({'scenario': sc_name, 'timestep': t,
                       'cka_score': round(cka_t, 4), 'cka_drop': round(drop, 4)})
        print(f"{sc_name:<20} {t:>4} {cka_t:>8.4f} {drop:>8.4f}")
    print()

e1_df = pd.DataFrame(e1_all)
e1_df.to_csv(os.path.join(RESULTS, 'E1_fidelity_collapse_all_scenarios.csv'), index=False)

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
colors = {'S1_targeted': '#7eccd6', 'S2_coordinated': '#4f98a3', 'S3_cascading': '#2a6b76'}
for sc_name in SCENARIOS:
    sub = e1_df[e1_df['scenario'] == sc_name]
    ax.plot(sub['timestep'], sub['cka_score'], marker='o', linewidth=2.5,
            color=colors[sc_name], label=sc_name)
ax.axhline(y=BASELINE_CKA, color='#4fa84a', linewidth=1.5, linestyle='--',
           label=f'Clean Baseline ({BASELINE_CKA:.4f})')
ax.axhline(y=BASELINE_CKA - 0.10, color='#cc4444', linewidth=1.5, linestyle=':',
           label='Detection Threshold θ=0.10')
ax.set_xlabel('Attack Timestep (minutes)', fontsize=12)
ax.set_ylabel('CKA Fidelity Score', fontsize=12)
ax.set_title('E1: KD Fidelity Collapse Under Cyberattack (KARAT)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, 'E1_fidelity_collapse.png'), dpi=150, bbox_inches='tight')
print(f"Saved E1 results to {RESULTS}")
