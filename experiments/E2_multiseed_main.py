"""
E2 — Multi-seed Main Experiment
================================
Fixed teacher/student pair (seed=42 trained model).
5 attack seeds × 3 scenarios × 6 timesteps = 90 evaluation rows.

Key design decisions:
  B1  = random baseline (lower bound)
  B2  = live student with frozen weights evaluated on attacked data
  B3  = L2-divergence-triggered teacher blend
  KARAT = CKA-triggered teacher blend (proposed method)
  Oracle = teacher scores (upper bound)

L2_THETA is set from the median of POSITIVE val L2 drops only,
because negative L2 drops (teacher/student co-movement) carry no
divergence signal and would cause over-triggering if threshold=0.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

from src.dataset import generate_kats_syn, KATS_FEATURES
from src.model import build_kats_ensemble
from src.attack import inject_adversarial
from src.metrics import precision_at_k, compute_cka, compute_l2_div
from src.karat import score_karat, score_l2_trigger

# ── Config ────────────────────────────────────────────────────────
SEED_MODEL  = 42
SEEDS       = [42, 7, 13, 99, 2024]   # attack seeds
K_FRAC      = 0.05
SCENARIOS   = ['S1_targeted', 'S2_coordinated', 'S3_cascading']
TIMESTEPS   = [0, 2, 4, 6, 8, 10]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Build dataset and models (seed=42 fixed) ─────────────────────
df = generate_kats_syn(n=75000, seed=SEED_MODEL)
le = LabelEncoder(); le.fit(['Low', 'Medium', 'High'])
y  = le.transform(df['priority_label'])
HIGH_IDX = list(le.classes_).index('High')

idx = np.arange(len(df))
tr_i, te_i = train_test_split(idx, test_size=0.20,
                               random_state=SEED_MODEL, stratify=y)
tr_i2, va_i = train_test_split(tr_i, test_size=0.25,
                                random_state=SEED_MODEL, stratify=y[tr_i])

df_train = df.iloc[tr_i2].reset_index(drop=True)
df_val   = df.iloc[va_i].reset_index(drop=True)
df_test  = df.iloc[te_i].reset_index(drop=True)
y_train  = y[tr_i2]

print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
print(f"High@K={K_FRAC}: {int(len(df_test)*K_FRAC)} slots, "
      f"{(df_test['priority_label']=='High').sum()} High services")

# Teacher
teacher = build_kats_ensemble(alpha=5, seed=SEED_MODEL)
teacher.fit(df_train[KATS_FEATURES], y_train)

# Student: 8% of training data, shallow MLP
scaler_s = StandardScaler()
X_tr = scaler_s.fit_transform(df_train[KATS_FEATURES].values)
rng_kd = np.random.RandomState(SEED_MODEL)
kd_idx = rng_kd.choice(len(df_train),
                        int(len(df_train) * 0.08), replace=False)
student = MLPClassifier(hidden_layer_sizes=(16,), activation='relu',
                         max_iter=60, random_state=SEED_MODEL,
                         learning_rate_init=0.008, alpha=0.05)
student.fit(X_tr[kd_idx], y_train[kd_idx])
print("Models trained.")

def teacher_proba(df_q):
    return teacher.predict_proba(df_q[KATS_FEATURES])

def student_proba(df_q):
    return student.predict_proba(
        scaler_s.transform(df_q[KATS_FEATURES].values))

def pak(df_q, scores):
    k = int(len(df_q) * K_FRAC)
    r = np.argsort(scores[:, HIGH_IDX])[::-1][:k]
    return (df_q['priority_label'].values[r] == 'High').mean()

def baseline_random(n, seed):
    return np.random.RandomState(seed).dirichlet(np.ones(3), n)

# ── Compute thresholds from val split ────────────────────────────
val_base_cka = compute_cka(teacher_proba(df_val), student_proba(df_val))
val_base_l2  = compute_l2_div(teacher_proba(df_val), student_proba(df_val))

val_cka_drops, val_l2_drops = [], []
for sc in SCENARIOS:
    for t in TIMESTEPS[1:]:
        df_tv = inject_adversarial(df_val, t, sc, SEED_MODEL)
        val_cka_drops.append(
            val_base_cka - compute_cka(teacher_proba(df_tv),
                                        student_proba(df_tv)))
        val_l2_drops.append(
            compute_l2_div(teacher_proba(df_tv), student_proba(df_tv))
            - val_base_l2)

THETA     = max(float(np.percentile(val_cka_drops, 20)), 1e-5)
trig_rate = np.mean([d > THETA for d in val_cka_drops])
pos_l2    = [d for d in val_l2_drops if d > 0]
if len(pos_l2) >= 3:
    L2_THETA = float(np.percentile(pos_l2, max(0, (1-trig_rate)*100)))
else:
    L2_THETA = float(np.percentile(val_l2_drops, 50))
L2_THETA = max(L2_THETA, 1e-5)

print(f"THETA={THETA:.5f}  L2_THETA={L2_THETA:.6f}  "
      f"trigger_rate={trig_rate:.2f}")

# ── Clean state ───────────────────────────────────────────────────
BASE_CKA = compute_cka(teacher_proba(df_test), student_proba(df_test))
BASE_L2  = compute_l2_div(teacher_proba(df_test), student_proba(df_test))
print(f"BASE_CKA={BASE_CKA:.4f}  BASE_L2={BASE_L2:.6f}")

# ── Main loop ─────────────────────────────────────────────────────
all_runs = []
for seed in SEEDS:
    for scenario in SCENARIOS:
        for t in TIMESTEPS:
            df_t  = inject_adversarial(df_test, t, scenario, seed)
            tp    = teacher_proba(df_t)
            sp    = student_proba(df_t)
            cka_d = BASE_CKA - compute_cka(tp, sp)
            l2_d  = compute_l2_div(tp, sp) - BASE_L2

            all_runs.append({
                'seed': seed, 'scenario': scenario, 'timestep': t,
                'cka_drop': round(cka_d, 4),
                'l2_drop':  round(l2_d, 6),
                'PAK_B1':     round(pak(df_t, baseline_random(len(df_t), seed)), 4),
                'PAK_B2':     round(pak(df_t, sp), 4),
                'PAK_B3_L2':  round(pak(df_t, score_l2_trigger(sp, tp, l2_d, L2_THETA)), 4),
                'PAK_KARAT':  round(pak(df_t, score_karat(sp, tp, cka_d, THETA)), 4),
                'PAK_B4_ORC': round(pak(df_t, tp), 4),
            })

runs_df = pd.DataFrame(all_runs)
out_path = os.path.join(RESULTS_DIR, 'E2_multiseed_all.csv')
runs_df.to_csv(out_path, index=False)
print(f"Saved {len(runs_df)} rows → {out_path}")

# ── Wilcoxon ─────────────────────────────────────────────────────
sig_rows = []
for scenario in SCENARIOS:
    sub = runs_df[runs_df['scenario'] == scenario]
    for bc, label in [('PAK_B1','B1_Random'), ('PAK_B2','B2_LiveStudent'),
                       ('PAK_B3_L2','B3_L2'),  ('PAK_B4_ORC','B4_Oracle')]:
        kv = sub['PAK_KARAT'].values; bv = sub[bc].values
        if np.all(kv == bv): stat, p = np.nan, np.nan
        else:
            try: _, p = wilcoxon(kv, bv, alternative='two-sided')
            except: p = np.nan
        sig = ('***' if (not np.isnan(p) and p < 0.001) else
               ('**'  if (not np.isnan(p) and p < 0.01)  else
               ('*'   if (not np.isnan(p) and p < 0.05)  else 'ns')))
        sig_rows.append({'scenario': scenario, 'baseline': label,
                         'karat_mean': round(kv.mean(), 4),
                         'base_mean':  round(bv.mean(), 4),
                         'delta':      round((kv-bv).mean(), 4),
                         'p_value':    round(p, 6) if not np.isnan(p) else np.nan,
                         'sig': sig})

sig_df = pd.DataFrame(sig_rows)
sig_df.to_csv(os.path.join(RESULTS_DIR, 'wilcoxon_significance.csv'), index=False)
print("Wilcoxon results saved.")

# ── Paper Table 2 ─────────────────────────────────────────────────
print("\n=== PAPER TABLE 2 ===")
print(f"{'Scenario':<22} {'B1':>13} {'B2':>13} {'B3_L2':>13} "
      f"{'KARAT':>13} {'Oracle':>13}")
for scenario in SCENARIOS:
    sub = runs_df[runs_df['scenario'] == scenario]
    def ms(c): return f"{sub[c].mean():.4f}\u00b1{sub[c].std():.4f}"
    print(f"{scenario:<22} {ms('PAK_B1'):>13} {ms('PAK_B2'):>13} "
          f"{ms('PAK_B3_L2'):>13} {ms('PAK_KARAT'):>13} "
          f"{ms('PAK_B4_ORC'):>13}")
