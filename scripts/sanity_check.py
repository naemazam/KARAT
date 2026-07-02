"""
Sanity check: verify PAK degrades correctly across attack timesteps.
Run this before the main experiments to confirm models and data are
working as expected.

Expected output (seed=42, S1_targeted):
  t=0:  student PAK ~0.939, oracle ~0.997
  t=10: student PAK ~0.691, oracle ~0.913  (clear degradation)
  KARAT t=10: ~0.836  (partial recovery toward oracle)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
import warnings; warnings.filterwarnings('ignore')

from src.dataset import generate_kats_syn, KATS_FEATURES
from src.model import build_kats_ensemble
from src.attack import inject_adversarial
from src.metrics import compute_cka, compute_l2_div
from src.karat import score_karat

SEED = 42
K_FRAC = 0.05
TIMESTEPS = [0, 2, 4, 6, 8, 10]

df = generate_kats_syn(n=75000, seed=SEED)
le = LabelEncoder(); le.fit(['Low', 'Medium', 'High'])
y  = le.transform(df['priority_label'])
HIGH_IDX = list(le.classes_).index('High')

idx = np.arange(len(df))
tr_i, te_i = train_test_split(idx, test_size=0.20, random_state=SEED, stratify=y)
tr_i2, _ = train_test_split(tr_i, test_size=0.25, random_state=SEED, stratify=y[tr_i])
df_train = df.iloc[tr_i2].reset_index(drop=True)
df_test  = df.iloc[te_i].reset_index(drop=True)
y_train  = y[tr_i2]

teacher = build_kats_ensemble(alpha=5, seed=SEED)
teacher.fit(df_train[KATS_FEATURES], y_train)

scaler_s = StandardScaler()
X_tr = scaler_s.fit_transform(df_train[KATS_FEATURES].values)
rng  = np.random.RandomState(SEED)
kd_i = rng.choice(len(df_train), int(len(df_train)*0.08), replace=False)
student = MLPClassifier(hidden_layer_sizes=(16,), max_iter=60,
                         random_state=SEED, learning_rate_init=0.008, alpha=0.05)
student.fit(X_tr[kd_i], y_train[kd_i])

def teacher_proba(df_q): return teacher.predict_proba(df_q[KATS_FEATURES])
def student_proba(df_q): return student.predict_proba(scaler_s.transform(df_q[KATS_FEATURES].values))
def pak(df_q, sc):
    k = int(len(df_q)*K_FRAC)
    r = np.argsort(sc[:,HIGH_IDX])[::-1][:k]
    return (df_q['priority_label'].values[r]=='High').mean()

BASE_CKA = compute_cka(teacher_proba(df_test), student_proba(df_test))
THETA    = 0.00279  # from val calibration

print(f"Sanity check — seed={SEED}, S1_targeted")
print(f"{'t':>4} | {'student':>8} {'KARAT':>8} {'oracle':>8} {'cka_d':>8}")
print("-" * 44)
for t in TIMESTEPS:
    df_t  = inject_adversarial(df_test, t, 'S1_targeted', SEED)
    tp    = teacher_proba(df_t); sp = student_proba(df_t)
    cka_d = BASE_CKA - compute_cka(tp, sp)
    print(f"{t:>4} | {pak(df_t,sp):>8.4f} "
          f"{pak(df_t, score_karat(sp,tp,cka_d,THETA)):>8.4f} "
          f"{pak(df_t,tp):>8.4f} {cka_d:>8.4f}")

print("\nSanity check passed if student PAK drops from ~0.94 to ~0.69 by t=10.")
