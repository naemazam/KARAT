# KARAT: Knowledge-Aware Resilience via Adaptive Teacher-Student Distillation

> **Maintaining cloud service priority ranking precision under adversarial label-drift using CKA-triggered teacher-student correction.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research-orange)]()

---

## Overview

KARAT is a resilience framework for **knowledge-distilled cloud service triage models** deployed under adversarial drift. A lightweight student model — a **Logistic Regression pipeline** (StandardScaler + LogisticRegression, max\_iter=1000) representing a compressed edge-deployed classifier with a fixed training snapshot — is continuously monitored via **Centred Kernel Alignment (CKA)** against a frozen Gradient Boosting teacher ensemble. When representation-space fidelity drops below a learned threshold θ, KARAT blends student and teacher scores to recover ranking precision.

This design deliberately maximises the teacher–student capacity gap, simulating realistic edge-deployment constraints where model compression is aggressive. KARAT closes the loop between fidelity degradation detection and adaptive migration re-prioritisation under active cyberattack — without retraining.

### Key Results (Precision@5K, 5 attack seeds — KARAT-SYN benchmark)

| Scenario | B2 Student | B3 L2 | **KARAT** | Oracle |
|---|---|---|---|---|
| S1 Targeted | 0.858 ± 0.090 | 0.921 ± 0.099 | **0.909 ± 0.038** | 0.977 ± 0.033 |
| S2 Coordinated | 0.909 ± 0.030 | 0.909 ± 0.030 | **0.914 ± 0.024** | 0.984 ± 0.019 |
| S3 Cascading | 0.912 ± 0.022 | 0.912 ± 0.022 | **0.915 ± 0.020** | 0.990 ± 0.007 |

### Cross-Domain Validation Summary (E5 + E5b)

| Dataset | Triggered / Total | B2 at t=10 | KARAT at t=10 | Mean Gain (triggered) |
|---|---|---|---|---|
| CIC-IDS2018 | 4/6 | 0.6550 | 0.7530 | +0.0695 |
| NSL-KDD | 4/6 | 0.6222 | 0.9971 | +0.3814 |

> KARAT is evaluated on **three datasets**: a calibrated synthetic cloud benchmark (KARAT-SYN, 75,000 services), CIC-IDS2018, and NSL-KDD. CKA threshold θ fitted on KARAT-SYN validation transfers without recalibration to both real-world datasets.

---

## Repository Structure

```
KARAT/
├── src/
│   ├── dataset.py           # Synthetic KARAT-SYN dataset generator + CIC loader
│   ├── model.py             # Teacher ensemble builder
│   ├── attack.py            # Adversarial injection (S1/S2/S3 scenarios)
│   ├── metrics.py           # Precision@K, CKA, L2 divergence, Wilcoxon
│   └── karat.py             # Core KARAT correction logic (KARATCorrector)
├── experiments/
│   ├── E2_multiseed.py      # Main 5-seed × 3-scenario experiment
│   ├── E5_cic_validation.py # CIC-IDS2018 cross-domain validation
│   └── E6_cka_vs_l2.py     # CKA vs L2 trigger ablation
├── results/
│   ├── e1_cka_monitoring.csv
│   ├── E1_fidelity_all_scenarios.csv / .pdf / .png
│   ├── E1_fidelity_collapse.csv / .png
│   ├── E1_fidelity_collapse_all_scenarios.csv
│   ├── E2_multiseed_all.csv
│   ├── E2_precision_all_methods.pdf / .png
│   ├── E2_survivability.csv / .png
│   ├── E2_survivability_all.csv
│   ├── e2_precision_at_k.csv
│   ├── E3_threshold_sweep.csv
│   ├── e3_rank_correlation.csv
│   ├── theta_sweep_val.csv
│   ├── E4_latency.csv
│   ├── e4_alert_recovery.csv
│   ├── E5_CIC_IDS2018_validation.csv
│   ├── E5b_NSL_KDD_validation.csv
│   ├── E5b_NSL_KDD_all_scenarios.csv
│   ├── E5_crossdomain_summary.csv
│   ├── E6_cka_vs_l2_ablation.csv
│   ├── wilcoxon_significance.csv
│   ├── KARAT_results_summary.csv
│   └── figures/
├── notebooks/
│   └── KARAT_full_pipeline.ipynb
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Datasets

### KARAT-SYN (Primary Benchmark)
- Calibrated synthetic cloud workload: **75,000 services**, 3 priority classes (High / Medium / Low)
- 8 features mapping cloud operational and service-criticality attributes
- Train / Val / Test split: 60% / 20% / 20% (45,000 / 15,000 / 15,000)
- CKA threshold θ selected on **validation split only**; all final results on **test split**

### CIC-IDS2018 (External Validation — E5)
- Real-world network intrusion dataset mapped to KARAT feature space
- Validates cross-domain generalisation without retraining or threshold recalibration
- [Download from Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2018.html)

### NSL-KDD (External Validation — E5b)
- Classic network security benchmark (KDDTest+, 22,544 records)
- NSL-KDD features mapped to KARAT 8-feature space (mapping documented in `notebooks/KARAT_full_pipeline.ipynb`)
- Priority label: `normal` → High (rescue targets), DoS → Low, Probe/R2L/U2R → Medium
- [Download from GitHub: defcom17/NSL_KDD](https://github.com/defcom17/NSL_KDD)

---

## Experimental Setup

| Parameter | Value |
|---|---|
| Synthetic dataset | KARAT-SYN, 75,000 services |
| Test set size | 15,000 |
| High-label services | 4,500 (30%) |
| K (primary, 5%) | 750 (16.7% of High) |
| K (secondary, 10%) | 1,500 (33.3% of High) |
| Attack seeds | 42, 7, 13, 99, 2024 |
| Timesteps | 0, 2, 4, 6, 8, 10 |
| Teacher | Gradient Boosting ensemble (100 estimators, max\_depth=4) |
| **Student** | **Logistic Regression pipeline (StandardScaler + LR, max\_iter=1000, 8% training data)** |
| CKA θ | 0.00279 (val p20, trigger rate 0.80) |
| L2 θ | 0.000988 (positive val drops only) |

### Attack Scenarios

- **S1 Targeted:** 75% of High-label services corrupted, noise scale 0.40
- **S2 Coordinated:** 70% of all services, noise scale 0.55
- **S3 Cascading:** High-dependency seed nodes → cascade to downstream\_critical services

---

## Sanity Check: PAK Degradation under S1 (seed=42)

```
t  | B2_frozen  live_s   KARAT   oracle   cka_d
------------------------------------------------
 0 |   0.9387   0.9387  0.9387  0.9973  0.0000
 2 |   0.9387   0.9293  0.9333  0.9960  0.0075
 4 |   0.9387   0.9133  0.9333  0.9947  0.0190
 6 |   0.9387   0.8680  0.9227  0.9920  0.0392
 8 |   0.9387   0.8120  0.8987  0.9747  0.0672
10 |   0.9387   0.6907  0.8360  0.9133  0.0909
```

---

## E5: CIC-IDS2018 Cross-Domain Validation

```
t  | B2_stud   KARAT   Oracle  Gain_B2   cka_d  trig
----------------------------------------------------
 0 |  0.8010  0.8010  0.9430  +0.0000  0.0000     ·
 2 |  0.7920  0.8000  0.9420  +0.0080  0.0046     ▲
 4 |  0.7900  0.8250  0.9320  +0.0350  0.0178     ▲
 6 |  0.7670  0.8240  0.9160  +0.0570  0.0361     ▲
 8 |  0.7250  0.8130  0.8670  +0.0880  0.0510     ▲
10 |  0.6550  0.7530  0.7300  +0.0980  0.0826     ▲
```

> **Notable:** At t=10, KARAT (0.753) exceeds the oracle (0.730), indicating CKA detects representational drift before teacher accuracy fully degrades.

---

## E5b: NSL-KDD Cross-Domain Validation (averaged across S1/S2/S3)

```
t  | B2_stud   KARAT   Oracle  Gain_B2   cka_d  trig
----------------------------------------------------
 0 |  0.9067  0.9067  1.0000  +0.0000  0.0000     ·
 2 |  0.7704  0.8948  1.0000  +0.1244  0.0826     ·
 4 |  0.5630  0.9689  1.0000  +0.4059  0.1428     ▲
 6 |  0.5852  0.9985  1.0000  +0.4133  0.2145     ▲
 8 |  0.6667  0.9985  1.0000  +0.3318  0.2653     ▲
10 |  0.6222  0.9971  1.0000  +0.3748  0.2864     ▲
```

> **Notable:** KARAT's correction benefit scales with the severity of student degradation. On NSL-KDD, the larger teacher–student capacity gap under attack produces stronger gains (+0.38 mean triggered) compared to CIC-IDS2018 (+0.07), confirming KARAT's value increases with attack intensity and model compression.

---

## E6: CKA vs L2 Trigger Ablation

Under **S1 targeted** attack, L2 divergence provides competitive signal at moderate drift (t=2–8) because teacher and student co-move positively on corrupted features. At extreme drift (t=10), CKA outperforms L2 by **+0.145 PAK**. Under **S2/S3**, L2 divergence turns negative (co-movement), disabling the L2 trigger entirely — CKA continues to trigger and correct.

---

## Threshold Selection and Validation vs Test Reporting

- **θ = 0.00279** was selected on the **validation split** (15,000 services) using the precision–recall tradeoff from E3.
- Validation-split precision at θ = 0.00279: **93.47%** (threshold selection metric only).
- All E1, E2, E4, E5, E5b, E6 results are reported on the **held-out test split**.
- θ is applied without modification to test split and both external datasets.

---

## Limitations

1. KARAT assumes access to a pre-trained teacher model at correction time.
2. The current correction cycle latency (~779 ms) may be insufficient for sub-second attack dynamics.
3. The CKA threshold assumes a stable clean baseline and may require recalibration under long-term concept drift.

---

## Installation

```bash
git clone https://github.com/naemazam/KARAT.git
cd KARAT
pip install -r requirements.txt
```

## Quickstart

```python
from src.dataset import generate_kats_syn
from src.karat import KARATCorrector, KARATConfig
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df = generate_kats_syn(n=75000, seed=42)
# ... fit teacher (GradientBoostingClassifier), then:
scaler  = StandardScaler()
student = LogisticRegression(max_iter=1000)
# fit scaler and student on 8% of training data, then:
corrector = KARATCorrector(teacher, student, scaler, features=FEATS,
                           config=KARATConfig(theta=0.00279))
corrected_scores = corrector.predict_proba(df_test)
```

## Run Experiments

```bash
python experiments/E2_multiseed.py
python experiments/E5_cic_validation.py
python experiments/E6_cka_vs_l2.py
```

---

## Citation

```bibtex
@misc{naemazam2026karat,
  title  = {KARAT: Knowledge-Aware Resilience via Adaptive Teacher-Student
             Distillation for Cloud Service Priority Ranking},
  author = {Chowdhury, Naem Azam},
  year   = {2026},
  url    = {https://github.com/naemazam/KARAT}
}
```

## License

MIT — see [LICENSE](LICENSE).
