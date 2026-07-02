# KARAT Experiment Results

This directory contains all CSV outputs from the KARAT experimental suite.
All results are reproducible by running the scripts in `experiments/`.

## Files

| File | Experiment | Description |
|---|---|---|
| `E2_multiseed_all.csv` | E2 | 90 rows: 5 attack seeds × 3 scenarios × 6 timesteps. Precision@5K for B1/B2/B3/KARAT/Oracle |
| `wilcoxon_significance.csv` | E2 | Wilcoxon signed-rank test results for KARAT vs each baseline |
| `E5_CIC_IDS2018_validation.csv` | E5 | Cross-domain validation on CIC-IDS2018, 6 timesteps |
| `E6_cka_vs_l2_ablation.csv` | E6 | CKA-trigger vs L2-trigger ablation, seed=42, 3 scenarios × 6 timesteps |

## Key Numbers (Paper)

### E2 — Table 2 (Mean Precision@5K ± std, 5 attack seeds)

| Scenario | B1_Random | B2_Student | B3_L2 | KARAT | Oracle |
|---|---|---|---|---|---|
| S1_targeted | 0.2899±0.0211 | 0.8583±0.0895 | 0.9205±0.0990 | **0.9087±0.0384** | 0.9768±0.0325 |
| S2_coordinated | 0.2899±0.0211 | 0.9090±0.0295 | 0.9090±0.0295 | **0.9136±0.0240** | 0.9835±0.0194 |
| S3_cascading | 0.2899±0.0211 | 0.9116±0.0215 | 0.9116±0.0215 | **0.9153±0.0199** | 0.9904±0.0072 |

KARAT vs B2 (live student): all scenarios p < 0.001 (Wilcoxon ***)

### E5 — CIC-IDS2018 Cross-Domain

| t | B2_Student | KARAT | Oracle | Gain |
|---|---|---|---|---|
| 0 | 0.8010 | 0.8010 | 0.9430 | +0.0000 |
| 4 | 0.7900 | 0.8250 | 0.9320 | +0.0350 |
| 8 | 0.7250 | 0.8130 | 0.8670 | +0.0880 |
| 10 | 0.6550 | **0.7530** | 0.7300 | **+0.0980** |

Triggered: 5/6 timesteps. Mean gain: +0.0477. KARAT exceeds oracle at t=10.

### E6 — CKA vs L2 (S1_targeted highlights)

| t | KARAT | B3_L2 | Δ |
|---|---|---|---|
| 6 | 0.9227 | 0.9853 | −0.063 (L2 wins: co-movement) |
| 10 | **0.8360** | 0.6907 | **+0.145** (CKA wins: L2 collapses) |

Note: L2 divergence goes negative under S2/S3 (teacher and student co-move),
rendering L2 unable to detect fidelity collapse. CKA remains effective.
