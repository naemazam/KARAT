#!/usr/bin/env bash
# ================================================================
# KARAT — Run all experiments in sequence
# ================================================================
# Usage: bash scripts/run_all_experiments.sh
# Requires: pip install -r requirements.txt
# Output:  results/ directory with all CSVs
# ================================================================

set -e
echo "================================================================"
echo " KARAT Experiment Runner"
echo "================================================================"

echo ""
echo "[1/3] E2 — Multi-seed main experiment (Table 2)..."
python -m experiments.E2_multiseed_main

echo ""
echo "[2/3] E6 — CKA vs L2 ablation..."
python -c "
from experiments.E2_multiseed_main import *
from experiments.E6_cka_vs_l2_ablation import run_e6
run_e6(teacher_proba, student_proba, df_test, THETA, L2_THETA, BASE_CKA, BASE_L2, HIGH_IDX)
"

echo ""
echo "[3/3] E5 — CIC cross-domain (requires CIC data)..."
if [ -f data/CIC-IDS2018-preprocessed.csv ]; then
    python -c "
from experiments.E2_multiseed_main import *
from experiments.E5_cic_validation import run_e5
from src.dataset import load_cic_ids2018
cic = load_cic_ids2018('data/CIC-IDS2018-preprocessed.csv')
run_e5(teacher, student_proba, THETA, HIGH_IDX, cic)
"
else
    echo "  Skipping E5: data/CIC-IDS2018-preprocessed.csv not found."
    echo "  Download from https://www.unb.ca/cic/datasets/ids-2018.html"
fi

echo ""
echo "================================================================"
echo " All done. Results saved to results/"
echo "================================================================"
