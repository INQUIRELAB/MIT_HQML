#!/usr/bin/env bash
# ==============================================================================
# One-Click Reproducibility Pipeline for QML MIT Classification
# ==============================================================================
set -e

# Resolve script directory and set ROOT
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "======================================================================"
echo "🚀 QML MIT Classification: Reproducibility Execution Pipeline"
echo "======================================================================"
echo "Working directory : ${SCRIPT_DIR}"
echo "Python binary     : ${PYTHON_BIN}"
echo "Start time        : $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
echo "======================================================================"

# ------------------------------------------------------------------------------
# Environment check
# ------------------------------------------------------------------------------
echo "🔍 Checking Python & dependency versions..."
"${PYTHON_BIN}" -c "
import torch, pennylane, sklearn, pandas, numpy
print(f'  - PyTorch: {torch.__version__}')
print(f'  - PennyLane: {pennylane.__version__}')
print(f'  - Scikit-Learn: {sklearn.__version__}')
print(f'  - Pandas: {pandas.__version__}')
print(f'  - NumPy: {numpy.__version__}')
"

# ------------------------------------------------------------------------------
# Step 1: Full-data benchmark runs for all head types (rho = 1.0)
# ------------------------------------------------------------------------------
echo ""
echo "----------------------------------------------------------------------"
echo "🔹 Step 1: Training heads on full data (rho = 1.0, seed 42)"
echo "----------------------------------------------------------------------"

for HEAD in linear mlp quantum reup; do
  echo "▶ Training '${HEAD}' head..."
  "${PYTHON_BIN}" scripts/train_heads_on_embeddings.py \
    --head_type "${HEAD}" \
    --seed 42 \
    --output_dir "results/heads/${HEAD}"
done

# ------------------------------------------------------------------------------
# Step 2: Training size (rho) ablation grid (rho = 0.25, 0.5, 1.0)
# ------------------------------------------------------------------------------
echo ""
echo "----------------------------------------------------------------------"
echo "🔹 Step 2: Running training size (rho) ablation grid"
echo "----------------------------------------------------------------------"

"${PYTHON_BIN}" scripts/run_mit_training_size_ablation.py \
  --fractions "0.25,0.5,1.0" \
  --heads "linear,mlp,quantum,reup" \
  --n_lowdata_seeds 3 \
  --lowdata_output_root "results/mit_training_size_ablation"

# ------------------------------------------------------------------------------
# Step 3: Result aggregation
# ------------------------------------------------------------------------------
echo ""
echo "----------------------------------------------------------------------"
echo "🔹 Step 3: Aggregating ablation metrics into markdown report"
echo "----------------------------------------------------------------------"

"${PYTHON_BIN}" scripts/aggregate_mit_training_size_ablation.py \
  --ablation_root "results/mit_training_size_ablation"

# ------------------------------------------------------------------------------
# Step 4: Generate figures
# ------------------------------------------------------------------------------
echo ""
echo "----------------------------------------------------------------------"
echo "🔹 Step 4: Generating visualization figures"
echo "----------------------------------------------------------------------"

mkdir -p figures
"${PYTHON_BIN}" scripts/generate_figures.py \
  --ablation_table "results/mit_training_size_ablation/ablation_table.json" \
  --output_dir "figures"

echo ""
echo "======================================================================"
echo "✅ Reproducibility Pipeline Execution Finished Successfully!"
echo "======================================================================"
echo "Aggregated report : results/mit_training_size_ablation/ablation_report.md"
echo "Ablation summary  : results/mit_training_size_ablation/ablation_table.json"
echo "Generated figures : figures/"
echo "======================================================================"
