# Quantum Machine Learning Post-Hoc Classifiers for Metal-Insulator Transition (QML-MIT) Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![PennyLane 0.30+](https://img.shields.io/badge/pennylane-0.30+-purple.svg)](https://pennylane.ai/)
[![License: Noncommercial](https://img.shields.io/badge/License-Noncommercial-blue.svg)](LICENSE)

Official reproducibility repository for **Quantum Machine Learning Post-Hoc Classifiers on Frozen Graph Embeddings for Metal-Insulator Transition (MIT) Classification in Crystalline Materials**.

---

## 📌 Abstract & Overview

Predicting Metal-Insulator Transitions (MIT) in crystalline solid-state compounds is a challenging problem in materials chemistry due to complex electron correlation and subtle structural distortions. 

This repository provides full source code, datasets, split configurations, and evaluation pipelines for training classical and quantum variational classifier heads on **64-dimensional frozen graph embeddings** produced by a Crystal Graph Convolutional Neural Network (CGCNN):

- **Classical Heads**: `Linear` logistic regression, Multi-Layer Perceptrons (`MLP`), Fourier features, and parameter-matched bottleneck controls.
- **Quantum Heads**: Variational Quantum Circuits (`VQC`) via PennyLane `AngleEmbedding(RY)` + `BasicEntanglerLayers`, and Data Re-Uploading (`REUP`) architectures.
- **Evaluation Protocol**: Grouped 5-fold cross-validation preventing chemical compound leakage, low-data positive fraction ($\rho$) ablations, fold-blocked cluster bootstrapping, and paired permutation significance testing.

---

## 📁 Repository Layout

```
github_upload_package/
├── data/
│   └── cgcnn_binary_embeddings.parquet    # 64-dim CGCNN graph embeddings (316 compounds)
├── splits/
│   └── grouped_5fold/                     # Grouped CV splits (fold_0.json ... fold_4.json)
├── models/
│   ├── __init__.py                        # Package init
│   └── posthoc_heads.py                   # PyTorch/PennyLane model classes (Linear, MLP, VQC, REUP)
├── evaluation/
│   ├── fold_utils.py                      # Split index loaders
│   └── thresholds.py                      # Validation threshold tuning routines
├── scripts/
│   ├── train_heads_on_embeddings.py       # Main cross-validation training script
│   ├── run_mit_training_size_ablation.py  # Training fraction (rho) ablation orchestrator
│   ├── aggregate_mit_training_size_ablation.py # Markdown report & JSON table aggregator
│   ├── compute_dependence_aware_stats.py  # Dependence-aware bootstrap & permutation statistics
│   ├── create_embedding_variants.py       # Per-fold PCA/autoencoder compression transforms
│   ├── compare_heads_vs_cgcnn.py          # CGCNN baseline comparison tools
│   ├── generate_figures.py                # Automated figure generation script
│   └── figure_style.py                    # Matplotlib visual styling config
├── requirements.txt                       # Dependency specifications
└── run_reproducibility_pipeline.sh        # One-click end-to-end reproducibility runner
```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites & Virtual Environment

Python 3.10+ is recommended. Create and activate a clean virtual environment:

```bash
python3 -m venv qml_env
source qml_env/bin/activate
pip install --upgrade pip
```

### 2. Install Dependencies

Install required PyTorch, PennyLane, Scikit-Learn, and plotting packages:

```bash
pip install -r requirements.txt
```

### 3. One-Click Reproducibility Pipeline

Run the automated reproducibility script to test model training, training-size ablation, metrics aggregation, and figure generation in one step:

```bash
PYTHON_BIN=python3 bash run_reproducibility_pipeline.sh
```

---

## ⚙️ Detailed Usage & Command Examples

### A. Training Single Classifier Heads (Full Data, $\rho = 1.0$)

Train individual head architectures across 5-fold grouped CV:

```bash
# Linear Logistic Head
python scripts/train_heads_on_embeddings.py --head_type linear --output_dir results/heads/linear

# Multi-Layer Perceptron (MLP)
python scripts/train_heads_on_embeddings.py --head_type mlp --hidden_dim 32 --output_dir results/heads/mlp

# Variational Quantum Circuit (VQC)
python scripts/train_heads_on_embeddings.py --head_type quantum --n_qubits 4 --n_q_layers 2 --output_dir results/heads/vqc

# Data Re-Uploading (REUP) Circuit
python scripts/train_heads_on_embeddings.py --head_type reup --reup_n_qubits 2 --reup_n_layers 1 --output_dir results/heads/reup
```

### B. Running Training Size ($\rho$) Ablation Sweeps

Subsample positive MIT training compounds across $\rho \in [0.05, 1.0]$ over 5 seeds:

```bash
python scripts/run_mit_training_size_ablation.py \
  --fractions "0.05,0.1,0.15,0.2,0.25,0.333333,0.5,0.75,1.0" \
  --heads "linear,mlp,quantum,reup" \
  --n_lowdata_seeds 5 \
  --lowdata_output_root "results/mit_training_size_ablation"
```

### C. Aggregating Results & Generating Reports

Build summary metrics and Markdown report comparison tables:

```bash
python scripts/aggregate_mit_training_size_ablation.py \
  --ablation_root "results/mit_training_size_ablation"
```

### D. Generating Figures

Generate publication-ready PDF and PNG figures in `paper/figures/`:

```bash
python scripts/generate_figures.py \
  --ablation_table "results/mit_training_size_ablation/ablation_table.json" \
  --output_dir "figures"
```

---

## 📊 Dataset Specifications

- **Compounds**: 316 solid-state materials with valid graph structure representations.
- **Classes**: 59 MIT positive (1) compounds, 257 non-MIT negative (0) compounds (~18.7% prevalence).
- **Features**: 64-dimensional graph embedding output from pre-trained CGCNN (`emb_0` ... `emb_63`).
- **Grouping**: Grouped by reduced chemical formula to prevent compound leakage between train, validation, and test folds.

---

## 📄 License & Citation

Original INQUIRE Lab code is licensed under PolyForm Noncommercial License 1.0.0. Original INQUIRE Lab datasets, figures, and documentation are licensed under CC BY-NC 4.0. See [LICENSE](LICENSE) for scope and the full license texts. Materials from other rights holders retain their original terms.

If you use this repository or dataset in your research, please cite:

```bibtex
@article{qml_mit_classification_2026,
  title={Quantum Machine Learning Post-Hoc Classifiers on Graph Embeddings for Metal-Insulator Transition Prediction},
  author={LoRA MIT Collaboration},
  journal={arXiv preprint},
  year={2026}
}
```
