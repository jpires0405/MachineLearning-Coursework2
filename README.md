# Active Learning on a Budget: TPC_RP and DynTPC

## Overview

This repository contains a from-scratch implementation of the **TPC_RP** (Typical Clustering with Representation learning and Parametric clustering) algorithm from:

> Hacohen, G., Dekel, A., & Weinshall, D. (2022). *Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets.* ICML 2022.

The codebase includes a full active learning evaluation pipeline on CIFAR-10, a multi-seed statistical comparison against a random selection baseline, and **DynTPC** — a novel dynamic phase-shift modification that encodes the paper's theoretical budget threshold directly into the sample selection rule, switching from typicality-maximising to typicality-minimising queries at a configurable phase threshold.

## Repository Structure

```
MachineLearning-Coursework2/
│
├── src/                    # Core Python modules
│   ├── data_pipeline.py    # ActiveLearningDataset: labeled/unlabeled pool management
│   ├── feature_extractor.py# ResNet-18 encoder producing L2-normalised 512-D embeddings
│   ├── tpcrp_sampler.py    # tpcrp_query(), dynamic_tpcrp_query(), random_query()
│   └── evaluator.py        # Linear probe evaluation via logistic regression
│
├── notebooks/              # Jupyter notebook with full pipeline and experiments
│   └── TPCRP_Active_Learning.ipynb
│
├── figures/                # Generated plots
│   ├── tpcrp_baseline_curve.png    # Single-seed TPC_RP learning curve
│   ├── statistical_comparison.png  # TPC_RP vs Random (mean ± std, 5 seeds)
│   └── dyntpc_comparison.png       # Three-way comparison: TPC_RP, Random, DynTPC
│
├── report/                 # LaTeX source for the 2-page coursework report
│   ├── main.tex
│   └── references.bib
│
├── docs/                   # PDF and HTML exports of the executed notebook
│
├── data/                   # (Gitignored) CIFAR-10 dataset downloads
│
├── requirements.txt        # Python dependencies
└── README.md
```

## Requirements

Install dependencies into a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Open `notebooks/TPCRP_Active_Learning.ipynb` and run cells sequentially.
The notebook is designed to run on **Google Colab** (mount your Drive at
`/content/drive/MyDrive/MachineLearning-Coursework2`) or locally with the
`.venv` activated.

## References

Hacohen, G., Dekel, A., & Weinshall, D. (2022). Active Learning on a Budget:
Opposite Strategies Suit High and Low Budgets. *ICML 2022*, PMLR 162.
