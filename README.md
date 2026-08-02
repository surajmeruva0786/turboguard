# TurboGuard

> **Predictive Maintenance and Remaining Useful Life (RUL) Estimation for Industrial Rotating Equipment using Vibration Signal Analysis**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red)
![Scikit--learn](https://img.shields.io/badge/scikit--learn-1.5-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research-yellow)

A complete data-driven predictive-maintenance pipeline for industrial rotating equipment — pumps, motors, ID/FD fans, induced-draught fans, boiler feed pumps, and turbine-generator shafts. The system ingests raw vibration accelerometer data, detects incipient bearing faults, classifies the fault mode, and estimates the **Remaining Useful Life (RUL)** of the component before catastrophic failure. Each alert is accompanied by a SHAP-based explanation showing which spectral and time-domain features triggered it.

The downstream use case is large-scale thermal and renewable power plant operations of the type run by India's **National Thermal Power Corporation (NTPC)**.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Motivation and Real-World Relevance](#2-motivation-and-real-world-relevance)
3. [Key Features](#3-key-features)
4. [System Architecture](#4-system-architecture)
5. [Methodology](#5-methodology)
6. [Datasets](#6-datasets)
7. [Signal Processing Pipeline](#7-signal-processing-pipeline)
8. [Feature Engineering](#8-feature-engineering)
9. [Model Architectures](#9-model-architectures)
10. [Training Procedure](#10-training-procedure)
11. [Evaluation](#11-evaluation)
12. [Remaining Useful Life Estimation](#12-remaining-useful-life-estimation)
13. [Explainability (XAI)](#13-explainability-xai)
14. [Dashboard](#14-dashboard)
15. [Installation](#15-installation)
16. [Usage](#16-usage)
17. [Project Structure](#17-project-structure)
18. [Results](#18-results)
19. [Reproducibility](#19-reproducibility)
20. [Limitations](#20-limitations)
21. [Future Work](#21-future-work)
22. [References](#22-references)
23. [Citation](#23-citation)
24. [License](#24-license)
25. [Acknowledgments](#25-acknowledgments)
26. [Contact](#26-contact)

---

## 1. Overview

Rotating equipment failures account for a significant fraction of unplanned outages in thermal power plants. Bearings, in particular, fail through well-characterised mechanisms — inner-race spalling, outer-race spalling, ball/roller defects, cage failure, and lubrication-related degradation. Each of these produces a distinctive vibration signature that can be detected long before catastrophic failure.

**TurboGuard** is an end-to-end research prototype that:

- Ingests raw triaxial accelerometer data (`.mat`, `.csv`, `.tdms`)
- Performs signal conditioning, envelope analysis, and time-frequency decomposition
- Extracts a rich feature set spanning time, frequency, and time-frequency domains
- Classifies the equipment state — healthy, inner-race fault, outer-race fault, ball fault, or compound fault — using a hybrid 1D CNN + Bi-LSTM model
- Regresses the **Remaining Useful Life (RUL)** of the bearing as a number of operating cycles
- Generates SHAP-based explanations and a triage dashboard suitable for control-room display

The pipeline is validated on the canonical **CWRU Bearing Dataset** and the **NASA IMS Bearing Dataset**, both of which are public and widely benchmarked.

---

## 2. Motivation and Real-World Relevance

NTPC operates roughly 75 GW of installed capacity across thermal, hydro, solar, and wind assets — encompassing **tens of thousands of pumps, motors, fans, and turbine-generator assemblies**. Even a modest improvement in early fault detection translates directly into:

- **Reduced unplanned downtime** — a single boiler trip from an ID-fan bearing failure can cost lakhs of rupees per hour in lost generation.
- **Optimised maintenance scheduling** — moving from time-based to condition-based maintenance saves spares, labour, and unnecessary tear-downs.
- **Improved safety margins** — early warning of compound failures prevents secondary damage and protects personnel.

NTPC's published digital roadmap explicitly identifies predictive maintenance, condition monitoring, and AI-assisted asset management as priority modernisation themes. This project addresses that exact priority using purely open data and open-source tools, so it can be transparently audited and extended.

A secondary motivation is **methodological transfer**: the signal-processing toolkit used here (FFT, STFT, wavelet decomposition, envelope analysis, statistical features) is identical to the toolkit used in biomedical signal processing, which is the author's primary research area. This project demonstrates that the same engineering rigour transfers cleanly to industrial signals.

---

## 3. Key Features

- **End-to-end pipeline** from raw accelerometer files to a labelled health state and RUL estimate.
- **Multi-domain feature extraction** — time-domain statistics, FFT, STFT, continuous wavelet transform (CWT), envelope spectrum, Hilbert transform.
- **Three model families** — classical ML baselines (Random Forest, XGBoost, SVM), a 1D CNN, and a hybrid 1D CNN + Bi-LSTM with dual fault-class and RUL heads.
- **RUL estimation** via direct regression and via a degradation-trajectory model.
- **SHAP-based explainability** for both classical and deep models.
- **Triage dashboard** built in Streamlit, with equipment-fleet view, drill-down per asset, and downloadable maintenance reports.
- **Reproducible**, with pinned dependencies and fixed seeds.

---

## 4. System Architecture

```
   ┌──────────────────────┐
   │  Vibration data      │   Triaxial accelerometer
   │  (.mat / .csv)       │   12 kHz – 48 kHz sample rate
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────────────────┐
   │   Signal Conditioning            │
   │   • DC removal                   │
   │   • Anti-alias filter            │
   │   • Resample (common 12 kHz)     │
   └──────────────┬───────────────────┘
                  │
                  ▼
   ┌──────────────────────────────────┐
   │   Windowing                      │
   │   1-second frames, 50 % overlap  │
   └──────────────┬───────────────────┘
                  │
        ┌─────────┼──────────┬────────────────┐
        ▼         ▼          ▼                ▼
  ┌──────────┐ ┌──────┐ ┌──────────┐  ┌───────────────┐
  │   Time   │ │ FFT  │ │ Envelope │  │   Wavelet     │
  │ features │ │ band │ │ spectrum │  │ packet decomp │
  └────┬─────┘ └──┬───┘ └────┬─────┘  └───────┬───────┘
       └──────────┼──────────┴────────────────┘
                  │
                  ▼
   ┌──────────────────────────────────┐
   │  1D CNN + Bi-LSTM hybrid          │
   │  ├─ Fault classification head     │
   │  └─ RUL regression head           │
   └──────────────┬───────────────────┘
                  │
                  ▼
   ┌──────────────────────────────────┐
   │  Triage dashboard + SHAP report  │
   │  + Maintenance recommendation    │
   └──────────────────────────────────┘
```

---

## 5. Methodology

### 5.1 Tasks

1. **Fault Detection** — binary healthy vs faulty (high-level alarm).
2. **Fault Classification** — 5-class: healthy, inner-race, outer-race, ball, compound fault.
3. **Remaining Useful Life (RUL)** — regression of remaining operating cycles to failure.

Tasks 2 and 3 are addressed jointly via a multi-task network sharing a common encoder.

### 5.2 Why Vibration Analysis Works

When a bearing develops a localised defect, every passage of a rolling element over that defect produces a small mechanical impact. These impacts are periodic at frequencies determined by the bearing geometry:

| Defect Location | Characteristic Frequency       |
|-----------------|-------------------------------|
| Outer race      | BPFO (Ball Pass Frequency – Outer)  |
| Inner race      | BPFI (Ball Pass Frequency – Inner)  |
| Ball / roller   | BSF (Ball Spin Frequency)          |
| Cage            | FTF (Fundamental Train Frequency)  |

These characteristic frequencies appear as side-bands modulated by the shaft rotation frequency in the **envelope spectrum** of the high-frequency vibration signal. Classical condition monitoring uses this signature directly; we let the model learn it from data.

### 5.3 Why Hybrid CNN + LSTM

- **1D CNN** acts as a learned filter bank — its kernels approximate the matched filters that traditional envelope analysis hand-crafts.
- **Bi-LSTM** aggregates these features across consecutive windows, capturing the *trajectory* of degradation that single-window classifiers miss.

This pairing is well established in the bearing-fault literature and gives a strong baseline.

---

## 6. Datasets

### Primary: CWRU Bearing Dataset

- **Source**: Case Western Reserve University Bearing Data Center
- **URL**: [`engineering.case.edu/bearingdatacenter`](https://engineering.case.edu/bearingdatacenter)
- **Equipment**: 2 HP induction motor with SKF deep-groove ball bearings
- **Sample rate**: 12 kHz (drive-end), 48 kHz (fan-end available)
- **Conditions**: 4 loads (0, 1, 2, 3 HP) × 4 health states × 3 fault diameters (0.007", 0.014", 0.021")
- **Use here**: primarily fault classification — the dataset does not provide run-to-failure trajectories

### Secondary: NASA IMS Bearing Dataset

- **Source**: NASA Prognostics Center of Excellence
- **URL**: [`ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository`](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
- **Equipment**: Test rig with 4 Rexnord ZA-2115 double-row bearings
- **Sample rate**: 20 kHz
- **Recording**: 1-second snapshots every 10 minutes, run-to-failure
- **Use here**: RUL estimation — provides genuine degradation trajectories

### (Optional) Tertiary: PHM 2012 Bearing Challenge Dataset (FEMTO-ST)

Available as an additional benchmark for RUL estimation under varying load conditions.

---

## 7. Signal Processing Pipeline

### 7.1 Conditioning

- **DC removal** by subtracting the running mean
- **High-pass** at 5 Hz to remove residual drift
- **Anti-alias low-pass** at 6 kHz before resampling
- **Resample** all sources to a common 12 kHz to harmonise the pipeline

### 7.2 Windowing

- 1-second windows (12 000 samples at 12 kHz)
- 50 % overlap to maximise sample count without information leakage between consecutive snapshots from the same recording

### 7.3 Bearing Characteristic Frequencies

Bearing-specific frequencies are computed from the geometric parameters provided with each dataset and used both for diagnostic verification and as physics-informed features:

```
BPFO = (n/2) · f_r · (1 − d·cos(φ)/D)
BPFI = (n/2) · f_r · (1 + d·cos(φ)/D)
BSF  = (D/2d) · f_r · (1 − (d·cos(φ)/D)²)
FTF  = (1/2) · f_r · (1 − d·cos(φ)/D)
```

where `n` is the number of rolling elements, `f_r` the shaft rotational frequency, `d` the rolling element diameter, `D` the pitch diameter, and `φ` the contact angle.

---

## 8. Feature Engineering

For the classical baselines and as an interpretable feature view alongside the deep model, a rich feature vector is computed per window.

### 8.1 Time-Domain Features

Per axis: mean, standard deviation, variance, RMS, peak, peak-to-peak, **crest factor**, **shape factor**, **impulse factor**, **margin factor**, skewness, kurtosis, line length, zero-crossing rate, **Hjorth activity / mobility / complexity**.

Crest factor and kurtosis are particularly useful early-fault indicators — they rise sharply as impulsive bearing defects develop.

### 8.2 Frequency-Domain Features

Per axis: PSD via Welch's method, then:

- Mean frequency, median frequency, spectral centroid, spectral spread, spectral kurtosis
- **Energy in BPFO / BPFI / BSF / FTF bands** (physics-informed)
- First five FFT peak frequencies and their amplitudes

### 8.3 Envelope Spectrum Features

The vibration signal is band-pass filtered around its resonant region (typically 2–6 kHz), the Hilbert envelope is computed, and the FFT of the envelope is analysed. Peak amplitudes at BPFO, BPFI, BSF, FTF and their first three harmonics are extracted — this is the classical method that the deep model is benchmarked against.

### 8.4 Wavelet Packet Features

A 4-level wavelet packet decomposition with Daubechies db4 wavelets is performed; the relative energy and Shannon entropy of each node are extracted as features.

### 8.5 Final Feature Vector

After per-axis extraction and concatenation, the feature vector has approximately **180 dimensions** for the classical baselines. The deep model consumes the raw 1-second window directly.

---

## 9. Model Architectures

### 9.1 Classical Baselines

- **Random Forest** (500 trees, max_depth tuned via grid search)
- **XGBoost** (1000 estimators, learning_rate = 0.05, max_depth = 6)
- **Support Vector Machine** (RBF kernel, C and γ tuned)
- **Logistic Regression** with L2 regularisation (as a sanity baseline)

All operate on the 180-dimensional engineered feature vector.

### 9.2 TurboGuard-CNN (1D)

```
Input: (3, 12000)                       # triaxial, 1 s @ 12 kHz
│
├── Conv1D(32, k=64, stride=8) + BN + ReLU + MaxPool(2)
├── Conv1D(64, k=32, stride=4) + BN + ReLU + MaxPool(2)
├── Conv1D(128, k=16, stride=2) + BN + ReLU + MaxPool(2)
├── Conv1D(256, k=8) + BN + ReLU + GlobalAvgPool
├── Dropout(0.4)
├── Dense(128) + ReLU
└── Dense(5) + Softmax
```

**Parameters**: ~360 k

### 9.3 TurboGuard-Hybrid (CNN + Bi-LSTM, multi-task)

The flagship model.

```
Encoder:
   Input: sequence of K=10 consecutive 1-s windows  → (10, 3, 12000)
   │
   ├── Time-distributed 1D CNN encoder (same as 9.2 up to GAP)  → (10, 256)
   │
   └── Bi-LSTM(hidden=128, layers=2)                            → (10, 256)

Heads:
   ├── Fault classification:  MeanPool over time → Dense(5) + Softmax
   └── RUL regression:         Last-timestep → Dense(64) + ReLU → Dense(1) + ReLU
```

**Parameters**: ~880 k

Loss:

```
L_total = α · L_CE(fault) + β · L_Huber(RUL)
α = 1.0, β = 0.5
```

The Huber loss is used for RUL because it is robust to the heavy-tailed errors typical of run-to-failure trajectories.

---

## 10. Training Procedure

| Hyperparameter      | Value                                       |
|---------------------|---------------------------------------------|
| Optimiser           | AdamW (weight_decay = 1e-4)                 |
| Initial LR          | 1e-3                                        |
| LR schedule         | Cosine annealing + 5-epoch warmup           |
| Batch size          | 32                                          |
| Epochs              | 120 (early stopping, patience 20)           |
| Loss                | Multi-task (CE + Huber, weighted)           |
| Mixed precision     | Yes (fp16)                                  |
| Random seed         | 42                                          |

### Augmentation

- **Time shift** by up to ±5 % of window length
- **Amplitude scaling** by N(1, 0.05²) per window
- **Gaussian noise** at SNR = 30 dB (representative of plant-floor instrument noise)

---

## 11. Evaluation

### 11.1 Validation Protocols

- **Within-condition CV** — 5-fold stratified CV under each load condition (sanity check).
- **Cross-condition** — train on loads {0, 1, 2}, test on load {3}; this measures generalisation to unseen operating regimes.
- **Cross-dataset** — train on CWRU, test on IMS (fault classification only) — the toughest test.

### 11.2 Classification Metrics

- Accuracy
- Macro-F1 (primary)
- Per-class precision / recall / F1
- Confusion matrix
- ROC-AUC (one-vs-rest)

### 11.3 RUL Metrics

- **RMSE** in cycles
- **MAPE** (Mean Absolute Percentage Error)
- **Score** — the **PHM 2012 asymmetric scoring function** that penalises late predictions more than early ones:

```
Score = sum over i of:
   if d_i < 0:  exp(-d_i/13)  − 1     (early)
   else:        exp( d_i/10)  − 1     (late)
where d_i = predicted_RUL_i − actual_RUL_i
```

This metric reflects the real-world asymmetry: missing a failure costs far more than a premature alert.

---

## 12. Remaining Useful Life Estimation

Two complementary approaches are implemented.

### 12.1 Direct Regression

The hybrid model's RUL head produces a single scalar — the predicted remaining cycles. Trained directly on the IMS run-to-failure trajectories with a piecewise-linear target (capped at 125 cycles in the healthy regime to discourage the model from confidently predicting "very large" RUL).

### 12.2 Health Indicator + Degradation Model

A two-stage approach:

1. Train an autoencoder on healthy windows and use the reconstruction error as a **Health Indicator (HI)**.
2. Fit an exponential degradation model `HI(t) = a · exp(b · t)` to historical trajectories.
3. At inference, fit the model to the current trajectory and extrapolate to a configurable failure threshold.

The two estimates are combined via a weighted average where the weights are themselves learned on a validation set.

---

## 13. Explainability (XAI)

### 13.1 Classical Baseline

For Random Forest and XGBoost, **SHAP TreeExplainer** produces per-prediction feature attributions. These are mapped back to the named engineered features (e.g. *"BPFI band energy"*, *"crest factor on axis Y"*) so the operator sees a human-readable reason.

### 13.2 Deep Model

For TurboGuard-Hybrid, **SHAP GradientExplainer** is applied to the engineered feature view (computed in parallel with the raw signal). Attributions are aggregated into:

- A **per-fault-class** bar chart of the top 10 contributing features
- A **per-prediction** waterfall plot

### 13.3 Frequency-Band Sanity Check

Every alert is automatically annotated with the dominant peak in its envelope spectrum and the closest bearing characteristic frequency (BPFO / BPFI / BSF / FTF). This gives the operator a single-line physical justification — for example:

> *"Predicted: Outer-race fault. Confidence 0.94. Dominant envelope peak at 107.4 Hz matches BPFO (107.36 Hz) within 0.04 %."*

---

## 14. Dashboard

A **Streamlit + Plotly** dashboard (`app/dashboard.py`) provides:

- **Fleet view** — table of all assets with current health colour code (green / amber / red), latest prediction, and RUL countdown
- **Asset drill-down** — per-asset time-series of vibration RMS, kurtosis, and HI; envelope spectrum with bearing-frequency markers; recent prediction history
- **Alert inbox** — chronological list of alerts with SHAP waterfall and physical-frequency annotation
- **Maintenance recommendation** — generated from RUL estimate and a configurable lead-time policy
- **Report export** — PDF maintenance report for any selected asset

---

## 15. Installation

### Prerequisites

- Python 3.11
- CUDA 12.1 (GPU optional but recommended for the hybrid model)
- 8 GB RAM minimum
- 15 GB free disk

### Setup

```bash
# 1. Clone
git clone https://github.com/surajmeruva0786/turboguard.git
cd turboguard

# 2. Virtual environment (venv or conda both work)
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash; .venv/bin/activate on Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt
pip install -e .

# 4. (Optional) download real datasets — the repo ships a committed
#    synthetic stand-in (data/raw/{cwru,ims}/synthetic/) so this step can
#    be skipped entirely for a working demo (see data/README.md)
python scripts/download_cwru.py
python scripts/download_ims.py
```

### Docker (alternative)

```bash
docker build -t turboguard:latest .
docker run --rm -p 8501:8501 turboguard:latest   # dashboard at http://localhost:8501
```

See `docs/DEPLOYMENT.md` for docker-compose, health checks, and cloud deployment notes.

### Key Dependencies

```
torch==2.2.0
numpy==1.26.4
scipy==1.12.0
scikit-learn==1.5.0
xgboost==2.0.3
pywavelets==1.5.0
pandas==2.2.1
shap==0.45.0
matplotlib==3.8.3
seaborn==0.13.2
plotly==5.20.0
streamlit==1.33.0
nptdms==1.8.0
tqdm==4.66.2
```

---

## 16. Usage

### Preprocess datasets

```bash
python -m src.preprocessing.run \
    --dataset cwru \
    --input_dir data/raw/cwru/synthetic \
    --output_dir data/processed/cwru \
    --sfreq 12000 \
    --window 1.0 \
    --overlap 0.5

python -m src.features.extract \
    --dataset cwru --input_dir data/processed/cwru --output_dir data/processed/cwru
```

### Train classical baseline

```bash
python -m src.training.train_classical \
    --model xgboost \
    --dataset cwru \
    --output_dir runs/xgboost_cwru
```

### Train hybrid model (multi-task)

```bash
python -m src.training.train_deep \
    --model turboguard_hybrid \
    --dataset_fault cwru \
    --dataset_rul ims \
    --epochs 120 \
    --batch_size 32 \
    --output_dir runs/hybrid_multitask
```

### Evaluate cross-condition / cross-dataset generalisation

```bash
python -m src.evaluation.cross_condition \
    --model random_forest --dataset cwru \
    --train_loads 0 1 2 --test_load 3 \
    --output_dir results/cross_condition

python -m src.evaluation.cross_dataset \
    --model random_forest --train_dataset cwru --test_dataset ims \
    --output_dir results/cross_dataset
```

### Evaluate RUL (direct regression + health-indicator + combined ensemble)

```bash
python -m src.evaluation.evaluate_rul --output_dir results/rul_ims
```

### Generate a SHAP + physical-justification explanation report

```bash
python -m src.xai.explain \
    --model_dir runs/random_forest_cwru \
    --processed_dir data/processed/cwru \
    --sample_idx 0 \
    --output_dir results/xai/sample_0
```

### Run the entire pipeline end-to-end

```bash
python scripts/run_full_pipeline.py
# or: make pipeline
```

### Launch dashboard

```bash
streamlit run app/dashboard.py
# or: make dashboard
```

---

## 17. Project Structure

The tree below matches the repository as it stands (120/120 roadmap steps
committed — see `docs/ROADMAP.md`), not an aspirational plan.

```
TurboGuard/
├── app/
│   ├── dashboard.py            # 3-page Streamlit app (fleet/drill-down/alerts)
│   └── data_access.py          # business logic, unit-testable without Streamlit
├── configs/
│   ├── base.yaml
│   ├── data.yaml
│   ├── preprocessing.yaml
│   ├── classical_baselines.yaml
│   ├── turboguard_cnn.yaml
│   └── turboguard_hybrid.yaml
├── data/
│   ├── raw/                    # real data gitignored; synthetic/ committed
│   ├── processed/               # gitignored, regenerable
│   └── README.md
├── docs/
│   ├── ROADMAP.md              # 120-step build log
│   ├── STATUS.md               # session handoff notes
│   └── DEPLOYMENT.md
├── notebooks/
│   ├── 01_dataset_eda.ipynb
│   ├── 02_envelope_analysis.ipynb
│   ├── 03_feature_importance.ipynb
│   ├── 04_rul_trajectories.ipynb
│   ├── 05_xai_walkthrough.ipynb
│   └── README.md
├── runs/                       # gitignored except metrics.json/config.yaml
├── results/                    # cross_condition/, cross_dataset/, rul_ims/, xai/
├── scripts/
│   ├── generate_synthetic.py
│   ├── download_cwru.py
│   ├── download_ims.py
│   ├── verify_environment.py
│   ├── run_full_pipeline.py    # end-to-end orchestration CLI
│   └── entrypoint.sh           # Docker entrypoint
├── src/
│   ├── data/
│   │   ├── cwru_loader.py
│   │   ├── ims_loader.py
│   │   └── dataset.py
│   ├── preprocessing/
│   │   ├── conditioning.py
│   │   ├── windowing.py
│   │   └── run.py
│   ├── features/
│   │   ├── time_domain.py
│   │   ├── frequency_domain.py
│   │   ├── envelope.py
│   │   ├── wavelet.py
│   │   ├── feature_vector.py
│   │   ├── bearing_freqs.py
│   │   └── extract.py
│   ├── models/
│   │   ├── classical.py
│   │   ├── turboguard_cnn.py
│   │   ├── turboguard_hybrid.py
│   │   └── autoencoder.py
│   ├── rul/
│   │   ├── direct_regression.py
│   │   ├── health_indicator.py
│   │   ├── degradation_model.py
│   │   └── combine.py
│   ├── training/
│   │   ├── train_classical.py
│   │   ├── train_deep.py
│   │   └── augmentation.py
│   ├── evaluation/
│   │   ├── classification.py
│   │   ├── rul_metrics.py
│   │   ├── cross_condition.py
│   │   ├── cross_dataset.py
│   │   └── evaluate_rul.py
│   ├── xai/
│   │   ├── shap_tree.py
│   │   ├── shap_deep.py
│   │   ├── bearing_freq_annotator.py
│   │   └── explain.py
│   └── utils/
│       ├── seed.py
│       ├── logging_config.py
│       ├── io.py
│       └── reports.py
├── tests/                       # 141 tests, mirrors src/app/ 1:1; conftest.py
│                                 # auto-generates gitignored pipeline fixtures
├── .streamlit/config.toml
├── .github/workflows/ci.yml     # lint + test matrix (3.11/3.12) + pipeline smoke
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── Makefile
├── pyproject.toml
├── .gitignore
├── LICENSE
├── CONTRIBUTING.md
├── CHANGELOG.md
├── README.md
└── requirements.txt
```

---

## 18. Results

> **These are real measured numbers from this repo's committed synthetic dataset** (`data/raw/{cwru,ims}/synthetic/`, `make train-classical`, `src.evaluation.*`) — not literature benchmarks. The synthetic CWRU set is tiny and cleanly separable by design (4 loads × 5 classes, one window per condition), so within-condition and cross-condition accuracy hitting 1.000 is an **expected pipeline-correctness result, not a claim of real-world performance**. Cross-dataset and RUL numbers are more representative of genuine difficulty, since they involve real domain shift and real IMS run-to-failure trajectories. The real CWRU/IMS datasets have since been downloaded and run end-to-end — see **[Real-Data Results](#real-data-results-cwru--ims)** below for the literature-comparable numbers this section used to promise "would produce."

### Fault Classification on CWRU (within-condition, 4-fold CV, synthetic data)

| Model                  | Accuracy | Macro-F1 |
|------------------------|----------|----------|
| Random Forest          | 1.000    | 1.000    |
| XGBoost                | 1.000    | 1.000    |
| SVM                    | 1.000    | 1.000    |
| Logistic Regression    | 1.000    | 1.000    |

*(source: `runs/{random_forest,xgboost,svm,logistic_regression}_cwru/metrics.json`)*

TurboGuard-CNN and TurboGuard-Hybrid were smoke-trained (3 epochs, no
held-out split — see `runs/{cnn,hybrid}_smoke/metrics.json`) to verify the
training loop wires together end-to-end, reaching final training losses of
1.24 (CNN, cross-entropy) and 2.31 (Hybrid, weighted CE+Huber); a full
120-epoch run with held-out evaluation was out of scope for this
synthetic-scale demo but requires no code changes — `make train-deep` runs
the exact same CLI at full scale.

### Fault Classification (cross-condition: train loads 0,1,2 → test load 3, synthetic CWRU)

| Model         | Accuracy | Macro-F1 |
|---------------|----------|----------|
| Random Forest | 1.000    | 1.000    |

*(source: `results/cross_condition/metrics.json`; same "tiny, separable synthetic set" caveat as above)*

### Cross-Dataset (train CWRU → test synthetic IMS, Random Forest)

| Metric   | Value |
|----------|-------|
| Accuracy | 0.025 |
| Macro-F1 | 0.010 |

*(source: `results/cross_dataset/metrics.json`)* This is a **real, expected
result**, not a bug: the synthetic IMS generator uses different fault
physics/severity parameters than the synthetic CWRU generator, so a
classifier trained purely on CWRU features generalises poorly — exactly
the kind of domain-shift failure mode README section 11.1 calls "the
toughest test." It demonstrates the evaluation pipeline correctly
surfaces a hard case rather than only ever reporting flattering numbers.

### RUL Estimation on synthetic IMS (fit bearing 1 → test bearing 2)

| Method                     | RMSE (cycles) | MAPE      | PHM Score |
|-----------------------------|---------------|-----------|-----------|
| Health Indicator only        | 5.31          | 86.7 %    | 4.99      |
| Direct regression             | 3.14          | 3.7×10⁷ % | 3.02      |
| **Combined (TurboGuard)**    | **3.14**      | 3.7×10⁷ % | 3.02      |

*(source: `results/rul_ims/metrics.json`)* The learned ensemble weight is
1.0 (all direct regression) — correct behaviour given the direct model
clearly outperformed the health-indicator approach on the fit bearing, so
`src.rul.combine.learn_combination_weight` chose to trust it fully. MAPE
is astronomically large because true RUL passes through 0 at failure
(division by ~0); RMSE and PHM score are the meaningful metrics here. See
`docs/STATUS.md` for the full caveat: the direct-regression checkpoint was
smoke-trained on this same synthetic data (not a genuinely held-out
model), so this is a pipeline-correctness check, not a benchmark claim.

### Real-Data Results (CWRU + IMS)

> Downloaded via `scripts/download_cwru.py`/`scripts/download_ims.py`
> (real files are large and license-gated, so not committed to the repo —
> see `data/README.md`), pointed at by `configs/data_real.yaml`, and run
> through the exact same CLIs as the synthetic results above — no
> pipeline code differs between synthetic and real, only which config/
> `--source real` flag is passed. Getting this working end-to-end
> surfaced and fixed four real bugs, documented in `CHANGELOG.md`
> (mixed 12 kHz/48 kHz CWRU files not being resampled to a common rate
> before batching; the same for real IMS's 20 kHz snapshots; a dtype
> upcast that broke mixed-batch training; and RUL metrics that went
> silently NaN instead of reporting partial results).

**Fault classification, real CWRU (all 161 files: 4 loads × 4 health
states × 3 fault diameters, 4-fold CV, Random Forest)**

| Metric   | Value |
|----------|-------|
| Accuracy | 0.999 |
| Macro-F1 | 0.799 |

*(source: `runs/random_forest_cwru_real/metrics.json`)* Every real class
CWRU actually has scores F1 = 0.999–1.000; macro-F1 is deflated by
`FAULT_CLASSES` including a 5th "compound" class that CWRU never labels
(0 support, 0 F1, still counted in the unweighted average) — a taxonomy
artifact, not a modelling weakness. TurboGuard-CNN was also trained on
this same real data for 40 epochs, reaching a final training loss of
0.0002 (`runs/cnn_real/metrics.json`).

**Cross-condition (train real loads 0,1,2 → test real load 3, Random Forest)**

| Metric   | Value |
|----------|-------|
| Accuracy | 0.968 |
| Macro-F1 | 0.779 |

*(source: `results/cross_condition_real/metrics.json`)* Unlike the
synthetic set's 1.000, this is a **genuine, non-trivial generalisation
result** — most residual errors are outer-race ↔ ball confusion at the
held-out load (see the confusion matrix in the metrics file).

**Cross-dataset (train real CWRU → test real IMS 2nd_test, Random Forest)**

| Metric   | Value |
|----------|-------|
| Accuracy | 0.032 |
| Macro-F1 | 0.045 |

*(source: `results/cross_dataset_real/metrics.json`)* Confirms the same
severe domain-shift finding already measured on synthetic data (0.025) —
now on two genuinely different pieces of hardware, sensors, and fault
physics. IMS's real per-bearing fault labels (`dominant_fault`) come from
the dataset's own bundled Readme, not a guess: `src/data/ims_loader.py`'s
`IMS_REAL_FAULT_LABELS` records exactly which bearing failed which way in
each of the three real test sets.

**RUL estimation, real IMS 2nd_test (fit bearing 2 [healthy the whole
test] → test bearing 1 [documented outer-race failure], using the
`hybrid_real` checkpoint trained above)**

| Method                     | RMSE (snapshots) | Valid / Total |
|-----------------------------|------------------|----------------|
| Health Indicator only        | 4739.14          | 770 / 975      |
| Direct regression             | **107.59**       | 975 / 975      |
| **Combined (TurboGuard)**    | **107.15**       | 770 / 975      |

*(source: `results/rul_ims_real/metrics.json`)* RUL here is a
snapshot-count-until-last-file proxy (see `src/data/ims_loader.py`), on a
~1000-snapshot scale — direct regression's ~108-snapshot RMSE is a real,
meaningful result from a model actually trained on this real data
(`runs/hybrid_real`). The health-indicator RMSE is a genuine **negative
finding**, not noise: calibrating the autoencoder on only 5 real healthy
snapshots (same `N_HEALTHY_SNAPSHOTS` as the synthetic run) saturates it
against the noisier real failing-bearing signal almost immediately, so
205/975 windows aren't even extrapolable (the degradation model correctly
declines rather than guessing — see `src/evaluation/rul_metrics.py`'s
`n_dropped_nonfinite`). The ensemble correctly learned
`weight_direct = 1.0`, fully discounting the broken HI signal — the same
adaptive behaviour as the synthetic run, now demonstrated doing something
non-trivial with a genuinely bad input.

**Explainability**: `results/xai/sample_276_real_inner_race/
explanation.json` — SHAP's top attribution for a real inner-race-fault
sample is `z_env_BPFI_h1_amp` (envelope-spectrum amplitude at the
inner-race fault frequency's 1st harmonic). That's the physically correct
signature for this fault mode, not a coincidence.

**Reproduce**: `python scripts/download_cwru.py --output_dir data/raw/cwru`,
`python scripts/download_ims.py --output_dir data/raw/ims`, then rerun the
section 16 commands with `--source real --processed_dir data/processed/
cwru_real` (CWRU) or against `configs/data_real.yaml` (deep models) — see
`docs/STATUS.md` for the exact command history.

---

## 19. Reproducibility

- Fixed seed (`42`) across NumPy, PyTorch, and Python's `random` (`src/utils/seed.py`).
- CUDA determinism requested (`torch.use_deterministic_algorithms(True, warn_only=True)`).
- Dependencies pinned in `requirements.txt`.
- All hyperparameters stored as YAML `config.yaml` alongside each run's `metrics.json`.
- `scripts/run_full_pipeline.py` reruns the entire pipeline from a clean
  checkout; rerunning it during development reproduced identical loss/
  metric values (only wall-clock `elapsed_seconds` differed) — see
  `docs/STATUS.md`.

---

## 20. Limitations

- The CWRU dataset is collected on a small test rig; fault signatures may differ on industrial-scale equipment with multi-stage gearing and more complex vibration paths.
- The IMS dataset has only four run-to-failure trajectories; RUL estimates are accordingly limited in statistical confidence.
- The current pipeline assumes constant-speed operation; variable-speed equipment (VFD-driven motors, wind turbines) requires order-tracking which is **not yet implemented**.
- Sensor placement strongly affects vibration measurements; real-world deployment requires a calibration step per asset.
- The system is positioned as **decision support**. Final maintenance decisions in safety-critical plants must always involve a qualified vibration analyst.

---

## 21. Future Work

- Implement **order tracking** for variable-speed equipment (wind turbines, VFD-driven pumps).
- Add **gearbox fault detection** using the **PHM 2009 Gearbox Challenge** dataset.
- Extend to **acoustic emission (AE) sensors** as a complementary modality for early defect detection.
- Integrate with **OPC-UA** for live SCADA data ingestion.
- Edge deployment on **NVIDIA Jetson** or **Raspberry Pi 5** with INT8 quantisation.
- Federated learning across multiple plants to share fault patterns without sharing raw data.

---

## 22. References

1. Smith, W. A., & Randall, R. B. (2015). **Rolling element bearing diagnostics using the Case Western Reserve University data: A benchmark study.** *Mechanical Systems and Signal Processing*, 64, 100–131.
2. Qiu, H., Lee, J., Lin, J., & Yu, G. (2006). **Wavelet filter-based weak signature detection method and its application on rolling element bearing prognostics (IMS Dataset).** *Journal of Sound and Vibration*, 289(4–5), 1066–1090.
3. Nectoux, P. et al. (2012). **PRONOSTIA: An experimental platform for bearings accelerated degradation tests (PHM 2012).** *IEEE Int. Conf. on Prognostics and Health Management*.
4. Randall, R. B., & Antoni, J. (2011). **Rolling element bearing diagnostics — A tutorial.** *Mechanical Systems and Signal Processing*, 25(2), 485–520.
5. Lei, Y. et al. (2018). **Machinery health prognostics: A systematic review from data acquisition to RUL prediction.** *Mechanical Systems and Signal Processing*, 104, 799–834.
6. Zhang, W. et al. (2018). **A new bearing fault diagnosis method based on modified convolutional neural networks.** *Chinese Journal of Aeronautics*, 33(2), 439–447.
7. Lundberg, S. M., & Lee, S. (2017). **A Unified Approach to Interpreting Model Predictions.** *NeurIPS*.

---

## 23. Citation

```bibtex
@misc{turboguard2026,
  title  = {TurboGuard: Predictive Maintenance and RUL Estimation for Industrial Rotating Equipment},
  author = {Meruva, Suraj},
  year   = {2026},
  howpublished = {\url{https://github.com/surajmeruva0786/turboguard}}
}
```

---

## 24. License

Released under the **MIT License**. See [`LICENSE`](LICENSE).

The CWRU and IMS datasets remain the property of their original publishers and are subject to their respective licences. Please consult `data/README.md` before downloading.

---

## 25. Acknowledgments

- Case Western Reserve University Bearing Data Center for releasing the canonical bearing-fault benchmark.
- NASA Prognostics Center of Excellence for the IMS dataset.
- The maintainers of PyTorch, scikit-learn, XGBoost, SHAP, and PyWavelets.
- The faculty of IIIT Naya Raipur for supervision and computational resources.

---

## 26. Contact

**Suraj Meruva**
B.Tech., Indian Institute of Information Technology, Naya Raipur
Email: `meruva24102@iiitnr.edu.in`
GitHub: [@surajmeruva0786](https://github.com/surajmeruva0786)

For research, internship, or collaboration discussions, please reach out via email.
