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
git clone https://github.com/<your-username>/TurboGuard.git
cd TurboGuard

# 2. Conda environment
conda create -n turboguard python=3.11 -y
conda activate turboguard

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download datasets
python scripts/download_cwru.py
python scripts/download_ims.py
```

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
    --input_dir data/raw/cwru \
    --output_dir data/processed/cwru \
    --sfreq 12000 \
    --window 1.0 \
    --overlap 0.5
```

### Train classical baseline

```bash
python -m src.training.train_classical \
    --model xgboost \
    --dataset cwru \
    --output_dir runs/xgb_cwru
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

### Evaluate cross-condition

```bash
python -m src.evaluation.cross_condition \
    --checkpoint runs/hybrid_multitask/best.ckpt \
    --train_loads 0 1 2 \
    --test_load 3 \
    --output_dir results/cross_condition
```

### Generate SHAP report

```bash
python -m src.xai.explain \
    --checkpoint runs/hybrid_multitask/best.ckpt \
    --asset_file data/processed/ims/asset_03.npz \
    --output_dir results/xai/asset_03
```

### Launch dashboard

```bash
streamlit run app/dashboard.py
```

---

## 17. Project Structure

```
TurboGuard/
├── app/
│   └── dashboard.py
├── configs/
│   ├── turboguard_cnn.yaml
│   ├── turboguard_hybrid.yaml
│   └── classical_baselines.yaml
├── data/
│   ├── raw/                       # gitignored
│   ├── processed/
│   └── README.md
├── notebooks/
│   ├── 01_dataset_eda.ipynb
│   ├── 02_envelope_analysis.ipynb
│   ├── 03_feature_importance.ipynb
│   ├── 04_rul_trajectories.ipynb
│   └── 05_xai_walkthrough.ipynb
├── runs/                          # gitignored
├── results/
├── scripts/
│   ├── download_cwru.py
│   ├── download_ims.py
│   └── verify_environment.py
├── src/
│   ├── data/
│   │   ├── cwru_loader.py
│   │   ├── ims_loader.py
│   │   └── dataset.py
│   ├── preprocessing/
│   │   └── run.py
│   ├── features/
│   │   ├── time_domain.py
│   │   ├── frequency_domain.py
│   │   ├── envelope.py
│   │   ├── wavelet.py
│   │   └── bearing_freqs.py
│   ├── models/
│   │   ├── classical.py
│   │   ├── turboguard_cnn.py
│   │   ├── turboguard_hybrid.py
│   │   └── autoencoder.py
│   ├── rul/
│   │   ├── direct_regression.py
│   │   ├── health_indicator.py
│   │   └── degradation_model.py
│   ├── training/
│   │   ├── train_classical.py
│   │   └── train_deep.py
│   ├── evaluation/
│   │   ├── classification.py
│   │   ├── rul_metrics.py
│   │   └── cross_condition.py
│   ├── xai/
│   │   ├── shap_tree.py
│   │   ├── shap_deep.py
│   │   └── bearing_freq_annotator.py
│   └── utils/
│       ├── seed.py
│       └── reports.py
├── tests/
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 18. Results

> *Numbers below are target benchmarks consistent with the published CWRU / IMS literature. Replace with your measured values after training.*

### Fault Classification on CWRU (within-condition, 5-fold CV)

| Model                  | Accuracy     | Macro-F1     |
|------------------------|--------------|--------------|
| Random Forest          | 0.96 ± 0.01  | 0.95 ± 0.02  |
| XGBoost                | 0.97 ± 0.01  | 0.96 ± 0.01  |
| TurboGuard-CNN         | 0.98 ± 0.01  | 0.98 ± 0.01  |
| **TurboGuard-Hybrid**  | **0.99 ± 0.005** | **0.99 ± 0.005** |

### Fault Classification (cross-condition: train on loads 0,1,2, test on 3)

| Model                  | Accuracy     | Macro-F1     |
|------------------------|--------------|--------------|
| Random Forest          | 0.85 ± 0.02  | 0.83 ± 0.03  |
| XGBoost                | 0.88 ± 0.02  | 0.86 ± 0.02  |
| TurboGuard-CNN         | 0.91 ± 0.02  | 0.90 ± 0.02  |
| **TurboGuard-Hybrid**  | **0.94 ± 0.01** | **0.93 ± 0.01** |

### RUL Estimation on IMS (Bearing 1)

| Model                  | RMSE (cycles) | MAPE       | PHM Score |
|------------------------|---------------|------------|-----------|
| Health Indicator only  | 38            | 22 %       | 410       |
| Direct regression      | 32            | 19 %       | 380       |
| **Combined (TurboGuard)** | **26**     | **15 %**   | **310**   |

### Confusion Matrix (Hybrid model, cross-condition)

```
                Pred_H   Pred_IR  Pred_OR  Pred_B   Pred_C
True_Healthy     0.97     0.01     0.01     0.00     0.01
True_InnerRace   0.01     0.94     0.02     0.02     0.01
True_OuterRace   0.01     0.02     0.95     0.01     0.01
True_Ball        0.01     0.03     0.02     0.92     0.02
True_Compound    0.01     0.02     0.02     0.03     0.92
```

---

## 19. Reproducibility

- Fixed seed (`42`) across NumPy, PyTorch, and Python's `random`.
- CUDA determinism enabled (`torch.use_deterministic_algorithms(True)`).
- Dependencies pinned in `requirements.txt`.
- All hyperparameters stored as YAML files and logged with each run.
- Pre-processed dataset checksums published with checkpoints.

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
  author = {<Your Name>},
  year   = {2026},
  howpublished = {\url{https://github.com/<your-username>/TurboGuard}}
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

**<Your Name>**
B.Tech., Indian Institute of Information Technology, Naya Raipur
Email: `<your.email@iiitnr.edu.in>`
GitHub: [@<your-username>](https://github.com/<your-username>)
LinkedIn: [linkedin.com/in/<your-username>](https://linkedin.com/in/<your-username>)

For research, internship, or collaboration discussions, please reach out via email.
