# Changelog

All notable changes to this project are documented in this file. Format
loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] — 2026-08-01

Initial release. All 120 roadmap steps complete (`docs/ROADMAP.md`), 141
tests passing, 76% coverage. End-to-end pipeline from raw vibration
signals to fault classification, RUL estimation, SHAP explanations, and a
Streamlit triage dashboard — validated against a committed physics-based
synthetic dataset (real CWRU/IMS downloaders included but not required).

### Added

- **Scaffolding**: packaging (`pyproject.toml`, `requirements.txt`), CI
  skeleton, `Makefile`, MIT license, contributing guide.
- **Data**: real CWRU/IMS downloaders (`scripts/download_*.py`) and a
  physics-based synthetic generator (`scripts/generate_synthetic.py`)
  producing a small committed dataset for tests/CI/demo; loaders for both
  sources behind one interface (`src/data/`).
- **Signal processing**: conditioning, windowing, and a CLI
  (`src/preprocessing/`).
- **Feature engineering**: time-domain, frequency-domain (with
  physics-informed BPFO/BPFI/BSF/FTF band energies), envelope-spectrum,
  and wavelet-packet features combined into a 176-dim vector
  (`src/features/`).
- **Models**: classical baselines (Random Forest, XGBoost, SVM, Logistic
  Regression), TurboGuard-CNN, TurboGuard-Hybrid (CNN + Bi-LSTM
  multi-task), and a feature-space autoencoder for health-indicator RUL
  (`src/models/`).
- **Training**: classical CV training CLI, deep-model training CLI
  (AdamW + cosine warmup + masked multi-task CE/Huber loss), waveform
  augmentation (`src/training/`).
- **RUL estimation**: direct regression, health-indicator +
  exponential-degradation extrapolation, and a learned-weight ensemble
  (`src/rul/`).
- **Evaluation**: classification metrics, RUL metrics (RMSE/MAPE/PHM
  score), cross-condition and cross-dataset generalisation evaluation
  (`src/evaluation/`).
- **Explainability**: SHAP TreeExplainer (classical models) and
  GradientExplainer (via a feature-view proxy MLP), plus a
  bearing-frequency physical-justification annotator (`src/xai/`).
- **Notebooks**: EDA, envelope analysis, feature importance, RUL
  trajectories, XAI walkthrough (`notebooks/`).
- **Dashboard**: Streamlit fleet view, asset drill-down, alert inbox, and
  PDF maintenance-report export (`app/`).
- **Deployment**: `Dockerfile`, `docker-compose.yml`, GitHub Actions CI
  (lint + test matrix on Python 3.11/3.12, plus an end-to-end
  full-pipeline smoke job), `scripts/run_full_pipeline.py` orchestration,
  `docs/DEPLOYMENT.md`.

### Fixed

- Repo-wide `ruff` import-sort violations that had gone uncaught because
  CI's lint step had been failing since the very first commit.
- `tests/test_dashboard_utils.py` / `test_dashboard_smoke.py` depended on
  gitignored, regenerable artifacts (`runs/random_forest_cwru/
  model.joblib`, `data/processed/*/features.parquet`) that only existed
  locally from earlier manual pipeline runs — a genuinely fresh checkout
  had no such files. Added a session-scoped autouse pytest fixture
  (`tests/conftest.py`) that generates them if missing.
- `src/rul/degradation_model.fit_exponential_degradation` used
  unconstrained nonlinear least squares (`scipy.optimize.curve_fit`),
  which diverged to absurd parameter values on a handful of noisy
  trajectory points; replaced with a log-linear fit and clamped growth
  rate.
- `src/rul/health_indicator.HealthIndicatorModel.compute` could return
  unbounded reconstruction error on severely out-of-distribution inputs
  from a tiny healthy-only training set; output is now clipped.

### Known limitations

- All measured results are on a small, cleanly-separable synthetic
  dataset — see README section 18 for exact numbers and caveats before
  treating anything here as a real-world benchmark.
- Deep model (CNN/Hybrid) classification metrics were not measured
  end-to-end (only smoke-trained for 3 epochs, no held-out split); a full
  120-epoch run needs no code changes but wasn't run at synthetic scale.
- Real CWRU/IMS datasets were not downloaded/run in this environment;
  downloaders are verified reachable but the full real-data pipeline is
  unexercised.
