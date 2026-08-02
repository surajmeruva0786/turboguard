# Changelog

All notable changes to this project are documented in this file. Format
loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] — 2026-08-02

Downloaded the real CWRU and NASA IMS datasets and ran the full pipeline
against them end-to-end (README section 18 "Real-Data Results") — the one
item `docs/RELEASE_CHECKLIST.md` had explicitly left out of 0.1.0. No
pipeline code differs between synthetic and real data, only which config
is pointed at (new `configs/data_real.yaml`) — but getting there for real
surfaced and fixed several bugs the synthetic-only path never exercised.

### Added

- `configs/data_real.yaml`: real-data variant of `configs/data.yaml`
  (`configs/data.yaml` itself is untouched — CI and the default dashboard
  demo still run against synthetic data).
- `evaluate_rul.py --source real --test_set N`: RUL evaluation against a
  real extracted IMS test-set directory, not just the synthetic
  `labels.csv` layout.
- `src.preprocessing.run --dataset ims --source real --test_set N` and
  `load_ims_dataset(source="real")` with no `bearing_id`: load every
  bearing in a real IMS test-set directory at once (previously required
  one explicit `bearing_id` per call).
- `src/data/ims_loader.py`'s `IMS_REAL_FAULT_LABELS`: per-(test_set,
  bearing) fault labels sourced from the dataset's own bundled "Readme
  Document for IMS Bearing Data.pdf" (Qiu et al. 2006), so real IMS
  snapshots get a real `dominant_fault` instead of `"unknown"` — needed
  to make cross-dataset evaluation against real IMS meaningful at all.
- `py7zr` dependency + multi-layer archive extraction in
  `scripts/download_ims.py`: the real IMS archive is undocumented as
  zip → nested `.7z` → three `.rar` files (one of which internally
  unpacks to a stale `4th_test` folder name), not a flat zip of test-set
  directories as originally assumed.
- Retry-with-backoff and Content-Length verification in
  `scripts/download_cwru.py`: the CWRU server intermittently truncated
  responses mid-download across the full 161-file set (`requests.
  exceptions.ChunkedEncodingError`), and a previous run's partial file
  was being silently treated as "already downloaded."
- `resample_and_fix_length()` (`src/preprocessing/conditioning.py`):
  resamples + pads/truncates a signal to a fixed shape regardless of its
  native rate/duration, wired through `train_deep.py` and
  `evaluate_rul.py` so real CWRU's mixed 12/48 kHz files and real IMS's
  20 kHz snapshots can batch against CWRU's 12 kHz windows. Previously
  `configs/data.yaml`'s `window.target_sample_rate_hz` was defined but
  never actually passed to the dataset classes — dead config.

### Fixed

- `resample_and_fix_length()` didn't preserve input dtype (scipy's
  `resample_poly`/`sosfiltfilt` upcast to float64 internally), so a
  mixed-rate batch silently became a float64 tensor and crashed at the
  model's float32 conv layer.
- `src/evaluation/rul_metrics.rul_metrics()` had no NaN handling — one
  correctly-declined-to-extrapolate prediction (see
  `degradation_model.py`) poisoned every aggregate metric via
  `sqrt(mean(...NaN...))`, hiding all the other valid predictions. Now
  computes over finite predictions only and reports `n_valid`/
  `n_dropped_nonfinite`. Also clipped `phm_score`'s `exp()` argument,
  which could overflow to a literal `Infinity` (invalid JSON) on RUL
  scales larger than the tiny synthetic set.
- `evaluate_rul.py`'s `evaluate_bearing_pair` called `load_ims_dataset`
  without a `source` argument, so it silently always loaded synthetic
  data regardless of what the caller intended.

### Results

See README section 18 "Real-Data Results" for full numbers. Headline:
real CWRU classification and cross-condition generalisation both hold up
(0.999 / 0.968 accuracy); real cross-dataset domain shift is confirmed
severe (0.032, matching the synthetic run's 0.025); real RUL direct
regression is meaningful (RMSE 107.6 on a ~1000-snapshot scale) while the
health-indicator approach shows a genuine, honestly-reported failure mode
(saturates when calibrated on only 5 real healthy snapshots).

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
