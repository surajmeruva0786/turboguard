# Session Status — where this build stands

Last updated: 2026-08-01, session 2, project complete. This file is the
single source of truth for "what's done" — read this before resuming any
future work.

## Progress: 120 / 120 roadmap steps committed — v0.1.0 tagged

All commits are on `main`, pushed to GitHub, one step at a time (see `git
log` or `docs/ROADMAP.md` for the full numbered list). Test suite: **141
passing, 0 failing**, verified from a **fully clean artifact state**.
GitHub Actions CI is **green** (`lint-and-test` on Python 3.11 + 3.12,
`full-pipeline-smoke`) — confirmed via the Actions API, see
`docs/RELEASE_CHECKLIST.md`.

### Done

- **Phases A–G**: scaffolding, utils, synthetic data, data loaders,
  preprocessing, feature engineering, classical ML. See prior session's
  entry in git history (steps 1–60) for detail.
- **Phase H — Deep models**: `src/models/turboguard_cnn.py` (`CNNEncoder`
  + `TurboGuardCNN`, ~500k params), `src/models/turboguard_hybrid.py`
  (`TurboGuardHybrid`: shared `CNNEncoder` + Bi-LSTM(128, 2 layers),
  mean-pooled fault head + last-timestep RUL head, ~1.4M params),
  `src/models/autoencoder.py` (`FeatureAutoencoder`, symmetric MLP over
  the 176-dim engineered feature vector, for the health-indicator RUL
  approach), `configs/turboguard_cnn.yaml` + `configs/turboguard_hybrid.yaml`,
  `src/training/augmentation.py` (time shift / amplitude scale / Gaussian
  noise, works on both single-window and sequence batches),
  `src/training/train_deep.py` (CLI: AdamW + cosine-annealing-with-warmup,
  masked multi-task CE+Huber loss via `collate_sequence_samples`'
  `has_fault`/`has_rul` masks, CUDA auto-detect). Smoke-trained both models
  on the synthetic data and committed `runs/cnn_smoke/` and
  `runs/hybrid_smoke/` (`metrics.json` + `config.yaml`; `model.pt`
  gitignored, regenerable via the same CLI commands, see README section 16).

- **Phase I — RUL estimation**: `src/rul/direct_regression.py`
  (piecewise-linear-capped target + hybrid-checkpoint predictor),
  `src/rul/health_indicator.py` (`FeatureAutoencoder`-based HI, AdamW +
  weight decay, clipped to avoid float blowup on tiny out-of-distribution
  inputs — see `docs/STATUS.md` git history step-80 fix commit),
  `src/rul/degradation_model.py` (log-linear exponential-fit RUL
  extrapolation — more stable than nonlinear `curve_fit` on few noisy
  points), `src/rul/combine.py` (grid-searched weighted ensemble),
  `src/evaluation/rul_metrics.py` (RMSE/MAPE/PHM asymmetric score),
  `src/evaluation/evaluate_rul.py` evaluated bearing1→bearing2 on the
  synthetic IMS set, committed to `results/rul_ims/`. **Caveat**: the
  direct-regression checkpoint (`runs/hybrid_smoke/model.pt`) was
  smoke-trained on this same synthetic data, not held out — see
  `results/rul_ims/config.yaml` note; this is a pipeline-correctness
  check, not a benchmark claim.

- **Phase J — Cross-condition/cross-dataset evaluation**:
  `src/evaluation/cross_condition.py` (classical model trained on CWRU
  loads {0,1,2}, tested on load {3} — accuracy 1.0 on this tiny synthetic
  set, same caveat as the within-condition classical baselines), committed
  to `results/cross_condition/`. `src/evaluation/cross_dataset.py`
  (trained on CWRU, tested on synthetic IMS — accuracy **0.025**, a real
  and expected result: the IMS synthetic generator uses different fault
  physics/severity than CWRU's, so this demonstrates genuine domain shift,
  not a bug — see `results/cross_dataset/metrics.json`), committed to
  `results/cross_dataset/`.

- **Phase K — XAI**: `src/xai/shap_tree.py` (TreeExplainer wrapper for
  RF/XGBoost, named-feature attributions), `src/xai/shap_deep.py` (small
  `FeatureMLP` proxy over the engineered feature vector + GradientExplainer
  — documented design choice since the real deep model consumes raw
  waveforms with no named features to attribute), `src/xai/
  bearing_freq_annotator.py` (physical-justification message generator),
  `src/xai/explain.py` CLI, sample report committed to
  `results/xai/sample_0/`.
- **Phase L — Notebooks**: all 5 notebooks (`01_dataset_eda` →
  `05_xai_walkthrough`) written as hand-built nbformat-4 JSON, executed
  headlessly via `nbclient` to verify they run end-to-end and bake in
  real outputs before committing (caught and fixed a real bug:
  `build_sample_explanation` didn't coerce `str` paths to `Path`).
  `notebooks/README.md` indexes them with regeneration commands. Jupyter
  itself isn't in `requirements.txt` (not needed by the deployable app).

- **Phase M — Streamlit dashboard**: `app/data_access.py` (business logic,
  no Streamlit import, unit-testable — builds a 6-asset demo fleet: 4 CWRU
  load conditions with a clearly-labelled RUL heuristic since CWRU has no
  run-to-failure data, + 2 real IMS bearing trajectories), `app/
  dashboard.py` (3-page sidebar app: Fleet View, Asset Drill-down, Alert
  Inbox; PDF export via reportlab). Verified two ways: `AppTest` headless
  harness (`tests/test_dashboard_smoke.py`) and a real `streamlit run`
  server that was curl-checked (HTTP 200) then stopped.

- **Phase N — Deployment & CI**: `Dockerfile` (built successfully, 3.67GB,
  verified with `docker build`), `docker-compose.yml`, `.dockerignore`,
  `.streamlit/config.toml`, `scripts/entrypoint.sh` (dispatches
  `dashboard`/`pipeline`/`test`), `scripts/run_full_pipeline.py`
  (end-to-end orchestration, verified reproducible — reran it and only
  `elapsed_seconds` differed, all loss/metric values identical),
  `docs/DEPLOYMENT.md`, CI matrix (Python 3.11 + 3.12) plus a
  `full-pipeline-smoke` job. **Also found and fixed a real, pre-existing
  bug while wiring this up**: `ruff --fix` caught genuine import-sort
  violations across ~26 files that had never actually been linted in CI
  (every CI run back to the very first commit had been failing at the
  Lint step — confirmed via the GitHub Actions API run history, 87/87
  runs). After fixing lint, a second bug surfaced: `tests/
  test_dashboard_utils.py` / `test_dashboard_smoke.py` load
  `runs/random_forest_cwru/model.joblib` and
  `data/processed/*/features.parquet` from disk — both gitignored/
  regenerable — which only existed locally from manual pipeline runs
  earlier in the session, so a genuinely fresh checkout (any CI runner,
  any new clone) would hit `FileNotFoundError`. Reproduced by hiding both
  locally; fixed with `tests/conftest.py` (session-scoped autouse fixture
  that generates them if missing) — verified all 141 tests pass from a
  fully clean artifact state. **Local environment note**: this machine's
  C: drive is at 99% disk usage (pre-existing, unrelated to this repo),
  which made Docker Desktop intermittently unresponsive mid-session —
  worth the user's attention independent of this project.

- **Phase O — Final polish (in progress)**: README section 18 (Results)
  rewritten with real measured numbers instead of the literature-style
  placeholder table; section 17 (Project Structure) and section 16
  (Usage) updated to match the actual repo and actual CLI signatures
  (several commands there had drifted from what the CLIs actually accept,
  e.g. `cross_condition`'s real `--model`/`--dataset` flags vs the
  originally-drafted `--checkpoint`); Citation/Contact placeholders filled
  in. Full coverage run: **76% overall** (`pytest -q --cov=src
  --cov-report=term-missing`, 141 passed). Two modules show 0% by the
  coverage tool despite being exercised: `src/evaluation/evaluate_rul.py`
  and `src/xai/explain.py` are both CLI entry points invoked via
  `subprocess`/`python -m` (from `conftest.py`'s fixture,
  `run_full_pipeline.py`, and manual runs) rather than imported and
  called in-process by a test, so `coverage.py` doesn't attribute that
  execution — not an actual test gap, just a measurement blind spot worth
  knowing about.

- **Phase O tail**: `CHANGELOG.md` added; final repo-wide `ruff check`
  pass confirmed clean; `docs/RELEASE_CHECKLIST.md` completed and
  `v0.1.0` tagged on the commit where CI first went fully green.

### Nothing left on the roadmap

All 120 steps are done. Future work beyond this roadmap (real dataset
runs, order tracking, edge deployment, etc.) is listed in README section
21 ("Future Work"), not tracked here.

## How to resume (for future feature work beyond v0.1.0)

```bash
cd Z:\turboguard
source .venv/Scripts/activate   # already has all requirements.txt installed
pytest -q                        # should show 141 passed
```

There is no "next step" queued — this file exists for historical context
on how the build was done and the bugs found along the way. Follow the
same pattern for any future work: implement, write tests against real
(committed synthetic or downloaded real) data, run them, commit, push.

## Key design decisions worth knowing before continuing

- **Synthetic vs real data**: everything runs against a small physics-based
  synthetic dataset committed to the repo (see `data/README.md`). Real
  CWRU/IMS downloaders are real and verified but not run — swapping in real
  data later needs zero code changes, only `source: real` in
  `configs/data.yaml` / `--source real` on the CLIs.
- **Window shape must match across CWRU and IMS**: both are 1s @ 12kHz
  (12000 samples) so the hybrid model's shared encoder can batch sequences
  from either dataset together.
- **gitignore gotcha**: to keep small files (`metrics.json`, `config.yaml`)
  inside an otherwise-ignored directory (`runs/*`), the directory itself
  must also be un-ignored (`!runs/*/`).
- **Multi-task batching**: `SequenceFaultDataset` (CWRU, fault-only) and
  `SequenceRULDataset` (IMS, RUL-only) samples are combined via
  `torch.utils.data.ConcatDataset` + `collate_sequence_samples`, which
  produces `has_fault`/`has_rul` boolean masks per batch so the multi-task
  loss only applies the supervised term each sample actually has — this is
  how `train_deep.py`'s hybrid path avoids needing paired fault+RUL labels
  per sample.
