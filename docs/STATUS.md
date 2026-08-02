# Session Status — where this build stands

Last updated: 2026-08-02, session 3 (post-v0.1.0 real-data run). This file
is the single source of truth for "what's done" — read this before
resuming any future work.

## Progress: 120 / 120 roadmap steps committed — v0.1.0 tagged; real-data run complete — v0.2.0

Session 3 downloaded the real CWRU + NASA IMS datasets and ran the full
pipeline against them (README section 18 "Real-Data Results",
`CHANGELOG.md` [0.2.0]) — see "Session 3" below for the detailed account.
The rest of this file (sessions 1-2) is unchanged history.

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
21 ("Future Work"), not tracked here. (Update, session 3: the real
dataset run item is now done — see below.)

## Session 3 (2026-08-02): real CWRU + IMS dataset run

Downloaded both real datasets and ran the full pipeline against them.
Full numbers: README section 18 "Real-Data Results". Full bug list:
`CHANGELOG.md` [0.2.0]. Summary of what happened, in order:

1. **Downloads weren't as documented.** `scripts/download_cwru.py`
   (161 real `.mat` files) hit `requests.exceptions.ChunkedEncodingError`
   twice mid-run — the CWRU server intermittently truncates responses
   under sustained sequential download. Added retry-with-backoff +
   Content-Length verification (a prior run's partial file was being
   silently treated as complete). `scripts/download_ims.py`'s single
   "~1GB zip" turned out to be zip → nested `IMS.7z` → three `.rar`
   files, one of which (`3rd_test.rar`) internally unpacks to a
   stale-named `4th_test/txt/` folder instead of `3rd_test/`. Added
   `py7zr` + WinRAR/`unrar` shellout + folder normalization to handle
   all three layers automatically.
2. **Read the dataset's own bundled Readme PDF** ("Readme Document for
   IMS Bearing Data.pdf", inside the IMS archive) for authoritative
   per-bearing fault ground truth, since the loader previously hardcoded
   `dominant_fault="unknown"` for all real IMS data: Set 1 bearing 3 =
   inner race, bearing 4 = ball; Set 2 bearing 1 = outer race; Set 3
   bearing 3 = outer race. Recorded in `IMS_REAL_FAULT_LABELS`
   (`src/data/ims_loader.py`). Without this, cross-dataset evaluation
   against real IMS would have been meaningless (single unlabelled
   class).
3. **Real data broke two assumptions the synthetic set made true by
   construction.** (a) `configs/data.yaml`'s `window.target_sample_rate_hz`
   was defined but never actually wired to the dataset classes — real
   CWRU mixes 12kHz and 48kHz files, real IMS is 20kHz, so batching them
   against CWRU's 12kHz windows crashed. Added and wired
   `resample_and_fix_length()` (`src/preprocessing/conditioning.py`).
   (b) That function's resampling upcast to float64 (scipy internals),
   which broke mixed-dtype batches at the model's float32 conv layer —
   fixed by preserving input dtype explicitly.
4. **RUL metrics went silently NaN on real data.** Fitting the
   health-indicator autoencoder on only 5 real (noisier, larger-scale)
   healthy snapshots saturates it almost immediately against the real
   failing bearing, so `src.rul.degradation_model` correctly declines to
   extrapolate on 205/975 windows (returns NaN rather than guessing) —
   but `rul_metrics()` had no NaN handling, so one NaN prediction turned
   every aggregate metric NaN, silently hiding the other 770 valid ones.
   Fixed to compute over finite predictions only, reporting
   `n_valid`/`n_dropped_nonfinite`. This is a genuine negative result
   (see README), not a bug being smoothed over — the direct-regression
   RUL model (trained on this same real data, `runs/hybrid_real`) does
   work, RMSE 107.6 on a ~1000-snapshot scale.
5. **Ran the full pipeline on real data**: classical CV (0.999 acc,
   `runs/random_forest_cwru_real`), cross-condition (0.968 acc,
   `results/cross_condition_real`), cross-dataset (0.032 acc, confirming
   the synthetic run's 0.025 domain-shift finding on real hardware,
   `results/cross_dataset_real`), CNN (40 epochs, `runs/cnn_real`),
   Hybrid (20 epochs, `runs/hybrid_real`), RUL (`results/rul_ims_real`),
   XAI (`results/xai/sample_276_real_inner_race` — top SHAP feature is
   `z_env_BPFI_h1_amp`, physically correct for an inner-race fault).
6. **Scoping decision**: real IMS work used `2nd_test` only (984
   snapshots × 4 bearings, one documented failure) rather than all three
   test sets — smallest, most tractable on CPU-only hardware, and
   `1st_test`/`3rd_test` remain downloaded and usable with the same
   `--test_set {1,3}` flags for anyone who wants to extend this.

Full test suite (144 tests, up from 141 — 3 new tests covering the
above fixes) and `ruff check` both clean after every change; see git log
for the exact commit-by-commit sequence (each fix/feature/result batch is
its own commit, pattern unchanged from sessions 1-2).

## How to resume (for future feature work beyond v0.1.0/v0.2.0)

```bash
cd Z:\turboguard
source .venv/Scripts/activate   # already has all requirements.txt installed
pytest -q                        # should show 144 passed
```

There is no "next step" queued — this file exists for historical context
on how the build was done and the bugs found along the way. Follow the
same pattern for any future work: implement, write tests against real
(committed synthetic or downloaded real) data, run them, commit, push.

## Key design decisions worth knowing before continuing

- **Synthetic vs real data**: CI, tests, and the default dashboard demo
  still run against the small physics-based synthetic dataset committed to
  the repo (see `data/README.md`). The real CWRU/IMS datasets have since
  been downloaded and run end-to-end (session 3, below;
  `configs/data_real.yaml`, `--source real` on the CLIs) — real data itself
  is gitignored (large, license-gated) so it isn't in the repo, only the
  resulting `results/*_real`/`runs/*_real` metrics are.
- **Window shape must match across CWRU and IMS**: both are 1s @ 12kHz
  (12000 samples) so the hybrid model's shared encoder can batch sequences
  from either dataset together. Enforced by
  `src.preprocessing.conditioning.resample_and_fix_length` (added in
  session 3) rather than just true-by-construction of the synthetic
  generator — real CWRU has mixed 12/48kHz files and real IMS is 20kHz.
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
