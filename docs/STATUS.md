# Session Status — where this build stands

Last updated: 2026-08-01, session 2, end of Phase M. This file is the
single source of truth for "what's done" and "what's next" — read this
before resuming.

## Progress: 105 / 120 roadmap steps committed (Phases A–M done, Phase N next)

All commits are on `main`, pushed to GitHub, one step at a time (see `git
log` or `docs/ROADMAP.md` for the full numbered list). Test suite: **141
passing, 0 failing** (`pytest -q` from repo root with `.venv` activated).

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

### Not started yet (Phase N → O)

Full detail in `docs/ROADMAP.md`. In order:

1. **Phase N — Deployment & CI**: Dockerfile, docker-compose,
   `.streamlit/config.toml`, full CI test matrix, `docs/DEPLOYMENT.md`,
   an end-to-end pipeline orchestration script.
2. **Phase O — Final polish**: update README's Results section with the
   real measured (synthetic-scale) numbers instead of the placeholder
   literature-style table, `CHANGELOG.md`, full lint pass, `v0.1.0` tag.

## How to resume

```bash
cd Z:\turboguard
source .venv/Scripts/activate   # already has all requirements.txt installed
pytest -q                        # should show 141 passed
```

Then continue at "Phase N" above — same pattern as every prior step:
implement, write tests against the committed synthetic data (not mocks),
run them, commit, push. Commit message convention: `feat(step-N/120): ...`
(see `git log` for exact numbering so far; next commit should be step 106).

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
