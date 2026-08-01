# Session Status — where this build stands

Last updated: 2026-08-01, session 2, end of Phase I. This file is the
single source of truth for "what's done" and "what's next" — read this
before resuming.

## Progress: 80 / 120 roadmap steps committed (Phases A–I done, Phase J next)

All commits are on `main`, pushed to GitHub, one step at a time (see `git
log` or `docs/ROADMAP.md` for the full numbered list). Test suite: **119
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

### Not started yet (Phases J → O)

Full detail in `docs/ROADMAP.md`. In order:

1. **Phase J — Cross-condition/cross-dataset evaluation**: train on loads
   {0,1,2} test on {3}; train on CWRU test on IMS.
2. **Phase K — XAI**: SHAP tree/deep explainers, the bearing-frequency
   physical-justification annotator (builds on
   `src/features/bearing_freqs.closest_characteristic_frequency` and
   `src/features/envelope.dominant_envelope_peak`, both already done).
3. **Phase L — Notebooks** (EDA, envelope analysis, feature importance,
   RUL trajectories, XAI walkthrough).
4. **Phase M — Streamlit dashboard** (fleet view, drill-down, alert
   inbox, maintenance recommendations — logic already exists in
   `src/utils/reports.recommend_from_rul` — PDF export).
5. **Phase N — Deployment & CI**: Dockerfile, docker-compose,
   `.streamlit/config.toml`, full CI test matrix, `docs/DEPLOYMENT.md`,
   an end-to-end pipeline orchestration script.
6. **Phase O — Final polish**: update README's Results section with the
   real measured (synthetic-scale) numbers instead of the placeholder
   literature-style table, `CHANGELOG.md`, full lint pass, `v0.1.0` tag.

## How to resume

```bash
cd Z:\turboguard
source .venv/Scripts/activate   # already has all requirements.txt installed
pytest -q                        # should show 101 passed
```

Then continue at "Phase I" above — same pattern as every prior step:
implement, write tests against the committed synthetic data (not mocks),
run them, commit, push. Commit message convention: `feat(step-N/120): ...`
(see `git log` for exact numbering so far; next commit should be step 73).

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
