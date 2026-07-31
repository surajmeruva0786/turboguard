# Session Status — where this build stands

Last updated: 2026-08-01, end of session 1. This file is the single source
of truth for "what's done" and "what's next" — read this before resuming.

## Progress: 37 / 120 roadmap steps committed (Phases A–G done, Phase H started)

All commits are on `main`, pushed to GitHub, one step at a time (see `git
log` or `docs/ROADMAP.md` for the full numbered list). Test suite: **85
passing, 0 failing** (`pytest -q` from repo root with `.venv` activated).

### Done

- **Phase A — Scaffolding**: `.gitignore`, `LICENSE`, `requirements.txt`,
  `pyproject.toml`, `docs/ROADMAP.md`, `CONTRIBUTING.md`, CI skeleton,
  `Makefile`, `src/` package tree, `configs/base.yaml`.
- **Phase B — Utils**: seeding, logging, YAML/JSON io, maintenance-
  recommendation logic. Fully tested.
- **Phase C — Synthetic data**: `scripts/generate_synthetic.py` (physics-
  based CWRU-like + IMS-like generator, verified against real CWRU BPFO
  values), real `download_cwru.py` (live-scrapes the real CWRU Bearing
  Data Center — verified against the live site, found all 161 real files)
  and `download_ims.py` (real NASA S3 archive URL, verified reachable),
  `verify_environment.py`. Synthetic datasets committed under
  `data/raw/{cwru,ims}/synthetic/`.
- **Phase D — Data loaders**: `src/data/cwru_loader.py`, `ims_loader.py`
  (both support real *and* synthetic sources), `src/data/dataset.py`
  (PyTorch `Dataset`s: single-window and K-sequence, with masked
  multi-task collation for mixed fault/RUL batches).
- **Phase E — Preprocessing**: `conditioning.py` (DC removal, filters,
  resample), `windowing.py`, CLI `src/preprocessing/run.py`.
- **Phase F — Feature engineering**: time-domain, frequency-domain
  (incl. physics-informed BPFO/BPFI/BSF/FTF band energies), envelope
  spectrum, wavelet packet (db4, level 4) — combined into a **176-dim**
  feature vector (README targets ~180) via `src/features/feature_vector.py`,
  plus CLI `src/features/extract.py`.
- **Phase G — Classical ML**: `ClassicalFaultClassifier` (RF/XGBoost/
  SVM/LogReg behind one interface), evaluation metrics
  (`src/evaluation/classification.py`), CLI `src/training/train_classical.py`.
  All 4 models trained on the synthetic CWRU features and committed under
  `runs/*_cwru/{metrics.json,config.yaml}` (model binaries gitignored,
  regenerable via `make train-classical`). **Note**: all 4 hit 1.000 CV
  accuracy — expected on this tiny (4 samples/class), cleanly-separable
  synthetic set; this is a pipeline smoke test, not a benchmark claim.
- **Phase H — started**: `src/models/turboguard_cnn.py` done
  (`CNNEncoder` + `TurboGuardCNN`, ~500k params, forward+backward
  verified). **This is the last thing committed.**

### Not started yet (Phases H tail → O)

Full detail in `docs/ROADMAP.md`. In order:

1. **Finish Phase H**: `src/models/turboguard_hybrid.py` (CNN+Bi-LSTM
   multi-task: fault head mean-pooled over time, RUL head from last
   timestep — reuse `CNNEncoder` from `turboguard_cnn.py` as the
   time-distributed encoder), `src/models/autoencoder.py` (feature-space
   autoencoder for the health-indicator RUL approach), configs
   (`turboguard_cnn.yaml`, `turboguard_hybrid.yaml`), `src/training/
   train_deep.py` (AdamW, cosine schedule + warmup, AMP, multi-task
   Huber+CE loss), `src/training/augmentation.py` (time shift, amplitude
   scaling, Gaussian noise), then actually run smoke-scale training runs
   on the synthetic data and commit their metrics.
2. **Phase I — RUL estimation**: direct regression, health-indicator +
   degradation-model approach (uses the autoencoder from Phase H),
   combined ensemble, RUL metrics (RMSE/MAPE/PHM score), evaluate on
   synthetic IMS.
3. **Phase J — Cross-condition/cross-dataset evaluation**.
4. **Phase K — XAI**: SHAP tree/deep explainers, the bearing-frequency
   physical-justification annotator (builds directly on
   `src/features/bearing_freqs.closest_characteristic_frequency` and
   `src/features/envelope.dominant_envelope_peak`, both already done).
5. **Phase L — Notebooks** (EDA, envelope analysis, feature importance,
   RUL trajectories, XAI walkthrough).
6. **Phase M — Streamlit dashboard** (fleet view, drill-down, alert
   inbox, maintenance recommendations — logic already exists in
   `src/utils/reports.recommend_from_rul` — PDF export).
7. **Phase N — Deployment & CI**: Dockerfile, docker-compose,
   `.streamlit/config.toml`, full CI test matrix, `docs/DEPLOYMENT.md`,
   an end-to-end pipeline orchestration script.
8. **Phase O — Final polish**: update README's Results section with the
   real measured (synthetic-scale) numbers instead of the placeholder
   literature-style table, `CHANGELOG.md`, full lint pass, `v0.1.0` tag.

## How to resume

```bash
cd Z:\turboguard
source .venv/Scripts/activate   # already has all requirements.txt installed
pytest -q                        # should show 85 passed
```

Then continue at "Finish Phase H" above — same pattern as every prior
step: implement, write tests against the committed synthetic data (not
mocks), run them, commit, push. Commit message convention:
`feat(step-N/120): ...` (see `git log` for exact numbering so far; next
commit should be step 38).

## Key design decisions worth knowing before continuing

- **Synthetic vs real data**: everything runs against a small physics-based
  synthetic dataset committed to the repo (see `data/README.md`). Real
  CWRU/IMS downloaders are real and verified but not run (real CWRU ≈500MB+,
  real IMS ≈1GB) — swapping in real data later needs zero code changes,
  only `source: real` in `configs/data.yaml` / `--source real` on the CLIs.
- **Window shape must match across CWRU and IMS**: both are 1s @ 12kHz
  (12000 samples) so the hybrid model's shared encoder can batch sequences
  from either dataset together (see the step-23 commit — this was a real
  bug caught by a test, not a hypothetical).
- **gitignore gotcha already fixed**: to keep small files (`metrics.json`,
  `config.yaml`) inside an otherwise-ignored directory (`runs/*`), the
  directory itself must also be un-ignored (`!runs/*/`) — a plain
  `!runs/**/metrics.json` negation is silently ineffective otherwise (see
  step-36 commit).
