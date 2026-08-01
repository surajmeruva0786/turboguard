# Deployment

TurboGuard ships as a single Docker image that can serve the Streamlit
dashboard, run the full pipeline, or run the test suite, dispatched via
`scripts/entrypoint.sh`.

## Local (no Docker)

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash; use .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
pip install -e .

python scripts/run_full_pipeline.py   # generates data, trains models, runs evaluations
streamlit run app/dashboard.py         # http://localhost:8501
```

## Docker

```bash
docker build -t turboguard:latest .

# Dashboard (default CMD)
docker run --rm -p 8501:8501 turboguard:latest

# Full pipeline
docker run --rm -v "$(pwd)/runs:/app/runs" -v "$(pwd)/results:/app/results" \
    turboguard:latest pipeline --skip-synthetic

# Test suite
docker run --rm turboguard:latest test
```

The image's `HEALTHCHECK` polls Streamlit's built-in health endpoint
(`/_stcore/health`); `docker ps` reports `healthy`/`unhealthy` accordingly.

## Docker Compose

```bash
docker compose up --build
```

Mounts `data/`, `runs/`, and `results/` as volumes so pipeline outputs
persist across container restarts and are inspectable from the host.
Dashboard is served at `http://localhost:8501`.

## Configuration

- `.streamlit/config.toml` — server (headless, port 8501, address
  `0.0.0.0` so it's reachable from outside the container) and theme
  settings. `.streamlit/secrets.toml` is gitignored — put any secrets
  there for local dev; in production, prefer environment variables or your
  platform's secret manager and reference them via `st.secrets` or `os.environ`.
- `configs/data.yaml` — switch `source: synthetic` to `source: real` per
  dataset once real CWRU/IMS data has been downloaded
  (`scripts/download_cwru.py`, `scripts/download_ims.py`); no other
  pipeline code needs to change (see `docs/ROADMAP.md`'s data note).

## CI

`.github/workflows/ci.yml` runs on every push/PR to `main`:

1. **lint-and-test** — ruff lint + full pytest suite (matrix: Python 3.11
   and 3.12) with coverage.
2. **full-pipeline-smoke** — runs `scripts/run_full_pipeline.py` end to
   end against the committed synthetic data, verifying the whole chain
   (preprocessing → features → classical training → deep smoke training →
   RUL/cross-condition/cross-dataset evaluation → XAI report) still wires
   together, not just each unit in isolation.

## Cloud deployment notes

The dashboard is a stateless Streamlit app reading from `data/`, `runs/`,
and `results/` — any platform that can run a Docker container and mount
(or bake in) those directories works:

- **Streamlit Community Cloud**: point it at this repo; it installs
  `requirements.txt` and runs `app/dashboard.py` directly (no Docker
  needed there, but the committed synthetic data + `runs/random_forest_cwru/metrics.json`
  give it something to display immediately — a fresh `model.joblib` still
  needs `make train-classical` run once, since binaries are gitignored).
- **Any container platform** (Cloud Run, ECS, Azure Container Apps,
  Render, Fly.io, etc.): build the image from the `Dockerfile` above,
  expose port 8501, and mount/seed `runs/` and `results/` however that
  platform handles persistent storage.
