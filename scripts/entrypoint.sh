#!/usr/bin/env sh
# Docker entrypoint (README/DEPLOYMENT.md, roadmap step 112). Dispatches on
# the first argument so the same image serves the dashboard, the full
# pipeline, or an ad-hoc command without needing separate images.
set -e

case "$1" in
    dashboard)
        exec streamlit run app/dashboard.py
        ;;
    pipeline)
        exec python scripts/run_full_pipeline.py "${@:2}"
        ;;
    test)
        exec pytest -q
        ;;
    *)
        exec "$@"
        ;;
esac
