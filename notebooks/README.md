# Notebooks

Exploratory notebooks accompanying the TurboGuard pipeline. All of them run
against the committed synthetic datasets and results — no external
downloads required. They assume the repo's other pipeline steps have
already been run at least once (`data/processed/*`, `runs/random_forest_cwru/`,
`results/rul_ims/`, `results/xai/`) so the artifacts they read exist; each
notebook notes how to regenerate its inputs if missing.

Jupyter itself isn't in `requirements.txt` (the deployable app doesn't
need it) — install it separately: `pip install jupyter`.

| Notebook | What it shows |
|---|---|
| [`01_dataset_eda.ipynb`](01_dataset_eda.ipynb) | Basic exploratory look at the synthetic CWRU-like and IMS-like datasets — per-class sample counts, example waveforms, IMS run-to-failure health-indicator trajectories. |
| [`02_envelope_analysis.ipynb`](02_envelope_analysis.ipynb) | The classical envelope-spectrum diagnostic method (README 8.3): band-pass → Hilbert envelope → FFT, with BPFO/BPFI/BSF/FTF markers overlaid on an outer-race fault window. |
| [`03_feature_importance.ipynb`](03_feature_importance.ipynb) | SHAP TreeExplainer feature importance for the Random-Forest baseline — top-10 contributing engineered features for the outer-race fault class. |
| [`04_rul_trajectories.ipynb`](04_rul_trajectories.ipynb) | Health-indicator and true-RUL trajectories for both synthetic IMS bearings, plus the evaluated direct/health-indicator/combined RUL metrics from `results/rul_ims/`. |
| [`05_xai_walkthrough.ipynb`](05_xai_walkthrough.ipynb) | End-to-end walkthrough of `src.xai.explain`: predict one sample, explain it with SHAP, and annotate it with the closest bearing characteristic frequency. |

## Regenerating inputs

```bash
python -m src.preprocessing.run --dataset cwru --input_dir data/raw/cwru --output_dir data/processed/cwru
python -m src.features.extract --dataset cwru --input_dir data/processed/cwru --output_dir data/processed/cwru
python -m src.training.train_classical --model random_forest --dataset cwru --output_dir runs/random_forest_cwru
python -m src.evaluation.evaluate_rul --output_dir results/rul_ims
python -m src.xai.explain --model_dir runs/random_forest_cwru --processed_dir data/processed/cwru --sample_idx 0 --output_dir results/xai/sample_0
```
