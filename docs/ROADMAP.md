# TurboGuard Build Roadmap

This document tracks the ~120-step build-out of TurboGuard from an empty
repository to a deployable, end-to-end predictive-maintenance system, as
described in the main [README](../README.md). Each step below corresponds to
one commit on `main`.

**Data note**: real CWRU / IMS datasets are large, license-gated downloads.
To make the full pipeline runnable and testable without them, a
physics-based synthetic vibration-signal generator
(`scripts/generate_synthetic.py`) produces small CWRU-like and IMS-like
datasets that are committed to the repo and used for CI, tests, and the
default dashboard demo. Real downloader scripts (`scripts/download_cwru.py`,
`scripts/download_ims.py`) are included and use the identical downstream
pipeline — swapping in real data requires no code changes, only pointing
`--input_dir` at the real data.

**Live progress**: see [`docs/STATUS.md`](STATUS.md) for exactly what's
done, what's next, and how to resume a session. Short version as of
2026-08-01: Phases A–H complete (101 tests passing), Phase I next.

## Phase A — Project scaffolding (1–10) ✅ done
1. `.gitignore`
2. `LICENSE` (MIT)
3. `requirements.txt`
4. `pyproject.toml`
5. `docs/ROADMAP.md` (this file)
6. `CONTRIBUTING.md`
7. `.github/workflows/ci.yml` (skeleton)
8. `Makefile`
9. `src/` package scaffolding (`__init__.py` files)
10. `configs/base.yaml`

## Phase B — Core utilities (11–15) ✅ done
11. `src/utils/seed.py`
12. `src/utils/logging_config.py`
13. `src/utils/reports.py`
14. `src/utils/io.py`
15. `tests/test_utils.py`

## Phase C — Bearing physics & synthetic data (16–25) ✅ done
16. `src/features/bearing_freqs.py`
17. `tests/test_bearing_freqs.py`
18. `scripts/generate_synthetic.py` (CWRU-like synthetic generator)
19. Synthetic IMS run-to-failure generator (extends step 18)
20. `data/README.md`
21. `scripts/download_cwru.py`
22. `scripts/download_ims.py`
23. `scripts/verify_environment.py`
24. Generate + commit synthetic CWRU dataset
25. Generate + commit synthetic IMS dataset

## Phase D — Data loaders (26–32) ✅ done
26. `src/data/cwru_loader.py`
27. `src/data/ims_loader.py`
28. `src/data/dataset.py` (PyTorch `Dataset`s)
29. `tests/test_cwru_loader.py`
30. `tests/test_ims_loader.py`
31. `tests/test_dataset.py`
32. `configs/data.yaml`

## Phase E — Signal preprocessing (33–40) ✅ done
33. `src/preprocessing/conditioning.py`
34. `src/preprocessing/windowing.py`
35. `src/preprocessing/run.py` (CLI)
36. `tests/test_preprocessing.py`
37. `configs/preprocessing.yaml`
38. Run preprocessing on synthetic CWRU
39. Run preprocessing on synthetic IMS
40. Validate processed outputs

## Phase F — Feature engineering (41–52) ✅ done
41. `src/features/time_domain.py`
42. `src/features/frequency_domain.py`
43. `src/features/envelope.py`
44. `src/features/wavelet.py`
45. `src/features/feature_vector.py`
46. `tests/test_time_domain.py`
47. `tests/test_frequency_domain.py`
48. `tests/test_envelope.py`
49. `tests/test_wavelet.py`
50. `tests/test_feature_vector.py`
51. `src/features/extract.py` + run extraction on processed data
52. Feature extraction validation

## Phase G — Classical ML models (53–60) ✅ done
53. `src/models/classical.py`
54. `src/training/train_classical.py`
55. `configs/classical_baselines.yaml`
56. `tests/test_classical.py`
57. Train classical baselines on synthetic CWRU
58. `src/evaluation/classification.py`
59. `tests/test_evaluation_classification.py`
60. Evaluate classical baselines, save results

## Phase H — Deep models (61–72) ✅ done
61. `src/models/turboguard_cnn.py`
62. `src/models/turboguard_hybrid.py`
63. `src/models/autoencoder.py`
64. `configs/turboguard_cnn.yaml`
65. `configs/turboguard_hybrid.yaml`
66. `tests/test_models.py`
67. `src/training/train_deep.py`
68. `src/training/augmentation.py`
69. `tests/test_augmentation.py`
70. Train CNN smoke run on synthetic CWRU
71. Train hybrid multitask smoke run
72. `tests/test_train_deep_smoke.py`

## Phase I — RUL estimation (73–80)
73. `src/rul/direct_regression.py`
74. `src/rul/health_indicator.py`
75. `src/rul/degradation_model.py`
76. `src/rul/combine.py`
77. `tests/test_rul.py`
78. `src/evaluation/rul_metrics.py`
79. `tests/test_rul_metrics.py`
80. Evaluate RUL on synthetic IMS

## Phase J — Cross-condition / cross-dataset evaluation (81–85)
81. `src/evaluation/cross_condition.py`
82. `tests/test_cross_condition.py`
83. Run cross-condition evaluation
84. `src/evaluation/cross_dataset.py`
85. Run cross-dataset evaluation

## Phase K — Explainability / XAI (86–92)
86. `src/xai/shap_tree.py`
87. `src/xai/shap_deep.py`
88. `src/xai/bearing_freq_annotator.py`
89. `src/xai/explain.py` (CLI)
90. `tests/test_xai.py`
91. Generate sample SHAP report
92. `notebooks/05_xai_walkthrough.ipynb`

## Phase L — Notebooks (93–97)
93. `notebooks/01_dataset_eda.ipynb`
94. `notebooks/02_envelope_analysis.ipynb`
95. `notebooks/03_feature_importance.ipynb`
96. `notebooks/04_rul_trajectories.ipynb`
97. Notebook README / index

## Phase M — Streamlit dashboard (98–105)
98. `app/dashboard.py` skeleton + fleet view
99. Asset drill-down page
100. Alert inbox + SHAP waterfall
101. Maintenance recommendation logic
102. PDF report export
103. `app/data_access.py` helpers
104. `tests/test_dashboard_utils.py`
105. Dashboard smoke test (headless)

## Phase N — Deployment & CI (106–114)
106. `Dockerfile`
107. `docker-compose.yml`
108. `.dockerignore`
109. `.github/workflows/ci.yml` (full: lint + test matrix)
110. `.streamlit/config.toml`
111. `docs/DEPLOYMENT.md`
112. Health-check / entrypoint script
113. `scripts/run_full_pipeline.py` (end-to-end orchestration)
114. Full pipeline smoke run in CI

## Phase O — Final polish (115–120)
115. Update README results with measured synthetic-scale numbers
116. Update README project structure to match reality
117. Full test suite run + coverage summary
118. `CHANGELOG.md`
119. Repo-wide lint/format pass
120. Final release checklist + `v0.1.0` tag
