.PHONY: install test lint format synthetic preprocess features train-classical train-deep dashboard clean

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest -q --cov=src --cov-report=term-missing

lint:
	ruff check src tests app scripts

format:
	ruff check --fix src tests app scripts

synthetic:
	python scripts/generate_synthetic.py --output_dir data/raw

preprocess:
	python -m src.preprocessing.run --dataset cwru --input_dir data/raw/cwru/synthetic --output_dir data/processed/cwru
	python -m src.preprocessing.run --dataset ims --input_dir data/raw/ims/synthetic --output_dir data/processed/ims

features:
	python -m src.features.extract --dataset cwru --input_dir data/processed/cwru --output_dir data/processed/cwru
	python -m src.features.extract --dataset ims --input_dir data/processed/ims --output_dir data/processed/ims

train-classical:
	python -m src.training.train_classical --model xgboost --dataset cwru --output_dir runs/xgb_cwru

train-deep:
	python -m src.training.train_deep --model turboguard_hybrid --dataset_fault cwru --dataset_rul ims --output_dir runs/hybrid_multitask

dashboard:
	streamlit run app/dashboard.py

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
