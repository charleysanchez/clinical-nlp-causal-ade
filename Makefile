.PHONY: env data train eval logits repro lint test

env:
	python -m venv .venv && source .venv/bin/activate && pip install -U pip wheel
	pip install -r requirements.txt

data:  ## synth
	python data/synth_clinical/create_synthetic_data.py --data_dir data/synth_clinical --n_samples 8000 --pos_ratio 0.5 --label_noise 0.05

train: ## BioClinical + ModernBERT on synth (fixed split)
	python -m scripts.train_compare --config configs/experiment.synthetic.yaml --paths configs/paths.local.yaml

eval:  ## aggregate metrics table
	python scripts/summarize_results.py --runs_dir reports

logits: ## export doc-level probabilities for causal pipeline
	python scripts/export_logits.py --data_dir data/synth_clinical --model thomas-sounack/BioClinical-ModernBERT-base --out data/synth_clinical/doc_logits.csv

repro: env data train eval

lint:
	rufflehog --version >/dev/null 2>&1 || pip install ruff
	ruff format && ruff check --fix

test:
	pytest -q
