.PHONY: env data train eval logits repro lint test

env:
\tpython -m venv .venv && . .venv/bin/activate && pip install -U pip wheel
\tpip install -r requirements.txt

data: ## synth
\tpython data/make_hard_v4.py --data_dir data/synth_clinical --n_samples 8000 --pos_ratio 0.5 --label_noise 0.05

train: ## BioClinical + ModernBERT on synth
\tpython scripts/train_compare.py --data_dir data/synth_clinical --max_len 128 --epochs 4 --batch 32 --lr 1e-5

eval: ## aggregate metrics table
\tpython scripts/summarize_results.py --runs_dir reports

logits: ## export doc-level probabilities for causal pipeline
\tpython scripts/export_logits.py --data_dir data/synth_clinical --model thomas-sounack/BioClinical-ModernBERT-base --out data/synth_clinical/doc_logits.csv

repro: env data train eval

lint:
\trufflehog --version >/dev/null 2>&1 || pip install ruff
\truff format && ruff check --fix

test:
\tpytest -q