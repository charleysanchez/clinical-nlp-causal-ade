import argparse
import json
from pathlib import Path

import yaml

from scripts.train_utils import build_dataset, run_model


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to experiment YAML")
    ap.add_argument(
        "--paths", required=False, help="Path to environment pahts YAML"
    )
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    paths = load_yaml(args.paths) if args.paths else {}

    # merge
    merged = {**cfg, **paths}

    data_dir = merged.get("data_dir", "data/synth_clinical")
    reports_root = Path(merged.get("reports_root", "reports"))
    reports_root.mkdir(parents=True, exist_ok=True)

    ds = build_dataset(data_dir)

    train_cfg = merged.get("train", {})
    models = merged["model_names"]

    results = {}
    for name, model_name in zip(["BioClinical", "ModernBERT"], models):
        outdir = (
            reports_root
            / f"doc_cls_{merged.get('dataset', 'synth')}_{model_name.replace('/', '_')}"
        )
        metrics = run_model(
            model_name,
            ds,
            max_len=merged.get("tokenizer_max_len", 128),
            epochs=train_cfg.get("epochs", 4),
            batch=train_cfg.get("batch", 32),
            lr=train_cfg.get("lr", 1e-5),
            weight_decay=train_cfg.get("weight_decay", 0.10),
            label_smoothing=train_cfg.get("label_smoothing", 0.10),
            eval_metric=train_cfg.get("eval_metric", "eval_auprc"),
            patience=train_cfg.get("early_stopping_patience", 2),
            fp16=True,
            seed=777,
            output_dir=outdir,
        )
        results[name] = metrics

    out_json = reports_root / f"metrics_{merged.get('dataset', 'synth')}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved:", out_json)


if __name__ == "__main__":
    main()
