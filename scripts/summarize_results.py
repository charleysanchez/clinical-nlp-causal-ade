#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="reports", help="Root directory to scan for metrics")
    ap.add_argument("--pattern", default="**/metrics*.json", help="Glob pattern under runs_dir")
    ap.add_argument("--out_csv", default="reports/summary.csv")
    ap.add_argument("--out_md", default="reports/summary.md")
    ap.add_argument("--sort_by", default="eval_auprc", help="Metric to sort by (desc)")
    ap.add_argument("--top_k", type=int, default=20, help="How many rows to show in printed preview")
    ap.add_argument("--round", dest="round_digits", type=int, default=6, help="Rounding digits for floats")
    return ap.parse_args()

def _maybe_float(v):
    try:
        return float(v)
    except Exception:
        return v
    
def _flatten_compare_dict(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Handles compare-style file produced by train_compare.py:
    {
      "BioClinical ModernBERT": {...metrics...},
      "ModernBERT (vanilla)": {...metrics...}
    }
    """
    rows = []
    for model_label, metrics in payload.items():
        row = {"model_label": model_label}
        row.update(metrics)
        rows.append(row)
    return rows


def _infer_model_from_path(p: Path) -> str:
    """
    Try to infer the model name from parent folders, e.g.:
    reports/doc_cls_synth_answerdotai_ModernBERT-base/... -> answerdotai/ModernBERT-base
    """
    # Look for doc_cls_*_<model> pattern
    for ancestor in [p.parent, p.parent.parent, p.parent.parent.parent]:
        if ancestor is None or ancestor == ancestor.parent:
            break
        m = re.search(r"doc_cls_[^/]+_(.+)$", ancestor.name)
        if m:
            # Convert '_' to '/' only for the last underscore-joined model path we produced earlier
            model_hint = m.group(1)
            # Heuristic: answerdotai_ModernBERT-base -> answerdotai/ModernBERT-base
            return model_hint.replace("_", "/", 1)
    return ""



def _infer_dataset_from_name(p: Path) -> str:
    # Prefer metrics_<dataset>.json file name
    m = re.match(r"metrics_(.+)\.json$", p.name)
    if m:
        return m.group(1)
    # Or from run folder doc_cls_<dataset>_...
    for ancestor in [p.parent, p.parent.parent]:
        if ancestor is None or ancestor == ancestor.parent:
            break
        m = re.search(r"doc_cls_([^_]+)_", ancestor.name)
        if m:
            return m.group(1)
    return ""


def load_metrics_file(p: Path) -> List[Dict[str, Any]]:
    payload = None
    with p.open("r") as f:
        payload = json.load(f)

    rows: List[Dict[str, Any]] = []
    # Case 1: compare dict
    if isinstance(payload, dict) and any(isinstance(v, dict) for v in payload.values()):
        rows = _flatten_compare_dict(payload)
    # Case 2: a single metrics dict
    elif isinstance(payload, dict):
        rows = [payload]
    # Case 3: a list of dicts
    elif isinstance(payload, list):
        rows = payload
    else:
        return []

    # Enrich with hints from path
    dataset_hint = _infer_dataset_from_name(p)
    model_hint_from_path = _infer_model_from_path(p)

    enriched = []
    for r in rows:
        r = dict(r)  # copy
        # Standardize common keys if possible
        # Accept either 'eval_*' or plain keys
        for k in ["eval_accuracy", "eval_f1", "eval_auroc", "eval_auprc", "eval_loss", "epoch", "seconds"]:
            if k not in r:
                plain = k.replace("eval_", "")
                if plain in r:
                    r[k] = r.pop(plain)

        # Attach labels
        r["dataset"] = r.get("dataset", dataset_hint)
        # 'model_label' may carry a human-friendly label; fall back to path inference
        r["model_label"] = r.get("model_label", "")
        r["model_name"] = r.get("model_name", model_hint_from_path)

        # Parse floats
        for k in list(r.keys()):
            if k.startswith("eval_") or k in ("epoch", "seconds", "accuracy", "f1", "auroc", "auprc"):
                r[k] = _maybe_float(r[k])

        enriched.append(r)
    return enriched


def main():
    args = parse_args()
    root = Path(args.runs_dir)
    files = sorted(root.glob(args.pattern))
    if not files:
        print(f"No metrics found under {root} matching '{args.pattern}'.")
        return

    all_rows: List[Dict[str, Any]] = []
    for p in files:
        try:
            rows = load_metrics_file(p)
            for r in rows:
                r["_source_file"] = str(p)
            all_rows.extend(rows)
        except Exception as e:
            print(f"[WARN] Failed to parse {p}: {e}")

    if not all_rows:
        print("No parsable metrics.")
        return

    df = pd.DataFrame(all_rows)

    # Canonical column ordering
    preferred_cols = [
        "dataset", "model_label", "model_name",
        "eval_accuracy", "eval_f1", "eval_auroc", "eval_auprc",
        "eval_loss", "epoch", "seconds",
        "_source_file"
    ]
    # Add any extra keys at the end
    cols = [c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]
    df = df[cols]

    # Rounding
    float_cols = [c for c in df.columns if df[c].dtype.kind in "fc"]
    df[float_cols] = df[float_cols].round(args.round_digits)

    # Sorting
    sort_key = args.sort_by if args.sort_by in df.columns else "eval_auprc"
    ascending = False
    df = df.sort_values(by=sort_key, ascending=ascending).reset_index(drop=True)

    # Write outputs
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    # Small markdown table
    md_cols = [c for c in ["dataset","model_label","model_name","eval_accuracy","eval_f1","eval_auroc","eval_auprc","eval_loss","seconds"] if c in df.columns]
    md = df[md_cols].to_markdown(index=False)
    out_md.write_text(md)

    # Console preview
    preview = df.head(args.top_k)
    print(f"\nSaved CSV: {out_csv}")
    print(f"Saved MD:  {out_md}\n")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(preview)


if __name__ == "__main__":
    main()
