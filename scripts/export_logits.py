#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

# we only need the checkpoint resolver; keep this import lightweight
try:
    from scripts.train_utils import best_checkpoint_dir
except Exception:
    # Fallback: allow running directly without package import
    from train_utils import best_checkpoint_dir  # type: ignore


def parse_args():
    ap = argparse.ArgumentParser(
        description="Export document-level logits/probabilities."
    )
    ap.add_argument(
        "--data_dir",
        default="data/synth_clinical",
        help="Folder with notes/labels CSVs",
    )
    ap.add_argument("--notes_file", default="notes.csv")
    ap.add_argument("--labels_file", default="doc_labels.csv")

    # Which data to score
    ap.add_argument(
        "--split", default="all", choices=["train", "val", "test", "all"]
    )

    # Model loading settings
    ap.add_argument(
        "--run_dir",
        default=None,
        help="Trainer output directory that contains checkpoint-* folders (takes precedence if provided).",
    )
    ap.add_argument(
        "--model",
        default="thomas-sounack/BioClinical-ModernBERT-base",
        help="HF model id or local path (used if --run_dir not provided).",
    )
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)

    # Output
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument(
        "--include_logits",
        action="store_true",
        help="Also save raw logits for each class",
    )
    return ap.parse_args()


def softmax_prob_1(logits: np.ndarray) -> np.ndarray:
    """stable softmax → P(class=1)"""
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e[:, 1] / e.sum(axis=1)


def load_model_and_tokenizer(run_dir: Optional[str], model_name: str):
    if run_dir:
        run_dir = str(run_dir)
        ckpt = best_checkpoint_dir(run_dir)
        if ckpt is None:
            # no checkpoint folders; allow loading the final saved model dir
            ckpt = Path(run_dir)
        print(f"Loading from checkpoint: {ckpt}")
        tok = AutoTokenizer.from_pretrained(ckpt, add_prefix_space=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(ckpt)
    else:
        print(f"Loading base model: {model_name}")
        tok = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
    return mdl, tok


def build_dataframe(
    data_dir: Path, notes_file: str, labels_file: str, split: str
) -> pd.DataFrame:
    notes = pd.read_csv(data_dir / notes_file)
    labels = pd.read_csv(data_dir / labels_file)

    # try to preserve standard columns if they exist
    keep_cols = [
        c
        for c in ["doc_id", "hadm_id", "subject_id", "text"]
        if c in notes.columns
    ]
    df = notes[keep_cols].merge(
        labels[
            [c for c in ["doc_id", "split", "label"] if c in labels.columns]
        ],
        on="doc_id",
        how="left",
    )

    if split != "all":
        if "split" not in df.columns:
            raise ValueError(
                "Requested a specific split but no 'split' column is present in labels."
            )
        df = df[df["split"] == split].reset_index(drop=True)
    return df


def tokenize_texts(
    tokenizer, texts: List[str], max_len: int, device: torch.device
):
    collator = DataCollatorWithPadding(tokenizer)

    class DS(torch.utils.data.Dataset):
        def __init__(self, X):
            self.X = X

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            return {"text": self.X[i]}

    def _preprocess(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_len,
        )

    ds = DS(texts)
    # We'll collate after tokenization to keep it simple
    # For speed with short texts, we can tokenize on the fly in the loop.

    return ds, collator, _preprocess


def predict(
    model,
    tokenizer,
    df: pd.DataFrame,
    max_len: int,
    batch_size: int,
    include_logits: bool = False,
) -> Dict[str, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    ds, collator, preprocess = tokenize_texts(
        tokenizer, df["text"].tolist(), max_len, device
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collator(
            preprocess({"text": [x["text"] for x in batch]})
        ),
    )

    all_logits: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            logits = out.logits.detach().cpu().numpy()
            all_logits.append(logits)

    logits = np.concatenate(all_logits, axis=0)
    p1 = softmax_prob_1(logits)

    result = {"prob_causal": p1}
    if include_logits:
        result["logit_0"] = logits[:, 0]
        result["logit_1"] = logits[:, 1]
    return result


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    # 1) Load model/tokenizer
    model, tok = load_model_and_tokenizer(args.run_dir, args.model)

    # 2) Load dataframe to score
    df = build_dataframe(
        data_dir, args.notes_file, args.labels_file, args.split
    )
    if df.empty:
        raise SystemExit(
            "No rows to score (empty dataframe after split selection)."
        )

    # 3) Predict
    out_dict = predict(
        model=model,
        tokenizer=tok,
        df=df,
        max_len=args.max_len,
        batch_size=args.batch,
        include_logits=args.include_logits,
    )

    # 4) Assemble output
    out_df = df[
        ["doc_id"]
        + [c for c in ["hadm_id", "subject_id", "split"] if c in df.columns]
    ].copy()
    out_df["prob_causal"] = out_dict["prob_causal"]
    if args.include_logits:
        out_df["logit_0"] = out_dict["logit_0"]
        out_df["logit_1"] = out_dict["logit_1"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved logits: {out_path}  (rows={len(out_df)})")


if __name__ == "__main__":
    main()
