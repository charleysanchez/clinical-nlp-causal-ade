from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


# --------------------
# Data loading helpers
# --------------------
def build_dataset(
    data_dir: str | Path,
    notes_file: str = "notes.csv",
    labels_file: str = "doc_labels.csv",
    text_col: str = "text",
    label_col: str = "label",
    split_col: str = "split",
) -> Dict[str, Dataset]:
    """
    Load fixed_split doc classification data and return HF Datasets dict with 'train' and 'test'.
    """
    data_dir = Path(data_dir)
    notes = pd.read_csv(data_dir / notes_file)
    labels = pd.read_csv(data_dir / labels_file)

    df = notes.merge(
        labels[["doc_id", split_col, label_col]], on="doc_id", how="left"
    )
    train_df = df[df[split_col] == "train"][[text_col, label_col]].reset_index(
        drop=True
    )
    test_df = df[df[split_col] == "test"][[text_col, label_col]].reset_index(
        drop=True
    )

    ds = {
        "train": Dataset.from_pandas(train_df),
        "test": Dataset.from_pandas(test_df),
    }

    return ds


# --------------------
# Metrics
# --------------------
def compute_metrics_fn(
    eval_pred: Tuple[np.ndarray, np.ndarray],
) -> Dict[str, float]:
    logits, y = eval_pred
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    p1 = p[:, 1] / p.sum(axis=1)
    yhat = (p1 >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y, yhat),
        "f1": f1_score(y, yhat),
        "auroc": roc_auc_score(y, p1),
        "auprc": average_precision_score(y, p1),
    }


# --------------------
# Core trainer
# --------------------
def run_model(
    model_name: str,
    ds: Dict[str, Dataset],
    *,
    max_len: int = 128,
    epochs: int = 4,
    batch: int = 32,
    lr: float = 1e-5,
    weight_decay: float = 0.10,
    label_smoothing: float = 0.10,
    eval_metric: str = "eval_auprc",
    patience: int = 2,
    fp16: bool = True,
    seed: int = 777,
    output_dir: str | Path = "./reports/run",
) -> Dict[str, float]:
    """
    Train and evaluate a ModernBERT-style classifier with compact, regularized defaults.
    Returns HF `trainer.evaluate()` metrics.
    """
    # Tokenizer + dynamic paddings
    tok = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

    def tok_map(batch):
        return tok(batch["text"], max_length=max_len, truncation=True)

    enc = {
        split: d.map(tok_map, batched=True, remove_columns=["text"])
        for split, d in ds.items()
    }
    data_collator = DataCollatorWithPadding(tok)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training args
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=weight_decay,
        label_smoothing_factor=label_smoothing,
        lr_scheduler_type="cosine",
        warmup_ratio=0.10,
        fp16=fp16,
        max_grad_norm=1.0,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=eval_metric,
        greater_is_better=True,
        logging_steps=100,
        report_to=[],
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=enc["train"],
        eval_dataset=enc["test"],
        tokenizer=tok,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )

    trainer.train()
    metrics = trainer.evaluate()
    return metrics


# ----------------------------
# Utility: find best checkpoint folder saved by Trainer
# ----------------------------
def best_checkpoint_dir(run_dir: str | Path) -> Optional[Path]:
    """
    Returns the path to the latest/best checkpoint directory (e.g., 'checkpoint-1600')
    if present inside `run_dir`. Returns None if not found.
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return None
    cpts = sorted(
        run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])
    )
    return cpts[-1] if cpts else None
