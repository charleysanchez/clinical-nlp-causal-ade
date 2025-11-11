import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

RNG_SEED = 1337
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

ADE_SYNS = [
    "acute kidney injury",
    "AKI",
    "renal injury",
    "worsening renal function",
    "rise in creatinine",
    "acute renal impairment",
]
CUE_TRAIN = [
    "after starting {drug}",
    "following initiation of {drug}",
    "soon after {drug} was begun",
]
CUE_TEST = [
    "temporal association with {drug}",
    "shortly post-initiation of {drug}",
    "in close proximity to {drug} start",
]
POS_LINK = [
    "patient developed {ade}",
    "clinical picture consistent with {ade}",
    "lab trend indicates {ade}",
    "no alternative explanation for {ade}",
]
NEG_LINK = [
    "creatinine remained stable, not {ade}",
    "findings argue against {ade}",
    "course not consistent with {ade}",
    "no evidence of {ade}",
]
IMPLICIT = [
    "{ade} noted during treatment with {drug}.",
    "creatinine increased while on {drug}, consistent with {ade}.",
    "on {drug} with subsequent {ade}.",
]
DISTRACT = [
    "family history of {ade}; no current symptoms.",
    "remote episode of {ade} years ago, resolved.",
    "education provided on recognizing {ade}, currently well.",
    "screening for {ade} negative today.",
]

FILLER_SENTENCES = [
    "Patient is alert and oriented.",
    "No acute events overnight.",
    "Vitals monitored closely.",
    "Care plan discussed with team.",
    "Symptoms reviewed with nursing staff.",
    "Labs reviewed and trended.",
]


def _dref(drug):
    return drug if isinstance(drug, str) and drug else "ACE inhibitor therapy"


def _pick(x):
    return random.choice(x)


def _mask_all(s: str) -> str:
    for d in ["lisinopril", "enalapril", "ramipril"]:
        s = s.replace(d, "[DRUG]")
    for a in ADE_SYNS:
        s = s.replace(a, "[ADE]").replace(a.upper(), "[ADE]")
    return s


def _typo(s: str, p=0.03):
    out = []
    for w in s.split():
        if random.random() < p and len(w) > 4:
            i = random.randrange(1, len(w) - 1)
            w = w[:i] + w[i + 1 :]  # drop char
        if random.random() < p and len(w) > 4:
            i = random.randrange(1, len(w) - 1)
            w = w[:i] + w[i + 1] + w[i] + w[i + 2 :]  # swap
        out.append(w)
    return " ".join(out)


def _shuffle_sentences(sents):
    random.shuffle(sents)
    return " ".join(sents)


def ensure_balanced_idx(n, pos_ratio=0.5):
    n_pos = int(n * pos_ratio)
    y = np.array([1] * n_pos + [0] * (n - n_pos))
    np.random.shuffle(y)
    return y


def make_note_text(
    base: str,
    drug: str,
    ade: str,
    split: str,
    is_positive: bool,
    implicit_bias_test: float,
    explicit_rate_train: float,
) -> str:
    """
    Create a single note’s text with randomized cues + filler.
    - Test positives are implicit-only with probability ~implicit_bias_test.
    - Train positives are explicit with probability ~explicit_rate_train.
    - Negatives sometimes include explicit cues but negate/link against ADE.
    """
    cue = _pick(CUE_TRAIN if split == "train" else CUE_TEST)

    if is_positive:
        # control explicit/implicit by split
        if split == "test" and random.random() < implicit_bias_test:
            main = _pick(IMPLICIT).format(ade=ade, drug=drug)
        else:
            if split == "train" and random.random() < explicit_rate_train:
                link = _pick(POS_LINK).format(ade=ade)
                main = f"{cue.capitalize()}, {link}."
            else:
                main = _pick(IMPLICIT).format(ade=ade, drug=drug)
    else:
        # negatives include cue/no-evidence combos often
        if random.random() < 0.6:
            nlink = _pick(NEG_LINK).format(ade=ade)
            main = f"{cue.capitalize()}, {nlink}."
        else:
            main = f"{_pick(IMPLICIT).format(ade=ade, drug=drug)} {_pick(NEG_LINK).format(ade=ade)}."

    # Add filler + distractors and shuffle to avoid fixed position shortcuts
    sents = [base, main, _pick(FILLER_SENTENCES), _pick(DISTRACT)]
    text = _shuffle_sentences(sents)
    text = _mask_all(text)
    text = _typo(text, p=0.05)
    return text


def main(
    data_dir: str,
    n_admissions: int,
    pos_ratio: float,
    label_noise: float,
    min_notes: int,
    max_notes: int,
    implicit_bias_test: float,
    explicit_rate_train: float,
    seed: int,
):
    random.seed(seed)
    np.random.seed(seed)

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # If real seed files exist, sample from them; else synth admissions
    notes_csv = data_dir / "notes.csv"
    cohort_csv = data_dir / "cohort.csv"
    base_rows = []

    if notes_csv.exists() and cohort_csv.exists():
        notes = pd.read_csv(notes_csv)
        cohort = pd.read_csv(cohort_csv)
        merged = notes.merge(
            cohort[["hadm_id", "subject_id", "T", "AKI", "drug_name"]],
            on="hadm_id",
            how="left",
        )
        pool = merged[
            ["subject_id", "hadm_id", "text", "T", "AKI", "drug_name"]
        ].copy()
        if pool.empty:
            raise RuntimeError(
                "Found notes/cohort but merge is empty. Check keys."
            )
        while len(base_rows) < n_admissions:
            r = pool.sample(
                1, replace=True, random_state=np.random.randint(10**9)
            ).iloc[0]
            base_rows.append(
                {
                    "subject_id": r["subject_id"],
                    "hadm_id": r["hadm_id"],
                    "base_text": (
                        r["text"] if isinstance(r["text"], str) else ""
                    ),
                    "T": int(r.get("T", 0)),
                    "AKI": int(r.get("AKI", 0)),
                    "drug_name": r.get("drug_name", ""),
                }
            )
    else:
        # Pure synthetic admissions
        for i in range(n_admissions):
            subj = f"P{i % 10000:06d}"  # allow more subjects as we scale
            hadm = f"H{i:07d}"
            T = np.random.binomial(1, 0.5)
            AKI = np.random.binomial(
                1, 0.4 + 0.2 * T
            )  # modest treatment-outcome link
            base = _shuffle_sentences(random.sample(FILLER_SENTENCES, k=3))
            drug = _pick(["lisinopril", "enalapril", "ramipril"]) if T else ""
            base_rows.append(
                {
                    "subject_id": subj,
                    "hadm_id": hadm,
                    "base_text": base,
                    "T": T,
                    "AKI": AKI,
                    "drug_name": drug,
                }
            )

    base_df = pd.DataFrame(base_rows)

    # Subject-disjoint split
    subjects = base_df["subject_id"].astype(str).unique().tolist()
    np.random.shuffle(subjects)
    cut = int(0.8 * len(subjects))
    train_subj, test_subj = set(subjects[:cut]), set(subjects[cut:])

    # Admission-level target, balanced independently of T/AKI to avoid shortcuts
    Y = ensure_balanced_idx(len(base_df), pos_ratio)
    base_df["target"] = Y

    # Generate multiple notes per admission
    rows = []
    doc_counter = 0
    for _, r in base_df.iterrows():
        subj = str(r["subject_id"])
        split = "train" if subj in train_subj else "test"
        base = r["base_text"] or ""
        drug = _dref(r["drug_name"])
        ade = _pick(ADE_SYNS)

        k_notes = random.randint(min_notes, max_notes)
        # Ensure at least one note actually carries the class signal
        sig_note_idx = random.randrange(k_notes)

        for j in range(k_notes):
            is_signal = j == sig_note_idx
            # signal note follows the positive/negative construction; others are mostly filler-ish with weaker cues
            is_positive = (
                bool(r["target"])
                if is_signal
                else bool(r["target"] and random.random() < 0.4)
            )
            text = make_note_text(
                base=base,
                drug=drug,
                ade=ade,
                split=split,
                is_positive=is_positive,
                implicit_bias_test=implicit_bias_test,
                explicit_rate_train=explicit_rate_train,
            )
            rows.append(
                {
                    "doc_id": f"D{doc_counter:09d}",
                    "subject_id": subj,
                    "hadm_id": r["hadm_id"],
                    "text": text,
                    "split": split,
                    "label": int(
                        r["target"]
                    ),  # doc inherits admission target for supervision
                    "T": int(r["T"]),
                    "AKI": int(r["AKI"]),
                }
            )
            doc_counter += 1

    df = pd.DataFrame(rows)

    # Label noise (document level)
    if label_noise > 0:
        flip = np.random.rand(len(df)) < label_noise
        df.loc[flip, "label"] = 1 - df.loc[flip, "label"]

    # Ensure both classes present in both splits
    for sp in ["train", "test"]:
        if (
            df[df.split == sp]["label"].nunique() < 2
            and len(df[df.split == sp]) > 0
        ):
            j = df.index[df.split == sp][0]
            df.loc[j, "label"] = 1 - df.loc[j, "label"]

    # Save
    (data_dir / "notes.csv").write_text("")  # touch for Windows cache
    df[["doc_id", "hadm_id", "subject_id", "text"]].to_csv(
        data_dir / "notes.csv", index=False
    )
    df[
        ["doc_id", "hadm_id", "subject_id", "split", "label", "T", "AKI"]
    ].to_csv(data_dir / "doc_labels.csv", index=False)

    # VERSION / provenance
    version_path = data_dir / "VERSION"
    cfg = {
        "n_admissions": int(n_admissions),
        "pos_ratio": float(pos_ratio),
        "label_noise": float(label_noise),
        "min_notes": int(min_notes),
        "max_notes": int(max_notes),
        "implicit_bias_test": float(implicit_bias_test),
        "explicit_rate_train": float(explicit_rate_train),
        "seed": int(seed),
    }
    version_path.write_text(json.dumps(cfg, indent=2))

    # Quick class balance print (admission-level intent → document-level labels)
    bal = (
        df.groupby(["split", "label"])
        .size()
        .groupby(level=0)
        .apply(lambda x: (x / x.sum()).round(3))
    )
    print("Saved at:", data_dir)
    print("Notes:", len(df))
    print("Admissions:", len(base_df))
    print("Class balance by split:\n", bal)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/synth_clinical")
    ap.add_argument("--n_admissions", type=int, default=8000)
    ap.add_argument("--pos_ratio", type=float, default=0.5)
    ap.add_argument("--label_noise", type=float, default=0.05)
    ap.add_argument("--min_notes", type=int, default=2)
    ap.add_argument("--max_notes", type=int, default=6)
    ap.add_argument("--implicit_bias_test", type=float, default=0.8)
    ap.add_argument("--explicit_rate_train", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()
    main(
        args.data_dir,
        args.n_admissions,
        args.pos_ratio,
        args.label_noise,
        args.min_notes,
        args.max_notes,
        args.implicit_bias_test,
        args.explicit_rate_train,
        args.seed,
    )
