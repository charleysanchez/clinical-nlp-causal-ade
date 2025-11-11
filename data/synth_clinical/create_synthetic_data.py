import os, random, argparse
from pathlib import Path
import numpy as np
import pandas as pd

RNG_SEED = 1337
random.seed(RNG_SEED); np.random.seed(RNG_SEED)

ADE_SYNS = [
    "acute kidney injury", "AKI", "renal injury", "worsening renal function",
    "rise in creatinine", "acute renal impairment"
]
CUE_TRAIN = [
    "after starting {drug}", "following initiation of {drug}", "soon after {drug} was begun"
]
CUE_TEST = [
    "temporal association with {drug}", "shortly post-initiation of {drug}",
    "in close proximity to {drug} start"
]
POS_LINK = [
    "patient developed {ade}", "clinical picture consistent with {ade}",
    "lab trend indicates {ade}", "no alternative explanation for {ade}"
]
NEG_LINK = [
    "creatinine remained stable, not {ade}", "findings argue against {ade}",
    "course not consistent with {ade}", "no evidence of {ade}"
]
IMPLICIT = [
    "{ade} noted during treatment with {drug}.",
    "creatinine increased while on {drug}, consistent with {ade}.",
    "on {drug} with subsequent {ade}."
]
DISTRACT = [
    "family history of {ade}; no current symptoms.",
    "remote episode of {ade} years ago, resolved.",
    "education provided on recognizing {ade}, currently well.",
    "screening for {ade} negative today."
]

FILLER_SENTENCES = [
    "Patient is alert and oriented.", "No acute events overnight.",
    "Vitals monitored closely.", "Care plan discussed with team.",
    "Symptoms reviewed with nursing staff.", "Labs reviewed and trended."
]

def _dref(drug):
    return drug if isinstance(drug, str) and drug else "ACE inhibitor therapy"

def _pick(x): return random.choice(x)

def _mask_all(s: str) -> str:
    for d in ["lisinopril", "enalapril", "ramipril"]:
        s = s.replace(d, "[DRUG]")
    for a in ADE_SYNS:
        s = s.replace(a, "[ADE]").replace(a.upper(), "[ADE]")
    return s

def _typo(s: str, p=0.03):
    # random character drops/swaps to add noise
    out = []
    for w in s.split():
        if random.random() < p and len(w) > 4:
            i = random.randrange(1, len(w)-1)
            w = w[:i] + w[i+1:]  # drop char
        if random.random() < p and len(w) > 4:
            i = random.randrange(1, len(w)-1)
            w = w[:i] + w[i+1] + w[i] + w[i+2:]  # swap
        out.append(w)
    return " ".join(out)

def _shuffle_sentences(sents):
    random.shuffle(sents)
    return " ".join(sents)

def ensure_balanced_idx(n, pos_ratio=0.5):
    n_pos = int(n * pos_ratio)
    y = np.array([1]*n_pos + [0]*(n-n_pos))
    np.random.shuffle(y)
    return y

def main(
    data_dir: str,
    n_samples: int,
    pos_ratio: float,
    label_noise: float,
):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Optional: use real cohort/notes to seed base text; else synth minimal bases
    notes_csv  = data_dir / "notes.csv"
    cohort_csv = data_dir / "cohort.csv"
    base_rows = []

    if notes_csv.exists() and cohort_csv.exists():
        notes = pd.read_csv(notes_csv)
        cohort = pd.read_csv(cohort_csv)
        merged = notes.merge(cohort[["hadm_id","subject_id","T","AKI","drug_name"]],
                             on="hadm_id", how="left")
        # If fewer than requested, we will sample with replacement
        pool = merged[["subject_id","hadm_id","text","T","AKI","drug_name"]].copy()
        if pool.empty:
            raise RuntimeError("Found notes/cohort but merge is empty. Check keys.")
        while len(base_rows) < n_samples:
            r = pool.sample(1, replace=True, random_state=np.random.randint(10**9)).iloc[0]
            base_rows.append({
                "subject_id": r["subject_id"],
                "hadm_id": r["hadm_id"],
                "base_text": (r["text"] if isinstance(r["text"], str) else ""),
                "T": int(r.get("T", 0)),
                "AKI": int(r.get("AKI", 0)),
                "drug_name": r.get("drug_name", ""),
            })
    else:
        # Pure synthetic bases
        for i in range(n_samples):
            subj = f"P{i%5000:06d}"        # ~5k subjects → multiple notes per subject
            hadm = f"H{i:07d}"
            T = np.random.binomial(1, 0.5)
            AKI = np.random.binomial(1, 0.4 + 0.2*T)  # modest treatment-outcome link
            base = _shuffle_sentences(random.sample(FILLER_SENTENCES, k=3))
            drug = _pick(["lisinopril","enalapril","ramipril"]) if T else ""
            base_rows.append({
                "subject_id": subj, "hadm_id": hadm, "base_text": base,
                "T": T, "AKI": AKI, "drug_name": drug
            })

    base_df = pd.DataFrame(base_rows)

    # Split by subject to avoid leakage and enforce template disjoint
    subjects = base_df["subject_id"].astype(str).unique().tolist()
    np.random.shuffle(subjects)
    cut = int(0.8 * len(subjects))
    train_subj, test_subj = set(subjects[:cut]), set(subjects[cut:])

    # Create balanced target labels (independent of T/AKI), then
    # enforce *textual* logic so that the label is not perfectly deducible from any single cue.
    Y = ensure_balanced_idx(len(base_df), pos_ratio)
    base_df["target"] = Y

    rows = []
    for i, r in base_df.iterrows():
        subj = str(r["subject_id"])
        is_train = subj in train_subj
        split = "train" if is_train else "test"
        base = r["base_text"] or ""
        drug = _dref(r["drug_name"])
        ade  = _pick(ADE_SYNS)

        cue = _pick(CUE_TRAIN if is_train else CUE_TEST)

        # draw ~half explicit, half implicit text per split
        if r["target"] == 1:
            # POS: (test is implicit-only 70% to block cue shortcut)
            if (not is_train and random.random() < 0.7) or (is_train and random.random() < 0.5):
                # implicit
                sent = _pick(IMPLICIT).format(ade=ade, drug=drug)
            else:
                # explicit with a positive link (includes a 'no' phrase sometimes to break "no==neg")
                link = _pick(POS_LINK).format(ade=ade)
                sent = f"{cue.capitalize()}, {link}."
        else:
            # NEG: include explicit cues often so cue≠label; also implicit-looking negatives
            if random.random() < 0.6:
                nlink = _pick(NEG_LINK).format(ade=ade)
                sent = f"{cue.capitalize()}, {nlink}."
            else:
                sent = f"{_pick(IMPLICIT).format(ade=ade, drug=drug)} {_pick(NEG_LINK).format(ade=ade)}."

        # Randomize position; add filler; mask + typos in BOTH splits
        sents = [base, sent, _pick(FILLER_SENTENCES)]
        text = _shuffle_sentences(sents)
        text = _mask_all(text)
        text = _typo(text, p=0.05)

        rows.append({
            "doc_id": f"D{i:08d}",
            "subject_id": subj,
            "hadm_id": r["hadm_id"],
            "text": text,
            "split": split,
            "label": int(r["target"]),
            "T": int(r["T"]),
            "AKI": int(r["AKI"])
        })

    df = pd.DataFrame(rows)

    # Inject small symmetric label noise to avoid memorization by tiny quirks
    if label_noise > 0:
        flip = np.random.rand(len(df)) < label_noise
        df.loc[flip, "label"] = 1 - df.loc[flip, "label"]

    # Ensure both classes present in both splits
    for sp in ["train","test"]:
        if df[df.split==sp]["label"].nunique() < 2:
            # flip one random label
            j = df.index[df.split==sp][0]
            df.loc[j, "label"] = 1 - df.loc[j, "label"]

    # Save
    (data_dir/"notes.csv").write_text("")  # touch on Windows
    df[["doc_id","hadm_id","subject_id","text"]].to_csv(data_dir/"notes.csv", index=False)
    df[["doc_id","hadm_id","subject_id","split","label","T","AKI"]].to_csv(data_dir/"doc_labels.csv", index=False)

    # Quick class balance print
    bal = df.groupby(["split","label"]).size().groupby(level=0).apply(lambda x: (x/x.sum()).round(3))
    print("Saved v4 at:", data_dir)
    print("Class balance by split:\n", bal)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/synth_clinical")
    ap.add_argument("--n_samples", type=int, default=8000)
    ap.add_argument("--pos_ratio", type=float, default=0.5)
    ap.add_argument("--label_noise", type=float, default=0.05)
    args = ap.parse_args()
    main(args.data_dir, args.n_samples, args.pos_ratio, args.label_noise)
