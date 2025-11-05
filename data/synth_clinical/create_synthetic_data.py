# data/make_hard_v2.py
import os, random, argparse
import numpy as np
import pandas as pd
from pathlib import Path

def main(data_dir: str):
    data_dir = Path(data_dir)
    notes_path  = data_dir / "notes.csv"
    cohort_path = data_dir / "cohort.csv"
    assert notes_path.exists() and cohort_path.exists(), f"Need {notes_path} and {cohort_path}"

    rng = np.random.default_rng(2025); random.seed(2025)
    notes   = pd.read_csv(notes_path)
    cohort  = pd.read_csv(cohort_path, parse_dates=["admittime","dischtime"])

    df = notes.merge(cohort[["hadm_id","subject_id","T","AKI","drug_name"]], on="hadm_id", how="left")

    # --- Lexica / templates ---
    ade_syns = [
        "acute kidney injury", "AKI", "renal injury", "worsening renal function",
        "rise in creatinine", "acute renal impairment"
    ]
    cue_A = ["after starting {drug}", "following initiation of {drug}", "soon after {drug} was begun"]
    cue_B = ["temporal association with {drug}", "shortly post-initiation of {drug}", "in close proximity to {drug} start"]
    link_pos = ["patient developed {ade}", "clinical picture consistent with {ade}",
                "lab trend indicates {ade}", "rapid creatinine rise indicating {ade}"]
    implicit = [
        "{ade} noted during treatment with {drug}.",
        "creatinine increased while on {drug}, consistent with {ade}.",
        "on {drug} with subsequent {ade}."
    ]
    ade_neg = ["no evidence of {ade}", "denies {ade}", "without signs of {ade}",
               "unlikely to represent {ade}", "creatinine stable; not {ade}"]
    distractors = [
        "family history of {ade}; no current symptoms.",
        "remote episode of {ade} years ago, resolved.",
        "education provided on recognizing {ade}, currently well.",
        "screening for {ade} negative today."
    ]
    neg_scope = ["no clear {ade} despite therapy.", "unable to confirm {ade}.", "not currently {ade}."]

    def dref(drug):
        return drug if isinstance(drug, str) and drug else "ACE inhibitor therapy"
    def pick_ade(): return random.choice(ade_syns)

    # --- Split by subject_id AND by template family (disjoint) ---
    subjects = df["subject_id"].dropna().astype(str).unique().tolist()
    rng.shuffle(subjects)
    cut = int(0.8 * len(subjects))
    train_subj, test_subj = set(subjects[:cut]), set(subjects[cut:])

    def pick_cue(is_train: bool):
        return random.choice(cue_A if is_train else cue_B)

    # Probabilities
    p_pos_explicit = 0.55       # positives: explicit cue vs implicit
    p_neg_with_cue = 0.40       # negatives: keep explicit cue but negated/no AKI/T
    p_mask_drug = 0.25          # mask [DRUG] in training subset
    p_mask_ade  = 0.25          # mask [ADE] in training subset

    def mask_drug(text: str) -> str:
        for d in ["lisinopril","enalapril","ramipril"]:
            text = text.replace(d, "[DRUG]")
        return text

    def mask_ade(text: str) -> str:
        t = text
        for a in ade_syns:
            t = t.replace(a, "[ADE]").replace(a.upper(), "[ADE]")
        return t

    rows = []
    for _, r in df.iterrows():
        base = str(r.get("text", "")).strip()
        is_train = str(r["subject_id"]) in train_subj
        drug = dref(r.get("drug_name", ""))
        ade  = pick_ade()

        text = base
        label = 0

        if r["T"] == 1 and r["AKI"] == 1:
            # Positive candidate
            if rng.uniform() < p_pos_explicit:
                cue  = pick_cue(is_train)
                link = random.choice(link_pos).format(ade=ade)
                text = f"{text} {cue.capitalize()}, {link}."
            else:
                phr = random.choice(implicit).format(ade=ade, drug=drug)
                text = f"{text} {phr}"
            label = 1
        else:
            # Negative
            if rng.uniform() < p_neg_with_cue:
                cue = pick_cue(is_train)
                neg = random.choice(ade_neg).format(ade=ade)
                text = f"{text} {cue.capitalize()}, {neg}."
            else:
                # mix distractors and neg-scope
                cand = distractors + [nv.format(ade=ade) for nv in neg_scope]
                text = f"{text} {random.choice(cand)}"
            label = 0

        # Optional masking (train only)
        if is_train and rng.uniform() < p_mask_drug:
            text = mask_drug(text)
        if is_train and rng.uniform() < p_mask_ade:
            text = mask_ade(text)

        rows.append({
            "doc_id": r["doc_id"],
            "hadm_id": r["hadm_id"],
            "subject_id": r["subject_id"],
            "charttime": r["charttime"],
            "category": r["category"],
            "text": text,
            "hard_v2_label": label,
            "split": "train" if is_train else "test",
        })

    df_v2 = pd.DataFrame(rows)

    # Save
    notes_v2  = data_dir / "notes_hard_v2.csv"
    labels_v2 = data_dir / "doc_labels_hard_v2.csv"
    split_v2  = data_dir / "split_hard_v2.csv"

    df_v2[["doc_id","hadm_id","subject_id","charttime","category","text"]].to_csv(notes_v2, index=False)
    df_v2[["doc_id","hadm_id","subject_id","hard_v2_label","split"]].to_csv(labels_v2, index=False)
    df_v2[["doc_id","split"]].to_csv(split_v2, index=False)
    print("Wrote:", notes_v2, labels_v2, split_v2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/synth_clinical")
    args = ap.parse_args()
    main(args.data_dir)
