#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

def main(in_labels: str, out_cohort: str):
    d = pd.read_csv(in_labels)  # columns: doc_id,hadm_id,subject_id,split,label,T,AKI
    # sanity: T/AKI/split should be constant within hadm_id
    gb = d.groupby("hadm_id")
    coh = gb.agg({
        "subject_id":"first",
        "split":"first",
        "T":"first",
        "AKI":"first"
    }).reset_index()
    # add note counts per admission (useful feature)
    coh["n_notes"] = gb.size().values
    Path(out_cohort).parent.mkdir(parents=True, exist_ok=True)
    coh.to_csv(out_cohort, index=False)
    print(f"✅ cohort saved: {out_cohort}  (rows={len(coh)})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_labels", default="data/synth_clinical/doc_labels.csv")
    ap.add_argument("--out_cohort", default="data/synth_clinical/cohort.csv")
    a = ap.parse_args()
    main(a.in_labels, a.out_cohort)
