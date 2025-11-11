#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

def main(cohort_csv: str, hadm_feats_csv: str, out_csv: str):
    coh = pd.read_csv(cohort_csv)
    feats = pd.read_csv(hadm_feats_csv)  # hadm_id, prob_mean, prob_max, prob_p90, frac_above_05
    out = coh.merge(feats, on="hadm_id", how="left").fillna({
        "prob_mean":0.0, "prob_max":0.0, "prob_p90":0.0, "frac_above_05":0.0
    })
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"cohort_with_logits saved: {out_csv}  (rows={len(out)})")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="data/synth_clinical/cohort.csv")
    ap.add_argument("--hadm_feats", default="data/synth_clinical/hadm_logits_features.csv")
    ap.add_argument("--out", default="data/synth_clinical/cohort_with_logits.csv")
    a = ap.parse_args()
    main(a.cohort, a.hadm_feats, a.out)
