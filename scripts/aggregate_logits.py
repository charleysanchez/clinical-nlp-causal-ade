import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--in_csv")
ap.add_argument("--out_csv")
a = ap.parse_args()

df = pd.read_csv(a.in_csv)
agg = (
    df.groupby("hadm_id")["prob_causal"]
    .agg(
        prob_mean="mean",
        prob_max="max",
        prob_p90=lambda x: np.percentile(x, 90),
        frac_above_05=lambda x: (x >= 0.5).mean(),
    )
    .reset_index()
)
Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
agg.to_csv(a.out_csv, index=False)
print("Saved to:", a.out_csv)
