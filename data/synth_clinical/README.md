# Synthetic Clinical NLP + Causal Inference Dataset

This synthetic dataset mirrors the plan:
- **Exposure (T)**: ACE inhibitor administration during admission (from `medications.csv` / `cohort.csv:T`).
- **Outcome (Y)**: `AKI` and `mortality_in_hosp` in `outcomes` and merged into `cohort.csv`.
- **Confounders (W)**: demographics, comorbidities, baseline labs/vitals in `patients.csv` and `cohort.csv`.
- **Notes**: clinical text in `notes.csv` with crude span annotations in `spans_bio.csv` (labels: DRUG, ADE).

## Files
- `patients.csv`: subject-level demographics & baseline state (age, sex, race, CKD, HTN, DM2, Charlson, SOFA at admit, baseline creatinine).
- `admissions.csv`: one admission per subject with admit/discharge times and length of stay.
- `icustays.csv`: ICU stay times (may be null if no ICU stay).
- `medications.csv`: exposure indicator via ACE inhibitor rows (drug_name/dose/time window).
- `labs_creatinine.csv`: creatinine trajectories with a bump if AKI.
- `notes.csv`: short synthetic notes containing drug/ADE mentions and negations.
- `spans_bio.csv`: character offsets for DRUG and ADE entities per document (BIO-friendly).
- `cohort.csv`: merged, analysis-ready table with T (0/1), Y outcomes, and W.

## Ground-truth data generating process (DGP)
- Propensity for ACEi depends on HTN, DM2, age, CKD, baseline creatinine (negative association), and SOFA (negative).
- AKI probability depends on T (true log-odds effect ~ 0.15), CKD, baseline creatinine, SOFA, and age.
- Mortality depends on SOFA, AKI, age, and a small T effect.
- Positivity enforced by bounding propensities away from 0/1.

*This dataset is fully synthetic, contains no PHI, and is safe to publish.*
