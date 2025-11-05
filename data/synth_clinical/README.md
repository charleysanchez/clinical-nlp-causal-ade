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

## Dataset details

### `patients.csv`

**Level:** one row per patient

**Purpose**: baseline demographics and comorbidities (confounders)

| Column                | Type          | Meaning                                                     |
| --------------------- | ------------- | ----------------------------------------------------------- |
| `subject_id`          | str           | Unique patient identifier (e.g., `P000123`)                 |
| `age`                 | int           | Age in years at admission (18–90)                           |
| `sex`                 | {F, M}        | Biological sex                                              |
| `race`                | categorical   | Race category (White, Black, Hispanic, Asian, Other)        |
| `ckd`                 | 0/1           | Chronic kidney disease flag                                 |
| `htn`                 | 0/1           | Hypertension flag                                           |
| `dm2`                 | 0/1           | Type 2 diabetes flag                                        |
| `charlson`            | int (0–12)    | Charlson Comorbidity Index (summary of comorbidity burden)  |
| `sofa_admit`          | int (0–20)    | SOFA score on ICU/ED admission — proxy for illness severity |
| `creatinine_baseline` | float (mg/dL) | Baseline serum creatinine prior to admission; higher in CKD |

<br>

---

### `admissions.csv`

**Level:** one row per hospital admission (1 per patient here)

| Column       | Type     | Meaning                               |
| ------------ | -------- | ------------------------------------- |
| `subject_id` | str      | Patient key (joins to `patients.csv`) |
| `hadm_id`    | str      | Hospital admission ID (`H000123`)     |
| `admittime`  | datetime | Admission timestamp                   |
| `dischtime`  | datetime | Discharge timestamp                   |
| `los_days`   | int      | Length of stay in days                |

<br>

---

### `icustays.csv`

**Level:** one row per ICU stay (subset of admissions)

| Column          | Type            | Meaning                              |
| --------------- | --------------- | ------------------------------------ |
| `subject_id`    | str             | Patient ID                           |
| `hadm_id`       | str             | Admission ID                         |
| `icustay_id`    | str             | ICU stay ID (`I000123`)              |
| `icu_admittime` | datetime / NULL | Start of ICU stay                    |
| `icu_dischtime` | datetime / NULL | End of ICU stay                      |
| `icu_los_days`  | int             | ICU length of stay (0 if not in ICU) |

<br>

---

### `medications.csv`

**Level:** one medication record per admission

**Purpose:** defines treatment/exposure variable T (ACE-inhibitor)

| Column                                                                 | Type     | Meaning                                                         |
| ---------------------------------------------------------------------- | -------- | --------------------------------------------------------------- |
| `hadm_id`                                                              | str      | Admission ID                                                    |
| `drug_class`                                                           | str      | Drug class label: `"ACE_INHIBITOR"` or `"NONE"`                 |
| `drug_name`                                                            | str      | Specific agent if given (`lisinopril`, `enalapril`, `ramipril`) |
| `dose_mg`                                                              | int      | Daily dose (mg)                                                 |
| `starttime`                                                            | datetime | First administration time                                       |
| `stoptime`                                                             | datetime | Stop time                            

**Derived variable:** `T = 1` if `drug_class=="ACE_INHIBITOR"`, else 0

<br>

---

### `labs_creatinine.csv`

**Level:** longitudinal lab observations per admission

**Purpose:** shows creatinine trajectories

| Column        | Type     | Meaning                                                     |
| ------------- | -------- | ----------------------------------------------------------- |
| `hadm_id`     | str      | Admission ID                                                |
| `item`        | str      | Lab test name (`"creatinine"`)                              |
| `charttime`   | datetime | Collection time                                             |
| `value_mg_dL` | float    | Serum creatinine (mg/dL); AKI patients exhibit later spikes |

<br>

---

### `notes.csv`

**Level:** one clinical note per admission

**Purpose:** text for NLP task

| Column                                                                              | Type     | Meaning                                                |
| ----------------------------------------------------------------------------------- | -------- | ------------------------------------------------------ |
| `doc_id`                                                                            | str      | Note/document ID (`D000123`)                           |
| `hadm_id`                                                                           | str      | Admission ID                                           |
| `charttime`                                                                         | datetime | Note timestamp                                         |
| `category`                                                                          | str      | Note type (`Progress note`, `Discharge summary`, etc.) |
| `text`                                                                              | str      | Synthetic note text containing:   SOFA, creatinine summary, Medication sentences (e.g., *“Started on lisinopril…”*), ADE mentions or negations (e.g., *“Concern for AKI…”* or *“Denies hypotension…”*)

<br>

---

### `spans_bio.csv`

**Level:** token-span annotations for named entities

**Purpose:** supervision for sequence-labeling models

| Column      | Type | Meaning                              |
| ----------- | ---- | ------------------------------------ |
| `doc_id`    | str  | Note/document key                    |
| `start`     | int  | Start character index of entity span |
| `end`       | int  | End character index (exclusive)      |
| `label`     | str  | Entity type: `"DRUG"` or `"ADE"`     |
| `span_text` | str  | Exact substring from the note text   |

These map directly back to `notes.text[start:end]` and can be converted to BIO or BILOU token labels.

<br>

---

### `cohort.csv`
**Level:** one row per analyzed admission (analysis-ready table)

**Purpose:** join of structured features, treatment T, and outcomes Y

| Column                | Type     | Meaning                                           |
| --------------------- | -------- | ------------------------------------------------- |
| `subject_id`          | str      | Patient ID                                        |
| `hadm_id`             | str      | Admission ID                                      |
| `admittime`           | datetime | Admission time                                    |
| `dischtime`           | datetime | Discharge time                                    |
| `los_days`            | int      | Length of stay                                    |
| `age`                 | int      | From `patients.csv`                               |
| `sex`                 | str      | From `patients.csv`                               |
| `race`                | str      | From `patients.csv`                               |
| `ckd`, `htn`, `dm2`   | 0/1      | Chronic kidney disease, hypertension, diabetes II |
| `charlson`            | int      | Charlson index                                    |
| `sofa_admit`          | int      | Admission SOFA score                              |
| `creatinine_baseline` | float    | Baseline creatinine                               |
| `drug_class`          | str      | `"ACE_INHIBITOR"` or `"NONE"`                     |
| `drug_name`           | str      | Specific ACE-inhibitor name (or blank)            |
| `dose_mg`             | int      | Dose (mg)                                         |
| `AKI`                 | 0/1      | Synthetic acute kidney injury outcome             |
| `mortality_in_hosp`   | 0/1      | In-hospital death                                 |
| `T`                   | 0/1      | Binary treatment indicator (1 = ACEi)             |

<br>

---

## Causal structure

| Role                       | Variable(s)                                                                                | Example use                                     |
| -------------------------- | ------------------------------------------------------------------------------------------ | ----------------------------------------------- |
| **Exposure T**             | `T` (ACE-inhibitor)                                                                        | Treatment variable for causal effect estimation |
| **Outcomes Y**             | `AKI`, `mortality_in_hosp`                                                                 | Dependent variables                             |
| **Confounders W**          | `age`, `sex`, `race`, `ckd`, `htn`, `dm2`, `charlson`, `sofa_admit`, `creatinine_baseline` | Used for propensity & outcome modeling          |
| **Text-derived proxies Z** | e.g. ADE mention flags, negations from `notes.csv`                                         | Optional unstructured covariates                |

<br>

---

## `convert_dataset_to_mds.py`

Courtesy of [BioClinical-ModernBERT](https://github.com/lindvalllab/BioClinical-ModernBERT/blob/main/pretraining_resources/convert_dataset_to_mds.py). Behaves
as filename would suggest. This is important as it follows previous papers' recommendations for increasing throughput of the model.

To run,

```bash
python convert_dataset_to_mds.py \
  --dataset synth_clinical/cohort.csv \
  --row_count N \
  --out_token_counts ../tokenized_datasets/synth_clinical/token_counts.txt \
  --out_dir tokenized_datasets/synth_clinical/

  python convert_dataset_to_mds.py \
  --dataset synth_clinical/icustays.csv \
  --row_count N \
  --out_token_counts ../tokenized_datasets/synth_clinical/token_counts.txt \
  --out_dir tokenized_datasets/synth_clinical/

  ...
  ```

  ## `create_synthetic_data.py`

  Generator script to createthe synthetic data. This has been iterated on and
  developed to create a synthetic dataset which sees a comparable difference in
  performance using BioClinical ModernBERT vs vanilla ModernBERT. 

  To run,

  ```bash
  python data/synth_clinical/create_synthetic_data.py
  ```

  from the base directory of the repo.

*This dataset is fully synthetic, contains no PHI, and is safe to publish.*
