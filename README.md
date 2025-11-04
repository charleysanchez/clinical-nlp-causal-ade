# Clinical NLP and Causal Inference: Adverse Drug Events and Outcomes

The purpose of this repo is to build and end-to-end pipeline that:

  * (1) extracts adverse drug events (ADEs) and key clinical variables from de-identified clinical notes using LLMs 
  * (2) assembles patient-level cohorts by linking notes to structured EHR
  * (3) estimates causal effects of specific medications/exposures on outcomes (e.g., ICU readmission, in-hospital mortality) using doubly-robust and orthogonal ML estimators (DML/DR-Learner), with rigorous diagnostics (overlap, sensitivity, falsification tests)

Results include an open, reproducible codebase, synthetic demo data, and a research paper.
