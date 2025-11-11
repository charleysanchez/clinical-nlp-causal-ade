# Clinical NLP and Causal Inference: Adverse Drug Events and Outcomes

The purpose of this repo is to build and end-to-end pipeline that:

  * (1) extracts adverse drug events (ADEs) and key clinical variables from de-identified clinical notes using LLMs 
  * (2) assembles patient-level cohorts by linking notes to structured EHR
  * (3) estimates causal effects of specific medications/exposures on outcomes (e.g., ICU readmission, in-hospital mortality) using doubly-robust and orthogonal ML estimators (DML/DR-Learner), with rigorous diagnostics (overlap, sensitivity, falsification tests)

Results include an open, reproducible codebase, synthetic demo data, and a research paper.

## Datasets and Access

Primary text source: [MIMIC IV v3.1](https://physionet.org/content/mimiciv/3.1/)

 	Johnson, Alistair, et al. "MIMIC-IV" (version 3.1). PhysioNet (2024). RRID:SCR_007345. https://doi.org/10.13026/kpb9-mt58

    Johnson, A.E.W., Bulgarelli, L., Shen, L. et al. MIMIC-IV, a freely accessible electronic health record dataset. Sci Data 10, 1 (2023). https://doi.org/10.1038/s41597-022-01899-x

    Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345.

**Note**: To follow along, this dataset requires specific credentials and completing the CITI training *Data or Specimens Only Research*. Full details for how to do this are attainable through the link above.

### (OPTIONAL) Reproduce on synthetic data (no MIMIC needed)
```bash
make repro
# outputs:
# - data/synth_clinical/notes_hard_v4.csv (and labels)
# - reports/*/metrics.json (BioClinical + ModernBERT)
# - tables/summary.csv


# References

@misc{sounack2025bioclinicalmodernbertstateoftheartlongcontext,
      title={BioClinical ModernBERT: A State-of-the-Art Long-Context Encoder for Biomedical and Clinical NLP}, 
      author={Thomas Sounack and Joshua Davis and Brigitte Durieux and Antoine Chaffin and Tom J. Pollard and Eric Lehman and Alistair E. W. Johnson and Matthew McDermott and Tristan Naumann and Charlotta Lindvall},
      year={2025},
      eprint={2506.10896},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.10896}, 
}

@misc{ling2023bioclinicalbertbertbase,
      title={Bio+Clinical BERT, BERT Base, and CNN Performance Comparison for Predicting Drug-Review Satisfaction}, 
      author={Yue Ling},
      year={2023},
      eprint={2308.03782},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2308.03782}, 
}