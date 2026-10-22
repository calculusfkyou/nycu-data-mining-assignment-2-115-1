# Model methodology

The authoritative settings and current results are in the root README, `models.py`, and `tables/`.

- T1: weighted error count / consecutive tool repetition, fixed cutoff 1.
- T2: 24-16-6-16-24 autoencoder, normal-only scaler and 32 successful fitting rows; AdamW, 150 epochs.
- T3: XGBoost fitted on 82 mixed training rows, never validation rows.
- Validation: 40 disjoint rows, threshold selection only. Test: 162 shared IDs.
- AP, ROC-AUC, F1, F2, precision, recall, warmed batched ms/sample, and training seconds are reported over five seeds. SD uses ddof=1.
- Seed t-tests measure optimization variation. Paired row bootstrap intervals use the seed-42 model and do not prove equivalence.

Run `experiment.py` → `ablations.py` → `error_analysis.py` → `paper_figures.py` from the root. Full commands and environment details are in the root README.

Contamination replaces 0/1/2/3 of 32 fitting rows with training-partition failures; the normal scaler is held fixed. Actual fractions are 0/3.125/6.25/9.375%, not a fictitious 1%. `ablation_fitting_ids.json` records those rows. Feature masking keeps 24 inputs and the network shape fixed. Threshold ablation reuses identical scores for F1, F2 and Normal95 calibration.

Error cases are actual rows selected by model outcomes; final-action/diff excerpts are archived. Semantic causes beyond those observations are not asserted as facts. Impact projections use measured confusion counts and charge for all alerts, including true alerts. The assumed costs favor reviewing every run; no production saving is established.
