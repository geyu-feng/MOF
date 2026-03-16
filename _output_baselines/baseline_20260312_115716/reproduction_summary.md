# Refined Reproduction Summary

## Selected display configuration
- Config: `score_matched`

## paper_faithful
- Best model: `RF`
- MAE: `18.26` mg/g
- RMSE: `22.12` mg/g
- R2: `0.74`

## score_matched
- Best model: `SVR`
- MAE: `14.96` mg/g
- RMSE: `21.82` mg/g
- R2: `0.84`

## Model selection
- Best model: `GBDT`
- Second-best model: `KNN`

## CoRE-MOF first adsorption dataset
- Candidate rows: `5382`
- Source file: `2019-11-01-ASR-internal_14142.csv`
- Top 10 by first model: `BIBBUL_clean, FIHHOV_clean, AVUPIR_SL, ITONAI_clean, OCIHIS_clean, FEPQEW_clean, ZONBAH_clean, JIVSIQ_clean, GEDSIT_clean, GEDRIS_clean`

## Initial screening(70)
- Rows: `70`
- Metal categories: `Zn, In, Fe, Cu, Ti, Zr, Nd`

## Second adsorption dataset
- Top 10 by second model: `TIDQIG_clean, VIXGAM01_clean, MAPFOY_clean, XIWYEH_clean, WOXROQ_clean, JIVSIQ_clean, WOYJID_clean, RAVCUO_clean, BUSSEP_clean, magnetochemistry3010001_PF290NdNO3_clean`

## Remaining gaps
- The exact IC/AR/Pol/Ele lookup table is not published, so the calibrated descriptor preset is an informed reconstruction.
- The local pipeline now uses the 5382 single-metal CoRE subset extracted from the official 14142-entry ASR table, which matches the user's requested candidate set but is still not the paper's unpublished 3833 subset.
- Model selection follows exhaustive parameter search with 10-fold cross-validation on the fully prepared 801-row training dataset; holdout-style grouped evaluation is retained separately for display figures.