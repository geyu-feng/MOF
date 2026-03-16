# Fig. 4 Refine Report

## Diagnostic
- The legacy Fig. 4 routine coupled PDP calculation and plotting in a single function, which made paper-style tuning fragile.
- The legacy single-variable panels used histogram-like density bands instead of rug marks, so the training-value distribution looked heavier than the paper.
- The legacy 2D panel appended a colorbar and used generic subplot spacing, both of which drifted away from the paper layout.
- Axis limits for CI, AD, Time, pH, and Tem were partly auto-expanded from the local data, so panel framing and whitespace were inconsistent with the paper.
- Output files did not encode the dataset/model/parameter version, and the refinement path had no dedicated logging or tests.

## Local Changes
- Added a dedicated `scripts/fig4_module.py` module and `scripts/make_fig4_only.py` entrypoint so Fig. 4 can be regenerated without rerunning the whole paper workflow.
- Moved Fig. 4 plotting parameters, axis limits, contour density, output formats, and validation settings into `config/fig4_config.json`.
- Split PDP computation from plotting into standalone `compute_*` and `plot_*` functions.
- Replaced density bands with rug marks derived from the prepared 801-row training distribution.
- Removed the 2D colorbar and locked the CI/AD/Time/pH/Tem display ranges to paper-style bounds from the current local training workflow.
- Added uncertainty CSV outputs from a CV ensemble and grouped validation summaries at DOI level and material-family-proxy level.

## Model Roles
- Model selection model: `GBDT` chosen by `run_model_grid_search_cv()` on the current 801-row training table.
- Display model: `GBDT` fitted once on the full prepared training table with params `{"learning_rate": 0.1, "max_depth": 5, "min_samples_leaf": 2, "n_estimators": 150}`.
- Deployment model: the same `GBDT` object family is reused downstream in the main screening workflow; this refinement script does not retrain a different display-only estimator.

## Final Figure
- Final Fig. 4 uses the current CV first-ranked model: `GBDT`.
- Output tag: `paper_43_calibrated_mixed_calibrated_gbdt_b4d4eb1a`.

## Comparison Models
- Additional reference renders were written for: `RF, XGB`.
- The source paper does not explicitly state which trained model produced Fig. 4, so using the current CV winner remains an engineering choice rather than a proven paper fact.

## External Validation
- `doi_level`: R2 `0.020`, MAE `42.141`, RMSE `62.826`, groups `24`, folds `5`
- `material_family_level`: R2 `0.204`, MAE `36.264`, RMSE `56.608`, groups `43`, folds `5`

## Remaining Uncertainty
- The exact visual palette and contour level policy are not described in the paper, so the chosen map is a paper-style approximation from the screenshot.
- The published article does not provide raw PDP arrays, so line shape agreement is limited by the local model and the paper screenshot only.
- The material-family validation uses the repository's `group_id` proxy derived from metal, modification method, and structural fields; it is not an author-supplied family label.