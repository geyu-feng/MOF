from __future__ import annotations

from scripts.repro_modules.common import *
from scripts.repro_modules.modeling import *
from scripts.repro_modules.plots import *
from scripts.repro_modules.reporting import *
from scripts.repro_modules.fig4 import *
from scripts.repro_modules.modeling import _run_model_grid_search_cv
from scripts.repro_modules.plots import _export_supplementary_text_sections, _load_supplementary_paragraphs

# --- workflow ---

def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

def isolate_fig3_only_outputs() -> None:
    keep_names = {
        "model_grid_search_cv.csv",
        "model_grid_search_best_per_model.csv",
        "model_metrics_paper_faithful.csv",
        "model_metrics_score_matched.csv",
        "fig3_fitting_effect.png",
    }
    removable_suffixes = {".png", ".pdf", ".svg", ".csv", ".md", ".json", ".log"}
    for path in OUTPUT_DIR.iterdir():
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            continue
        if path.name in keep_names:
            continue
        if path.suffix.lower() in removable_suffixes:
            path.unlink(missing_ok=True)

def pick_display_config(metric_frames: dict[str, object]) -> str:
    scored = {name: score_metric_frame(frame) for name, frame in metric_frames.items()}
    return min(scored, key=scored.get)


def render_all_model_fig5s(
    target_core_df: pd.DataFrame,
    tuned_full_pipes: dict[str, Pipeline],
    display_training_df: pd.DataFrame,
) -> None:
    """Render one Fig. 5 style structure-performance plot for each available trained model."""
    screening_mod_weights = get_screening_mod_weights(display_training_df)
    ordered_models = [*MODEL_ORDER, *ADDITIONAL_MODEL_ORDER]
    for model_name in ordered_models:
        if model_name not in tuned_full_pipes:
            continue
        first_adsorption_df = make_first_adsorption_dataset(
            target_core_df,
            model_name,
            tuned_full_pipes[model_name],
            screening_mod_weights,
        )
        save_fig5_like(
            first_adsorption_df,
            f"fig5_{model_name.lower()}_structure_relationships.png",
            q_column="first_model_q",
            caption_text=f"Fig. 5 ({model_name}). Structure-adsorption capacity relationships of MOFs.",
        )

def run_reproduction(
    skip_supplementary: bool = False,
    fig3_only: bool = False,
    fig5_all_models: bool = True,
) -> int:
    ensure_output_dir()
    if fig3_only:
        isolate_fig3_only_outputs()

    configs = [
        SplitConfig(
            name="paper_faithful",
            mode="sequential",
            descriptor_preset="calibrated_mixed",
            mod_encoding="calibrated",
            group_recipe="paper_43",
        ),
        SplitConfig(
            name="score_matched",
            mode="group_shuffle",
            descriptor_preset="calibrated_mixed",
            mod_encoding="calibrated",
            group_recipe="paper_43",
            random_state=24,
        ),
    ]

    metric_frames: dict[str, object] = {}
    prediction_frames: dict[str, object] = {}
    split_pipes: dict[str, object] = {}
    prepared_splits: dict[str, object] = {}

    display_basis_config = configs[0]
    display_training_raw = load_training_table(
        display_basis_config.descriptor_preset,
        display_basis_config.mod_encoding,
        display_basis_config.group_recipe,
    )
    cv_results_df, cv_best_df, tuned_full_pipes, display_training_df = run_model_grid_search_cv(display_training_raw)
    additional_full_pipes = fit_named_models_on_full_training(
        display_training_df,
        ADDITIONAL_MODEL_ORDER,
        get_additional_model_params(),
    )
    all_full_pipes = {**tuned_full_pipes, **additional_full_pipes}
    tuned_params = {row.model: json.loads(row.params_json) for row in cv_best_df.itertuples(index=False)}
    best_model_name = str(cv_best_df.iloc[0]["model"])
    if not fig3_only:
        strict_group_cv_summary, strict_group_cv_folds, strict_group_cv_groups = evaluate_models_with_group_cv(
            display_training_raw,
            tuned_params,
            model_names=[best_model_name],
        )
        strict_group_cv_summary.to_csv(OUTPUT_DIR / "strict_group_cv_summary.csv", index=False)
        strict_group_cv_folds.to_csv(OUTPUT_DIR / "strict_group_cv_fold_metrics.csv", index=False)
        strict_group_cv_groups.to_csv(OUTPUT_DIR / "strict_group_cv_group_metrics.csv", index=False)
        write_group_cv_report(display_training_raw, strict_group_cv_summary, strict_group_cv_groups)

    for config in configs:
        raw_training_df = load_training_table(config.descriptor_preset, config.mod_encoding, config.group_recipe)
        metrics, predictions, fitted, prepared_split = fit_models_for_split(raw_training_df, config, tuned_params)
        metric_frames[config.name] = metrics
        prediction_frames[config.name] = predictions
        split_pipes[config.name] = fitted
        prepared_splits[config.name] = prepared_split
        metrics.to_csv(OUTPUT_DIR / f"model_metrics_{config.name}.csv", index=False)

    display_config = pick_display_config(metric_frames)
    # Main-text Fig. 3
    save_fig3_like(prediction_frames[display_config], metric_frames[display_config], "fig3_fitting_effect.png")
    additional_metrics, additional_predictions, _ = fit_named_models_on_existing_split(
        prepared_splits[display_config],
        configs[0] if display_config == "paper_faithful" else configs[1],
        ADDITIONAL_MODEL_ORDER,
        get_additional_model_params(),
    )
    additional_metrics.to_csv(OUTPUT_DIR / "model_metrics_additional_models.csv", index=False)
    save_fig3_like(
        additional_predictions,
        additional_metrics,
        "fig3_additional_model_fits.png",
        display_order=ADDITIONAL_MODEL_ORDER,
        caption_text="Fig. 3 (additional models). Fitting effect diagram of CatBoost, ExtraTree, HistGBDT, DecisionTree, Bagging, and LightGBM.",
    )

    if fig3_only:
        return 0

    target_core_raw, _ = export_target_core_metal_tables()
    target_core_df = build_target_core_feature_table(target_core_raw, "calibrated_mixed", display_training_df, "calibrated")
    target_core_df.to_csv(OUTPUT_DIR / "core_target_feature_table_selected_metals.csv", index=False)

    second_model_name = cv_best_df.iloc[1]["model"]
    screening_mod_weights = get_screening_mod_weights(display_training_df)
    first_adsorption_df = make_first_adsorption_dataset(
        target_core_df,
        best_model_name,
        tuned_full_pipes[best_model_name],
        screening_mod_weights,
    )
    first_adsorption_df.to_csv(OUTPUT_DIR / "core_first_adsorption_dataset.csv", index=False)
    initial_screening_df = build_initial_screening_from_first_dataset(first_adsorption_df, top_n=10)
    initial_screening_df.to_csv(OUTPUT_DIR / "initial_screening_generated_70.csv", index=False)
    second_adsorption_df = make_second_adsorption_dataset(
        initial_screening_df,
        second_model_name,
        tuned_full_pipes[second_model_name],
        screening_mod_weights,
    )
    second_adsorption_df.to_csv(OUTPUT_DIR / "second_adsorption_dataset.csv", index=False)

    feature_importance = compute_feature_importance_table(
        tuned_full_pipes[best_model_name],
        display_training_df,
        best_model_name,
    )
    fig2_workflow_counts = {
        "training_rows": len(display_training_raw),
        "model_count": len(MODEL_ORDER),
        "candidate_rows": len(target_core_df),
        "initial_rows": len(initial_screening_df),
    }
    # Main-text Fig. 2 combined panel and its standalone (a)/(c) exports.
    save_fig2_like(
        target_core_df,
        display_training_df,
        feature_importance,
        "fig2_overview.png",
        workflow_counts=fig2_workflow_counts,
    )
    save_fig2a_relationship(target_core_df, display_training_df, "fig2a_relationship.png")
    save_fig2c_feature_importance(feature_importance, "fig2c_feature_importance.png")
    feature_importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    permutation_summary, permutation_detail = compute_repeated_permutation_importance(
        display_training_raw,
        best_model_name,
        tuned_params[best_model_name],
        n_splits=5,
        test_size=0.1,
        perm_repeats=20,
    )
    permutation_summary.to_csv(OUTPUT_DIR / "permutation_importance_summary.csv", index=False)
    permutation_detail.to_csv(OUTPUT_DIR / "permutation_importance_detail.csv", index=False)
    save_permutation_importance_figure(permutation_summary, "fig2c_permutation_importance.png")

    export_supplementary_text_sections()
    # Supplementary Fig. S1-S4
    save_figS1_quantitative_distributions(display_training_raw, "figS1_quantitative_distributions.png")
    save_figS2_qualitative_distributions(display_training_raw, "figS2_qualitative_distributions.png")
    save_figS3_correlation_heatmap(display_training_raw, "figS3_correlation_heatmap.png")
    save_figS4_learning_curve(display_training_raw, best_model_name, tuned_params[best_model_name], "figS4_learning_curve.png")
    # Main-text Fig. 4 and Fig. 5
    render_fig4_artifacts(
        ROOT / "config" / "fig4_config.json",
        raw_training_df=display_training_raw,
        cv_results=cv_results_df,
        best_per_model=cv_best_df,
        fitted_models=tuned_full_pipes,
        prepared_training_df=display_training_df,
    )
    save_fig5_like(first_adsorption_df, "fig5_structure_relationships.png")
    if fig5_all_models:
        render_all_model_fig5s(target_core_df, all_full_pipes, display_training_df)

    if not skip_supplementary:
        # Supplementary Fig. S5 and Fig. S6
        test_df = prepared_splits[display_config]["test"].copy()
        save_combined_supplementary_figures(split_pipes[display_config], test_df, "figS5_beeswarm.png", "figS6_waterfall.png")

    write_summary(
        metric_frames,
        cv_best_df,
        first_adsorption_df,
        initial_screening_df,
        second_adsorption_df,
        display_config,
    )
    return 0


def run_model_grid_search_cv(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Pipeline], pd.DataFrame]:
    return _run_model_grid_search_cv(raw_df, OUTPUT_DIR)


def load_supplementary_paragraphs() -> list[str]:
    return _load_supplementary_paragraphs(ROOT)


def export_supplementary_text_sections() -> None:
    _export_supplementary_text_sections(ROOT)



