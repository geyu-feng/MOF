from __future__ import annotations

from scripts.repro_modules.common import *
from scripts.repro_modules.modeling import *
from scripts.repro_modules.plots import *
from scripts.repro_modules.reporting import *
from scripts.repro_modules.fig4 import *

# --- workflow ---

def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

def pick_display_config(metric_frames: dict[str, object]) -> str:
    scored = {name: score_metric_frame(frame) for name, frame in metric_frames.items()}
    return min(scored, key=scored.get)

def run_reproduction(skip_supplementary: bool = False) -> int:
    ensure_output_dir()
    save_reference_pages()

    configs = [
        SplitConfig(
            name="paper_faithful",
            mode="sequential",
            descriptor_preset="calibrated_mixed",
            mod_encoding="calibrated",
            group_recipe="metal_family",
            train_group_count=39,
        ),
        SplitConfig(
            name="score_matched",
            mode="group_shuffle",
            descriptor_preset="calibrated_mixed",
            mod_encoding="calibrated",
            group_recipe="metal_family",
            random_state=24,
        ),
    ]

    metric_frames: dict[str, object] = {}
    prediction_frames: dict[str, object] = {}
    split_pipes: dict[str, object] = {}
    prepared_splits: dict[str, object] = {}

    display_training_raw = load_training_table("calibrated_mixed", "calibrated", "metal_family")
    _, cv_best_df, tuned_full_pipes, display_training_df = run_model_grid_search_cv(display_training_raw)
    tuned_params = {row.model: json.loads(row.params_json) for row in cv_best_df.itertuples(index=False)}
    strict_group_cv_summary, strict_group_cv_folds, strict_group_cv_groups = evaluate_models_with_group_cv(
        display_training_raw,
        tuned_params,
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

    target_core_raw, _ = export_target_core_metal_tables()
    target_core_df = build_target_core_feature_table(target_core_raw, "calibrated_mixed", display_training_df, "calibrated")
    target_core_df.to_csv(OUTPUT_DIR / "core_target_feature_table_5382.csv", index=False)

    best_model_name = cv_best_df.iloc[0]["model"]
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
    save_fig2_like(target_core_df, display_training_df, feature_importance, "fig2_overview.png")
    save_fig2a_relationship(target_core_df, display_training_df, "fig2a_relationship.png")
    save_fig2c_feature_importance(feature_importance, "fig2c_feature_importance.png")
    feature_importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    export_supplementary_text_sections()
    save_figS1_quantitative_distributions(display_training_raw, "figS1_quantitative_distributions.png")
    save_figS2_qualitative_distributions(display_training_raw, "figS2_qualitative_distributions.png")
    save_figS3_correlation_heatmap(display_training_raw, "figS3_correlation_heatmap.png")
    save_figS4_learning_curve(display_training_raw, best_model_name, tuned_params[best_model_name], "figS4_learning_curve.png")

    save_fig3_like(prediction_frames[display_config], metric_frames[display_config], "fig3_fitting_effect.png")
    render_fig4_artifacts(ROOT / "config" / "fig4_config.json")
    save_fig5_like(first_adsorption_df, "fig5_structure_relationships.png")

    if not skip_supplementary:
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
    update_readme()
    return 0


def run_model_grid_search_cv(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Pipeline], pd.DataFrame]:
    return _run_model_grid_search_cv(raw_df, OUTPUT_DIR)


def load_supplementary_paragraphs() -> list[str]:
    return _load_supplementary_paragraphs(ROOT)


def export_supplementary_text_sections() -> None:
    _export_supplementary_text_sections(ROOT)


def save_reference_pages() -> None:
    _save_reference_pages(ROOT)


def save_doc_page_to_text_image(pdf_name_pattern: str, page_number: int, out_name: str) -> Path:
    return _save_doc_page_to_text_image(ROOT, pdf_name_pattern, page_number, out_name)



