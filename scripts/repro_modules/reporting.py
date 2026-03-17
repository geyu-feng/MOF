from __future__ import annotations

from scripts.repro_modules.common import *

# --- core.py ---

def write_summary(
    metric_frames: dict[str, object],
    cv_best_df,
    first_df,
    initial_screening_df,
    second_df,
    display_config: str,
) -> None:
    lines = [
        "# Refined Reproduction Summary",
        "",
        "## Selected display configuration",
        f"- Config: `{display_config}`",
        "",
    ]
    for config_name, metrics in metric_frames.items():
        best = metrics.sort_values("r2", ascending=False).iloc[0]
        lines.extend(
            [
                f"## {config_name}",
                f"- Best model: `{best['model']}`",
                f"- MAE: `{best['mae']:.2f}` mg/g",
                f"- RMSE: `{best['rmse']:.2f}` mg/g",
                f"- R2: `{best['r2']:.2f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Model selection",
            f"- Best model: `{cv_best_df.iloc[0]['model']}`",
            f"- Second-best model: `{cv_best_df.iloc[1]['model']}`",
            "",
            "## CoRE-MOF first adsorption dataset",
            f"- Candidate rows: `{len(first_df)}`",
            f"- Source file: `{first_df['source_file'].iloc[0] if not first_df.empty else 'N/A'}`",
            f"- ModM handling: `{SCREENING_MOD_STRATEGY}`",
            f"- Top 10 by first model: `{', '.join(first_df.head(10)['cif_file'].tolist()) if not first_df.empty else 'N/A'}`",
            "",
            "## Initial screening(70)",
            f"- Rows: `{len(initial_screening_df)}`",
            f"- Metal categories: `{', '.join(TARGET_CORE_METALS)}`",
            "",
            "## Second adsorption dataset",
            f"- Top 10 by second model: `{', '.join(second_df.head(10)['cif_file'].tolist()) if not second_df.empty else 'N/A'}`",
            "",
            "## Remaining gaps",
            "- The exact IC/AR/Pol/Ele lookup table is not published, so the calibrated descriptor preset is an informed reconstruction.",
            "- The local pipeline now uses the 5382 single-metal CoRE subset extracted from the official 14142-entry ASR table, which matches the user's requested candidate set but is still not the paper's unpublished 3833 subset.",
            "- Model selection follows grouped cross-validation with fold-wise train-only preprocessing on the current local training dataset; holdout-style grouped evaluation is retained separately for display figures.",
        ]
    )
    (OUTPUT_DIR / "reproduction_summary.md").write_text("\n".join(lines), encoding="utf-8")

