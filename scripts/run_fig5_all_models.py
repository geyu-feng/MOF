from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.repro_modules.common import OUTPUT_DIR, build_target_core_feature_table, export_target_core_metal_tables, load_training_table
from scripts.repro_modules.modeling import get_screening_mod_weights
from scripts.repro_modules.workflow import render_all_model_fig5s, run_model_grid_search_cv


def main() -> int:
    OUTPUT_DIR.mkdir(exist_ok=True)
    display_training_raw = load_training_table("calibrated_mixed", "calibrated", "paper_43")
    _, _, tuned_full_pipes, display_training_df = run_model_grid_search_cv(display_training_raw)
    target_core_raw, _ = export_target_core_metal_tables()
    target_core_df = build_target_core_feature_table(
        target_core_raw,
        "calibrated_mixed",
        display_training_df,
        "calibrated",
    )
    render_all_model_fig5s(target_core_df, tuned_full_pipes, display_training_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
