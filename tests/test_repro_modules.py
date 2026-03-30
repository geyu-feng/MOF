from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.repro_modules.common import encode_modification_codes, prepare_model_table
from scripts.repro_modules.modeling import _run_model_grid_search_cv


class ReproModuleTests(unittest.TestCase):
    def test_encode_modification_codes_handles_unknown_values(self) -> None:
        mods = pd.Series(["Unmodified", "Experimental", None, ""])
        codes, mapping, unknown = encode_modification_codes(mods, "calibrated")
        self.assertEqual(codes.notna().sum(), 4)
        self.assertIn("Experimental", mapping)
        self.assertIn("Unknown", mapping)
        self.assertEqual(sorted(unknown), ["Experimental", "Unknown"])

    def test_holdout_requires_enough_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "group_id": [0, 0],
                "q": [1.0, 2.0],
                "mod_code": [0, 0],
                "ionic_charge": [1.0, 1.0],
                "atomic_radius": [1.0, 1.0],
                "polarizability": [1.0, 1.0],
                "electronegativity": [1.0, 1.0],
                "sa": [1.0, 2.0],
                "mpd": [1.0, 2.0],
                "pd": [1.0, 2.0],
                "pv": [1.0, 2.0],
                "ci": [1.0, 2.0],
                "ad": [1.0, 2.0],
                "time": [1.0, 2.0],
                "ph": [7.0, 7.0],
                "temp": [298.0, 298.0],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "requires at least .* rows"):
                _run_model_grid_search_cv(frame, Path(temp_dir))

    def test_prepare_model_table_uses_stratified_and_positive_imputation(self) -> None:
        frame = pd.DataFrame(
            {
                "metal": ["Zr", "Zr", "Zr", "Cu", "Cu", "Cu"],
                "modification": ["Unmodified", "Unmodified", "Unmodified", "Composite", "Composite", "Composite"],
                "doi": ["a", "a", "a", "b", "b", "b"],
                "group_id": [0, 0, 0, 1, 1, 1],
                "q": [100.0, 120.0, 110.0, 80.0, 90.0, 85.0],
                "mod_code": [0, 0, 0, 3, 3, 3],
                "ionic_charge": [4.0, 4.0, 4.0, 2.0, 2.0, 2.0],
                "atomic_radius": [160.0, 160.0, 160.0, 128.0, 128.0, 128.0],
                "polarizability": [11.1, 11.1, 11.1, 6.1, 6.1, 6.1],
                "electronegativity": [1.33, 1.33, 1.33, 1.90, 1.90, 1.90],
                "sa": [500.0, 550.0, 520.0, 300.0, 320.0, 310.0],
                "mpd": [4.0, 4.2, 4.1, 3.0, 3.1, 3.05],
                "pd": [1.5, 1.6, None, 1.2, 1.25, 1.22],
                "pv": [0.50, 0.55, None, 0.30, 0.31, 0.29],
                "ci": [100.0, 110.0, None, 200.0, 220.0, 210.0],
                "ad": [500.0, 520.0, None, 800.0, 820.0, 810.0],
                "time": [60.0, 60.0, 60.0, 120.0, 120.0, 120.0],
                "ph": [5.0, 5.0, 5.0, 6.0, 6.0, 6.0],
                "temp": [298.0, 298.0, 298.0, 298.0, 298.0, 298.0],
            }
        )
        prepared = prepare_model_table(frame, fit_df=frame)
        self.assertFalse(prepared[["ci", "ad", "pd", "pv"]].isna().any().any())
        self.assertTrue((prepared[["ci", "ad", "pd", "pv"]] > 0).all().all())
        self.assertAlmostEqual(float(prepared.loc[2, "ci"]), 105.0, places=6)
        self.assertAlmostEqual(float(prepared.loc[2, "ad"]), 510.0, places=6)


if __name__ == "__main__":
    unittest.main()
