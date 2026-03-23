from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.repro_modules.common import encode_modification_codes
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
            with self.assertRaisesRegex(ValueError, "requires at least 6 rows"):
                _run_model_grid_search_cv(frame, Path(temp_dir))


if __name__ == "__main__":
    unittest.main()
