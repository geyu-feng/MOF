from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.reproduce_paper import (
    TRAINING_FEATURES,
    build_output_tag,
    compute_partial_dependence_1d,
    compute_rug_values,
    predict_with_mod_marginalization,
)


class DummyPredictor:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["ci"].to_numpy(dtype=float) + frame["ad"].to_numpy(dtype=float) * 2.0


class DummyScreeningPredictor:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["sa"].to_numpy(dtype=float) + frame["mod_code"].to_numpy(dtype=float) * 10.0


class Fig4ModuleTests(unittest.TestCase):
    def test_build_output_tag_is_stable(self) -> None:
        config = {
            "version": "fig4_refine_v1",
            "dataset": {
                "group_recipe": "paper_43",
                "descriptor_preset": "calibrated_mixed",
                "mod_encoding": "calibrated",
            },
        }
        params = {"n_estimators": 150, "max_depth": 5}
        tag_a = build_output_tag(config, "GBDT", params)
        tag_b = build_output_tag(config, "GBDT", params)
        self.assertEqual(tag_a, tag_b)
        self.assertIn("gbdt", tag_a)

    def test_compute_rug_values_respects_limits(self) -> None:
        frame = pd.DataFrame({"ci": [0.0, 5.0, 10.0, 15.0, np.nan, 30.0]})
        rug = compute_rug_values(frame, "ci", (5.0, 15.0))
        self.assertTrue(np.array_equal(rug, np.asarray([5.0, 10.0, 15.0])))

    def test_compute_partial_dependence_1d_uses_feature_grid(self) -> None:
        base = pd.DataFrame(
            {
                "ci": [1.0, 2.0],
                "ad": [10.0, 10.0],
            }
        )
        grid = np.asarray([0.0, 5.0, 10.0])
        pdp = compute_partial_dependence_1d(DummyPredictor(), base, "ci", grid)
        expected = np.asarray([20.0, 25.0, 30.0])
        np.testing.assert_allclose(pdp, expected)

    def test_predict_with_mod_marginalization_uses_weighted_average(self) -> None:
        frame = pd.DataFrame({feature: [0.0, 0.0] for feature in TRAINING_FEATURES})
        frame["sa"] = [5.0, 7.0]
        preds = predict_with_mod_marginalization(DummyScreeningPredictor(), frame, {0: 0.25, 4: 0.75})
        expected = np.asarray([35.0, 37.0])
        np.testing.assert_allclose(preds, expected)


if __name__ == "__main__":
    unittest.main()
