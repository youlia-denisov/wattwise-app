import unittest
import numpy as np
import pandas as pd

from src.outliers import (
    detect_outliers_3sigma,
    detect_outliers_iqr,
    calculate_outlier_summary,
)


class TestOutliers(unittest.TestCase):
    """
    Tests for src/outliers.py.

    Outlier detection flags kWh readings that are unusually high or low.
    Two methods: 3-sigma (mean ± 3 std) and IQR (Q1 - 1.5*IQR … Q3 + 1.5*IQR).
    """

    def _make_df_with_spike(self):
        """Normal distribution with one obvious spike."""
        np.random.seed(42)
        kWh = np.append(np.random.normal(loc=1.0, scale=0.1, size=99), [50.0])
        return pd.DataFrame({
            "kWh": kWh,
            "datetime": pd.date_range("2024-01-01", periods=100, freq="15min"),
            "hour": [t.hour for t in pd.date_range("2024-01-01", periods=100, freq="15min")],
        })

    def test_3sigma_detects_spike(self):
        """The 50 kWh spike is >> 3 std from mean and must be flagged."""
        df = self._make_df_with_spike()
        outliers = detect_outliers_3sigma(df)
        self.assertGreater(len(outliers), 0, "Expected at least one outlier")
        self.assertIn(50.0, outliers["kWh"].values)

    def test_iqr_detects_spike(self):
        """IQR should also catch the spike."""
        df = self._make_df_with_spike()
        outliers = detect_outliers_iqr(df)
        self.assertGreater(len(outliers), 0)
        self.assertIn(50.0, outliers["kWh"].values)

    def test_3sigma_no_outliers_in_flat_data(self):
        """All-identical values → std = 0 → no value is > 3 std away."""
        df = pd.DataFrame({"kWh": [1.0] * 100})
        outliers = detect_outliers_3sigma(df)
        self.assertEqual(len(outliers), 0)

    def test_outlier_summary_columns(self):
        """calculate_outlier_summary should return a 2-row DataFrame."""
        df = self._make_df_with_spike()
        o3 = detect_outliers_3sigma(df)
        oq = detect_outliers_iqr(df)
        summary = calculate_outlier_summary(df, o3, oq)
        self.assertEqual(len(summary), 2)
        self.assertIn("method", summary.columns)
        self.assertIn("outlier_percentage", summary.columns)

    def test_outlier_summary_percentages_bounded(self):
        """Percentages must be between 0 and 100."""
        df = self._make_df_with_spike()
        o3 = detect_outliers_3sigma(df)
        oq = detect_outliers_iqr(df)
        summary = calculate_outlier_summary(df, o3, oq)
        self.assertTrue((summary["outlier_percentage"] >= 0).all())
        self.assertTrue((summary["outlier_percentage"] <= 100).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
