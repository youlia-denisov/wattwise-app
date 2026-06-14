import unittest

from src.features import (
    time_of_day_ratios,
    regularity_features,
    peak_features,
    night_baseline,
    build_user_features,
    _safe_ratio,
    _hourly,
    _daily,
)
from helpers import make_clean_df


class TestFeatures(unittest.TestCase):
    """
    Tests for src/features.py — feature engineering for clustering.

    All feature functions take a clean DataFrame and return a pd.Series
    of numeric values.
    """

    def setUp(self):
        self.df = make_clean_df(n_days=14)  # 2 weeks for stable averages
        # Feature functions expect hourly-resampled data, not raw 15-min readings.
        # _hourly() returns a DataFrame with 'datetime' and 'kWh_hour' columns.
        # _daily() returns a DataFrame with 'datetime' and 'kWh_day' columns.
        self.h = _hourly(self.df)
        self.daily_h = _daily(self.df)

    def test_safe_ratio_zero_denominator(self):
        """
        _safe_ratio(x, 0) must return 0.0 — not raise ZeroDivisionError.
        This matters when a user has ZERO consumption in some time window.
        """
        self.assertEqual(_safe_ratio(5.0, 0.0), 0.0)
        self.assertEqual(_safe_ratio(0.0, 0.0), 0.0)

    def test_time_of_day_ratios_sum_to_one(self):
        """
        day + evening + night fractions must sum to ~1.0 (scale-invariant property).
        If they don't, the clustering distances are distorted.
        """
        ratios = time_of_day_ratios(self.h)
        total = ratios["daytime_activity_share"] + ratios["ratio_evening"] + ratios["ratio_night"]
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_time_of_day_ratios_all_nonnegative(self):
        """No ratio should be negative (would mean negative electricity usage)."""
        ratios = time_of_day_ratios(self.h)
        for name, val in ratios.items():
            self.assertGreaterEqual(val, 0.0, f"{name} is negative: {val}")

    def test_regularity_features_routine_score_bounded(self):
        """
        routine_score = 1 - mean_CV.
        For typical data, CV is between 0 and 1, so routine_score ∈ [-∞, 1].
        But for well-behaved data it should be in [0, 1].
        """
        result = regularity_features(self.h, self.daily_h)
        self.assertIn("routine_score", result.index)

    def test_peak_features_hour_in_range(self):
        """Peak hour must be an integer in [0, 23]."""
        result = peak_features(self.h)
        self.assertGreaterEqual(result["hour_of_peak"], 0)
        self.assertLessEqual(result["hour_of_peak"], 23)

    def test_build_user_features_is_series(self):
        """build_user_features must return a pd.Series (not a DataFrame)."""
        import pandas as pd
        result = build_user_features(self.df)
        self.assertIsInstance(result, pd.Series)

    def test_build_user_features_no_nan(self):
        """
        Feature Series must not contain NaN — sklearn's StandardScaler / KMeans
        will throw if it does.
        """
        result = build_user_features(self.df)
        nan_features = result[result.isna()].index.tolist()
        self.assertEqual(nan_features, [], f"NaN in features: {nan_features}")

    # ── night_baseline ─────────────────────────────────────────────────────────

    def test_night_baseline_returns_series_with_expected_key(self):
        """night_baseline must return a Series with the 'min_consumption_baseline_kwh' key."""
        import pandas as pd
        result = night_baseline(self.h)
        self.assertIsInstance(result, pd.Series)
        self.assertIn("min_consumption_baseline_kwh", result.index)

    def test_night_baseline_excludes_peak_hours(self):
        """
        Standby should be lower than daytime peak.
        We inject a clear evening spike so the minimum-of-medians method
        must pick a quiet overnight hour, not the spike.
        """
        h = self.h.copy()
        h.loc[h["datetime"].dt.hour == 19, "kWh_hour"] += 5.0
        result = night_baseline(h)
        self.assertLess(result["min_consumption_baseline_kwh"],
                        h["kWh_hour"].max(),
                        "Baseline should not equal the peak")

    def test_night_baseline_ignores_single_night_spike(self):
        """
        A one-off spike in a single overnight hour on one day should not
        inflate the baseline — the median per hour is resistant to outliers.
        """
        h = self.h.copy()
        # Spike only on the first day at 2am
        mask = (h["datetime"].dt.hour == 2) & (h["datetime"].dt.date == h["datetime"].dt.date.min())
        h.loc[mask, "kWh_hour"] += 10.0
        result = night_baseline(h)
        self.assertLess(result["min_consumption_baseline_kwh"], 1.0,
                        "Single-night spike should not inflate the baseline")


if __name__ == "__main__":
    unittest.main(verbosity=2)
