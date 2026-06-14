import unittest

from src.aggregation import (
    compute_hourly_stats,
    compute_daily_stats,
    compute_daily_totals,
    compute_summary,
)
from helpers import make_clean_df


class TestAggregation(unittest.TestCase):
    """
    Tests for src/aggregation.py.
    These functions group the cleaned DataFrame to produce stats tables.
    """

    def setUp(self):
        self.df = make_clean_df(n_days=7)

    def test_compute_hourly_stats_columns(self):
        """Must return the expected aggregate columns."""
        result = compute_hourly_stats(self.df)
        for col in ["weekday", "hour", "avg_kWh", "std_kWh", "min_kWh", "max_kWh"]:
            self.assertIn(col, result.columns)

    def test_compute_hourly_stats_hour_range(self):
        """Hours should be 0–23 only (no negative or >23 values)."""
        result = compute_hourly_stats(self.df)
        self.assertTrue((result["hour"] >= 0).all() and (result["hour"] <= 23).all())

    def test_compute_daily_stats_weekday_categorical(self):
        """
        Weekday column must be a Categorical ordered Sun → Sat.
        Without this, bar charts show days in alphabetical order (Fri, Mon, …).
        """
        result = compute_daily_stats(self.df)
        self.assertTrue(hasattr(result["weekday"], "cat"),
                        "weekday should be a pandas Categorical")

    def test_compute_daily_totals_sum(self):
        """
        Total kWh across all days must be the same whether you sum the raw
        readings or the pre-aggregated daily totals.
        """
        result = compute_daily_totals(self.df)
        grand_total_raw = round(float(self.df["kWh"].sum()), 4)
        grand_total_agg = round(float(result["daily_kWh"].sum()), 4)
        self.assertAlmostEqual(grand_total_agg, grand_total_raw, places=3)

    def test_compute_summary_keys(self):
        """compute_summary must return a dict with all expected keys."""
        hourly = compute_hourly_stats(self.df)
        daily = compute_daily_stats(self.df)
        summary = compute_summary(self.df, hourly, daily)
        for key in ["start_date", "end_date", "records", "days",
                    "total_kWh", "avg_daily_kWh", "peak_weekday", "peak_hour"]:
            self.assertIn(key, summary, f"Missing key: {key}")

    def test_compute_summary_days_count(self):
        """days in summary should equal the number of unique dates."""
        hourly = compute_hourly_stats(self.df)
        daily = compute_daily_stats(self.df)
        summary = compute_summary(self.df, hourly, daily)
        self.assertEqual(summary["days"], self.df["date"].nunique())


if __name__ == "__main__":
    unittest.main(verbosity=2)
