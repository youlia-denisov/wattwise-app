import unittest
import pandas as pd

from src.discount_calculator import (
    calculate_plan_savings,
    compare_all_plans,
    extrapolate_annual,
    build_custom_pattern_df,
)
from helpers import make_clean_df, make_offers_df


class TestDiscountCalculator(unittest.TestCase):
    """
    Tests for src/discount_calculator.py.

    calculate_plan_savings does the actual NIS arithmetic:
    how much would you save if you switched to plan X?
    """

    def setUp(self):
        self.df = make_clean_df(n_days=30)
        self.offer = pd.Series({
            "supplier_name": "TestCo",
            "plan_name": "Day Plan",
            "discount_pct": 20,
            "time_restriction": "sun-thu 07:00-17:00",
            "requires_smart_meter": True,
        })

    # ── calculate_plan_savings ─────────────────────────────────────────────────

    def test_plan_savings_returns_expected_keys(self):
        """Result dict must contain all keys the UI and reports rely on."""
        result = calculate_plan_savings(self.df, self.offer)
        for key in [
            "kwh_in_window", "kwh_outside_window", "total_kwh",
            "cost_baseline_nis", "cost_with_plan_nis",
            "nis_saved", "effective_discount_pct", "matching_usage_share_pct"
        ]:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_plan_savings_nis_saved_positive(self):
        """A 20% discount on any usage should always save money (≥ 0 NIS)."""
        result = calculate_plan_savings(self.df, self.offer)
        self.assertGreaterEqual(result["nis_saved"], 0.0)

    def test_plan_savings_zero_consumption_no_crash(self):
        """
        Edge case: all kWh = 0. Division by zero would crash here without
        the guard in _safe_ratio.
        """
        df_zero = self.df.copy()
        df_zero["kWh"] = 0.0
        result = calculate_plan_savings(df_zero, self.offer)
        self.assertEqual(result["effective_discount_pct"], 0.0)
        self.assertEqual(result["nis_saved"], 0.0)

    def test_plan_savings_cost_with_plan_lte_baseline(self):
        """With a discount, the plan cost should never exceed the baseline cost."""
        result = calculate_plan_savings(self.df, self.offer)
        self.assertLessEqual(result["cost_with_plan_nis"], result["cost_baseline_nis"])

    def test_plan_savings_matching_share_bounded(self):
        """matching_usage_share_pct must be between 0 and 100."""
        result = calculate_plan_savings(self.df, self.offer)
        self.assertGreaterEqual(result["matching_usage_share_pct"], 0.0)
        self.assertLessEqual(result["matching_usage_share_pct"], 100.0)

    # ── extrapolate_annual ─────────────────────────────────────────────────────

    def test_extrapolate_annual_basic(self):
        """Scaling 30 days to 365 should multiply nis_saved by ~12.17."""
        offers = make_offers_df()
        results = compare_all_plans(self.df, offers, has_smart_meter=False)
        annual = extrapolate_annual(results, observation_days=30)
        for _, row in annual.iterrows():
            expected = round(row["nis_saved"] * 365 / 30, 2)
            self.assertAlmostEqual(row["annual_nis_saved"], expected, places=1)

    def test_extrapolate_annual_zero_days_raises(self):
        """observation_days=0 must raise ValueError (would cause division by zero)."""
        offers = make_offers_df()
        results = compare_all_plans(self.df, offers, has_smart_meter=False)
        with self.assertRaises(ValueError):
            extrapolate_annual(results, observation_days=0)

    def test_extrapolate_annual_negative_days_raises(self):
        """observation_days < 0 is also invalid."""
        offers = make_offers_df()
        results = compare_all_plans(self.df, offers, has_smart_meter=False)
        with self.assertRaises(ValueError):
            extrapolate_annual(results, observation_days=-5)

    # ── build_custom_pattern_df ────────────────────────────────────────────────

    def test_build_custom_pattern_df_basic(self):
        """Should return a DataFrame with columns hour, weekday, kWh."""
        df = build_custom_pattern_df(
            monthly_kwh=300,
            pct_weekday_day=40,
            pct_weekday_evening=30,
            pct_weekday_night=10,
            pct_weekend=20,
        )
        for col in ["hour", "weekday", "kWh"]:
            self.assertIn(col, df.columns)
        self.assertGreater(len(df), 0)

    def test_build_custom_pattern_df_total_kwh(self):
        """The kWh values must sum to monthly_kwh (after normalisation)."""
        monthly = 300.0
        df = build_custom_pattern_df(
            monthly_kwh=monthly,
            pct_weekday_day=40,
            pct_weekday_evening=30,
            pct_weekday_night=10,
            pct_weekend=20,
        )
        self.assertAlmostEqual(df["kWh"].sum(), monthly, places=3)

    def test_build_custom_pattern_df_all_zeros_raises(self):
        """
        All percentages = 0 is invalid (no usage anywhere).
        The function must raise ValueError to prevent division by zero
        in the normalisation step.
        """
        with self.assertRaises(ValueError):
            build_custom_pattern_df(
                monthly_kwh=300,
                pct_weekday_day=0,
                pct_weekday_evening=0,
                pct_weekday_night=0,
                pct_weekend=0,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
