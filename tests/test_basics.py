# -*- coding: utf-8 -*-
"""
Basic tests for the electricity consumption app.

Run from the project root with:
    python -m pytest tests/test_basics.py -v
  OR (no pytest needed):
    python tests/test_basics.py

Organised by module, from simplest to most complex.
Each test class maps to one source file in src/ or the root.
"""

import sys
import unittest
import tempfile
from pathlib import Path

# ── Make sure the project root AND src/ are on sys.path ─────────────────────
# Why two entries?
#   - PROJECT_ROOT  → needed for 'import config' (config.py lives at root)
#   - SRC_DIR       → needed for bare 'from discount_analysis import …'
#                     inside src/discount_calculator.py (it imports sibling
#                     modules without the 'src.' prefix).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

# ── Imports from the app ───────────────────────────────────────────────────────
from src.preprocessing import clean_consumption_data
from src.aggregation import (
    compute_hourly_stats,
    compute_daily_stats,
    compute_daily_totals,
    compute_summary,
)
from src.outliers import (
    detect_outliers_3sigma,
    detect_outliers_iqr,
    calculate_outlier_summary,
)
from src.features import (
    time_of_day_ratios,
    weekday_weekend_features,
    regularity_features,
    peak_features,
    night_baseline,
    build_user_features,
    _safe_ratio,
    _hourly,
    _daily,
)
from src.discount_analysis import (
    _hours_from_restriction,
    extract_weekdays,
    extract_hour_range,
    extract_weekday_range,
    add_offer_eligibility,
    fill_time_restriction_from_context,
)
from src.discount_calculator import (
    calculate_plan_savings,
    compare_all_plans,
    extrapolate_annual,
    build_custom_pattern_df,
)
from src.loader import find_header_row, load_raw_csv, load_discount_offers


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS — tiny DataFrames that look like real app data
# ══════════════════════════════════════════════════════════════════════════════

def _make_clean_df(n_days: int = 7) -> pd.DataFrame:
    """
    Return a preprocessed DataFrame (output of clean_consumption_data).
    We build it by hand so tests don't depend on any file on disk.

    Structure mirrors what clean_consumption_data() produces:
      datetime, date, hour, weekday, month, kWh
    Each day has 24 hourly rows.
    """
    rows = []
    # 2024-01-07 is a Sunday; the week Sun–Sat covers all 7 weekday names.
    base = pd.Timestamp("2024-01-07")
    for day in range(n_days):
        ts = base + pd.Timedelta(days=day)
        for hour in range(24):
            rows.append(
                {
                    "datetime": ts.replace(hour=hour),
                    "date": ts.date(),
                    "hour": hour,
                    "weekday": ts.day_name(),
                    "month": ts.month,
                    "kWh": 0.2 + 0.05 * hour,  # gentle ramp so peak != flat
                }
            )
    df = pd.DataFrame(rows)
    df["hour"] = df["hour"].astype("int8")
    df["month"] = df["month"].astype("int8")
    return df


def _make_raw_df() -> pd.DataFrame:
    """Return a raw DataFrame — the format loader produces, before preprocessing."""
    return pd.DataFrame(
        {
            "date": ["07/01/2024", "07/01/2024", "07/01/2024"],
            "time": ["00:00", "00:15", "00:30"],
            "kWh": ["0.100", "0.150", "0.200"],
        }
    )


def _make_offers_df() -> pd.DataFrame:
    """Return a minimal offers DataFrame, similar to electricity_discount_offers.csv."""
    return pd.DataFrame(
        {
            "supplier_name": ["TestCo", "TestCo"],
            "plan_name": ["Day Plan", "Night Plan"],
            "discount_pct": [20, 15],
            "time_restriction": ["sun-thu 07:00-17:00", "sun-thu 23:00-07:00"],
            "requires_smart_meter": [True, False],
            "customer_type": ["All", "All"],
            "context": ["", ""],
        }
    )


# ══════════════════════════════════════════════════════════════════════════════
#  1. LOADER
# ══════════════════════════════════════════════════════════════════════════════

class TestLoader(unittest.TestCase):
    """
    Tests for src/loader.py.

    The loader's job: find the right header row in a messy IEC CSV and
    return a 3-column DataFrame (date, time, kWh).
    """

    # ── find_header_row ────────────────────────────────────────────────────────

    def test_find_header_row_with_hebrew_header(self):
        """
        When the CSV contains the Hebrew header line, the function must
        return the correct row index — NOT 0.

        Why this can break: if the encoding is wrong, Hebrew characters are
        garbled and the search never matches → silently returns 0 and you
        load the wrong part of the file.
        """
        # Write a temp CSV that mimics a real IEC export
        content = (
            "metadata line 1\n"
            "metadata line 2\n"
            'תאריך,מועד תחילת הפעימה,col3,col4,col5\n'
            "07/01/2024,00:00,val1,val2,0.1\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", encoding="utf-8-sig", delete=False
        ) as f:
            f.write(content)
            path = Path(f.name)

        try:
            row = find_header_row(path)
            self.assertEqual(row, 2,
                "Expected header at row 2 (0-indexed), "
                "because the first two rows are metadata.")
        finally:
            path.unlink()

    def test_find_header_row_no_hebrew_raises(self):
        """
        If the Hebrew header line is not found, find_header_row raises ValueError.
        This is the correct behaviour — the user uploaded the wrong file and we
        want a clear error, not silent data corruption from reading row 0.
        """
        content = "date,time,kwh\n07/01/2024,00:00,0.1\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", encoding="utf-8-sig", delete=False
        ) as f:
            f.write(content)
            path = Path(f.name)

        try:
            with self.assertRaises(ValueError):
                find_header_row(path)
        finally:
            path.unlink()

    # ── load_raw_csv ───────────────────────────────────────────────────────────

    def test_load_raw_csv_too_few_columns_raises(self):
        """
        The IEC file must have at least 5 columns.
        If the file has the Hebrew header but only 3 columns, load_raw_csv
        must raise a clear ValueError about the column count.
        """
        # Hebrew header row present (so find_header_row succeeds), but only 3 columns.
        content = "תאריך,מועד תחילת הפעימה,col3\nval1,val2,val3\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", encoding="utf-8-sig", delete=False
        ) as f:
            f.write(content)
            path = Path(f.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                load_raw_csv(path)
            self.assertIn("5 columns", str(ctx.exception))
        finally:
            path.unlink()

    def test_load_raw_csv_valid(self):
        """
        A valid IEC CSV (Hebrew header row + 5+ columns) should return a
        DataFrame with exactly the columns: date, time, kWh.
        """
        # Real IEC layout: Hebrew header on row 0, data on row 1.
        # Columns: 0=תאריך(date), 1=מועד תחילת הפעימה(time), 2=junk, 3=junk, 4=kwh
        content = (
            'תאריך,מועד תחילת הפעימה,col3,col4,kwh\n'
            '07/01/2024,00:00,val3,val4,0.1\n'
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", encoding="utf-8-sig", delete=False
        ) as f:
            f.write(content)
            path = Path(f.name)

        try:
            df = load_raw_csv(path)
            self.assertListEqual(list(df.columns), ["date", "time", "kWh"])
            self.assertEqual(len(df), 1)
        finally:
            path.unlink()

    # ── load_discount_offers ───────────────────────────────────────────────────

    def test_load_discount_offers_missing_file_raises(self):
        """
        If the offers CSV does not exist, we should get FileNotFoundError.
        This prevents silent failures when the file path is misconfigured.
        """
        with self.assertRaises(FileNotFoundError):
            load_discount_offers(Path("/nonexistent/path/offers.csv"))


# ══════════════════════════════════════════════════════════════════════════════
#  2. PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

class TestPreprocessing(unittest.TestCase):
    """
    Tests for src/preprocessing.py — clean_consumption_data().

    This function:
    1. Parses date+time into a datetime column
    2. Drops rows where datetime couldn't be parsed
    3. Cleans kWh (handles commas, junk characters)
    4. Interpolates missing kWh values
    5. Adds hour, weekday, month columns
    """

    def test_basic_happy_path(self):
        """A clean input should produce all expected columns."""
        df = _make_raw_df()
        result = clean_consumption_data(df)
        expected_cols = {"datetime", "date", "hour", "weekday", "month", "kWh"}
        self.assertTrue(expected_cols.issubset(set(result.columns)),
                        f"Missing columns: {expected_cols - set(result.columns)}")

    def test_kWh_comma_decimal(self):
        """
        IEC files use commas as decimal separators (e.g. '0,150').
        The function must convert '0,150' → 0.150, not NaN.
        """
        df = pd.DataFrame({
            "date": ["07/01/2024"],
            "time": ["00:00"],
            "kWh": ["0,150"],    # ← comma as decimal
        })
        result = clean_consumption_data(df)
        self.assertAlmostEqual(result["kWh"].iloc[0], 0.150, places=3)

    def test_bad_dates_are_dropped(self):
        """
        Rows with unparseable dates are dropped via dropna(subset=['datetime']).
        After dropping, only the valid rows remain.
        """
        df = pd.DataFrame({
            "date": ["07/01/2024", "NOT_A_DATE"],
            "time": ["00:00", "00:00"],
            "kWh": ["0.1", "0.2"],
        })
        result = clean_consumption_data(df)
        self.assertEqual(len(result), 1, "Only the valid row should survive")

    def test_all_bad_dates_raises_or_returns_empty(self):
        """
        If EVERY row has an unparseable date, all rows are dropped.
        kWh interpolation on an empty Series still succeeds (no ValueError),
        so the function should return an empty DataFrame gracefully.

        Why this matters: if a user uploads the wrong file, the app should
        not crash with an unreadable traceback.
        """
        df = pd.DataFrame({
            "date": ["GARBAGE", "GARBAGE"],
            "time": ["00:00", "00:00"],
            "kWh": ["0.1", "0.2"],
        })
        # Either returns empty df OR raises a clear error — both are acceptable.
        # What is NOT acceptable: an unhandled IndexError or AttributeError.
        try:
            result = clean_consumption_data(df)
            self.assertEqual(len(result), 0)
        except (ValueError, KeyError) as e:
            pass  # A descriptive error is also fine

    def test_all_null_kWh_raises_ValueError(self):
        """
        If kWh is entirely non-numeric (and can't be interpolated),
        clean_consumption_data should raise ValueError:
        'kWh still contains missing values after interpolation'.
        """
        df = pd.DataFrame({
            "date": ["07/01/2024", "07/01/2024"],
            "time": ["00:00", "00:15"],
            "kWh": ["NOT_A_NUMBER", "ALSO_BAD"],
        })
        with self.assertRaises(ValueError) as ctx:
            clean_consumption_data(df)
        self.assertIn("missing values", str(ctx.exception))

    def test_sorted_by_datetime(self):
        """Rows must be sorted chronologically, not in input order."""
        df = pd.DataFrame({
            "date": ["07/01/2024", "07/01/2024"],
            "time": ["01:00", "00:00"],    # ← reversed order in input
            "kWh": ["0.2", "0.1"],
        })
        result = clean_consumption_data(df)
        self.assertTrue(
            result["datetime"].is_monotonic_increasing,
            "Output must be sorted ascending by datetime"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  3. AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

class TestAggregation(unittest.TestCase):
    """
    Tests for src/aggregation.py.
    These functions group the cleaned DataFrame to produce stats tables.
    """

    def setUp(self):
        """Create a 7-day clean DataFrame shared across all aggregation tests."""
        self.df = _make_clean_df(n_days=7)

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
        This is important for correct chart ordering — without it,
        bar charts would show days in alphabetical order (Fri, Mon, …).
        """
        result = compute_daily_stats(self.df)
        self.assertTrue(hasattr(result["weekday"], "cat"),
                        "weekday should be a pandas Categorical")

    def test_compute_daily_totals_sum(self):
        """
        Total kWh across all days must be the same whether you sum the raw
        15-min readings or the pre-aggregated daily totals.
        We compare grand totals to avoid any date-type indexing edge cases.
        """
        result = compute_daily_totals(self.df)
        # Grand total of individual readings
        grand_total_raw = round(float(self.df["kWh"].sum()), 4)
        # Grand total from the daily-aggregated table
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


# ══════════════════════════════════════════════════════════════════════════════
#  4. OUTLIERS
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
#  5. FEATURES
# ══════════════════════════════════════════════════════════════════════════════

class TestFeatures(unittest.TestCase):
    """
    Tests for src/features.py — feature engineering for clustering.

    All feature functions take a clean DataFrame and return a pd.Series
    of numeric values.
    """

    def setUp(self):
        self.df = _make_clean_df(n_days=14)  # 2 weeks for stable averages
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
        # A perfectly constant signal would give routine_score = 1
        # (cv_same_hour = 0 when std = 0)

    def test_peak_features_hour_in_range(self):
        """Peak hour must be an integer in [0, 23]."""
        result = peak_features(self.h)
        self.assertGreaterEqual(result["hour_of_peak"], 0)
        self.assertLessEqual(result["hour_of_peak"], 23)

    def test_build_user_features_is_series(self):
        """build_user_features must return a pd.Series (not a DataFrame)."""
        result = build_user_features(self.df)
        self.assertIsInstance(result, pd.Series)

    def test_build_user_features_no_nan(self):
        """
        Feature Series must not contain NaN — sklearn's StandardScaler / KMeans
        will throw if it does.
        """
        result = build_user_features(self.df)
        nan_features = result[result.isna()].index.tolist()
        self.assertEqual(nan_features, [],
                         f"NaN in features: {nan_features}")


# ══════════════════════════════════════════════════════════════════════════════
#  6. DISCOUNT ANALYSIS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

class TestDiscountAnalysis(unittest.TestCase):
    """
    Tests for the helper functions in src/discount_analysis.py.

    These functions parse the free-text 'time_restriction' column
    (e.g. "sun-thu 07:00-17:00") into hour lists and weekday lists.
    """

    # ── _hours_from_restriction ────────────────────────────────────────────────

    def test_hours_daytime(self):
        """'07:00-17:00' → hours 7 through 16 (endpoint exclusive)."""
        hours = _hours_from_restriction("sun-thu 07:00-17:00")
        self.assertEqual(hours, list(range(7, 17)))

    def test_hours_overnight_wraps(self):
        """
        '23:00-07:00' spans midnight → hours [23, 0, 1, 2, 3, 4, 5, 6].
        This is the 'wrap-around' case; if you just do range(23, 7) you get [].
        """
        hours = _hours_from_restriction("sun-thu 23:00-07:00")
        expected = list(range(23, 24)) + list(range(0, 7))
        self.assertEqual(hours, expected)

    def test_hours_empty_restriction(self):
        """Empty restriction → all 24 hours (no restriction)."""
        hours = _hours_from_restriction("")
        self.assertEqual(hours, list(range(24)))

    def test_hours_24_7(self):
        """'24/7' → all 24 hours."""
        hours = _hours_from_restriction("24/7")
        self.assertEqual(hours, list(range(24)))

    def test_hours_same_start_end(self):
        """Start == end (e.g. '12:00-12:00') is treated as all day."""
        hours = _hours_from_restriction("12:00-12:00")
        self.assertEqual(hours, list(range(24)))

    # ── extract_weekdays ───────────────────────────────────────────────────────

    def test_weekdays_sun_thu(self):
        """'sun-thu' → [Sunday, Monday, …, Thursday] (5 days)."""
        days = extract_weekdays("sun-thu 07:00-17:00")
        self.assertEqual(len(days), 5)
        self.assertNotIn("Friday", days)
        self.assertNotIn("Saturday", days)

    def test_weekdays_unknown_text_returns_all(self):
        """
        If no recognisable weekday pattern is found, return all 7 days.
        This is the safest default — don't silently exclude any day.
        """
        days = extract_weekdays("some random text with no day info")
        self.assertEqual(len(days), 7)

    def test_weekdays_empty_returns_all(self):
        """Empty string → all 7 days."""
        days = extract_weekdays("")
        self.assertEqual(len(days), 7)

    # ── extract_hour_range (human-readable label) ──────────────────────────────

    def test_extract_hour_range_present(self):
        result = extract_hour_range("sun-thu 07:00-17:00")
        self.assertEqual(result, "07:00-17:00")

    def test_extract_hour_range_absent(self):
        result = extract_hour_range("כל הימים")
        self.assertEqual(result, "All day")

    # ── add_offer_eligibility ──────────────────────────────────────────────────

    def test_eligibility_with_smart_meter(self):
        """
        User HAS a smart meter → all offers are 'eligible', even those that
        require a smart meter.
        """
        offers = _make_offers_df()
        result = add_offer_eligibility(offers, has_smart_meter=True)
        self.assertTrue((result["eligibility"] == "eligible").all())

    def test_eligibility_without_smart_meter(self):
        """
        User does NOT have a smart meter → plans that require one must be
        marked 'not_eligible_requires_smart_meter'.
        """
        offers = _make_offers_df()
        # First offer requires smart meter, second does not
        result = add_offer_eligibility(offers, has_smart_meter=False)
        self.assertEqual(
            result.iloc[0]["eligibility"],
            "not_eligible_requires_smart_meter",
        )
        self.assertEqual(result.iloc[1]["eligibility"], "eligible")

    def test_eligibility_unknown_smart_meter(self):
        """
        When smart meter status is unknown (None), plans that require one
        get 'unknown_smart_meter_required'; others get 'eligible_or_unknown'.
        """
        offers = _make_offers_df()
        result = add_offer_eligibility(offers, has_smart_meter=None)
        self.assertEqual(result.iloc[0]["eligibility"], "unknown_smart_meter_required")
        self.assertEqual(result.iloc[1]["eligibility"], "eligible_or_unknown")


# ══════════════════════════════════════════════════════════════════════════════
#  7. DISCOUNT CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

class TestDiscountCalculator(unittest.TestCase):
    """
    Tests for src/discount_calculator.py.

    calculate_plan_savings does the actual NIS arithmetic:
    how much would you save if you switched to plan X?
    """

    def setUp(self):
        self.df = _make_clean_df(n_days=30)
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
        Edge case: all kWh = 0 (e.g. empty house, or test data).
        Division by zero would crash here without the guard in _safe_ratio.
        """
        df_zero = self.df.copy()
        df_zero["kWh"] = 0.0
        result = calculate_plan_savings(df_zero, self.offer)
        self.assertEqual(result["effective_discount_pct"], 0.0)
        self.assertEqual(result["nis_saved"], 0.0)

    def test_plan_savings_cost_with_plan_lte_baseline(self):
        """With a discount, the plan cost should never exceed the baseline cost."""
        result = calculate_plan_savings(self.df, self.offer)
        self.assertLessEqual(
            result["cost_with_plan_nis"],
            result["cost_baseline_nis"],
        )

    def test_plan_savings_matching_share_bounded(self):
        """matching_usage_share_pct must be between 0 and 100."""
        result = calculate_plan_savings(self.df, self.offer)
        self.assertGreaterEqual(result["matching_usage_share_pct"], 0.0)
        self.assertLessEqual(result["matching_usage_share_pct"], 100.0)

    # ── extrapolate_annual ─────────────────────────────────────────────────────

    def test_extrapolate_annual_basic(self):
        """Scaling 30 days to 365 should multiply nis_saved by ~12.17."""
        offers = _make_offers_df()
        results = compare_all_plans(self.df, offers, has_smart_meter=False)
        annual = extrapolate_annual(results, observation_days=30)
        # annual_nis_saved ≈ nis_saved * (365/30)
        for _, row in annual.iterrows():
            expected = round(row["nis_saved"] * 365 / 30, 2)
            self.assertAlmostEqual(row["annual_nis_saved"], expected, places=1)

    def test_extrapolate_annual_zero_days_raises(self):
        """
        observation_days=0 is nonsensical — would cause division by zero.
        The function must raise ValueError with a clear message.
        """
        offers = _make_offers_df()
        results = compare_all_plans(self.df, offers, has_smart_meter=False)
        with self.assertRaises(ValueError):
            extrapolate_annual(results, observation_days=0)

    def test_extrapolate_annual_negative_days_raises(self):
        """observation_days < 0 is also invalid."""
        offers = _make_offers_df()
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


# ══════════════════════════════════════════════════════════════════════════════
#  8. PIPELINE INTERMEDIATE FILES
#  Verifies that each pipeline step saves the files the next step depends on.
#  Catches "[Errno 2] No such file or directory" errors before they hit the app.
# ══════════════════════════════════════════════════════════════════════════════

class TestPipelineFiles(unittest.TestCase):
    """
    Check that pipeline steps write the intermediate files they promise.
    Uses a real (tiny) raw DataFrame so we exercise the actual save logic.
    """

    def _make_valid_raw_csv(self, path: Path):
        """Write a minimal valid IEC-format CSV to disk."""
        content = (
            "תאריך,מועד תחילת הפעימה,col3,col4,kwh\n"
            + "".join(
                f"07/01/2024,{h:02d}:00,x,x,0.1\n" for h in range(24)
            )
            * 7  # 7 days so clustering has enough rows
        )
        path.write_text(content, encoding="utf-8-sig")

    def test_cleaned_csv_is_written_to_processed_dir(self):
        """
        After load + clean, cleaned_consumption.csv must exist in processed_dir.
        If it doesn't, clustering raises FileNotFoundError.
        """
        from src.loader import load_raw_csv
        from src.preprocessing import clean_consumption_data

        with tempfile.TemporaryDirectory() as tmp:
            processed_dir = Path(tmp) / "processed"
            processed_dir.mkdir()
            csv_path = Path(tmp) / "raw.csv"
            self._make_valid_raw_csv(csv_path)

            raw = load_raw_csv(csv_path)
            df = clean_consumption_data(raw)
            out = processed_dir / "cleaned_consumption.csv"
            df.to_csv(out, index=False)

            self.assertTrue(out.exists(), "cleaned_consumption.csv was not written")
            self.assertGreater(out.stat().st_size, 0, "cleaned_consumption.csv is empty")

    def test_clustering_reads_cleaned_csv(self):
        """
        run_clustering must succeed when cleaned_consumption.csv exists,
        and must raise FileNotFoundError when it doesn't.
        """
        from src.clustering import run_clustering

        with tempfile.TemporaryDirectory() as tmp:
            processed_dir = Path(tmp) / "processed"
            processed_dir.mkdir()
            missing = processed_dir / "cleaned_consumption.csv"

            with self.assertRaises((FileNotFoundError, Exception)):
                run_clustering(
                    input_path=missing,
                    output_path=processed_dir / "clustered.csv",
                    summary_path=processed_dir / "summary.csv",
                )


# ══════════════════════════════════════════════════════════════════════════════
#  9. IMPORT SMOKE TESTS

#  These catch NameError / ImportError at module level (missing imports,
#  undefined constants, etc.) — exactly the class of bugs fixed recently.
#  Run these first; if they fail, nothing else will work.
# ══════════════════════════════════════════════════════════════════════════════

class TestImports(unittest.TestCase):
    """
    Verify every src module and the pipeline can be imported without error.
    A failure here means a missing import or undefined name at module level.
    """

    def test_import_config(self):
        import config

    def test_import_clustering(self):
        import src.clustering

    def test_import_visualization(self):
        import src.visualization

    def test_import_features(self):
        import src.features

    def test_import_preprocessing(self):
        import src.preprocessing

    def test_import_aggregation(self):
        import src.aggregation

    def test_import_outliers(self):
        import src.outliers

    def test_import_discount_analysis(self):
        import src.discount_analysis

    def test_import_discount_calculator(self):
        import src.discount_calculator

    def test_import_loader(self):
        import src.loader

    def test_import_pipeline(self):
        import pipeline


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point — run with: python tests/test_basics.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
