import unittest
import pandas as pd

from src.preprocessing import clean_consumption_data
from helpers import make_raw_df


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
        df = make_raw_df()
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
            "kWh": ["0,150"],
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
        The function should return an empty DataFrame gracefully — not crash.
        """
        df = pd.DataFrame({
            "date": ["GARBAGE", "GARBAGE"],
            "time": ["00:00", "00:00"],
            "kWh": ["0.1", "0.2"],
        })
        try:
            result = clean_consumption_data(df)
            self.assertEqual(len(result), 0)
        except (ValueError, KeyError):
            pass  # A descriptive error is also acceptable

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
            "time": ["01:00", "00:00"],    # reversed order in input
            "kWh": ["0.2", "0.1"],
        })
        result = clean_consumption_data(df)
        self.assertTrue(
            result["datetime"].is_monotonic_increasing,
            "Output must be sorted ascending by datetime"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
