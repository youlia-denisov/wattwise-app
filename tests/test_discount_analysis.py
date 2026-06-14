import unittest

from src.discount_analysis import (
    _hours_from_restriction,
    extract_weekdays,
    extract_hour_range,
    add_offer_eligibility,
)
from helpers import make_offers_df


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
        If you just do range(23, 7) you get [] — the wrap-around must be handled.
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
        Safe default — don't silently exclude any day.
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
        """User HAS a smart meter → all offers are 'eligible'."""
        offers = make_offers_df()
        result = add_offer_eligibility(offers, has_smart_meter=True)
        self.assertTrue((result["eligibility"] == "eligible").all())

    def test_eligibility_without_smart_meter(self):
        """
        User does NOT have a smart meter → plans that require one must be
        marked 'not_eligible_requires_smart_meter'.
        """
        offers = make_offers_df()
        result = add_offer_eligibility(offers, has_smart_meter=False)
        self.assertEqual(result.iloc[0]["eligibility"], "not_eligible_requires_smart_meter")
        self.assertEqual(result.iloc[1]["eligibility"], "eligible")

    def test_eligibility_unknown_smart_meter(self):
        """
        When smart meter status is unknown (None), plans that require one
        get 'unknown_smart_meter_required'; others get 'eligible_or_unknown'.
        """
        offers = make_offers_df()
        result = add_offer_eligibility(offers, has_smart_meter=None)
        self.assertEqual(result.iloc[0]["eligibility"], "unknown_smart_meter_required")
        self.assertEqual(result.iloc[1]["eligibility"], "eligible_or_unknown")


if __name__ == "__main__":
    unittest.main(verbosity=2)
