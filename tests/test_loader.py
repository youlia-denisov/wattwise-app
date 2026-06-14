import unittest
import tempfile
from pathlib import Path

from src.loader import find_header_row, load_raw_csv, load_discount_offers


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
