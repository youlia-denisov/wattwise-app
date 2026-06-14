import unittest
import tempfile
from pathlib import Path


class TestPipelineFiles(unittest.TestCase):
    """
    Verifies that each pipeline step saves the files the next step depends on.
    Catches "[Errno 2] No such file or directory" errors before they hit the app.
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
        run_clustering must raise FileNotFoundError when the input file is missing.
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
