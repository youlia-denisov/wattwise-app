"""Loader functions for electricity consumption data and discount offers."""

import logging
from pathlib import Path
import pandas as pd
from config import RAW_DIR, EXTERNAL_DIR
from src.text_parsers import fill_time_restriction_from_context

log = logging.getLogger(__name__)

def find_header_row(csv_path: Path) -> int:
    """Find the real IEC table header row in a CSV export."""
    with open(csv_path, encoding="utf-8-sig", errors="ignore") as f:
        for i, line in enumerate(f):
            if "תאריך" in line and "מועד תחילת הפעימה" in line:
                return i
    raise ValueError(
        f"Could not find IEC header row in {csv_path.name}. "
        "Is this an IEC consumption CSV?"
    )

def detect_kwh_col(df: pd.DataFrame) -> str | None:
    """Return the name of the kWh/consumption column, or None if not found."""
    return next(
        (c for c in df.columns
         if any(w in c.lower() for w in ["kwh", "kwatt", "consumption"])),
        None,
    )

def find_consumption_file() -> Path:
    """Find the newest IEC consumption CSV in data/raw/."""

    csv_files = sorted(
        RAW_DIR.glob("*.csv"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )

    if not csv_files:
        raise FileNotFoundError(f"No consumption CSV found in {RAW_DIR}")

    return csv_files[0]


def load_raw_csv(csv_path: Path | None = None) -> pd.DataFrame:
    """
    Load IEC consumption CSV and return only:
    date, time, kWh.

    If csv_path is not provided, the newest CSV in RAW_DIR is used.
    """

    if csv_path is None:
        csv_path = find_consumption_file()

    csv_path = Path(csv_path)
    header_row = find_header_row(csv_path)

    df = pd.read_csv(csv_path, skiprows=header_row, encoding="utf-8-sig")
    
    # Cleaning column names, changing type, removing excess spaces
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace('"', "", regex=False)
    )
    # Checking if df has correct column count.
    if df.shape[1] < 5:
        raise ValueError(
            f"Expected at least 5 columns in {csv_path.name}, "
            f"got {df.shape[1]}"
        )

    df = df.iloc[:, [2, 3, 4]].copy()
    df.columns = ["date", "time", "kWh"]

    return df


def load_discount_offers(
    csv_path: Path | None = None,
) -> pd.DataFrame:
    """
    Load electricity discount offers.

    If csv_path is not provided, loads:
    data/external/electricity_discount_offers.csv
    """

    if csv_path is None:
        csv_path = EXTERNAL_DIR / "electricity_discount_offers.csv"

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Discount offers file not found: {csv_path}"
        )

    offers = pd.read_csv(
        csv_path,
        encoding="utf-8-sig",
    )

    offers.columns = offers.columns.str.strip()

    # Fill missing time_restriction values by parsing the Hebrew context column.
    
    offers = fill_time_restriction_from_context(offers)

    return offers