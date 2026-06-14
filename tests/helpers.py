# helpers.py — shared factory functions used across test files.
# Import from here instead of duplicating in each test module.

import pandas as pd


def make_clean_df(n_days: int = 7) -> pd.DataFrame:
    """
    Return a preprocessed DataFrame (output of clean_consumption_data).
    Built by hand so tests don't depend on any file on disk.

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
            rows.append({
                "datetime": ts.replace(hour=hour),
                "date": ts.date(),
                "hour": hour,
                "weekday": ts.day_name(),
                "month": ts.month,
                "kWh": 0.2 + 0.05 * hour,  # gentle ramp so peak != flat
            })
    df = pd.DataFrame(rows)
    df["hour"] = df["hour"].astype("int8")
    df["month"] = df["month"].astype("int8")
    return df


def make_raw_df() -> pd.DataFrame:
    """Return a raw DataFrame — the format loader produces, before preprocessing."""
    return pd.DataFrame({
        "date": ["07/01/2024", "07/01/2024", "07/01/2024"],
        "time": ["00:00", "00:15", "00:30"],
        "kWh": ["0.100", "0.150", "0.200"],
    })


def make_offers_df() -> pd.DataFrame:
    """Return a minimal offers DataFrame, similar to electricity_discount_offers.csv."""
    return pd.DataFrame({
        "supplier_name": ["TestCo", "TestCo"],
        "plan_name": ["Day Plan", "Night Plan"],
        "discount_pct": [20, 15],
        "time_restriction": ["sun-thu 07:00-17:00", "sun-thu 23:00-07:00"],
        "requires_smart_meter": [True, False],
        "customer_type": ["All", "All"],
        "context": ["", ""],
    })
