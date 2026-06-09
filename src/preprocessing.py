"""
Preprocessing functions for electricity consumption data.
This module includes functions  such as parsing datetime, cleaning kWh values, interpolating missing readings, 
and adding time-related columns.
Data types are optimized for memory efficiency, and the code is designed to handle common issues in IEC files,
such as blank rows and non-standard kWh formats.
Input: DataFrame with columns 'date', 'time', and 'kWh'.
Output: Cleaned DataFrame with additional columns 'datetime', 'date', 'hour', 'weekday', and 'month'.
"""

import pandas as pd


def clean_consumption_data(df: pd.DataFrame) -> pd.DataFrame:
    """Parse datetime, clean kWh values, interpolate missing readings, and add time columns."""
    df = df.copy()
    df["datetime"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        format="%d/%m/%Y %H:%M",
        errors="coerce",
    )
    # IEC files sometimes contain one blank row directly after the header.
    df = df.dropna(subset=["datetime"]).copy()

    df["kWh"] = (
        df["kWh"].astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
        .str.replace(r"[^0-9.\-]", "", regex=True)
    )
    df["kWh"] = pd.to_numeric(df["kWh"], errors="coerce")

    df = df.sort_values("datetime").reset_index(drop=True)
    df["kWh"] = df["kWh"].interpolate(method="linear").bfill().ffill()
    if df["kWh"].isna().any():
        raise ValueError("kWh still contains missing values after interpolation.")

    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour.astype("int8")
    df["weekday"] = df["datetime"].dt.day_name()
    df["month"] = df["datetime"].dt.month.astype("int8")
    return df
