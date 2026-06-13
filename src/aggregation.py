"""
This module contains functions to compute aggregated statistics from the cleaned electricity consumption data.
Includes:
- compute_hourly_stats: average, std, min, max consumption by weekday and hour.
- compute_daily_stats: average, std, min, max daily consumption by weekday.
- compute_daily_totals: total daily consumption for each date.
- compute_summary: overall summary statistics for the report.

Input: cleaned DataFrame with columns like 'date', 'weekday', 'hour', 'kWh'.

Output: DataFrames with aggregated statistics and a summary dictionary.
"""

import pandas as pd
import config


def compute_hourly_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Computes average, std, min, max kWh consumption by weekday and hour."""
    hourly_totals = df.groupby(["date", "weekday", "hour"], as_index=False)["kWh"].sum()
    hourly = (
        hourly_totals.groupby(["weekday", "hour"], as_index=False)["kWh"]
        .agg(avg_kWh="mean", std_kWh="std", min_kWh="min", max_kWh="max", days_count="count")
    )
    hourly["weekday"] = pd.Categorical(hourly["weekday"], categories=config.WEEKDAY_ORDER, ordered=True)
    return hourly.sort_values(["weekday", "hour"])


def compute_daily_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Computes average, std, min, max daily kWh consumption by weekday."""
    daily_totals = df.groupby(["date", "weekday"], as_index=False)["kWh"].sum()
    daily = (
        daily_totals.groupby("weekday", as_index=False)["kWh"]
        .agg(avg_daily_kWh="mean", std_daily_kWh="std", min_daily_kWh="min", max_daily_kWh="max", days_count="count")
    )
    daily["weekday"] = pd.Categorical(daily["weekday"], categories=config.WEEKDAY_ORDER, ordered=True)
    return daily.sort_values("weekday")


def compute_daily_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Sums total daily kWh consumption for each date."""
    return df.groupby(["date", "weekday"], as_index=False)["kWh"].sum().rename({"kWh": "daily_kWh"}, axis=1)


def compute_summary(df: pd.DataFrame, hourly: pd.DataFrame, daily: pd.DataFrame) -> dict:
    """Computes overall summary statistics for the report."""
    peak = hourly.loc[hourly["avg_kWh"].idxmax()]
    peak_hour = int(peak["hour"].item())
    return {
        "start_date": str(df["date"].min()),
        "end_date": str(df["date"].max()),
        "records": int(len(df)),
        "days": int(df["date"].nunique()),
        "total_kWh": float(df["kWh"].sum()),
        "avg_daily_kWh": float(daily["avg_daily_kWh"].mean()),
        "peak_weekday": str(peak["weekday"]),
        "peak_hour": peak_hour,
        "peak_avg_kWh": float(peak["avg_kWh"]),
    }
