"""
This module contains functions to analyze discount offers and estimate which one fits best based on the measured consumption patterns.
The source of discount offers is currently hardcoded to Kamaze's public table (pulled by ai_prompt via Claude, check README.md for instructions).
The main function is `estimate_discount_scenarios` which checks how much of the real consumption falls inside each discount's weekday/time window.
Discount table should help identify which plans are most relevant to the user.

- Input: cleaned consumption data and discount offers table (data/external/"electricity_discount_offers.csv").
- Output: 
    summarizing table of discount scenarios with estimated fit and a recommendation for the best plan.
    visual comparisons of the user's consumption profile against the discount tariff matrices, saved as images for reporting.
"""

from pathlib import Path
import logging
import re
import numpy as np
import pandas as pd
from src.text_parsers import fill_time_restriction_from_context
from config import TARIFF, WEEKDAY_ORDER

log = logging.getLogger(__name__)

# Approximate 2026 IEC standard rate per kWh (in NIS)
BASE_RATE = TARIFF
HOURS_OF_DAY = [f"{h:02d}:00" for h in range(24)]

# Maps weekday text patterns (Hebrew, English, shorthands) to day lists
WEEKDAY_MAP = {
    "sun-thu": ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"],
    "sun–thu": ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"],
    "sunday-thursday": ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"],
    "fri": ["Friday"],
    "friday": ["Friday"],
    "sat": ["Saturday"],
    "saturday": ["Saturday"],
    "כל הימים": WEEKDAY_ORDER,
    "כל השבוע": WEEKDAY_ORDER,
    "כל ימות השבוע": WEEKDAY_ORDER,
    "24/7": WEEKDAY_ORDER,
    "all days": WEEKDAY_ORDER,
    "all week": WEEKDAY_ORDER,
    # Hebrew day-range patterns (aleph=Sun … heh=Thu, vav=Fri)
    "א'-ה'": ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"],
    "א'-ו'": ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
    "ו'": ["Friday"],
    "שישי": ["Friday"],
    "שבת": ["Saturday"],
}
def get_user_smart_meter_status() -> bool:
     """
    Prompts the user via the terminal. Only valid when running pipeline.py
    as a standalone script — raises RuntimeError in a web context.
    """
     raise RuntimeError(
        "get_user_smart_meter_status() cannot be called in a web context. "
        "Pass has_smart_meter explicitly."
    )

def _bool_or_none(value):
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None

def add_offer_eligibility(offers: pd.DataFrame, has_smart_meter=None) -> pd.DataFrame:
    offers = offers.copy()
    offers["requires_smart_meter"] = offers["requires_smart_meter"].map(_bool_or_none)

    if has_smart_meter is None:
        offers["eligibility"] = offers["requires_smart_meter"].map(
            lambda x: "unknown_smart_meter_required" if x is True else "eligible_or_unknown"
        )
    elif has_smart_meter:
        offers["eligibility"] = "eligible"
    else:
        offers["eligibility"] = offers["requires_smart_meter"].map(
            lambda x: "not_eligible_requires_smart_meter" if x is True else "eligible"
        )
    return offers

def _hours_from_restriction(text):
    if not isinstance(text, str) or not text.strip() or text.lower() in {"nan", "24/7"}:
        return list(range(24))

    match = re.search(r"(\d{1,2}):\d{2}\s*[-–]\s*(\d{1,2}):\d{2}", text)
    if not match:
        return list(range(24))

    start = int(match.group(1))
    end = int(match.group(2))

    if start == end:
        return list(range(24))
    if start < end:
        return list(range(start, end))
    return list(range(start, 24)) + list(range(0, end))

def extract_weekdays(text):
    if not isinstance(text, str) or not text.strip() or text.lower() in {"nan", ""}:
        return WEEKDAY_ORDER.copy()

    text_lower = text.lower()
    weekdays = []
    for key, values in WEEKDAY_MAP.items():
        if key.lower() in text_lower:
            weekdays.extend(values)

    if not weekdays:
        return WEEKDAY_ORDER.copy()

    return [day for day in WEEKDAY_ORDER if day in set(weekdays)]

def extract_hour_range(text):
    if not isinstance(text, str) or not text.strip() or text.lower() in {"nan", "24/7"}:
        return "All day"
    match = re.search(r"(\d{1,2}:\d{2})\s*[-–]\s*(\d{1,2}:\d{2})", text)
    return f"{match.group(1)}-{match.group(2)}" if match else "All day"


def extract_weekday_range(text):
    weekdays = extract_weekdays(text)
    if len(weekdays) == 7:
        return "All week"
    if weekdays[:5] == WEEKDAY_ORDER[:5]:
        return "Sunday-Thursday"
    if weekdays == ["Friday"]:
        return "Friday"
    if weekdays == ["Saturday"]:
        return "Saturday"
    return ", ".join(weekdays)

def estimate_discount_scenarios(
    df: pd.DataFrame,
    offers: pd.DataFrame,
    has_smart_meter=None
) -> pd.DataFrame:
    """
    For each offer, calculate how much of the actual consumption falls within its
    weekday/time window, then score it by:
      weighted_discount_score = eligible_kwh * discount_pct / 100
      matching_usage_share_pct = eligible_kwh / total_kwh * 100

    Returns a DataFrame sorted by eligibility and weighted score descending.
    """
    offers = add_offer_eligibility(offers, has_smart_meter)
    total_kWh = df["kWh"].sum()
    rows = []
    for _, offer in offers.iterrows():
        restriction = offer.get("time_restriction")
        hours = _hours_from_restriction(restriction)

        mask = pd.Series(True, index=df.index)
        if hours is not None:
            mask &= df["hour"].isin(hours)
        mask &= df["weekday"].isin(extract_weekdays(restriction))

        eligible_kWh = df.loc[mask, "kWh"].sum()
        discount_pct = pd.to_numeric(offer["discount_pct"], errors="coerce")

        rows.append({
            **offer.to_dict(),
            "weekdays_applicable": extract_weekday_range(restriction),
            "hours_applicable": extract_hour_range(restriction),
            "total_measured_kWh": total_kWh,
            "consumption_matching_plan_hours": eligible_kWh,
            "matching_usage_share_pct": round(eligible_kWh / total_kWh * 100, 2) if total_kWh else 0,
            "weighted_discount_score": round(eligible_kWh * (discount_pct or 0) / 100, 2),
        })

    scenarios = pd.DataFrame(rows)
    return scenarios.sort_values(
        ["eligibility", "weighted_discount_score", "discount_pct"],
        ascending=[True, False, False]
    ).reset_index(drop=True)

def choose_recommendation(scenarios: pd.DataFrame) -> dict:
    if scenarios.empty:
        return {"message": "No discount offers were available."}

    candidates = scenarios[~scenarios["eligibility"].eq("not_eligible_requires_smart_meter")].copy()
    if candidates.empty:
        candidates = scenarios.copy()

    best = candidates.sort_values("weighted_discount_score", ascending=False).iloc[0]

    return {
        "supplier_name": best.get("supplier_name", "N/A"),
        "plan_name": best.get("plan_name", "N/A"),
        "discount_pct": best.get("discount_pct", "N/A"),
        "weekdays_applicable": best.get("weekdays_applicable", "N/A"),
        "hours_applicable": best.get("hours_applicable", "N/A"),
        "time_restriction": best.get("time_restriction", "N/A"),
        "requires_smart_meter": best.get("requires_smart_meter", "N/A"),
        "eligibility": best.get("eligibility", "N/A"),
        "matching_usage_share_pct": best.get("matching_usage_share_pct", 0),
        "weighted_discount_score": best.get("weighted_discount_score", 0),
        "source_url": best.get("source_url", ""),
    }

def _load_plot_inputs(
    processed_dir: Path,
    table_dir: Path,
    has_smart_meter,
) -> tuple:
    """
    Load and prepare the two inputs needed for side-by-side plots.

    Returns
    -------
    consumption_matrix : pd.DataFrame
        Pivoted weekday × hour average-kWh matrix (rows = days, cols = hours).
    unique_plans : pd.DataFrame
        Deduplicated table of eligible plans with columns:
        supplier_name, plan_name, discount_pct, time_restriction.

    Raises
    ------
    FileNotFoundError
        If weekly_hourly_stats.csv or discount_scenarios.csv are missing.
    """
    stats_path = processed_dir / "weekly_hourly_stats.csv"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Missing processed consumption profile at: {stats_path}. "
            "Run the main pipeline first."
        )

    stats_df = pd.read_csv(stats_path)
    consumption_matrix = (
        stats_df
        .pivot(index="weekday", columns="hour", values="avg_kWh")
        .reindex(WEEKDAY_ORDER)
    )
    consumption_matrix.columns = HOURS_OF_DAY

    scenarios_path = table_dir / "discount_scenarios.csv"
    offers_df = pd.read_csv(scenarios_path)
    offers_df = add_offer_eligibility(offers_df, has_smart_meter=has_smart_meter)

    eligible = offers_df[offers_df["eligibility"] != "not_eligible_requires_smart_meter"]
    unique_plans = (
        eligible[["supplier_name", "plan_name", "discount_pct", "time_restriction"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return consumption_matrix, unique_plans


def build_price_matrix(restriction: str, discount: float, tariff: float = BASE_RATE) -> pd.DataFrame:
    """
    Build a weekday × hour tariff-rate DataFrame for one discount plan.

    All cells start at `tariff`. Cells inside the plan's discount window
    are set to tariff × (1 - discount / 100).

    Parameters
    ----------
    restriction : str
        Raw time_restriction string (e.g. "sun-thu 07:00-17:00").
    discount : float
        Discount percentage, e.g. 15.0 for 15 %.
    tariff : float
        Base rate in ₪/kWh. Defaults to BASE_RATE from config.
        Pass the sidebar value to reflect the user's custom tariff.

    Returns
    -------
    pd.DataFrame  shape (7, 24), rows = WEEKDAY_ORDER, cols = HOURS_OF_DAY.
    """
    discounted_rate = round(tariff * (1 - discount / 100.0), 4)
    day_to_idx = {day: i for i, day in enumerate(WEEKDAY_ORDER)}

    matrix = np.full((len(WEEKDAY_ORDER), 24), tariff)
    target_days  = [day_to_idx[d] for d in extract_weekdays(restriction) if d in day_to_idx]
    target_hours = _hours_from_restriction(restriction)

    for d in target_days:
        for h in target_hours:
            matrix[int(d), int(h)] = discounted_rate

    return pd.DataFrame(matrix, index=WEEKDAY_ORDER, columns=HOURS_OF_DAY)


# _save_one_plot and generate_side_by_side_plots have moved to src/visualization.py
