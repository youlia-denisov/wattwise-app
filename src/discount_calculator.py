"""
Calculator module for comparing actual (or hypothetical) electricity usage
against every available discount plan, expressed in NIS.

The key improvement over discount_analysis.estimate_discount_scenarios is that
results here are money-denominated: you see NIS saved and effective discount %,
not a dimensionless weighted score.

Typical call chain:
    df = pd.read_csv(...)          # cleaned_consumption.csv with hour/weekday cols
    offers = pd.read_csv(...)      # electricity_discount_offers.csv
    results = compare_all_plans(df, offers, tariff=0.666)
    annual = extrapolate_annual(results, observation_days=df["date"].nunique())
"""

import pandas as pd
from config import (
    TARIFF, WEEKDAY_ORDER,
    DAY_START_HOUR, DAY_END_HOUR, EVENING_START_HOUR, EVENING_END_HOUR, NIGHT_START_HOUR,
)

from src.discount_analysis import (
    _hours_from_restriction,
    extract_weekdays,
    extract_hour_range,
    extract_weekday_range,
    add_offer_eligibility,
)


def calculate_plan_savings(
    df: pd.DataFrame,
    offer: pd.Series,
    tariff: float = TARIFF,
) -> dict:
    """
    For a single discount plan, compute the realistic NIS savings
    against the user's actual consumption pattern.

    Returns a dict with:
        kwh_in_window       -- kWh that falls inside the discount window
        kwh_outside_window  -- kWh outside (billed at full rate)
        cost_baseline       -- what the user would pay without any discount (NIS)
        cost_with_plan      -- what the user pays with this plan (NIS)
        nis_saved           -- cost_baseline - cost_with_plan
        effective_discount_pct -- nis_saved / cost_baseline * 100
        matching_usage_share_pct -- kwh_in_window / total_kwh * 100
    """
    restriction = offer.get("time_restriction", "")
    discount_pct = pd.to_numeric(offer.get("discount_pct", 0), errors="coerce") or 0.0

    target_hours = _hours_from_restriction(restriction)
    target_days = extract_weekdays(restriction)

    mask = df["hour"].isin(target_hours) & df["weekday"].isin(target_days)

    kwh_in = df.loc[mask, "kWh"].sum()
    kwh_out = df.loc[~mask, "kWh"].sum()
    total_kwh = kwh_in + kwh_out

    cost_baseline = total_kwh * tariff
    # kWh outside window billed at full rate; kWh inside window billed at discounted rate
    discounted_rate = tariff * (1 - discount_pct / 100)
    cost_with_plan = kwh_out * tariff + kwh_in * discounted_rate

    nis_saved = cost_baseline - cost_with_plan
    effective_discount_pct = (nis_saved / cost_baseline * 100) if cost_baseline > 0 else 0.0

    return {
        "kwh_in_window": round(kwh_in, 3),
        "kwh_outside_window": round(kwh_out, 3),
        "total_kwh": round(total_kwh, 3),
        "cost_baseline_nis": round(cost_baseline, 2),
        "cost_with_plan_nis": round(cost_with_plan, 2),
        "nis_saved": round(nis_saved, 2),
        "effective_discount_pct": round(effective_discount_pct, 2),
        "matching_usage_share_pct": round(kwh_in / total_kwh * 100, 2) if total_kwh > 0 else 0.0,
    }


def compare_all_plans(
    df: pd.DataFrame,
    offers: pd.DataFrame,
    tariff: float = TARIFF,
    has_smart_meter=None,
) -> pd.DataFrame:
    """
    Run calculate_plan_savings for every unique plan and return a single
    DataFrame sorted by nis_saved descending.

    Deduplication: if the same (supplier_name, plan_name) appears in both
    Kamaze and IsraelElectricity sources, we keep only the first occurrence
    since the discount terms are the same.
    """
    offers = add_offer_eligibility(offers.copy(), has_smart_meter)

    # Keep one row per (supplier + plan) to avoid double-counting
    unique_plans = (
        offers
        .drop_duplicates(subset=["supplier_name", "plan_name"])
        .reset_index(drop=True)
    )

    rows = []
    for _, offer in unique_plans.iterrows():
        savings = calculate_plan_savings(df, offer, tariff)
        rows.append({
            "supplier_name": offer.get("supplier_name", ""),
            "plan_name": offer.get("plan_name", ""),
            "discount_pct": offer.get("discount_pct", 0),
            "time_restriction": offer.get("time_restriction", ""),
            "weekdays_applicable": extract_weekday_range(offer.get("time_restriction", "")),
            "hours_applicable": extract_hour_range(offer.get("time_restriction", "")),
            "requires_smart_meter": offer.get("requires_smart_meter"),
            "eligibility": offer.get("eligibility", ""),
            "customer_type": offer.get("customer_type", ""),
            **savings,
        })

    result = pd.DataFrame(rows)
    return result.sort_values("nis_saved", ascending=False).reset_index(drop=True)


def extrapolate_annual(
    results: pd.DataFrame,
    observation_days: int,
) -> pd.DataFrame:
    """
    Scale the savings from the observed period to a full year (365 days).

    Adds columns:
        annual_nis_saved        -- projected annual saving in NIS
        annual_cost_baseline    -- projected annual baseline cost in NIS
        annual_cost_with_plan   -- projected annual cost with the plan in NIS

    observation_days should be df["date"].nunique() from the original data.
    """
    if observation_days <= 0:
        raise ValueError("observation_days must be positive")

    scale = 365 / observation_days
    out = results.copy()
    out["annual_nis_saved"] = (out["nis_saved"] * scale).round(2)
    out["annual_cost_baseline"] = (out["cost_baseline_nis"] * scale).round(2)
    out["annual_cost_with_plan"] = (out["cost_with_plan_nis"] * scale).round(2)
    out["observation_days"] = observation_days
    return out


def build_custom_pattern_df(
    monthly_kwh: float,
    pct_weekday_day: float,       # % of kWh in Sun-Thu 07:00-17:00
    pct_weekday_evening: float,   # % of kWh in Sun-Thu 17:00-23:00
    pct_weekday_night: float,     # % of kWh in Sun-Thu 23:00-07:00
    pct_weekend: float,           # % of kWh on Fri-Sat (any hour)
) -> pd.DataFrame:
    """
    Build a synthetic hourly DataFrame that mimics cleaned_consumption.csv
    structure, from user-supplied percentages. This lets the calculator work
    without needing real meter data.

    Percentages are normalised to 100% internally so they don't have to sum
    exactly (the user is using sliders and it's easy to drift slightly).

    Returns a DataFrame with columns: hour, weekday, kWh.
    """
    total = pct_weekday_day + pct_weekday_evening + pct_weekday_night + pct_weekend
    if total <= 0:
        raise ValueError("At least one usage percentage must be > 0")

    # Normalise
    wd_day     = pct_weekday_day     / total * monthly_kwh
    wd_evening = pct_weekday_evening / total * monthly_kwh
    wd_night   = pct_weekday_night   / total * monthly_kwh
    wknd       = pct_weekend         / total * monthly_kwh

    weekdays   = WEEKDAY_ORDER[:5]   # Sun-Thu
    weekend    = WEEKDAY_ORDER[5:]   # Fri-Sat

    day_hours     = list(range(DAY_START_HOUR, DAY_END_HOUR + 1))              # 07-16
    evening_hours = list(range(EVENING_START_HOUR, EVENING_END_HOUR + 1))    # 17-22
    night_hours   = list(range(NIGHT_START_HOUR, 24)) + list(range(0, DAY_START_HOUR))  # 23, 00-06

    rows = []

    def _spread(kwh_total, days, hours):
        n = len(days) * len(hours)
        if n == 0:
            return
        per_slot = kwh_total / n
        for d in days:
            for h in hours:
                rows.append({"weekday": d, "hour": h, "kWh": per_slot})

    # 4 weeks in a month ≈ a reasonable approximation for the synthetic pattern
    _spread(wd_day,     weekdays, day_hours)
    _spread(wd_evening, weekdays, evening_hours)
    _spread(wd_night,   weekdays, night_hours)
    _spread(wknd,       weekend,  list(range(24)))

    return pd.DataFrame(rows)
