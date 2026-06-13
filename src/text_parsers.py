"""
Neutral text-parsing utilities shared across modules.

Kept separate so low-level modules (loader.py) can import these helpers
without creating a circular dependency on discount_analysis.py.
"""

import re
import pandas as pd
from config import WEEKDAY_ORDER

WEEKDAY_MAP = {
    "sun-thu": ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"],
    "sunday-thursday": ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"],
    "fri": ["Friday"],
    "friday": ["Friday"],
    "sat": ["Saturday"],
    "saturday": ["Saturday"],
    "24/7": WEEKDAY_ORDER,
    "all days": WEEKDAY_ORDER,
    "all week": WEEKDAY_ORDER,
}


def fill_time_restriction_from_context(offers: pd.DataFrame) -> pd.DataFrame:
    """
    Where time_restriction is missing, parse it from the Hebrew context column.
    Returns the DataFrame with time_restriction filled in for NaN rows.
    """
    offers = offers.copy()
    if "context" not in offers.columns:
        return offers

    time_re = re.compile(r"(\d{1,2}:\d{2})\s*(?:ועד|עד|-|–)\s*(\d{1,2}:\d{2})")

    def _parse_one(row):
        ctx = row.get("context", "")
        if not isinstance(ctx, str) or not ctx.strip():
            return row["time_restriction"]
        if pd.notna(row["time_restriction"]) and str(row["time_restriction"]).strip() not in {"", "nan"}:
            return row["time_restriction"]

        weekday_tag = None
        for key in WEEKDAY_MAP:
            if key in ctx:
                days = WEEKDAY_MAP[key]
                if days == WEEKDAY_ORDER:
                    weekday_tag = "all week"
                elif days == ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"]:
                    weekday_tag = "sun-thu"
                elif days == ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
                    weekday_tag = "sun-fri"
                elif days == ["Friday"]:
                    weekday_tag = "fri"
                elif days == ["Saturday"]:
                    weekday_tag = "sat"
                else:
                    weekday_tag = "sun-thu"
                break

        m = time_re.search(ctx)
        time_tag = None
        if m:
            h1 = m.group(1).zfill(5)
            h2 = m.group(2).zfill(5)
            time_tag = f"{h1}-{h2}"

        if weekday_tag and time_tag:
            return f"{weekday_tag} {time_tag}"
        if time_tag:
            return time_tag
        if weekday_tag:
            return weekday_tag
        return row["time_restriction"]

    offers["time_restriction"] = offers.apply(_parse_one, axis=1)
    return offers
