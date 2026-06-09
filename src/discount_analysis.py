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
import traceback
import numpy as np
import pandas as pd
import seaborn as sns
import re
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

from config import TARIFF, WEEKDAY_ORDER, PROCESSED_DIR, TABLE_DIR, FIGURE_DIR

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


def fill_time_restriction_from_context(offers: pd.DataFrame) -> pd.DataFrame:
    """
    Where `time_restriction` is missing, parse it from the Hebrew `context` column.

    Extracts:
      - time range  e.g. "7:00 ועד 17:00"  →  "07:00-17:00"
      - weekday range from Hebrew day letters or keyword phrases

    Returns the DataFrame with `time_restriction` filled in-place for NaN rows.
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
            return row["time_restriction"]  # already filled

        parts = []

        # ── weekday ────────────────────────────────────────────────────────────
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

        # ── time range ─────────────────────────────────────────────────────────
        m = time_re.search(ctx)
        time_tag = None
        if m:
            h1 = m.group(1).zfill(5)   # "7:00" → "07:00"
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

def get_user_smart_meter_status() -> bool:
    """
    Interactively prompts the user via the terminal if smart meter status 
    is not pre-configured or passed down from the master orchestrator pipeline.
    """
    while True:
        user_input = input("Do you have a smart electricity meter installed? (Y/N): ").strip().upper()
        if user_input in {'Y', 'YES'}:
            return True
        if user_input in {'N', 'NO'}:
            return False
        print("❌ Invalid input. Please enter Y or N.")

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
    if has_smart_meter is None:
        has_smart_meter = get_user_smart_meter_status()

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

def generate_side_by_side_plots(has_smart_meter=None, processed_dir=None, table_dir=None, figure_dir=None):
    """
    Save a side-by-side heatmap for each eligible plan: actual consumption (left)
    vs. the plan's tariff rate matrix (right). Outputs saved to figure_dir.

    Parameters
    ----------
    processed_dir : Path or None
        Directory containing weekly_hourly_stats.csv. Defaults to config.PROCESSED_DIR.
    table_dir : Path or None
        Directory containing discount_scenarios.csv. Defaults to config.TABLE_DIR.
    figure_dir : Path or None
        Directory to save comparison images. Defaults to config.FIGURE_DIR.
    """
    if has_smart_meter is None:
        has_smart_meter = get_user_smart_meter_status()

    _processed_dir = Path(processed_dir) if processed_dir is not None else PROCESSED_DIR
    _table_dir = Path(table_dir) if table_dir is not None else TABLE_DIR
    _figure_dir = Path(figure_dir) if figure_dir is not None else FIGURE_DIR

    stats_path = _processed_dir / "weekly_hourly_stats.csv"

    if not stats_path.exists():
        raise FileNotFoundError(f"Missing processed consumption profile at: {stats_path}."
                                f"Please run the main pipeline to generate this file before creating visual")

    stats_df = pd.read_csv(stats_path)

    # Pivot to weekday × hour matrix
    consumption_matrix = stats_df.pivot(index='weekday', columns='hour', values='avg_kWh')

    consumption_matrix = consumption_matrix.reindex(WEEKDAY_ORDER)
    consumption_matrix.columns = HOURS_OF_DAY

    # Load scenarios
    scenarios_path = _table_dir / "discount_scenarios.csv"
    offers_df = pd.read_csv(scenarios_path)
    offers_df = add_offer_eligibility(offers_df, has_smart_meter=has_smart_meter)

    # Exclude plans the home is ineligible to utilize
    eligible_plans = offers_df[offers_df["eligibility"] != "not_eligible_requires_smart_meter"]
    unique_plans = eligible_plans[['supplier_name', 'plan_name', 'discount_pct', 'time_restriction']].drop_duplicates().reset_index(drop=True)

    output_images = []

    figure_dir_path = _figure_dir
    figure_dir_path.mkdir(parents=True, exist_ok=True)

    day_to_idx = {day: i for i, day in enumerate(WEEKDAY_ORDER)}
    
    for idx, row in unique_plans.iterrows():
        supplier = row['supplier_name']
        plan = row['plan_name']
        discount = float(row['discount_pct'])
        restriction = row['time_restriction']

        log.info("  [plot %d] %s — %s | discount=%.1f%% | restriction=%r (type=%s)",
                 idx, supplier, plan, discount, restriction, type(restriction).__name__)

        try:
            # Calculate matching cost variables
            discounted_rate = round(BASE_RATE * (1 - (discount / 100.0)), 4)
            log.info("    discounted_rate=%s", discounted_rate)

            price_matrix = np.full((len(WEEKDAY_ORDER), 24), BASE_RATE)
            log.info("    price_matrix shape=%s dtype=%s", price_matrix.shape, price_matrix.dtype)

            # Extract target days and hours from the restriction text
            target_days = [day_to_idx[d] for d in extract_weekdays(restriction) if d in day_to_idx]
            target_hours = _hours_from_restriction(restriction)
            log.info("    target_days=%s (types=%s)", target_days, [type(x).__name__ for x in target_days[:3]])
            log.info("    target_hours=%s (types=%s)", target_hours[:5], [type(x).__name__ for x in target_hours[:3]])

            for d in target_days:
                for h in target_hours:
                    price_matrix[int(d), int(h)] = discounted_rate

            log.info("    price_matrix filled OK")
            plan_matrix_df = pd.DataFrame(price_matrix, index=WEEKDAY_ORDER, columns=HOURS_OF_DAY)
            log.info("    plan_matrix_df created OK")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5.5))
            log.info("    subplots created OK")

            # Left: actual consumption
            sns.heatmap(consumption_matrix, annot=False, fmt=".2f", cmap="Oranges", ax=ax1, cbar_kws={'label': 'Usage (kWh)'})
            log.info("    left heatmap OK")
            ax1.set_title("Your Family's Real Consumption Profile", fontsize=11, fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)

            # Right: tariff rate matrix
            sns.heatmap(plan_matrix_df, annot=False, fmt=".3f", cmap="RdYlGn_r", vmin=0.450, vmax=BASE_RATE, ax=ax2, cbar_kws={'label': 'Tariff Rate (NIS)'})
            log.info("    right heatmap OK")
            ax2.set_title(f"{supplier} — {plan} ({discount}% Min Discount)", fontsize=11, fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)

            plt.tight_layout()

            safe_plan_name = re.sub(r'[^a-zA-Z0-9]', '_', f"{supplier}_{plan}").lower()
            image_filename = f"comparison_{safe_plan_name}.png"
            image_save_path = figure_dir_path / image_filename

            plt.savefig(image_save_path, dpi=150)
            plt.close()
            log.info("    saved %s", image_filename)

        except Exception:
            log.error("    FAILED on plan '%s — %s':\n%s", supplier, plan, traceback.format_exc())
            plt.close()
            continue

        output_images.append({
            "supplier": supplier,
            "plan": plan,
            "filename": image_filename
        })

    print(f"Saved {len(output_images)} comparison plots to {figure_dir_path}")
    return output_images
