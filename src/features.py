"""
Feature engineering for electricity usage pattern clustering.

Each public function takes a preprocessed DataFrame (output of clean_consumption_data)
and returns either a single-row Series of features (for one user) or a full feature
DataFrame (when building the matrix across many users).

Data assumptions:
- 15-minute resolution kWh readings
- Columns: datetime, kWh, hour, weekday, is_weekend (from preprocessing.py)
- A 'user_id' column is expected only in build_feature_matrix()
"""

import math

import numpy as np
import pandas as pd

try:
    from config import (
        DAY_START_HOUR, DAY_END_HOUR,
        EVENING_START_HOUR, EVENING_END_HOUR,
        NIGHT_START_HOUR,
    )
except ImportError:
    # Fallback when run as a standalone script outside the project root
    DAY_START_HOUR, DAY_END_HOUR = 7, 16
    EVENING_START_HOUR, EVENING_END_HOUR = 17, 22
    NIGHT_START_HOUR = 23


# ---- helpers ----------------------------------------------------------------

def _safe_ratio(numerator: float, denominator: float) -> float:
    """Avoid ZeroDivisionError when a period has zero total consumption."""
    return numerator / denominator if denominator > 0 else 0.0


def _hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 15-min readings to hourly sums.

    Why: ratios and peaks are easier to reason about at hourly resolution,
    and hour-of-day labels map cleanly to 0-23.
    """
    return (
        df.set_index("datetime")["kWh"]
        .resample("h")
        .sum()
        .reset_index()
        .rename(columns={"kWh": "kWh_hour"})
    )


def _daily(df: pd.DataFrame) -> pd.DataFrame:
    """Sum kWh per calendar day."""
    return (
        df.groupby(df["datetime"].dt.date)["kWh"]
        .sum()
        .reset_index()
        .rename(columns={"index": "date", "kWh": "kWh_day"})
    )


# ---- time-of-day features ---------------------------------------------------

def time_of_day_ratios(h: pd.DataFrame) -> pd.Series:
    """
    What fraction of daily electricity falls in each part of the day?

    These three windows sum to 1 by construction, which makes them scale-invariant —
    a heavy user and a light user with the same routine get the same ratios.
    That is exactly what we want for clustering behaviour, not quantity.

    Windows (hour ranges, 24-h clock):
        day       07-16  — daytime activity
        evening   17-22  — after-work peak (cooking, TV, charging)
        night     23-06  — baseline / standby
    """
    h = h.copy()
    h["hour"] = h["datetime"].dt.hour

    total = h["kWh_hour"].sum()

    day     = h.loc[h["hour"].between(DAY_START_HOUR, DAY_END_HOUR), "kWh_hour"].sum()
    evening = h.loc[h["hour"].between(EVENING_START_HOUR, EVENING_END_HOUR), "kWh_hour"].sum()
    night   = h.loc[(h["hour"] >= NIGHT_START_HOUR) | (h["hour"] < DAY_START_HOUR), "kWh_hour"].sum()

    return pd.Series(
        {
            "daytime_activity_share": _safe_ratio(day,     total),
            "ratio_evening":          _safe_ratio(evening, total),
            "ratio_night":            _safe_ratio(night,   total),
        }
    )


# ---- weekday vs. weekend features -------------------------------------------

def weekday_weekend_features(h: pd.DataFrame) -> pd.Series:
    """
    Compare behaviour on weekdays vs. weekends.

    weekend_ratio > 1  → person uses more electricity on weekends (home on weekends)
    weekend_ratio ≈ 1  → constant usage regardless of day
    weekend_ratio < 1  → higher weekday usage (could be WFH, or heavy weekday appliances)

    morning_shift: do they start using electricity later on weekends?
    A positive value means the weekend morning peak is later than the weekday one —
    typical of people who sleep in.
    """
    h = h.copy()
    h["hour"] = h["datetime"].dt.hour
    h["is_weekend"] = (h["datetime"].dt.dayofweek >= 5).astype(int)

    weekday_mean = h.loc[h["is_weekend"] == 0, "kWh_hour"].mean()
    weekend_mean = h.loc[h["is_weekend"] == 1, "kWh_hour"].mean()

    # Hour of peak usage on an average weekday vs. weekend morning (06-12)
    morning_hours = h[h["hour"].between(6, 12)]

    def _peak_hour(sub: pd.DataFrame) -> float:
        if sub.empty:
            return np.nan
        return float(sub.groupby("hour")["kWh_hour"].mean().idxmax())

    wd_peak_hour = _peak_hour(morning_hours[morning_hours["is_weekend"] == 0])
    we_peak_hour = _peak_hour(morning_hours[morning_hours["is_weekend"] == 1])

    return pd.Series(
        {
            "weekend_ratio":        _safe_ratio(weekend_mean, weekday_mean),
            "morning_shift_hours":  we_peak_hour - wd_peak_hour,  # + = later on weekends
        }
    )


# ---- regularity / variability features --------------------------------------

def regularity_features(h: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
    """
    How predictable and routine is this person's electricity use?

    Coefficient of Variation (CV) = std / mean.
    Low CV → consistent user (religious observance, fixed work schedule).
    High CV → erratic usage (irregular schedule, parties, guests).

    cv_daily:        variability of total kWh from day to day
    cv_same_hour:    average CV of the same hour across all days
                     (e.g., is 8am always similar, or wildly different each day?)
    routine_score:   1 - mean(cv_same_hour), so higher = more routine.
                     Easier to read than raw CV.
    """
    cv_daily = _safe_ratio(daily["kWh_day"].std(), daily["kWh_day"].mean())

    h = h.copy()
    h["hour"] = h["datetime"].dt.hour

    # For each hour slot (0-23), compute CV across days
    cv_per_hour = (
        h.groupby("hour")["kWh_hour"]
        .agg(["std", "mean"])
        .apply(lambda row: _safe_ratio(row["std"], row["mean"]), axis=1)
    )
    mean_cv_same_hour = cv_per_hour.mean()

    return pd.Series(
        {
            "cv_daily":       cv_daily,
            "cv_same_hour":   mean_cv_same_hour,
            "routine_score":  1 - mean_cv_same_hour,  # higher = more routine
        }
    )


# ---- peak behaviour features ------------------------------------------------

def peak_features(h: pd.DataFrame) -> pd.Series:
    """
    When and how sharply does this person peak?

    hour_of_peak:      the hour (0-23) with the highest average usage —
                       morning people vs. night owls cluster differently here.
    peak_to_mean:      how extreme is the peak relative to average hourly usage?
                       High ratio → person has concentrated, spiky usage.
                       Low ratio → usage is spread evenly (always-on devices, constant load).
    evening_chores_score: ratio of Fri/Sat evening usage (18-22h) to weekday evenings.
                       High → person does chores/cooking on weekend evenings.
    """
    h = h.copy()

    h["hour"]      = h["datetime"].dt.hour
    h["dayofweek"] = h["datetime"].dt.dayofweek

    hourly_mean = h.groupby("hour")["kWh_hour"].mean()
    overall_mean = h["kWh_hour"].mean()

    hour_of_peak  = hourly_mean.idxmax()
    peak_to_mean  = _safe_ratio(hourly_mean.max(), overall_mean)

    # Fri=4, Sat=5 evenings vs. Mon-Thu=0-3 evenings
    evening = h[h["hour"].between(EVENING_START_HOUR, EVENING_END_HOUR)]
    we_evening = evening.loc[evening["dayofweek"].isin([4, 5]), "kWh_hour"].mean()
    wd_evening = evening.loc[evening["dayofweek"].isin([0, 1, 2, 3]), "kWh_hour"].mean()

    return pd.Series(
        {
            "hour_of_peak":          hour_of_peak,
            "peak_to_mean":          peak_to_mean,
            "evening_chores_score":  _safe_ratio(we_evening, wd_evening),
        }
    )


# ---- night baseline feature -------------------------------------------------

def night_baseline(h: pd.DataFrame) -> pd.Series:
    """
    Minimal Consumption Baseline — average kWh/hour between 23:00 and 07:00.

    This is the electricity your household cannot switch off: fridges, routers,
    heating/cooling systems, medical devices, fish tanks, servers, standby
    electronics. Because everyone is typically asleep during this window,
    intentional activity is near zero — what remains is the irreducible floor.

    It is a raw absolute value (not a ratio) because the *scale* matters here:
        < 0.10 kWh/h  — very low standby; efficient household
        0.10–0.20     — typical modern home
        > 0.20 kWh/h  — worth investigating; always-on appliances may be costing
                        money even when no one is actively using electricity

    Unlike the time-of-day ratios, this number does not change when overall
    consumption rises or falls — it isolates the baseline independent of habits.
    """
    h = h.copy()

    h["hour"] = h["datetime"].dt.hour
    night_mean = h.loc[(h["hour"] >= NIGHT_START_HOUR) | (h["hour"] < DAY_START_HOUR), "kWh_hour"].mean()

    return pd.Series({"min_consumption_baseline_kwh": night_mean})


# ---- master function ---------------------------------------------------------

def build_user_features(df: pd.DataFrame) -> pd.Series:
    """
    Combine all feature groups into one Series for a single user.

    Call this per user, then stack the results into a DataFrame for clustering:

        feature_df = (
            raw_data
            .groupby("user_id")
            .apply(lambda g: build_user_features(g))
        )

    Each row of feature_df = one user, each column = one feature.
    That matrix goes directly into StandardScaler → KMeans (or DBSCAN).
    """
    h = _hourly(df)
    daily = _daily(df)
    return pd.concat(
        [
            time_of_day_ratios(h),
            weekday_weekend_features(h),
            regularity_features(h, daily),
            peak_features(h),
            night_baseline(h),
        ]
    )
# ----- PERSONA INSIGHTS ----------------------------------------------------------------

# ── Persona classification thresholds ─────────────────────────────────────────
# Change a value here and every persona-related function updates automatically.
_HOME_DWELLER_DAY_SHARE   = 0.45   # daytime_activity_share above this → home dweller
_NIGHT_OWL_NIGHT_SHARE    = 0.25   # ratio_night above this → night owl
_WEEKEND_WARRIOR_RATIO    = 1.35   # weekend_ratio above this → weekend warrior
_CLOCKWORK_ROUTINE_SCORE  = 0.70   # routine_score above this → clockwork household
_AFTER_WORK_EVENING_SHARE = 0.40   # ratio_evening above this → after-work household


def derive_persona(f: pd.Series) -> dict:
    """
    Map feature values to a single human-readable household persona.
    Returns a dict with: emoji, label, tagline, color.
    """
    ratio_day     = f.get("daytime_activity_share", 0)
    ratio_night   = f.get("ratio_night",            0)
    ratio_evening = f.get("ratio_evening",          0)
    weekend_ratio = f.get("weekend_ratio", 1)
    routine_score = f.get("routine_score", 0.5)

    if ratio_day > _HOME_DWELLER_DAY_SHARE:
        return dict(emoji="🏠", label="Home Dweller",
                    tagline="Most of your electricity is used during the day — you're home a lot!",
                    color="#FFB74D")
    if ratio_night > _NIGHT_OWL_NIGHT_SHARE:
        return dict(emoji="🌙", label="Night Owl",
                    tagline="Your household comes alive at night — late evenings are your peak time.",
                    color="#9575CD")
    if weekend_ratio > _WEEKEND_WARRIOR_RATIO:
        return dict(emoji="🎉", label="Weekend Warrior",
                    tagline="You use noticeably more electricity on weekends — home is where the weekend is.",
                    color="#4DB6AC")
    if routine_score > _CLOCKWORK_ROUTINE_SCORE:
        return dict(emoji="🕐", label="Clockwork Household",
                    tagline="Your daily routine is very consistent — predictable and efficient.",
                    color="#64B5F6")
    if ratio_evening > _AFTER_WORK_EVENING_SHARE:
        return dict(emoji="🌆", label="After-Work Household",
                    tagline="Evenings are your busiest time — classic after-work energy spike.",
                    color="#F06292")
    return dict(emoji="⚖️", label="Balanced Household",
                tagline="Your usage is spread fairly evenly — no single pattern dominates.",
                color="#81C784")

def build_insights(f):
    """Return a list of insight dicts from feature values (up to 5 cards)."""
    insights = []

    hop = f.get("hour_of_peak", None)
    if hop is not None:
        hour_label = f"{int(hop):02d}:00"
        if hop < 10:
            desc = "You peak in the morning — early riser or morning appliances."
        elif hop < 15:
            desc = "Midday is your busiest time — typical of home workers."
        elif hop < 20:
            desc = "Your peak is in the late afternoon or evening — common after-work pattern."
        else:
            desc = "You peak late at night — night-owl household."
        insights.append(dict(icon="⏰", title="Peak hour", value=hour_label, desc=desc, status="info"))

    rs = f.get("routine_score", None)
    if rs is not None:
        if rs > 0.70:
            r_val, r_desc, r_st = "Very routine", "Your schedule is highly predictable — same pattern day after day.", "good"
        elif rs > 0.45:
            r_val, r_desc, r_st = "Moderately routine", "Some variation in your daily pattern, but broadly consistent.", "info"
        else:
            r_val, r_desc, r_st = "Unpredictable", "Your usage varies a lot day-to-day — irregular schedule.", "warn"
        insights.append(dict(icon="📅", title="Routine level", value=r_val, desc=r_desc, status=r_st))

    wr = f.get("weekend_ratio", None)
    if wr is not None:
        if wr > 1.2:
            wr_val, wr_desc = f"{wr:.1f}x more on weekends", "You use more electricity on weekends — you're probably home more then."
        elif wr < 0.85:
            wr_val, wr_desc = f"{wr:.1f}x less on weekends", "Weekdays dominate — could be a home-office setup."
        else:
            wr_val, wr_desc = "Similar on both days", "Your weekday and weekend usage are about the same."
        insights.append(dict(icon="📆", title="Weekday vs weekend", value=wr_val, desc=wr_desc, status="info"))

    nb = f.get("min_consumption_baseline_kwh", None)
    if nb is not None and not math.isnan(float(nb)):
        if nb > 0.20:
            nb_val  = f"{nb:.2f} kWh/h"
            nb_desc = (
                "This is your Minimal Consumption Baseline — electricity your home "
                "uses even when everyone is asleep (fridges, routers, standby devices). "
                "Your level is above average: worth checking for always-on appliances "
                "that could be switched off or replaced."
            )
            nb_st = "warn"
        elif nb > 0.10:
            nb_val  = f"{nb:.2f} kWh/h"
            nb_desc = (
                "This is your Minimal Consumption Baseline — the unavoidable overnight "
                "draw from fridges, routers, and standby electronics. "
                "Your level is typical for a modern home."
            )
            nb_st = "info"
        else:
            nb_val  = f"{nb:.2f} kWh/h"
            nb_desc = (
                "This is your Minimal Consumption Baseline — electricity consumed "
                "while the household sleeps. Your standby load is very low, "
                "suggesting well-managed or energy-efficient appliances."
            )
            nb_st = "good"
        insights.append(dict(icon="🔌", title="Minimal Consumption Baseline", value=nb_val, desc=nb_desc, status=nb_st))

    ms = f.get("morning_shift_hours", None)
    if ms is not None and not math.isnan(float(ms)):
        if ms > 1.5:
            ms_val, ms_desc = f"+{ms:.1f} h later on weekends", "You start your day noticeably later on weekends — classic sleep-in."
        elif ms < -0.5:
            ms_val, ms_desc = f"{ms:.1f} h earlier on weekends", "Weekend mornings start earlier — early bird!"
        else:
            ms_val, ms_desc = "No shift", "Morning routine is similar on weekdays and weekends."
        insights.append(dict(icon="😴", title="Weekend sleep-in", value=ms_val, desc=ms_desc, status="info"))

    return insights