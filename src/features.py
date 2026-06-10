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

import numpy as np
import pandas as pd


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

def time_of_day_ratios(df: pd.DataFrame) -> pd.Series:
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
    h = _hourly(df)
    h["hour"] = h["datetime"].dt.hour

    total = h["kWh_hour"].sum()

    day     = h.loc[h["hour"].between(7, 16), "kWh_hour"].sum()
    evening = h.loc[h["hour"].between(17, 22), "kWh_hour"].sum()
    night   = h.loc[h["hour"].isin(list(range(23, 24)) + list(range(0, 7))), "kWh_hour"].sum()

    return pd.Series(
        {
            "ratio_day":     _safe_ratio(day,     total),
            "ratio_evening": _safe_ratio(evening, total),
            "ratio_night":   _safe_ratio(night,   total),
        }
    )


# ---- weekday vs. weekend features -------------------------------------------

def weekday_weekend_features(df: pd.DataFrame) -> pd.Series:
    """
    Compare behaviour on weekdays vs. weekends.

    weekend_ratio > 1  → person uses more electricity on weekends (home on weekends)
    weekend_ratio ≈ 1  → constant usage regardless of day
    weekend_ratio < 1  → higher weekday usage (could be WFH, or heavy weekday appliances)

    morning_shift: do they start using electricity later on weekends?
    A positive value means the weekend morning peak is later than the weekday one —
    typical of people who sleep in.
    """
    h = _hourly(df)
    h["hour"]       = h["datetime"].dt.hour
    h["is_weekend"] = (h["datetime"].dt.dayofweek >= 5).astype(int)

    weekday_mean = h.loc[h["is_weekend"] == 0, "kWh_hour"].mean()
    weekend_mean = h.loc[h["is_weekend"] == 1, "kWh_hour"].mean()

    # Hour of peak usage on an average weekday vs. weekend morning (06-12)
    morning_hours = h[h["hour"].between(6, 12)]

    def _peak_hour(sub: pd.DataFrame) -> float:
        if sub.empty:
            return np.nan
        return sub.groupby("hour")["kWh_hour"].mean().idxmax()

    wd_peak_hour = _peak_hour(morning_hours[morning_hours["is_weekend"] == 0])
    we_peak_hour = _peak_hour(morning_hours[morning_hours["is_weekend"] == 1])

    return pd.Series(
        {
            "weekend_ratio":        _safe_ratio(weekend_mean, weekday_mean),
            "morning_shift_hours":  we_peak_hour - wd_peak_hour,  # + = later on weekends
        }
    )


# ---- regularity / variability features --------------------------------------

def regularity_features(df: pd.DataFrame) -> pd.Series:
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
    daily = _daily(df)
    cv_daily = _safe_ratio(daily["kWh_day"].std(), daily["kWh_day"].mean())

    h = _hourly(df)
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

def peak_features(df: pd.DataFrame) -> pd.Series:
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
    h = _hourly(df)
    h["hour"]      = h["datetime"].dt.hour
    h["dayofweek"] = h["datetime"].dt.dayofweek

    hourly_mean = h.groupby("hour")["kWh_hour"].mean()
    overall_mean = h["kWh_hour"].mean()

    hour_of_peak  = hourly_mean.idxmax()
    peak_to_mean  = _safe_ratio(hourly_mean.max(), overall_mean)

    # Fri=4, Sat=5 evenings vs. Mon-Thu=0-3 evenings
    evening = h[h["hour"].between(17, 22)]
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

def night_baseline(df: pd.DataFrame) -> pd.Series:
    """
    Average kWh between midnight and 5am.

    This captures standby/always-on load: servers, fish tanks, medical devices,
    or simply poor insulation with always-on heating. It's a raw value (not a ratio)
    because absolute nighttime draw is meaningful on its own.
    """
    h = _hourly(df)
    h["hour"] = h["datetime"].dt.hour
    night_mean = h.loc[h["hour"].isin(list(range(23, 24)) + list(range(0, 7))), "kWh_hour"].mean()

    return pd.Series({"night_baseline_kwh": night_mean})


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
    return pd.concat(
        [
            time_of_day_ratios(df),
            weekday_weekend_features(df),
            regularity_features(df),
            peak_features(df),
            night_baseline(df),
        ]
    )
