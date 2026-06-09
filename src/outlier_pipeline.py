"""
outlier_pipeline.py
-------------------
Runs four outlier/anomaly detection methods on cleaned electricity consumption
data and auto-selects the most appropriate one based on data characteristics.

Methods
-------
1. 3-Sigma  — classic mean ± 3σ threshold. Reliable when data is roughly normal.
2. IQR      — Q1 − 1.5×IQR / Q3 + 1.5×IQR. Robust to skewed distributions.
3. DBSCAN   — density-based; flags points that don't fit any dense region.
              Uses the same feature space as the main clustering pipeline.
4. Isolation Forest — tree-based anomaly detection. Works well on small
              datasets and non-linear patterns. Doesn't assume any distribution.

Auto-selection logic
--------------------
The selector scores each method against three data properties:
  - skewness      : how symmetric the kWh distribution is
  - n_samples     : total number of readings
  - outlier_rate  : fraction flagged as outliers (sanity check)

Rules (applied in order):
  1. If skewness > 1.5 (highly skewed)      → IQR or Isolation Forest
  2. If n_samples < 500                     → IQR (simple, no hyperparams)
  3. If skewness between 0.5 and 1.5        → IQR preferred over 3-sigma
  4. If skewness < 0.5 (near-normal)        → 3-sigma is fine
  5. DBSCAN and Isolation Forest are shown as alternatives but not
     auto-selected as primary because they require feature engineering
     that may not always be available in the df passed to this module.

Returns
-------
OutlierResults dataclass containing:
  - results_by_method : dict[str, pd.DataFrame]  — flagged rows per method
  - summary           : pd.DataFrame             — counts + % per method
  - recommended       : str                      — name of selected method
  - reason            : str                      — plain-English explanation
  - stats             : dict                     — skewness, n_samples, etc.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest


# ── constants ─────────────────────────────────────────────────────────────────

VALUE_COL = "kWh"

# Feature columns used for DBSCAN and Isolation Forest.
# These are the same engineered features produced by the main pipeline's
# features.py / clustering.py, so they're always present in
# cleaned_consumption_clustered.csv (the file the Streamlit app loads).
FEATURE_COLS = [
    "kWh",
    "hour_sin", "hour_cos",
    "weekday_sin", "weekday_cos",
    "daily_total_kwh",
    "evening_peak",
    "night_baseline",
    "peak_to_baseline",
]

# DBSCAN: eps chosen to avoid both mega-clusters (>80% in one cluster)
# and near-total noise (>50% flagged). Auto-tuned via k-distance if possible.
DBSCAN_EPS_CANDIDATES = [0.3, 0.5, 0.8, 1.2, 1.5, 1.8, 2.0, 2.5]
DBSCAN_MIN_SAMPLES = 5
DBSCAN_MAX_DOMINANCE = 0.80   # skip eps if one cluster holds > 80% of points
DBSCAN_MAX_NOISE_FRAC = 0.50  # skip eps if > 50% points are noise

# Isolation Forest: contamination is the expected fraction of anomalies.
# "auto" lets sklearn estimate it from the data (equivalent to ~ 0.1 for most
# datasets). You can override with a float, e.g. 0.05.
IF_CONTAMINATION = "auto"
IF_N_ESTIMATORS = 200
IF_RANDOM_STATE = 42


# ── result container ───────────────────────────────────────────────────────────

@dataclass
class OutlierResults:
    results_by_method: dict[str, pd.DataFrame] = field(default_factory=dict)
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    recommended: str = ""
    reason: str = ""
    stats: dict = field(default_factory=dict)


# ── individual detectors ───────────────────────────────────────────────────────

def _detect_3sigma(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Flag readings outside mean ± 3 standard deviations."""
    s = df[VALUE_COL]
    mean, std = s.mean(), s.std()
    lower, upper = mean - 3 * std, mean + 3 * std
    mask = (s < lower) | (s > upper)
    flagged = df[mask].copy()
    flagged["outlier_score"] = ((s[mask] - mean) / std).abs()  # z-score as score
    params = {"mean": round(mean, 4), "std": round(std, 4),
              "lower": round(lower, 4), "upper": round(upper, 4)}
    return flagged, params


def _detect_iqr(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Flag readings outside Q1 − 1.5×IQR / Q3 + 1.5×IQR."""
    s = df[VALUE_COL]
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (s < lower) | (s > upper)
    flagged = df[mask].copy()
    # Score: how many IQR-lengths away from the nearest fence
    dist_lower = (lower - s[mask]).clip(lower=0) / iqr
    dist_upper = (s[mask] - upper).clip(lower=0) / iqr
    flagged["outlier_score"] = (dist_lower + dist_upper)
    params = {"q1": round(q1, 4), "q3": round(q3, 4),
              "iqr": round(iqr, 4), "lower": round(lower, 4), "upper": round(upper, 4)}
    return flagged, params


def _auto_tune_eps(X_scaled: np.ndarray) -> tuple[float, str]:
    """
    Choose eps by sweeping candidates and picking the loosest eps that:
      - has < DBSCAN_MAX_NOISE_FRAC noise points, AND
      - has < DBSCAN_MAX_DOMINANCE fraction in one cluster.

    Falls back to eps=0.8 if no candidate satisfies both constraints.
    """
    best_eps = 0.8  # sensible default
    best_label = "fallback default (eps=0.8)"

    for eps in sorted(DBSCAN_EPS_CANDIDATES):
        labels = DBSCAN(eps=eps, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(X_scaled)
        n_total = len(labels)
        noise_frac = (labels == -1).sum() / n_total

        non_noise = labels[labels != -1]
        if len(non_noise) == 0:
            continue  # all noise — skip

        counts = pd.Series(non_noise).value_counts()
        dominance = counts.iloc[0] / len(non_noise)

        if noise_frac < DBSCAN_MAX_NOISE_FRAC and dominance < DBSCAN_MAX_DOMINANCE:
            best_eps = eps
            best_label = (
                f"eps={eps:.2f} → {noise_frac*100:.1f}% noise, "
                f"dominance {dominance*100:.1f}%"
            )
            break

    return best_eps, best_label


def _detect_dbscan(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Run DBSCAN on the multi-feature space (time, consumption, engineered features).
    Points labelled -1 (noise) are the outliers.
    """
    available_features = [c for c in FEATURE_COLS if c in df.columns]

    if len(available_features) < 3:
        return _detect_iqr(df)[0].assign(method="DBSCAN_fallback_IQR"), {
            "note": "Insufficient feature columns; fell back to IQR"
        }

    df_work = df.copy()
    if "datetime" in df_work.columns:
        df_work["datetime"] = pd.to_datetime(df_work["datetime"])
        df_work["hour_dt"] = df_work["datetime"].dt.floor("h")
        agg_dict = {c: "mean" for c in available_features}
        agg_dict["kWh"] = "sum"
        df_hourly = (
            df_work.groupby("hour_dt")
            .agg(agg_dict)
            .reset_index()
            .rename(columns={"hour_dt": "datetime"})
        )
    else:
        df_hourly = df_work.copy()

    X = RobustScaler().fit_transform(df_hourly[available_features])

    eps, eps_note = _auto_tune_eps(X)
    labels = DBSCAN(eps=eps, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(X)
    df_hourly["dbscan_label"] = labels

    noise_rows = df_hourly[df_hourly["dbscan_label"] == -1].copy()
    if len(noise_rows) > 0:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(DBSCAN_MIN_SAMPLES, len(df_hourly)))
        nn.fit(X)
        distances, _ = nn.kneighbors(X[noise_rows.index])
        noise_rows["outlier_score"] = distances[:, -1]

    n_clusters = len(set(labels) - {-1})
    n_noise = (labels == -1).sum()
    params = {
        "eps": eps,
        "eps_selection_note": eps_note,
        "min_samples": DBSCAN_MIN_SAMPLES,
        "n_clusters_found": n_clusters,
        "n_noise": int(n_noise),
        "noise_pct": round(100 * n_noise / len(labels), 1),
        "features_used": available_features,
        "resampled_to_hourly": "datetime" in df.columns,
    }
    return noise_rows, params


def _detect_isolation_forest(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Isolation Forest: builds random trees that isolate points.
    Anomalies are isolated in fewer splits — they get a high anomaly score.
    """
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    feature_cols = available_features if len(available_features) >= 3 else [VALUE_COL]

    X = RobustScaler().fit_transform(df[feature_cols])

    iso = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        contamination=IF_CONTAMINATION,
        random_state=IF_RANDOM_STATE,
    )
    preds = iso.fit_predict(X)
    scores = iso.score_samples(X)

    df_result = df.copy()
    df_result["if_pred"] = preds
    df_result["outlier_score"] = -scores

    flagged = df_result[df_result["if_pred"] == -1].copy()

    params = {
        "n_estimators": IF_N_ESTIMATORS,
        "contamination": IF_CONTAMINATION,
        "features_used": feature_cols,
        "n_flagged": len(flagged),
    }
    return flagged.drop(columns=["if_pred"]), params


# ── auto-selector ──────────────────────────────────────────────────────────────

def _select_method(skewness: float, n_samples: int, rates: dict[str, float]) -> tuple[str, str]:
    """Return (method_name, plain_english_reason) for the recommended method."""
    sigma_rate = rates.get("3-Sigma", 0)

    if n_samples < 200:
        return (
            "IQR",
            f"Dataset is small ({n_samples} readings). IQR makes no assumptions about "
            "the distribution shape, making it the most reliable choice here."
        )

    if skewness > 1.5:
        return (
            "IQR",
            f"The kWh distribution is strongly right-skewed (skewness = {skewness:.2f}). "
            "High consumption spikes pull the mean upward, which distorts the 3-sigma fences. "
            "IQR uses the median and quartiles, which are not affected by extreme values."
        )

    if skewness > 0.5:
        return (
            "IQR",
            f"The distribution is moderately skewed (skewness = {skewness:.2f}). "
            "IQR is more robust than 3-sigma here because the mean and standard deviation "
            "are slightly inflated by the right tail of the consumption data."
        )

    if sigma_rate > 0.05:
        return (
            "IQR",
            f"3-sigma flagged {sigma_rate*100:.1f}% of readings — well above the expected 0.3% "
            "for a normal distribution. This confirms the distribution is too skewed for "
            "3-sigma to be reliable. Switching to IQR."
        )

    return (
        "3-Sigma",
        f"The distribution is close to symmetric (skewness = {skewness:.2f}) and 3-sigma "
        f"flagged {sigma_rate*100:.1f}% of readings, close to the expected 0.3%. "
        "3-sigma is appropriate for this data."
    )


# ── main entry point ───────────────────────────────────────────────────────────

def run_outlier_pipeline(df: pd.DataFrame) -> OutlierResults:
    """
    Run all four outlier detection methods and return an OutlierResults object.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned consumption data. Must contain a 'kWh' column.
        If engineered feature columns (hour_sin, daily_total_kwh, etc.) are
        present, DBSCAN and Isolation Forest will use the full feature space.
        Otherwise they fall back to univariate detection on kWh alone.

    Returns
    -------
    OutlierResults
        .results_by_method  — {method_name: DataFrame of flagged rows}
        .summary            — one row per method with count and %
        .recommended        — name of the auto-selected method
        .reason             — plain-English explanation of the selection
        .stats              — dict with skewness, n_samples, method params
    """
    if VALUE_COL not in df.columns:
        raise ValueError(f"DataFrame must contain a '{VALUE_COL}' column.")

    kwh = pd.to_numeric(df[VALUE_COL], errors="coerce").dropna()
    n = len(kwh)
    skewness = float(kwh.skew())

    sigma_flagged, sigma_params = _detect_3sigma(df)
    iqr_flagged, iqr_params = _detect_iqr(df)
    dbscan_flagged, dbscan_params = _detect_dbscan(df)
    if_flagged, if_params = _detect_isolation_forest(df)

    results = {
        "3-Sigma": sigma_flagged,
        "IQR": iqr_flagged,
        "DBSCAN": dbscan_flagged,
        "Isolation Forest": if_flagged,
    }

    rates = {name: len(df_) / n for name, df_ in results.items()}
    recommended, reason = _select_method(skewness, n, rates)

    summary_rows = []
    for method_name, flagged_df in results.items():
        summary_rows.append({
            "Method": method_name,
            "Flagged": len(flagged_df),
            "% of total": round(100 * len(flagged_df) / n, 1),
            "Recommended": "✓" if method_name == recommended else "",
        })
    summary = pd.DataFrame(summary_rows)

    stats = {
        "skewness": round(skewness, 3),
        "n_samples": n,
        "3sigma_params": sigma_params,
        "iqr_params": iqr_params,
        "dbscan_params": dbscan_params,
        "if_params": if_params,
    }

    return OutlierResults(
        results_by_method=results,
        summary=summary,
        recommended=recommended,
        reason=reason,
        stats=stats,
    )
