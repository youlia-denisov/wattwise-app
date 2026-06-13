"""
Detecting outliers using 3sigma method.
The 3-sigma method marks values as outliers when they are more than 3 standard deviations away from the mean.
In a normal distribution, about 99.7% of values are expected to fall within ±3 standard deviations from the mean.
Therefore, only about 0.3% of values are expected to be outside this range.
If the percentage of 3-sigma outliers is much higher than 0.3%, the data may not be normally distributed,
or it may contain unusual consumption spikes, measurement errors, or strong behavioral patterns.
If the data is skewed, the mean and standard deviation can be pulled toward extreme values.
In that case, the 3-sigma method may be less reliable.

Skewness shows whether the data is symmetric or pulled toward one side.

Skewness close to 0 means the distribution is approximately symmetric.
Positive skew means there are unusually high values pulling the distribution to the right.
Negative skew means there are unusually low values pulling the distribution to the left.

A common rough interpretation:
- between -0.5 and 0.5: approximately symmetric
- between 0.5 and 1 or -0.5 and -1: moderately skewed
- above 1 or below -1: highly skewed

For strongly skewed electricity consumption data, IQR-based outlier detection may be more appropriate.
"""
import pandas as pd


def _apply_limits(
    df: pd.DataFrame,
    lower_limit: float,
    upper_limit: float,
    method: str,
    value_col: str,
    extra_cols: dict | None = None,
) -> pd.DataFrame:
    """Filter outlier rows, add metadata columns, and sort by consumption."""
    outliers = df[
        (df[value_col] < lower_limit) |
        (df[value_col] > upper_limit)
    ].copy()

    outliers["method"] = method
    outliers["lower_limit"] = lower_limit
    outliers["upper_limit"] = upper_limit

    if extra_cols:
        for col_name, values in extra_cols.items():
            outliers[col_name] = values.loc[outliers.index]

    return outliers.sort_values(value_col, ascending=False)


def detect_outliers_3sigma(df: pd.DataFrame, value_col: str = "kWh") -> pd.DataFrame:
    """Detect outliers using mean ± 3 standard deviations."""
    mean_val = df[value_col].mean()
    std_val  = df[value_col].std()

    lower_limit = mean_val - 3 * std_val
    upper_limit = mean_val + 3 * std_val
    z_score = (df[value_col] - mean_val) / std_val

    return _apply_limits(
        df=df,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        method="3sigma",
        value_col=value_col,
        extra_cols={"z_score": z_score},
    )


def detect_outliers_iqr(df: pd.DataFrame, value_col: str = "kWh") -> pd.DataFrame:
    """Detect outliers using the IQR rule: Q1 - 1.5*IQR, Q3 + 1.5*IQR."""
    q1 = df[value_col].quantile(0.25)
    q3 = df[value_col].quantile(0.75)
    iqr = q3 - q1

    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    return _apply_limits(
        df=df,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        method="IQR",
        value_col=value_col,
    )


def calculate_outlier_summary(
    df: pd.DataFrame,
    outliers_3sigma: pd.DataFrame,
    outliers_iqr: pd.DataFrame,
    value_col: str = "kWh",
) -> pd.DataFrame:
    """Summarize outlier counts and percentages for both methods."""
    total_rows = len(df)
    skewness = df[value_col].skew()

    return pd.DataFrame({
        "method": ["3sigma", "IQR"],
        "total_rows": [total_rows, total_rows],
        "outlier_count": [len(outliers_3sigma), len(outliers_iqr)],
        "outlier_percentage": [
            100 * len(outliers_3sigma) / total_rows,
            100 * len(outliers_iqr) / total_rows,
        ],
        "skewness": [skewness, skewness],
    })
