"""
Reporting module for electricity consumption analysis.
Summarizes key stats, outlier findings, weather correlations, and discount plan recommendations into a Markdown report.
The report is saved to the `reports/` directory and includes:
- Overview of the data and consumption patterns
- Outlier detection results with tables
- Weather cross-analysis (if weather data is available)
- Discount plan recommendation based on the user's consumption pattern and the scraped discount offers
- A table of top discount scenarios with their estimated savings
- Links to generated visualizations (heatmaps, trends, etc.)

"""

from pathlib import Path
import pandas as pd
from datetime import datetime


def _fmt(value, digits=2):
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _small_table(
    df: pd.DataFrame,
    columns: list[str],
    n: int = 8,
) -> str:

    if df is None or df.empty:
        return "No rows available."

    small = df[
        [c for c in columns if c in df.columns]
    ].head(n).astype(str)

    widths = {
        c: max(int(v) if not pd.isna(v := small[c].str.len().max()) else 0, len(c))
        for c in small.columns
    }

    fmt = lambda row: "| " + " | ".join(
        str(row[c]).ljust(widths[c])
        for c in small.columns
    ) + " |"

    return "\n".join([
        fmt(dict(zip(small.columns, small.columns))),
        "|-" + "-|-".join(
            "-" * widths[c]
            for c in small.columns
        ) + "-|",
        *(fmt(r) for r in small.to_dict("records"))
    ])

def _extract_outlier_stats(outliers: pd.DataFrame, method: str):
    """Return count and percentage for one outlier method."""
    if outliers is None or outliers.empty or "method" not in outliers.columns:
        return 0, 0

    method_rows = outliers[outliers["method"].eq(method)]
    count = len(method_rows)

    if "outlier_percentage" in method_rows.columns:
        pct = method_rows["outlier_percentage"].iloc[0]
    else:
        pct = None

    return count, pct


def _format_outlier_lines(outliers: pd.DataFrame) -> str:
    """Build Markdown text for the outlier section."""
    if outliers is None or outliers.empty:
        return "No outliers were detected."

    lines = []

    for method in outliers["method"].dropna().unique():
        method_rows = outliers[outliers["method"].eq(method)]

        lines.append(f"### {method}")
        lines.append(f"- Outliers detected: **{len(method_rows)}**")

        if "lower_limit" in method_rows.columns:
            lines.append(f"- Lower limit: **{_fmt(method_rows['lower_limit'].iloc[0])}**")

        if "upper_limit" in method_rows.columns:
            lines.append(f"- Upper limit: **{_fmt(method_rows['upper_limit'].iloc[0])}**")

        top_cols = [
            "datetime",
            "date",
            "time",
            "weekday",
            "hour",
            "kWh",
            "z_score",
            "lower_limit",
            "upper_limit",
        ]

        lines.append("")
        lines.append("Top outlier rows:")
        lines.append("")
        lines.append(_small_table(method_rows, top_cols, n=8))
        lines.append("")

    return "\n".join(lines)

def write_report(summary: dict, 
                 outliers: pd.DataFrame, 
                 weather_summary: dict | None, 
                 scenarios: pd.DataFrame, 
                 recommendation: dict, 
                 report_dir: Path,
                 generated_plots: list[Path] | None = None) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "summary_report.md"


    # Build the outlier bullet list from every row in the summary DataFrame.
    
    outlier_lines = _format_outlier_lines(outliers)
    sigma_count, sigma_pct = _extract_outlier_stats(outliers, "3sigma")
    iqr_count,   iqr_pct   = _extract_outlier_stats(outliers, "IQR")


    # Discounts revised and summarized data
    top_cols = ["supplier_name", "plan_name", "discount_pct",
                "time_restriction", "requires_smart_meter", "eligibility",
                "matching_usage_share_pct", "weighted_discount_score"]
    
    # Local weather analysis
    weather_text = "Weather analysis was not run or failed."
    if weather_summary:
        weather_text = "\n".join([
            f"- Correlation kWh vs temperature: **{_fmt(weather_summary.get('corr_kWh_temp'), 3)}**",
            f"- Correlation kWh  vs humidity: **{_fmt(weather_summary.get('corr_kWh_humidity'), 3)}**",
            f"- Average temperature: **{_fmt(weather_summary.get('avg_temp'), 1)}°C**",
            f"- Avg kWh  during hot hours ≥30°C: **{_fmt(weather_summary.get('avg_kWh_hot_hours'), 3)}**",
            f"- Avg kWh  during cooler hours <30°C: **{_fmt(weather_summary.get('avg_kWh_normal_hours'), 3)}**",
        ])

    # The main report content    
    report_content = f"""# Electricity Consumption Analysis Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

- Data period: **{summary['start_date']} → {summary['end_date']}**
- Records analyzed: **{summary['records']:,}**
- Days analyzed: **{summary['days']}**
- Total measured consumption: **{_fmt(summary['total_kWh'])} kWh**
- Average daily consumption: **{_fmt(summary['avg_daily_kWh'])} kWh**
- Peak average hour: **{summary['peak_weekday']} {summary['peak_hour']:02d}:00** with **{_fmt(summary['peak_avg_kWh'])} kWh**

## Outliers


- Outlier tables saved to `data/processed/outliers_3sigma.csv` and `outliers_iqr.csv`.

## Weather Cross-Analysis

{weather_text}

## Discount Recommendation

Recommended plan based on measured consumption pattern and scraped discount table:

- Supplier: **{recommendation.get('supplier_name', 'N/A')}**
- Plan: **{recommendation.get('plan_name', 'N/A')}**
- Max discount: **{recommendation.get('discount_pct', 'N/A')}%**
- Time restriction: **{recommendation.get('time_restriction', 'N/A')}**
- Requires smart meter: **{recommendation.get('requires_smart_meter', 'N/A')}**
- Eligibility status: **{recommendation.get('eligibility', 'N/A')}**
- Share of your measured consumption matching plan hours: **{_fmt(recommendation.get('matching_usage_share_pct', 0))}%**

Important: this is not a bill forecast because tariff prices were not provided. The score applies the scraped discount percentage to measured kWh only.

## Top Discount Scenarios

{_small_table(scenarios, top_cols, 10)}

## Generated Visuals

  
- `outputs/html/heatmap_weekday_hour.html`
- `outputs/html/heatmap_variability_weekday_hour.html`
- `outputs/html/hourly_consumption_by_weekday.html`
- `outputs/html/daily_consumption_distribution.html`
- `outputs/html/daily_consumption_trend.html`
- `outputs/html/load_duration_curve.html`
- `outputs/html/outliers_timeline.html`
- `outputs/html/weather_temperature_scatter.html` if weather API succeeds
- `outputs/html/daily_consumption_vs_temperature.html` if weather API succeeds
"""
    # Dynamically inject side-by-side tariff comparisons using Path syntax
    if generated_plots is None:
        generated_plots = []
        
    if len(generated_plots) > 0:  
        report_content += "\n## Plan Comparison Matrices\n"
        report_content += "The visual profiles below map your actual household consumption matrix (left) next to the targeted supplier discount window structures (right).\n\n"
    
    for plot in generated_plots:
        # Safe Check: If it's a dictionary, extract the fields safely using .get()
        if isinstance(plot, dict):
            supplier = plot.get("supplier", "Unknown Supplier")
            plan = plot.get("plan", "Discount Plan")
            filename = plot.get("filename", "")
        else:
            # If it's just a raw string filename, handle it gracefully without crashing
            supplier = "Tariff Plan"
            plan = "Comparison"
            filename = str(plot)
            
        # Only append to markdown if a valid filename exists
        if filename:
            report_content += f"### {supplier} — {plan}\n"
            report_content += f"![Consumption vs Plan Tariff Matrix](figures/{filename})\n\n"
    
    report_path.write_text(report_content, encoding="utf-8")
    return report_path
