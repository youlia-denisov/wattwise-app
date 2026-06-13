"""
Reporting module for electricity consumption analysis.

Generates a Markdown summary report saved to report_dir/summary_report.md.
The report renders cleanly in the Streamlit Report tab, where it is
converted to HTML and styled by the .report-box CSS.

Sections
--------
- Overview        : key consumption stats for the period
- User Habits     : persona banner + insight cards (requires df_clean)
- Outliers        : count and share of unusual readings by method
- Weather         : temperature correlations (if weather data was collected)
- Best Plan       : top recommendation with eligibility details
- Top 10 Plans    : comparison table
"""

from pathlib import Path
from datetime import datetime

import pandas as pd

from src.features import derive_persona, build_insights
from src.loader import detect_kwh_col

# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt(value, digits=2):
    """Format a numeric value or return 'N/A' if missing."""
    try:
        if pd.isna(value):
            return "N/A"
        return "{:.{}f}".format(float(value), digits)
    except Exception:
        return str(value)


def _gfm_table(df, columns, n=10):
    """Render a GitHub-Flavored Markdown table (n rows, selected columns)."""
    cols = [c for c in columns if c in df.columns]
    if not cols or df.empty:
        return "_No data available._"

    rename = {
        "supplier_name":            "Supplier",
        "plan_name":                "Plan",
        "discount_pct":             "Discount %",
        "time_restriction":         "Time window",
        "requires_smart_meter":     "Smart meter?",
        "matching_usage_share_pct": "Your usage match %",
    }
    subset = df[cols].head(n).copy()
    subset = subset.rename(
        columns={c: rename.get(c, c.replace("_", " ").title()) for c in cols}
    )

    header = "| " + " | ".join(subset.columns) + " |"
    sep    = "| " + " | ".join(["---"] * len(subset.columns)) + " |"
    rows   = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in subset.itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep] + rows)


# ── User Habits section ───────────────────────────────────────────────────────

_STATUS_COLORS = {"good": "#21c354", "warn": "#e6a817", "info": "#1c83e1"}


def _card_html(icon, title, value, desc, color):
    return (
        '<div style="background:#fff;border-radius:10px;'
        'box-shadow:0 1px 5px rgba(0,0,0,0.08);padding:14px 16px;'
        'border-left:4px solid ' + color + ';margin-bottom:0;">'
        '<div style="font-size:1.15rem;">' + icon + ' <strong>' + title + '</strong></div>'
        '<div style="font-size:1.05rem;font-weight:600;margin:4px 0;">' + value + '</div>'
        '<div style="font-size:0.88rem;color:#666;">' + desc + '</div>'
        '</div>'
    )


def _habits_html(df_clean):
    """Compute user features and return styled HTML. Returns empty string on failure."""
    if df_clean is None:
        return ""
    try:
        from src.features import build_user_features
        df = df_clean.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        if "kWh" not in df.columns:
            kwh_col = detect_kwh_col(df)
            if kwh_col is None:
                return ""
            df = df.rename(columns={kwh_col: "kWh"})
        f = build_user_features(df)
    except Exception:
        return ""

    persona  = derive_persona(f)
    insights = build_insights(f)
    color    = persona["color"]

    banner = (
        '<div style="background:' + color + '22;border-left:5px solid ' + color + ';'
        'border-radius:10px;padding:18px 22px;margin:16px 0 20px 0;">'
        '<div style="font-size:2rem;line-height:1;">' + persona["emoji"] + '</div>'
        '<div style="font-size:1.3rem;font-weight:700;margin-top:6px;">' + persona["label"] + '</div>'
        '<div style="font-size:0.97rem;color:#555;margin-top:4px;">' + persona["tagline"] + '</div>'
        '</div>'
    )

    card_els = [
        _card_html(ins["icon"], ins["title"], ins["value"], ins["desc"],
                   _STATUS_COLORS.get(ins["status"], "#888"))
        for ins in insights[:6]
    ]

    rows = []
    for i in range(0, len(card_els), 2):
        pair = card_els[i:i+2]
        if len(pair) == 2:
            rows.append(
                '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px;">'
                + pair[0] + pair[1] + '</div>'
            )
        else:
            rows.append('<div style="margin-bottom:12px;">' + pair[0] + '</div>')

    return banner + "\n".join(rows)


# ── Outlier section ───────────────────────────────────────────────────────────

def _outlier_section(outlier_summary):
    """Build the outlier paragraph from the summary DataFrame."""
    if outlier_summary is None or outlier_summary.empty:
        return "_No outlier data available._"

    lines = []
    for _, row in outlier_summary.iterrows():
        method = row.get("method", "Unknown")
        count  = int(row.get("outlier_count", 0))
        pct    = _fmt(row.get("outlier_percentage"), 1)
        total  = int(row.get("total_rows", 0))
        label  = "3-Sigma" if str(method).lower() == "3sigma" else str(method).upper()
        lines.append(
            "**{}** — {:,} unusual readings out of {:,} ({}%)".format(
                label, count, total, pct
            )
        )

    skew_note = ""
    if "skewness" in outlier_summary.columns:
        skew_val = outlier_summary["skewness"].iloc[0]
        if not pd.isna(skew_val):
            direction = (
                "right-skewed (occasional high-use spikes)" if skew_val > 0.5
                else "left-skewed (occasional dips)"        if skew_val < -0.5
                else "roughly symmetric"
            )
            skew_note = "\n\nDistribution skew: **{}** ({}).".format(
                _fmt(skew_val, 2), direction
            )

    return "\n\n".join(lines) + skew_note


# ── Weather section ───────────────────────────────────────────────────────────

def _weather_section(weather_summary):
    """Build a Markdown table for the weather section, or a fallback string."""
    if not weather_summary:
        return "_Weather analysis was not run or data was unavailable._"

    def _corr_label(v_str):
        try:
            v = float(v_str)
        except Exception:
            return ""
        if abs(v) >= 0.6:
            return " _(strong)_"
        if abs(v) >= 0.3:
            return " _(moderate)_"
        return " _(weak)_"

    corr_temp  = _fmt(weather_summary.get("corr_kWh_temp"),        3)
    corr_hum   = _fmt(weather_summary.get("corr_kWh_humidity"),    3)
    avg_temp   = _fmt(weather_summary.get("avg_temp"),             1)
    hot_kwh    = _fmt(weather_summary.get("avg_kWh_hot_hours"),    3)
    normal_kwh = _fmt(weather_summary.get("avg_kWh_normal_hours"), 3)

    rows = [
        ("Correlation with temperature",        corr_temp  + _corr_label(corr_temp)),
        ("Correlation with humidity",            corr_hum   + _corr_label(corr_hum)),
        ("Average temperature in dataset",       avg_temp   + " C"),
        ("Avg use during hot hours (>= 30 C)",   hot_kwh    + " kWh"),
        ("Avg use during cooler hours (< 30 C)", normal_kwh + " kWh"),
    ]
    lines = ["| Metric | Value |", "| --- | --- |"]
    lines += ["| {} | {} |".format(k, v) for k, v in rows]
    return "\n".join(lines)


# ── Main entry point ──────────────────────────────────────────────────────────

def write_report(
    summary,
    outliers,
    weather_summary,
    scenarios,
    recommendation,
    report_dir,
    df_clean=None,
    generated_plots=None,
):
    """
    Write a Markdown summary report to report_dir/summary_report.md.

    Parameters
    ----------
    summary         : dict from compute_summary()
    outliers        : outlier_summary DataFrame from calculate_outlier_summary()
    weather_summary : dict from summarize_weather(), or None
    scenarios       : discount_scenarios DataFrame
    recommendation  : best-match plan dict from choose_recommendation()
    report_dir      : destination folder (created if missing)
    df_clean        : cleaned DataFrame — used to compute and render the habits section
    generated_plots : accepted but ignored (charts cannot render in the report tab)
    """
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "summary_report.md"

    # ── Build variable sections ───────────────────────────────────────────────
    habits_html  = _habits_html(df_clean)
    outlier_text = _outlier_section(outliers)
    weather_text = _weather_section(weather_summary)

    scenario_cols = [
        "supplier_name", "plan_name", "discount_pct",
        "time_restriction", "requires_smart_meter",
    ]
    # Add clickable links to supplier names using company_page (per supplier)
    # or deal_url (per plan) if available in the scenarios DataFrame.
    scenarios_display = scenarios.copy()
    url_col = next((c for c in ["company_page", "deal_url"] if c in scenarios.columns), None)
    if url_col:
        def _linked(row):
            url = str(row.get(url_col, ""))
            name = str(row.get("supplier_name", ""))
            return "[{}]({})".format(name, url) if url.startswith("http") else name
        scenarios_display["supplier_name"] = scenarios_display.apply(_linked, axis=1)
    top_scenarios_table = _gfm_table(scenarios_display, scenario_cols, n=10)

    rec = recommendation
    rec_supplier = rec.get("supplier_name",          "N/A")
    rec_plan     = rec.get("plan_name",              "N/A")
    rec_discount = rec.get("discount_pct",           "N/A")
    rec_window   = rec.get("time_restriction",       "N/A")
    rec_smart    = rec.get("requires_smart_meter",   "N/A")
    rec_eligible = rec.get("eligibility",            "N/A")
    rec_match    = _fmt(rec.get("matching_usage_share_pct", 0), 1)

    # ── Compose report as a list of sections ─────────────────────────────────
    parts = []

    parts.append("# Electricity Consumption Report\n")
    parts.append("_Generated {}_\n".format(
        datetime.now().strftime("%d %B %Y, %H:%M")
    ))
    parts.append("---\n")

    # Overview
    parts.append("## Overview\n")
    overview_rows = [
        ("Period",            "{} to {}".format(summary["start_date"], summary["end_date"])),
        ("Days analysed",     str(summary["days"])),
        ("Total readings",    "{:,}".format(summary["records"])),
        ("Total consumption", "{} kWh".format(_fmt(summary["total_kWh"]))),
        ("Average daily use", "{} kWh/day".format(_fmt(summary["avg_daily_kWh"]))),
        ("Busiest hour",      "{} at {:02d}:00 ({} kWh avg)".format(
            summary["peak_weekday"], summary["peak_hour"], _fmt(summary["peak_avg_kWh"])
        )),
    ]
    tbl = ["| | |", "| --- | --- |"]
    tbl += ["| {} | {} |".format(k, v) for k, v in overview_rows]
    parts.append("\n".join(tbl) + "\n")
    parts.append("---\n")

    # User Habits — HTML markers let report.py render this block directly
    if habits_html:
        parts.append("## Your Household Profile\n")
        parts.append("<!-- RAW_HTML_START -->\n")
        parts.append(habits_html + "\n")
        parts.append("<!-- RAW_HTML_END -->\n")
        parts.append("---\n")

    # Outliers
    parts.append("## Unusual Readings (Outliers)\n")
    parts.append(
        "Outliers are hourly readings that fall unusually far from your typical pattern. "
        "A small number is normal — they often reflect holidays, parties, or appliance faults.\n"
    )
    parts.append(outlier_text + "\n")
    parts.append("---\n")

    # Weather
    parts.append("## Weather Analysis\n")
    parts.append(weather_text + "\n")
    parts.append("---\n")

    # Best plan
    parts.append("## Best Discount Plan for You\n")
    parts.append("Based on your actual usage pattern, the best matching plan is:\n")
    plan_rows = [
        ("Supplier",                            rec_supplier),
        ("Plan",                                rec_plan),
        ("Max discount",                        "{}%".format(rec_discount)),
        ("Discount window",                     rec_window),
        ("Requires smart meter",                str(rec_smart)),
        ("Your eligibility",                    rec_eligible),
        ("Share of your use in discount hours", "{}%".format(rec_match)),
    ]
    tbl2 = ["| | |", "| --- | --- |"]
    tbl2 += ["| {} | {} |".format(k, v) for k, v in plan_rows]
    parts.append("\n".join(tbl2) + "\n")
    parts.append(
        "> **Note:** Savings are estimated from measured kWh and the scraped discount "
        "percentage. Actual bill savings depend on your current tariff rate.\n"
    )
    parts.append("---\n")

    # Top 10 plans
    parts.append("## Top 10 Plans Compared\n")
    parts.append(top_scenarios_table + "\n")

    report_path.write_text("\n".join(parts), encoding="utf-8")
    return report_path
