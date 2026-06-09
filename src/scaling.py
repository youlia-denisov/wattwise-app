"""
Scaling, in the case of my data was needed to overcome right skew (2.3% of samples over 3sigma threshold)
I use RobustScaler to treat the data

"""

import pandas as pd
import plotly.express as px
from sklearn.preprocessing import RobustScaler

from config import PROCESSED_DIR, HTML_DIR


def scale_kWh_by_weekday_hour(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["kWh_scaled"] = (
        df.groupby(["weekday", "hour"])["kWh"]
        .transform(
            lambda x: RobustScaler()
            .fit_transform(x.to_frame())
            .ravel()
        )
    )

    return df

def save_heatmap(df, value_col, title, output_file):
    pivot = df.pivot_table(
        index="weekday",
        columns="hour",
        values=value_col,
        aggfunc="mean"
    )

    fig = px.imshow(
        pivot,
        aspect="auto",
        title=title,
        labels={
            "x": "Hour",
            "y": "Weekday",
            "color": value_col
        }
    )

    fig.update_layout(width=1100, height=600)
    fig.write_html(output_file)
    fig.show()


def main() -> None:
    input_path = PROCESSED_DIR / "cleaned_consumption.csv"
    output_path = PROCESSED_DIR / "cleaned_consumption_scaled.csv"

    HTML_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, parse_dates=["datetime"])

    df_scaled = scale_kWh_by_weekday_hour(df)

    df_scaled.to_csv(output_path, index=False)

    save_heatmap(
        df_scaled,
        "kWh",
        "Before RobustScaler — Average kWh by Weekday and Hour",
        HTML_DIR / "heatmap_before_scaling.html"
    )

    save_heatmap(
        df_scaled,
        "kWh_scaled",
        "After RobustScaler — Average Scaled kWh by Weekday and Hour",
        HTML_DIR / "heatmap_after_scaling.html"
    )

    print(f"Saved scaled data to: {output_path}")
    print(f"Saved heatmaps to: {HTML_DIR}")


if __name__ == "__main__":
    main()
    