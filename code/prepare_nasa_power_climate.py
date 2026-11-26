from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"D:\project\shuozhouinflucefactor")
DATA_DIR = ROOT / "data" / "climate" / "nasa_power"
RAW_CSV = DATA_DIR / "nasa_power_daily_202211_202512.csv"
DAILY_CSV = DATA_DIR / "climate_daily_clean_202211_202512.csv"
MONTHLY_CSV = DATA_DIR / "climate_monthly_202211_202512.csv"
METADATA_JSON = DATA_DIR / "climate_metadata.json"


def read_power_csv() -> pd.DataFrame:
    lines = RAW_CSV.read_text(encoding="utf-8").splitlines()
    header_index = next(i for i, line in enumerate(lines) if line.startswith("YEAR,"))
    frame = pd.read_csv(RAW_CSV, skiprows=header_index)
    frame.columns = [column.strip() for column in frame.columns]
    frame = frame.rename(
        columns={
            "T2M": "tmean_c",
            "T2M_MAX": "tmax_c",
            "T2M_MIN": "tmin_c",
            "PRECTOTCORR": "precip_mm_day",
        }
    )
    if {"MO", "DY"}.issubset(frame.columns):
        frame["date"] = pd.to_datetime(
            dict(year=frame["YEAR"], month=frame["MO"], day=frame["DY"])
        )
    else:
        frame["date"] = pd.to_datetime(
            frame["YEAR"].astype(str) + frame["DOY"].astype(str).str.zfill(3),
            format="%Y%j",
        )
    columns = ["date", "tmean_c", "tmax_c", "tmin_c", "precip_mm_day"]
    frame = frame[columns].replace(-999, np.nan)
    expected = pd.date_range("2022-11-01", "2025-12-31", freq="D")
    frame = frame.set_index("date").reindex(expected).rename_axis("date").reset_index()
    return frame


def monthly_summary(daily: pd.DataFrame) -> pd.DataFrame:
    source = daily.set_index("date")
    monthly = source.resample("MS").agg(
        tmean_c=("tmean_c", "mean"),
        tmax_c=("tmax_c", "max"),
        tmin_c=("tmin_c", "min"),
        precip_mm=("precip_mm_day", "sum"),
        rainy_days_ge1mm=("precip_mm_day", lambda values: int((values >= 1.0).sum())),
        valid_temp_days=("tmean_c", "count"),
        valid_precip_days=("precip_mm_day", "count"),
    )
    monthly.index = monthly.index.strftime("%Y-%m")
    monthly.index.name = "month"
    return monthly.reset_index()


def make_preview(daily: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), dpi=150, sharex=True)
    axes[0].plot(daily["date"], daily["tmean_c"], color="#d95f02", linewidth=0.7)
    axes[0].set_ylabel("Daily mean temperature (°C)")
    axes[0].grid(alpha=0.25)
    axes[1].bar(
        daily["date"],
        daily["precip_mm_day"],
        width=1.0,
        color="#2b8cbe",
        linewidth=0,
    )
    axes[1].set_ylabel("Daily precipitation (mm)")
    axes[1].set_xlabel("Date (UTC)")
    axes[1].grid(alpha=0.25)
    fig.suptitle("NASA POWER climate series at study-area centroid")
    fig.tight_layout()
    fig.savefig(DATA_DIR / "climate_daily_preview.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    daily = read_power_csv()
    monthly = monthly_summary(daily)
    daily.to_csv(DAILY_CSV, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    monthly.to_csv(MONTHLY_CSV, index=False, encoding="utf-8-sig")
    make_preview(daily)

    metadata = {
        "source": "NASA POWER Daily API",
        "source_url": "https://power.larc.nasa.gov/api/temporal/daily/point",
        "location": {
            "description": "centroid of the 1 km Great Wall buffer study area",
            "longitude": 112.794431,
            "latitude": 39.201233,
        },
        "time_range": ["2022-11-01", "2025-12-31"],
        "time_standard": "UTC",
        "variables": {
            "tmean_c": "2 m daily mean air temperature, degree Celsius",
            "tmax_c": "2 m daily maximum air temperature, degree Celsius",
            "tmin_c": "2 m daily minimum air temperature, degree Celsius",
            "precip_mm_day": "corrected daily precipitation, mm/day",
        },
        "spatial_note": "The study area is much smaller than the native meteorological grid, so a centroid time series is used; this is a temporal driver, not a spatial susceptibility factor.",
        "daily_rows": int(len(daily)),
        "monthly_rows": int(len(monthly)),
        "missing_daily_values": {
            column: int(daily[column].isna().sum())
            for column in ["tmean_c", "tmax_c", "tmin_c", "precip_mm_day"]
        },
    }
    METADATA_JSON.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
