"""Download/read raw Vykhodnoy GIC and download Wind L1 solar-wind data.

GIC is downloaded automatically from the public GIC website into:

    data/raw/GIC

The Vykhodnoy product used here covers 2012-01-01 through 2022-12-31,
has a 2 Hz sampling rate, and has a +/-120 A dynamic range.

Wind MFI/SWE ASCII files are downloaded from the Wind science data service
into ``data/raw/Wind_L1``.

This file does not create processed Parquet files. It only downloads raw GIC,
reads raw GIC records into pandas DataFrames one source file at a time,
and does not resample, interpolate, clean, or merge them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator
from datetime import date, datetime, timedelta
import time
from dateutil.relativedelta import relativedelta

import pandas as pd
import requests
import argparse


# This section defines data folders, the GIC website URL, and the requested period.
# Keeping paths relative to this script makes the code work on another machine.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT
GIC_DIR = SCRIPT_DIR / "data" / "raw" / "GIC"
GIC_SUFFIXES = {".txt", ".dat", ".csv"}
GIC_COLUMNS = ["year", "month", "day", "hour", "minute", "second", "gic_a"]
GIC_BASE_URL = "http://gic.en51.ru/data/vkh_gic"
GIC_START_DATE = date(2012, 1, 1)
GIC_END_DATE = date(2022, 12, 31)
DOWNLOAD_TIMEOUT_SECONDS = (20, 180)
DOWNLOAD_RETRIES = 3

WIND_START_DATE = datetime(2012, 1, 1)
WIND_END_DATE = datetime(2022, 12, 31)
WIND_BASE_URL = "https://wind.nasa.gov/mfi_swe_display_results.php"
WIND_DIR = SCRIPT_DIR / "data" / "raw" / "Wind_L1"
WIND_MFI_PARAMS = {
    "output": "ascii", "system": "gsm", "time1": "1_min", "time2": "92_sec",
    "p01": "b", "p02": "bx", "p03": "by", "p04": "bz",
}
WIND_SWE_PARAMS = {
    "output": "ascii", "system": "gsm", "time1": "1_min", "time2": "92_sec",
    "p07": "v", "p08": "vx", "p09": "vy", "p10": "vz", "p11": "vtheta",
    "p12": "vphi", "p13": "np", "p14": "vth", "p15": "psw", "p16": "pb",
    "p17": "pth", "p18": "beta", "p19": "va", "p20": "ma", "p21": "mms",
    "p22": "epsilon", "p23": "vbs",
}


# This function lists source files in a stable order.
# The files are not copied or changed; only their paths are returned.
def find_files(folder: Path, suffixes: set[str]) -> list[Path]:
    """Return all supported files below *folder*, ordered by path."""

    if not folder.exists():
        raise FileNotFoundError(
            f"Input folder does not exist: {folder}. Create it and put the raw files there."
        )
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


# This section downloads one daily Vykhodnoy raw GIC file.
# Existing non-empty files are skipped, so rerunning the script can continue safely.
def download_gic_file(day: date, output_folder: Path) -> str:
    """Download one official YYYYMMDD.txt file and return its status."""

    year_month = f"{day.year}-{day.month:02d}"
    filename = f"{day:%Y%m%d}.txt"
    url = f"{GIC_BASE_URL}/{day.year}/{year_month}/{filename}"
    output_path = output_folder / str(day.year) / year_month / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and output_path.stat().st_size > 0:
        return "skipped"

    temporary_path = output_path.with_name(output_path.name + ".part")
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
                if response.status_code == 404:
                    return "missing"
                response.raise_for_status()
                with temporary_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if temporary_path.stat().st_size == 0:
                raise OSError("server returned an empty file")
            temporary_path.replace(output_path)
            return "downloaded"
        except Exception as exc:
            if attempt == DOWNLOAD_RETRIES:
                print(f"GIC download failed: {url} ({exc})")
            else:
                print(f"GIC download retry {attempt}/{DOWNLOAD_RETRIES}: {url} ({exc})")
    if temporary_path.exists():
        temporary_path.unlink()
    return "failed"


# This section downloads every day in the requested 2012-2022 interval.
# The raw 2 Hz files are saved only in data/raw/GIC; no one-minute aggregation is done.
def download_gic_data(
    output_folder: Path = GIC_DIR,
    start_day: date = GIC_START_DATE,
    end_day: date = GIC_END_DATE,
) -> dict[str, int]:
    """Download the inclusive daily Vykhodnoy date range."""

    output_folder.mkdir(parents=True, exist_ok=True)
    counts = {"downloaded": 0, "skipped": 0, "missing": 0, "failed": 0}
    current = start_day
    total_days = (end_day - start_day).days + 1
    number = 0
    while current <= end_day:
        number += 1
        status = download_gic_file(current, output_folder)
        counts[status] += 1
        if number == 1 or number % 100 == 0 or current == end_day:
            print(f"GIC download progress: {number}/{total_days} days")
        current += timedelta(days=1)
    return counts


# This section reads one official Vykhodnoy text file.
# Each row keeps its original GIC value; timestamp_utc is added only for later alignment.
def read_gic_file(path: Path) -> pd.DataFrame:
    """Read one Vykhodnoy file with columns year ... second and GIC (A)."""

    if path.suffix.lower() == ".csv":
        first_line = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()[0]
        has_header = any(word in first_line.lower() for word in ("year", "second", "gic"))
        if has_header:
            raw = pd.read_csv(path, comment="#", low_memory=False)
            names = {str(column).strip().lower(): column for column in raw.columns}
            if "gic_a" not in names and "gic" in names:
                names["gic_a"] = names["gic"]
            missing = [column for column in GIC_COLUMNS if column not in names]
            if missing:
                raise ValueError(f"GIC file {path.name} is missing columns: {missing}")
            raw = raw[[names[column] for column in GIC_COLUMNS]]
            raw.columns = GIC_COLUMNS
        else:
            raw = pd.read_csv(path, header=None, names=GIC_COLUMNS, usecols=range(7))
    else:
        # The website format is whitespace-delimited and has no header row.
        raw = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=GIC_COLUMNS,
            usecols=range(7),
            comment="#",
            engine="c",
            on_bad_lines="skip",
        )

    # Convert the seven source columns to numbers. Invalid rows become NaN and
    # are removed only when no usable timestamp can be constructed.
    for column in GIC_COLUMNS:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    calendar_date = pd.to_datetime(
        {"year": raw["year"], "month": raw["month"], "day": raw["day"]},
        errors="coerce",
        utc=True,
    )
    timestamp = (
        calendar_date
        + pd.to_timedelta(raw["hour"], unit="h")
        + pd.to_timedelta(raw["minute"], unit="m")
        + pd.to_timedelta(raw["second"], unit="s")
    )
    valid = timestamp.notna()
    result = raw.loc[valid].copy()
    result.insert(0, "timestamp_utc", timestamp.loc[valid])
    result.insert(1, "source_file", path.name)
    return result.reset_index(drop=True)


# This generator reads GIC files one at a time.
# It avoids concatenating the entire 2012-2022 2 Hz record set into memory.
def read_gic_data(folder: Path = GIC_DIR) -> Iterator[tuple[Path, pd.DataFrame]]:
    """Yield ``(file_path, dataframe)`` for every raw GIC file."""

    for path in find_files(folder, GIC_SUFFIXES):
        yield path, read_gic_file(path)


def fetch_wind_data(start_dt: datetime, end_dt: datetime, params: dict[str, str], data_type: str) -> str | None:
    payload = params.copy()
    payload.update({"startdate": start_dt.strftime("%Y%m%d"), "starttime": "000000",
                    "enddate": end_dt.strftime("%Y%m%d"), "endtime": "235959"})
    print(f"[{data_type}] request: {start_dt:%Y-%m-%d} ~ {end_dt:%Y-%m-%d}")
    try:
        response = requests.post(WIND_BASE_URL, data=payload, timeout=60)
        response.raise_for_status()
        text = response.text
        if "<html" in text.lower() or "error" in text.lower():
            print(f"[{data_type}] server returned an error response")
            return None
        return text
    except Exception as exc:
        print(f"[{data_type}] request failed: {exc}")
        return None


def save_wind_data(text: str, start_dt: datetime, end_dt: datetime, data_type: str) -> None:
    save_dir = WIND_DIR / ("MFI" if data_type == "MFI" else "SWE")
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = (f"{start_dt:%Y%m}_{data_type}.txt" if start_dt.year == end_dt.year and start_dt.month == end_dt.month
                else f"{start_dt:%Y%m}_{end_dt:%Y%m}_{data_type}.txt")
    (save_dir / filename).write_text(text, encoding="utf-8")
    print(f"[{data_type}] saved: {save_dir / filename}")


def download_wind_range(start_dt: datetime, end_dt: datetime, params: dict[str, str], data_type: str, retry_daily: bool = False) -> None:
    if retry_daily:
        current = start_dt
        while current <= end_dt:
            day_end = min(current + timedelta(days=1) - timedelta(microseconds=1), end_dt)
            data = fetch_wind_data(current, day_end, params, data_type)
            if data:
                save_wind_data(data, current, day_end, data_type)
            current = day_end + timedelta(microseconds=1)
            time.sleep(1)
        return
    data = fetch_wind_data(start_dt, end_dt, params, data_type)
    if data:
        save_wind_data(data, start_dt, end_dt, data_type)
    else:
        download_wind_range(start_dt, end_dt, params, data_type, retry_daily=True)


def download_wind_data(start_dt: datetime = WIND_START_DATE, end_dt: datetime = WIND_END_DATE) -> None:
    current = start_dt
    while current <= end_dt:
        next_month = current + relativedelta(months=1)
        month_end = min(next_month - timedelta(microseconds=1), end_dt)
        download_wind_range(current, month_end, WIND_MFI_PARAMS, "MFI")
        time.sleep(2)
        download_wind_range(current, month_end, WIND_SWE_PARAMS, "SWE")
        time.sleep(2)
        current = next_month
        if current <= end_dt:
            time.sleep(3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download GIC and Wind source files.")
    args = parser.parse_args()
    if not args.download:
        parser.print_help()
        raise SystemExit(0)
    download_summary = download_gic_data()
    print(f"GIC download complete: {download_summary}")

    gic_count = 0
    gic_rows = 0
    for file_path, data in read_gic_data():
        gic_count += 1
        gic_rows += len(data)

    print(f"Read complete: GIC files={gic_count}, GIC rows={gic_rows:,}")
    download_wind_data()
