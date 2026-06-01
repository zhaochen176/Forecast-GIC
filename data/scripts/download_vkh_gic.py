"""Download and aggregate public VKH GIC observations.

The public database stores one text file per day. This script downloads the
files and aggregates the 1-second GIC records to 1-minute values by keeping the
sample with the largest absolute magnitude in each minute.

Output:
    data/interim/vkh_gic_2012_2022_1min.parquet
"""

from __future__ import annotations

import argparse
import calendar
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


DEFAULT_BASE_URL = "http://gic.en51.ru/data/lkh_gic/"
SENTINEL_VALUE = 99999.9
INVALID_ABS_THRESHOLD = 1e4


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def download_file(url: str, path: Path, timeout: int, retries: int) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return True

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            if response.status_code == 404:
                return False
            response.raise_for_status()
            with path.open("wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
            return True
        except requests.RequestException as exc:
            print(f"[download] {url} failed ({attempt}/{retries}): {exc}")
            time.sleep(1.0)
    return False


def download_daily_files(
    raw_dir: Path,
    start_year: int,
    end_year: int,
    base_url: str,
    timeout: int,
    retries: int,
) -> None:
    total = ok = skipped = failed = 0
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            month_label = f"{year}-{month:02d}"
            for day in range(1, calendar.monthrange(year, month)[1] + 1):
                total += 1
                filename = f"{year}{month:02d}{day:02d}.txt"
                local_path = raw_dir / str(year) / month_label / filename
                if local_path.exists() and local_path.stat().st_size > 0:
                    skipped += 1
                    continue
                url = f"{base_url}{year}/{month_label}/{filename}"
                if download_file(url, local_path, timeout=timeout, retries=retries):
                    ok += 1
                else:
                    failed += 1
                    if local_path.exists():
                        local_path.unlink()
    print(f"[download] total={total}, ok={ok}, skipped={skipped}, failed={failed}")


def sanitize_gic(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype("float32")
    bad = np.isclose(values, SENTINEL_VALUE, atol=1.0) | (
        np.abs(values) >= INVALID_ABS_THRESHOLD
    )
    return values.mask(bad, np.nan)


def aggregate_to_minute(raw_dir: Path, output_path: Path) -> pd.DataFrame:
    parts = []
    for path in sorted(raw_dir.rglob("*.txt")):
        frame = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=["year", "month", "day", "hour", "minute", "second", "gic"],
        )
        frame["gic"] = sanitize_gic(frame["gic"])
        frame = frame.dropna(subset=["gic"])
        if frame.empty:
            continue

        timestamp = pd.to_datetime(frame[["year", "month", "day", "hour", "minute"]])
        timestamp = timestamp + pd.to_timedelta(frame["second"], unit="s")
        frame = pd.DataFrame({"timestamp": timestamp, "gic": frame["gic"]})
        frame["minute"] = frame["timestamp"].dt.floor("min")
        frame["abs_gic"] = frame["gic"].abs()
        idx = frame.groupby("minute")["abs_gic"].idxmax()
        minute_frame = (
            frame.loc[idx, ["minute", "gic"]]
            .set_index("minute")
            .sort_index()
            .rename(columns={"gic": "gic"})
        )
        parts.append(minute_frame)

    if not parts:
        raise RuntimeError(f"No readable VKH GIC text files found under {raw_dir}")

    out = pd.concat(parts).sort_index()
    out = out.groupby(level=0)["gic"].apply(lambda x: x.iloc[np.argmax(np.abs(x))])
    out = out.to_frame("gic").astype("float32")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path)
    print(f"[aggregate] rows={len(out):,}, output={output_path}")
    return out


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2012)
    parser.add_argument("--end-year", type=int, default=2022)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--raw-dir", type=Path, default=root / "data" / "raw" / "vkh_gic")
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "interim" / "vkh_gic_2012_2022_1min.parquet",
    )
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--skip-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_download:
        download_daily_files(
            raw_dir=args.raw_dir,
            start_year=args.start_year,
            end_year=args.end_year,
            base_url=args.base_url,
            timeout=args.timeout,
            retries=args.retries,
        )
    aggregate_to_minute(args.raw_dir, args.output)


if __name__ == "__main__":
    main()
