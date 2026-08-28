"""Preprocess Vykhodnoy GIC and Wind L1 solar-wind data."""

from __future__ import annotations

from pathlib import Path
import re
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 固定输入、输出路径和时间范围。
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
GIC_DIR = DATA_DIR / "raw" / "GIC"
WIND_MFI_DIR = DATA_DIR / "raw" / "Wind_L1" / "MFI"
WIND_SWE_DIR = DATA_DIR / "raw" / "Wind_L1" / "SWE"
OUTPUT_DIR = DATA_DIR / "processed" / "preprocessing_outputs"
GIC_OUTPUT_DIR = OUTPUT_DIR / "GIC"
WIND_OUTPUT_DIR = OUTPUT_DIR / "Wind"
GIC_1MIN_OUTPUT = DATA_DIR / "processed" / "GIC_1min_2012_2022.csv"
WIND_1MIN_OUTPUT = DATA_DIR / "processed" / "Wind_L1_1min_2012_2022.csv"
AUDIT_OUTPUT_DIR = OUTPUT_DIR / "time_audit"
START_TIME = pd.Timestamp("2012-01-01 00:00:00")
END_TIME = pd.Timestamp("2022-12-31 23:59:00")
GIC_MAX_A = 120.0
GIC_SENTINELS = (9.99, 99.99, 999.99, 9999.99, 999.9, 9999.9, 99999.9)
GIC_COLUMNS = ["year", "month", "day", "hour", "minute", "second", "gic"]
GIC_FILE_SUFFIXES = {".txt", ".dat", ".csv"}
GIC_PLOT_EVENT_DATES = (
    pd.Timestamp("2013-06-07"),
    pd.Timestamp("2017-05-28"),
    pd.Timestamp("2021-03-01"),
)
DIST_EDGES = np.linspace(-GIC_MAX_A, GIC_MAX_A, 2401)
ABS_HIST_EDGES = np.arange(0.0, 6.0, 1.0)
MFI_SKIPROWS = 250
SWE_SKIPROWS = 268
WIND_FILE_SUFFIXES = {".txt", ".dat", ".csv"}
WIND_TIME_COLUMNS = {"year", "doy", "millisecs"}


# 识别由连续数字 9 构成的哨兵值，避免误删 19.2、399 等正常值。
def repeated_nine_mask(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip().str.replace('"', "", regex=False)
    mantissa = text.str.lower().str.split("e").str[0]
    all_nines = text.str.fullmatch(r"[+-]?9+(?:\.9+)?(?:e[+-]?\d+)?", na=False)
    nine_count = mantissa.str.count("9")
    decimal_form = mantissa.str.contains(".", regex=False) & nine_count.ge(3)
    large_integer_form = ~mantissa.str.contains(".", regex=False) & nine_count.ge(4)
    return (all_nines & (decimal_form | large_integer_form)).fillna(False)


def missing_intervals(
    times: pd.DatetimeIndex,
    missing: np.ndarray | pd.Series,
    min_duration_min: int = 60,
) -> pd.DataFrame:
    """Summarize contiguous missing periods on an explicitly supplied time axis."""
    mask = np.asarray(missing, dtype=bool)
    if len(times) != len(mask):
        raise ValueError("Time axis and missing mask lengths differ.")
    transitions = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    starts, stops = transitions[::2], transitions[1::2]
    durations = stops - starts
    keep = durations >= min_duration_min
    return pd.DataFrame({
        "missing_start": times[starts[keep]],
        "missing_end": times[stops[keep] - 1],
        "duration_min": durations[keep],
    })


def file_date_from_name(path: Path) -> object:
    match = re.search(r"(?<!\d)(\d{8})(?!\d)", path.name)
    return pd.to_datetime(match.group(1), format="%Y%m%d", errors="coerce") if match else pd.NaT


# 读取一个官网日文件，保留每个 0.5 s 样本。
def read_gic_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            first = handle.readline().lower()
        if any(word in first for word in ("year", "second", "gic")):
            raw = pd.read_csv(path, comment="#", low_memory=False)
            names = {str(c).strip().lower(): c for c in raw.columns}
            if "gic" not in names and "gic_a" in names:
                names["gic"] = names["gic_a"]
            if any(c not in names for c in GIC_COLUMNS):
                raise ValueError(f"GIC header is incomplete: {path}")
            raw = raw[[names[c] for c in GIC_COLUMNS]]
            raw.columns = GIC_COLUMNS
        else:
            raw = pd.read_csv(path, header=None, names=GIC_COLUMNS, usecols=range(7))
    else:
        raw = pd.read_csv(
            path, sep=r"\s+", header=None, names=GIC_COLUMNS, usecols=range(7),
            comment="#", engine="c", on_bad_lines="skip"
        )

    for column in GIC_COLUMNS:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    calendar = pd.to_datetime(
        {"year": raw.year, "month": raw.month, "day": raw.day},
        errors="coerce"
    )
    timestamp = (
        calendar + pd.to_timedelta(raw.hour, unit="h")
        + pd.to_timedelta(raw.minute, unit="m")
        + pd.to_timedelta(raw.second, unit="s")
    )
    valid_time = timestamp.notna() & raw.hour.between(0, 23) & raw.minute.between(0, 59)
    frame = pd.DataFrame({"time": timestamp, "GIC": raw.gic})
    frame = frame.loc[valid_time].reset_index(drop=True)
    source_values = frame["GIC"].copy()
    values = pd.to_numeric(source_values, errors="coerce")
    sentinel = repeated_nine_mask(source_values).to_numpy(dtype=bool)
    for value in GIC_SENTINELS:
        sentinel |= np.isclose(values.abs(), value, atol=1e-6, rtol=0.0)
    invalid = values.isna() | ~np.isfinite(values) | (values.abs() > GIC_MAX_A) | sentinel
    # 删除异常单元格，保留时间行，再按清洗结果生成缺失标记。
    frame["GIC"] = values.mask(invalid)
    frame["missing_flag"] = frame["GIC"].isna().astype("int8")
    return frame


# 按日期流式遍历 GIC，避免把 11 年 2 Hz 数据同时放入内存。
def iter_gic_files() -> list[Path]:
    return sorted(
        path for path in GIC_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in GIC_FILE_SUFFIXES
    )


# 统计有效值分布和缺失比例，直方图边界在原始/1 min 数据间保持一致。
def update_gic_statistics(
    frame: pd.DataFrame,
    distribution_counts: np.ndarray,
    abs_counts: np.ndarray,
    totals: dict[str, int],
) -> None:
    totals["rows"] += len(frame)
    totals["missing"] += int(frame["missing_flag"].sum())
    values = frame.loc[frame["missing_flag"].eq(0), "GIC"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values):
        distribution_counts += np.histogram(values, DIST_EDGES)[0]
        abs_values = np.abs(values)
        abs_counts += np.histogram(abs_values, ABS_HIST_EDGES)[0]


# 选择每个显示区间内绝对值最大的原始样本，只用于绘图数据表，不改变处理数据。
def select_plot_points(frame: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(frame) <= max_points:
        return frame.copy()
    edges = np.linspace(0, len(frame), max_points + 1, dtype=int)
    selected = []
    for left, right in zip(edges[:-1], edges[1:]):
        block = frame.iloc[left:right]
        valid = block.loc[block["missing_flag"].eq(0)]
        selected.append(block.loc[[valid["GIC"].abs().idxmax()]] if len(valid) else block.iloc[[0]])
    return pd.concat(selected, ignore_index=True)


# 保存时序绘图表和 PNG 图；年尺度原始数据使用显示采样，日尺度保留全部样本。
def save_time_plot(frame: pd.DataFrame, stem: str, title: str) -> None:
    output_csv = GIC_OUTPUT_DIR / f"{stem}_plot_data.csv"
    output_png = GIC_OUTPUT_DIR / f"{stem}.png"
    frame.to_csv(output_csv, index=False, date_format="%Y-%m-%d %H:%M:%S.%f")
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(frame["time"], frame["GIC"], linewidth=0.45, color="tab:blue")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("GIC (A)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def write_selected_gic_event_plots() -> None:
    """Compare native-sampling and 1-minute GIC series for three paper events."""
    if not GIC_1MIN_OUTPUT.exists():
        raise FileNotFoundError(f"Selected-event plots require: {GIC_1MIN_OUTPUT}")
    GIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    minute = pd.read_csv(GIC_1MIN_OUTPUT, usecols=["time", "GIC", "missing_flag"], parse_dates=["time"])
    files = iter_gic_files()
    summary_rows = []
    for event_day in GIC_PLOT_EVENT_DATES:
        event_end = event_day + pd.Timedelta(days=1)
        date_key = event_day.strftime("%Y%m%d")
        matching_files = [path for path in files if file_date_from_name(path) == event_day]
        raw_parts = [
            read_gic_file(path).loc[lambda frame: frame["time"].ge(event_day) & frame["time"].lt(event_end)]
            for path in matching_files
        ]
        raw = pd.concat(raw_parts, ignore_index=True).sort_values("time") if raw_parts else pd.DataFrame(columns=["time", "GIC", "missing_flag"])
        one_minute = minute.loc[minute["time"].ge(event_day) & minute["time"].lt(event_end)].copy()
        raw_stem = f"raw_gic_{date_key}"
        minute_stem = f"gic_1min_{date_key}"
        raw.to_csv(GIC_OUTPUT_DIR / f"{raw_stem}_plot_data.csv", index=False, date_format="%Y-%m-%d %H:%M:%S.%f")
        one_minute.to_csv(GIC_OUTPUT_DIR / f"{minute_stem}_plot_data.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
        if raw.empty:
            expected_minute = pd.DataFrame(columns=["time", "raw_valid_sample_count", "raw_selected_GIC", "raw_abs_max_A"])
        else:
            raw_valid = raw.loc[raw["missing_flag"].eq(0)].copy()
            raw_valid["minute"] = raw_valid["time"].dt.floor("min")
            raw_counts = raw_valid.groupby("minute").size().rename("raw_valid_sample_count")
            selected_index = raw_valid["GIC"].abs().groupby(raw_valid["minute"]).idxmax()
            selected = raw_valid.loc[selected_index, ["minute", "GIC"]].rename(columns={"minute": "time", "GIC": "raw_selected_GIC"})
            expected_minute = selected.merge(raw_counts.rename_axis("time").reset_index(), on="time", how="outer")
            expected_minute["raw_abs_max_A"] = expected_minute["raw_selected_GIC"].abs()
        resampling_check = one_minute[["time", "GIC", "missing_flag"]].merge(expected_minute, on="time", how="left")
        resampling_check["abs_difference_A"] = (resampling_check["GIC"] - resampling_check["raw_selected_GIC"]).abs()
        resampling_check["resampling_match"] = (
            resampling_check["abs_difference_A"].le(1e-6)
            | (resampling_check["missing_flag"].eq(1) & resampling_check["raw_selected_GIC"].isna())
        ).astype("int8")
        resampling_check.to_csv(
            GIC_OUTPUT_DIR / f"gic_1min_resampling_check_{date_key}.csv", index=False, date_format="%Y-%m-%d %H:%M:%S",
        )
        if not raw.empty:
            save_time_plot(raw, raw_stem, f"VKH GIC native sampling: {event_day:%Y-%m-%d} UTC")
        else:
            print(f"No GIC raw file matched {date_key}; raw plot was not written.")
        save_time_plot(one_minute, minute_stem, f"VKH GIC 1-minute maximum-|GIC| resampling: {event_day:%Y-%m-%d} UTC")
        summary_rows.append({
            "event_date": event_day,
            "raw_file_count": len(matching_files),
            "raw_rows": len(raw),
            "raw_valid_rows": int(raw["missing_flag"].eq(0).sum()) if len(raw) else 0,
            "raw_first_time": raw["time"].min() if len(raw) else pd.NaT,
            "raw_last_time": raw["time"].max() if len(raw) else pd.NaT,
            "one_minute_rows": len(one_minute),
            "one_minute_valid_rows": int(one_minute["missing_flag"].eq(0).sum()),
            "one_minute_missing_rows": int(one_minute["missing_flag"].eq(1).sum()),
            "resampling_match_minutes": int(resampling_check["resampling_match"].sum()),
            "resampling_mismatch_minutes": int(resampling_check["resampling_match"].eq(0).sum()),
        })
    pd.DataFrame(summary_rows).to_csv(
        GIC_OUTPUT_DIR / "selected_event_raw_vs_1min_summary.csv", index=False, date_format="%Y-%m-%d %H:%M:%S",
    )


# 保存分布表和 log10(count) 图。
def save_distribution(counts: np.ndarray, stem: str, title: str) -> None:
    centers = (DIST_EDGES[:-1] + DIST_EDGES[1:]) / 2
    log_counts = np.full(len(counts), np.nan, dtype=float)
    positive = counts > 0
    log_counts[positive] = np.log10(counts[positive])
    table = pd.DataFrame({
        "gic_bin_left_A": DIST_EDGES[:-1],
        "gic_bin_right_A": DIST_EDGES[1:],
        "gic_bin_center_A": centers,
        "count": counts,
        "log10_count": log_counts,
    })
    table.to_csv(GIC_OUTPUT_DIR / f"{stem}_distribution.csv", index=False)
    visible = table[table["count"] > 0]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(visible["gic_bin_center_A"], visible["log10_count"], linewidth=0.8)
    ax.set_xlabel("GIC (A)")
    ax.set_ylabel("log10(count)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(GIC_OUTPUT_DIR / f"{stem}_distribution.png", dpi=180)
    plt.close(fig)


# 保存 0-5 A 绝对值分段表和图。
def save_abs_histogram(counts: np.ndarray, stem: str, title: str) -> None:
    labels = ["[0-1A)", "[1-2A)", "[2-3A)", "[3-4A)", "[4-5A]"]
    table = pd.DataFrame({"interval": labels, "count": counts[:5]})
    table.to_csv(GIC_OUTPUT_DIR / f"{stem}_abs_0_5_histogram.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, counts[:5], color="tab:orange")
    ax.set_xlabel("|GIC| interval")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(GIC_OUTPUT_DIR / f"{stem}_abs_0_5_histogram.png", dpi=180)
    plt.close(fig)


# 处理全部 GIC 文件，按分钟选择绝对值最大样本并保留其原始符号。
def preprocess_gic(force_rebuild: bool = False) -> pd.DataFrame:
    # Reuse a completed minute-level product on reruns.  This is especially
    # important because rebuilding GIC requires scanning the full 0.5 s archive
    # even when only the ACE reader has changed.
    if not force_rebuild and GIC_1MIN_OUTPUT.exists() and GIC_1MIN_OUTPUT.stat().st_size > 0:
        try:
            header = pd.read_csv(GIC_1MIN_OUTPUT, nrows=0)
            required = {"time", "GIC", "missing_flag"}
            if required.issubset(header.columns):
                cached = pd.read_csv(
                    GIC_1MIN_OUTPUT,
                    usecols=["time", "GIC", "missing_flag"],
                    parse_dates=["time"],
                )
                expected_rows = len(pd.date_range(START_TIME, END_TIME, freq="min"))
                if len(cached) == expected_rows:
                    cached["missing_flag"] = cached["missing_flag"].astype("int8")
                    print(f"GIC 1-minute output exists; skipped GIC preprocessing: {GIC_1MIN_OUTPUT}")
                    return cached
                print(
                    f"Existing GIC output has {len(cached):,} rows; expected {expected_rows:,}. "
                    "Rebuilding GIC."
                )
        except Exception as exc:
            print(f"Existing GIC output could not be reused ({exc}); rebuilding GIC.")

    GIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = iter_gic_files()
    if not files:
        raise FileNotFoundError(f"No GIC files found in {GIC_DIR}")
    raw_dist = np.zeros(len(DIST_EDGES) - 1, dtype=np.int64)
    raw_abs = np.zeros(len(ABS_HIST_EDGES) - 1, dtype=np.int64)
    raw_totals = {"rows": 0, "missing": 0}
    minute_parts = []
    file_audit_rows = []
    plot_windows = {
        "raw_2015_year": (pd.Timestamp("2015-01-01"), pd.Timestamp("2016-01-01"), 1_500),
        "raw_2015_03": (pd.Timestamp("2015-03-01"), pd.Timestamp("2015-04-01"), 10_000),
        "raw_2015_03_17": (pd.Timestamp("2015-03-17"), pd.Timestamp("2015-03-18"), 500_000),
    }
    window_parts = {name: [] for name in plot_windows}

    for number, path in enumerate(files, start=1):
        frame = read_gic_file(path)
        valid_values = frame.loc[frame["missing_flag"].eq(0), "GIC"]
        file_audit_rows.append({
            "source": "GIC",
            "file": str(path.relative_to(GIC_DIR)),
            "filename_date": file_date_from_name(path),
            "parsed_first_time": frame["time"].min(),
            "parsed_last_time": frame["time"].max(),
            "parsed_row_count": len(frame),
            "valid_gic_row_count": int(valid_values.notna().sum()),
            "parsed_first_date_matches_filename": bool(
                pd.notna(file_date_from_name(path)) and pd.notna(frame["time"].min())
                and frame["time"].min().normalize() == file_date_from_name(path)
            ),
        })
        update_gic_statistics(frame, raw_dist, raw_abs, raw_totals)
        valid = frame.loc[frame["missing_flag"].eq(0)].copy()
        if len(valid):
            valid["time"] = valid["time"].dt.floor("min")
            idx = valid["GIC"].abs().groupby(valid["time"]).idxmax()
            minute_parts.append(valid.loc[idx, ["time", "GIC"]].set_index("time"))
        for name, (start, end, per_file_limit) in plot_windows.items():
            selected = frame[(frame["time"] >= start) & (frame["time"] < end)]
            if len(selected):
                window_parts[name].append(select_plot_points(selected, per_file_limit))
        if number == 1 or number % 100 == 0 or number == len(files):
            print(f"GIC files processed: {number}/{len(files)}")

    raw_missing_ratio = raw_totals["missing"] / raw_totals["rows"] if raw_totals["rows"] else np.nan
    save_distribution(raw_dist, "raw_gic", "Raw GIC distribution")
    save_abs_histogram(raw_abs, "raw_gic", "Raw |GIC| distribution from 0 to 5 A")
    for name, parts in window_parts.items():
        if parts:
            plot_frame = pd.concat(parts, ignore_index=True).sort_values("time")
            save_time_plot(plot_frame, name, name.replace("_", " "))

    minute = pd.concat(minute_parts).sort_index() if minute_parts else pd.DataFrame(columns=["GIC"])
    if len(minute.index):
        duplicate = minute.index.duplicated(keep=False)
        if duplicate.any():
            minute = (
                minute.reset_index()
                .sort_values(["time", "GIC"], key=lambda s: s.abs() if s.name == "GIC" else s)
                .drop_duplicates("time", keep="last")
                .set_index("time")
            )
    full_index = pd.date_range(START_TIME, END_TIME, freq="min")
    minute = minute.reindex(full_index).rename_axis("time").reset_index()
    minute["missing_flag"] = minute["GIC"].isna().astype("int8")
    minute = minute[["time", "GIC", "missing_flag"]]
    minute.to_csv(GIC_1MIN_OUTPUT, index=False, date_format="%Y-%m-%d %H:%M:%S")

    AUDIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gic_file_audit = pd.DataFrame(file_audit_rows)
    gic_file_audit.to_csv(AUDIT_OUTPUT_DIR / "GIC_raw_file_time_audit.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    expected_days = pd.DataFrame({"expected_date": pd.date_range(START_TIME.normalize(), END_TIME.normalize(), freq="D")})
    daily_file_coverage = expected_days.merge(
        gic_file_audit,
        left_on="expected_date",
        right_on="filename_date",
        how="left",
    )
    daily_file_coverage["raw_file_found"] = daily_file_coverage["file"].notna().astype("int8")
    daily_file_coverage["content_date_matches_expected"] = pd.to_datetime(
        daily_file_coverage["parsed_first_time"], errors="coerce"
    ).dt.normalize().eq(daily_file_coverage["expected_date"])
    daily_file_coverage.to_csv(
        AUDIT_OUTPUT_DIR / "GIC_expected_daily_file_coverage.csv", index=False, date_format="%Y-%m-%d %H:%M:%S",
    )
    missing_intervals(
        pd.DatetimeIndex(minute["time"]), minute["missing_flag"].to_numpy(dtype=bool),
    ).to_csv(AUDIT_OUTPUT_DIR / "GIC_1min_missing_intervals.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")

    minute_dist = np.histogram(minute.loc[minute.missing_flag.eq(0), "GIC"], DIST_EDGES)[0]
    minute_abs = np.histogram(np.abs(minute.loc[minute.missing_flag.eq(0), "GIC"]), ABS_HIST_EDGES)[0]
    save_distribution(minute_dist, "gic_1min", "1-minute GIC distribution")
    save_abs_histogram(minute_abs, "gic_1min", "1-minute |GIC| distribution from 0 to 5 A")
    for name, (start, end, _) in plot_windows.items():
        selected = minute[(minute.time >= start) & (minute.time < end)]
        if len(selected):
            save_time_plot(selected, name.replace("raw_", "gic_1min_"), name.replace("raw_", "1-minute ").replace("_", " "))

    minute_totals = {"rows": len(minute), "missing": int(minute.missing_flag.sum())}
    pd.DataFrame([
        {"stage": "raw_0_5s", "rows": raw_totals["rows"], "missing_rows": raw_totals["missing"], "missing_ratio": raw_missing_ratio},
        {"stage": "gic_1min", "rows": minute_totals["rows"], "missing_rows": minute_totals["missing"], "missing_ratio": minute_totals["missing"] / minute_totals["rows"]},
    ]).to_csv(GIC_OUTPUT_DIR / "missing_ratio_comparison.csv", index=False)
    return minute


# MFI 跳过前 250 行，SWE 跳过前 268 行，下一行作为特征名。
def read_ace_file(path: Path, instrument: str) -> pd.DataFrame:
    skiprows = MFI_SKIPROWS if instrument.upper() == "MFI" else SWE_SKIPROWS
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for _ in range(skiprows):
            next(handle, None)
        header_line = next(handle, "")
    if not header_line:
        raise ValueError(f"ACE file has no header after {skiprows} lines: {path}")
    separator = "," if header_line.count(",") >= header_line.count("\t") else "\t"
    frame = pd.read_csv(path, skiprows=skiprows, sep=separator, dtype="string", low_memory=False)
    frame.columns = [str(c).strip().strip('"') for c in frame.columns]
    time_col = next((c for c in frame.columns if re.search(r"epoch|time|date", c, re.I)), None)
    if time_col is None:
        raise ValueError(f"No time column found in ACE header: {path}")
    # CDAWeb exports may mix ISO-8601 precision (e.g. ``.0Z`` and
    # ``.000Z``) within one file; ``format='mixed'`` handles both while
    # retaining UTC semantics.
    timestamp = pd.to_datetime(frame[time_col], errors="coerce", utc=True, format="mixed")
    invalid_time = int(timestamp.isna().sum())
    if invalid_time:
        # CDAWeb ASCII exports can contain a few trailing/metadata rows that
        # are not observations.  Keep a reportable count, but do not let those
        # rows abort the entire 11-year import.
        print(f"{path.name}: skipping {invalid_time} rows with invalid timestamps")
        valid = timestamp.notna()
        frame = frame.loc[valid].copy()
        timestamp = timestamp.loc[valid]
    frame = frame.drop(columns=[time_col])
    frame.index = timestamp.dt.tz_localize(None)
    frame.index.name = "time"
    return frame.sort_index()


# 统一 V1 使用的核心太阳风字段名，其他原始字段保持不变。
def standardize_ace_columns(frame: pd.DataFrame, instrument: str) -> pd.DataFrame:
    def key(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")

    magnetic_aliases = {
        "Btot": {"magnitude", "bmag", "btot", "bt", "b_total", "b_nt"},
        "Bx_gse": {"bgsec_0", "bgse_0", "bgsec_x", "bgse_x", "bx_gse", "bx_gse_x_component_nt"},
        "By_gse": {"bgsec_1", "bgse_1", "bgsec_y", "bgse_y", "by_gse", "by_gse_y_component_nt"},
        "Bz_gse": {"bgsec_2", "bgse_2", "bgsec_z", "bgse_z", "bz_gse", "bz_gse_z_component_nt"},
    }
    plasma_aliases = {
        "Vp": {"vp", "v_p", "speed", "solar_wind_speed", "proton_speed"},
        "Np": {
            "np", "n_p", "density", "proton_density", "proton_number_density",
            "h_density_cc", "h_density_cm_3", "h_density_cm3",
        },
    }
    aliases = magnetic_aliases if instrument.upper() == "MFI" else plasma_aliases
    rename = {}
    used = set(frame.columns)

    def fuzzy_match(target: str) -> str | None:
        candidates = []
        for column in frame.columns:
            normalized = key(column)
            if column in rename or column in used and column in {"Btot", "Bx_gse", "By_gse", "Bz_gse", "Vp", "Np"}:
                continue
            if "cnt" in normalized or "quality" in normalized or "flag" in normalized:
                continue
            if target == "Btot" and (
                normalized in {"b_nt", "magnitude_nt", "bmag_nt"}
                or normalized.startswith("btot_")
                or (normalized.startswith("b_") and normalized.endswith("_nt"))
            ):
                candidates.append(column)
            elif target == "Bx_gse" and normalized.startswith("bx_gse") and normalized.endswith("_nt"):
                candidates.append(column)
            elif target == "By_gse" and normalized.startswith("by_gse") and normalized.endswith("_nt"):
                candidates.append(column)
            elif target == "Bz_gse" and normalized.startswith("bz_gse") and normalized.endswith("_nt"):
                candidates.append(column)
            elif target == "Vp" and (
                normalized == "vp"
                or normalized.startswith("vp_")
                or (("speed" in normalized or "velocity" in normalized) and ("proton" in normalized or "ion" in normalized or "sw" in normalized))
            ):
                candidates.append(column)
            elif target == "Np" and (
                normalized == "np"
                or normalized.startswith("np_")
                or normalized.startswith("h_density_")
                or ("density" in normalized and ("proton" in normalized or "ion" in normalized or "sw" in normalized))
            ):
                candidates.append(column)
        return candidates[0] if candidates else None

    for target, names in aliases.items():
        if target in frame.columns:
            continue
        match = next((column for column in frame.columns if key(column) in names), None)
        if match is None:
            match = fuzzy_match(target)
        if match is not None and target not in used:
            rename[match] = target
            used.add(target)
    return frame.rename(columns=rename)


def align_ace_to_minute(frame: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """Map CDAWeb minute-center timestamps to their UTC minute bins."""

    if not len(frame):
        return frame
    original = pd.DatetimeIndex(frame.index)
    aligned = original.floor("min")
    offsets = (original - aligned).total_seconds()
    if np.any((offsets < 0) | (offsets >= 60)):
        raise ValueError(f"{instrument} contains timestamps outside minute bins")
    frame = frame.copy()
    frame.index = pd.DatetimeIndex(aligned, name="time")
    if frame.index.has_duplicates:
        duplicates = int(frame.index.duplicated(keep=False).sum())
        raise ValueError(
            f"{instrument} contains {duplicates} rows that collide after minute alignment"
        )
    unique_offsets = np.unique(offsets)
    shown = ", ".join(f"{value:g}" for value in unique_offsets[:10])
    suffix = "..." if len(unique_offsets) > 10 else ""
    print(f"{instrument} UTC timestamp offsets within minute (s): {shown}{suffix}")
    return frame


# 检查单个仪器是否覆盖完整、连续且唯一的 2012-2022 分钟轴。
def validate_ace_time_axis(frame: pd.DataFrame, instrument: str) -> pd.DatetimeIndex:
    if frame.index.has_duplicates:
        duplicates = int(frame.index.duplicated(keep=False).sum())
        raise ValueError(f"{instrument} contains {duplicates} duplicate time rows")
    expected = pd.date_range(START_TIME, END_TIME, freq="min", name="time")
    actual = pd.DatetimeIndex(frame.index, name="time")
    if not actual.equals(expected):
        missing = expected.difference(actual)
        extra = actual.difference(expected)
        first_missing = str(missing[0]) if len(missing) else "none"
        first_extra = str(extra[0]) if len(extra) else "none"
        raise ValueError(
            f"{instrument} minute axis is incomplete: rows={len(actual):,}, "
            f"expected={len(expected):,}, missing={len(missing):,}, extra={len(extra):,}, "
            f"first_missing={first_missing}, first_extra={first_extra}"
        )
    return actual


# 异常/缺失特征值置为 NaN，每个特征追加独立的 0/1 缺失标记。
def clean_wind_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned = pd.DataFrame(index=frame.index)
    flags = pd.DataFrame(index=frame.index)
    report_rows = []
    for column in frame.columns:
        source = frame[column]
        numeric = pd.to_numeric(source, errors="coerce")
        source_missing = source.isna() | source.str.strip().eq("")
        nine_sentinel = repeated_nine_mask(source)
        cdf_fill = numeric.abs().ge(1e30)
        nonnumeric = numeric.isna() & ~source_missing
        invalid = source_missing | nine_sentinel | cdf_fill | nonnumeric
        cleaned[column] = numeric.mask(invalid)
        flag_name = f"{column}_missing_flag"
        flags[flag_name] = cleaned[column].isna().astype("int8")
        missing_count = int(flags[flag_name].sum())
        report_rows.append({
            "feature": column,
            "rows": len(frame),
            "source_missing_count": int(source_missing.sum()),
            "repeated_nine_count": int(nine_sentinel.sum()),
            "cdf_fill_count": int(cdf_fill.sum()),
            "nonnumeric_count": int(nonnumeric.sum()),
            "missing_count": missing_count,
            "missing_ratio": missing_count / len(frame) if len(frame) else np.nan,
            "missing_percent": missing_count / len(frame) * 100 if len(frame) else np.nan,
        })
    return pd.concat([cleaned, flags], axis=1), pd.DataFrame(report_rows)


# 严格对齐 MFI/SWE 分钟行，按 MFI 特征、SWE 特征、缺失标记顺序合并。
def preprocess_ace() -> pd.DataFrame:
    """Deprecated ACE path retained for historical compatibility; main uses Wind."""
    ACE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mfi_files = sorted(p for p in L1_DIR.rglob("AC_H0_MFI*") if p.is_file())
    swe_files = sorted(p for p in L1_DIR.rglob("AC_H0_SWE*") if p.is_file())
    if not mfi_files or not swe_files:
        raise FileNotFoundError("AC_H0_MFI* and AC_H0_SWE* files are required in data/L1")
    mfi = pd.concat([read_ace_file(p, "MFI") for p in mfi_files]).sort_index()
    swe = pd.concat([read_ace_file(p, "SWE") for p in swe_files]).sort_index()
    mfi = standardize_ace_columns(mfi, "MFI")
    swe = standardize_ace_columns(swe, "SWE")
    mfi = align_ace_to_minute(mfi, "MFI")
    swe = align_ace_to_minute(swe, "SWE")
    mfi_axis = validate_ace_time_axis(mfi, "MFI")
    swe_axis = validate_ace_time_axis(swe, "SWE")
    if not mfi_axis.equals(swe_axis):
        raise ValueError("MFI and SWE minute rows do not align exactly")

    overlap = set(mfi.columns) & set(swe.columns)
    mfi = mfi.rename(columns={c: f"{c}_MFI" for c in overlap})
    swe = swe.rename(columns={c: f"{c}_SWE" for c in overlap})
    feature_values = pd.concat([mfi, swe], axis=1)
    cleaned, missing_report = clean_ace_features(feature_values)
    value_columns = list(feature_values.columns)
    flag_columns = [f"{column}_missing_flag" for column in value_columns]
    cleaned = cleaned[value_columns + flag_columns]
    cleaned.index.name = "time"
    cleaned.reset_index().to_csv(
        ACE_1MIN_OUTPUT, index=False, date_format="%Y-%m-%d %H:%M:%S"
    )
    missing_report.to_csv(ACE_OUTPUT_DIR / "ACE_feature_missing_ratio.csv", index=False)
    return cleaned


# 依次处理 GIC 与 ACE，并打印结果规模。
def _find_wind_header(path: Path) -> tuple[int, list[str]]:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line_number, line in enumerate(handle):
            columns = line.split()
            if WIND_TIME_COLUMNS.issubset({column.lower() for column in columns}):
                return line_number, columns
    raise ValueError(f"Could not find Wind time header in {path}")


def read_wind_ascii(path: Path, requested: dict[str, str] | None) -> pd.DataFrame:
    header_row, columns = _find_wind_header(path)
    raw = pd.read_csv(path, sep=r"\s+", header=None, names=columns, skiprows=header_row + 2,
                      dtype="string", comment="#", engine="c", on_bad_lines="skip")
    lookup = {str(column).lower(): column for column in raw.columns}
    if requested is None:
        requested = {str(column): str(column) for column in raw.columns if str(column).lower() not in WIND_TIME_COLUMNS}
    missing = [source for source in requested if source.lower() not in lookup]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(missing)}")
    year = pd.to_numeric(raw[lookup["year"]], errors="coerce")
    doy = pd.to_numeric(raw[lookup["doy"]], errors="coerce")
    millisecs = pd.to_numeric(raw[lookup["millisecs"]], errors="coerce")
    calendar = pd.to_datetime({"year": year, "month": pd.Series(1, index=raw.index), "day": pd.Series(1, index=raw.index)}, errors="coerce")
    timestamp = calendar + pd.to_timedelta(doy - 1, unit="D") + pd.to_timedelta(millisecs, unit="ms")
    valid = timestamp.notna() & year.between(1900, 2200) & doy.between(1, 366) & millisecs.between(0, 86_399_999)
    frame = pd.DataFrame(index=pd.DatetimeIndex(timestamp.loc[valid], name="time"))
    for source, target in requested.items():
        frame[target] = raw.loc[valid, lookup[source.lower()]].array
    return frame.sort_index()


def read_wind_product(folder: Path, requested: dict[str, str] | None, source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in WIND_FILE_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"No Wind ASCII files found in {folder}")
    parts = []
    audit_rows = []
    for path in files:
        part = read_wind_ascii(path, requested)
        parts.append(part)
        audit_rows.append({
            "source": source,
            "file": str(path.relative_to(folder)),
            "parsed_first_time": part.index.min(),
            "parsed_last_time": part.index.max(),
            "parsed_row_count": len(part),
            "duplicate_time_count": int(part.index.duplicated(keep=False).sum()),
        })
    frame = pd.concat(parts).sort_index()
    duplicate_count = int(frame.index.duplicated(keep=False).sum())
    if duplicate_count:
        print(f"{source}: dropped {duplicate_count:,} duplicate raw timestamps, keeping the final file value.")
    return frame.loc[~frame.index.duplicated(keep="last")], pd.DataFrame(audit_rows)


def align_wind_mfi(frame: pd.DataFrame) -> pd.DataFrame:
    aligned = frame.copy()
    aligned.index = pd.DatetimeIndex(aligned.index.floor("min"), name="time")
    if aligned.index.has_duplicates:
        raise ValueError("Wind MFI contains duplicate minute timestamps")
    return aligned


def nearest_wind_swe(swe: pd.DataFrame, minute_index: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict[str, float | int]]:
    source = swe.sort_index().loc[~swe.index.duplicated(keep="last")].copy()
    if source.empty:
        return pd.DataFrame(index=minute_index, columns=swe.columns, dtype="string"), {"source_rows": 0, "continuous_segments": 0, "gap_break_seconds": np.nan, "mapped_minutes": 0, "max_abs_time_offset_seconds": np.nan, "median_abs_time_offset_seconds": np.nan}
    intervals = source.index.to_series().diff().dt.total_seconds().dropna()
    normal = intervals[intervals.between(30.0, 180.0)]
    native = float(normal.median()) if len(normal) else 92.0
    upper = float(normal.quantile(0.99)) if len(normal) else native
    gap_break = max(1.5 * native, 1.05 * upper)
    source["_segment"] = intervals.gt(gap_break).cumsum().reindex(source.index, fill_value=0).to_numpy(dtype=np.int64)
    mapped_parts, offsets = [], []
    for _, block in source.groupby("_segment", sort=False):
        block = block.drop(columns="_segment")
        first = max(minute_index[0], block.index.min().floor("min")); last = min(minute_index[-1], block.index.max().floor("min"))
        if first > last: continue
        targets = pd.DataFrame({"time": pd.date_range(first, last, freq="min")})
        observations = block.reset_index(names="source_time")
        nearest = pd.merge_asof(targets, observations, left_on="time", right_on="source_time", direction="nearest")
        offsets.extend((nearest["time"] - nearest["source_time"]).abs().dt.total_seconds().to_numpy())
        mapped_parts.append(nearest.set_index("time")[list(swe.columns)])
    mapped = pd.concat(mapped_parts).sort_index() if mapped_parts else pd.DataFrame(columns=swe.columns)
    mapped.index = pd.DatetimeIndex(mapped.index, name="time")
    mapped = mapped.reindex(minute_index)
    values = np.asarray(offsets, dtype=float)
    return mapped, {"source_rows": int(len(source)), "continuous_segments": int(source["_segment"].nunique()), "gap_break_seconds": float(gap_break), "mapped_minutes": int(mapped.notna().any(axis=1).sum()), "max_abs_time_offset_seconds": float(np.nanmax(values)) if len(values) else np.nan, "median_abs_time_offset_seconds": float(np.nanmedian(values)) if len(values) else np.nan}


def preprocess_wind(start: pd.Timestamp = START_TIME, end: pd.Timestamp = END_TIME) -> pd.DataFrame:
    minute_index = pd.date_range(start, end, freq="min", name="time")
    mfi, mfi_file_audit = read_wind_product(
        WIND_MFI_DIR, {"B": "Btot", "Bx": "Bx_gsm", "By": "By_gsm", "Bz": "Bz_gsm"}, "Wind_MFI",
    )
    swe, swe_file_audit = read_wind_product(WIND_SWE_DIR, None, "Wind_SWE")
    mfi = align_wind_mfi(mfi).reindex(minute_index)
    swe, mapping = nearest_wind_swe(swe, minute_index)
    merged = pd.concat([mfi, swe], axis=1)
    cleaned, report = clean_wind_features(merged)
    value_columns = list(merged.columns)
    cleaned = cleaned[value_columns + [f"{column}_missing_flag" for column in value_columns]]
    cleaned.index.name = "time"
    WIND_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.reset_index().to_csv(WIND_1MIN_OUTPUT, index=False, date_format="%Y-%m-%d %H:%M:%S")
    report.to_csv(WIND_OUTPUT_DIR / "Wind_feature_missing_ratio.csv", index=False)
    pd.DataFrame([mapping]).to_csv(WIND_OUTPUT_DIR / "Wind_SWE_nearest_mapping_report.csv", index=False)
    AUDIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat([mfi_file_audit, swe_file_audit], ignore_index=True).to_csv(
        AUDIT_OUTPUT_DIR / "Wind_raw_file_time_audit.csv", index=False, date_format="%Y-%m-%d %H:%M:%S",
    )
    interval_parts = []
    for column in value_columns:
        intervals = missing_intervals(minute_index, cleaned[f"{column}_missing_flag"].to_numpy(dtype=bool))
        intervals.insert(0, "feature", column)
        interval_parts.append(intervals)
    pd.concat(interval_parts, ignore_index=True).to_csv(
        AUDIT_OUTPUT_DIR / "Wind_feature_missing_intervals.csv", index=False, date_format="%Y-%m-%d %H:%M:%S",
    )
    return cleaned


def write_combined_missing_audit(gic: pd.DataFrame, wind: pd.DataFrame) -> None:
    """Audit all final GIC/Wind features on the same explicit minute axis."""
    AUDIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gic_indexed = gic.set_index("time")
    feature_flags = {"GIC": gic_indexed["missing_flag"].astype("int8")}
    for column in wind.columns:
        if column.endswith("_missing_flag"):
            feature = column[: -len("_missing_flag")]
            feature_flags[feature] = wind[column].astype("int8")
    flags = pd.DataFrame(feature_flags).reindex(pd.date_range(START_TIME, END_TIME, freq="min", name="time"))
    summary_rows = []
    interval_parts = []
    for feature in flags.columns:
        missing = flags[feature].fillna(1).astype(bool)
        summary_rows.append({
            "feature": feature,
            "rows": len(flags),
            "missing_count": int(missing.sum()),
            "missing_ratio": float(missing.mean()),
            "first_missing_time": flags.index[missing.to_numpy()][0] if missing.any() else pd.NaT,
        })
        intervals = missing_intervals(flags.index, missing.to_numpy())
        intervals.insert(0, "feature", feature)
        interval_parts.append(intervals)
    pd.DataFrame(summary_rows).to_csv(AUDIT_OUTPUT_DIR / "combined_feature_missing_summary.csv", index=False)
    pd.concat(interval_parts, ignore_index=True).to_csv(
        AUDIT_OUTPUT_DIR / "combined_feature_missing_intervals.csv", index=False, date_format="%Y-%m-%d %H:%M:%S",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-rebuild", action="store_true",
        help="Rebuild the GIC one-minute product from raw files instead of reusing an existing CSV.",
    )
    parser.add_argument(
        "--audit-only", action="store_true",
        help="Only write the final combined missingness audit from existing 1-minute GIC and Wind outputs.",
    )
    parser.add_argument(
        "--plot-gic-events-only", action="store_true",
        help="Write native-sampling and 1-minute GIC plots/data for 2013-06-07, 2017-05-28, and 2021-03-01.",
    )
    args = parser.parse_args()
    if args.audit_only and args.plot_gic_events_only:
        raise ValueError("Use --audit-only or --plot-gic-events-only, not both.")
    if args.plot_gic_events_only:
        write_selected_gic_event_plots()
        print(f"Selected GIC event plots written: {GIC_OUTPUT_DIR}")
        raise SystemExit(0)
    if args.audit_only:
        if not GIC_1MIN_OUTPUT.exists() or not WIND_1MIN_OUTPUT.exists():
            raise FileNotFoundError("--audit-only requires existing GIC and Wind 1-minute outputs.")
        gic = pd.read_csv(GIC_1MIN_OUTPUT, usecols=["time", "missing_flag"], parse_dates=["time"])
        wind_header = pd.read_csv(WIND_1MIN_OUTPUT, nrows=0)
        wind_columns = ["time", *[column for column in wind_header.columns if column.endswith("_missing_flag")]]
        wind = pd.read_csv(WIND_1MIN_OUTPUT, usecols=wind_columns, parse_dates=["time"]).set_index("time")
        if "missing_flag" not in gic.columns or len(wind_columns) == 1:
            raise ValueError("Existing 1-minute outputs do not contain the required missing-flag columns.")
        write_combined_missing_audit(gic, wind)
        print(f"Combined missingness audit written: {AUDIT_OUTPUT_DIR}")
        raise SystemExit(0)
    gic = preprocess_gic(force_rebuild=args.force_rebuild)
    wind = preprocess_wind()
    write_combined_missing_audit(gic, wind)
    print(f"GIC 1-minute output: {GIC_1MIN_OUTPUT} | rows={len(gic):,}")
    print(f"Wind 1-minute output: {WIND_1MIN_OUTPUT} | rows={len(wind):,}")
