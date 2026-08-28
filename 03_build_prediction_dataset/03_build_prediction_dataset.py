"""Build aligned Wind-GIC lead-time prediction datasets for VKH.

The script preserves the native 1-minute UTC grid and never interpolates a
missing value.  At prediction time t, a solar-wind input sequence ends at
t-L and the binary target is based on max(|GIC|) over (t, t+30 min].

By default, outputs are written below ``data/prediction_dataset``.  The canonical
timeline contains only 10-day CME/CIR event blocks; each (lag, window)
combination is encoded by a validity mask rather than duplicating the same
sequences many times.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
GIC_PATH = DATA_DIR / "processed" / "GIC_1min_2012_2022.csv"
WIND_PATH = DATA_DIR / "processed" / "Wind_L1_1min_2012_2022.csv"
OUTPUT_DIR = DATA_DIR / "prediction_dataset"
PLOT_DIR = OUTPUT_DIR / "yearly_feature_plots"
SUPPORTING_INFO_PATH = ROOT / "data" / "raw" / "supporting_information.docx"

START_TIME = pd.Timestamp("2012-01-01 00:00:00")
END_TIME = pd.Timestamp("2022-12-31 23:59:00")
TARGET_WINDOW_MINUTES = 30
THRESHOLDS_A = (3, 5, 10, 20)
LAGS_MINUTES = (30, 45, 60, 90)
WINDOWS_MINUTES = (30, 60, 120)
LONG_GAP_MINUTES = 24 * 60
EVENT_WINDOW_DAYS = 5
EVENT_EXCLUSIONS = {18, 19}
DEFAULT_SPLIT_RATIOS = (0.70, 0.20, 0.10)

WIND_FEATURES = (
    "Btot", "Bx_gsm", "By_gsm", "Bz_gsm", "V", "Np", "Psw", "Ma",
    "Mms", "Epsilon", "VBs",
)
GIC_FEATURE = "GIC"

# The CNN-BiLSTM input consists only of upstream solar-wind quantities.  GIC
# is intentionally excluded from input features to preserve L1 lead time.
SEQUENCE_FEATURES = (
    *WIND_FEATURES,
    "Bt_gsm", "Bz_south", "clock_angle_rad", "Newell_coupling",
    "Borovsky_Rquick_mV_m",
)
WINDOW_STAT_FEATURES = (
    # Keep the window moments aligned with the exact 13-field F1 input.  In
    # particular, Bx_gsm is included and Mms is deliberately excluded.
    "Btot", "Bx_gsm", "By_gsm", "Bz_gsm", "V", "Np", "Psw", "Ma",
    "Epsilon", "VBs", "Bz_south", "Newell_coupling", "Borovsky_Rquick_mV_m",
)
# These are calculated at more than one scale inside a sequence.  They retain
# the duration and integrated strength of geoeffective solar-wind forcing,
# which a minute-level encoder can otherwise dilute when it sees a quiet and a
# disturbed interval in the same input window.
COUPLING_ACCUMULATION_FEATURES = (
    "Bz_south", "Epsilon", "Newell_coupling", "Borovsky_Rquick_mV_m",
)
ACCUMULATION_WINDOWS_MINUTES = (30, 60, 120)


def expected_time_index() -> pd.DatetimeIndex:
    return pd.date_range(START_TIME, END_TIME, freq="min", name="time")


DOCX_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def _docx_cell_text(cell: ET.Element) -> str:
    return " ".join(text.text.strip() for text in cell.findall(".//w:t", DOCX_NS) if text.text).strip()


def _parse_event_time(date_text: str, time_text: str) -> pd.Timestamp:
    normalized_time = re.sub(r"\s+", "", time_text).replace(".", ":")
    return pd.to_datetime(f"{date_text.strip()} {normalized_time}", format="%d.%m.%Y %H:%M")


def load_vkh_event_table() -> pd.DataFrame:
    """Read VKH Table 1 directly from the supplied supporting-information DOCX."""
    if not SUPPORTING_INFO_PATH.exists():
        raise FileNotFoundError(f"VKH event table is required: {SUPPORTING_INFO_PATH}")
    with ZipFile(SUPPORTING_INFO_PATH) as archive:
        document = ET.fromstring(archive.read("word/document.xml"))
    tables = document.findall(".//w:tbl", DOCX_NS)
    if not tables:
        raise ValueError("Supporting information does not contain a VKH event table.")
    rows = []
    for row in tables[0].findall("./w:tr", DOCX_NS):
        cells = [_docx_cell_text(cell) for cell in row.findall("./w:tc", DOCX_NS)]
        if len(cells) >= 14 and re.fullmatch(r"\d+\*?", cells[0]):
            rows.append(cells[:14])
    if len(rows) != 92:
        raise ValueError(f"Expected 92 VKH events in Table 1, found {len(rows)}.")
    events = pd.DataFrame(rows, columns=[
        "event_number", "date", "time_ut", "gic_peak_A", "duration_min", "dbdt_nT_min",
        "sym_h_nT", "sym_h_max_nT", "bz_nT", "storm_type", "ae_nT", "ie_nT",
        "ie_max_nT", "gd_type",
    ])
    events["event_id"] = events["event_number"].str.extract(r"(\d+)").astype("int16")
    events["peak_time"] = [
        _parse_event_time(date, time_ut) for date, time_ut in zip(events["date"], events["time_ut"])
    ]
    events["gic_peak_A"] = pd.to_numeric(events["gic_peak_A"].str.replace(",", ".", regex=False), errors="coerce")
    events["gic_peak_abs_A"] = events["gic_peak_A"].abs()
    storm = events["storm_type"].str.upper()
    is_cir = storm.str.contains("CIR", regex=False)
    is_cme = storm.str.contains("CME", regex=False) & ~storm.str.startswith("SC")
    events["driver"] = pd.Series(pd.NA, index=events.index, dtype="string")
    events.loc[is_cme, "driver"] = "CME"
    events.loc[is_cir, "driver"] = "CIR"
    events["selection_status"] = "excluded_non_CME_CIR"
    events.loc[events["event_id"].isin(EVENT_EXCLUSIONS), "selection_status"] = "excluded_engineering_event"
    events.loc[is_cme | is_cir, "selection_status"] = "selected"
    events.loc[events["event_id"].isin(EVENT_EXCLUSIONS), "selection_status"] = "excluded_engineering_event"
    return events.sort_values("peak_time").reset_index(drop=True)


def maximum_missing_run_minutes(values: pd.Series) -> int:
    """Return the longest contiguous missing period on the native minute grid."""
    missing = values.isna().to_numpy(dtype=bool)
    transitions = np.flatnonzero(np.diff(np.r_[False, missing, False]))
    return int(np.max(transitions[1::2] - transitions[::2])) if len(transitions) else 0


def assess_event_data_availability(events: pd.DataFrame, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exclude selected CME/CIR events whose +/-5 day blocks contain long data gaps."""
    assessed = events.copy()
    assessed["event_window_start"] = assessed["peak_time"] - pd.Timedelta(days=EVENT_WINDOW_DAYS)
    assessed["event_window_end"] = assessed["peak_time"] + pd.Timedelta(days=EVENT_WINDOW_DAYS)
    assessed["max_missing_GIC_min"] = 0
    assessed["max_missing_solar_wind_min"] = 0
    assessed["data_selection_status"] = "not_assessed_non_CME_CIR_or_engineering"
    long_gap_rows = []
    candidate_mask = assessed["selection_status"].eq("selected") & ~assessed["event_id"].isin(EVENT_EXCLUSIONS)
    for index, event in assessed.loc[candidate_mask].iterrows():
        block = frame.loc[
            frame["time"].ge(event["event_window_start"]) & frame["time"].lt(event["event_window_end"])
        ]
        if block.empty:
            raise ValueError(f"No source data covers selected event {event['event_id']}.")
        gic_gap = maximum_missing_run_minutes(block[GIC_FEATURE])
        solar_gaps = {feature: maximum_missing_run_minutes(block[feature]) for feature in SEQUENCE_FEATURES}
        solar_gap = max(solar_gaps.values(), default=0)
        assessed.loc[index, "max_missing_GIC_min"] = gic_gap
        assessed.loc[index, "max_missing_solar_wind_min"] = solar_gap
        failed_features = {GIC_FEATURE: gic_gap, **solar_gaps}
        long_features = {feature: length for feature, length in failed_features.items() if length >= LONG_GAP_MINUTES}
        if long_features:
            assessed.loc[index, "data_selection_status"] = "excluded_long_GIC_or_solar_wind_gap"
            for feature, length in long_features.items():
                long_gap_rows.append({
                    "event_id": int(event["event_id"]), "driver": event["driver"], "peak_time": event["peak_time"],
                    "feature": feature, "longest_missing_minutes": int(length),
                    "longest_missing_days": length / 1440.0,
                })
        else:
            assessed.loc[index, "data_selection_status"] = "selected_data_complete_for_long_gap_rule"
    return assessed, pd.DataFrame(
        long_gap_rows,
        columns=["event_id", "driver", "peak_time", "feature", "longest_missing_minutes", "longest_missing_days"],
    )


def build_event_catalog(
    events: pd.DataFrame,
    split_ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select CME/CIR events, merge overlapping 10-day windows, and split blocks."""
    selected = events.loc[
        events["selection_status"].eq("selected")
        & events["data_selection_status"].eq("selected_data_complete_for_long_gap_rule")
        & ~events["event_id"].isin(EVENT_EXCLUSIONS)
    ].copy()
    selected["event_window_start"] = selected["peak_time"] - pd.Timedelta(days=EVENT_WINDOW_DAYS)
    selected["event_window_end"] = selected["peak_time"] + pd.Timedelta(days=EVENT_WINDOW_DAYS)

    group_ids: list[int] = []
    current_end: pd.Timestamp | None = None
    group_id = 0
    for row in selected.itertuples():
        if current_end is None or row.event_window_start >= current_end:
            group_id += 1
            current_end = row.event_window_end
        else:
            current_end = max(current_end, row.event_window_end)
        group_ids.append(group_id)
    selected["event_group"] = np.asarray(group_ids, dtype=np.int16)
    group_summary = selected.groupby("event_group", sort=True).agg(
        segment_start=("event_window_start", "min"),
        segment_end=("event_window_end", "max"),
        event_count=("event_id", "size"),
        event_ids=("event_id", lambda values: ",".join(map(str, values))),
    ).reset_index()

    # Use whole chronological blocks near the requested target proportions;
    # splitting an overlapping block would leak event context.
    cumulative = group_summary["event_count"].cumsum().to_numpy()
    train_ratio, validation_ratio, _ = split_ratios
    train_target = len(selected) * train_ratio
    train_boundary = min(range(1, len(group_summary) - 1), key=lambda i: abs(cumulative[i - 1] - train_target))
    validation_target = len(selected) * validation_ratio
    validation_boundary = min(
        range(train_boundary + 1, len(group_summary)),
        key=lambda i: abs((cumulative[i - 1] - cumulative[train_boundary - 1]) - validation_target),
    )
    split_names = np.full(len(group_summary), "test", dtype=object)
    split_names[:train_boundary] = "train"
    split_names[train_boundary:validation_boundary] = "validation"
    group_summary["split"] = split_names
    selected = selected.merge(group_summary[["event_group", "segment_start", "segment_end", "split"]], on="event_group", how="left")
    return selected, group_summary


def _read_source(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required 02 output does not exist: {path}. Run 02_data_preprocessing.py first."
        )
    header = pd.read_csv(path, nrows=0)
    missing = sorted(set(required_columns) - set(header.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {', '.join(missing)}")
    dtypes = {column: "float32" for column in required_columns if column != "time"}
    frame = pd.read_csv(path, usecols=required_columns, dtype=dtypes, parse_dates=["time"])
    frame = frame.rename(columns={"time": "time"})
    return frame


def inspect_time_axis(frame: pd.DataFrame, source: str) -> dict[str, object]:
    time = pd.DatetimeIndex(frame["time"])
    expected = expected_time_index()
    duplicates = int(time.duplicated(keep=False).sum())
    monotonic = bool(time.is_monotonic_increasing)
    unique_time = time.drop_duplicates()
    missing = expected.difference(unique_time)
    extra = unique_time.difference(expected)
    return {
        "source": source,
        "rows": int(len(frame)),
        "expected_rows": int(len(expected)),
        "duplicate_rows": duplicates,
        "monotonic_increasing": monotonic,
        "missing_timestamp_count": int(len(missing)),
        "extra_timestamp_count": int(len(extra)),
        "first_missing_timestamp": str(missing[0]) if len(missing) else None,
        "first_extra_timestamp": str(extra[0]) if len(extra) else None,
        "exact_expected_axis": bool(
            len(frame) == len(expected) and duplicates == 0 and time.equals(expected)
        ),
    }


def load_and_align() -> tuple[pd.DataFrame, dict[str, object]]:
    wind = _read_source(WIND_PATH, ["time", *WIND_FEATURES])
    gic = _read_source(GIC_PATH, ["time", GIC_FEATURE])
    wind_report = inspect_time_axis(wind, WIND_PATH.name)
    gic_report = inspect_time_axis(gic, GIC_PATH.name)
    report = {
        "wind": wind_report,
        "gic": gic_report,
        "time_columns_exactly_aligned": bool(pd.DatetimeIndex(wind["time"]).equals(pd.DatetimeIndex(gic["time"]))),
    }
    # Preserve diagnostics even when the script must stop on an invalid grid.
    (OUTPUT_DIR / "time_alignment_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if not wind_report["exact_expected_axis"] or not gic_report["exact_expected_axis"]:
        raise ValueError(
            "Input time axes are not the complete, unique 2012-01-01 through "
            "2022-12-31 1-minute grid. See time_alignment_report.json."
        )
    if not pd.DatetimeIndex(wind["time"]).equals(pd.DatetimeIndex(gic["time"])):
        raise ValueError("Wind and GIC time columns do not align exactly.")
    merged = wind.merge(gic, on="time", how="inner", validate="one_to_one")
    merged.index = pd.DatetimeIndex(merged["time"], name="time")
    report["time_columns_exactly_aligned"] = bool(len(merged) == len(expected_time_index()))
    report["start_time"] = str(merged["time"].iloc[0])
    report["end_time"] = str(merged["time"].iloc[-1])
    return merged, report


def calculate_coupling_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add physical IMF geometry plus Newell and Borovsky coupling functions.

    Newell et al. (2007) is stored in its conventional empirical input units:
      V[km s^-1]^(4/3) * Bt[nT]^(2/3) * sin(theta/2)^(8/3).

    The Borovsky rapid dayside-reconnection driver is:
      Rquick = V * Bt * sin(theta/2)^2.
    With Wind inputs V in km/s and Bt in nT, multiplication by 1e-3 gives
    mV/m. Epsilon remains the Wind-provided Akasofu value and is not
    recomputed.
    """

    result = frame.copy()
    by = result["By_gsm"].astype("float64")
    bz = result["Bz_gsm"].astype("float64")
    speed_km_s = result["V"].astype("float64")

    bt = np.hypot(by, bz)
    valid_imf = bt.notna() & speed_km_s.gt(0)
    # atan2(By, Bz) produces the GSM clock angle measured from northward Bz.
    clock_angle = np.mod(np.arctan2(by, bz), 2.0 * np.pi)
    sin_half = np.sin(clock_angle / 2.0)
    sin_half = pd.Series(sin_half, index=result.index).where(valid_imf, np.nan)

    result["Bt_gsm"] = bt.astype("float32")
    result["Bz_south"] = (-bz).clip(lower=0.0).where(bz.notna(), np.nan).astype("float32")
    result["clock_angle_rad"] = pd.Series(clock_angle, index=result.index).where(valid_imf).astype("float32")
    result["Newell_coupling"] = (
        speed_km_s.pow(4.0 / 3.0) * bt.pow(2.0 / 3.0) * sin_half.pow(8.0 / 3.0)
    ).where(valid_imf).astype("float32")

    result["Borovsky_Rquick_mV_m"] = (
        speed_km_s * bt * sin_half.pow(2.0) * 1e-3
    ).where(valid_imf).astype("float32")
    return result


def add_prediction_targets(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    gic_abs = result[GIC_FEATURE].astype("float64").abs()
    gic_valid = gic_abs.notna()
    # At row t this is max(|GIC(t+1)|, ..., |GIC(t+30)|), never including t.
    peak = gic_abs.rolling(TARGET_WINDOW_MINUTES, min_periods=TARGET_WINDOW_MINUTES).max().shift(-TARGET_WINDOW_MINUTES)
    result["future_gic_peak_abs_30min_A"] = peak.astype("float32")
    result["target_window_complete_30min"] = peak.notna().astype("int8")
    # An onset is the first valid minute crossing from below a threshold to
    # above it.  The auxiliary target asks whether that crossing occurs in
    # (t, t+30], aligning training with the catalogue's early-warning metric
    # while retaining the original future-peak target unchanged.
    complete_future_window = (
        gic_valid.astype("int8")
        .rolling(TARGET_WINDOW_MINUTES, min_periods=TARGET_WINDOW_MINUTES)
        .min()
        .shift(-TARGET_WINDOW_MINUTES)
        .eq(1)
    )
    for threshold in THRESHOLDS_A:
        label = pd.Series(pd.NA, index=result.index, dtype="Int8")
        valid = peak.notna()
        label.loc[valid] = (peak.loc[valid] >= threshold).astype("int8")
        result[f"target_exceeds_{threshold}A_30min"] = label
        above = gic_abs.ge(threshold)
        previous_above = above.shift(1, fill_value=False)
        onset = (above & ~previous_above & gic_valid & gic_valid.shift(1, fill_value=False)).astype("float32")
        future_onset = onset.rolling(TARGET_WINDOW_MINUTES, min_periods=TARGET_WINDOW_MINUTES).max().shift(-TARGET_WINDOW_MINUTES)
        onset_label = pd.Series(pd.NA, index=result.index, dtype="Int8")
        # The crossing at t+1 is defined relative to the observed state at t.
        # Without GIC(t), a transition immediately after t cannot be labelled
        # unambiguously even if all future values are present.
        onset_valid = gic_valid & complete_future_window & future_onset.notna()
        onset_label.loc[onset_valid] = future_onset.loc[onset_valid].astype("int8")
        result[f"target_onset_{threshold}A_30min"] = onset_label
    return result


def find_long_missing_intervals(frame: pd.DataFrame, feature: str) -> list[dict[str, object]]:
    missing = frame[feature].isna().to_numpy(dtype=bool)
    if not missing.any():
        return []
    times = pd.DatetimeIndex(frame["time"])
    previous_contiguous = np.r_[False, np.diff(times.asi8) == pd.Timedelta(minutes=1).value]
    next_contiguous = np.r_[previous_contiguous[1:], False]
    starts = np.flatnonzero(missing & ~(np.r_[False, missing[:-1]] & previous_contiguous))
    ends = np.flatnonzero(missing & ~(np.r_[missing[1:], False] & next_contiguous))
    if len(starts) != len(ends):
        raise RuntimeError(f"Could not pair missing intervals for {feature}.")
    rows: list[dict[str, object]] = []
    for start, end in zip(starts, ends):
        minutes = int((times[end] - times[start]) / pd.Timedelta(minutes=1)) + 1
        if minutes >= LONG_GAP_MINUTES:
            rows.append({
                "feature": feature,
                "start_time": frame["time"].iloc[start],
                "end_time": frame["time"].iloc[end],
                "missing_minutes": minutes,
                "missing_days": minutes / 1440.0,
            })
    return rows


def add_lag_window_masks(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = frame.copy()
    all_input_valid = result.loc[:, SEQUENCE_FEATURES].notna().all(axis=1).astype("int8")
    configurations = []
    for lag in LAGS_MINUTES:
        for window in WINDOWS_MINUTES:
            name = f"sample_valid_L{lag}_W{window}"
            window_complete_at_end = all_input_valid.rolling(window, min_periods=window).min()
            # At t, rolling window must end at t-L: [t-L-W+1, t-L].
            valid = window_complete_at_end.shift(lag).fillna(0).astype("int8")
            valid = (valid.astype(bool) & result["target_window_complete_30min"].astype(bool)).astype("int8")
            result[name] = valid
            configurations.append({
                "lag_minutes": lag,
                "window_minutes": window,
                "input_start": "t-L-W+1 min",
                "input_end": "t-L min",
                "target_start": "t+1 min",
                "target_end": "t+30 min",
                "validity_column": name,
                "valid_samples": int(valid.sum()),
            })
    return result, pd.DataFrame(configurations)


def build_event_timeline(frame: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only selected event blocks and invalidate samples at block edges."""
    group_columns = events[["event_group", "segment_start", "segment_end", "split"]].drop_duplicates()
    parts = []
    for group in group_columns.sort_values("event_group").itertuples(index=False):
        part = frame.loc[
            frame["time"].ge(group.segment_start) & frame["time"].lt(group.segment_end)
        ].copy()
        if part.empty:
            raise ValueError(f"No source rows found for event group {group.event_group}.")
        part["event_group"] = int(group.event_group)
        part["split"] = group.split
        part["segment_start"] = group.segment_start
        part["segment_end"] = group.segment_end
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    result = result.sort_values("time").reset_index(drop=True)
    configurations = []
    for lag in LAGS_MINUTES:
        for window in WINDOWS_MINUTES:
            name = f"sample_valid_L{lag}_W{window}"
            input_start = result["time"] - pd.Timedelta(minutes=lag + window - 1)
            target_end = result["time"] + pd.Timedelta(minutes=TARGET_WINDOW_MINUTES)
            inside_block = input_start.ge(result["segment_start"]) & target_end.lt(result["segment_end"])
            result[name] = (result[name].astype(bool) & inside_block).astype("int8")
            configurations.append({
                "lag_minutes": lag,
                "window_minutes": window,
                "input_start": "t-L-W+1 min",
                "input_end": "t-L min",
                "target_start": "t+1 min",
                "target_end": "t+30 min",
                "validity_column": name,
                "valid_samples": int(result[name].sum()),
            })
    return result, pd.DataFrame(configurations)


def build_window_statistics(frame: pd.DataFrame) -> None:
    """Write feature statistics indexed by window end time for W=30/60/120.

    Statistics are calculated from solar-wind features only.  ``min_periods``
    always equals the statistic's own window, so incomplete histories remain
    NaN rather than being partly filled.  In addition to same-window moments,
    each output contains cumulative Bz-south, Epsilon, Newell, and Borovsky
    forcing plus Bz-south duration at all scales no longer than the sequence
    window.  These values end at the input timestamp and therefore cannot
    expose target-period information.
    """

    for window in WINDOWS_MINUTES:
        output = pd.DataFrame({"time": frame["time"]})
        for feature in WINDOW_STAT_FEATURES:
            series = frame[feature].astype("float64")
            grouped = series.groupby(frame["event_group"], sort=False)
            output[f"{feature}_mean_W{window}"] = grouped.transform(lambda values: values.rolling(window, min_periods=window).mean()).astype("float32")
            output[f"{feature}_max_W{window}"] = grouped.transform(lambda values: values.rolling(window, min_periods=window).max()).astype("float32")
            output[f"{feature}_std_W{window}"] = grouped.transform(lambda values: values.rolling(window, min_periods=window).std(ddof=0)).astype("float32")
        for accumulation_window in (value for value in ACCUMULATION_WINDOWS_MINUTES if value <= window):
            for feature in COUPLING_ACCUMULATION_FEATURES:
                grouped = frame[feature].astype("float64").groupby(frame["event_group"], sort=False)
                unit_suffix = "_nT_min" if feature == "Bz_south" else ""
                output[f"{feature}_sum{unit_suffix}_W{accumulation_window}"] = grouped.transform(
                    lambda values: values.rolling(accumulation_window, min_periods=accumulation_window).sum()
                ).astype("float32")
            output[f"Bz_south_duration_min_W{accumulation_window}"] = (
                frame["Bz_south"].gt(0).astype("float64").groupby(frame["event_group"], sort=False).transform(
                    lambda values: values.rolling(accumulation_window, min_periods=accumulation_window).sum()
                ).astype("float32")
            )
        output.to_csv(OUTPUT_DIR / f"window_statistics_W{window}.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")


def plot_yearly_features(frame: pd.DataFrame) -> None:
    plot_features = (*WIND_FEATURES, GIC_FEATURE, "Newell_coupling", "Borovsky_Rquick_mV_m")
    units = {
        "Btot": "nT", "Bx_gsm": "nT", "By_gsm": "nT", "Bz_gsm": "nT",
        "V": "km s-1", "Np": "cm-3", "Psw": "nPa", "Ma": "1", "Mms": "1",
        "Epsilon": "10^11 W", "VBs": "mV m-1", "GIC": "A",
        "Newell_coupling": "(km s-1)^(4/3) nT^(2/3)", "Borovsky_Rquick_mV_m": "mV m-1",
    }
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for year in range(START_TIME.year, END_TIME.year + 1):
        year_frame = frame.loc[frame["time"].dt.year.eq(year)]
        fig, axes = plt.subplots(7, 2, figsize=(22, 26), sharex=True, constrained_layout=True)
        for axis, feature in zip(axes.ravel(), plot_features):
            # Matplotlib breaks a line at NaN, leaving missing time intervals blank.
            axis.plot(year_frame["time"], year_frame[feature], linewidth=0.22, color="#1f4e79")
            axis.set_ylabel(f"{feature}\n[{units[feature]}]", fontsize=8)
            axis.grid(alpha=0.25, linewidth=0.4)
        locator = mdates.MonthLocator(interval=1)
        formatter = mdates.DateFormatter("%Y-%m")
        for axis in axes[-1, :]:
            axis.xaxis.set_major_locator(locator)
            axis.xaxis.set_major_formatter(formatter)
            axis.tick_params(axis="x", rotation=45, labelsize=8)
        fig.suptitle(f"VKH GIC and Wind solar-wind features: {year} (UTC)", fontsize=14)
        fig.savefig(PLOT_DIR / f"features_{year}.png", dpi=180)
        plt.close(fig)


def write_metadata(
    alignment_report: dict[str, object],
    configurations: pd.DataFrame,
    selected_events: pd.DataFrame,
    event_groups: pd.DataFrame,
    split_ratios: tuple[float, float, float],
) -> None:
    metadata = {
        "time_grid": "UTC, 1 minute, 2012-01-01 00:00 through 2022-12-31 23:59",
        "dataset_type": "event_window_blocks",
        "event_window": "[peak_time - 5 days, peak_time + 5 days)",
        "target": {
            "definition": "max(abs(GIC)) over (t, t+30 min]",
            "thresholds_A": list(THRESHOLDS_A),
            "onset_auxiliary_definition": "a below-to-at-or-above-threshold |GIC| crossing over (t, t+30 min]",
        },
        "input_window": "[t-L-W+1 min, t-L min] inclusive",
        "lags_minutes": list(LAGS_MINUTES),
        "window_lengths_minutes": list(WINDOWS_MINUTES),
        "sequence_features": list(SEQUENCE_FEATURES),
        "source_features": list(WIND_FEATURES) + [GIC_FEATURE],
        "coupling_functions": {
            "Newell_coupling": "V[km/s]^(4/3) Bt[nT]^(2/3) sin(theta/2)^(8/3), Bt=sqrt(By_gsm^2+Bz_gsm^2), theta=atan2(By_gsm,Bz_gsm) mod 2pi",
            "Borovsky_Rquick_mV_m": "V[km/s] * Bt[nT] * sin(theta/2)^2 * 1e-3, the Borovsky rapid dayside-reconnection driver Rquick in mV/m.",
            "Akasofu": "Wind Epsilon is retained exactly as supplied (Wind SWE display unit: 10^11 W); it is not recomputed.",
        },
        "alignment": alignment_report,
        "event_selection": {
            "source_event_count": 92,
            "selected_event_count": int(len(selected_events)),
            "selected_driver_counts": selected_events["driver"].value_counts().astype(int).to_dict(),
            "excluded_engineering_event_ids": sorted(EVENT_EXCLUSIONS),
            "event_group_count": int(len(event_groups)),
            "split_event_counts": selected_events.groupby("split").size().astype(int).to_dict(),
            "split_group_counts": event_groups.groupby("split").size().astype(int).to_dict(),
            "requested_split_ratios": {
                "train": split_ratios[0],
                "validation": split_ratios[1],
                "test": split_ratios[2],
            },
            "split_method": "chronological whole event blocks nearest the requested event-count ratios",
        },
        "feature_count_note": "The requested source selection contains 12 variables (11 Wind plus GIC). Adding Newell and Borovsky produces 14 plotted variables.",
        "event_selection_status": "Applied from the supplied VKH Table 1; event 18 and 19 engineering-affected records are excluded.",
    }
    (OUTPUT_DIR / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    configurations.to_csv(OUTPUT_DIR / "lag_window_configurations.csv", index=False)


def write_split_manifests(
    selected_events: pd.DataFrame,
    event_groups: pd.DataFrame,
    event_timeline: pd.DataFrame,
    configurations: pd.DataFrame,
) -> None:
    """Write the exact events, event blocks, time ranges, and valid samples in each split."""
    manifest_columns = [
        "split", "event_id", "driver", "peak_time", "gic_peak_abs_A", "event_group",
        "event_window_start", "event_window_end", "segment_start", "segment_end",
        "max_missing_GIC_min", "max_missing_solar_wind_min",
    ]
    selected_events.loc[:, manifest_columns].sort_values(["split", "peak_time"]).to_csv(
        OUTPUT_DIR / "event_split_manifest.csv", index=False, date_format="%Y-%m-%d %H:%M:%S",
    )
    rows = []
    for split in ("train", "validation", "test"):
        events = selected_events.loc[selected_events["split"].eq(split)]
        blocks = event_groups.loc[event_groups["split"].eq(split)]
        samples = event_timeline.loc[event_timeline["split"].eq(split)]
        row = {
            "split": split,
            "event_count": int(len(events)),
            "event_ids": ",".join(map(str, events.sort_values("peak_time")["event_id"])),
            "event_group_count": int(len(blocks)),
            "event_group_ids": ",".join(map(str, blocks["event_group"])),
            "timeline_start": samples["time"].min() if len(samples) else pd.NaT,
            "timeline_end": samples["time"].max() if len(samples) else pd.NaT,
            "timeline_rows": int(len(samples)),
        }
        for config in configurations.itertuples(index=False):
            row[f"valid_samples_L{config.lag_minutes}_W{config.window_minutes}"] = int(
                samples[config.validity_column].sum()
            )
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "split_dataset_summary.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")


def main() -> None:
    global OUTPUT_DIR, PLOT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-plots", action="store_true", help="Do not generate diagnostic event plots.")
    parser.add_argument("--skip-window-statistics", action="store_true", help="Do not write W=30/60/120 rolling-statistic tables.")
    parser.add_argument(
        "--split-ratios", type=float, nargs=3, metavar=("TRAIN", "VALIDATION", "TEST"),
        default=DEFAULT_SPLIT_RATIOS,
        help="Chronological train/validation/test event-count ratios; defaults to 0.7 0.2 0.1.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Output directory relative to this script, unless an absolute path is supplied.",
    )
    args = parser.parse_args()

    split_ratios = tuple(float(value) for value in args.split_ratios)
    if any(value <= 0.0 for value in split_ratios) or not np.isclose(sum(split_ratios), 1.0):
        raise ValueError("--split-ratios must contain three positive values that sum to 1.")
    OUTPUT_DIR = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    OUTPUT_DIR = OUTPUT_DIR.resolve()
    PLOT_DIR = OUTPUT_DIR / "yearly_feature_plots"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timeline, alignment_report = load_and_align()
    (OUTPUT_DIR / "time_alignment_report.json").write_text(json.dumps(alignment_report, indent=2), encoding="utf-8")

    timeline = calculate_coupling_features(timeline)
    timeline = add_prediction_targets(timeline)
    timeline, _ = add_lag_window_masks(timeline)
    # This report describes the complete 02 one-minute sources, before event
    # selection removes long periods from the modelling timeline.
    missing_rows = []
    for feature in (*WIND_FEATURES, GIC_FEATURE, "Newell_coupling", "Borovsky_Rquick_mV_m"):
        missing_rows.extend(find_long_missing_intervals(timeline, feature))
    pd.DataFrame(
        missing_rows,
        columns=["feature", "start_time", "end_time", "missing_minutes", "missing_days"],
    ).to_csv(OUTPUT_DIR / "continuous_missing_over_1day.csv", index=False)
    source_events = load_vkh_event_table()
    source_events.to_csv(OUTPUT_DIR / "vkh_event_catalog_source.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    assessed_events, long_gap_events = assess_event_data_availability(source_events, timeline)
    assessed_events.to_csv(OUTPUT_DIR / "vkh_event_data_eligibility_audit.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    long_gap_events.to_csv(OUTPUT_DIR / "excluded_events_long_gap_details.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    selected_events, event_groups = build_event_catalog(assessed_events, split_ratios=split_ratios)
    event_timeline, configurations = build_event_timeline(timeline, selected_events)
    event_timeline["GIC_abs"] = event_timeline[GIC_FEATURE].abs().astype("float32")
    selected_events.to_csv(OUTPUT_DIR / "vkh_event_catalog_selected.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    event_groups.to_csv(OUTPUT_DIR / "vkh_event_groups.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    write_split_manifests(selected_events, event_groups, event_timeline, configurations)

    output_columns = [
        "time", *WIND_FEATURES, GIC_FEATURE, "GIC_abs", "Bt_gsm", "Bz_south", "clock_angle_rad",
        "Newell_coupling", "Borovsky_Rquick_mV_m", "future_gic_peak_abs_30min_A",
        "target_window_complete_30min",
        *[f"target_exceeds_{threshold}A_30min" for threshold in THRESHOLDS_A],
        *[f"target_onset_{threshold}A_30min" for threshold in THRESHOLDS_A],
        *configurations["validity_column"].tolist(), "event_group", "split", "segment_start", "segment_end",
    ]
    event_timeline.loc[:, output_columns].to_csv(OUTPUT_DIR / "prediction_timeline.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    write_metadata(alignment_report, configurations, selected_events, event_groups, split_ratios)
    if not args.skip_window_statistics:
        build_window_statistics(event_timeline)
    if not args.skip_plots:
        print("Event-window dataset: skipped connected yearly plots to avoid joining separate event blocks.")
    print(f"Event prediction timeline: {OUTPUT_DIR / 'prediction_timeline.csv'} | rows={len(event_timeline):,}")
    print(f"Selected events: {len(selected_events)} | event blocks={len(event_groups)} | split counts={selected_events['split'].value_counts().to_dict()}")
    print(f"Time alignment: {OUTPUT_DIR / 'time_alignment_report.json'}")
    print(f"Lag/window configurations: {OUTPUT_DIR / 'lag_window_configurations.csv'}")


if __name__ == "__main__":
    main()
