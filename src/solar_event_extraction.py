"""Solar-wind driver event extraction and event-level forecast metrics."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float32")
    return pd.to_numeric(df[col], errors="coerce").astype("float32")


def _q(series: pd.Series, q: float, fallback: float) -> float:
    arr = series.to_numpy(dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float(fallback)
    return float(max(np.nanquantile(arr, q), fallback))


def build_solar_driver_score(
    df: pd.DataFrame,
    threshold_frame: Optional[pd.DataFrame] = None,
    roll_min: int = 30,
    core_quantile: float = 0.92,
    assist_quantile: float = 0.95,
    min_score: float = 1.05,
    require_southward: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Add solar-driver rolling features, flags and a normalized score.

    Thresholds are estimated from ``threshold_frame`` when provided, so callers
    can fit event criteria on the training era and reuse them on held-out years.
    """
    out = df.copy()
    ref = threshold_frame if threshold_frame is not None and len(threshold_frame) else df
    win = max(int(roll_min), 1)

    out["Bz_south"] = _numeric(out, "Bz_gse").clip(upper=0).abs()
    ref_bz_south = _numeric(ref, "Bz_gse").clip(upper=0).abs()

    specs = {
        "Bz_south": (out["Bz_south"], ref_bz_south, core_quantile, 4.0),
        "Ey_mV/m": (_numeric(out, "Ey_mV/m"), _numeric(ref, "Ey_mV/m"), core_quantile, 1.5),
        "Newell": (_numeric(out, "Newell"), _numeric(ref, "Newell"), core_quantile, 0.0),
        "epsilon_norm": (_numeric(out, "epsilon_norm"), _numeric(ref, "epsilon_norm"), core_quantile, 0.0),
        "P_dyn_nPa": (_numeric(out, "P_dyn_nPa"), _numeric(ref, "P_dyn_nPa"), assist_quantile, 0.0),
        "Btot": (_numeric(out, "Btot"), _numeric(ref, "Btot"), assist_quantile, 0.0),
        "Vp": (_numeric(out, "Vp"), _numeric(ref, "Vp"), assist_quantile, 0.0),
    }

    thresholds: Dict[str, float] = {}
    score_parts = []
    for name, (series, ref_series, quantile, fallback) in specs.items():
        roll = series.rolling(win, min_periods=max(1, win // 3)).mean()
        ref_roll = ref_series.rolling(win, min_periods=max(1, win // 3)).mean()
        thr = _q(ref_roll, quantile, fallback)
        thresholds[f"{name}_roll{win}_thr"] = thr
        out[f"{name}_roll{win}"] = roll.astype("float32")
        out[f"{name}_high"] = (roll >= thr).fillna(False)
        denom = max(abs(thr), 1e-6)
        score_parts.append((roll / denom).clip(lower=0, upper=3).fillna(0.0))

    bz_high = out["Bz_south_high"]
    coupling_high = out["Ey_mV/m_high"] | out["Newell_high"] | out["epsilon_norm_high"]
    if require_southward:
        core_flag = bz_high & coupling_high
    else:
        core_flag = bz_high | coupling_high
    assist_flag = out["P_dyn_nPa_high"] | out["Btot_high"] | out["Vp_high"]
    bz_soft = out[f"Bz_south_roll{win}"] >= (0.7 * thresholds[f"Bz_south_roll{win}_thr"])
    out["solar_core_flag"] = core_flag.fillna(False)
    out["solar_assist_flag"] = assist_flag.fillna(False)
    out["solar_driver_score"] = pd.concat(score_parts, axis=1).mean(axis=1).astype("float32")
    score_flag = out["solar_driver_score"] >= float(min_score)
    out["solar_event_flag"] = ((core_flag | (assist_flag & bz_soft & coupling_high)) & score_flag).fillna(False)
    return out, thresholds


def _bool_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start = None
    for i, val in enumerate(mask.astype(bool)):
        if val and start is None:
            start = i
        elif not val and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _merge_time_intervals(
    intervals: List[Tuple[pd.Timestamp, pd.Timestamp]],
    max_gap: pd.Timedelta,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda item: item[0])
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + max_gap:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def extract_solar_wind_driver_events(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    roll_min: int = 30,
    core_quantile: float = 0.92,
    assist_quantile: float = 0.95,
    min_score: float = 1.05,
    require_southward: bool = True,
    min_peak_score: float = 1.25,
    min_core_hit_ratio: float = 0.20,
    min_duration_min: int = 60,
    merge_gap_min: int = 180,
    pre_context_min: int = 360,
    post_context_min: int = 1440,
    threshold_frame: Optional[pd.DataFrame] = None,
) -> Tuple[List[Tuple[pd.Timestamp, pd.Timestamp]], List[Tuple[pd.Timestamp, pd.Timestamp]], pd.DataFrame, Dict[str, float]]:
    scored, thresholds = build_solar_driver_score(
        df,
        threshold_frame=threshold_frame,
        roll_min=roll_min,
        core_quantile=core_quantile,
        assist_quantile=assist_quantile,
        min_score=min_score,
        require_southward=require_southward,
    )
    if len(scored) == 0:
        return [], [], pd.DataFrame(), thresholds

    index = pd.DatetimeIndex(scored.index)
    mask = scored["solar_event_flag"].to_numpy(dtype=bool)
    min_len = max(int(min_duration_min), 1)
    core_intervals: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for start_i, stop_i in _bool_runs(mask):
        if stop_i - start_i < min_len:
            continue
        core_intervals.append((index[start_i], index[stop_i - 1]))

    core_intervals = _merge_time_intervals(
        core_intervals, pd.Timedelta(minutes=max(int(merge_gap_min), 0))
    )
    data_start, data_end = index.min(), index.max()
    expanded = [
        (
            max(data_start, start - pd.Timedelta(minutes=int(pre_context_min))),
            min(data_end, end + pd.Timedelta(minutes=int(post_context_min))),
        )
        for start, end in core_intervals
    ]

    records = []
    for event_id, ((core_start, core_end), (start, end)) in enumerate(zip(core_intervals, expanded), start=1):
        core = scored.loc[core_start:core_end]
        event_type_guess = "mixed"
        if len(core) > 0:
            core_hit_ratio = float(core["solar_event_flag"].mean())
            vp_high = float(core["Vp_high"].mean()) if "Vp_high" in core else 0.0
            pdyn_high = float(core["P_dyn_nPa_high"].mean()) if "P_dyn_nPa_high" in core else 0.0
            btot_high = float(core["Btot_high"].mean()) if "Btot_high" in core else 0.0
            if vp_high >= 0.35 and pdyn_high < 0.35:
                event_type_guess = "CIR_like"
            elif pdyn_high >= 0.25 or btot_high >= 0.35:
                event_type_guess = "CME_sheath_like"
            peak_idx = core["solar_driver_score"].idxmax()
            peak_score = float(core["solar_driver_score"].max())
        else:
            peak_idx = core_start
            peak_score = float("nan")
            core_hit_ratio = float("nan")
        records.append(
            {
                "event_id": event_id,
                "split": "pending",
                "start": start,
                "end": end,
                "solar_onset": core_start,
                "solar_end": core_end,
                "peak_driver_time": peak_idx,
                "driver_peak_score": peak_score,
                "core_hit_ratio": core_hit_ratio,
                "event_type_guess": event_type_guess,
                "n_rows": int(len(scored.loc[start:end])),
            }
        )

    report = pd.DataFrame.from_records(records)
    if report.empty:
        return [], [], report, thresholds
    keep_mask = (
        pd.to_numeric(report["driver_peak_score"], errors="coerce").ge(float(min_peak_score))
        & pd.to_numeric(report["core_hit_ratio"], errors="coerce").ge(float(min_core_hit_ratio))
    )
    report = report[keep_mask].reset_index(drop=True)
    if report.empty:
        return [], [], report, thresholds
    report["event_id"] = np.arange(1, len(report) + 1)
    cut = max(1, min(len(report) - 1, int(round(len(report) * float(train_ratio)))))
    report.loc[: cut - 1, "split"] = "train"
    report.loc[cut:, "split"] = "test"
    train_intervals = [(r.start, r.end) for r in report[report["split"].eq("train")].itertuples()]
    test_intervals = [(r.start, r.end) for r in report[report["split"].eq("test")].itertuples()]
    return train_intervals, test_intervals, report, thresholds


def rows_from_intervals(df: pd.DataFrame, intervals: Iterable[Tuple[pd.Timestamp, pd.Timestamp]]) -> pd.DataFrame:
    parts = []
    for start, end in intervals:
        part = df.loc[pd.Timestamp(start):pd.Timestamp(end)]
        if len(part):
            parts.append(part)
    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts).sort_index()


def event_level_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    time_index: pd.DatetimeIndex,
    events_report: pd.DataFrame,
    threshold_a: float,
    decision_threshold: float,
) -> Dict[str, float]:
    if events_report is None or events_report.empty or len(y_true) == 0:
        return {}
    frame = pd.DataFrame(
        {"y_true": y_true, "y_score": y_score},
        index=pd.DatetimeIndex(time_index),
    ).sort_index()
    truth, pred, scores = [], [], []
    for row in events_report.itertuples(index=False):
        if getattr(row, "split", "") != "test":
            continue
        start = pd.Timestamp(getattr(row, "start"))
        end = pd.Timestamp(getattr(row, "end"))
        sub = frame.loc[start:end]
        if sub.empty:
            continue
        truth.append(int(float(sub["y_true"].max()) >= float(threshold_a)))
        max_score = float(sub["y_score"].max())
        scores.append(max_score)
        pred.append(int(max_score >= float(decision_threshold)))
    if len(truth) == 0:
        return {}
    truth_arr = np.asarray(truth, dtype=int)
    pred_arr = np.asarray(pred, dtype=int)
    score_arr = np.asarray(scores, dtype=float)
    tp = int(np.sum((truth_arr == 1) & (pred_arr == 1)))
    fp = int(np.sum((truth_arr == 0) & (pred_arr == 1)))
    fn = int(np.sum((truth_arr == 1) & (pred_arr == 0)))
    tn = int(np.sum((truth_arr == 0) & (pred_arr == 0)))
    pod = tp / (tp + fn) if (tp + fn) else 0.0
    pofd = fp / (fp + tn) if (fp + tn) else 0.0
    far = fp / (tp + fp) if (tp + fp) else 0.0
    csi = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    tss = pod - pofd
    denom_hss = 2 * ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    hss = 2 * (tp * tn - fp * fn) / denom_hss if denom_hss else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    bias = (tp + fp) / (tp + fn) if (tp + fn) else np.nan
    auc = roc_auc_score(truth_arr, score_arr) if len(np.unique(truth_arr)) == 2 else np.nan
    return {
        "event_n": int(len(truth_arr)),
        "event_TP": tp,
        "event_FP": fp,
        "event_FN": fn,
        "event_TN": tn,
        "event_POD": float(pod),
        "event_POFD": float(pofd),
        "event_FAR": float(far),
        "event_CSI": float(csi),
        "event_TSS": float(tss),
        "event_HSS": float(hss),
        "event_F1": float(f1),
        "event_AUC": float(auc),
        "event_Bias": float(bias),
    }
