from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


DEFAULT_GROUPS = [
    "solar_lag0_source_window_plus_time",
    "solar_lag30_source_window_plus_time",
    "solar_lag45_source_window_plus_time",
    "solar_lag60_source_window_plus_time",
    "solar_lag90_source_window_plus_time",
    "solar_source_windows_plus_time",
]


def _extract_intervals(
    time_index: pd.DatetimeIndex,
    mask: np.ndarray,
    min_duration_min: int = 1,
    merge_gap_min: int = 0,
    max_time_gap_min: int = 2,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, int]]:
    mask = np.asarray(mask, dtype=bool)
    if len(time_index) == 0 or len(mask) == 0:
        return []
    if len(time_index) != len(mask):
        raise ValueError("time_index and mask length mismatch")
    intervals: List[Tuple[pd.Timestamp, pd.Timestamp, int]] = []
    start = None
    last_true = None
    for i, flag in enumerate(mask):
        gap_break = False
        if i > 0:
            gap_min = (time_index[i] - time_index[i - 1]) / pd.Timedelta(minutes=1)
            gap_break = bool(gap_min > float(max_time_gap_min))
        if gap_break and start is not None:
            end = last_true if last_true is not None else i - 1
            duration = int(round((time_index[end] - time_index[start]) / pd.Timedelta(minutes=1))) + 1
            if duration >= int(min_duration_min):
                intervals.append((pd.Timestamp(time_index[start]), pd.Timestamp(time_index[end]), duration))
            start = None
            last_true = None
        if flag and start is None:
            start = i
        if flag:
            last_true = i
        if (not flag or i == len(mask) - 1) and start is not None:
            end = last_true if last_true is not None else i
            duration = int(round((time_index[end] - time_index[start]) / pd.Timedelta(minutes=1))) + 1
            if duration >= int(min_duration_min):
                intervals.append((pd.Timestamp(time_index[start]), pd.Timestamp(time_index[end]), duration))
            start = None
            last_true = None

    if not intervals or merge_gap_min <= 0:
        return intervals

    merged: List[Tuple[pd.Timestamp, pd.Timestamp, int]] = []
    current_start, current_end, _ = intervals[0]
    for start_time, end_time, _ in intervals[1:]:
        gap = (start_time - current_end) / pd.Timedelta(minutes=1) - 1
        if gap < float(merge_gap_min):
            current_end = end_time
        else:
            duration = int(round((current_end - current_start) / pd.Timedelta(minutes=1))) + 1
            merged.append((current_start, current_end, duration))
            current_start, current_end = start_time, end_time
    duration = int(round((current_end - current_start) / pd.Timedelta(minutes=1))) + 1
    merged.append((current_start, current_end, duration))
    return merged


def _is_match(
    pred_start: pd.Timestamp,
    pred_end: pd.Timestamp,
    true_start: pd.Timestamp,
    true_end: pd.Timestamp,
    tolerance_min: int,
) -> bool:
    tolerance = pd.Timedelta(minutes=int(tolerance_min))
    if pred_start <= true_end and pred_end >= true_start:
        return True
    if abs(pred_start - true_end) <= tolerance:
        return True
    if abs(pred_end - true_start) <= tolerance:
        return True
    if abs(pred_start - true_start) <= tolerance:
        return True
    if abs(pred_end - true_end) <= tolerance:
        return True
    return False


def _event_metrics(
    true_events: List[Tuple[pd.Timestamp, pd.Timestamp, int]],
    pred_events: List[Tuple[pd.Timestamp, pd.Timestamp, int]],
    tolerance_min: int,
) -> Tuple[Dict[str, float], List[Dict[str, object]], List[Dict[str, object]]]:
    hit_true = set()
    pred_rows = []
    for pred_id, (pred_start, pred_end, pred_duration) in enumerate(pred_events, start=1):
        candidates = []
        for true_id, (true_start, true_end, true_duration) in enumerate(true_events, start=1):
            if _is_match(pred_start, pred_end, true_start, true_end, tolerance_min):
                overlap_start = max(pred_start, true_start)
                overlap_end = min(pred_end, true_end)
                overlap_min = max(0.0, (overlap_end - overlap_start) / pd.Timedelta(minutes=1) + 1)
                center_gap = abs(
                    ((pred_start + (pred_end - pred_start) / 2) - (true_start + (true_end - true_start) / 2))
                    / pd.Timedelta(minutes=1)
                )
                candidates.append((overlap_min, -center_gap, true_id, true_start, true_end, true_duration))
        if candidates:
            candidates.sort(reverse=True)
            _, _, true_id, true_start, true_end, true_duration = candidates[0]
            for _, _, matched_id, _, _, _ in candidates:
                hit_true.add(matched_id)
            pred_rows.append(
                {
                    "pred_event_id": pred_id,
                    "pred_start": pred_start,
                    "pred_end": pred_end,
                    "pred_duration_min": pred_duration,
                    "matched_true_event_id": true_id,
                    "true_start": true_start,
                    "true_end": true_end,
                    "true_duration_min": true_duration,
                    "outcome": "TP",
                }
            )
        else:
            pred_rows.append(
                {
                    "pred_event_id": pred_id,
                    "pred_start": pred_start,
                    "pred_end": pred_end,
                    "pred_duration_min": pred_duration,
                    "matched_true_event_id": np.nan,
                    "true_start": pd.NaT,
                    "true_end": pd.NaT,
                    "true_duration_min": np.nan,
                    "outcome": "FP",
                }
            )

    true_rows = []
    for true_id, (true_start, true_end, true_duration) in enumerate(true_events, start=1):
        outcome = "TP" if true_id in hit_true else "FN"
        true_rows.append(
            {
                "true_event_id": true_id,
                "true_start": true_start,
                "true_end": true_end,
                "true_duration_min": true_duration,
                "outcome": outcome,
            }
        )

    tp = int(len(hit_true))
    fp = int(sum(row["outcome"] == "FP" for row in pred_rows))
    fn = int(len(true_events) - tp)
    pod = tp / (tp + fn) if (tp + fn) else 0.0
    far = fp / (tp + fp) if (tp + fp) else 0.0
    csi = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    bias = (tp + fp) / (tp + fn) if (tp + fn) else np.nan
    return (
        {
            "event_TP": tp,
            "event_FP": fp,
            "event_FN": fn,
            "event_true_n": int(len(true_events)),
            "event_pred_n": int(len(pred_events)),
            "event_POD": float(pod),
            "event_FAR": float(far),
            "event_CSI": float(csi),
            "event_F1": float(f1),
            "event_Bias": float(bias),
        },
        true_rows,
        pred_rows,
    )


def _sample_metrics(y_true_binary: np.ndarray, y_pred_binary: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    truth = np.asarray(y_true_binary, dtype=int)
    pred = np.asarray(y_pred_binary, dtype=int)
    tp = int(np.sum((truth == 1) & (pred == 1)))
    fp = int(np.sum((truth == 0) & (pred == 1)))
    fn = int(np.sum((truth == 1) & (pred == 0)))
    tn = int(np.sum((truth == 0) & (pred == 0)))
    pod = tp / (tp + fn) if (tp + fn) else 0.0
    pofd = fp / (fp + tn) if (fp + tn) else 0.0
    far = fp / (tp + fp) if (tp + fp) else 0.0
    csi = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    tss = pod - pofd
    hss_denom = 2 * ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    hss = 2 * (tp * tn - fp * fn) / hss_denom if hss_denom else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    bias = (tp + fp) / (tp + fn) if (tp + fn) else np.nan
    try:
        auc = roc_auc_score(truth, y_score) if len(np.unique(truth)) == 2 else np.nan
    except ValueError:
        auc = np.nan
    return {
        "sample_TP": tp,
        "sample_FP": fp,
        "sample_TN": tn,
        "sample_FN": fn,
        "sample_POD": float(pod),
        "sample_POFD": float(pofd),
        "sample_FAR": float(far),
        "sample_CSI": float(csi),
        "sample_TSS": float(tss),
        "sample_HSS": float(hss),
        "sample_F1": float(f1),
        "sample_AUC": float(auc),
        "sample_Bias": float(bias),
    }


def evaluate_file(
    pred_path: str,
    group_name: str,
    thresholds: Iterable[float],
    probability_threshold: float,
    min_alarm_duration_min: int,
    merge_gap_min: int,
    match_tolerance_min: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    pred = pd.read_csv(pred_path)
    pred["datetime"] = pd.to_datetime(pred["datetime"], errors="coerce")
    pred = pred.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    time_index = pd.DatetimeIndex(pred.index)
    summary_rows, true_event_rows, pred_event_rows = [], [], []
    for threshold in thresholds:
        score_col = f"prob_ge_{float(threshold):g}A"
        if score_col not in pred.columns:
            continue
        y_true_binary = pd.to_numeric(pred["y_true"], errors="coerce").to_numpy(dtype=float) >= float(threshold)
        score = pd.to_numeric(pred[score_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        raw_alarm = score >= float(probability_threshold)
        true_events = _extract_intervals(time_index, y_true_binary, min_duration_min=1, merge_gap_min=0)
        pred_events = _extract_intervals(
            time_index,
            raw_alarm,
            min_duration_min=min_alarm_duration_min,
            merge_gap_min=merge_gap_min,
        )
        pred_event_mask = np.zeros(len(time_index), dtype=bool)
        for start, end, _ in pred_events:
            pred_event_mask[(time_index >= start) & (time_index <= end)] = True
        event_metric, true_rows, pred_rows = _event_metrics(true_events, pred_events, match_tolerance_min)
        sample_metric = _sample_metrics(y_true_binary.astype(int), pred_event_mask.astype(int), score)
        summary_rows.append(
            {
                "feature_group": group_name,
                "threshold_A": float(threshold),
                "score_column": score_col,
                "probability_threshold": float(probability_threshold),
                "min_alarm_duration_min": int(min_alarm_duration_min),
                "merge_gap_min": int(merge_gap_min),
                "match_tolerance_min": int(match_tolerance_min),
                **event_metric,
                **sample_metric,
            }
        )
        for row in true_rows:
            true_event_rows.append({"feature_group": group_name, "threshold_A": float(threshold), **row})
        for row in pred_rows:
            pred_event_rows.append({"feature_group": group_name, "threshold_A": float(threshold), **row})
    return summary_rows, true_event_rows, pred_event_rows


def _evaluate_one_threshold(
    pred: pd.DataFrame,
    group_name: str,
    threshold: float,
    probability_threshold: float,
    min_alarm_duration_min: int,
    merge_gap_min: int,
    match_tolerance_min: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    time_index = pd.DatetimeIndex(pred.index)
    score_col = f"prob_ge_{float(threshold):g}A"
    y_true_binary = pd.to_numeric(pred["y_true"], errors="coerce").to_numpy(dtype=float) >= float(threshold)
    score = pd.to_numeric(pred[score_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    raw_alarm = score >= float(probability_threshold)
    true_events = _extract_intervals(time_index, y_true_binary, min_duration_min=1, merge_gap_min=0)
    pred_events = _extract_intervals(
        time_index,
        raw_alarm,
        min_duration_min=min_alarm_duration_min,
        merge_gap_min=merge_gap_min,
    )
    pred_event_mask = np.zeros(len(time_index), dtype=bool)
    for start, end, _ in pred_events:
        pred_event_mask[(time_index >= start) & (time_index <= end)] = True
    event_metric, true_rows, pred_rows = _event_metrics(true_events, pred_events, match_tolerance_min)
    sample_metric = _sample_metrics(y_true_binary.astype(int), pred_event_mask.astype(int), score)
    summary = {
        "feature_group": group_name,
        "threshold_A": float(threshold),
        "score_column": score_col,
        "probability_threshold": float(probability_threshold),
        "min_alarm_duration_min": int(min_alarm_duration_min),
        "merge_gap_min": int(merge_gap_min),
        "match_tolerance_min": int(match_tolerance_min),
        **event_metric,
        **sample_metric,
    }
    return summary, true_rows, pred_rows


def evaluate_file_low_far(
    pred_path: str,
    group_name: str,
    thresholds: Iterable[float],
    max_event_far: float,
    threshold_grid_step: float,
    min_alarm_duration_min: int,
    merge_gap_min: int,
    match_tolerance_min: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    pred = pd.read_csv(pred_path)
    pred["datetime"] = pd.to_datetime(pred["datetime"], errors="coerce")
    pred = pred.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    summary_rows, pred_event_rows = [], []
    grid_step = max(float(threshold_grid_step), 1e-3)
    candidates = np.arange(0.5, 1.0, grid_step)
    candidates = np.unique(np.clip(np.concatenate([candidates, [0.5]]), 0.5, 0.999))
    for threshold in thresholds:
        score_col = f"prob_ge_{float(threshold):g}A"
        if score_col not in pred.columns:
            continue
        best = None
        fallback = None
        for prob_threshold in candidates:
            summary, _, pred_rows = _evaluate_one_threshold(
                pred=pred,
                group_name=group_name,
                threshold=float(threshold),
                probability_threshold=float(prob_threshold),
                min_alarm_duration_min=min_alarm_duration_min,
                merge_gap_min=merge_gap_min,
                match_tolerance_min=match_tolerance_min,
            )
            feasible_key = (
                summary["event_CSI"],
                summary["event_POD"],
                summary["event_F1"],
                -float(prob_threshold),
            )
            fallback_key = (
                -summary["event_FAR"],
                summary["event_CSI"],
                summary["event_POD"],
                float(prob_threshold),
            )
            if summary["event_FAR"] <= float(max_event_far):
                if best is None or feasible_key > best[0]:
                    best = (feasible_key, summary, pred_rows, "far_control")
            if fallback is None or fallback_key > fallback[0]:
                fallback = (fallback_key, summary, pred_rows, "fallback_min_far")
        _, chosen_summary, chosen_pred_rows, status = best if best is not None else fallback
        chosen_summary = dict(chosen_summary)
        chosen_summary["event_threshold_strategy"] = "low-far"
        chosen_summary["max_event_FAR"] = float(max_event_far)
        chosen_summary["selection_status"] = status
        chosen_summary["threshold_grid_step"] = float(grid_step)
        summary_rows.append(chosen_summary)
        for row in chosen_pred_rows:
            pred_event_rows.append(
                {
                    "feature_group": group_name,
                    "threshold_A": float(threshold),
                    "probability_threshold": float(chosen_summary["probability_threshold"]),
                    "max_event_FAR": float(max_event_far),
                    **row,
                }
            )
    return summary_rows, pred_event_rows


def evaluate_file_low_far_overall(
    pred_path: str,
    group_name: str,
    thresholds: Iterable[float],
    max_event_far: float,
    threshold_grid_step: float,
    min_alarm_duration_min: int,
    merge_gap_min: int,
    match_tolerance_min: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    pred = pd.read_csv(pred_path)
    pred["datetime"] = pd.to_datetime(pred["datetime"], errors="coerce")
    pred = pred.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    thresholds = [float(x) for x in thresholds]
    grid_step = max(float(threshold_grid_step), 1e-3)
    candidates = np.arange(0.5, 1.0, grid_step)
    candidates = np.unique(np.clip(np.concatenate([candidates, [0.5]]), 0.5, 0.999))
    best = None
    fallback = None
    scan_rows = []
    for prob_threshold in candidates:
        summaries = []
        pred_rows_all = []
        for threshold in thresholds:
            score_col = f"prob_ge_{float(threshold):g}A"
            if score_col not in pred.columns:
                continue
            summary, _, pred_rows = _evaluate_one_threshold(
                pred=pred,
                group_name=group_name,
                threshold=float(threshold),
                probability_threshold=float(prob_threshold),
                min_alarm_duration_min=min_alarm_duration_min,
                merge_gap_min=merge_gap_min,
                match_tolerance_min=match_tolerance_min,
            )
            summaries.append(summary)
            for row in pred_rows:
                pred_rows_all.append(
                    {
                        "feature_group": group_name,
                        "threshold_A": float(threshold),
                        "probability_threshold": float(prob_threshold),
                        "max_event_FAR": float(max_event_far),
                        **row,
                    }
                )
        if not summaries:
            continue
        frame = pd.DataFrame(summaries)
        overall = {
            "feature_group": group_name,
            "event_threshold_strategy": "overall-low-far",
            "probability_threshold": float(prob_threshold),
            "max_event_FAR": float(max_event_far),
            "threshold_grid_step": float(grid_step),
            "mean_event_POD": float(frame["event_POD"].mean()),
            "mean_event_FAR": float(frame["event_FAR"].mean()),
            "mean_event_CSI": float(frame["event_CSI"].mean()),
            "mean_event_F1": float(frame["event_F1"].mean()),
            "mean_event_Bias": float(frame["event_Bias"].mean()),
            "sum_event_TP": int(frame["event_TP"].sum()),
            "sum_event_FP": int(frame["event_FP"].sum()),
            "sum_event_FN": int(frame["event_FN"].sum()),
            "sum_event_true_n": int(frame["event_true_n"].sum()),
            "sum_event_pred_n": int(frame["event_pred_n"].sum()),
        }
        scan_rows.append(overall)
        feasible_key = (
            overall["mean_event_CSI"],
            overall["mean_event_F1"],
            overall["mean_event_POD"],
            -overall["mean_event_FAR"],
            -float(prob_threshold),
        )
        fallback_key = (
            -overall["mean_event_FAR"],
            overall["mean_event_CSI"],
            overall["mean_event_F1"],
            overall["mean_event_POD"],
            -float(prob_threshold),
        )
        if overall["mean_event_FAR"] <= float(max_event_far):
            if best is None or feasible_key > best[0]:
                best = (feasible_key, summaries, overall, pred_rows_all, "far_control")
        if fallback is None or fallback_key > fallback[0]:
            fallback = (fallback_key, summaries, overall, pred_rows_all, "fallback_min_far")
    _, summaries, overall, pred_rows_all, status = best if best is not None else fallback
    overall = dict(overall)
    overall["selection_status"] = status
    chosen_rows = []
    for summary in summaries:
        row = dict(summary)
        row["event_threshold_strategy"] = "overall-low-far"
        row["max_event_FAR"] = float(max_event_far)
        row["selection_status"] = status
        row["threshold_grid_step"] = float(grid_step)
        chosen_rows.append(row)
    return chosen_rows, overall, pred_rows_all, scan_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Contiguous threshold-event matching metrics from prediction cache.")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--thresholds", nargs="*", type=float, default=[3.0, 5.0, 10.0, 20.0])
    parser.add_argument("--groups", nargs="*", default=DEFAULT_GROUPS)
    parser.add_argument("--probability-threshold", type=float, default=0.5)
    parser.add_argument("--low-far-max-event-far", type=float, default=None)
    parser.add_argument("--low-far-threshold-grid-step", type=float, default=0.01)
    parser.add_argument(
        "--low-far-mode",
        choices=["per-threshold", "overall", "both"],
        default="both",
        help="Select probability thresholds separately per GIC threshold or one threshold by overall mean event metrics.",
    )
    parser.add_argument("--min-alarm-duration-min", type=int, default=3)
    parser.add_argument("--merge-gap-min", type=int, default=10)
    parser.add_argument("--match-tolerance-min", type=int, default=10)
    args = parser.parse_args()

    experiment_dir = os.path.abspath(args.experiment_dir)
    cache_dir = os.path.join(experiment_dir, "prediction_cache")
    out_dir = os.path.join(experiment_dir, "contiguous_event_eval")
    os.makedirs(out_dir, exist_ok=True)
    summary_rows, true_rows, pred_rows = [], [], []
    low_far_rows, low_far_pred_rows = [], []
    overall_low_far_rows, overall_low_far_summary_rows, overall_low_far_pred_rows, overall_scan_rows = [], [], [], []
    for group in args.groups:
        pred_path = os.path.join(cache_dir, f"{group}_predictions.csv")
        if not os.path.exists(pred_path):
            print(f"[ContiguousEvent] Skip missing cache: {pred_path}")
            continue
        group_summary, group_true, group_pred = evaluate_file(
            pred_path=pred_path,
            group_name=group,
            thresholds=args.thresholds,
            probability_threshold=args.probability_threshold,
            min_alarm_duration_min=args.min_alarm_duration_min,
            merge_gap_min=args.merge_gap_min,
            match_tolerance_min=args.match_tolerance_min,
        )
        summary_rows.extend(group_summary)
        true_rows.extend(group_true)
        pred_rows.extend(group_pred)
        if args.low_far_max_event_far is not None and args.low_far_mode in {"per-threshold", "both"}:
            group_low_far, group_low_far_pred = evaluate_file_low_far(
                pred_path=pred_path,
                group_name=group,
                thresholds=args.thresholds,
                max_event_far=float(args.low_far_max_event_far),
                threshold_grid_step=float(args.low_far_threshold_grid_step),
                min_alarm_duration_min=args.min_alarm_duration_min,
                merge_gap_min=args.merge_gap_min,
                match_tolerance_min=args.match_tolerance_min,
            )
            low_far_rows.extend(group_low_far)
            low_far_pred_rows.extend(group_low_far_pred)
        if args.low_far_max_event_far is not None and args.low_far_mode in {"overall", "both"}:
            group_rows, group_overall, group_pred, group_scan = evaluate_file_low_far_overall(
                pred_path=pred_path,
                group_name=group,
                thresholds=args.thresholds,
                max_event_far=float(args.low_far_max_event_far),
                threshold_grid_step=float(args.low_far_threshold_grid_step),
                min_alarm_duration_min=args.min_alarm_duration_min,
                merge_gap_min=args.merge_gap_min,
                match_tolerance_min=args.match_tolerance_min,
            )
            overall_low_far_rows.extend(group_rows)
            overall_low_far_summary_rows.append(group_overall)
            overall_low_far_pred_rows.extend(group_pred)
            overall_scan_rows.extend(group_scan)

    summary = pd.DataFrame(summary_rows)
    true_events = pd.DataFrame(true_rows)
    pred_events = pd.DataFrame(pred_rows)
    summary.to_csv(os.path.join(out_dir, "contiguous_event_level_metrics.csv"), index=False, encoding="utf-8-sig")
    true_events.to_csv(os.path.join(out_dir, "true_contiguous_events.csv"), index=False, encoding="utf-8-sig")
    pred_events.to_csv(os.path.join(out_dir, "predicted_contiguous_events.csv"), index=False, encoding="utf-8-sig")
    if low_far_rows:
        pd.DataFrame(low_far_rows).to_csv(
            os.path.join(out_dir, "low_far_contiguous_event_level_metrics.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(low_far_pred_rows).to_csv(
            os.path.join(out_dir, "low_far_predicted_contiguous_events.csv"),
            index=False,
            encoding="utf-8-sig",
        )
    if overall_low_far_rows:
        pd.DataFrame(overall_low_far_rows).to_csv(
            os.path.join(out_dir, "overall_low_far_contiguous_event_level_metrics.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(overall_low_far_summary_rows).to_csv(
            os.path.join(out_dir, "overall_low_far_summary.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(overall_low_far_pred_rows).to_csv(
            os.path.join(out_dir, "overall_low_far_predicted_contiguous_events.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(overall_scan_rows).to_csv(
            os.path.join(out_dir, "overall_low_far_threshold_scan.csv"),
            index=False,
            encoding="utf-8-sig",
        )
    print(f"[ContiguousEvent] Saved: {out_dir}")
    if not summary.empty:
        print(summary.sort_values(["threshold_A", "feature_group"]).to_string(index=False))


if __name__ == "__main__":
    main()
