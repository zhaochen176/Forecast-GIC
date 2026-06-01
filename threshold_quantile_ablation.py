"""
Threshold-event classification + upper-quantile regression ablation.

This is a lightweight diagnostic experiment. It does not use the deep sequence
model. It tests whether tabular features can support:

1. threshold event prediction: GIC >= 3/5/10/20 A
2. upper-envelope prediction: Q80/Q90/Q95 quantile regression
3. feature ablation: solar coupling vs geomagnetic response vs time vs all-D

Outputs CSV reports under:
outputs/experiments/threshold_quantile_ablation/<scope>/<target>/
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    EXPERIMENT_DIR,
    PROCESSED_DATA_FILE,
    SEED,
    TARGET_VYK_COL,
)
from src.data_loader import build_vkh_event_type_report
from src.feature_engineering import get_feature_columns
from src.solar_event_extraction import event_level_classification_metrics


DEEP_BACKENDS = {"bilstm", "cnn_bilstm", "cnn_bilstm_attention", "cnn_bilstm_attention_gatefusion"}
DEEP_EPOCHS = int(os.environ.get("GIC_DEEP_EPOCHS", "25"))
DEEP_BATCH_SIZE = int(os.environ.get("GIC_DEEP_BATCH_SIZE", "2048"))
DEEP_LR = float(os.environ.get("GIC_DEEP_LR", "0.001"))
DEEP_HIDDEN_SIZE = int(os.environ.get("GIC_DEEP_HIDDEN_SIZE", "64"))
DEEP_DROPOUT = float(os.environ.get("GIC_DEEP_DROPOUT", "0.15"))


def _safe_name(text: str) -> str:
    return str(text).replace("/", "_").replace("\\", "_").replace(" ", "_")


def _rows_from_intervals(
    df: pd.DataFrame,
    intervals: Iterable[Tuple[pd.Timestamp, pd.Timestamp]],
) -> pd.DataFrame:
    parts = []
    for start, end in intervals:
        if pd.isna(start) or pd.isna(end):
            continue
        part = df.loc[pd.Timestamp(start):pd.Timestamp(end)]
        if len(part) > 0:
            parts.append(part)
    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts).sort_index()


def _add_future_window_max_label(
    df: pd.DataFrame,
    target_col: str,
    horizon: int,
) -> Tuple[pd.DataFrame, str]:
    if target_col not in df.columns:
        raise KeyError(f"Missing target column: {target_col}")
    label_col = f"{target_col}_future_max_H{int(horizon)}"
    s = pd.to_numeric(df[target_col], errors="coerce").astype(np.float32)
    future = s.shift(-1)
    df = df.copy()
    df[label_col] = (
        future.iloc[::-1]
        .rolling(window=int(horizon), min_periods=1)
        .max()
        .iloc[::-1]
        .astype(np.float32)
    )
    return df, label_col


def _split_events(
    df: pd.DataFrame,
    scope: str,
    event_types: Optional[List[str]],
    train_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    scope = scope.lower()
    if scope == "full-time":
        n = len(df)
        cut = max(1, min(n - 1, int(round(n * train_ratio))))
        return df.iloc[:cut].copy(), df.iloc[cut:].copy(), "full_time"

    events = build_vkh_event_type_report(df)
    events = events[
        events["in_data_range"].astype(bool)
        & events["quality_keep"].astype(bool)
    ].sort_values("peak_time")

    if scope == "event-type":
        keep = set(event_types or ["CME", "CIR"])
        events = events[events["event_type"].isin(keep)].copy()
        label = "event_type_" + "_".join(sorted(keep))
    elif scope == "fixed-events":
        label = "fixed_events"
    else:
        raise ValueError("scope must be fixed-events, event-type, or full-time")

    if len(events) < 2:
        raise RuntimeError(f"Not enough events for split: {len(events)}")

    cut = max(1, min(len(events) - 1, int(round(len(events) * train_ratio))))
    train_events = events.iloc[:cut]
    test_events = events.iloc[cut:]

    train_df = _rows_from_intervals(
        df,
        [(r["start"], r["end"]) for _, r in train_events.iterrows()],
    )
    test_df = _rows_from_intervals(
        df,
        [(r["start"], r["end"]) for _, r in test_events.iterrows()],
    )
    return train_df, test_df, label


def _root_groups(feature_cols: List[str]) -> Dict[str, List[str]]:
    def has_any(col: str, needles: List[str]) -> bool:
        c = col.lower()
        return any(n.lower() in c for n in needles)

    solar_keys = [
        "bz", "by", "btot", "vp", "np", "p_dyn", "ey", "newell",
        "epsilon", "borovsky", "ma", "vp_bz",
    ]
    geomag_keys = [
        "x_pert", "y_pert", "z_pert", "h_pert", "dx_pert", "dy_pert",
        "dz_pert", "dh_pert", "dbhdt", "dbhd", "dh_dt", "d2_d",
        "dH_dt".lower(),
    ]
    time_keys = ["hour_", "doy_"]

    groups = {
        "solar_coupling": [c for c in feature_cols if has_any(c, solar_keys)],
        "geomag_response": [c for c in feature_cols if has_any(c, geomag_keys)],
        "time_only": [c for c in feature_cols if has_any(c, time_keys)],
        "solar_plus_time": [
            c for c in feature_cols if has_any(c, solar_keys + time_keys)
        ],
        "geomag_plus_time": [
            c for c in feature_cols if has_any(c, geomag_keys + time_keys)
        ],
        "all_D": list(feature_cols),
    }
    return {k: list(dict.fromkeys(v)) for k, v in groups.items() if len(v) > 0}


def _sample_rows(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed).sort_index()


def _prepare_xy(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    max_train_rows: int,
    max_test_rows: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    cols = feature_cols + [target_col]
    train = _sample_rows(
        train_df[cols].replace([np.inf, -np.inf], np.nan).dropna(),
        max_train_rows,
        SEED,
    )
    test = _sample_rows(
        test_df[cols].replace([np.inf, -np.inf], np.nan).dropna(),
        max_test_rows,
        SEED + 1,
    )
    x_train = train[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y_train = train[target_col].to_numpy(dtype=np.float32, copy=True)
    x_test = test[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y_test = test[target_col].to_numpy(dtype=np.float32, copy=True)
    return x_train, y_train, x_test, y_test, pd.DatetimeIndex(test.index)


def _unique_feature_columns(feature_groups: Dict[str, List[str]]) -> List[str]:
    return list(dict.fromkeys(c for cols in feature_groups.values() for c in cols))


def _prepare_common_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_groups: Dict[str, List[str]],
    target_col: str,
    max_train_rows: int,
    max_test_rows: int,
    extra_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build one shared train/test row set for all feature groups.

    This keeps ablation metrics and prediction plots comparable. Without this,
    each group can drop a different set of NaN rows and then sample a different
    test subset, so plots may show different event times under the same split.
    """
    all_features = _unique_feature_columns(feature_groups)
    cols = all_features + [target_col]
    for col in extra_cols or []:
        if col in train_df.columns and col in test_df.columns and col not in cols:
            cols.append(col)
    train = _sample_rows(
        train_df[cols].replace([np.inf, -np.inf], np.nan).dropna(),
        max_train_rows,
        SEED,
    )
    full_test = test_df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    test = _sample_rows(full_test, max_test_rows, SEED + 1)
    return train, test, full_test


def _prepare_xy_from_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    x_train = train[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y_train = train[target_col].to_numpy(dtype=np.float32, copy=True)
    x_test = test[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y_test = test[target_col].to_numpy(dtype=np.float32, copy=True)
    return x_train, y_train, x_test, y_test, pd.DatetimeIndex(test.index)


def _classification_metrics(
    y_true_binary: np.ndarray,
    y_prob: np.ndarray,
    decision_threshold: float = 0.5,
) -> Dict[str, float]:
    y_pred = (y_prob >= float(decision_threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred, labels=[0, 1]).ravel()
    pod = tp / max(tp + fn, 1)
    pofd = fp / max(fp + tn, 1)
    far = fp / max(tp + fp, 1)
    bias = (tp + fp) / max(tp + fn, 1)
    hss_denom = 2 * ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    hss = 2 * (tp * tn - fp * fn) / max(hss_denom, 1)
    try:
        auc = roc_auc_score(y_true_binary, y_prob)
    except ValueError:
        auc = np.nan
    return {
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "AUC": float(auc),
        "POD": float(pod),
        "POFD": float(pofd),
        "FAR": float(far),
        "Bias": float(bias),
        "TSS": float(pod - pofd),
        "HSS": float(hss),
        "precision": float(precision_score(y_true_binary, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_binary, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true_binary, y_pred, zero_division=0)),
        "positive_rate_true": float(np.mean(y_true_binary)),
        "positive_rate_pred": float(np.mean(y_pred)),
    }


def _event_level_arrays(
    y_true: np.ndarray,
    y_score: np.ndarray,
    time_index: pd.DatetimeIndex,
    events_report: Optional[pd.DataFrame],
    threshold_a: float,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    if events_report is None or events_report.empty or len(y_true) == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=float), []
    frame = pd.DataFrame(
        {"y_true": y_true, "y_score": y_score},
        index=pd.DatetimeIndex(time_index),
    ).sort_index()
    truth, scores, rows = [], [], []
    for row in events_report.itertuples(index=False):
        if getattr(row, "split", "") != "test":
            continue
        start = pd.Timestamp(getattr(row, "start"))
        end = pd.Timestamp(getattr(row, "end"))
        sub = frame.loc[start:end]
        if sub.empty:
            continue
        true_peak = float(sub["y_true"].max())
        score_peak = float(sub["y_score"].max())
        event_truth = int(true_peak >= float(threshold_a))
        truth.append(event_truth)
        scores.append(score_peak)
        rows.append(
            {
                "event_id": getattr(row, "event_id", len(rows) + 1),
                "start": start,
                "end": end,
                "event_type": getattr(row, "event_type", getattr(row, "type", "")),
                "true_peak": true_peak,
                "score_peak": score_peak,
                "truth": event_truth,
            }
        )
    return np.asarray(truth, dtype=int), np.asarray(scores, dtype=float), rows


def _event_metrics_from_arrays(
    truth_arr: np.ndarray,
    score_arr: np.ndarray,
    decision_threshold: float,
) -> Dict[str, float]:
    if len(truth_arr) == 0:
        return {}
    pred_arr = (score_arr >= float(decision_threshold)).astype(int)
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
    try:
        auc = roc_auc_score(truth_arr, score_arr) if len(np.unique(truth_arr)) == 2 else np.nan
    except ValueError:
        auc = np.nan
    return {
        "event_n": int(len(truth_arr)),
        "event_TP": tp,
        "event_FP": fp,
        "event_TN": tn,
        "event_FN": fn,
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


def _select_low_far_event_threshold(
    truth_arr: np.ndarray,
    score_arr: np.ndarray,
    max_event_far: float,
) -> Tuple[float, Dict[str, float], str]:
    if len(truth_arr) == 0:
        return 0.5, {}, "empty"
    candidates = np.unique(np.clip(score_arr, 0.0, 1.0))
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], candidates)))
    best_thr = 0.5
    best_metrics = _event_metrics_from_arrays(truth_arr, score_arr, best_thr)
    best_key = None
    fallback_thr = best_thr
    fallback_metrics = best_metrics
    fallback_key = None
    for thr in candidates:
        metrics = _event_metrics_from_arrays(truth_arr, score_arr, float(thr))
        feasible = metrics.get("event_FAR", np.inf) <= float(max_event_far)
        key = (
            metrics.get("event_TSS", -np.inf),
            metrics.get("event_CSI", -np.inf),
            metrics.get("event_F1", -np.inf),
            -float(thr),
        )
        fallback = (
            -metrics.get("event_FAR", np.inf),
            metrics.get("event_POD", -np.inf),
            metrics.get("event_CSI", -np.inf),
            float(thr),
        )
        if feasible and (best_key is None or key > best_key):
            best_key = key
            best_thr = float(thr)
            best_metrics = metrics
        if fallback_key is None or fallback > fallback_key:
            fallback_key = fallback
            fallback_thr = float(thr)
            fallback_metrics = metrics
    if best_key is None:
        return fallback_thr, fallback_metrics, "fallback_min_far"
    return best_thr, best_metrics, "far_control"


def _split_fit_calibration(
    x_train: np.ndarray,
    y_train: np.ndarray,
    calibration_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ratio = float(np.clip(calibration_ratio, 0.0, 0.5))
    if ratio <= 0.0 or len(y_train) < 20:
        return x_train, y_train, x_train, y_train
    n_cal = max(1, int(round(len(y_train) * ratio)))
    n_fit = max(1, len(y_train) - n_cal)
    if n_fit < 2:
        return x_train, y_train, x_train, y_train
    return x_train[:n_fit], y_train[:n_fit], x_train[n_fit:], y_train[n_fit:]


def _strong_event_sample_weight(
    y: np.ndarray,
    strong_event_weight: float,
    extreme_event_weight: float,
) -> np.ndarray:
    weights = np.ones(len(y), dtype=np.float32)
    if strong_event_weight > 1.0:
        weights[y >= 10.0] = np.maximum(weights[y >= 10.0], float(strong_event_weight))
    if extreme_event_weight > 1.0:
        weights[y >= 20.0] = np.maximum(weights[y >= 20.0], float(extreme_event_weight))
    return weights


def _select_decision_threshold(
    y_true_binary: np.ndarray,
    y_prob: np.ndarray,
    strategy: str,
    max_far: float,
) -> Tuple[float, Dict[str, float]]:
    strategy = str(strategy).lower()
    if strategy == "fixed" or len(y_true_binary) == 0 or len(np.unique(y_true_binary)) < 2:
        thr = 0.5
        return thr, _classification_metrics(y_true_binary, y_prob, thr)

    candidates = np.unique(np.clip(y_prob, 0.0, 1.0))
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], candidates)))
    best_thr = 0.5
    best_metrics = _classification_metrics(y_true_binary, y_prob, best_thr)
    best_score = -np.inf
    fallback_score = -np.inf
    fallback_thr = best_thr
    fallback_metrics = best_metrics

    for thr in candidates:
        metrics = _classification_metrics(y_true_binary, y_prob, float(thr))
        if strategy == "max-f1":
            score = metrics["F1"]
        elif strategy == "far-control":
            score = metrics["TSS"] if metrics["FAR"] <= float(max_far) else -np.inf
            fallback = metrics["TSS"] - max(metrics["FAR"] - float(max_far), 0.0)
            if fallback > fallback_score:
                fallback_score = fallback
                fallback_thr = float(thr)
                fallback_metrics = metrics
        else:
            score = metrics["TSS"]

        if score > best_score:
            best_score = score
            best_thr = float(thr)
            best_metrics = metrics

    if strategy == "far-control" and not np.isfinite(best_score):
        return fallback_thr, fallback_metrics
    return best_thr, best_metrics


def _quantile_calibration_offset(
    y_cal: np.ndarray,
    pred_cal: np.ndarray,
    quantile: float,
    enabled: bool,
) -> float:
    if not enabled or len(y_cal) == 0:
        return 0.0
    residual = y_cal - pred_cal
    residual = residual[np.isfinite(residual)]
    if len(residual) == 0:
        return 0.0
    return float(np.quantile(residual, float(quantile)))


def _continuous_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "corr": np.nan}
    yt = y_true[mask]
    yp = y_pred[mask]
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - np.sum(err ** 2) / denom) if denom > 1e-12 else np.nan
    corr = float(np.corrcoef(yt, yp)[0, 1]) if len(yt) > 1 and np.std(yp) > 1e-12 else np.nan
    high = yt >= np.quantile(yt, 0.9)
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "corr": corr,
        "bias": float(np.mean(err)),
        "true_mean": float(np.mean(yt)),
        "pred_mean": float(np.mean(yp)),
        "true_q90": float(np.quantile(yt, 0.9)),
        "pred_q90": float(np.quantile(yp, 0.9)),
        "top10_MAE": float(np.mean(np.abs(err[high]))) if np.any(high) else np.nan,
        "top10_bias": float(np.mean(err[high])) if np.any(high) else np.nan,
    }


def _regression_threshold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: List[float],
) -> List[Dict[str, float]]:
    rows = []
    for thr in thresholds:
        y_bin = (y_true >= float(thr)).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue
        metrics = _classification_metrics(y_bin, y_pred, float(thr))
        rows.append({"threshold_A": float(thr), **metrics})
    return rows


def _plot_continuous_prediction(
    time_index: Optional[pd.DatetimeIndex],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_name: str,
    out_dir: str,
    max_points: int = 6000,
) -> None:
    plot_dir = os.path.join(out_dir, "continuous_regression_plots")
    os.makedirs(plot_dir, exist_ok=True)
    n = len(y_true)
    if n == 0:
        return
    if n <= max_points:
        idx = np.arange(n)
    else:
        peak = int(np.nanargmax(y_true))
        half = max_points // 2
        start = max(0, peak - half)
        end = min(n, start + max_points)
        start = max(0, end - max_points)
        idx = np.arange(start, end)
    x = np.arange(len(idx))
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(x, y_true[idx], color="black", linewidth=1.0, label="True future max")
    ax.plot(x, y_pred[idx], color="tab:red", linewidth=1.0, label="Predicted continuous")
    for thr in [3.0, 5.0, 10.0, 20.0]:
        ax.axhline(thr, linestyle="--", linewidth=0.8, alpha=0.5, label=f"{thr:g} A")
    ax.set_title(f"{group_name}: continuous Huber regression")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("GIC future max (A)")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=4, fontsize=9)
    fig.tight_layout()
    path = os.path.join(plot_dir, f"{group_name}_continuous_prediction.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    cache = pd.DataFrame({
        "time": time_index.astype(str) if time_index is not None and len(time_index) == n else np.arange(n),
        "y_true": y_true,
        "y_pred": y_pred,
    })
    cache.to_csv(os.path.join(plot_dir, f"{group_name}_continuous_prediction.csv"), index=False, encoding="utf-8-sig")


def _require_lightgbm():
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except ImportError as exc:
        raise ImportError(
            "LightGBM is not installed. Install it first, e.g. `pip install lightgbm`, "
            "or run with `--model-backend hgb`."
        ) from exc
    return LGBMClassifier, LGBMRegressor


def _normalize_backend(model_backend: str) -> str:
    name = str(model_backend).lower().replace("-", "_")
    aliases = {
        "model3": "cnn_bilstm_attention",
        "m3": "cnn_bilstm_attention",
        "model5": "cnn_bilstm_attention_gatefusion",
        "m5": "cnn_bilstm_attention_gatefusion",
        "gatefusion": "cnn_bilstm_attention_gatefusion",
        "cnn_bilstm_attention_gate": "cnn_bilstm_attention_gatefusion",
    }
    return aliases.get(name, name)


class _LaggedDeepNet(nn.Module):
    def __init__(self, n_features: int, backend: str, output_size: int):
        super().__init__()
        self.backend = _normalize_backend(backend)
        self.use_cnn = self.backend in {"cnn_bilstm", "cnn_bilstm_attention"}
        self.use_attention = self.backend in {"cnn_bilstm_attention", "cnn_bilstm_attention_gatefusion"}
        self.use_gatefusion = self.backend == "cnn_bilstm_attention_gatefusion"
        self.use_cnn = self.use_cnn or self.use_gatefusion
        self.input_proj = nn.Linear(1, DEEP_HIDDEN_SIZE)
        if self.use_cnn:
            self.conv = nn.Sequential(
                nn.Conv1d(DEEP_HIDDEN_SIZE, DEEP_HIDDEN_SIZE, kernel_size=3, padding=1),
                nn.BatchNorm1d(DEEP_HIDDEN_SIZE),
                nn.GELU(),
                nn.Dropout(DEEP_DROPOUT),
                nn.Conv1d(DEEP_HIDDEN_SIZE, DEEP_HIDDEN_SIZE, kernel_size=5, padding=2),
                nn.BatchNorm1d(DEEP_HIDDEN_SIZE),
                nn.GELU(),
            )
        self.lstm = nn.LSTM(
            input_size=DEEP_HIDDEN_SIZE,
            hidden_size=DEEP_HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        rep_size = DEEP_HIDDEN_SIZE * 2
        if self.use_attention:
            self.attn_score = nn.Sequential(
                nn.Linear(rep_size, DEEP_HIDDEN_SIZE),
                nn.Tanh(),
                nn.Linear(DEEP_HIDDEN_SIZE, 1),
            )
        if self.use_gatefusion:
            self.gate = nn.Sequential(
                nn.LayerNorm(rep_size * 2),
                nn.Linear(rep_size * 2, rep_size),
                nn.Sigmoid(),
            )
        self.head = nn.Sequential(
            nn.LayerNorm(rep_size),
            nn.Linear(rep_size, DEEP_HIDDEN_SIZE),
            nn.GELU(),
            nn.Dropout(DEEP_DROPOUT),
            nn.Linear(DEEP_HIDDEN_SIZE, output_size),
        )

    def forward(self, x):
        x = self.input_proj(x.unsqueeze(-1))
        if self.use_cnn:
            residual = x
            x = self.conv(x.transpose(1, 2)).transpose(1, 2) + residual
        seq_out, _ = self.lstm(x)
        if self.use_attention:
            weights = torch.softmax(self.attn_score(seq_out).squeeze(-1), dim=1)
            attn_pooled = torch.bmm(weights.unsqueeze(1), seq_out).squeeze(1)
            if self.use_gatefusion:
                last_step = seq_out[:, -1, :]
                gate = self.gate(torch.cat([last_step, attn_pooled], dim=1))
                pooled = gate * attn_pooled + (1.0 - gate) * last_step
            else:
                pooled = attn_pooled
        else:
            pooled = seq_out[:, -1, :]
        return self.head(pooled)


class _DeepBase:
    def __init__(self, model_backend: str, output_size: int, task: str):
        self.model_backend = _normalize_backend(model_backend)
        self.output_size = int(output_size)
        self.task = task
        self.model: Optional[_LaggedDeepNet] = None
        self.x_mean: Optional[np.ndarray] = None
        self.x_std: Optional[np.ndarray] = None
        self.y_scale: float = 1.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history_: List[Dict[str, float]] = []

    def save_checkpoint(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save an unfitted deep model.")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "model_backend": self.model_backend,
                "output_size": self.output_size,
                "task": self.task,
                "state_dict": self.model.state_dict(),
                "x_mean": self.x_mean,
                "x_std": self.x_std,
                "y_scale": self.y_scale,
                "history": self.history_,
                "extra": self._checkpoint_extra(),
            },
            path,
        )

    def load_checkpoint(self, path: str) -> "_DeepBase":
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)
        self.model_backend = str(ckpt.get("model_backend", self.model_backend))
        self.output_size = int(ckpt.get("output_size", self.output_size))
        self.task = str(ckpt.get("task", self.task))
        self.x_mean = ckpt["x_mean"]
        self.x_std = ckpt["x_std"]
        self.y_scale = float(ckpt.get("y_scale", 1.0))
        self.history_ = list(ckpt.get("history", []))
        self._load_checkpoint_extra(dict(ckpt.get("extra", {})))
        self.model = _LaggedDeepNet(len(self.x_mean), self.model_backend, self.output_size).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        return self

    def _checkpoint_extra(self) -> Dict[str, object]:
        return {}

    def _load_checkpoint_extra(self, extra: Dict[str, object]) -> None:
        return None

    def _restore_best_state(self, best_state: Optional[Dict[str, torch.Tensor]]) -> None:
        if best_state is not None and self.model is not None:
            self.model.load_state_dict(best_state)

    @staticmethod
    def _best_loss_key(row: Dict[str, float]) -> float:
        for key in ("cal_loss", "test_loss", "train_loss"):
            val = row.get(key, np.nan)
            if np.isfinite(val):
                return float(val)
        return float("inf")

    def _standardize_fit(self, x: np.ndarray) -> np.ndarray:
        self.x_mean = np.nanmean(x, axis=0).astype(np.float32)
        self.x_std = np.nanstd(x, axis=0).astype(np.float32)
        self.x_std[self.x_std < 1e-6] = 1.0
        return self._standardize(x)

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        if self.x_mean is None or self.x_std is None:
            raise RuntimeError("Deep model has not been fitted.")
        z = (x.astype(np.float32) - self.x_mean) / self.x_std
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _iter_batches(self, x: np.ndarray, y: np.ndarray, weight: Optional[np.ndarray] = None, shuffle: bool = True):
        order = np.arange(len(x))
        if shuffle:
            rng = np.random.default_rng(SEED)
            rng.shuffle(order)
        for start in range(0, len(order), DEEP_BATCH_SIZE):
            idx = order[start:start + DEEP_BATCH_SIZE]
            xb = torch.from_numpy(x[idx]).to(self.device)
            yb = torch.from_numpy(y[idx]).to(self.device)
            wb = None if weight is None else torch.from_numpy(weight[idx].astype(np.float32)).to(self.device)
            yield xb, yb, wb


def _as_eval_weight(weight: Optional[np.ndarray], n: int) -> Optional[np.ndarray]:
    if weight is None:
        return None
    arr = weight.astype(np.float32).reshape(-1, 1)
    return arr if len(arr) == n else None


def _save_deep_history(
    model: _DeepBase,
    out_dir: Optional[str],
    group_name: str,
    task_name: str,
) -> None:
    if out_dir is None or not getattr(model, "history_", None):
        return
    hist_dir = os.path.join(out_dir, "training_history", _safe_name(group_name))
    os.makedirs(hist_dir, exist_ok=True)
    hist = pd.DataFrame(model.history_)
    csv_path = os.path.join(hist_dir, f"{_safe_name(task_name)}_loss_history.csv")
    hist.to_csv(csv_path, index=False, encoding="utf-8-sig")

    loss_cols = [c for c in hist.columns if c.endswith("_loss")]
    if loss_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        for col in loss_cols:
            ax.plot(hist["epoch"], hist[col], marker="o", linewidth=1.2, label=col)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{group_name} | {task_name}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig_path = os.path.join(hist_dir, f"{_safe_name(task_name)}_loss_history.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Ablation] Saved training history: {csv_path}")


class DeepClassifier(_DeepBase):
    def __init__(self, model_backend: str):
        super().__init__(model_backend=model_backend, output_size=1, task="classification")

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        eval_sets: Optional[Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]] = None,
    ):
        torch.manual_seed(SEED)
        x_train = self._standardize_fit(x)
        y_train = y.astype(np.float32).reshape(-1, 1)
        weights = None if sample_weight is None else sample_weight.astype(np.float32).reshape(-1, 1)
        pos_rate = float(np.mean(y_train))
        pos_weight = torch.tensor([(1.0 - pos_rate) / max(pos_rate, 1e-3)], device=self.device)
        self.model = _LaggedDeepNet(x_train.shape[1], self.model_backend, 1).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=DEEP_LR, weight_decay=1e-4)
        eval_arrays: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = {}
        for name, pack in (eval_sets or {}).items():
            ex, ey, ew = pack
            eval_arrays[str(name)] = (
                self._standardize(ex),
                ey.astype(np.float32).reshape(-1, 1),
                _as_eval_weight(ew, len(ey)),
            )
        self.history_ = []
        best_loss = float("inf")
        best_state = None
        for epoch in range(1, DEEP_EPOCHS + 1):
            self.model.train()
            train_losses = []
            for xb, yb, wb in self._iter_batches(x_train, y_train, weights, shuffle=True):
                logits = self.model(xb)
                loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight, reduction="none")
                if wb is not None:
                    loss = loss * wb
                loss = loss.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu().item()))
            row = {"epoch": float(epoch), "train_loss": float(np.mean(train_losses)) if train_losses else np.nan}
            self.model.eval()
            with torch.no_grad():
                for name, (ex, ey, ew) in eval_arrays.items():
                    losses = []
                    for xb, yb, wb in self._iter_batches(ex, ey, ew, shuffle=False):
                        logits = self.model(xb)
                        loss = F.binary_cross_entropy_with_logits(
                            logits, yb, pos_weight=pos_weight, reduction="none"
                        )
                        if wb is not None:
                            loss = loss * wb
                        losses.append(float(loss.mean().detach().cpu().item()))
                    row[f"{name}_loss"] = float(np.mean(losses)) if losses else np.nan
            self.history_.append(row)
            score = self._best_loss_key(row)
            if score < best_loss:
                best_loss = score
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        self._restore_best_state(best_state)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Deep classifier has not been fitted.")
        x_test = self._standardize(x)
        probs = []
        self.model.eval()
        with torch.no_grad():
            dummy_y = np.zeros((len(x_test), 1), dtype=np.float32)
            for xb, _, _ in self._iter_batches(x_test, dummy_y, shuffle=False):
                probs.append(torch.sigmoid(self.model(xb)).cpu().numpy().reshape(-1))
        p1 = np.concatenate(probs) if probs else np.array([], dtype=np.float32)
        return np.column_stack([1.0 - p1, p1])


class DeepQuantileRegressor(_DeepBase):
    def __init__(self, model_backend: str, quantile: float):
        super().__init__(model_backend=model_backend, output_size=1, task="quantile")
        self.quantile = float(quantile)

    def _checkpoint_extra(self) -> Dict[str, object]:
        return {"quantile": self.quantile}

    def _load_checkpoint_extra(self, extra: Dict[str, object]) -> None:
        if "quantile" in extra:
            self.quantile = float(extra["quantile"])

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        eval_sets: Optional[Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]] = None,
    ):
        torch.manual_seed(SEED)
        x_train = self._standardize_fit(x)
        y_raw = y.astype(np.float32).reshape(-1, 1)
        self.y_scale = float(np.nanpercentile(np.abs(y_raw), 95))
        if not np.isfinite(self.y_scale) or self.y_scale < 1.0:
            self.y_scale = 1.0
        y_train = y_raw / self.y_scale
        weights = None if sample_weight is None else sample_weight.astype(np.float32).reshape(-1, 1)
        self.model = _LaggedDeepNet(x_train.shape[1], self.model_backend, 1).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=DEEP_LR, weight_decay=1e-4)
        q = torch.tensor(self.quantile, device=self.device)
        eval_arrays: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = {}
        for name, pack in (eval_sets or {}).items():
            ex, ey, ew = pack
            eval_arrays[str(name)] = (
                self._standardize(ex),
                ey.astype(np.float32).reshape(-1, 1) / self.y_scale,
                _as_eval_weight(ew, len(ey)),
            )
        self.history_ = []
        for epoch in range(1, DEEP_EPOCHS + 1):
            self.model.train()
            train_losses = []
            for xb, yb, wb in self._iter_batches(x_train, y_train, weights, shuffle=True):
                pred = F.softplus(self.model(xb))
                err = yb - pred
                loss = torch.maximum(q * err, (q - 1.0) * err)
                if wb is not None:
                    loss = loss * wb
                loss = loss.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu().item()))
            row = {"epoch": float(epoch), "train_loss": float(np.mean(train_losses)) if train_losses else np.nan}
            self.model.eval()
            with torch.no_grad():
                for name, (ex, ey, ew) in eval_arrays.items():
                    losses = []
                    for xb, yb, wb in self._iter_batches(ex, ey, ew, shuffle=False):
                        pred = F.softplus(self.model(xb))
                        err = yb - pred
                        loss = torch.maximum(q * err, (q - 1.0) * err)
                        if wb is not None:
                            loss = loss * wb
                        losses.append(float(loss.mean().detach().cpu().item()))
                    row[f"{name}_loss"] = float(np.mean(losses)) if losses else np.nan
            self.history_.append(row)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Deep quantile regressor has not been fitted.")
        x_test = self._standardize(x)
        preds = []
        self.model.eval()
        with torch.no_grad():
            dummy_y = np.zeros((len(x_test), 1), dtype=np.float32)
            for xb, _, _ in self._iter_batches(x_test, dummy_y, shuffle=False):
                preds.append(F.softplus(self.model(xb)).cpu().numpy().reshape(-1))
        pred = np.concatenate(preds) if preds else np.array([], dtype=np.float32)
        return pred * self.y_scale


class DeepMultiQuantileRegressor(_DeepBase):
    def __init__(self, model_backend: str, quantiles: List[float]):
        super().__init__(model_backend=model_backend, output_size=len(quantiles), task="multi_quantile")
        self.quantiles = [float(q) for q in quantiles]

    def _checkpoint_extra(self) -> Dict[str, object]:
        return {"quantiles": self.quantiles}

    def _load_checkpoint_extra(self, extra: Dict[str, object]) -> None:
        if "quantiles" in extra:
            self.quantiles = [float(q) for q in extra["quantiles"]]
            self.output_size = len(self.quantiles)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        eval_sets: Optional[Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]] = None,
    ):
        torch.manual_seed(SEED)
        x_train = self._standardize_fit(x)
        y_raw = y.astype(np.float32).reshape(-1, 1)
        self.y_scale = float(np.nanpercentile(np.abs(y_raw), 95))
        if not np.isfinite(self.y_scale) or self.y_scale < 1.0:
            self.y_scale = 1.0
        y_train = y_raw / self.y_scale
        weights = None if sample_weight is None else sample_weight.astype(np.float32).reshape(-1, 1)
        self.model = _LaggedDeepNet(x_train.shape[1], self.model_backend, len(self.quantiles)).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=DEEP_LR, weight_decay=1e-4)
        q = torch.tensor(self.quantiles, dtype=torch.float32, device=self.device).view(1, -1)
        eval_arrays: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = {}
        for name, pack in (eval_sets or {}).items():
            ex, ey, ew = pack
            eval_arrays[str(name)] = (
                self._standardize(ex),
                ey.astype(np.float32).reshape(-1, 1) / self.y_scale,
                _as_eval_weight(ew, len(ey)),
            )
        self.history_ = []
        best_loss = float("inf")
        best_state = None
        for epoch in range(1, DEEP_EPOCHS + 1):
            self.model.train()
            train_losses = []
            for xb, yb, wb in self._iter_batches(x_train, y_train, weights, shuffle=True):
                raw = self.model(xb)
                first = F.softplus(raw[:, :1])
                if raw.shape[1] > 1:
                    increments = F.softplus(raw[:, 1:])
                    pred = torch.cat([first, first + torch.cumsum(increments, dim=1)], dim=1)
                else:
                    pred = first
                err = yb - pred
                loss = torch.maximum(q * err, (q - 1.0) * err)
                if wb is not None:
                    loss = loss * wb
                loss = loss.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu().item()))
            row = {"epoch": float(epoch), "train_loss": float(np.mean(train_losses)) if train_losses else np.nan}
            self.model.eval()
            with torch.no_grad():
                for name, (ex, ey, ew) in eval_arrays.items():
                    losses = []
                    for xb, yb, wb in self._iter_batches(ex, ey, ew, shuffle=False):
                        raw = self.model(xb)
                        first = F.softplus(raw[:, :1])
                        if raw.shape[1] > 1:
                            increments = F.softplus(raw[:, 1:])
                            pred = torch.cat([first, first + torch.cumsum(increments, dim=1)], dim=1)
                        else:
                            pred = first
                        err = yb - pred
                        loss = torch.maximum(q * err, (q - 1.0) * err)
                        if wb is not None:
                            loss = loss * wb
                        losses.append(float(loss.mean().detach().cpu().item()))
                    row[f"{name}_loss"] = float(np.mean(losses)) if losses else np.nan
            self.history_.append(row)
            score = self._best_loss_key(row)
            if score < best_loss:
                best_loss = score
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        self._restore_best_state(best_state)
        return self

    def predict_all(self, x: np.ndarray) -> Dict[float, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Deep multi-quantile regressor has not been fitted.")
        x_test = self._standardize(x)
        preds = []
        self.model.eval()
        with torch.no_grad():
            dummy_y = np.zeros((len(x_test), 1), dtype=np.float32)
            for xb, _, _ in self._iter_batches(x_test, dummy_y, shuffle=False):
                raw = self.model(xb)
                first = F.softplus(raw[:, :1])
                if raw.shape[1] > 1:
                    increments = F.softplus(raw[:, 1:])
                    pred = torch.cat([first, first + torch.cumsum(increments, dim=1)], dim=1)
                else:
                    pred = first
                preds.append(pred.cpu().numpy())
        arr = np.vstack(preds) * self.y_scale if preds else np.zeros((0, len(self.quantiles)), dtype=np.float32)
        return {q: arr[:, i] for i, q in enumerate(self.quantiles)}


class DeepHuberRegressor(_DeepBase):
    def __init__(self, model_backend: str, huber_delta: float = 1.0):
        super().__init__(model_backend=model_backend, output_size=1, task="continuous_regression")
        self.huber_delta = float(huber_delta)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        eval_sets: Optional[Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]] = None,
    ):
        torch.manual_seed(SEED)
        x_train = self._standardize_fit(x)
        y_raw = y.astype(np.float32).reshape(-1, 1)
        self.y_scale = float(np.nanpercentile(np.abs(y_raw), 95))
        if not np.isfinite(self.y_scale) or self.y_scale < 1.0:
            self.y_scale = 1.0
        y_train = y_raw / self.y_scale
        weights = None if sample_weight is None else sample_weight.astype(np.float32).reshape(-1, 1)
        self.model = _LaggedDeepNet(x_train.shape[1], self.model_backend, 1).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=DEEP_LR, weight_decay=1e-4)

        eval_arrays: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = {}
        for name, pack in (eval_sets or {}).items():
            ex, ey, ew = pack
            eval_arrays[str(name)] = (
                self._standardize(ex),
                ey.astype(np.float32).reshape(-1, 1) / self.y_scale,
                _as_eval_weight(ew, len(ey)),
            )

        self.history_ = []
        for epoch in range(1, DEEP_EPOCHS + 1):
            self.model.train()
            train_losses = []
            for xb, yb, wb in self._iter_batches(x_train, y_train, weights, shuffle=True):
                pred = F.softplus(self.model(xb))
                loss = F.huber_loss(pred, yb, delta=self.huber_delta, reduction="none")
                if wb is not None:
                    loss = loss * wb
                loss = loss.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu().item()))

            row = {"epoch": float(epoch), "train_loss": float(np.mean(train_losses)) if train_losses else np.nan}
            self.model.eval()
            with torch.no_grad():
                for name, (ex, ey, ew) in eval_arrays.items():
                    losses = []
                    for xb, yb, wb in self._iter_batches(ex, ey, ew, shuffle=False):
                        pred = F.softplus(self.model(xb))
                        loss = F.huber_loss(pred, yb, delta=self.huber_delta, reduction="none")
                        if wb is not None:
                            loss = loss * wb
                        losses.append(float(loss.mean().detach().cpu().item()))
                    row[f"{name}_loss"] = float(np.mean(losses)) if losses else np.nan
            self.history_.append(row)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Deep Huber regressor has not been fitted.")
        x_test = self._standardize(x)
        preds = []
        self.model.eval()
        with torch.no_grad():
            dummy_y = np.zeros((len(x_test), 1), dtype=np.float32)
            for xb, _, _ in self._iter_batches(x_test, dummy_y, shuffle=False):
                preds.append(F.softplus(self.model(xb)).cpu().numpy().reshape(-1))
        pred = np.concatenate(preds) if preds else np.array([], dtype=np.float32)
        return pred * self.y_scale


def _build_classifier(model_backend: str):
    backend = _normalize_backend(model_backend)
    if backend in DEEP_BACKENDS:
        return DeepClassifier(backend)
    if backend == "lightgbm":
        LGBMClassifier, _ = _require_lightgbm()
        return LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=0.1,
            objective="binary",
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )
    if backend != "hgb":
        raise ValueError("model_backend must be one of: hgb, lightgbm, bilstm, cnn_bilstm, cnn_bilstm_attention, model3, model5")
    return HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=SEED,
    )


def _build_quantile_regressor(model_backend: str, quantile: float):
    backend = _normalize_backend(model_backend)
    if backend in DEEP_BACKENDS:
        return DeepQuantileRegressor(backend, float(quantile))
    if backend == "lightgbm":
        _, LGBMRegressor = _require_lightgbm()
        return LGBMRegressor(
            objective="quantile",
            alpha=float(quantile),
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=0.1,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )
    if backend != "hgb":
        raise ValueError("model_backend must be one of: hgb, lightgbm, bilstm, cnn_bilstm, cnn_bilstm_attention, model3, model5")
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=float(quantile),
        max_iter=220,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=SEED,
    )


def run_ablation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_groups: Dict[str, List[str]],
    target_col: str,
    thresholds: List[float],
    quantiles: List[float],
    max_train_rows: int,
    max_test_rows: int,
    out_dir: Optional[str] = None,
    plot_groups: Optional[List[str]] = None,
    common_eval_index: bool = True,
    calibration_ratio: float = 0.2,
    threshold_strategy: str = "max-tss",
    max_far: float = 0.4,
    strong_event_weight: float = 3.0,
    extreme_event_weight: float = 8.0,
    quantile_calibration: bool = True,
    model_backend: str = "hgb",
    events_report: Optional[pd.DataFrame] = None,
    eval_only: bool = False,
    raw_target_col: Optional[str] = None,
    final_quantile: float = 0.5,
    low_far_event_max_far: float = 0.2,
    save_prediction_outputs: bool = True,
    skip_prediction_plots: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cls_rows = []
    q_rows = []
    low_far_event_rows = []
    low_far_event_detail_rows = []

    common_train: Optional[pd.DataFrame] = None
    common_test: Optional[pd.DataFrame] = None
    common_full_test: Optional[pd.DataFrame] = None
    if common_eval_index:
        common_train, common_test, common_full_test = _prepare_common_frames(
            train_df=train_df,
            test_df=test_df,
            feature_groups=feature_groups,
            target_col=target_col,
            max_train_rows=max_train_rows,
            max_test_rows=max_test_rows,
            extra_cols=[raw_target_col] if raw_target_col else None,
        )
        print(
            f"[Ablation] common_eval_index=True, "
            f"train_rows={len(common_train):,}, eval_test_rows={len(common_test):,}, "
            f"plot_test_rows={len(common_full_test):,}"
        )

    for group_name, cols in feature_groups.items():
        print(f"[Ablation] group={group_name}, n_features={len(cols)}")
        full_test_index: Optional[pd.DatetimeIndex] = None
        x_full_test: Optional[np.ndarray] = None
        y_full_test: Optional[np.ndarray] = None
        y_raw_full_test: Optional[np.ndarray] = None
        if common_train is not None and common_test is not None:
            x_train, y_train, x_test, y_test, test_index = _prepare_xy_from_frames(
                common_train, common_test, cols, target_col
            )
            if common_full_test is not None:
                x_full_test = common_full_test[cols].to_numpy(dtype=np.float32, copy=True)
                y_full_test = common_full_test[target_col].to_numpy(dtype=np.float32, copy=True)
                if raw_target_col and raw_target_col in common_full_test.columns:
                    y_raw_full_test = common_full_test[raw_target_col].to_numpy(dtype=np.float32, copy=True)
                full_test_index = pd.DatetimeIndex(common_full_test.index)
        else:
            x_train, y_train, x_test, y_test, test_index = _prepare_xy(
                train_df, test_df, cols, target_col, max_train_rows, max_test_rows
            )
            full_cols = cols + [target_col]
            if raw_target_col and raw_target_col in test_df.columns and raw_target_col not in full_cols:
                full_cols.append(raw_target_col)
            full = test_df[full_cols].replace([np.inf, -np.inf], np.nan).dropna(subset=cols + [target_col])
            x_full_test = full[cols].to_numpy(dtype=np.float32, copy=True)
            y_full_test = full[target_col].to_numpy(dtype=np.float32, copy=True)
            if raw_target_col and raw_target_col in full.columns:
                y_raw_full_test = full[raw_target_col].to_numpy(dtype=np.float32, copy=True)
            full_test_index = pd.DatetimeIndex(full.index)
        if len(y_train) == 0 or len(y_test) == 0:
            continue

        x_fit, y_fit, x_cal, y_cal = _split_fit_calibration(
            x_train, y_train, calibration_ratio
        )
        fit_weights = _strong_event_sample_weight(
            y_fit,
            strong_event_weight=strong_event_weight,
            extreme_event_weight=extreme_event_weight,
        )
        eval_sets_reg = {
            "cal": (x_cal, y_cal, None),
            "test": (x_test, y_test, None),
        }

        threshold_probs: Dict[float, np.ndarray] = {}
        full_threshold_probs: Dict[float, np.ndarray] = {}
        for thr in thresholds:
            y_bin_train = (y_fit >= thr).astype(int)
            y_bin_cal = (y_cal >= thr).astype(int)
            y_bin_test = (y_test >= thr).astype(int)
            if len(np.unique(y_bin_train)) < 2:
                continue
            clf = _build_classifier(model_backend)
            ckpt_path = None
            if isinstance(clf, _DeepBase):
                ckpt_path = os.path.join(
                    out_dir or ".",
                    "checkpoints",
                    _safe_name(group_name),
                    f"classification_thr{float(thr):g}A.pt",
                )
            if eval_only and isinstance(clf, _DeepBase):
                if not os.path.exists(ckpt_path):
                    raise FileNotFoundError(f"Missing checkpoint for eval-only: {ckpt_path}")
                print(f"[Ablation] Loading checkpoint: {ckpt_path}", flush=True)
                clf.load_checkpoint(ckpt_path)
            else:
                fit_kwargs = {}
                if isinstance(clf, _DeepBase):
                    fit_kwargs["eval_sets"] = {
                        "cal": (x_cal, y_bin_cal, None),
                        "test": (x_test, y_bin_test, None),
                    }
                clf.fit(x_fit, y_bin_train, sample_weight=fit_weights, **fit_kwargs)
                if isinstance(clf, _DeepBase):
                    clf.save_checkpoint(ckpt_path)
                    print(f"[Ablation] Saved checkpoint: {ckpt_path}")
                    _save_deep_history(
                        clf,
                        out_dir,
                        group_name,
                        f"classification_thr{float(thr):g}A",
                    )
            print(
                f"[Ablation] Predict classification group={group_name}, "
                f"thr={float(thr):g}A, cal_rows={len(x_cal):,}, test_rows={len(x_test):,}, "
                f"full_rows={(len(x_full_test) if x_full_test is not None else 0):,}",
                flush=True,
            )
            cal_prob = clf.predict_proba(x_cal)[:, 1]
            selected_thr, cal_metrics = _select_decision_threshold(
                y_bin_cal,
                cal_prob,
                strategy=threshold_strategy,
                max_far=max_far,
            )
            prob = clf.predict_proba(x_test)[:, 1]
            threshold_probs[float(thr)] = prob
            full_prob = None
            if x_full_test is not None:
                if (
                    full_test_index is not None
                    and len(x_full_test) == len(x_test)
                    and len(full_test_index) == len(test_index)
                    and pd.DatetimeIndex(full_test_index).equals(pd.DatetimeIndex(test_index))
                ):
                    full_prob = prob
                else:
                    full_prob = clf.predict_proba(x_full_test)[:, 1]
                full_threshold_probs[float(thr)] = full_prob
            print(
                f"[Ablation] Done classification group={group_name}, thr={float(thr):g}A",
                flush=True,
            )
            metrics = _classification_metrics(y_bin_test, prob, selected_thr)
            event_metrics = event_level_classification_metrics(
                y_true=y_full_test if y_full_test is not None else y_test,
                y_score=full_prob if full_prob is not None else prob,
                time_index=full_test_index if full_test_index is not None else test_index,
                events_report=events_report,
                threshold_a=float(thr),
                decision_threshold=float(selected_thr),
            )
            event_truth, event_scores, event_detail = _event_level_arrays(
                y_true=y_full_test if y_full_test is not None else y_test,
                y_score=full_prob if full_prob is not None else prob,
                time_index=full_test_index if full_test_index is not None else test_index,
                events_report=events_report,
                threshold_a=float(thr),
            )
            low_far_thr, low_far_metrics, low_far_status = _select_low_far_event_threshold(
                event_truth,
                event_scores,
                max_event_far=low_far_event_max_far,
            )
            if low_far_metrics:
                low_far_event_rows.append(
                    {
                        "feature_group": group_name,
                        "n_features": len(cols),
                        "threshold_A": float(thr),
                        "model_backend": model_backend,
                        "event_decision_threshold": float(low_far_thr),
                        "event_threshold_strategy": "low-far",
                        "max_event_FAR": float(low_far_event_max_far),
                        "selection_status": low_far_status,
                        **low_far_metrics,
                    }
                )
                for detail in event_detail:
                    pred = int(float(detail["score_peak"]) >= float(low_far_thr))
                    truth_value = int(detail["truth"])
                    if truth_value == 1 and pred == 1:
                        outcome = "TP"
                    elif truth_value == 0 and pred == 1:
                        outcome = "FP"
                    elif truth_value == 1 and pred == 0:
                        outcome = "FN"
                    else:
                        outcome = "TN"
                    low_far_event_detail_rows.append(
                        {
                            "feature_group": group_name,
                            "threshold_A": float(thr),
                            "model_backend": model_backend,
                            "event_decision_threshold": float(low_far_thr),
                            "max_event_FAR": float(low_far_event_max_far),
                            **detail,
                            "pred": pred,
                            "outcome": outcome,
                        }
                    )
            cls_rows.append(
                {
                    "feature_group": group_name,
                    "n_features": len(cols),
                    "threshold_A": float(thr),
                    "model_backend": model_backend,
                    "decision_threshold": float(selected_thr),
                    "threshold_strategy": threshold_strategy,
                    "calibration_ratio": float(calibration_ratio),
                    "cal_TSS": float(cal_metrics.get("TSS", np.nan)),
                    "cal_F1": float(cal_metrics.get("F1", np.nan)),
                    "cal_FAR": float(cal_metrics.get("FAR", np.nan)),
                    "n_train": int(len(y_train)),
                    "n_fit": int(len(y_fit)),
                    "n_cal": int(len(y_cal)),
                    "n_test": int(len(y_test)),
                    **metrics,
                    **event_metrics,
                }
            )

        quantile_preds: Dict[float, np.ndarray] = {}
        full_quantile_preds: Dict[float, np.ndarray] = {}
        backend = _normalize_backend(model_backend)
        if backend in DEEP_BACKENDS:
            reg_multi = DeepMultiQuantileRegressor(backend, [float(q) for q in quantiles])
            ckpt_path = os.path.join(
                out_dir or ".",
                "checkpoints",
                _safe_name(group_name),
                "multi_quantile.pt",
            )
            if eval_only:
                if not os.path.exists(ckpt_path):
                    raise FileNotFoundError(f"Missing checkpoint for eval-only: {ckpt_path}")
                print(f"[Ablation] Loading checkpoint: {ckpt_path}", flush=True)
                reg_multi.load_checkpoint(ckpt_path)
            else:
                reg_multi.fit(x_fit, y_fit, sample_weight=fit_weights, eval_sets=eval_sets_reg)
                reg_multi.save_checkpoint(ckpt_path)
                print(f"[Ablation] Saved checkpoint: {ckpt_path}")
                _save_deep_history(reg_multi, out_dir, group_name, "multi_quantile")
            print(
                f"[Ablation] Predict quantiles group={group_name}, "
                f"cal_rows={len(x_cal):,}, test_rows={len(x_test):,}, "
                f"full_rows={(len(x_full_test) if x_full_test is not None else 0):,}",
                flush=True,
            )
            cal_preds = reg_multi.predict_all(x_cal)
            test_preds = reg_multi.predict_all(x_test)
            if x_full_test is not None:
                if (
                    full_test_index is not None
                    and len(x_full_test) == len(x_test)
                    and len(full_test_index) == len(test_index)
                    and pd.DatetimeIndex(full_test_index).equals(pd.DatetimeIndex(test_index))
                ):
                    full_preds = test_preds
                else:
                    full_preds = reg_multi.predict_all(x_full_test)
            else:
                full_preds = {}
            print(f"[Ablation] Done quantiles group={group_name}", flush=True)
        else:
            cal_preds = {}
            test_preds = {}
            full_preds = {}

        for q in quantiles:
            if backend in DEEP_BACKENDS:
                cal_pred = cal_preds[float(q)]
                raw_pred = test_preds[float(q)]
                raw_full_pred = full_preds.get(float(q))
            else:
                reg = _build_quantile_regressor(model_backend, float(q))
                fit_kwargs = {"eval_sets": eval_sets_reg} if isinstance(reg, _DeepBase) else {}
                reg.fit(x_fit, y_fit, sample_weight=fit_weights, **fit_kwargs)
                if isinstance(reg, _DeepBase):
                    _save_deep_history(reg, out_dir, group_name, f"quantile_Q{int(round(float(q) * 100))}")
                cal_pred = reg.predict(x_cal)
                raw_pred = reg.predict(x_test)
                raw_full_pred = reg.predict(x_full_test) if x_full_test is not None else None
            offset = _quantile_calibration_offset(
                y_cal,
                cal_pred,
                quantile=float(q),
                enabled=quantile_calibration,
            )
            pred = raw_pred + offset
            quantile_preds[float(q)] = pred
            if raw_full_pred is not None:
                full_quantile_preds[float(q)] = raw_full_pred + offset
            pinball = np.mean(np.maximum(q * (y_test - pred), (q - 1) * (y_test - pred)))
            coverage = np.mean(y_test <= pred)
            exceed_mask = y_test > pred
            mae_upper = np.mean(np.abs(y_test - pred))
            high_mask = y_test >= np.quantile(y_test, 0.9)
            high_exceed = exceed_mask & high_mask
            q_rows.append(
                {
                    "feature_group": group_name,
                    "n_features": len(cols),
                    "quantile": float(q),
                    "model_backend": model_backend,
                    "n_train": int(len(y_train)),
                    "n_fit": int(len(y_fit)),
                    "n_cal": int(len(y_cal)),
                    "n_test": int(len(y_test)),
                    "calibration_offset": float(offset),
                    "quantile_calibration": bool(quantile_calibration),
                    "pinball_loss": float(pinball),
                    "coverage": float(coverage),
                    "coverage_error": float(coverage - q),
                    "underprediction_rate": float(np.mean(exceed_mask)),
                    "exceedance_mae": float(np.mean(y_test[exceed_mask] - pred[exceed_mask]))
                    if np.any(exceed_mask) else 0.0,
                    "mae_to_quantile_pred": float(mae_upper),
                    "top10_mae_to_quantile_pred": float(
                        np.mean(np.abs(y_test[high_mask] - pred[high_mask]))
                    ) if np.any(high_mask) else np.nan,
                    "top10_underprediction_rate": float(np.mean(exceed_mask[high_mask]))
                    if np.any(high_mask) else np.nan,
                    "top10_exceedance_mae": float(
                        np.mean(y_test[high_exceed] - pred[high_exceed])
                    ) if np.any(high_exceed) else 0.0,
                    "pred_q50": float(np.quantile(pred, 0.50)),
                    "pred_q90": float(np.quantile(pred, 0.90)),
                    "true_q90": float(np.quantile(y_test, 0.90)),
                }
            )

        if out_dir is not None and save_prediction_outputs and _should_plot_group(group_name, plot_groups):
            plot_quantile_preds = full_quantile_preds or quantile_preds
            save_prediction_cache(
                time_index=full_test_index if full_test_index is not None else test_index,
                y_true=y_full_test if y_full_test is not None else y_test,
                y_raw=y_raw_full_test,
                threshold_probs=full_threshold_probs or threshold_probs,
                quantile_preds=plot_quantile_preds,
                group_name=group_name,
                out_dir=out_dir,
                final_quantile=final_quantile,
            )
            if not skip_prediction_plots:
                plot_prediction_comparison(
                    time_index=full_test_index if full_test_index is not None else test_index,
                    y_true=y_full_test if y_full_test is not None else y_test,
                    y_raw=y_raw_full_test,
                    threshold_probs=full_threshold_probs or threshold_probs,
                    quantile_preds=plot_quantile_preds,
                    thresholds=thresholds,
                    quantiles=quantiles,
                    group_name=group_name,
                    out_dir=out_dir,
                    events_report=events_report,
                    final_quantile=final_quantile,
                )

    if out_dir is not None and low_far_event_rows:
        low_far_dir = os.path.join(out_dir, "low_false_alarm_event_eval")
        os.makedirs(low_far_dir, exist_ok=True)
        low_far_report = pd.DataFrame(low_far_event_rows)
        low_far_detail = pd.DataFrame(low_far_event_detail_rows)
        low_far_report.to_csv(
            os.path.join(low_far_dir, "low_far_event_level_ablation.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        if not low_far_detail.empty:
            low_far_detail.to_csv(
                os.path.join(low_far_dir, "low_far_event_predictions_long.csv"),
                index=False,
                encoding="utf-8-sig",
            )
        _plot_low_far_event_reports(low_far_report, low_far_dir)
        print(f"[Ablation] Saved low-FAR event evaluation: {low_far_dir}")

    return pd.DataFrame(cls_rows), pd.DataFrame(q_rows)


def run_continuous_regression_ablation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_groups: Dict[str, List[str]],
    target_col: str,
    thresholds: List[float],
    max_train_rows: int,
    max_test_rows: int,
    out_dir: Optional[str] = None,
    plot_groups: Optional[List[str]] = None,
    common_eval_index: bool = True,
    calibration_ratio: float = 0.2,
    strong_event_weight: float = 3.0,
    extreme_event_weight: float = 8.0,
    model_backend: str = "cnn_bilstm",
    huber_delta: float = 1.0,
) -> pd.DataFrame:
    backend = _normalize_backend(model_backend)
    if backend not in DEEP_BACKENDS:
        raise ValueError("Continuous Huber regression currently supports deep backends only.")

    rows = []
    common_train: Optional[pd.DataFrame] = None
    common_test: Optional[pd.DataFrame] = None
    common_full_test: Optional[pd.DataFrame] = None
    if common_eval_index:
        common_train, common_test, common_full_test = _prepare_common_frames(
            train_df=train_df,
            test_df=test_df,
            feature_groups=feature_groups,
            target_col=target_col,
            max_train_rows=max_train_rows,
            max_test_rows=max_test_rows,
        )

    for group_name, cols in feature_groups.items():
        print(f"[ContinuousReg] group={group_name}, n_features={len(cols)}")
        full_test_index: Optional[pd.DatetimeIndex] = None
        x_full_test: Optional[np.ndarray] = None
        y_full_test: Optional[np.ndarray] = None
        if common_train is not None and common_test is not None:
            x_train, y_train, x_test, y_test, test_index = _prepare_xy_from_frames(
                common_train, common_test, cols, target_col
            )
            if common_full_test is not None:
                x_full_test = common_full_test[cols].to_numpy(dtype=np.float32, copy=True)
                y_full_test = common_full_test[target_col].to_numpy(dtype=np.float32, copy=True)
                full_test_index = pd.DatetimeIndex(common_full_test.index)
        else:
            x_train, y_train, x_test, y_test, test_index = _prepare_xy(
                train_df, test_df, cols, target_col, max_train_rows, max_test_rows
            )
            full = test_df[cols + [target_col]].replace([np.inf, -np.inf], np.nan).dropna()
            x_full_test = full[cols].to_numpy(dtype=np.float32, copy=True)
            y_full_test = full[target_col].to_numpy(dtype=np.float32, copy=True)
            full_test_index = pd.DatetimeIndex(full.index)
        if len(y_train) == 0 or len(y_test) == 0:
            continue

        x_fit, y_fit, x_cal, y_cal = _split_fit_calibration(
            x_train, y_train, calibration_ratio
        )
        fit_weights = _strong_event_sample_weight(
            y_fit,
            strong_event_weight=strong_event_weight,
            extreme_event_weight=extreme_event_weight,
        )
        eval_sets = {
            "cal": (x_cal, y_cal, None),
            "test": (x_test, y_test, None),
        }
        reg = DeepHuberRegressor(backend, huber_delta=huber_delta)
        reg.fit(x_fit, y_fit, sample_weight=fit_weights, eval_sets=eval_sets)
        _save_deep_history(reg, out_dir, group_name, "continuous_huber")
        y_pred = reg.predict(x_test)
        base_metrics = _continuous_regression_metrics(y_test, y_pred)
        threshold_rows = _regression_threshold_metrics(y_test, y_pred, thresholds)
        if not threshold_rows:
            threshold_rows = [{"threshold_A": np.nan}]
        for thr_row in threshold_rows:
            rows.append(
                {
                    "feature_group": group_name,
                    "n_features": len(cols),
                    "model_backend": backend,
                    "huber_delta": float(huber_delta),
                    "n_train": int(len(y_train)),
                    "n_fit": int(len(y_fit)),
                    "n_cal": int(len(y_cal)),
                    "n_test": int(len(y_test)),
                    **base_metrics,
                    **thr_row,
                }
            )
        if out_dir is not None and _should_plot_group(group_name, plot_groups):
            if x_full_test is not None and y_full_test is not None:
                y_full_pred = reg.predict(x_full_test)
                _plot_continuous_prediction(full_test_index, y_full_test, y_full_pred, group_name, out_dir)
            else:
                _plot_continuous_prediction(test_index, y_test, y_pred, group_name, out_dir)

    return pd.DataFrame(rows)


def _plot_low_far_event_reports(report: pd.DataFrame, out_dir: str) -> None:
    if report.empty:
        return
    metrics = ["event_POD", "event_FAR", "event_CSI", "event_TSS", "event_HSS", "event_F1", "event_Bias"]
    for thr, sub in report.groupby("threshold_A"):
        sub = sub.sort_values("feature_group")
        fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 5.0))
        axes = np.atleast_1d(axes)
        for ax, metric in zip(axes, metrics):
            ax.barh(np.arange(len(sub)), sub[metric].to_numpy(dtype=float))
            ax.set_yticks(np.arange(len(sub)))
            ax.set_yticklabels(sub["feature_group"], fontsize=8)
            ax.invert_yaxis()
            ax.set_title(metric.replace("event_", ""))
            ax.grid(True, axis="x", alpha=0.25)
        fig.suptitle(f"Low false alarm event-level metrics >= {float(thr):g} A")
        fig.tight_layout()
        path = os.path.join(out_dir, f"low_far_event_metrics_thr{float(thr):g}A.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    matrix_rows = []
    for row in report.itertuples(index=False):
        for cell, value in [
            ("TP", getattr(row, "event_TP", 0)),
            ("FP", getattr(row, "event_FP", 0)),
            ("TN", getattr(row, "event_TN", 0)),
            ("FN", getattr(row, "event_FN", 0)),
        ]:
            matrix_rows.append(
                {
                    "feature_group": getattr(row, "feature_group"),
                    "threshold_A": float(getattr(row, "threshold_A")),
                    "matrix_type": "low_far_event",
                    "cell": cell,
                    "count": int(value),
                }
            )
    if matrix_rows:
        pd.DataFrame(matrix_rows).to_csv(
            os.path.join(out_dir, "low_far_event_confusion_matrices_long.csv"),
            index=False,
            encoding="utf-8-sig",
        )


def _should_plot_group(group_name: str, plot_groups: Optional[List[str]]) -> bool:
    if plot_groups is None or len(plot_groups) == 0:
        return group_name == "all_D"
    wanted = {str(g) for g in plot_groups}
    return "all" in wanted or group_name in wanted


def _plot_segments(
    x,
    y_true: np.ndarray,
    y_raw: Optional[np.ndarray],
    quantile_preds: Dict[float, np.ndarray],
    threshold_probs: Dict[float, np.ndarray],
    thresholds: List[float],
    quantiles: List[float],
    title: str,
    path: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
    ax = axes[0]
    ax.plot(x, y_true, color="black", linewidth=1.0, label="True future max")
    if y_raw is not None and len(y_raw) == len(y_true):
        ax.plot(x, y_raw, color="tab:gray", linewidth=0.9, alpha=0.85, label="True GIC")
    for q in sorted(quantiles):
        pred = quantile_preds.get(float(q))
        if pred is not None:
            ax.plot(x, pred, linewidth=1.0, label=f"Q{int(round(q * 100))}")
    for thr in thresholds:
        ax.axhline(float(thr), color="gray", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.set_ylabel("GIC future max (A)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncol=4, fontsize=9)

    ax2 = axes[1]
    for thr in thresholds:
        prob = threshold_probs.get(float(thr))
        if prob is not None:
            ax2.plot(x, prob, linewidth=1.0, label=f"P(>={thr:g}A)")
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_ylabel("Event probability")
    ax2.set_xlabel("Sample index in plotted test subset")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper right", ncol=4, fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Ablation] Saved prediction plot: {path}")


def _safe_time_token(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y%m%d_%H%M")


def _contiguous_time_segments(
    time_index: pd.DatetimeIndex,
    max_gap: pd.Timedelta = pd.Timedelta(minutes=2),
) -> List[Tuple[int, int]]:
    if len(time_index) == 0:
        return []
    if not isinstance(time_index, pd.DatetimeIndex):
        return [(0, len(time_index))]
    if len(time_index) == 1:
        return [(0, 1)]

    gaps = time_index.to_series().diff().to_numpy()
    breaks = [i for i in range(1, len(time_index)) if pd.notna(gaps[i]) and gaps[i] > max_gap]
    starts = [0] + breaks
    ends = breaks + [len(time_index)]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e > s]


def _save_full_event_plots(
    time_index: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_raw: Optional[np.ndarray],
    threshold_probs: Dict[float, np.ndarray],
    quantile_preds: Dict[float, np.ndarray],
    thresholds: List[float],
    quantiles: List[float],
    group_name: str,
    plot_dir: str,
) -> None:
    if len(time_index) != len(y_true) or not isinstance(time_index, pd.DatetimeIndex):
        return

    segments = _contiguous_time_segments(time_index, max_gap=pd.Timedelta(minutes=30))
    if len(segments) <= 1:
        return

    segment_dir = os.path.join(plot_dir, f"{group_name}_test_events")
    os.makedirs(segment_dir, exist_ok=True)
    rows = []
    for event_no, (start, end) in enumerate(segments, start=1):
        idx = np.arange(start, end)
        start_time = time_index[start]
        end_time = time_index[end - 1]
        event_max = float(np.nanmax(y_true[idx])) if len(idx) > 0 else np.nan
        rows.append(
            {
                "test_event_no": event_no,
                "start_time": start_time,
                "end_time": end_time,
                "n_points": int(end - start),
                "true_future_max_A": event_max,
                "start_row": int(start),
                "end_row_exclusive": int(end),
            }
        )
        x = time_index[idx]
        q_slice = {q: pred[idx] for q, pred in quantile_preds.items()}
        p_slice = {thr: prob[idx] for thr, prob in threshold_probs.items()}
        raw_slice = y_raw[idx] if y_raw is not None and len(y_raw) == len(y_true) else None
        title = (
            f"{group_name}: event {event_no:02d}, "
            f"{start_time} ~ {end_time}, max={event_max:.2f}A"
        )
        name = (
            f"{group_name}_test_event_{event_no:02d}_"
            f"{_safe_time_token(start_time)}_{_safe_time_token(end_time)}.png"
        )
        _plot_segments(
            x=x,
            y_true=y_true[idx],
            y_raw=raw_slice,
            quantile_preds=q_slice,
            threshold_probs=p_slice,
            thresholds=thresholds,
            quantiles=quantiles,
            title=title,
            path=os.path.join(segment_dir, name),
        )

    pd.DataFrame(rows).to_csv(
        os.path.join(segment_dir, f"{group_name}_test_events.csv"),
        index=False,
        encoding="utf-8-sig",
    )


def _save_report_event_plots(
    time_index: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_raw: Optional[np.ndarray],
    threshold_probs: Dict[float, np.ndarray],
    quantile_preds: Dict[float, np.ndarray],
    thresholds: List[float],
    quantiles: List[float],
    group_name: str,
    plot_dir: str,
    events_report: Optional[pd.DataFrame],
) -> None:
    if events_report is None or events_report.empty:
        return
    if len(time_index) != len(y_true) or not isinstance(time_index, pd.DatetimeIndex):
        return

    event_dir = os.path.join(plot_dir, f"{group_name}_paper_events")
    os.makedirs(event_dir, exist_ok=True)
    rows = []
    for row in events_report.itertuples(index=False):
        if getattr(row, "split", "") != "test":
            continue
        start_time = pd.Timestamp(getattr(row, "start"))
        end_time = pd.Timestamp(getattr(row, "end"))
        mask = (time_index >= start_time) & (time_index <= end_time)
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            continue
        paper_id = getattr(row, "paper_event_id", getattr(row, "event_id", len(rows) + 1))
        peak_time = getattr(row, "peak_time", pd.NaT)
        event_type = getattr(row, "event_type", "")
        event_type_raw = getattr(row, "event_type_raw", "")
        event_max = float(np.nanmax(y_true[idx]))
        rows.append(
            {
                "paper_event_id": paper_id,
                "event_type": event_type,
                "event_type_raw": event_type_raw,
                "peak_time": peak_time,
                "start_time": start_time,
                "end_time": end_time,
                "n_points": int(len(idx)),
                "true_future_max_A": event_max,
                "start_row": int(idx[0]),
                "end_row_exclusive": int(idx[-1] + 1),
            }
        )
        q_slice = {q: pred[idx] for q, pred in quantile_preds.items()}
        p_slice = {thr: prob[idx] for thr, prob in threshold_probs.items()}
        raw_slice = y_raw[idx] if y_raw is not None and len(y_raw) == len(y_true) else None
        title = (
            f"{group_name}: paper event {int(paper_id):02d} | {event_type} ({event_type_raw}) | "
            f"peak={peak_time} | max={event_max:.2f}A"
        )
        name = (
            f"{group_name}_paper_event_{int(paper_id):02d}_"
            f"{_safe_time_token(start_time)}_{_safe_time_token(end_time)}.png"
        )
        _plot_segments(
            x=time_index[idx],
            y_true=y_true[idx],
            y_raw=raw_slice,
            quantile_preds=q_slice,
            threshold_probs=p_slice,
            thresholds=thresholds,
            quantiles=quantiles,
            title=title,
            path=os.path.join(event_dir, name),
        )

    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(event_dir, f"{group_name}_paper_events.csv"),
            index=False,
            encoding="utf-8-sig",
        )


def save_prediction_cache(
    time_index: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_raw: Optional[np.ndarray],
    threshold_probs: Dict[float, np.ndarray],
    quantile_preds: Dict[float, np.ndarray],
    group_name: str,
    out_dir: str,
    final_quantile: float = 0.5,
) -> Optional[str]:
    if len(y_true) == 0:
        return None
    cache_dir = os.path.join(out_dir, "prediction_cache")
    os.makedirs(cache_dir, exist_ok=True)
    df = pd.DataFrame(index=time_index if len(time_index) == len(y_true) else None)
    if len(time_index) == len(y_true):
        df.index.name = "datetime"
    df["y_true"] = np.asarray(y_true, dtype=np.float32)
    if y_raw is not None and len(y_raw) == len(y_true):
        df["gic_true"] = np.asarray(y_raw, dtype=np.float32)
    for thr, prob in sorted(threshold_probs.items()):
        df[f"prob_ge_{float(thr):g}A"] = np.asarray(prob, dtype=np.float32)
    for q, pred in sorted(quantile_preds.items()):
        df[f"Q{int(round(float(q) * 100))}"] = np.asarray(pred, dtype=np.float32)
    if quantile_preds:
        q_keys = sorted(quantile_preds.keys())
        chosen_q = min(q_keys, key=lambda q: abs(float(q) - float(final_quantile)))
        df["final_pred"] = np.asarray(quantile_preds[chosen_q], dtype=np.float32)
        df["final_pred_quantile"] = float(chosen_q)
    path = os.path.join(cache_dir, f"{group_name}_predictions.parquet")
    df.to_parquet(path)
    csv_path = os.path.join(cache_dir, f"{group_name}_predictions.csv")
    df.to_csv(csv_path, encoding="utf-8-sig")
    print(f"[Ablation] Saved prediction cache: {path}")
    print(f"[Ablation] Saved prediction CSV: {csv_path}")
    return path


def plot_prediction_comparison(
    time_index: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_raw: Optional[np.ndarray],
    threshold_probs: Dict[float, np.ndarray],
    quantile_preds: Dict[float, np.ndarray],
    thresholds: List[float],
    quantiles: List[float],
    group_name: str,
    out_dir: str,
    events_report: Optional[pd.DataFrame] = None,
    final_quantile: float = 0.5,
    max_points: int = 6000,
) -> None:
    if len(y_true) == 0:
        return
    plot_dir = os.path.join(out_dir, "prediction_plots")
    os.makedirs(plot_dir, exist_ok=True)

    n = len(y_true)
    if n <= max_points:
        idx = np.arange(n)
    else:
        # Keep a compact view centered on the largest true event.
        peak = int(np.nanargmax(y_true))
        half = max_points // 2
        start = max(0, peak - half)
        end = min(n, start + max_points)
        start = max(0, end - max_points)
        idx = np.arange(start, end)

    # Keep the overview plot identical to the original 6000-point figure:
    # x-axis is the sample index inside the plotted test subset, not datetime.
    x = np.arange(len(idx))
    y_slice = y_true[idx]
    raw_slice = y_raw[idx] if y_raw is not None and len(y_raw) == len(y_true) else None
    q_slice = {q: pred[idx] for q, pred in quantile_preds.items()}
    p_slice = {thr: prob[idx] for thr, prob in threshold_probs.items()}
    q_labels = "/".join(f"Q{int(round(float(q) * 100))}" for q in sorted(quantiles))
    title = f"{group_name}: threshold probability + {q_labels} envelope"
    path = os.path.join(plot_dir, f"{group_name}_prediction_comparison.png")
    _plot_segments(
        x=x,
        y_true=y_slice,
        y_raw=raw_slice,
        quantile_preds=q_slice,
        threshold_probs=p_slice,
        thresholds=thresholds,
        quantiles=quantiles,
        title=title,
        path=path,
    )

    _save_full_event_plots(
        time_index=time_index,
        y_true=y_true,
        y_raw=y_raw,
        threshold_probs=threshold_probs,
        quantile_preds=quantile_preds,
        thresholds=thresholds,
        quantiles=quantiles,
        group_name=group_name,
        plot_dir=plot_dir,
    )

    _save_report_event_plots(
        time_index=time_index,
        y_true=y_true,
        y_raw=y_raw,
        threshold_probs=threshold_probs,
        quantile_preds=quantile_preds,
        thresholds=thresholds,
        quantiles=quantiles,
        group_name=group_name,
        plot_dir=plot_dir,
        events_report=events_report,
    )

    # Event-wise panels around the largest true peaks, for easier inspection.
    peak_order = np.argsort(y_true)[::-1]
    centers = []
    min_gap = 180
    for p in peak_order:
        p = int(p)
        if all(abs(p - c) > min_gap for c in centers):
            centers.append(p)
        if len(centers) >= 6:
            break
    if not centers:
        return

    fig, axes = plt.subplots(len(centers), 1, figsize=(16, 2.8 * len(centers)), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, center in zip(axes, centers):
        start = max(0, center - 180)
        end = min(n, center + 240)
        xx = np.arange(start, end) - start
        ax.plot(xx, y_true[start:end], color="black", linewidth=1.0, label="True future max")
        if y_raw is not None and len(y_raw) == len(y_true):
            ax.plot(xx, y_raw[start:end], color="tab:gray", linewidth=0.9, alpha=0.8, label="True GIC")
        for q in sorted(quantiles):
            pred = quantile_preds.get(float(q))
            if pred is not None:
                ax.plot(xx, pred[start:end], linewidth=1.0, label=f"Q{int(round(q * 100))}")
        ax.axvline(center - start, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(
            f"peak@{center}, true={float(y_true[center]):.2f}A"
            + (f", time={time_index[center]}" if center < len(time_index) else "")
        )
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", ncol=4, fontsize=9)
    fig.tight_layout()
    path = os.path.join(plot_dir, f"{group_name}_top_events_quantile_detail.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Ablation] Saved prediction plot: {path}")


def build_summary(cls_report: pd.DataFrame, quantile_report: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = sorted(
        set(cls_report.get("feature_group", pd.Series(dtype=str)).dropna().unique()).union(
            set(quantile_report.get("feature_group", pd.Series(dtype=str)).dropna().unique())
        )
    )
    for group in groups:
        cls_g = cls_report[cls_report["feature_group"].eq(group)] if not cls_report.empty else pd.DataFrame()
        q_g = quantile_report[quantile_report["feature_group"].eq(group)] if not quantile_report.empty else pd.DataFrame()
        row = {"feature_group": group}
        if not cls_g.empty:
            row.update(
                {
                    "best_AUC": float(cls_g["AUC"].max()),
                    "best_F1": float(cls_g["F1"].max()),
                    "best_TSS": float(cls_g["TSS"].max()),
                    "best_HSS": float(cls_g["HSS"].max()),
                }
            )
            for thr in sorted(cls_g["threshold_A"].unique()):
                s = cls_g[cls_g["threshold_A"].eq(thr)].iloc[0]
                row[f"thr{thr:g}_AUC"] = float(s["AUC"])
                row[f"thr{thr:g}_POD"] = float(s["POD"])
                row[f"thr{thr:g}_FAR"] = float(s["FAR"])
                row[f"thr{thr:g}_F1"] = float(s["F1"])
                row[f"thr{thr:g}_TSS"] = float(s["TSS"])
                for metric in ["POD", "POFD", "FAR", "CSI", "TSS", "HSS", "F1", "AUC", "Bias"]:
                    col = f"event_{metric}"
                    if col in s.index:
                        row[f"thr{thr:g}_event_{metric}"] = float(s[col])
        if not q_g.empty:
            for q in sorted(q_g["quantile"].unique()):
                s = q_g[q_g["quantile"].eq(q)].iloc[0]
                label = f"Q{int(round(q * 100))}"
                row[f"{label}_pinball"] = float(s["pinball_loss"])
                row[f"{label}_coverage"] = float(s["coverage"])
                row[f"{label}_coverage_error"] = float(s["coverage_error"])
                row[f"{label}_under_rate"] = float(s["underprediction_rate"])
                row[f"{label}_top10_under_rate"] = float(s["top10_underprediction_rate"])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_ablation_reports(cls_report: pd.DataFrame, quantile_report: pd.DataFrame, out_dir: str) -> None:
    if not cls_report.empty:
        metrics = ["AUC", "POD", "FAR", "F1", "TSS"]
        for thr, sub in cls_report.groupby("threshold_A"):
            sub = sub.sort_values("AUC", ascending=False)
            fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 5))
            for ax, metric in zip(axes, metrics):
                ax.barh(np.arange(len(sub)), sub[metric].to_numpy())
                ax.set_yticks(np.arange(len(sub)))
                ax.set_yticklabels(sub["feature_group"], fontsize=8)
                ax.invert_yaxis()
                ax.set_title(metric)
                ax.grid(True, axis="x", alpha=0.25)
            fig.suptitle(f"Threshold classification >= {thr:g} A")
            fig.tight_layout()
            path = os.path.join(out_dir, f"classification_threshold_{thr:g}A.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[Ablation] Saved figure: {path}")

    save_confusion_matrix_reports(cls_report, out_dir)

    if not quantile_report.empty:
        metrics = ["pinball_loss", "coverage", "underprediction_rate", "top10_underprediction_rate"]
        for q, sub in quantile_report.groupby("quantile"):
            sub = sub.sort_values("pinball_loss", ascending=True)
            fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 5))
            for ax, metric in zip(axes, metrics):
                ax.barh(np.arange(len(sub)), sub[metric].to_numpy())
                ax.set_yticks(np.arange(len(sub)))
                ax.set_yticklabels(sub["feature_group"], fontsize=8)
                ax.invert_yaxis()
                ax.set_title(metric)
                if metric == "coverage":
                    ax.axvline(float(q), color="red", linestyle="--", linewidth=1)
                ax.grid(True, axis="x", alpha=0.25)
            fig.suptitle(f"Upper quantile regression Q{int(round(q * 100))}")
            fig.tight_layout()
            path = os.path.join(out_dir, f"quantile_regression_Q{int(round(q * 100))}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[Ablation] Saved figure: {path}")


def save_confusion_matrix_reports(cls_report: pd.DataFrame, out_dir: str) -> None:
    if cls_report.empty:
        return
    cm_dir = os.path.join(out_dir, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)
    rows = []

    def _add_matrix(row, prefix: str, label: str) -> Optional[np.ndarray]:
        keys = [f"{prefix}TN", f"{prefix}FP", f"{prefix}FN", f"{prefix}TP"]
        if not all(k in row.index and pd.notna(row[k]) for k in keys):
            return None
        tn, fp, fn, tp = [int(row[k]) for k in keys]
        rows.extend(
            [
                {
                    "feature_group": row["feature_group"],
                    "threshold_A": float(row["threshold_A"]),
                    "matrix_type": label,
                    "actual": "negative",
                    "predicted": "negative",
                    "count": tn,
                },
                {
                    "feature_group": row["feature_group"],
                    "threshold_A": float(row["threshold_A"]),
                    "matrix_type": label,
                    "actual": "negative",
                    "predicted": "positive",
                    "count": fp,
                },
                {
                    "feature_group": row["feature_group"],
                    "threshold_A": float(row["threshold_A"]),
                    "matrix_type": label,
                    "actual": "positive",
                    "predicted": "negative",
                    "count": fn,
                },
                {
                    "feature_group": row["feature_group"],
                    "threshold_A": float(row["threshold_A"]),
                    "matrix_type": label,
                    "actual": "positive",
                    "predicted": "positive",
                    "count": tp,
                },
            ]
        )
        return np.array([[tn, fp], [fn, tp]], dtype=int)

    for _, row in cls_report.iterrows():
        matrices = [
            ("sample", _add_matrix(row, "", "sample")),
            ("event", _add_matrix(row, "event_", "event")),
        ]
        for matrix_type, matrix in matrices:
            if matrix is None:
                continue
            group = str(row["feature_group"])
            thr = float(row["threshold_A"])
            fig, ax = plt.subplots(figsize=(4.8, 4.2))
            im = ax.imshow(matrix, cmap="Blues")
            ax.set_xticks([0, 1], labels=["Pred Neg", "Pred Pos"])
            ax.set_yticks([0, 1], labels=["Actual Neg", "Actual Pos"])
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
            title = f"{group} | {matrix_type} confusion | >= {thr:g}A"
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            path = os.path.join(
                cm_dir,
                f"{_safe_name(group)}_{matrix_type}_thr{thr:g}A_confusion.png",
            )
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)

    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(cm_dir, "confusion_matrices_long.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        print(f"[Ablation] Saved confusion matrices: {cm_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Threshold classification and quantile upper-envelope ablation"
    )
    parser.add_argument("--data", default=PROCESSED_DATA_FILE)
    parser.add_argument("--feature-set", default="D", choices=["A", "B", "C", "D"])
    parser.add_argument("--target", default=TARGET_VYK_COL)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--scope", default="event-type", choices=["fixed-events", "event-type", "full-time"])
    parser.add_argument("--event-types", nargs="*", default=["CME", "CIR"])
    parser.add_argument(
        "--event-type-batches",
        nargs="*",
        choices=["CME", "CIR", "CME_CIR"],
        default=None,
        help="Run separate event-type models in one command, e.g. CME CIR CME_CIR.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--thresholds", nargs="*", type=float, default=[3.0, 5.0, 10.0, 20.0])
    parser.add_argument("--quantiles", nargs="*", type=float, default=[0.80, 0.90, 0.95])
    parser.add_argument("--max-train-rows", type=int, default=250000)
    parser.add_argument("--max-test-rows", type=int, default=120000)
    parser.add_argument(
        "--model-backend",
        default="hgb",
        choices=["hgb", "lightgbm", "bilstm", "cnn_bilstm", "cnn_bilstm_attention"],
    )
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument(
        "--threshold-strategy",
        default="max-tss",
        choices=["fixed", "max-tss", "max-f1", "far-control"],
    )
    parser.add_argument("--max-far", type=float, default=0.4)
    parser.add_argument(
        "--low-far-event-max-far",
        type=float,
        default=0.2,
        help="Maximum event-level FAR used by the additional low-false-alarm event evaluation.",
    )
    parser.add_argument("--strong-event-weight", type=float, default=3.0)
    parser.add_argument("--extreme-event-weight", type=float, default=8.0)
    parser.add_argument(
        "--no-quantile-calibration",
        action="store_true",
        help="Disable post-hoc Q80/Q90/Q95 coverage calibration.",
    )
    parser.add_argument(
        "--independent-group-samples",
        action="store_true",
        help="Use old behavior: each feature group drops NaNs and samples rows independently.",
    )
    parser.add_argument(
        "--plot-groups",
        nargs="*",
        default=["geomag_response", "geomag_plus_time", "solar_plus_time", "all_D"],
        help="Feature groups to draw prediction comparisons for. Use 'all' for every group.",
    )
    parser.add_argument(
        "--skip-prediction-plots",
        action="store_true",
        help="Save prediction_cache CSV/parquet but skip expensive prediction/event plots.",
    )
    parser.add_argument(
        "--skip-report-plots",
        action="store_true",
        help="Skip ablation summary/confusion figure generation.",
    )
    parser.add_argument(
        "--run-continuous-regression",
        action="store_true",
        help="Also train weighted-Huber continuous deep regressors for gic future max.",
    )
    parser.add_argument("--huber-delta", type=float, default=1.0)
    args = parser.parse_args()

    print(f"[Ablation] Loading: {args.data}")
    df = pd.read_parquet(args.data)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    base_features = get_feature_columns(df, feature_set=args.feature_set, target_name=args.target)
    df, label_col = _add_future_window_max_label(df, args.target, args.horizon)

    batch_map = {"CME": ["CME"], "CIR": ["CIR"], "CME_CIR": ["CME", "CIR"]}
    if args.scope == "event-type" and args.event_type_batches:
        event_type_batches = [batch_map[name] for name in args.event_type_batches]
    else:
        event_type_batches = [args.event_types]

    for event_types in event_type_batches:
        train_df, test_df, scope_label = _split_events(
            df,
            scope=args.scope,
            event_types=event_types,
            train_ratio=args.train_ratio,
        )
        groups = _root_groups(base_features)

        out_dir = os.path.join(
        EXPERIMENT_DIR,
        f"threshold_quantile_ablation_{args.model_backend}",
        _safe_name(scope_label),
        _safe_name(f"{args.target}_H{args.horizon}"),
        )
        os.makedirs(out_dir, exist_ok=True)
        print(
            f"[Ablation] scope={scope_label}, train_rows={len(train_df):,}, "
            f"test_rows={len(test_df):,}, out={out_dir}"
        )

        cls_report, quantile_report = run_ablation(
            train_df=train_df,
            test_df=test_df,
            feature_groups=groups,
            target_col=label_col,
            thresholds=args.thresholds,
            quantiles=args.quantiles,
            max_train_rows=args.max_train_rows,
            max_test_rows=args.max_test_rows,
            out_dir=out_dir,
            plot_groups=args.plot_groups,
            common_eval_index=not args.independent_group_samples,
            calibration_ratio=args.calibration_ratio,
            threshold_strategy=args.threshold_strategy,
            max_far=args.max_far,
            strong_event_weight=args.strong_event_weight,
            extreme_event_weight=args.extreme_event_weight,
            quantile_calibration=not args.no_quantile_calibration,
            model_backend=args.model_backend,
            low_far_event_max_far=args.low_far_event_max_far,
            skip_prediction_plots=args.skip_prediction_plots,
        )

        regression_report = pd.DataFrame()
        if args.run_continuous_regression:
            regression_report = run_continuous_regression_ablation(
                train_df=train_df,
                test_df=test_df,
                feature_groups=groups,
                target_col=label_col,
                thresholds=args.thresholds,
                max_train_rows=args.max_train_rows,
                max_test_rows=args.max_test_rows,
                out_dir=out_dir,
                plot_groups=args.plot_groups,
                common_eval_index=not args.independent_group_samples,
                calibration_ratio=args.calibration_ratio,
                strong_event_weight=args.strong_event_weight,
                extreme_event_weight=args.extreme_event_weight,
                model_backend=args.model_backend,
                huber_delta=args.huber_delta,
            )

        cls_path = os.path.join(out_dir, "threshold_classification_ablation.csv")
        q_path = os.path.join(out_dir, "quantile_regression_ablation.csv")
        reg_path = os.path.join(out_dir, "continuous_regression_ablation.csv")
        summary_path = os.path.join(out_dir, "feature_ablation_summary.csv")
        summary = build_summary(cls_report, quantile_report)
        cls_report.to_csv(cls_path, index=False, encoding="utf-8-sig")
        quantile_report.to_csv(q_path, index=False, encoding="utf-8-sig")
        if not regression_report.empty:
            regression_report.to_csv(reg_path, index=False, encoding="utf-8-sig")
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        if not args.skip_report_plots:
            plot_ablation_reports(cls_report, quantile_report, out_dir)
        print(f"[Ablation] Saved: {cls_path}")
        print(f"[Ablation] Saved: {q_path}")
        if not regression_report.empty:
            print(f"[Ablation] Saved: {reg_path}")
        print(f"[Ablation] Saved: {summary_path}")
        if not cls_report.empty:
            print(cls_report.sort_values(["threshold_A", "AUC"], ascending=[True, False]).to_string(index=False))
        if not quantile_report.empty:
            print(quantile_report.sort_values(["quantile", "pinball_loss"]).to_string(index=False))
        if not regression_report.empty:
            print(regression_report.sort_values(["threshold_A", "TSS"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
