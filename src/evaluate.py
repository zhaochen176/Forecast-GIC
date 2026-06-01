"""
GIC 预测项目 - 评估模块

支持:
1) 绝对阈值评估 (2/3/4/5A，可按目标启停)
2) 分位数阈值评估 (Q90/Q95/Q97/Q99，输出具体阈值数值)
3) 事件导向指标 (命中率、峰值时间误差、峰值幅值误差、突变检测F1)
4) 区间指标 (PICP/MPIW/Winkler/ACE/CWC)
"""
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable

from src.config import (
    AMP_DTYPE,
    DEVICE,
    PEAK_EVAL_THRESHOLDS,
    PI_ALPHA_LIST,
    POINT_PREDICTION_MODE,
    POINT_BLEND_LOW_Q,
    POINT_BLEND_HIGH_Q,
    POINT_BLEND_MAX_ALPHA,
    POINT_BLEND_LOGIT_SCALE,
    POINT_BLEND_LOGIT_BIAS,
    QUANTILES,
    REPORT_DIR,
    USE_AMP,
)


def _get_loader_aligned_raw_targets(loader) -> Optional[np.ndarray]:
    ds = getattr(loader, "dataset", None)
    if ds is None:
        return None
    getter = getattr(ds, "get_aligned_raw_targets", None)
    if getter is None:
        return None
    try:
        arr = getter()
        return np.asarray(arr, dtype=np.float32)
    except Exception:
        return None


@torch.no_grad()
def get_predictions(model, loader, device=DEVICE):
    """返回: y_true, y_pred_quantiles, y_pred_point, sample_weights, peak_logits"""
    model.eval()
    all_true, all_pred_q, all_pred_point, all_weights = [], [], [], []
    all_peak_logits = []
    amp_ctx = torch.amp.autocast("cuda", enabled=USE_AMP, dtype=AMP_DTYPE)
    def _q_idx(qv: float) -> int:
        if qv in QUANTILES:
            return QUANTILES.index(qv)
        arr = np.asarray(QUANTILES, dtype=np.float32)
        return int(np.argmin(np.abs(arr - float(qv))))

    def _unpack_model_outputs(model_out):
        if not isinstance(model_out, (tuple, list)):
            raise TypeError(f"Unexpected model output type: {type(model_out)}")
        if len(model_out) == 4:
            pred, attn, peak_logit, peak_expert_out = model_out
            point_reg_out = None
            return pred, attn, peak_logit, peak_expert_out, point_reg_out
        if len(model_out) >= 5:
            pred, attn, peak_logit, peak_expert_out, point_reg_out = model_out[:5]
            return pred, attn, peak_logit, peak_expert_out, point_reg_out
        raise ValueError(f"Unexpected model output tuple length: {len(model_out)}")

    def _build_point_prediction(pred_q_t, peak_logit_t, peak_expert_out_t, point_reg_out_t):
        q50 = pred_q_t[:, _q_idx(0.5)]
        if POINT_PREDICTION_MODE == "median":
            return q50
        if POINT_PREDICTION_MODE == "point_reg_head":
            if point_reg_out_t is None:
                return q50
            return point_reg_out_t.squeeze(-1).cpu().float()
        if POINT_PREDICTION_MODE == "quantile_blend":
            q_low = pred_q_t[:, _q_idx(POINT_BLEND_LOW_Q)]
            q_high = pred_q_t[:, _q_idx(POINT_BLEND_HIGH_Q)]
            blend_logit = (
                peak_logit_t.squeeze(-1).cpu().float() * POINT_BLEND_LOGIT_SCALE
                + POINT_BLEND_LOGIT_BIAS
            )
            alpha = torch.sigmoid(blend_logit) * POINT_BLEND_MAX_ALPHA
            return q_low + alpha * (q_high - q_low)
        gate = torch.sigmoid(peak_logit_t.squeeze(-1).cpu().float())
        expert = peak_expert_out_t.squeeze(-1).cpu().float()
        if POINT_PREDICTION_MODE == "gated_residual":
            return q50 + gate * (expert - q50)
        if POINT_PREDICTION_MODE == "gated_positive":
            return q50 + gate * torch.relu(expert - q50)
        raise ValueError(f"Unsupported POINT_PREDICTION_MODE={POINT_PREDICTION_MODE}")

    for X, y, w in tqdm(loader, desc="  预测", leave=False):
        X = X.to(device, non_blocking=True)
        with amp_ctx:
            pred_q, _, peak_logit, peak_expert_out, point_reg_out = _unpack_model_outputs(model(X))
        pred_q = pred_q.cpu().float()

        pred_point = _build_point_prediction(
            pred_q, peak_logit, peak_expert_out, point_reg_out
        )
        has_cls = getattr(model, "has_classification", True)
        if has_cls:
            all_peak_logits.append(peak_logit.squeeze(-1).cpu().float().numpy())
        else:
            all_peak_logits.append(np.zeros(len(y), dtype=np.float32))

        all_true.append(y.numpy())
        all_pred_q.append(pred_q.numpy())
        all_pred_point.append(pred_point.numpy())
        all_weights.append(w.numpy())

    return (
        np.concatenate(all_true),
        np.concatenate(all_pred_q),
        np.concatenate(all_pred_point),
        np.concatenate(all_weights),
        np.concatenate(all_peak_logits),
    )


def compute_point_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold=None):
    if threshold is not None:
        mask = y_true >= threshold
        if mask.sum() == 0:
            return {"n": 0, "mae": np.nan, "rmse": np.nan, "mape": np.nan}
        y_t = y_true[mask]
        y_p = y_pred[mask]
    else:
        y_t = y_true
        y_p = y_pred

    abs_err = np.abs(y_t - y_p)
    return {
        "n": int(len(y_t)),
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt((abs_err ** 2).mean())),
        "mape": float((abs_err / (np.abs(y_t) + 1e-8)).mean() * 100.0),
    }


def compute_peak_metrics(y_true, y_pred_median, y_pred_quantiles, threshold, quantiles=QUANTILES):
    true_peaks = y_true >= threshold
    pred_peaks = y_pred_median >= threshold * 0.8
    n_true = int(true_peaks.sum())
    n_pred = int(pred_peaks.sum())
    if n_true == 0:
        return {"n_peak": 0}

    tp = int((true_peaks & pred_peaks).sum())
    fp = int((pred_peaks & ~true_peaks).sum())

    errs = y_true[true_peaks] - y_pred_median[true_peaks]
    q05i = quantiles.index(0.05) if 0.05 in quantiles else 0
    q95i = quantiles.index(0.95) if 0.95 in quantiles else -1

    return {
        "n_peak": n_true,
        "capture": float(tp / max(n_true, 1)),
        "false_alarm": float(fp / max(n_pred, 1)),
        "peak_mae": float(np.mean(np.abs(errs))),
        "peak_rmse": float(np.sqrt(np.mean(errs ** 2))),
        "peak_mape": float(np.mean(np.abs(errs) / (y_true[true_peaks] + 1e-8)) * 100),
        "under_ratio": float((y_pred_median[true_peaks] < y_true[true_peaks]).sum() / max(n_true, 1)),
        "peak_picp_90": float(
            (
                (y_true[true_peaks] >= y_pred_quantiles[true_peaks, q05i])
                & (y_true[true_peaks] <= y_pred_quantiles[true_peaks, q95i])
            ).mean()
        ),
    }


def compute_classification_metrics(y_true, peak_logits, threshold_a, pred_threshold=0.5):
    y_bin = (y_true >= threshold_a).astype(int)
    prob = 1.0 / (1.0 + np.exp(-np.clip(peak_logits, -50, 50)))
    pred = (prob >= pred_threshold).astype(int)

    TP = int(((pred == 1) & (y_bin == 1)).sum())
    FP = int(((pred == 1) & (y_bin == 0)).sum())
    FN = int(((pred == 0) & (y_bin == 1)).sum())
    TN = int(((pred == 0) & (y_bin == 0)).sum())

    POD = TP / max(TP + FN, 1)
    POFD = FP / max(FP + TN, 1)
    TSS = POD - POFD
    num_hss = 2 * (TP * TN - FN * FP)
    den_hss = (TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)
    HSS = num_hss / max(den_hss, 1)
    Bias = (TP + FP) / max(TP + FN, 1)
    F1 = 2 * TP / max(2 * TP + FP + FN, 1)

    fpr, tpr = np.array([0, 1]), np.array([0, 1])
    auc_val = float("nan")
    try:
        from sklearn.metrics import roc_auc_score, roc_curve

        if len(np.unique(y_bin)) > 1:
            auc_val = float(roc_auc_score(y_bin, prob))
            fpr, tpr, _ = roc_curve(y_bin, prob)
    except Exception:
        pass

    return {
        "n_events": int(y_bin.sum()),
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "POD": float(POD),
        "POFD": float(POFD),
        "HSS": float(HSS),
        "TSS": float(TSS),
        "Bias": float(Bias),
        "F1": float(F1),
        "AUC": float(auc_val),
        "fpr": fpr,
        "tpr": tpr,
    }


def compute_interval_metrics(y_true, y_pred_quantiles, quantiles=QUANTILES, alpha=0.1, subset_mask=None):
    if subset_mask is not None:
        y = y_true[subset_mask]
        yq = y_pred_quantiles[subset_mask]
    else:
        y = y_true
        yq = y_pred_quantiles
    if len(y) == 0:
        return {}

    q_lo = alpha / 2
    q_hi = 1 - alpha / 2
    lo_idx = min(range(len(quantiles)), key=lambda i: abs(quantiles[i] - q_lo))
    hi_idx = min(range(len(quantiles)), key=lambda i: abs(quantiles[i] - q_hi))
    L = yq[:, lo_idx]
    U = yq[:, hi_idx]
    width = U - L

    covered = ((y >= L) & (y <= U))
    picp = float(covered.mean())
    mpiw = float(width.mean())
    y_range = float(y.max() - y.min()) if len(y) else 0.0
    nmpiw = mpiw / (y_range + 1e-8)

    penalty_lo = np.maximum(L - y, 0)
    penalty_hi = np.maximum(y - U, 0)
    ws = float((width + (2.0 / alpha) * (penalty_lo + penalty_hi)).mean())

    nominal = 1 - alpha
    ace = float(picp - nominal)
    cwc = float(nmpiw * (1 + 50.0 * max(0, nominal - picp) ** 2))

    return {
        "picp": picp,
        "mpiw": mpiw,
        "nmpiw": float(nmpiw),
        "winkler_score": ws,
        "ace": ace,
        "cwc": cwc,
        "n_samples": int(len(y)),
    }


def _extract_events(series: np.ndarray, threshold: float):
    events = []
    idxs = np.where(series >= threshold)[0]
    if len(idxs) == 0:
        return events

    start = idxs[0]
    prev = idxs[0]
    for idx in idxs[1:]:
        if idx != prev + 1:
            seg = np.arange(start, prev + 1)
            peak_idx = int(seg[np.argmax(series[seg])])
            events.append((int(start), int(prev), peak_idx, float(series[peak_idx])))
            start = idx
        prev = idx
    seg = np.arange(start, prev + 1)
    peak_idx = int(seg[np.argmax(series[seg])])
    events.append((int(start), int(prev), peak_idx, float(series[peak_idx])))
    return events


def _compute_jump_f1(y_true: np.ndarray, y_pred: np.ndarray):
    if len(y_true) < 3:
        return 0.0
    d_true = np.abs(np.diff(y_true))
    d_pred = np.abs(np.diff(y_pred))
    thr = float(np.quantile(d_true, 0.95))
    t = (d_true >= thr).astype(int)
    p = (d_pred >= thr).astype(int)
    tp = int(((t == 1) & (p == 1)).sum())
    fp = int(((t == 0) & (p == 1)).sum())
    fn = int(((t == 1) & (p == 0)).sum())
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def compute_event_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> Dict:
    true_events = _extract_events(y_true, threshold)
    pred_events = _extract_events(y_pred, threshold)

    n_true = len(true_events)
    n_pred = len(pred_events)
    if n_true == 0:
        return {
            "n_true_events": 0,
            "n_pred_events": n_pred,
            "event_hit_rate": np.nan,
            "peak_time_mae_idx": np.nan,
            "peak_value_mae": np.nan,
            "jump_f1": _compute_jump_f1(y_true, y_pred),
        }

    hits = 0
    time_err = []
    amp_err = []
    for t_start, t_end, t_peak_idx, t_peak_val in true_events:
        overlap = [
            pe for pe in pred_events
            if not (pe[1] < t_start or pe[0] > t_end)
        ]
        if not overlap:
            continue
        hits += 1
        best = min(overlap, key=lambda pe: abs(pe[2] - t_peak_idx))
        time_err.append(abs(best[2] - t_peak_idx))
        amp_err.append(abs(best[3] - t_peak_val))

    return {
        "n_true_events": n_true,
        "n_pred_events": n_pred,
        "event_hit_rate": float(hits / max(n_true, 1)),
        "peak_time_mae_idx": float(np.mean(time_err)) if time_err else np.nan,
        "peak_value_mae": float(np.mean(amp_err)) if amp_err else np.nan,
        "jump_f1": _compute_jump_f1(y_true, y_pred),
    }


def get_top_peaks(y_true, y_pred, n=10):
    idx = np.argsort(y_true)[::-1][:n]
    return [
        {
            "rank": i + 1,
            "true": float(y_true[j]),
            "pred": float(y_pred[j]),
            "error_pct": float((y_pred[j] - y_true[j]) / (y_true[j] + 1e-8) * 100),
        }
        for i, j in enumerate(idx)
    ]


def evaluate_model(
    model,
    loader,
    quantiles=QUANTILES,
    absolute_thresholds: Optional[List[float]] = None,
    quantile_threshold_values: Optional[Dict[str, float]] = None,
    quantile_threshold_stats: Optional[List[Dict]] = None,
    device=DEVICE,
    dataset_name="测试集",
    target_scaler=None,
    model_name=None,
    target_name="target",
    horizon: int = 30,
):
    if model_name is None:
        model_name = getattr(model, "model_name", "未知模型")
    if absolute_thresholds is None:
        absolute_thresholds = PEAK_EVAL_THRESHOLDS
    if quantile_threshold_values is None:
        quantile_threshold_values = {}
    if quantile_threshold_stats is None:
        quantile_threshold_stats = []

    print(f"\n{'='*72}")
    print(
        f"[评估] {model_name} | {dataset_name} | target={target_name} | H={horizon}"
    )
    print(f"{'='*72}")

    y_true, y_pred_q, y_pred_point, _, peak_logits = get_predictions(model, loader, device)
    if target_scaler is not None:
        y_true = target_scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
        for i in range(y_pred_q.shape[1]):
            y_pred_q[:, i] = target_scaler.inverse_transform(y_pred_q[:, i].reshape(-1, 1)).ravel()
        y_pred_point = target_scaler.inverse_transform(y_pred_point.reshape(-1, 1)).ravel()

    y_pred_median = y_pred_point
    y_true_plot = y_true.copy()
    raw_true_aligned = _get_loader_aligned_raw_targets(loader)
    if raw_true_aligned is not None and len(raw_true_aligned) == len(y_true_plot):
        y_true_plot = raw_true_aligned

    pm_all = compute_point_metrics(y_true, y_pred_median, threshold=None)
    print(
        f"  全局: MAE={pm_all['mae']:.3f}, RMSE={pm_all['rmse']:.3f}, "
        f"MAPE={pm_all['mape']:.1f}%"
    )

    point_metrics = {"all": pm_all}
    peak_metrics = {}
    cls_metrics = {}
    event_metrics = {}
    threshold_records = []

    # 绝对阈值
    if absolute_thresholds:
        for thr in absolute_thresholds:
            pm = compute_point_metrics(y_true, y_pred_median, threshold=thr)
            pk = compute_peak_metrics(y_true, y_pred_median, y_pred_q, thr, quantiles)
            ev = compute_event_metrics(y_true, y_pred_median, thr)
            cls = compute_classification_metrics(y_true, peak_logits, thr)

            key = f"abs_{thr:.2f}"
            point_metrics[key] = pm
            peak_metrics[key] = pk
            cls_metrics[key] = cls
            event_metrics[key] = ev
            threshold_records.append({
                "threshold_type": "absolute",
                "threshold_name": f"{thr:.2f}A",
                "threshold_value": float(thr),
                "point": pm,
                "peak": pk,
                "classification": cls,
                "event": ev,
            })

    # 分位数阈值
    for q_name, q_val in quantile_threshold_values.items():
        pm = compute_point_metrics(y_true, y_pred_median, threshold=q_val)
        pk = compute_peak_metrics(y_true, y_pred_median, y_pred_q, q_val, quantiles)
        ev = compute_event_metrics(y_true, y_pred_median, q_val)
        cls = compute_classification_metrics(y_true, peak_logits, q_val)

        key = f"quantile_{q_name}"
        point_metrics[key] = pm
        peak_metrics[key] = pk
        cls_metrics[key] = cls
        event_metrics[key] = ev
        threshold_records.append({
            "threshold_type": "quantile",
            "threshold_name": q_name,
            "threshold_value": float(q_val),
            "point": pm,
            "peak": pk,
            "classification": cls,
            "event": ev,
        })

    interval_metrics = {}
    for alpha in PI_ALPHA_LIST:
        conf = int((1 - alpha) * 100)
        interval_metrics[f"global_{conf}"] = compute_interval_metrics(
            y_true, y_pred_q, quantiles, alpha=alpha
        )

    top_peaks = get_top_peaks(y_true, y_pred_median, n=10)

    return {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "target_name": target_name,
        "horizon": int(horizon),
        "point_metrics": point_metrics,
        "peak": peak_metrics,
        "classification": cls_metrics,
        "event_metrics": event_metrics,
        "interval": interval_metrics,
        "threshold_records": threshold_records,
        "quantile_threshold_values": quantile_threshold_values,
        "quantile_threshold_stats": quantile_threshold_stats,
        "top_peaks": top_peaks,
        "global_mae": pm_all["mae"],
        "global_rmse": pm_all["rmse"],
        "global_corr": float(np.corrcoef(y_true, y_pred_median)[0, 1]),
        "y_true": y_true,
        "y_true_plot": y_true_plot,
        "y_pred_quantiles": y_pred_q,
        "y_pred_median": y_pred_median,
        "peak_logits": peak_logits,
    }


def save_evaluation_report(results, save_name="evaluation_report.csv", save_path: Optional[str] = None):
    rows = []
    model_name = results.get("model_name", "未知")
    target_name = results.get("target_name", "target")
    horizon = results.get("horizon", -1)

    rows.append({
        "模型": model_name, "目标": target_name, "H": horizon,
        "类别": "全局", "指标": "MAE", "值": f"{results['global_mae']:.4f}",
    })
    rows.append({
        "模型": model_name, "目标": target_name, "H": horizon,
        "类别": "全局", "指标": "RMSE", "值": f"{results['global_rmse']:.4f}",
    })

    for q_name, q_val in results.get("quantile_threshold_values", {}).items():
        rows.append({
            "模型": model_name, "目标": target_name, "H": horizon,
            "类别": "分位数阈值", "指标": q_name, "值": f"{q_val:.6f}",
        })

    for item in results.get("quantile_threshold_stats", []):
        rows.append({
            "模型": model_name,
            "目标": target_name,
            "H": horizon,
            "类别": f"quantile_candidate:{item.get('label', '')}",
            "指标": "threshold_value",
            "值": f"{item.get('threshold_value', np.nan)}",
        })
        rows.append({
            "模型": model_name,
            "目标": target_name,
            "H": horizon,
            "类别": f"quantile_candidate:{item.get('label', '')}",
            "指标": "event_count",
            "值": f"{item.get('event_count', np.nan)}",
        })
        rows.append({
            "模型": model_name,
            "目标": target_name,
            "H": horizon,
            "类别": f"quantile_candidate:{item.get('label', '')}",
            "指标": "event_ratio",
            "值": f"{item.get('event_ratio', np.nan)}",
        })
        rows.append({
            "模型": model_name,
            "目标": target_name,
            "H": horizon,
            "类别": f"quantile_candidate:{item.get('label', '')}",
            "指标": "selected",
            "值": f"{item.get('selected', False)}",
        })

    for rec in results.get("threshold_records", []):
        th_type = rec["threshold_type"]
        th_name = rec["threshold_name"]
        th_val = rec["threshold_value"]
        pm = rec["point"]
        pk = rec["peak"]
        ev = rec["event"]
        cls = rec["classification"]

        for metric_name, metric_val in {
            "threshold_value": th_val,
            "point_mae": pm.get("mae", np.nan),
            "point_rmse": pm.get("rmse", np.nan),
            "point_mape": pm.get("mape", np.nan),
            "peak_capture": pk.get("capture", np.nan),
            "peak_mae": pk.get("peak_mae", np.nan),
            "peak_mape": pk.get("peak_mape", np.nan),
            "event_hit_rate": ev.get("event_hit_rate", np.nan),
            "peak_time_mae_idx": ev.get("peak_time_mae_idx", np.nan),
            "peak_value_mae": ev.get("peak_value_mae", np.nan),
            "jump_f1": ev.get("jump_f1", np.nan),
            "cls_AUC": cls.get("AUC", np.nan),
            "cls_F1": cls.get("F1", np.nan),
        }.items():
            rows.append({
                "模型": model_name,
                "目标": target_name,
                "H": horizon,
                "类别": f"{th_type}:{th_name}",
                "指标": metric_name,
                "值": f"{metric_val}",
            })

    for key, iv in results.get("interval", {}).items():
        for m in ["picp", "mpiw", "winkler_score", "ace"]:
            rows.append({
                "模型": model_name,
                "目标": target_name,
                "H": horizon,
                "类别": f"区间:{key}",
                "指标": m,
                "值": f"{iv.get(m, np.nan)}",
            })

    for p in results.get("top_peaks", []):
        rows.append({
            "模型": model_name,
            "目标": target_name,
            "H": horizon,
            "类别": f"TOP-{p['rank']}",
            "指标": "真实/预测/误差%",
            "值": f"{p['true']:.4f}/{p['pred']:.4f}/{p['error_pct']:+.2f}%",
        })

    out = save_path or os.path.join(REPORT_DIR, save_name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[评估报告] 已保存: {out}")
    return out


def save_comparison_report(all_results: Dict[int, Dict], save_name="comparison_report.csv", save_path=None):
    rows = []
    for key, res in all_results.items():
        rows.append({
            "exp_key": key,
            "model_name": res.get("model_name", ""),
            "target_name": res.get("target_name", ""),
            "horizon": res.get("horizon", -1),
            "global_mae": res.get("global_mae", np.nan),
            "global_rmse": res.get("global_rmse", np.nan),
            "global_corr": res.get("global_corr", np.nan),
        })
    out = save_path or os.path.join(REPORT_DIR, save_name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[对比报告] 已保存: {out}")
    return out
