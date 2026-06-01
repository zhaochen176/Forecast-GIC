"""
GIC 预测项目 - 绘图模块
"""
import os
import math
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURE_DIR, QUANTILES

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def _out_dir(output_dir: Optional[str]) -> str:
    d = output_dir or FIGURE_DIR
    os.makedirs(d, exist_ok=True)
    return d


def _split_segments_by_time_gap(
    x,
    gap_factor: float = 12.0,
    min_gap: str = "30min",
):
    """
    Split a datetime-like x-axis into contiguous segments.
    Used to avoid drawing misleading lines across large time gaps
    (for example, concatenated event windows from different dates).
    """
    n = len(x)
    if n <= 1:
        return [(0, n)]
    if not isinstance(x, pd.DatetimeIndex):
        return [(0, n)]

    diffs = np.diff(x.asi8)
    positive = diffs[diffs > 0]
    if len(positive) == 0:
        return [(0, n)]

    typical_step = int(np.median(positive))
    min_gap_ns = int(pd.Timedelta(min_gap).value)
    gap_threshold = max(int(typical_step * float(gap_factor)), min_gap_ns)
    cut_points = np.where(diffs > gap_threshold)[0]
    if len(cut_points) == 0:
        return [(0, n)]

    segments = []
    start = 0
    for cp in cut_points:
        end = int(cp) + 1
        if end > start:
            segments.append((start, end))
        start = end
    if start < n:
        segments.append((start, n))
    return segments


def plot_training_history(history: Dict, save_name: str = "training_history.png", output_dir: Optional[str] = None):
    if not history.get("train_losses"):
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(history["train_losses"]) + 1)
    ax.plot(epochs, history["train_losses"], "b-", label="Train Loss")
    if history.get("val_losses"):
        ax.plot(epochs, history["val_losses"], "r-", label="Val Loss")
    if history.get("best_epoch", 0) > 0:
        ax.axvline(history["best_epoch"], color="gray", linestyle="--", alpha=0.6)
    ax.set_title("Training History")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_path = os.path.join(_out_dir(output_dir), save_name)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_with_intervals(
    y_true,
    y_pred_quantiles,
    y_pred_point=None,
    quantiles=QUANTILES,
    time_index=None,
    start_idx=0,
    length=1440,
    title="Prediction With Intervals",
    save_name="prediction_intervals.png",
    output_dir: Optional[str] = None,
):
    end_idx = min(start_idx + length, len(y_true))
    y_t = y_true[start_idx:end_idx]
    y_q = y_pred_quantiles[start_idx:end_idx]
    y_p = None if y_pred_point is None else y_pred_point[start_idx:end_idx]
    x = time_index[start_idx:end_idx] if time_index is not None else np.arange(start_idx, end_idx)

    q05i = min(range(len(quantiles)), key=lambda i: abs(quantiles[i] - 0.05))
    q95i = min(range(len(quantiles)), key=lambda i: abs(quantiles[i] - 0.95))
    q25i = min(range(len(quantiles)), key=lambda i: abs(quantiles[i] - 0.25))
    q75i = min(range(len(quantiles)), key=lambda i: abs(quantiles[i] - 0.75))
    q50i = min(range(len(quantiles)), key=lambda i: abs(quantiles[i] - 0.5))

    fig, ax = plt.subplots(figsize=(16, 6))
    segments = _split_segments_by_time_gap(x) if time_index is not None else [(0, len(y_t))]
    for i, (s, e) in enumerate(segments):
        seg_x = x[s:e]
        seg_t = y_t[s:e]
        seg_q = y_q[s:e]
        seg_p = None if y_p is None else y_p[s:e]

        ax.fill_between(
            seg_x,
            seg_q[:, q05i],
            seg_q[:, q95i],
            alpha=0.15,
            color="#4c78a8",
            label="90% PI" if i == 0 else None,
        )
        ax.fill_between(
            seg_x,
            seg_q[:, q25i],
            seg_q[:, q75i],
            alpha=0.28,
            color="#4c78a8",
            label="50% PI" if i == 0 else None,
        )
        if seg_p is not None:
            ax.plot(
                seg_x,
                seg_p,
                "b-",
                linewidth=1.2,
                label="Final Prediction" if i == 0 else None,
            )
            ax.plot(
                seg_x,
                seg_q[:, q50i],
                color="#1f77b4",
                linewidth=0.8,
                alpha=0.5,
                linestyle="--",
                label="Q50" if i == 0 else None,
            )
        else:
            ax.plot(
                seg_x,
                seg_q[:, q50i],
                "b-",
                linewidth=1.2,
                label="Median" if i == 0 else None,
            )
        ax.plot(seg_x, seg_t, "r-", linewidth=1.0, label="True" if i == 0 else None)

    ax.set_title(title)
    ax.set_ylabel("Amplitude (A)")
    ax.set_xlabel("Time" if time_index is not None else "Sample Index")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    if time_index is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        plt.xticks(rotation=30)
    save_path = os.path.join(_out_dir(output_dir), save_name)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_comparison_full(
    y_true,
    y_pred,
    time_index=None,
    title="Full Test Comparison",
    save_name="prediction_full_test.png",
    output_dir: Optional[str] = None,
):
    n = len(y_true)
    if n == 0:
        return
    x = time_index if time_index is not None else np.arange(n)

    fig, ax = plt.subplots(figsize=(24, 7))
    segments = _split_segments_by_time_gap(x) if time_index is not None else [(0, n)]
    for i, (s, e) in enumerate(segments):
        ax.plot(
            x[s:e],
            y_true[s:e],
            color="#d62728",
            linewidth=0.7,
            alpha=0.9,
            label="True" if i == 0 else None,
        )
        ax.plot(
            x[s:e],
            y_pred[s:e],
            color="#1f77b4",
            linewidth=0.7,
            alpha=0.9,
            label="Final Prediction" if i == 0 else None,
        )

    if time_index is not None and len(segments) > 1:
        ax.set_title(f"{title} (event-only timeline, discontinuous)")
    else:
        ax.set_title(title)
    ax.set_ylabel("Amplitude")
    ax.set_xlabel("Time" if time_index is not None else "Sample Index")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    if time_index is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=25)

    save_path = os.path.join(_out_dir(output_dir), save_name)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_comparison_all_events(
    y_true,
    y_pred,
    time_index=None,
    title="Event-wise Prediction Comparison",
    save_name_prefix="prediction_eventwise",
    output_dir: Optional[str] = None,
    ncols: int = 4,
    max_events_per_page: int = 24,
):
    """
    Plot true/pred comparison for every detected event segment.
    Segments are split by large datetime gaps when time_index is provided.
    """
    n = len(y_true)
    if n == 0:
        return

    x = time_index if time_index is not None else np.arange(n)
    segments = _split_segments_by_time_gap(x) if time_index is not None else [(0, n)]
    if len(segments) == 0:
        return

    ncols = max(int(ncols), 1)
    max_events_per_page = max(int(max_events_per_page), 1)
    pages = int(math.ceil(len(segments) / max_events_per_page))

    out_dir = _out_dir(output_dir)
    for page_idx in range(pages):
        start_event = page_idx * max_events_per_page
        end_event = min((page_idx + 1) * max_events_per_page, len(segments))
        page_segments = segments[start_event:end_event]
        n_events_page = len(page_segments)
        nrows = int(math.ceil(n_events_page / ncols))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.8 * ncols, 2.8 * nrows),
            sharey=False,
        )
        axes = np.atleast_1d(axes).reshape(-1)

        for ax_i, (s, e) in enumerate(page_segments):
            ax = axes[ax_i]
            seg_true = y_true[s:e]
            seg_pred = y_pred[s:e]
            if len(seg_true) == 0:
                ax.set_axis_off()
                continue

            if time_index is not None:
                seg_time = x[s:e]
                rel_min = (
                    (seg_time - seg_time[0]).total_seconds().astype(np.float64) / 60.0
                )
                x_plot = rel_min
                t0 = pd.Timestamp(seg_time[0]).strftime("%Y-%m-%d %H:%M")
                t1 = pd.Timestamp(seg_time[-1]).strftime("%Y-%m-%d %H:%M")
                evt_title = f"E{start_event + ax_i + 1} | {t0} ~ {t1}"
                ax.set_xlabel("Minutes From Event Start")
            else:
                x_plot = np.arange(len(seg_true))
                evt_title = f"E{start_event + ax_i + 1} | n={len(seg_true)}"
                ax.set_xlabel("Sample Index")

            ax.plot(x_plot, seg_true, color="#d62728", linewidth=1.0, label="True")
            ax.plot(x_plot, seg_pred, color="#1f77b4", linewidth=1.0, label="Pred")
            ax.set_title(evt_title, fontsize=9)
            ax.grid(True, alpha=0.25)

            if ax_i % ncols == 0:
                ax.set_ylabel("Amplitude")

        for ax in axes[n_events_page:]:
            ax.set_axis_off()

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", frameon=False)
        fig.suptitle(
            f"{title} | page {page_idx + 1}/{pages} | events {start_event + 1}-{end_event}",
            fontsize=12,
            y=0.995,
        )

        if pages == 1:
            save_name = f"{save_name_prefix}.png"
        else:
            save_name = f"{save_name_prefix}_p{page_idx + 1:02d}.png"
        save_path = os.path.join(out_dir, save_name)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_peak_comparison(y_true, y_pred_median, threshold=2.0, save_name="peak_comparison.png", output_dir=None):
    pm = y_true >= threshold
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(y_true, y_pred_median, s=2, alpha=0.1, color="#4c78a8")
    if pm.any():
        axes[0].scatter(y_true[pm], y_pred_median[pm], s=8, alpha=0.6, color="#e45756", label=f">={threshold}")
    mx = max(float(np.max(y_true)), float(np.max(y_pred_median)))
    axes[0].plot([0, mx], [0, mx], "k--", alpha=0.5)
    axes[0].set_title("All Samples")
    axes[0].set_xlabel("True")
    axes[0].set_ylabel("Pred")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left")

    if pm.any():
        axes[1].scatter(y_true[pm], y_pred_median[pm], s=8, alpha=0.6, color="#e45756")
    axes[1].plot([threshold, mx], [threshold, mx], "k--", alpha=0.5)
    axes[1].set_title("Peak Region")
    axes[1].set_xlabel("True")
    axes[1].set_ylabel("Pred")
    axes[1].grid(True, alpha=0.3)

    save_path = os.path.join(_out_dir(output_dir), save_name)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_peak_events(
    y_true,
    y_pred_quantiles,
    quantiles=QUANTILES,
    threshold=5.0,
    context_window=120,
    max_events=8,
    save_name="peak_events_detail.png",
    output_dir: Optional[str] = None,
):
    peak_indices = np.where(y_true >= threshold)[0]
    if len(peak_indices) == 0:
        return

    events = []
    event_start = peak_indices[0]
    for i in range(1, len(peak_indices)):
        if peak_indices[i] - peak_indices[i - 1] > context_window:
            peak_idx = event_start + np.argmax(y_true[event_start:peak_indices[i - 1] + 1])
            events.append(peak_idx)
            event_start = peak_indices[i]
    events.append(event_start + np.argmax(y_true[event_start:peak_indices[-1] + 1]))
    events.sort(key=lambda x: y_true[x], reverse=True)
    events = events[:max_events]

    q05i = min(range(len(quantiles)), key=lambda i: abs(quantiles[i] - 0.05))
    q95i = min(range(len(quantiles)), key=lambda i: abs(quantiles[i] - 0.95))
    q50i = min(range(len(quantiles)), key=lambda i: abs(quantiles[i] - 0.5))

    fig, axes = plt.subplots(len(events), 1, figsize=(16, 3.5 * len(events)))
    if len(events) == 1:
        axes = [axes]
    for i, peak_idx in enumerate(events):
        s = max(0, peak_idx - context_window)
        e = min(len(y_true), peak_idx + context_window)
        x = np.arange(s, e)
        ax = axes[i]
        ax.fill_between(x, y_pred_quantiles[s:e, q05i], y_pred_quantiles[s:e, q95i], alpha=0.2, color="#72b7b2")
        ax.plot(x, y_pred_quantiles[s:e, q50i], "b-", linewidth=1.2, label="Pred Median")
        ax.plot(x, y_true[s:e], "r-", linewidth=1.2, label="True")
        ax.axvline(peak_idx, color="k", linestyle="--", alpha=0.4)
        ax.set_title(f"Event {i + 1} | true peak={y_true[peak_idx]:.2f}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Sample Index")
    save_path = os.path.join(_out_dir(output_dir), save_name)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_quantile_calibration(y_true, y_pred_quantiles, quantiles=QUANTILES, save_name="quantile_calibration.png", output_dir=None):
    actual = [(y_true <= y_pred_quantiles[:, i]).mean() for i, _ in enumerate(quantiles)]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.plot(quantiles, actual, "o-", color="#4c78a8")
    ax.set_xlabel("Nominal Quantile")
    ax.set_ylabel("Actual Coverage")
    ax.set_title("Quantile Calibration")
    ax.grid(True, alpha=0.3)
    save_path = os.path.join(_out_dir(output_dir), save_name)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_distribution(y_true, y_pred_median, save_name="error_distribution.png", output_dir=None):
    err = y_true - y_pred_median
    abs_err = np.abs(err)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(err, bins=200, density=True, color="#4c78a8", alpha=0.8)
    axes[0].axvline(0, color="k", linestyle="--", alpha=0.5)
    axes[0].set_title("Error Distribution")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(y_true, abs_err, s=2, alpha=0.1, color="#f58518")
    axes[1].set_title("|Error| vs True")
    axes[1].set_xlabel("True")
    axes[1].set_ylabel("|Error|")
    axes[1].grid(True, alpha=0.3)

    bins = np.quantile(y_true, [0, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
    labels = [f"{bins[i]:.1f}-{bins[i + 1]:.1f}" for i in range(len(bins) - 1)]
    groups = pd.cut(y_true, bins=bins, labels=labels, include_lowest=True)
    stat = []
    for label in labels:
        vals = abs_err[groups == label]
        if len(vals) > 0:
            stat.append(vals)
    if stat:
        axes[2].boxplot(stat, labels=[labels[i] for i in range(len(stat))], showfliers=False)
        axes[2].tick_params(axis="x", rotation=30)
    axes[2].set_title("|Error| by True Bin")
    axes[2].grid(True, alpha=0.3)

    save_path = os.path.join(_out_dir(output_dir), save_name)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance_shap(
    model,
    sample_data,
    feature_names,
    device,
    n_samples=500,
    save_name="shap_feature_importance.png",
    output_dir: Optional[str] = None,
):
    try:
        import shap
    except Exception:
        print("[可视化] shap 未安装，跳过 SHAP 图。")
        return None

    model.eval()
    if len(sample_data) > n_samples:
        idx = np.random.choice(len(sample_data), n_samples, replace=False)
    else:
        idx = np.arange(len(sample_data))

    X = []
    for i in idx:
        x, _, _ = sample_data[i]
        X.append(x.numpy().mean(axis=0))
    X = np.asarray(X)

    import torch as _torch

    def predict_fn(x_np):
        x_tensor = _torch.tensor(x_np, dtype=_torch.float32)
        x_tensor = x_tensor.unsqueeze(1).expand(-1, sample_data.seq_len, -1).to(device)
        with _torch.no_grad():
            pred, _, _, _ = model(x_tensor)
        m_idx = model.quantiles.index(0.5)
        return pred[:, m_idx].cpu().numpy()

    bg = X[: min(100, len(X))]
    explainer = shap.KernelExplainer(predict_fn, bg)
    shap_values = explainer.shap_values(X[: min(200, len(X))])
    mean_abs = np.abs(shap_values).mean(axis=0)
    imp = pd.DataFrame({"feature": feature_names, "importance": mean_abs}).sort_values("importance", ascending=False)

    top = imp.head(min(30, len(imp)))
    fig, ax = plt.subplots(figsize=(10, max(8, 0.3 * len(top))))
    ax.barh(np.arange(len(top)), top["importance"].values, color="#4c78a8")
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top["feature"].values, fontsize=8)
    ax.invert_yaxis()
    ax.set_title("SHAP Feature Importance")
    ax.set_xlabel("Mean |SHAP|")
    save_path = os.path.join(_out_dir(output_dir), save_name)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return imp


def plot_target_distribution_triplet(
    df: pd.DataFrame,
    target_cols: List[str],
    save_name: str = "target_distribution_triplet.png",
    output_dir: Optional[str] = None,
):
    titles = ["GIC1 (Vykhodnoy)", "GIC2 (Loukhi)", "dBH/dt (SOD)"]
    subplot_tags = ["(a)", "(b)", "(c)"]
    out_dir = _out_dir(output_dir)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))
    hist_rows = []
    summary_rows = []

    print(f"[Plot] Target distribution columns: {target_cols}")
    for i, col in enumerate(target_cols):
        if col not in df.columns:
            axes[i].text(0.5, 0.5, f"Missing: {col}", ha="center", va="center")
            axes[i].set_axis_off()
            continue
        vals = df[col].dropna().to_numpy(dtype=np.float32)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            axes[i].text(0.5, 0.5, f"Empty: {col}", ha="center", va="center")
            axes[i].set_axis_off()
            continue

        axes[i].hist(vals, bins=120, color="#4c78a8", alpha=0.85)
        axes[i].set_yscale("log")
        axes[i].set_title(titles[i] if i < len(titles) else col, fontsize=13)
        if i in (0, 1):
            axes[i].set_xlabel("GIC (A)", fontsize=12)
        else:
            axes[i].set_xlabel("dBH/dt (nT/min)", fontsize=12)
        axes[i].set_ylabel("Count (log scale)")
        axes[i].grid(True, alpha=0.25)
        if i < len(subplot_tags):
            axes[i].text(
                0.03,
                0.96,
                subplot_tags[i],
                transform=axes[i].transAxes,
                ha="left",
                va="top",
                fontsize=14,
                fontweight="bold",
            )

        counts, edges = np.histogram(vals, bins=120)
        for j in range(len(counts)):
            hist_rows.append({
                "target": col,
                "bin_left": float(edges[j]),
                "bin_right": float(edges[j + 1]),
                "count": int(counts[j]),
            })

        summary_rows.append({
            "target": col,
            "n": int(len(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "q50": float(np.quantile(vals, 0.50)),
            "q75": float(np.quantile(vals, 0.75)),
            "q80": float(np.quantile(vals, 0.80)),
            "q85": float(np.quantile(vals, 0.85)),
            "q90": float(np.quantile(vals, 0.90)),
            "q93": float(np.quantile(vals, 0.93)),
            "q95": float(np.quantile(vals, 0.95)),
            "q97": float(np.quantile(vals, 0.97)),
            "q99": float(np.quantile(vals, 0.99)),
            "zero_ratio": float(np.mean(vals == 0)),
            "positive_ratio": float(np.mean(vals > 0)),
            "negative_ratio": float(np.mean(vals < 0)),
        })

    save_path = os.path.join(out_dir, save_name)
    fig.savefig(save_path, dpi=320, bbox_inches="tight")
    plt.close(fig)

    save_stem, _ = os.path.splitext(save_name)
    hist_path = os.path.join(out_dir, f"{save_stem}_histogram.csv")
    summary_path = os.path.join(out_dir, f"{save_stem}_summary.csv")
    if hist_rows:
        pd.DataFrame(hist_rows).to_csv(hist_path, index=False, encoding="utf-8-sig")
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[Plot] Saved distribution figure: {save_path}")
    print(f"[Plot] Saved histogram csv: {hist_path}")
    print(f"[Plot] Saved summary csv: {summary_path}")


def generate_all_plots(
    results: Dict,
    history: Dict,
    quantiles: List[float] = QUANTILES,
    time_index=None,
    output_dir: Optional[str] = None,
):
    y_true = results["y_true"]
    y_true_plot = results.get("y_true_plot", y_true)
    y_pred_q = results["y_pred_quantiles"]
    y_pred_median = results["y_pred_median"]

    plot_training_history(history, output_dir=output_dir)
    plot_peak_comparison(y_true, y_pred_median, threshold=5.0, output_dir=output_dir)
    plot_peak_events(y_true, y_pred_q, quantiles=quantiles, threshold=5.0, output_dir=output_dir)
    plot_quantile_calibration(y_true, y_pred_q, quantiles=quantiles, output_dir=output_dir)
    plot_error_distribution(y_true, y_pred_median, output_dir=output_dir)
    plot_prediction_comparison_full(
        y_true_plot,
        y_pred_median,
        time_index=time_index,
        title=f"{results.get('target_name', 'Target')} | H={results.get('horizon', '')} | Full Test",
        save_name="prediction_full_test.png",
        output_dir=output_dir,
    )
    plot_prediction_comparison_all_events(
        y_true_plot,
        y_pred_median,
        time_index=time_index,
        title=f"{results.get('target_name', 'Target')} | H={results.get('horizon', '')}",
        save_name_prefix="prediction_eventwise_all",
        output_dir=output_dir,
    )

    if len(y_true_plot):
        safe_true = np.nan_to_num(np.abs(y_true_plot), nan=-1.0, posinf=-1.0, neginf=-1.0)
        peak_idx = int(np.argmax(safe_true))
    else:
        peak_idx = 0
    plot_prediction_with_intervals(
        y_true_plot,
        y_pred_q,
        y_pred_point=y_pred_median,
        quantiles=quantiles,
        time_index=time_index,
        start_idx=max(0, peak_idx - 2160),
        length=4320,
        title=f"{results.get('target_name', 'Target')} | H={results.get('horizon', '')}",
        save_name="prediction_intervals_peak.png",
        output_dir=output_dir,
    )


def generate_comparison_plots(all_results: Dict[int, Dict], output_dir: Optional[str] = None):
    if not all_results:
        return
    rows = []
    for k, v in all_results.items():
        rows.append((str(k), v.get("global_mae", np.nan), v.get("global_rmse", np.nan)))
    labels = [r[0] for r in rows]
    mae = [r[1] for r in rows]
    rmse = [r[2] for r in rows]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.35
    ax.bar(x - w / 2, mae, width=w, label="MAE", color="#4c78a8")
    ax.bar(x + w / 2, rmse, width=w, label="RMSE", color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Experiment Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    save_path = os.path.join(_out_dir(output_dir), "experiment_comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
