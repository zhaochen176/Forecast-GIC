from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate_contiguous_event_metrics import _event_metrics, _extract_intervals


DEFAULT_GROUPS = [
    "solar_lag30_source_window_plus_time",
    "solar_lag45_source_window_plus_time",
    "solar_lag60_source_window_plus_time",
    "solar_lag90_source_window_plus_time",
    "solar_source_windows_plus_time",
]

GROUP_LABELS = {
    "solar_lag0_source_window_plus_time": "lag 0",
    "solar_lag30_source_window_plus_time": "lag 30",
    "solar_lag45_source_window_plus_time": "lag 45",
    "solar_lag60_source_window_plus_time": "lag 60",
    "solar_lag90_source_window_plus_time": "lag 90",
    "solar_source_windows_plus_time": "lag all",
}

GROUP_MARKERS = {
    "solar_lag0_source_window_plus_time": "X",
    "solar_lag30_source_window_plus_time": "o",
    "solar_lag45_source_window_plus_time": "s",
    "solar_lag60_source_window_plus_time": "^",
    "solar_lag90_source_window_plus_time": "D",
    "solar_source_windows_plus_time": "P",
}


def _read_prediction(path: str) -> pd.DataFrame:
    pred = pd.read_csv(path)
    pred["datetime"] = pd.to_datetime(pred["datetime"], errors="coerce")
    return pred.dropna(subset=["datetime"]).set_index("datetime").sort_index()


def _threshold_grid(start: float, stop: float, step: float) -> np.ndarray:
    step = max(float(step), 1e-4)
    grid = np.arange(float(start), float(stop) + 0.5 * step, step)
    return np.unique(np.clip(grid, 0.0, 0.999))


def build_tradeoff_table(
    experiment_dir: str,
    groups: Iterable[str],
    thresholds: Iterable[float],
    probability_thresholds: Iterable[float],
    min_alarm_duration_min: int,
    merge_gap_min: int,
    match_tolerance_min: int,
) -> pd.DataFrame:
    cache_dir = os.path.join(experiment_dir, "prediction_cache")
    rows: List[Dict[str, object]] = []
    for group in groups:
        pred_path = os.path.join(cache_dir, f"{group}_predictions.csv")
        if not os.path.exists(pred_path):
            print(f"[Tradeoff] Skip missing cache: {pred_path}")
            continue
        pred = _read_prediction(pred_path)
        time_index = pd.DatetimeIndex(pred.index)
        for gic_threshold in thresholds:
            score_col = f"prob_ge_{float(gic_threshold):g}A"
            if score_col not in pred.columns:
                print(f"[Tradeoff] Skip missing score column {score_col}: {pred_path}")
                continue
            y_true_binary = pd.to_numeric(pred["y_true"], errors="coerce").to_numpy(dtype=float) >= float(gic_threshold)
            true_events = _extract_intervals(time_index, y_true_binary, min_duration_min=1, merge_gap_min=0)
            score = pd.to_numeric(pred[score_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            for alarm_threshold in probability_thresholds:
                raw_alarm = score >= float(alarm_threshold)
                pred_events = _extract_intervals(
                    time_index,
                    raw_alarm,
                    min_duration_min=min_alarm_duration_min,
                    merge_gap_min=merge_gap_min,
                )
                event_metric, _, _ = _event_metrics(
                    true_events=true_events,
                    pred_events=pred_events,
                    tolerance_min=match_tolerance_min,
                )
                rows.append(
                    {
                        "feature_group": group,
                        "threshold_A": float(gic_threshold),
                        "score_column": score_col,
                        "probability_threshold": float(alarm_threshold),
                        "min_alarm_duration_min": int(min_alarm_duration_min),
                        "merge_gap_min": int(merge_gap_min),
                        "match_tolerance_min": int(match_tolerance_min),
                        **event_metric,
                    }
                )
    return pd.DataFrame(rows)


def select_points(
    table: pd.DataFrame,
    max_far: float,
) -> pd.DataFrame:
    selected = []
    for (group, threshold), sub in table.groupby(["feature_group", "threshold_A"]):
        sub = sub.copy()
        row = sub.sort_values(
            ["event_FAR", "event_CSI", "event_F1", "event_POD"],
            ascending=[True, False, False, False],
        ).iloc[0].copy()
        row["is_feasible"] = bool(float(row["event_FAR"]) <= float(max_far))
        row["selected_reason"] = f"uniform strategy: min event_FAR; feasible means event_FAR<={max_far:g}"
        selected.append(row)
    return pd.DataFrame(selected)


def _axis_limits(values: pd.Series, default_max: float) -> Tuple[float, float]:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return 0.0, default_max
    upper = min(max(default_max, float(np.nanquantile(arr, 0.99)) * 1.12), 1.0)
    return 0.0, upper


def plot_tradeoff(table: pd.DataFrame, selected: pd.DataFrame, out_path: str, thresholds: Iterable[float]) -> None:
    thresholds = [float(x) for x in thresholds]
    nrows = 2 if len(thresholds) > 2 else 1
    ncols = int(np.ceil(len(thresholds) / nrows))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5.8 * nrows), squeeze=False)
    all_handles = {}
    for ax, threshold in zip(axes.ravel(), thresholds):
        sub_thr = table[table["threshold_A"].eq(float(threshold))]
        sel_thr = selected[selected["threshold_A"].eq(float(threshold))]
        xlim = _axis_limits(sub_thr["event_FAR"], default_max=0.8)
        for group, sub in sub_thr.groupby("feature_group"):
            label = GROUP_LABELS.get(group, group)
            marker = GROUP_MARKERS.get(group, "o")
            sc = ax.scatter(
                sub["event_FAR"],
                sub["event_POD"],
                c=sub["event_CSI"],
                cmap="viridis",
                marker=marker,
                s=42,
                alpha=0.48,
                edgecolors="none",
                label=label,
            )
            all_handles[label] = plt.Line2D(
                [0],
                [0],
                marker=marker,
                linestyle="None",
                markersize=8,
                color="gray",
                label=label,
            )
        if not sel_thr.empty:
            ax.scatter(
                sel_thr["event_FAR"],
                sel_thr["event_POD"],
                marker="*",
                s=460,
                facecolors="none",
                edgecolors="red",
                linewidths=2.0,
                label="final selected",
                zorder=5,
            )
            for _, row in sel_thr.iterrows():
                ax.text(
                    float(row["event_FAR"]) + 0.01,
                    float(row["event_POD"]) + 0.01,
                    f"{float(row['probability_threshold']):.2f}",
                    color="red",
                    fontsize=9,
                    weight="bold",
                )
        ax.set_title(f"{threshold:g}A")
        ax.axvline(0.10, color="red", linestyle="--", linewidth=1.0, alpha=0.65)
        ax.set_xlabel("event FAR")
        ax.set_ylabel("event POD")
        ax.set_xlim(*xlim)
        ax.set_ylim(0, 1.03)
        ax.grid(True, alpha=0.28)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("event CSI")
    for ax in axes.ravel()[len(thresholds):]:
        ax.axis("off")
    handles = list(all_handles.values()) + [
        plt.Line2D([0], [0], marker="*", linestyle="None", markersize=14, markeredgewidth=1.8,
                   markerfacecolor="none", markeredgecolor="red", color="red", label="final selected")
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 7), frameon=False)
    fig.suptitle("Low-false-alarm POD-FAR tradeoff: candidate thresholds colored by event CSI", fontsize=16)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot POD-FAR-CSI tradeoff for event alarm probability thresholds.")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--thresholds", nargs="*", type=float, default=[3.0, 5.0, 10.0, 20.0])
    parser.add_argument("--groups", nargs="*", default=DEFAULT_GROUPS)
    parser.add_argument("--probability-threshold-start", type=float, default=0.05)
    parser.add_argument("--probability-threshold-stop", type=float, default=0.95)
    parser.add_argument("--probability-threshold-step", type=float, default=0.01)
    parser.add_argument("--select-max-far", type=float, default=0.10)
    parser.add_argument("--min-alarm-duration-min", type=int, default=3)
    parser.add_argument("--merge-gap-min", type=int, default=10)
    parser.add_argument("--match-tolerance-min", type=int, default=10)
    args = parser.parse_args()

    experiment_dir = os.path.abspath(args.experiment_dir)
    out_dir = os.path.join(experiment_dir, "contiguous_event_eval", "threshold_tradeoff")
    os.makedirs(out_dir, exist_ok=True)
    grid = _threshold_grid(
        args.probability_threshold_start,
        args.probability_threshold_stop,
        args.probability_threshold_step,
    )
    table = build_tradeoff_table(
        experiment_dir=experiment_dir,
        groups=args.groups,
        thresholds=args.thresholds,
        probability_thresholds=grid,
        min_alarm_duration_min=args.min_alarm_duration_min,
        merge_gap_min=args.merge_gap_min,
        match_tolerance_min=args.match_tolerance_min,
    )
    selected = select_points(
        table,
        max_far=args.select_max_far,
    )
    table_path = os.path.join(out_dir, "event_probability_threshold_tradeoff.csv")
    selected_path = os.path.join(out_dir, "selected_event_probability_thresholds.csv")
    fig_path = os.path.join(out_dir, "pod_far_csi_tradeoff_selected_points.png")
    table.to_csv(table_path, index=False, encoding="utf-8-sig")
    selected.to_csv(selected_path, index=False, encoding="utf-8-sig")
    plot_tradeoff(table, selected, fig_path, args.thresholds)
    print(f"[Tradeoff] Saved data: {table_path}")
    print(f"[Tradeoff] Saved selected: {selected_path}")
    print(f"[Tradeoff] Saved figure: {fig_path}")
    print(selected.sort_values(["threshold_A", "feature_group"])[
        [
            "feature_group",
            "threshold_A",
            "probability_threshold",
            "event_POD",
            "event_FAR",
            "event_CSI",
            "event_F1",
            "event_Bias",
        ]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
