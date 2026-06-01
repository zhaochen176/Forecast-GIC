"""
Nonlinear dependence analysis between features and VKH GIC.

This script is intended for the paper section "2.3 Correlation analysis".
It does not train a forecasting model. By default it uses same-time features
without propagation lags, constructs a future-window VKH GIC target, and exports
both figures and the CSV data needed to redraw those figures in external
plotting software.

For a row at GIC time t, the default label is max(|GIC_VKH|[t+1 : t+H]).

Feature-set presets:
  - solar_coupling_geomag_current: solar wind + coupling + geomagnetic features, no lag
  - solar_coupling_current: solar wind + coupling features, no lag
  - lagged_corrected: legacy corrected-lag feature analysis

Examples
--------
python nonlinear_feature_gic_analysis.py --run-presets --scope event-type --event-types CME CIR --horizon 30
python nonlinear_feature_gic_analysis.py --feature-set solar_coupling_geomag_current --scope event-type --horizon 30
python nonlinear_feature_gic_analysis.py --feature-set solar_coupling_current --scope event-type --horizon 30
python nonlinear_feature_gic_analysis.py --scope fixed-events --horizon 30 --top-n 30
python nonlinear_feature_gic_analysis.py --target-mode current --scope event-type
"""
from __future__ import annotations

import argparse
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from solar_wind_lagged_ablation import build_lagged_solar_features
from src.config import (
    DATA_FILE,
    OUTPUT_DIR,
    SEED,
    TARGET_VYK_COL,
)
from src.data_loader import build_vkh_event_type_report, prepare_targets
from threshold_quantile_ablation import _add_future_window_max_label


SOLAR_RAW_BASE_COLS = [
    "Btot",
    "Bx_gse",
    "By_gse",
    "Bz_gse",
    "Vp",
    "Np_filled",
    "P_dyn_nPa",
    "Ey_mV/m",
    "Ma",
]
SOLAR_COUPLING_BASE_COLS = ["epsilon_norm", "Newell", "Borovsky"]
GEOMAG_RESPONSE_COLS = [
    "X",
    "Y",
    "Z",
    "X_pert",
    "Y_pert",
    "Z_pert",
    "H_pert",
    "dX_pert_dt",
    "dY_pert_dt",
    "dZ_pert_dt",
    "dH_pert_dt",
    "dbhdt_abs_feature",
]
TIME_COLS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
FEATURE_SET_PRESETS = {
    "solar_coupling_geomag_current": {
        "include_current_solar": True,
        "include_lagged_solar": False,
        "include_geomag": True,
        "label": "solar_coupling_geomag_current_no_lag",
    },
    "solar_coupling_current": {
        "include_current_solar": True,
        "include_lagged_solar": False,
        "include_geomag": False,
        "label": "solar_coupling_current_no_lag",
    },
    "lagged_corrected": {
        "include_current_solar": False,
        "include_lagged_solar": True,
        "include_geomag": True,
        "label": "lagged_corrected",
    },
}
DEFAULT_FEATURE_PRESETS = [
    "solar_coupling_geomag_current",
    "solar_coupling_current",
]
DEFAULT_OUTPUT_ROOT = os.path.join(
    OUTPUT_DIR,
    "不同模型预测结果",
    "nonlinear_correlation_corrected",
)


def _safe_name(text: str) -> str:
    return (
        str(text)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def _sample_frame(df: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    if n <= 0 or len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed).sort_index()


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


def select_scope(
    df: pd.DataFrame,
    scope: str,
    event_types: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, str, pd.DataFrame]:
    scope = str(scope).lower()
    if scope in {"full", "full-time", "full_time"}:
        return df, "full_time", pd.DataFrame()

    events = build_vkh_event_type_report(df)
    events = events[
        events["in_data_range"].astype(bool)
        & events["quality_keep"].astype(bool)
    ].copy()

    if scope == "fixed-events":
        label = "fixed_events"
    elif scope == "event-type":
        keep_types = set(event_types or ["CME", "CIR"])
        events = events[events["event_type"].isin(keep_types)].copy()
        label = "event_type_" + "_".join(sorted(keep_types))
    else:
        raise ValueError("scope must be one of: full, full-time, fixed-events, event-type")

    intervals = [
        (pd.Timestamp(r["start"]), pd.Timestamp(r["end"]))
        for _, r in events.iterrows()
    ]
    scoped = _rows_from_intervals(df, intervals)
    return scoped, label, events


def _standardize_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mask = np.isfinite(x)
    if not mask.all():
        med = np.nanmedian(x)
        x = np.where(mask, x, med)
    std = np.std(x)
    if std <= 1e-12:
        return np.zeros_like(x)
    return (x - np.mean(x)) / std


def distance_corr_1d(x: np.ndarray, y: np.ndarray) -> float:
    x = _standardize_1d(x)
    y = _standardize_1d(y)
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    ax = np.abs(x[:, None] - x[None, :])
    ay = np.abs(y[:, None] - y[None, :])
    ax = ax - ax.mean(axis=0, keepdims=True) - ax.mean(axis=1, keepdims=True) + ax.mean()
    ay = ay - ay.mean(axis=0, keepdims=True) - ay.mean(axis=1, keepdims=True) + ay.mean()
    dcov2 = np.mean(ax * ay)
    dvarx = np.mean(ax * ax)
    dvary = np.mean(ay * ay)
    denom = np.sqrt(max(dvarx * dvary, 0.0))
    if denom <= 1e-18:
        return 0.0
    return float(np.sqrt(max(dcov2, 0.0) / np.sqrt(denom)))


def _rbf_kernel_1d(z: np.ndarray) -> np.ndarray:
    z = _standardize_1d(z)
    d2 = (z[:, None] - z[None, :]) ** 2
    vals = d2[np.triu_indices_from(d2, k=1)]
    sigma2 = float(np.median(vals[vals > 0])) if np.any(vals > 0) else 1.0
    sigma2 = max(sigma2, 1e-6)
    return np.exp(-d2 / (2.0 * sigma2))


def hsic_rbf_1d(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    kx = _rbf_kernel_1d(x)
    ky = _rbf_kernel_1d(y)
    n = len(x)
    h = np.eye(n) - np.ones((n, n), dtype=np.float64) / n
    val = np.trace(kx @ h @ ky @ h) / max((n - 1) ** 2, 1)
    return float(max(val, 0.0))


def _normalise_mi_by_entropy(values: pd.Series, y: np.ndarray, bins: int = 64) -> pd.Series:
    finite_y = y[np.isfinite(y)]
    if len(finite_y) < 2:
        return pd.Series(np.nan, index=values.index)
    counts, _ = np.histogram(finite_y, bins=bins)
    probs = counts[counts > 0] / max(counts.sum(), 1)
    entropy_y = -float(np.sum(probs * np.log(probs)))
    if entropy_y <= 1e-12:
        return pd.Series(np.nan, index=values.index)
    return values / entropy_y


def infer_feature_group(feature: str) -> str:
    if feature in TIME_COLS:
        return "time_periodic"
    if feature in GEOMAG_RESPONSE_COLS or "pert" in feature.lower() or "dbhdt" in feature.lower():
        return "geomag_response"
    if "lag" in feature:
        if any(token in feature for token in ["Newell", "Borovsky", "epsilon", "Ey_mV/m", "Vp_Bz_south", "Bz_south", "imf_clock"]):
            return "lagged_solar_coupling"
        return "lagged_solar_raw"
    if feature in SOLAR_COUPLING_BASE_COLS:
        return "solar_coupling_current"
    if feature in SOLAR_RAW_BASE_COLS:
        return "solar_raw_current"
    return "other"


def _lag_from_feature(feature: str) -> float:
    match = re.search(r"lag(\d+)", feature)
    return float(match.group(1)) if match else np.nan


def _window_from_feature(feature: str) -> float:
    match = re.search(r"roll(\d+)", feature)
    return float(match.group(1)) if match else np.nan


def _base_from_feature(feature: str) -> str:
    base = re.sub(r"_lag\d+.*$", "", feature)
    return base


def _dedupe_existing(cols: Iterable[str], df: pd.DataFrame) -> List[str]:
    seen = set()
    out = []
    for col in cols:
        if col in df.columns and col not in seen:
            seen.add(col)
            out.append(col)
    return out


def build_analysis_table(
    data_path: str,
    target: str,
    target_mode: str,
    horizon: int,
    lags: List[int],
    rolling_windows: List[int],
    rolling_stats: List[str],
    include_current_solar: bool,
    include_lagged_solar: bool,
    include_geomag: bool,
    include_time: bool,
) -> Tuple[pd.DataFrame, List[str], str, pd.DataFrame]:
    print(f"[NonlinearCorr] Loading source data: {data_path}")
    df = pd.read_parquet(data_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = prepare_targets(df)

    solar_groups: Dict[str, List[str]] = {}
    metadata = pd.DataFrame()
    if include_lagged_solar:
        df, solar_groups, metadata = build_lagged_solar_features(
            df,
            solar_cols=SOLAR_RAW_BASE_COLS + SOLAR_COUPLING_BASE_COLS,
            lags=lags,
            rolling_windows=rolling_windows,
            rolling_stats=rolling_stats,
        )

    if target_mode == "future-max":
        df, analysis_target = _add_future_window_max_label(df, target, horizon)
    elif target_mode == "current":
        analysis_target = target
    else:
        raise ValueError("target_mode must be future-max or current")

    feature_cols: List[str] = []
    if include_current_solar:
        feature_cols.extend(SOLAR_RAW_BASE_COLS)
        feature_cols.extend(SOLAR_COUPLING_BASE_COLS)
    if include_lagged_solar:
        feature_cols.extend(solar_groups.get("solar_all_lags", []))
    if include_geomag:
        feature_cols.extend(GEOMAG_RESPONSE_COLS)
    if include_time:
        feature_cols.extend(TIME_COLS)
    feature_cols = _dedupe_existing(feature_cols, df)
    feature_cols = [c for c in feature_cols if c != analysis_target]
    return df, feature_cols, analysis_target, metadata


def compute_scores(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    mi_sample_size: int,
    kernel_sample_size: int,
    top_kernel_candidates: int,
    seed: int = SEED,
) -> pd.DataFrame:
    data = df[feature_cols + [target_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        raise RuntimeError("No finite rows available for nonlinear correlation analysis.")

    pearson = data[feature_cols].corrwith(data[target_col], method="pearson")
    spearman = data[feature_cols].corrwith(data[target_col], method="spearman")

    mi_data = _sample_frame(data, mi_sample_size, seed)
    x_mi = mi_data[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y_mi = mi_data[target_col].to_numpy(dtype=np.float32, copy=True)
    mi = mutual_info_regression(
        x_mi,
        y_mi,
        random_state=seed,
        n_neighbors=5,
    )
    mi_series = pd.Series(mi, index=feature_cols)

    report = pd.DataFrame(
        {
            "feature": feature_cols,
            "pearson": pearson.reindex(feature_cols).values,
            "abs_pearson": pearson.abs().reindex(feature_cols).values,
            "spearman": spearman.reindex(feature_cols).values,
            "abs_spearman": spearman.abs().reindex(feature_cols).values,
            "mutual_info": mi_series.reindex(feature_cols).values,
        }
    ).set_index("feature")
    report["mi_normalized_by_target_entropy"] = _normalise_mi_by_entropy(
        report["mutual_info"], y_mi
    )

    candidates = (
        report.sort_values("mutual_info", ascending=False)
        .head(int(top_kernel_candidates))
        .index.tolist()
    )
    kernel_data = _sample_frame(data[candidates + [target_col]], kernel_sample_size, seed + 17)
    y_kernel = kernel_data[target_col].to_numpy(dtype=np.float64, copy=True)
    dcor = pd.Series(np.nan, index=feature_cols, dtype=np.float64)
    hsic = pd.Series(np.nan, index=feature_cols, dtype=np.float64)
    for col in candidates:
        x = kernel_data[col].to_numpy(dtype=np.float64, copy=True)
        dcor.loc[col] = distance_corr_1d(x, y_kernel)
        hsic.loc[col] = hsic_rbf_1d(x, y_kernel)

    report["distance_corr"] = dcor
    report["hsic_rbf"] = hsic
    report["feature_group"] = [infer_feature_group(f) for f in report.index]
    report["lag_min"] = [_lag_from_feature(f) for f in report.index]
    report["window_min"] = [_window_from_feature(f) for f in report.index]
    report["base_feature"] = [_base_from_feature(f) for f in report.index]
    report["rank_mi"] = report["mutual_info"].rank(ascending=False, method="min")
    report["rank_abs_spearman"] = report["abs_spearman"].rank(ascending=False, method="min")
    report["rank_distance_corr"] = report["distance_corr"].rank(ascending=False, method="min")
    report["rank_hsic_rbf"] = report["hsic_rbf"].rank(ascending=False, method="min")
    report["nonlinear_rank_score"] = (
        report["mutual_info"].rank(ascending=False, pct=True).rsub(1.0)
        + report["distance_corr"].fillna(0).rank(ascending=False, pct=True).rsub(1.0)
        + report["hsic_rbf"].fillna(0).rank(ascending=False, pct=True).rsub(1.0)
    ) / 3.0
    return report.sort_values("mutual_info", ascending=False)


def make_group_summary(report: pd.DataFrame, top_per_group: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    group_summary = (
        report.groupby("feature_group")
        .agg(
            n_features=("mutual_info", "size"),
            mean_mi=("mutual_info", "mean"),
            median_mi=("mutual_info", "median"),
            max_mi=("mutual_info", "max"),
            mean_abs_spearman=("abs_spearman", "mean"),
            max_distance_corr=("distance_corr", "max"),
            max_hsic_rbf=("hsic_rbf", "max"),
        )
        .sort_values("max_mi", ascending=False)
    )

    top_rows = []
    for group, part in report.groupby("feature_group", sort=False):
        top = part.sort_values("mutual_info", ascending=False).head(top_per_group).copy()
        top.insert(0, "feature", top.index)
        top_rows.append(top)
    group_top = pd.concat(top_rows, axis=0) if top_rows else pd.DataFrame()
    return group_summary, group_top


def make_lag_mi_table(report: pd.DataFrame) -> pd.DataFrame:
    lagged = report.dropna(subset=["lag_min"]).copy()
    if lagged.empty:
        return pd.DataFrame()
    lagged["stat_type"] = lagged.index.to_series().str.extract(r"_roll\d+_([^_]+)$")[0]
    lagged["stat_type"] = lagged["stat_type"].fillna("point")
    cols = [
        "base_feature",
        "lag_min",
        "window_min",
        "stat_type",
        "feature_group",
        "mutual_info",
        "mi_normalized_by_target_entropy",
        "abs_spearman",
        "distance_corr",
        "hsic_rbf",
    ]
    out = lagged[cols].copy()
    out.insert(0, "feature", out.index)
    return out.sort_values(["base_feature", "lag_min", "window_min", "stat_type"])


def make_scatter_data(
    df: pd.DataFrame,
    report: pd.DataFrame,
    target_col: str,
    top_n: int,
    sample_size: int,
    bins: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    top_features = report.head(top_n).index.tolist()
    data = _sample_frame(
        df[top_features + [target_col]].replace([np.inf, -np.inf], np.nan).dropna(),
        sample_size,
        SEED + 29,
    )
    scatter_parts = []
    binned_parts = []
    for col in top_features:
        part = pd.DataFrame(
            {
                "time": data.index.astype(str),
                "feature": col,
                "feature_value": data[col].to_numpy(),
                "target_value": data[target_col].to_numpy(),
                "mutual_info": report.loc[col, "mutual_info"],
                "spearman": report.loc[col, "spearman"],
                "feature_group": report.loc[col, "feature_group"],
            }
        )
        scatter_parts.append(part)

        x = data[col]
        y = data[target_col]
        try:
            bin_id = pd.qcut(x.rank(method="first"), q=min(bins, len(x)), labels=False, duplicates="drop")
        except ValueError:
            continue
        binned = (
            pd.DataFrame({"feature_value": x, "target_value": y, "bin_id": bin_id})
            .dropna()
            .groupby("bin_id")
            .agg(
                feature_x_median=("feature_value", "median"),
                feature_x_mean=("feature_value", "mean"),
                target_y_mean=("target_value", "mean"),
                target_y_median=("target_value", "median"),
                target_y_q25=("target_value", lambda s: s.quantile(0.25)),
                target_y_q75=("target_value", lambda s: s.quantile(0.75)),
                n=("target_value", "size"),
            )
            .reset_index()
        )
        binned.insert(0, "feature", col)
        binned.insert(1, "feature_group", report.loc[col, "feature_group"])
        binned_parts.append(binned)
    scatter = pd.concat(scatter_parts, ignore_index=True) if scatter_parts else pd.DataFrame()
    binned = pd.concat(binned_parts, ignore_index=True) if binned_parts else pd.DataFrame()
    return scatter, binned


def plot_top_bars(report: pd.DataFrame, out_dir: str, top_n: int) -> None:
    metrics = [
        ("mutual_info", "Mutual Information"),
        ("abs_spearman", "|Spearman|"),
        ("distance_corr", "Distance Correlation"),
        ("hsic_rbf", "RBF-HSIC"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 8))
    for ax, (col, title) in zip(axes, metrics):
        s = report[col].dropna().sort_values(ascending=False).head(top_n)
        colors = [
            plt.cm.tab10(hash(report.loc[idx, "feature_group"]) % 10)
            for idx in s.index
        ]
        ax.barh(np.arange(len(s)), s.values, color=colors)
        ax.set_yticks(np.arange(len(s)))
        ax.set_yticklabels(s.index, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    path = os.path.join(out_dir, "nonlinear_dependence_top_bars.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[NonlinearCorr] Saved figure: {path}")


def plot_group_summary(group_summary: pd.DataFrame, out_dir: str) -> None:
    if group_summary.empty:
        return
    plot_df = group_summary.sort_values("max_mi", ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4, len(plot_df) * 0.55)))
    ax.barh(np.arange(len(plot_df)), plot_df["max_mi"].values, label="Max MI")
    ax.scatter(plot_df["median_mi"].values, np.arange(len(plot_df)), color="black", s=28, label="Median MI")
    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels(plot_df.index)
    ax.set_xlabel("Mutual information")
    ax.set_title("Feature-group nonlinear dependence summary")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "feature_group_mi_summary.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[NonlinearCorr] Saved figure: {path}")


def plot_lag_mi(lag_table: pd.DataFrame, out_dir: str, top_base_n: int = 8) -> None:
    if lag_table.empty:
        return
    point_or_mean = lag_table[
        (lag_table["stat_type"].isin(["point", "mean", "sum"]))
        & (lag_table["base_feature"].notna())
    ].copy()
    if point_or_mean.empty:
        return
    top_bases = (
        point_or_mean.groupby("base_feature")["mutual_info"]
        .max()
        .sort_values(ascending=False)
        .head(top_base_n)
        .index.tolist()
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    for base in top_bases:
        part = point_or_mean[point_or_mean["base_feature"] == base]
        curve = part.groupby("lag_min")["mutual_info"].max().sort_index()
        ax.plot(curve.index, curve.values, marker="o", label=base)
    ax.set_xlabel("Propagation lag (min)")
    ax.set_ylabel("Mutual information")
    ax.set_title("Lag-dependent nonlinear dependence")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = os.path.join(out_dir, "lag_mi_comparison.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[NonlinearCorr] Saved figure: {path}")


def plot_top_scatter_binned(
    scatter: pd.DataFrame,
    binned: pd.DataFrame,
    out_dir: str,
    top_n: int,
) -> None:
    if scatter.empty:
        return
    features = scatter["feature"].drop_duplicates().head(top_n).tolist()
    ncols = 3
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.2 * nrows))
    axes = np.asarray(axes).reshape(-1)
    for ax, feature in zip(axes, features):
        part = scatter[scatter["feature"] == feature]
        ax.scatter(part["feature_value"], part["target_value"], s=3, alpha=0.2)
        trend = binned[binned["feature"] == feature]
        if not trend.empty:
            ax.plot(trend["feature_x_median"], trend["target_y_median"], color="red", linewidth=1.8)
            ax.fill_between(
                trend["feature_x_median"].to_numpy(),
                trend["target_y_q25"].to_numpy(),
                trend["target_y_q75"].to_numpy(),
                color="red",
                alpha=0.15,
            )
        mi = part["mutual_info"].iloc[0]
        rho = part["spearman"].iloc[0]
        ax.set_title(f"{feature}\nMI={mi:.4f}, rho={rho:.3f}", fontsize=9)
        ax.set_xlabel("Feature value")
        ax.set_ylabel("Target")
        ax.grid(True, alpha=0.25)
    for ax in axes[len(features):]:
        ax.axis("off")
    fig.tight_layout()
    path = os.path.join(out_dir, "nonlinear_dependence_scatter_binned.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[NonlinearCorr] Saved figure: {path}")


def save_run_metadata(
    out_dir: str,
    args: argparse.Namespace,
    rows_before_scope: int,
    rows_after_scope: int,
    feature_count: int,
    analysis_target: str,
    events: pd.DataFrame,
) -> None:
    meta = pd.DataFrame(
        [
            {
                "data": args.data,
                "scope": args.scope,
                "event_types": ",".join(args.event_types or []),
                "target": args.target,
                "target_mode": args.target_mode,
                "analysis_target": analysis_target,
                "horizon": args.horizon,
                "feature_set": getattr(args, "feature_set", "custom"),
                "include_current_solar": bool(getattr(args, "include_current_solar", False)),
                "include_lagged_solar": bool(getattr(args, "include_lagged_solar", False)),
                "include_geomag": bool(getattr(args, "include_geomag", False)),
                "include_time": bool(getattr(args, "include_time", False)),
                "lags": ",".join(map(str, args.lags)),
                "rolling_windows": ",".join(map(str, args.rolling_windows)),
                "rolling_stats": ",".join(args.rolling_stats),
                "rows_before_scope": rows_before_scope,
                "rows_after_scope": rows_after_scope,
                "feature_count": feature_count,
                "event_count": len(events),
                "sample_size_mi": args.sample_size,
                "sample_size_kernel": args.kernel_sample_size,
                "seed": SEED,
            }
        ]
    )
    meta.to_csv(os.path.join(out_dir, "run_metadata.csv"), index=False, encoding="utf-8-sig")
    if not events.empty:
        events.to_csv(os.path.join(out_dir, "selected_events.csv"), index=False, encoding="utf-8-sig")


def _apply_feature_preset(args: argparse.Namespace, feature_set: str) -> argparse.Namespace:
    if feature_set not in FEATURE_SET_PRESETS:
        raise ValueError(f"Unknown feature_set: {feature_set}")
    run_args = argparse.Namespace(**vars(args))
    preset = FEATURE_SET_PRESETS[feature_set]
    run_args.feature_set = feature_set
    run_args.feature_set_label = str(preset["label"])
    run_args.include_current_solar = bool(preset["include_current_solar"])
    run_args.include_lagged_solar = bool(preset["include_lagged_solar"])
    run_args.include_geomag = bool(preset["include_geomag"])
    run_args.include_time = not bool(args.no_time)
    return run_args


def run_analysis(args: argparse.Namespace) -> str:
    df, feature_cols, analysis_target, solar_metadata = build_analysis_table(
        data_path=args.data,
        target=args.target,
        target_mode=args.target_mode,
        horizon=args.horizon,
        lags=args.lags,
        rolling_windows=args.rolling_windows,
        rolling_stats=args.rolling_stats,
        include_current_solar=args.include_current_solar,
        include_lagged_solar=args.include_lagged_solar,
        include_geomag=args.include_geomag,
        include_time=args.include_time,
    )
    rows_before_scope = len(df)
    scoped_df, scope_label, selected_events = select_scope(df, args.scope, args.event_types)
    if scoped_df.empty:
        raise RuntimeError(f"Scope {args.scope} produced no rows.")

    feature_cols = [c for c in feature_cols if c in scoped_df.columns and c != analysis_target]
    print(
        f"[NonlinearCorr] feature_set={args.feature_set}, scope={scope_label}, rows={len(scoped_df):,}, "
        f"features={len(feature_cols)}, target={analysis_target}"
    )

    out_dir = os.path.join(
        args.output_root,
        _safe_name(args.feature_set_label),
        _safe_name(scope_label),
        _safe_name(args.target),
        f"H{int(args.horizon)}" if args.target_mode == "future-max" else "current",
    )
    os.makedirs(out_dir, exist_ok=True)

    report = compute_scores(
        scoped_df,
        feature_cols=feature_cols,
        target_col=analysis_target,
        mi_sample_size=args.sample_size,
        kernel_sample_size=args.kernel_sample_size,
        top_kernel_candidates=args.top_kernel_candidates,
    )
    report_export = report.copy()
    report_export.insert(0, "feature", report_export.index)
    report_path = os.path.join(out_dir, "nonlinear_correlation_report.csv")
    report_export.to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"[NonlinearCorr] Saved report: {report_path}")
    print(report.head(args.top_n).to_string())

    group_summary, group_top = make_group_summary(report, args.top_per_group)
    group_summary.to_csv(os.path.join(out_dir, "feature_group_summary.csv"), encoding="utf-8-sig")
    group_top.to_csv(os.path.join(out_dir, "feature_group_top_features.csv"), index=False, encoding="utf-8-sig")

    lag_table = make_lag_mi_table(report)
    if not lag_table.empty:
        lag_table.to_csv(os.path.join(out_dir, "lag_mi_comparison_data.csv"), index=False, encoding="utf-8-sig")

    scatter, binned = make_scatter_data(
        scoped_df,
        report,
        analysis_target,
        top_n=min(args.scatter_top_n, args.top_n),
        sample_size=args.scatter_sample_size,
        bins=args.bins,
    )
    scatter.to_csv(os.path.join(out_dir, "top_scatter_sample_data.csv"), index=False, encoding="utf-8-sig")
    binned.to_csv(os.path.join(out_dir, "top_scatter_binned_trend_data.csv"), index=False, encoding="utf-8-sig")

    if not solar_metadata.empty:
        solar_metadata.to_csv(os.path.join(out_dir, "corrected_solar_feature_metadata.csv"), index=False, encoding="utf-8-sig")

    save_run_metadata(
        out_dir,
        args,
        rows_before_scope=rows_before_scope,
        rows_after_scope=len(scoped_df),
        feature_count=len(feature_cols),
        analysis_target=analysis_target,
        events=selected_events,
    )

    plot_top_bars(report, out_dir, args.top_n)
    plot_group_summary(group_summary, out_dir)
    plot_lag_mi(lag_table, out_dir)
    plot_top_scatter_binned(scatter, binned, out_dir, min(args.scatter_top_n, args.top_n))

    print(f"[NonlinearCorr] Finished. Outputs: {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Current-feature nonlinear feature-vs-VKH-GIC dependence analysis"
    )
    parser.add_argument("--data", default=DATA_FILE)
    parser.add_argument("--target", default=TARGET_VYK_COL)
    parser.add_argument("--target-mode", default="future-max", choices=["future-max", "current"])
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--scope", default="event-type", choices=["full", "full-time", "fixed-events", "event-type"])
    parser.add_argument("--event-types", nargs="*", default=["CME", "CIR"])
    parser.add_argument("--feature-set", default="solar_coupling_geomag_current", choices=sorted(FEATURE_SET_PRESETS))
    parser.add_argument("--run-presets", action="store_true", help="Run both no-lag paper presets.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--lags", nargs="*", type=int, default=[30, 45, 60])
    parser.add_argument("--rolling-windows", nargs="*", type=int, default=[15, 30, 60])
    parser.add_argument("--rolling-stats", nargs="*", default=["mean", "std", "max"])
    parser.add_argument("--no-time", action="store_true")
    parser.add_argument("--sample-size", type=int, default=80000, help="Rows for mutual information.")
    parser.add_argument("--kernel-sample-size", type=int, default=2500, help="Rows for distance corr/HSIC.")
    parser.add_argument("--top-kernel-candidates", type=int, default=50)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--top-per-group", type=int, default=10)
    parser.add_argument("--scatter-top-n", type=int, default=12)
    parser.add_argument("--scatter-sample-size", type=int, default=30000)
    parser.add_argument("--bins", type=int, default=30)
    args = parser.parse_args()

    feature_sets = DEFAULT_FEATURE_PRESETS if args.run_presets else [args.feature_set]
    out_dirs = []
    for feature_set in feature_sets:
        out_dirs.append(run_analysis(_apply_feature_preset(args, feature_set)))
    print("[NonlinearCorr] All requested runs finished:")
    for out_dir in out_dirs:
        print(f"  - {out_dir}")


if __name__ == "__main__":
    main()
