"""
Solar-wind propagation-lag threshold and quantile ablation.

This script is intentionally separate from threshold_quantile_ablation.py.
It keeps the earlier geomagnetic-response experiment intact as a short-time
estimation baseline, and rebuilds true lead-time solar-wind features from
raw 1-minute columns only.

For a row at GIC time t:
  - point lag L uses solar wind at t-L minutes
  - lagged rolling window (L, W) uses solar wind from t-L-W+1 to t-L
  - the label is max(GIC[t+1 : t+H])

Outputs CSV reports under:
outputs/experiments/solar_wind_lagged_ablation/<scope>/<target>/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import (
    DATA_FILE,
    EXPERIMENT_DIR,
    SOLAR_WIND_FLAG_COLS,
    TARGET_COL,
    TARGET_COLUMNS,
    TARGET_DBHDT_RAW_COL,
    TARGET_VYK_COL,
)
from src.data_loader import build_vkh_event_type_report, prepare_targets
from src.solar_event_extraction import (
    extract_solar_wind_driver_events,
    rows_from_intervals,
)
from threshold_quantile_ablation import (
    _add_future_window_max_label,
    _rows_from_intervals,
    _safe_name,
    _split_events,
    build_summary,
    plot_ablation_reports,
    run_ablation,
    run_continuous_regression_ablation,
)


SOLAR_RAW_COLS = [
    "Btot",
    "Bx_gse",
    "By_gse",
    "Bz_gse",
    "Vp",
    "Np_filled",
    "P_dyn_nPa",
    "Ey_mV/m",
    "Ma",
    "epsilon_norm",
    "Newell",
    "Borovsky",
]

TIME_COLS = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
CACHE_VERSION = "solar_lagged_features_v2"

TARGET_SOURCE_COLS = [
    TARGET_COL,
    TARGET_DBHDT_RAW_COL,
    "dH_pert_dt",
    "dBH_dt",
    "dB_dt",
    "dH_dt",
]


def load_raw_minute_frame(path: str, target_col: str) -> pd.DataFrame:
    """
    Load a 1-minute source table and keep only columns needed for this
    lead-time experiment.

    Old engineered lag/rolling columns may exist in cached files; they are not
    used as model inputs and are dropped before new lagged solar-wind features
    are generated.
    """
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    required_targets = set(TARGET_COLUMNS + [target_col])
    if not required_targets.issubset(df.columns):
        df = prepare_targets(df)

    if target_col not in df.columns:
        raise KeyError(f"Missing target column after target preparation: {target_col}")

    keep_cols = []
    for col in SOLAR_RAW_COLS + SOLAR_WIND_FLAG_COLS + TARGET_SOURCE_COLS + TARGET_COLUMNS + [target_col]:
        if col in df.columns and col not in keep_cols:
            keep_cols.append(col)
    return df[keep_cols].copy()


def _default_feature_cache_path(args: argparse.Namespace) -> str:
    cfg = {
        "version": CACHE_VERSION,
        "data": os.path.abspath(args.data),
        "target": args.target,
        "horizon": int(args.horizon),
        "lags": [int(x) for x in args.lags],
        "rolling_windows": [int(x) for x in args.rolling_windows],
        "rolling_stats": [str(x) for x in args.rolling_stats],
        "solar_cols": SOLAR_RAW_COLS,
        "source_window_features": bool(args.source_window_features),
        "source_window_only": bool(args.source_window_only),
    }
    key = hashlib.md5(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(args.data)), "solar_lagged_feature_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"solar_lagged_{key}.parquet")


def _cache_sidecar_paths(cache_path: str) -> Tuple[str, str]:
    stem, _ = os.path.splitext(cache_path)
    return f"{stem}.groups.json", f"{stem}.metadata.csv"


def _load_feature_cache(cache_path: str) -> Tuple[pd.DataFrame, Dict[str, List[str]], pd.DataFrame, str]:
    groups_path, metadata_path = _cache_sidecar_paths(cache_path)
    if not (os.path.exists(cache_path) and os.path.exists(groups_path) and os.path.exists(metadata_path)):
        raise FileNotFoundError(cache_path)
    df = pd.read_parquet(cache_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    with open(groups_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    groups = {str(k): list(v) for k, v in payload["groups"].items()}
    label_col = str(payload["label_col"])
    metadata = pd.read_csv(metadata_path)
    print(f"[SolarLagCache] Loaded feature cache: {cache_path}")
    print(f"[SolarLagCache] shape={df.shape}, label={label_col}")
    return df, groups, metadata, label_col


def _save_feature_cache(
    cache_path: str,
    df: pd.DataFrame,
    groups: Dict[str, List[str]],
    metadata: pd.DataFrame,
    label_col: str,
    args: argparse.Namespace,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    groups_path, metadata_path = _cache_sidecar_paths(cache_path)
    df.to_parquet(cache_path)
    payload = {
        "version": CACHE_VERSION,
        "label_col": label_col,
        "groups": groups,
        "config": {
            "data": os.path.abspath(args.data),
            "target": args.target,
            "horizon": int(args.horizon),
            "lags": [int(x) for x in args.lags],
            "rolling_windows": [int(x) for x in args.rolling_windows],
            "rolling_stats": [str(x) for x in args.rolling_stats],
            "source_window_features": bool(args.source_window_features),
            "source_window_only": bool(args.source_window_only),
        },
    }
    with open(groups_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    metadata.to_csv(metadata_path, index=False, encoding="utf-8-sig")
    print(f"[SolarLagCache] Saved feature cache: {cache_path}")
    print(f"[SolarLagCache] Saved groups: {groups_path}")


def _add_time_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    hour = (df.index.hour + df.index.minute / 60.0).astype(np.float32)
    day_of_year = df.index.dayofyear.astype(np.float32)
    out = pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24.0).astype(np.float32),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0).astype(np.float32),
            "doy_sin": np.sin(2 * np.pi * day_of_year / 365.25).astype(np.float32),
            "doy_cos": np.cos(2 * np.pi * day_of_year / 365.25).astype(np.float32),
        },
        index=df.index,
    )
    return pd.concat([df, out], axis=1), TIME_COLS.copy()


def _rolling_stat(s: pd.Series, window: int, stat: str) -> pd.Series:
    roller = s.rolling(window=int(window), min_periods=1)
    if stat == "mean":
        return roller.mean()
    if stat == "std":
        return roller.std().fillna(0)
    if stat == "max":
        return roller.max()
    if stat == "min":
        return roller.min()
    raise ValueError(f"Unsupported rolling stat: {stat}")


def build_lagged_solar_features(
    df: pd.DataFrame,
    solar_cols: Iterable[str],
    lags: Iterable[int],
    rolling_windows: Iterable[int],
    rolling_stats: Iterable[str],
    horizon: int = 30,
    source_window_features: bool = True,
    source_window_only: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, List[str]], pd.DataFrame]:
    """
    Build true propagation-lag features.

    Every generated solar feature has an explicit source time no later than
    t-lag. Existing engineered lag/rolling columns in df are ignored.
    """
    out = df.copy()
    available = [c for c in solar_cols if c in out.columns]
    if not available:
        raise RuntimeError("No requested solar-wind columns were found in the input data.")

    groups: Dict[str, List[str]] = {}
    metadata_rows = []
    new_cols = {}
    horizon = max(int(horizon), 1)

    for lag in [int(x) for x in lags]:
        group_cols: List[str] = []
        source_window_cols: List[str] = []
        shifted_cache: Dict[str, pd.Series] = {}

        for col in available:
            s_lag = pd.to_numeric(out[col], errors="coerce").shift(lag)
            shifted_cache[col] = s_lag
            if not source_window_only:
                name = f"{col}_lag{lag}"
                new_cols[name] = s_lag.astype(np.float32)
                group_cols.append(name)
                metadata_rows.append(
                    {
                        "feature": name,
                        "source_column": col,
                        "lag_min": lag,
                        "window_min": 1,
                        "stat": "point",
                        "source_start_min_before_t": lag,
                        "source_end_min_before_t": lag,
                    }
                )

        if not source_window_only:
            for col, s_lag in shifted_cache.items():
                for win in [int(x) for x in rolling_windows]:
                    for stat in rolling_stats:
                        name = f"{col}_lag{lag}_roll{win}_{stat}"
                        new_cols[name] = _rolling_stat(s_lag, win, str(stat)).astype(np.float32)
                        group_cols.append(name)
                        metadata_rows.append(
                            {
                                "feature": name,
                                "source_column": col,
                                "lag_min": lag,
                                "window_min": win,
                                "stat": str(stat),
                                "source_start_min_before_t": lag + win - 1,
                                "source_end_min_before_t": lag,
                            }
                        )

        if source_window_features:
            src_start_before_t = lag - 1
            src_end_before_t = lag - horizon
            if lag == 0:
                # No propagation-lag baseline: use information available up to
                # forecast issue time t only, i.e. [t-H+1, t]. This is not a
                # true L1 lead-time setting, but is useful as a no-lag ablation.
                src_start_before_t = horizon - 1
                src_end_before_t = 0
            if src_end_before_t >= 0:
                for col in available:
                    shifted = pd.to_numeric(out[col], errors="coerce").shift(src_end_before_t)
                    roller = shifted.rolling(horizon, min_periods=1)
                    stat_values = {
                        "mean": roller.mean(),
                        "max": roller.max(),
                        "std": roller.std().fillna(0.0),
                    }
                    for stat_name, values in stat_values.items():
                        name = f"{col}_lag{lag}_srcH{horizon}_{stat_name}"
                        new_cols[name] = values.astype(np.float32)
                        group_cols.append(name)
                        source_window_cols.append(name)
                        metadata_rows.append(
                            {
                                "feature": name,
                                "source_column": col,
                                "lag_min": lag,
                                "window_min": horizon,
                                "stat": f"future_source_{stat_name}",
                                "source_start_min_before_t": src_start_before_t,
                                "source_end_min_before_t": src_end_before_t,
                            }
                        )
                if "Bz_gse" in available:
                    shifted_bz = pd.to_numeric(out["Bz_gse"], errors="coerce").shift(src_end_before_t)
                    bz_south = shifted_bz.clip(upper=0).abs()
                    name = f"Bz_south_lag{lag}_srcH{horizon}_sum"
                    new_cols[name] = bz_south.rolling(horizon, min_periods=1).sum().astype(np.float32)
                    group_cols.append(name)
                    source_window_cols.append(name)
                    metadata_rows.append(
                        {
                            "feature": name,
                            "source_column": "Bz_gse",
                            "lag_min": lag,
                            "window_min": horizon,
                            "stat": "future_source_southward_sum",
                            "source_start_min_before_t": src_start_before_t,
                            "source_end_min_before_t": src_end_before_t,
                        }
                    )

        if "Bz_gse" in shifted_cache and not source_window_only:
            bz_south = shifted_cache["Bz_gse"].clip(upper=0).abs()
            name = f"Bz_south_lag{lag}"
            new_cols[name] = bz_south.astype(np.float32)
            group_cols.append(name)
            metadata_rows.append(
                {
                    "feature": name,
                    "source_column": "Bz_gse",
                    "lag_min": lag,
                    "window_min": 1,
                    "stat": "southward_abs",
                    "source_start_min_before_t": lag,
                    "source_end_min_before_t": lag,
                }
            )

            for win in [int(x) for x in rolling_windows]:
                name = f"Bz_south_lag{lag}_roll{win}_sum"
                new_cols[name] = bz_south.rolling(win, min_periods=1).sum().astype(np.float32)
                group_cols.append(name)
                metadata_rows.append(
                    {
                        "feature": name,
                        "source_column": "Bz_gse",
                        "lag_min": lag,
                        "window_min": win,
                        "stat": "southward_sum",
                        "source_start_min_before_t": lag + win - 1,
                        "source_end_min_before_t": lag,
                    }
                )

        if "Vp" in shifted_cache and "Bz_gse" in shifted_cache and not source_window_only:
            name = f"Vp_Bz_south_lag{lag}"
            new_cols[name] = (
                shifted_cache["Vp"] * shifted_cache["Bz_gse"].clip(upper=0).abs()
            ).astype(np.float32)
            group_cols.append(name)
            metadata_rows.append(
                {
                    "feature": name,
                    "source_column": "Vp*Bz_gse",
                    "lag_min": lag,
                    "window_min": 1,
                    "stat": "product",
                    "source_start_min_before_t": lag,
                    "source_end_min_before_t": lag,
                }
            )

        if "By_gse" in shifted_cache and "Bz_gse" in shifted_cache and not source_window_only:
            clock = np.arctan2(shifted_cache["By_gse"], shifted_cache["Bz_gse"])
            for suffix, values in {
                "sin": np.sin(clock),
                "cos": np.cos(clock),
            }.items():
                name = f"imf_clock_{suffix}_lag{lag}"
                new_cols[name] = pd.Series(values, index=out.index).astype(np.float32)
                group_cols.append(name)
                metadata_rows.append(
                    {
                        "feature": name,
                        "source_column": "By_gse,Bz_gse",
                        "lag_min": lag,
                        "window_min": 1,
                        "stat": f"clock_{suffix}",
                        "source_start_min_before_t": lag,
                        "source_end_min_before_t": lag,
                    }
                )

        if "P_dyn_nPa" in shifted_cache and not source_window_only:
            name = f"dP_dyn_dt_lag{lag}"
            new_cols[name] = shifted_cache["P_dyn_nPa"].diff().astype(np.float32)
            group_cols.append(name)
            metadata_rows.append(
                {
                    "feature": name,
                    "source_column": "P_dyn_nPa",
                    "lag_min": lag,
                    "window_min": 2,
                    "stat": "diff",
                    "source_start_min_before_t": lag + 1,
                    "source_end_min_before_t": lag,
                }
            )

        if "Newell" in shifted_cache and not source_window_only:
            name = f"Newell_diff30_lag{lag}"
            new_cols[name] = shifted_cache["Newell"].diff(30).astype(np.float32)
            group_cols.append(name)
            metadata_rows.append(
                {
                    "feature": name,
                    "source_column": "Newell",
                    "lag_min": lag,
                    "window_min": 31,
                    "stat": "diff30",
                    "source_start_min_before_t": lag + 30,
                    "source_end_min_before_t": lag,
                }
            )

        if group_cols:
            groups[f"solar_lag{lag}"] = list(dict.fromkeys(group_cols))
        if source_window_cols:
            groups[f"solar_lag{lag}_source_window"] = list(dict.fromkeys(source_window_cols))

    feature_frame = pd.DataFrame(new_cols, index=out.index)
    out = pd.concat([out, feature_frame], axis=1)
    groups["solar_all_lags"] = list(dict.fromkeys(c for cols in groups.values() for c in cols))
    source_cols_all = [c for c in groups["solar_all_lags"] if f"srcH{horizon}" in c]
    if source_cols_all:
        groups["solar_source_windows"] = list(dict.fromkeys(source_cols_all))
    out, time_cols = _add_time_features(out)
    groups["time_only"] = time_cols
    groups["solar_all_lags_plus_time"] = groups["solar_all_lags"] + time_cols
    if source_cols_all:
        groups["solar_source_windows_plus_time"] = groups["solar_source_windows"] + time_cols
    for lag in [int(x) for x in lags]:
        if f"solar_lag{lag}" in groups:
            groups[f"solar_lag{lag}_plus_time"] = groups[f"solar_lag{lag}"] + time_cols
        if f"solar_lag{lag}_source_window" in groups:
            groups[f"solar_lag{lag}_source_window_plus_time"] = (
                groups[f"solar_lag{lag}_source_window"] + time_cols
            )

    metadata = pd.DataFrame(metadata_rows)
    return out, groups, metadata


def _validate_feature_logic(groups: Dict[str, List[str]], metadata: pd.DataFrame) -> None:
    blocked = ["x_pert", "y_pert", "z_pert", "h_pert", "dbhdt", "dh_dt", "dbhd"]
    for group, cols in groups.items():
        if group == "time_only":
            continue
        bad = [c for c in cols if any(token in c.lower() for token in blocked)]
        if bad:
            raise RuntimeError(f"Geomagnetic leakage in {group}: {bad[:5]}")

    if not metadata.empty:
        invalid = metadata[
            metadata["source_end_min_before_t"].astype(float)
            > metadata["lag_min"].astype(float)
        ]
        if len(invalid) > 0:
            raise RuntimeError("Found lagged features whose source window ends after t-lag.")


def split_paper_vkh_driver_events(
    df: pd.DataFrame,
    train_ratio: float,
    include_types: Iterable[str],
    pre_context_min: int,
    post_context_min: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, str, pd.DataFrame]:
    report = build_vkh_event_type_report(
        df,
        pre_context_min=int(pre_context_min),
        post_context_min=int(post_context_min),
    )
    report = report[
        report["in_data_range"].astype(bool)
        & report["quality_keep"].astype(bool)
    ].sort_values("peak_time").copy()
    include = {str(x).upper() for x in include_types}
    if include and "ALL" not in include:
        report = report[report["event_type"].astype(str).str.upper().isin(include)].copy()
    if len(report) < 2:
        raise RuntimeError(f"Not enough paper VKH events after filtering: {len(report)}")
    cut = max(1, min(len(report) - 1, int(round(len(report) * float(train_ratio)))))
    report["split"] = "test"
    report.iloc[:cut, report.columns.get_loc("split")] = "train"
    train_df = _rows_from_intervals(
        df,
        [(r["start"], r["end"]) for _, r in report[report["split"].eq("train")].iterrows()],
    )
    test_df = _rows_from_intervals(
        df,
        [(r["start"], r["end"]) for _, r in report[report["split"].eq("test")].iterrows()],
    )
    type_label = "all" if not include or "ALL" in include else "_".join(sorted(include))
    return train_df, test_df, f"paper_vkh_drivers_{type_label}", report


def save_lag_ablation_event_overlay_plots(
    out_dir: str,
    group_names: List[str],
    events_report: pd.DataFrame,
    target_name: str,
) -> None:
    if events_report is None or events_report.empty or not group_names:
        return
    cache_dir = os.path.join(out_dir, "prediction_cache")
    frames: Dict[str, pd.DataFrame] = {}
    for group in group_names:
        path = os.path.join(cache_dir, f"{group}_predictions.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")
        elif df.columns[0].lower().startswith("unnamed"):
            df = df.rename(columns={df.columns[0]: "datetime"})
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")
        if "final_pred" not in df.columns:
            q_cols = [c for c in df.columns if c.startswith("Q")]
            if "Q50" in q_cols:
                df["final_pred"] = df["Q50"]
            elif "Q80" in q_cols:
                df["final_pred"] = df["Q80"]
            elif q_cols:
                df["final_pred"] = df[sorted(q_cols)[0]]
        frames[group] = df
    if not frames:
        return

    plot_dir = os.path.join(out_dir, "lag_ablation_event_overlay")
    os.makedirs(plot_dir, exist_ok=True)
    summary_rows = []
    for row in events_report.itertuples(index=False):
        if getattr(row, "split", "") != "test":
            continue
        start = pd.Timestamp(getattr(row, "start"))
        end = pd.Timestamp(getattr(row, "end"))
        paper_id = int(getattr(row, "paper_event_id", getattr(row, "event_id", 0)))
        peak_time = getattr(row, "peak_time", pd.NaT)
        event_type = getattr(row, "event_type", "")
        event_type_raw = getattr(row, "event_type_raw", "")

        fig, ax = plt.subplots(figsize=(16, 5))
        plotted = False
        raw_plotted = False
        for group, df in frames.items():
            sub = df.loc[(df.index >= start) & (df.index <= end)]
            if sub.empty:
                continue
            if not raw_plotted:
                if "gic_true" in sub.columns:
                    ax.plot(sub.index, sub["gic_true"], color="black", linewidth=1.2, label="True GIC")
                    raw_plotted = True
                elif "y_true" in sub.columns:
                    ax.plot(sub.index, sub["y_true"], color="black", linewidth=1.2, label="True future max")
                    raw_plotted = True
            if "final_pred" in sub.columns:
                label = group.replace("solar_", "").replace("_source_window_plus_time", "")
                ax.plot(sub.index, sub["final_pred"], linewidth=1.0, label=label)
                plotted = True
                summary_rows.append(
                    {
                        "paper_event_id": paper_id,
                        "event_type": event_type,
                        "event_type_raw": event_type_raw,
                        "group": group,
                        "n_points": int(len(sub)),
                        "true_gic_max": float(sub["gic_true"].max()) if "gic_true" in sub.columns else np.nan,
                        "true_future_max": float(sub["y_true"].max()) if "y_true" in sub.columns else np.nan,
                        "pred_max": float(sub["final_pred"].max()),
                        "pred_at_true_gic_peak": float(
                            sub["final_pred"].iloc[int(np.nanargmax(sub["gic_true"].to_numpy()))]
                        ) if "gic_true" in sub.columns and len(sub) else np.nan,
                    }
                )
        if not plotted:
            plt.close(fig)
            continue
        for thr in [3.0, 5.0, 10.0, 20.0]:
            ax.axhline(thr, color="gray", linestyle="--", linewidth=0.7, alpha=0.35)
        ax.set_title(
            f"{target_name} | paper event {paper_id:02d} | {event_type} ({event_type_raw}) | peak={peak_time}"
        )
        ax.set_ylabel("GIC / predicted future max (A)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", ncol=3, fontsize=8)
        fig.tight_layout()
        name = f"paper_event_{paper_id:02d}_{pd.Timestamp(start).strftime('%Y%m%d_%H%M')}_{pd.Timestamp(end).strftime('%Y%m%d_%H%M')}.png"
        fig.savefig(os.path.join(plot_dir, name), dpi=150, bbox_inches="tight")
        plt.close(fig)
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(plot_dir, "lag_ablation_event_overlay_summary.csv"),
            index=False,
            encoding="utf-8-sig",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="True propagation-lag solar-wind threshold and quantile ablation"
    )
    parser.add_argument("--data", default=DATA_FILE)
    parser.add_argument("--target", default=TARGET_VYK_COL)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--scope", default="paper-vkh-drivers", choices=["paper-vkh-drivers", "solar-driver", "fixed-events", "event-type", "full-time"])
    parser.add_argument("--event-types", nargs="*", default=["CME", "CIR"])
    parser.add_argument(
        "--paper-driver-types",
        nargs="*",
        default=["ALL"],
        help="Paper VKH driver groups to include: ALL CME CIR NO_WEAK SC_SI.",
    )
    parser.add_argument(
        "--event-type-batches",
        nargs="*",
        choices=["CME", "CIR", "CME_CIR"],
        default=None,
        help="Run separate event-type models in one command, e.g. CME CIR CME_CIR.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--lags", nargs="*", type=int, default=[30, 45, 60, 90])
    parser.add_argument("--rolling-windows", nargs="*", type=int, default=[15, 30, 60])
    parser.add_argument("--rolling-stats", nargs="*", default=["mean", "std", "max"])
    parser.add_argument("--thresholds", nargs="*", type=float, default=[3.0, 5.0, 10.0, 20.0])
    parser.add_argument("--quantiles", nargs="*", type=float, default=[0.80, 0.90, 0.95])
    parser.add_argument("--final-quantile", type=float, default=0.50)
    parser.add_argument("--max-train-rows", type=int, default=250000)
    parser.add_argument("--max-test-rows", type=int, default=120000)
    parser.add_argument(
        "--model-backend",
        default="model3",
        choices=["hgb", "lightgbm", "bilstm", "cnn_bilstm", "cnn_bilstm_attention", "model3", "model5", "gatefusion"],
    )
    parser.add_argument(
        "--model-backends",
        nargs="*",
        default=None,
        help="Run multiple backends for comparison, e.g. model3 model5.",
    )
    parser.add_argument(
        "--feature-cache",
        default="auto",
        help="Parquet cache for lagged feature table. Use 'auto' for config-based path or 'none' to disable.",
    )
    parser.add_argument("--rebuild-feature-cache", action="store_true")
    parser.add_argument("--eval-only", action="store_true", help="Load saved deep checkpoints and only evaluate/plot.")
    source_window_group = parser.add_mutually_exclusive_group()
    source_window_group.add_argument(
        "--source-window-features",
        dest="source_window_features",
        action="store_true",
        help="Enable future-label source-window solar-wind features.",
    )
    source_window_group.add_argument(
        "--no-source-window-features",
        dest="source_window_features",
        action="store_false",
        help="Disable future-label source-window solar-wind features.",
    )
    parser.set_defaults(source_window_features=True)
    parser.add_argument(
        "--source-window-only",
        action="store_true",
        help="Build only future-source-window lag features to reduce memory use.",
    )
    parser.add_argument("--solar-roll-min", type=int, default=30)
    parser.add_argument("--solar-core-quantile", type=float, default=0.92)
    parser.add_argument("--solar-assist-quantile", type=float, default=0.95)
    parser.add_argument("--solar-min-score", type=float, default=1.05)
    parser.add_argument("--solar-min-peak-score", type=float, default=1.25)
    parser.add_argument("--solar-min-core-hit-ratio", type=float, default=0.20)
    parser.add_argument("--no-require-southward", action="store_true")
    parser.add_argument("--solar-min-duration-min", type=int, default=60)
    parser.add_argument("--solar-merge-gap-min", type=int, default=180)
    parser.add_argument("--solar-pre-context-min", type=int, default=360)
    parser.add_argument("--solar-post-context-min", type=int, default=1440)
    parser.add_argument("--paper-pre-context-min", type=int, default=720)
    parser.add_argument("--paper-post-context-min", type=int, default=1440)
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
        default=None,
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
        "--feature-groups",
        nargs="*",
        default=None,
        help="Only train selected feature groups, e.g. solar_source_windows_plus_time time_only.",
    )
    parser.add_argument(
        "--run-continuous-regression",
        action="store_true",
        help="Also train weighted-Huber continuous deep regressors for GIC future max.",
    )
    parser.add_argument("--huber-delta", type=float, default=1.0)
    args = parser.parse_args()
    model_backends = args.model_backends if args.model_backends else [args.model_backend]
    if args.plot_groups is None:
        args.plot_groups = ["solar_source_windows_plus_time", "solar_all_lags_plus_time"] + [
            f"solar_lag{int(lag)}_source_window_plus_time" for lag in args.lags
        ] + [
            f"solar_lag{int(lag)}_plus_time" for lag in args.lags
        ]

    cache_path = None
    if str(args.feature_cache).lower() != "none":
        cache_path = _default_feature_cache_path(args) if args.feature_cache == "auto" else args.feature_cache

    if cache_path and os.path.exists(cache_path) and not args.rebuild_feature_cache:
        df, groups, metadata, label_col = _load_feature_cache(cache_path)
        _validate_feature_logic(groups, metadata)
    else:
        print(f"[SolarLag] Loading raw 1-minute source: {args.data}")
        df = load_raw_minute_frame(args.data, args.target)
        print(
            f"[SolarLag] Using raw solar-wind columns only; shape={df.shape}, "
            f"range={df.index.min()} ~ {df.index.max()}"
        )

        df, groups, metadata = build_lagged_solar_features(
            df=df,
            solar_cols=SOLAR_RAW_COLS,
            lags=args.lags,
            rolling_windows=args.rolling_windows,
            rolling_stats=args.rolling_stats,
            horizon=args.horizon,
            source_window_features=args.source_window_features,
            source_window_only=args.source_window_only,
        )
        _validate_feature_logic(groups, metadata)
        df, label_col = _add_future_window_max_label(df, args.target, args.horizon)

        max_required_history = max(args.lags) + max(args.rolling_windows) - 1
        df = df.iloc[int(max_required_history):].copy()
        if cache_path:
            _save_feature_cache(cache_path, df, groups, metadata, label_col, args)

    batch_map = {"CME": ["CME"], "CIR": ["CIR"], "CME_CIR": ["CME", "CIR"]}
    if args.scope == "event-type" and args.event_type_batches:
        event_type_batches = [batch_map[name] for name in args.event_type_batches]
    else:
        event_type_batches = [args.event_types]

    for event_types in event_type_batches:
        events_report = pd.DataFrame()
        solar_thresholds = {}
        if args.scope == "solar-driver":
            train_intervals, test_intervals, events_report, solar_thresholds = extract_solar_wind_driver_events(
                df,
                train_ratio=args.train_ratio,
                roll_min=args.solar_roll_min,
                core_quantile=args.solar_core_quantile,
                assist_quantile=args.solar_assist_quantile,
                min_score=args.solar_min_score,
                require_southward=not args.no_require_southward,
                min_peak_score=args.solar_min_peak_score,
                min_core_hit_ratio=args.solar_min_core_hit_ratio,
                min_duration_min=args.solar_min_duration_min,
                merge_gap_min=args.solar_merge_gap_min,
                pre_context_min=args.solar_pre_context_min,
                post_context_min=args.solar_post_context_min,
            )
            train_df = rows_from_intervals(df, train_intervals)
            test_df = rows_from_intervals(df, test_intervals)
            scope_label = "solar_driver_events"
        elif args.scope == "paper-vkh-drivers":
            train_df, test_df, scope_label, events_report = split_paper_vkh_driver_events(
                df=df,
                train_ratio=args.train_ratio,
                include_types=args.paper_driver_types,
                pre_context_min=args.paper_pre_context_min,
                post_context_min=args.paper_post_context_min,
            )
        else:
            train_df, test_df, scope_label = _split_events(
                df,
                scope=args.scope,
                event_types=event_types,
                train_ratio=args.train_ratio,
            )

        for model_backend in model_backends:
            run_groups = groups
            if args.feature_groups:
                missing = [g for g in args.feature_groups if g not in groups]
                if missing:
                    raise KeyError(f"Unknown feature groups: {missing}. Available: {list(groups)}")
                run_groups = {g: groups[g] for g in args.feature_groups}
            out_dir = os.path.join(
                EXPERIMENT_DIR,
                "solar_wind_lagged_ablation" if model_backend == "hgb"
                else f"solar_wind_lagged_ablation_{model_backend}",
                _safe_name(scope_label),
                _safe_name(f"{args.target}_H{args.horizon}"),
            )
            os.makedirs(out_dir, exist_ok=True)
            metadata.to_csv(os.path.join(out_dir, "lagged_feature_metadata.csv"), index=False, encoding="utf-8-sig")
            if not events_report.empty:
                events_report.to_csv(os.path.join(out_dir, "events_report.csv"), index=False, encoding="utf-8-sig")
                if solar_thresholds:
                    pd.DataFrame([solar_thresholds]).to_csv(os.path.join(out_dir, "solar_driver_thresholds.csv"), index=False, encoding="utf-8-sig")

            feature_rows = [
                {"feature_group": group, "n_features": len(cols), "features": ",".join(cols)}
                for group, cols in run_groups.items()
            ]
            pd.DataFrame(feature_rows).to_csv(
                os.path.join(out_dir, "feature_groups.csv"), index=False, encoding="utf-8-sig"
            )

            print(
                f"[SolarLag] scope={scope_label}, train_rows={len(train_df):,}, "
                f"test_rows={len(test_df):,}, model={model_backend}, out={out_dir}"
            )
            for group, cols in run_groups.items():
                print(f"[SolarLag] group={group}, n_features={len(cols)}")

            cls_report, quantile_report = run_ablation(
                train_df=train_df,
                test_df=test_df,
                feature_groups=run_groups,
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
                model_backend=model_backend,
                events_report=events_report if not events_report.empty else None,
                eval_only=args.eval_only,
                raw_target_col=args.target,
                final_quantile=args.final_quantile,
                low_far_event_max_far=args.low_far_event_max_far,
                skip_prediction_plots=args.skip_prediction_plots,
            )

            regression_report = pd.DataFrame()
            if args.run_continuous_regression:
                regression_report = run_continuous_regression_ablation(
                    train_df=train_df,
                    test_df=test_df,
                    feature_groups=run_groups,
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
                    model_backend=model_backend,
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
            print(f"[SolarLag] Saved: {cls_path}")
            print(f"[SolarLag] Saved: {q_path}")
            if not regression_report.empty:
                print(f"[SolarLag] Saved: {reg_path}")
            print(f"[SolarLag] Saved: {summary_path}")
            if args.feature_groups and len(args.feature_groups) > 1 and not events_report.empty:
                save_lag_ablation_event_overlay_plots(
                    out_dir=out_dir,
                    group_names=list(run_groups.keys()),
                    events_report=events_report,
                    target_name=args.target,
                )


if __name__ == "__main__":
    main()
