"""Gradient-SHAP style feature analysis for saved final envelope predictions.

The installed environment may not provide the external ``shap`` package, so
this script implements an Expected Gradients / GradientSHAP approximation for
the saved PyTorch multi-quantile model. It explains the final Q90 envelope
prediction by default, matching ``final_pred`` in the current prediction cache.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from solar_wind_lagged_ablation import (
    _add_future_window_max_label,
    _load_feature_cache,
    load_raw_minute_frame,
    split_paper_vkh_driver_events,
)
from threshold_quantile_ablation import DeepMultiQuantileRegressor


DEFAULT_RUN_DIR = Path("outputs/experiments/paper_vkh_drivers_all/gic_vyk_abs_H30")
DEFAULT_DATA = Path("data/merged_2012_2022_processed.parquet")
DEFAULT_GROUP = "solar_lag45_source_window_plus_time"


def _parse_lags_from_groups(groups: Sequence[str]) -> List[int]:
    lags = []
    for group in groups:
        for match in re.finditer(r"lag(\d+)", group):
            lags.append(int(match.group(1)))
    return sorted(set(lags)) or [45]


def _load_feature_groups(run_dir: Path, selected_groups: Sequence[str]) -> Dict[str, List[str]]:
    path = run_dir / "feature_groups.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature_groups.csv: {path}")
    frame = pd.read_csv(path)
    groups: Dict[str, List[str]] = {}
    for _, row in frame.iterrows():
        name = str(row["feature_group"])
        if selected_groups and name not in selected_groups:
            continue
        features = [x for x in str(row["features"]).split(",") if x]
        groups[name] = features
    missing = [g for g in selected_groups if g not in groups]
    if missing:
        raise KeyError(f"Unknown groups in feature_groups.csv: {missing}")
    return groups


def _build_test_features(
    data_path: Path,
    feature_cache: Optional[Path],
    target: str,
    horizon: int,
    groups: Dict[str, List[str]],
    lags: Sequence[int],
    rolling_windows: Sequence[int],
    rolling_stats: Sequence[str],
    train_ratio: float,
    paper_driver_types: Sequence[str],
    paper_pre_context_min: int,
    paper_post_context_min: int,
) -> Tuple[pd.DataFrame, str]:
    needed_features = sorted(set(feature for cols in groups.values() for feature in cols))
    if feature_cache is not None and feature_cache.exists():
        print(f"[SHAP] Loading feature cache: {feature_cache}", flush=True)
        df, built_groups, _, label_col = _load_feature_cache(str(feature_cache))
    else:
        print(f"[SHAP] Loading raw data: {data_path}", flush=True)
        df = load_raw_minute_frame(str(data_path), target)
        df = _add_required_source_window_features(df, needed_features, horizon)
        df, label_col = _add_future_window_max_label(df, target, horizon)
        built_groups = groups

    missing = [feature for feature in needed_features if feature not in df.columns]
    if missing:
        available_hint = [name for name in built_groups if name in groups]
        raise KeyError(f"Missing rebuilt features: {missing[:10]}; rebuilt matching groups={available_hint}")

    if feature_cache is None or not feature_cache.exists():
        max_required_history = max(int(x) for x in lags) + max(int(x) for x in rolling_windows) - 1
        df = df.iloc[int(max_required_history):].copy()
    _, test_df, _, _ = split_paper_vkh_driver_events(
        df=df,
        train_ratio=float(train_ratio),
        include_types=paper_driver_types,
        pre_context_min=int(paper_pre_context_min),
        post_context_min=int(paper_post_context_min),
    )
    return test_df, label_col


def _add_required_source_window_features(df: pd.DataFrame, needed_features: Sequence[str], horizon: int) -> pd.DataFrame:
    """Build only the feature columns needed for SHAP to avoid high memory use."""
    out = df.copy()
    new_cols: Dict[str, pd.Series] = {}
    horizon = int(horizon)
    for feature in needed_features:
        if feature in out.columns or feature in new_cols:
            continue
        if feature in {"hour_sin", "hour_cos", "doy_sin", "doy_cos"}:
            hour = (out.index.hour + out.index.minute / 60.0).astype(np.float32)
            day_of_year = out.index.dayofyear.astype(np.float32)
            values = {
                "hour_sin": np.sin(2 * np.pi * hour / 24.0).astype(np.float32),
                "hour_cos": np.cos(2 * np.pi * hour / 24.0).astype(np.float32),
                "doy_sin": np.sin(2 * np.pi * day_of_year / 365.25).astype(np.float32),
                "doy_cos": np.cos(2 * np.pi * day_of_year / 365.25).astype(np.float32),
            }
            new_cols[feature] = pd.Series(values[feature], index=out.index)
            continue

        south_match = re.fullmatch(r"Bz_south_lag(\d+)_srcH(\d+)_sum", feature)
        if south_match:
            lag = int(south_match.group(1))
            window = int(south_match.group(2))
            source_end_before_t = lag - horizon
            shifted = pd.to_numeric(out["Bz_gse"], errors="coerce").shift(source_end_before_t)
            new_cols[feature] = shifted.clip(upper=0).abs().rolling(window, min_periods=1).sum().astype(np.float32)
            continue

        src_match = re.fullmatch(r"(.+)_lag(\d+)_srcH(\d+)_(mean|max|std)", feature)
        if src_match:
            source_col = src_match.group(1)
            lag = int(src_match.group(2))
            window = int(src_match.group(3))
            stat = src_match.group(4)
            if source_col not in out.columns:
                raise KeyError(f"Missing source column {source_col} for feature {feature}")
            source_end_before_t = lag - horizon
            shifted = pd.to_numeric(out[source_col], errors="coerce").shift(source_end_before_t)
            roller = shifted.rolling(window, min_periods=1)
            if stat == "mean":
                values = roller.mean()
            elif stat == "max":
                values = roller.max()
            elif stat == "std":
                values = roller.std().fillna(0.0)
            else:
                raise ValueError(f"Unsupported feature stat: {feature}")
            new_cols[feature] = values.astype(np.float32)
            continue

        raise ValueError(f"Cannot rebuild required feature: {feature}")

    if new_cols:
        out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)
    return out


def _checkpoint_quantile_index(model: DeepMultiQuantileRegressor, quantile: float) -> int:
    quantiles = [float(q) for q in model.quantiles]
    return int(np.argmin(np.abs(np.asarray(quantiles) - float(quantile))))


def _forward_quantile_from_z(
    model: DeepMultiQuantileRegressor,
    z_tensor: torch.Tensor,
    quantile_index: int,
) -> torch.Tensor:
    if model.model is None:
        raise RuntimeError("Model checkpoint has not been loaded.")
    raw = model.model(z_tensor)
    first = F.softplus(raw[:, :1])
    if raw.shape[1] > 1:
        increments = F.softplus(raw[:, 1:])
        pred = torch.cat([first, first + torch.cumsum(increments, dim=1)], dim=1)
    else:
        pred = first
    return pred[:, quantile_index] * float(model.y_scale)


def _stratified_sample_indices(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    valid = np.where(np.isfinite(y))[0]
    if n <= 0 or len(valid) <= n:
        return valid
    rng = np.random.default_rng(seed)
    quantiles = np.nanquantile(y[valid], [0.0, 0.5, 0.8, 0.9, 0.97, 1.0])
    selected = []
    per_bin = max(1, int(math.ceil(n / (len(quantiles) - 1))))
    for lo, hi in zip(quantiles[:-1], quantiles[1:]):
        if hi == quantiles[-1]:
            members = valid[(y[valid] >= lo) & (y[valid] <= hi)]
        else:
            members = valid[(y[valid] >= lo) & (y[valid] < hi)]
        if len(members) == 0:
            continue
        take = min(per_bin, len(members))
        selected.extend(rng.choice(members, size=take, replace=False).tolist())
    selected = list(dict.fromkeys(selected))
    if len(selected) < n:
        remain = np.setdiff1d(valid, np.asarray(selected, dtype=int), assume_unique=False)
        extra = rng.choice(remain, size=min(n - len(selected), len(remain)), replace=False)
        selected.extend(extra.tolist())
    return np.asarray(selected[:n], dtype=int)


def _expected_gradients(
    model: DeepMultiQuantileRegressor,
    x_original: np.ndarray,
    background_original: np.ndarray,
    quantile: float,
    nsamples: int,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    if model.x_mean is None or model.x_std is None:
        raise RuntimeError("Loaded model has no standardization parameters.")
    device = model.device
    quantile_index = _checkpoint_quantile_index(model, quantile)
    x_mean = np.asarray(model.x_mean, dtype=np.float32)
    x_std = np.asarray(model.x_std, dtype=np.float32)
    x_z = np.nan_to_num((x_original.astype(np.float32) - x_mean) / x_std, nan=0.0, posinf=0.0, neginf=0.0)
    bg_z = np.nan_to_num((background_original.astype(np.float32) - x_mean) / x_std, nan=0.0, posinf=0.0, neginf=0.0)

    rng = np.random.default_rng(seed)
    shap_values = np.zeros_like(x_z, dtype=np.float32)
    if len(bg_z) == 0:
        raise ValueError("Empty background sample.")

    for start in range(0, len(x_z), batch_size):
        end = min(len(x_z), start + batch_size)
        x_batch = x_z[start:end]
        attr_accum = np.zeros_like(x_batch, dtype=np.float32)
        for _ in range(nsamples):
            bg = bg_z[rng.integers(0, len(bg_z), size=len(x_batch))]
            alpha = rng.random((len(x_batch), 1), dtype=np.float32)
            interpolated = bg + alpha * (x_batch - bg)
            z_tensor = torch.tensor(interpolated, dtype=torch.float32, device=device, requires_grad=True)
            output = _forward_quantile_from_z(model, z_tensor, quantile_index).sum()
            if z_tensor.grad is not None:
                z_tensor.grad.zero_()
            output.backward()
            grad = z_tensor.grad.detach().cpu().numpy().astype(np.float32)
            attr_accum += (x_batch - bg) * grad
        shap_values[start:end] = attr_accum / float(nsamples)
        print(f"[SHAP] Expected gradients rows {end}/{len(x_z)}", flush=True)
    return shap_values


def _is_excluded_feature(feature: str, exclude_patterns: Sequence[str]) -> bool:
    return any(re.search(pattern, feature) for pattern in exclude_patterns)


def _filter_plot_importance(importance: pd.DataFrame, exclude_patterns: Sequence[str]) -> pd.DataFrame:
    if not exclude_patterns:
        return importance.copy()
    mask = ~importance["feature"].astype(str).map(lambda x: _is_excluded_feature(x, exclude_patterns))
    return importance.loc[mask].copy()


def _write_summary_plot(importance: pd.DataFrame, out_dir: Path, group: str, top_n: int) -> None:
    plot_frame = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(5, 0.32 * len(plot_frame))))
    ax.barh(plot_frame["feature"], plot_frame["mean_abs_shap"], color="#4477AA")
    ax.set_xlabel("mean(|SHAP|) for Q90 final envelope (A)")
    ax.set_ylabel("Feature")
    ax.set_title(f"SHAP feature importance: {group}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"{group}_shap_feature_ranking.png", dpi=220)
    plt.close(fig)


def _write_dependence_plots(
    sample_frame: pd.DataFrame,
    shap_frame: pd.DataFrame,
    importance: pd.DataFrame,
    out_dir: Path,
    group: str,
    top_n: int,
) -> None:
    top_features = importance["feature"].head(top_n).tolist()
    for feature in top_features:
        dep = pd.DataFrame(
            {
                "datetime": sample_frame["datetime"].values,
                "feature": feature,
                "feature_value": sample_frame[feature].to_numpy(dtype=float),
                "shap_value": shap_frame[feature].to_numpy(dtype=float),
                "y_true": sample_frame["y_true"].to_numpy(dtype=float),
                "final_pred": sample_frame["final_pred"].to_numpy(dtype=float),
            }
        )
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", feature)
        dep.to_csv(out_dir / f"{group}_dependence_{safe}_data.csv", index=False, encoding="utf-8-sig")

        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        color = dep["final_pred"].to_numpy(dtype=float)
        sc = ax.scatter(dep["feature_value"], dep["shap_value"], c=color, s=16, cmap="viridis", alpha=0.78)
        ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
        ax.set_xlabel(feature)
        ax.set_ylabel("SHAP value for Q90 final envelope (A)")
        ax.set_title(f"SHAP dependence: {feature}")
        ax.grid(alpha=0.2)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("final_pred Q90 (A)")
        fig.tight_layout()
        fig.savefig(out_dir / f"{group}_dependence_{safe}.png", dpi=220)
        plt.close(fig)


def run_group_shap(
    run_dir: Path,
    group: str,
    test_df: pd.DataFrame,
    label_col: str,
    target: str,
    quantile: float,
    nsamples: int,
    background_size: int,
    explain_size: int,
    batch_size: int,
    top_n: int,
    exclude_plot_patterns: Sequence[str],
    seed: int,
) -> Dict[str, object]:
    feature_groups = _load_feature_groups(run_dir, [group])
    features = feature_groups[group]
    ckpt_path = run_dir / "checkpoints" / group / "multi_quantile.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing multi_quantile checkpoint: {ckpt_path}")
    pred_path = run_dir / "prediction_cache" / f"{group}_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction cache: {pred_path}")

    print(f"[SHAP] Preparing group={group}, n_features={len(features)}", flush=True)
    cache = pd.read_csv(pred_path, parse_dates=["datetime"])
    cache_keyed = cache.copy()
    cache_keyed["_occurrence"] = cache_keyed.groupby("datetime").cumcount()
    frame = test_df[features + [label_col, target]].replace([np.inf, -np.inf], np.nan).dropna(subset=features + [label_col])
    frame = frame.copy()
    frame["datetime"] = frame.index
    frame["_occurrence"] = frame.groupby("datetime").cumcount()
    aligned = cache_keyed[["datetime", "_occurrence", "final_pred", "Q90"]].merge(
        frame.reset_index(drop=True),
        on=["datetime", "_occurrence"],
        how="inner",
        sort=False,
    )
    if len(aligned) == 0:
        raise RuntimeError(f"No aligned rows between rebuilt features and prediction cache for {group}.")
    frame = aligned.drop(columns=["_occurrence"]).copy()
    frame["y_true"] = frame[label_col].astype(float)
    frame["gic_true"] = frame[target].astype(float)

    y_for_sampling = frame["y_true"].to_numpy(dtype=float)
    bg_idx = _stratified_sample_indices(y_for_sampling, background_size, seed)
    explain_idx = _stratified_sample_indices(y_for_sampling, explain_size, seed + 11)
    background = frame.iloc[bg_idx][features].to_numpy(dtype=np.float32, copy=True)
    explain_frame = frame.iloc[explain_idx].copy()
    x_explain = explain_frame[features].to_numpy(dtype=np.float32, copy=True)

    model = DeepMultiQuantileRegressor("model3", [0.80, 0.90, 0.95])
    model.load_checkpoint(str(ckpt_path))
    shap_values = _expected_gradients(
        model=model,
        x_original=x_explain,
        background_original=background,
        quantile=quantile,
        nsamples=nsamples,
        batch_size=batch_size,
        seed=seed,
    )

    out_dir = run_dir / "shap_analysis" / group
    out_dir.mkdir(parents=True, exist_ok=True)
    shap_frame = pd.DataFrame(shap_values, columns=features)
    sample_out = explain_frame[["datetime", "y_true", "gic_true", "final_pred", "Q90"] + features].reset_index(drop=True)
    sample_out.to_csv(out_dir / f"{group}_shap_sample_features.csv", index=False, encoding="utf-8-sig")
    shap_frame.insert(0, "datetime", sample_out["datetime"])
    shap_frame.to_csv(out_dir / f"{group}_shap_values.csv", index=False, encoding="utf-8-sig")

    importance = pd.DataFrame(
        {
            "feature": features,
            "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
            "mean_shap": np.mean(shap_values, axis=0),
            "std_shap": np.std(shap_values, axis=0),
            "mean_feature_value": np.nanmean(x_explain, axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(out_dir / f"{group}_shap_feature_importance.csv", index=False, encoding="utf-8-sig")
    plot_importance = _filter_plot_importance(importance, exclude_plot_patterns)
    plot_importance.to_csv(
        out_dir / f"{group}_shap_feature_importance_for_plots.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _write_summary_plot(plot_importance, out_dir, group, top_n)
    _write_dependence_plots(sample_out, shap_frame.drop(columns=["datetime"]), plot_importance, out_dir, group, top_n)

    return {
        "group": group,
        "n_features": len(features),
        "n_background": len(background),
        "n_explained": len(x_explain),
        "nsamples": nsamples,
        "quantile": quantile,
        "top_feature": str(plot_importance.iloc[0]["feature"]) if len(plot_importance) else "",
        "top_mean_abs_shap": float(plot_importance.iloc[0]["mean_abs_shap"]) if len(plot_importance) else np.nan,
        "excluded_plot_patterns": ",".join(exclude_plot_patterns),
        "out_dir": str(out_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument("--target", default="gic_vyk_abs")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--groups", nargs="+", default=[DEFAULT_GROUP])
    parser.add_argument("--lags", nargs="*", type=int, default=None)
    parser.add_argument("--rolling-windows", nargs="+", type=int, default=[30])
    parser.add_argument("--rolling-stats", nargs="+", default=["mean"])
    parser.add_argument("--paper-driver-types", nargs="+", default=["ALL"])
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--paper-pre-context-min", type=int, default=720)
    parser.add_argument("--paper-post-context-min", type=int, default=720)
    parser.add_argument("--quantile", type=float, default=0.90)
    parser.add_argument("--background-size", type=int, default=96)
    parser.add_argument("--explain-size", type=int, default=512)
    parser.add_argument("--nsamples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--exclude-plot-patterns", nargs="*", default=[r"^hour_", r"^doy_"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = _load_feature_groups(args.run_dir, args.groups)
    lags = args.lags if args.lags else _parse_lags_from_groups(list(groups))
    test_df, label_col = _build_test_features(
        data_path=args.data,
        feature_cache=args.feature_cache,
        target=args.target,
        horizon=args.horizon,
        groups=groups,
        lags=lags,
        rolling_windows=args.rolling_windows,
        rolling_stats=args.rolling_stats,
        train_ratio=args.train_ratio,
        paper_driver_types=args.paper_driver_types,
        paper_pre_context_min=args.paper_pre_context_min,
        paper_post_context_min=args.paper_post_context_min,
    )
    rows = []
    for group in groups:
        rows.append(
            run_group_shap(
                run_dir=args.run_dir,
                group=group,
                test_df=test_df,
                label_col=label_col,
                target=args.target,
                quantile=args.quantile,
                nsamples=args.nsamples,
                background_size=args.background_size,
                explain_size=args.explain_size,
                batch_size=args.batch_size,
                top_n=args.top_n,
                exclude_plot_patterns=args.exclude_plot_patterns,
                seed=args.seed,
            )
        )
    summary = pd.DataFrame(rows)
    out_path = args.run_dir / "shap_analysis" / "shap_analysis_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
