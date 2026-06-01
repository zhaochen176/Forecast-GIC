"""
Feature engineering utilities for GIC forecasting.
"""
from __future__ import annotations

import gc
import re
from typing import List, Optional

import numpy as np
import pandas as pd

from src.config import (
    LAG_MINUTES,
    LAG_FEATURE_COLS,
    ROLLING_WINDOWS,
    ROLLING_FEATURE_COLS,
    ROLLING_STATS,
    FEATURE_SET_DEFINITIONS,
    TARGET_COLUMNS,
    RAW_TARGET_COLUMNS,
    TARGET_DBHDT_FEATURE_COL,
    EVENT_TYPE_FEATURE_COLS,
)


def _to_float32(df: pd.DataFrame) -> pd.DataFrame:
    float64_cols = df.select_dtypes(include=["float64"]).columns
    if len(float64_cols) > 0:
        df[float64_cols] = df[float64_cols].astype(np.float32)
    return df


def add_lag_features(
    df: pd.DataFrame,
    cols: List[str] = LAG_FEATURE_COLS,
    lags: List[int] = LAG_MINUTES,
) -> pd.DataFrame:
    new_cols = {}
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col]
        for lag in lags:
            new_cols[f"{col}_lag{lag}"] = s.shift(lag).astype(np.float32)

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        gc.collect()
    return df


def add_rolling_features(
    df: pd.DataFrame,
    cols: List[str] = ROLLING_FEATURE_COLS,
    windows: List[int] = ROLLING_WINDOWS,
    stats: List[str] = ROLLING_STATS,
) -> pd.DataFrame:
    new_cols = {}
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col]
        for win in windows:
            roller = s.rolling(window=win, min_periods=1)
            for stat in stats:
                name = f"{col}_roll{win}_{stat}"
                if stat == "mean":
                    new_cols[name] = roller.mean().astype(np.float32)
                elif stat == "std":
                    new_cols[name] = roller.std().fillna(0).astype(np.float32)
                elif stat == "max":
                    new_cols[name] = roller.max().astype(np.float32)
                elif stat == "min":
                    new_cols[name] = roller.min().astype(np.float32)

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        gc.collect()
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    hour = (df.index.hour + df.index.minute / 60.0).astype(np.float32)
    day_of_year = df.index.dayofyear.astype(np.float32)

    new_cols = pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24.0).astype(np.float32),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0).astype(np.float32),
            "doy_sin": np.sin(2 * np.pi * day_of_year / 365.25).astype(np.float32),
            "doy_cos": np.cos(2 * np.pi * day_of_year / 365.25).astype(np.float32),
        },
        index=df.index,
    )
    return pd.concat([df, new_cols], axis=1)


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {}

    if "Bz_gse" in df.columns:
        new_cols["Bz_south"] = df["Bz_gse"].clip(upper=0).abs().astype(np.float32)

    if "dX_pert_dt" in df.columns and "dY_pert_dt" in df.columns:
        new_cols["dH_dt_magnitude"] = np.sqrt(
            df["dX_pert_dt"] ** 2 + df["dY_pert_dt"] ** 2
        ).astype(np.float32)

    if "Newell" in df.columns:
        new_cols["Newell_diff30"] = (
            df["Newell"] - df["Newell"].shift(30)
        ).astype(np.float32)

    if not new_cols:
        return df
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def add_advanced_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {}

    for col in ["dX_pert_dt", "dY_pert_dt", "dH_pert_dt"]:
        if col in df.columns:
            new_cols[f"d2_{col}"] = df[col].diff().fillna(0).astype(np.float32)

    if "By_gse" in df.columns and "Bz_gse" in df.columns:
        clock_angle = np.arctan2(df["By_gse"].values, df["Bz_gse"].values)
        new_cols["imf_clock_sin"] = np.sin(clock_angle).astype(np.float32)
        new_cols["imf_clock_cos"] = np.cos(clock_angle).astype(np.float32)

    if "Vp" in df.columns and "Bz_gse" in df.columns:
        bz_south = df["Bz_gse"].clip(upper=0).abs()
        new_cols["Vp_Bz_south"] = (df["Vp"] * bz_south).astype(np.float32)

    if "Bz_gse" in df.columns:
        bz_neg = df["Bz_gse"].clip(upper=0)
        for win in [30, 60]:
            new_cols[f"Bz_south_cumsum_{win}"] = (
                bz_neg.rolling(win, min_periods=1).sum().astype(np.float32)
            )

    for col in ["dH_pert_dt", "dX_pert_dt"]:
        if col in df.columns:
            abs_col = df[col].abs()
            for win in [5, 15]:
                new_cols[f"{col}_absmax_{win}"] = (
                    abs_col.rolling(win, min_periods=1).max().astype(np.float32)
                )

    if "P_dyn_nPa" in df.columns:
        new_cols["dP_dyn_dt"] = df["P_dyn_nPa"].diff().fillna(0).astype(np.float32)

    if not new_cols:
        return df
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def engineer_features(df: pd.DataFrame, save_path: Optional[str] = None) -> pd.DataFrame:
    df = _to_float32(df)

    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_temporal_features(df)
    df = add_interaction_features(df)
    df = add_advanced_physics_features(df)

    max_lag = max(LAG_MINUTES + ROLLING_WINDOWS)
    df = df.iloc[max_lag:].reset_index(drop=False)
    if "index" in df.columns:
        df = df.rename(columns={"index": "datetime"}).set_index("datetime")

    if df.isnull().sum().sum() > 0:
        df = df.fillna(0)

    if save_path:
        df.to_parquet(save_path)
    return df


def _infer_feature_root(col: str) -> str:
    lag_match = re.match(r"^(.*)_lag\d+$", col)
    if lag_match:
        return lag_match.group(1)

    roll_match = re.match(r"^(.*)_roll\d+_(mean|std|max|min)$", col)
    if roll_match:
        return roll_match.group(1)

    if col.startswith("d2_dX_pert_dt"):
        return "dX_pert_dt"
    if col.startswith("d2_dY_pert_dt"):
        return "dY_pert_dt"
    if col.startswith("d2_dH_pert_dt"):
        return "dH_pert_dt"

    if col.startswith("imf_clock"):
        return "Bz_gse"
    if col.startswith("Vp_Bz_south"):
        return "Vp"
    if col.startswith("Bz_south"):
        return "Bz_gse"
    if col.startswith("Newell_diff30"):
        return "Newell"
    if col.startswith("dP_dyn_dt"):
        return "P_dyn_nPa"
    if col.startswith("dH_dt_magnitude"):
        return "dH_pert_dt"
    if col.startswith("dH_pert_dt_absmax"):
        return "dH_pert_dt"
    if col.startswith("dX_pert_dt_absmax"):
        return "dX_pert_dt"

    return col


def get_feature_columns(
    df: pd.DataFrame,
    feature_set: Optional[str] = None,
    target_name: Optional[str] = None,
) -> List[str]:
    base_exclude = {
        "gic",
        "gic_abs",
        "density_missing",
        "filled_by_model",
        *TARGET_COLUMNS,
    }
    exclude_raw_targets = base_exclude | set(RAW_TARGET_COLUMNS)

    if feature_set is None:
        return [c for c in df.columns if c not in exclude_raw_targets]

    if feature_set not in FEATURE_SET_DEFINITIONS:
        raise ValueError(
            f"Unknown feature_set={feature_set}, choices={list(FEATURE_SET_DEFINITIONS.keys())}"
        )

    allowed_roots = set(FEATURE_SET_DEFINITIONS[feature_set])
    exclude_cols = set(exclude_raw_targets)
    if feature_set == "D":
        # D group explicitly keeps |dBH/dt| feature as an input feature.
        exclude_cols.discard(TARGET_DBHDT_FEATURE_COL)

    all_feature_cols = [c for c in df.columns if c not in exclude_cols]
    always_keep = {"hour_sin", "hour_cos", "doy_sin", "doy_cos"}
    always_keep.update(c for c in EVENT_TYPE_FEATURE_COLS if c in df.columns)

    selected = []
    for col in all_feature_cols:
        if col in always_keep:
            selected.append(col)
            continue
        root = _infer_feature_root(col)
        if root in allowed_roots:
            selected.append(col)

    return selected
