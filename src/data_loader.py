"""
GIC 预测项目 - 数据加载与预处理模块

包含:
1) 基础 parquet 数据加载
2) Loukhi 站秒级 GIC 自动下载 + 聚合到分钟级(|GIC|最大值)
3) 三目标构造: gic_vyk_abs / gic_lou_abs / dbhdt_abs
"""
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from src.config import (
    ALL_FEATURE_COLS,
    DATA_END,
    DATA_FILE,
    DATA_START,
    DBHDT_SOURCE_CANDIDATES,
    DEFAULT_STORM_TARGET_COL,
    LOUKHI_BASE_URL,
    LOUKHI_CACHE_FILE,
    LOUKHI_RAW_DIR,
    LOUKHI_REQUEST_TIMEOUT,
    LOUKHI_RETRY_SLEEP,
    LOUKHI_RETRY_TIMES,
    TARGET_COL,
    TARGET_COLUMNS,
    TARGET_DBHDT_COL,
    TARGET_DBHDT_FEATURE_COL,
    TARGET_DBHDT_RAW_COL,
    TARGET_LOU_COL,
    TARGET_LOU_RAW_COL,
    TARGET_MAX_CLIP,
    USE_TARGET_CLIP,
    TARGET_VYK_COL,
    TARGET_VYK_RAW_COL,
    TRAIN_END,
    USE_RATIO_SPLIT,
    VAL_END,
    SPLIT_RATIOS,
    DRIVER_SW_ROLL_MIN,
    DRIVER_GM_ROLL_MIN,
    DRIVER_PROPAGATION_MIN,
    DRIVER_BZ_THRESHOLD,
    DRIVER_EY_THRESHOLD,
    DRIVER_HIGH_QUANTILE,
    DRIVER_DBHDT_THRESHOLD,
    DRIVER_MERGE_GAP_MIN,
    DRIVER_PRE_CONTEXT_MIN,
    DRIVER_POST_CONTEXT_MIN,
    DRIVER_MIN_EVENT_MIN,
    FIXED_VKH_PRE_DAYS,
    FIXED_VKH_POST_DAYS,
    FIXED_VKH_QUALITY_FILTER,
    FIXED_VKH_QUALITY_CORE_HOURS,
    FIXED_VKH_MIN_CORE_PEAK_A,
    FIXED_VKH_MAX_CORE_DENSITY_MISSING,
    FIXED_VKH_MAX_CORE_FILLED_RATIO,
    EVENT_TYPE_COL,
    EVENT_TYPE_FEATURE_COLS,
    EVENT_TYPE_RAW_COL,
    EVENT_TYPE_SPLIT_RATIOS,
    EVENT_TYPE_TRAIN_TYPES,
)

LOUKHI_SENTINEL_VALUE = 99999.9
LOUKHI_SENTINEL_ATOL = 1.0
LOUKHI_INVALID_ABS_THRESHOLD = 1e4


def sanitize_loukhi_raw_series(series: pd.Series) -> Tuple[pd.Series, int]:
    """
    Sanitize Loukhi raw GIC:
    - parse numeric
    - sentinel-like values around 99999.9 -> NaN
    - any unrealistic huge magnitude (>= 1e4) -> NaN
    """
    s = pd.to_numeric(series, errors="coerce").astype(np.float32)
    bad_mask = np.isclose(s, LOUKHI_SENTINEL_VALUE, atol=LOUKHI_SENTINEL_ATOL) | (
        np.abs(s) >= LOUKHI_INVALID_ABS_THRESHOLD
    )
    bad_count = int(np.sum(bad_mask))
    if bad_count > 0:
        s = s.mask(bad_mask, np.nan)
    return s, bad_count


def sanitize_loukhi_column_inplace(df: pd.DataFrame, column: str = TARGET_LOU_RAW_COL) -> int:
    if column not in df.columns:
        return 0
    cleaned, bad_count = sanitize_loukhi_raw_series(df[column])
    df[column] = cleaned
    return bad_count


def _pick_absmax_keep_sign(values: pd.Series) -> float:
    """
    Pick the sample with largest absolute magnitude and keep its sign.
    """
    if values is None or len(values) == 0:
        return np.nan
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float32)
    if arr.size == 0:
        return np.nan
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return np.nan
    arr = arr[finite_mask]
    idx = int(np.argmax(np.abs(arr)))
    return float(arr[idx])


def _get_month_days(year: int, month: int) -> int:
    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)
    return (next_month - datetime(year, month, 1)).days


def _download_file(url: str, local_path: str) -> bool:
    """
    下载单文件，支持简单断点续传和重试。
    返回 True 表示成功或已完整存在。
    """
    for attempt in range(LOUKHI_RETRY_TIMES):
        try:
            headers = {}
            if os.path.exists(local_path):
                existing_size = os.path.getsize(local_path)
                if existing_size > 0:
                    headers["Range"] = f"bytes={existing_size}-"
                else:
                    existing_size = 0
            else:
                existing_size = 0

            resp = requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=LOUKHI_REQUEST_TIMEOUT,
            )
            if resp.status_code == 416:
                return True
            if resp.status_code == 404:
                return False
            if resp.status_code in (200, 206):
                mode = "ab" if existing_size > 0 else "wb"
                with open(local_path, mode) as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                return True

            print(
                f"[Loukhi 下载] HTTP {resp.status_code} | "
                f"重试 {attempt + 1}/{LOUKHI_RETRY_TIMES}"
            )
            time.sleep(LOUKHI_RETRY_SLEEP)
        except Exception as exc:
            print(
                f"[Loukhi 下载] 异常: {exc} | "
                f"重试 {attempt + 1}/{LOUKHI_RETRY_TIMES}"
            )
            time.sleep(LOUKHI_RETRY_SLEEP)
    return False


def _download_loukhi_raw(start_year: int = 2012, end_year: int = 2022) -> None:
    os.makedirs(LOUKHI_RAW_DIR, exist_ok=True)
    total = 0
    ok = 0
    skipped = 0
    failed = 0

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            days = _get_month_days(year, month)
            month_str = f"{year}-{month:02d}"
            for day in range(1, days + 1):
                total += 1
                filename = f"{year}{month:02d}{day:02d}.txt"
                url = f"{LOUKHI_BASE_URL}{year}/{month_str}/{filename}"
                local_dir = os.path.join(LOUKHI_RAW_DIR, str(year), month_str)
                os.makedirs(local_dir, exist_ok=True)
                local_path = os.path.join(local_dir, filename)

                if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                    skipped += 1
                    continue

                if _download_file(url, local_path):
                    ok += 1
                else:
                    failed += 1
                    if os.path.exists(local_path):
                        try:
                            os.remove(local_path)
                        except OSError:
                            pass

    print(
        f"[Loukhi 下载] 总日期={total}, 成功={ok}, 跳过={skipped}, 失败={failed}"
    )


def _build_loukhi_minute_cache() -> pd.DataFrame:
    """
    将原始秒级 txt 文件聚合成分钟级:
    每分钟取 |GIC| 最大对应的原始值（保留正负号，不取绝对值）。
    """
    parts = []
    txt_files = 0
    for root, _, files in os.walk(LOUKHI_RAW_DIR):
        for file in files:
            if not file.endswith(".txt"):
                continue
            txt_files += 1
            path = os.path.join(root, file)
            try:
                raw = pd.read_csv(
                    path,
                    sep=r"\s+",
                    header=None,
                    names=["year", "month", "day", "hour", "minute", "second", "gic"],
                )
                raw["gic"], bad_count = sanitize_loukhi_raw_series(raw["gic"])
                if bad_count > 0:
                    print(f"[Loukhi 聚合] 异常值清洗: {path} | bad={bad_count}")

                ts = pd.to_datetime(raw[["year", "month", "day", "hour", "minute"]])
                ts = ts + pd.to_timedelta(raw["second"], unit="s")
                sec_df = pd.DataFrame(
                    {"timestamp": ts, "gic": raw["gic"].astype(np.float32)}
                ).dropna(subset=["gic"])
                if sec_df.empty:
                    continue
                sec_df = sec_df.sort_values("timestamp")
                sec_df["minute"] = sec_df["timestamp"].dt.floor("min")
                sec_df["gic_abs"] = sec_df["gic"].abs()
                idx = sec_df.groupby("minute")["gic_abs"].idxmax()
                one_min = (
                    sec_df.loc[idx, ["minute", "gic"]]
                    .set_index("minute")
                    .sort_index()
                    .rename(columns={"gic": TARGET_LOU_RAW_COL})
                )
                parts.append(one_min)
            except Exception as exc:
                print(f"[Loukhi 聚合] 读取失败: {path} | {exc}")

    if not parts:
        raise RuntimeError("Loukhi 原始文件为空或无法解析，无法构建分钟级缓存。")

    lou = pd.concat(parts).sort_index()
    lou = (
        lou.groupby(level=0)[TARGET_LOU_RAW_COL]
        .apply(_pick_absmax_keep_sign)
        .to_frame(TARGET_LOU_RAW_COL)
    )
    lou = lou.loc[DATA_START:DATA_END]

    os.makedirs(os.path.dirname(LOUKHI_CACHE_FILE), exist_ok=True)
    lou.to_parquet(LOUKHI_CACHE_FILE)
    valid = lou[TARGET_LOU_RAW_COL].dropna()
    if len(valid) > 0:
        neg_ratio = float((valid < 0).mean())
        print(
            f"[Loukhi 聚合] 值域检查: min={valid.min():.4f}, max={valid.max():.4f}, "
            f"neg_ratio={neg_ratio:.4%}"
        )
    print(
        f"[Loukhi 聚合] 文件数={txt_files}, 分钟数据={len(lou):,}, "
        f"缓存={LOUKHI_CACHE_FILE}"
    )
    return lou


def load_loukhi_minute_data() -> pd.DataFrame:
    if os.path.exists(LOUKHI_CACHE_FILE):
        lou = pd.read_parquet(LOUKHI_CACHE_FILE)
        if not isinstance(lou.index, pd.DatetimeIndex):
            lou.index = pd.to_datetime(lou.index)
        if TARGET_LOU_RAW_COL not in lou.columns and len(lou.columns) >= 1:
            lou = lou.rename(columns={lou.columns[0]: TARGET_LOU_RAW_COL})
        bad_count = sanitize_loukhi_column_inplace(lou, TARGET_LOU_RAW_COL)
        if bad_count > 0:
            print(f"[Loukhi 缓存] 检测到历史异常值并已清洗: bad={bad_count}")
            try:
                lou.to_parquet(LOUKHI_CACHE_FILE)
                print(f"[Loukhi 缓存] 已回写清洗后的缓存: {LOUKHI_CACHE_FILE}")
            except Exception as exc:
                print(f"[Loukhi 缓存] 回写失败(不影响继续运行): {exc}")

        # Legacy cache guard: old versions stored abs() minute max.
        lou_vals = lou[TARGET_LOU_RAW_COL].dropna()
        if len(lou_vals) > 0 and bool((lou_vals >= 0).all()):
            print("[Loukhi 缓存] 检测到疑似旧版绝对值缓存，自动从原始文件重建分钟级缓存。")
            try:
                lou = _build_loukhi_minute_cache()
            except Exception as exc:
                print(f"[Loukhi 缓存] 自动重建失败，继续使用现有缓存: {exc}")
        return lou.sort_index()

    print("[Loukhi] 未找到分钟缓存，开始自动下载并聚合...")
    _download_loukhi_raw(2012, 2022)
    try:
        return _build_loukhi_minute_cache()
    except Exception as exc:
        print(f"[Loukhi] 下载/聚合失败，将返回空列并由上层决定是否跳过实验: {exc}")
        return pd.DataFrame(columns=[TARGET_LOU_RAW_COL])


def load_raw_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    加载基础 parquet，并自动拼接 Loukhi 分钟数据。
    返回 DatetimeIndex 的 DataFrame。
    """
    filepath = filepath or DATA_FILE
    print(f"[数据加载] 正在读取基础数据: {filepath}")
    df = pd.read_parquet(filepath)

    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")
        else:
            df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    df = df.loc[DATA_START:DATA_END]

    lou = load_loukhi_minute_data()
    lou = lou.loc[DATA_START:DATA_END]
    df = df.join(lou.reindex(df.index), how="left")

    print(
        f"[数据加载] 形状={df.shape}, 时间范围={df.index.min()} ~ {df.index.max()}"
    )
    return df


def print_missing_report(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(4)
    report = pd.DataFrame({"缺失数": missing, "缺失比例(%)": missing_pct})
    report = report[report["缺失数"] > 0].sort_values("缺失比例(%)", ascending=False)
    print("\n[缺失值报告]")
    if len(report) == 0:
        print("  无缺失值")
    else:
        print(report.to_string())
    return report


def handle_missing_values(df: pd.DataFrame, max_gap: int = 5) -> pd.DataFrame:
    print(f"\n[缺失值处理] 处理前缺失总数: {df.isnull().sum().sum()}")

    # Keep Loukhi raw missing points as NaN (do not interpolate sentinel-missing).
    target_fill_candidates = [TARGET_COL]
    cols_to_fill = ALL_FEATURE_COLS + target_fill_candidates
    cols_to_fill = [c for c in cols_to_fill if c in df.columns]

    df[cols_to_fill] = df[cols_to_fill].interpolate(
        method="linear", limit=max_gap, limit_direction="both"
    )
    df[cols_to_fill] = df[cols_to_fill].ffill(limit=max_gap)
    df[cols_to_fill] = df[cols_to_fill].bfill(limit=max_gap)

    remaining = df[cols_to_fill].isnull().sum().sum()
    print(f"[缺失值处理] 插值+填充后剩余缺失: {remaining}")

    if remaining > 0:
        for col in cols_to_fill:
            if df[col].isnull().any():
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                df[col] = df[col].fillna(median_val)

    float64_cols = df.select_dtypes(include=["float64"]).columns
    if len(float64_cols) > 0:
        df[float64_cols] = df[float64_cols].astype(np.float32)
        print(f"[缺失值处理] 已将 {len(float64_cols)} 列转为 float32")

    print(f"[缺失值处理] 处理后缺失总数: {df.isnull().sum().sum()}")
    return df


def _resolve_dbhdt_source(df: pd.DataFrame) -> str:
    for col in DBHDT_SOURCE_CANDIDATES:
        if col in df.columns:
            return col
    raise KeyError(
        f"未找到 dBH/dt 代理列，可选: {DBHDT_SOURCE_CANDIDATES}"
    )


def prepare_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    构造三目标并统一截断到 20A:
    - gic_vyk_abs
    - gic_lou_abs
    - dbhdt_abs
    """
    if TARGET_COL not in df.columns:
        raise KeyError(f"基础数据缺少目标列: {TARGET_COL}")
    if TARGET_LOU_RAW_COL not in df.columns:
        print("[目标处理] 警告: 缺少 Loukhi 列，将填充为 NaN（对应实验可在主流程跳过）")
        df[TARGET_LOU_RAW_COL] = np.nan

    # Raw series are stored for distribution analysis and export.
    df[TARGET_VYK_RAW_COL] = df[TARGET_COL].astype(np.float32)
    if TARGET_LOU_RAW_COL in df.columns:
        bad_count = sanitize_loukhi_column_inplace(df, TARGET_LOU_RAW_COL)
        if bad_count > 0:
            print(f"[目标处理] Loukhi 原始列异常值清洗: bad={bad_count}")
    else:
        df[TARGET_LOU_RAW_COL] = np.nan

    # Training targets: default no clip (only abs). Optional clip is configurable.
    if USE_TARGET_CLIP:
        df[TARGET_VYK_COL] = df[TARGET_VYK_RAW_COL].abs().clip(0, TARGET_MAX_CLIP)
        df[TARGET_LOU_COL] = df[TARGET_LOU_RAW_COL].abs().clip(0, TARGET_MAX_CLIP)
    else:
        df[TARGET_VYK_COL] = df[TARGET_VYK_RAW_COL].abs()
        df[TARGET_LOU_COL] = df[TARGET_LOU_RAW_COL].abs()

    try:
        dbhdt_col = _resolve_dbhdt_source(df)
    except KeyError:
        dbhdt_col = None
    if dbhdt_col is not None:
        df[TARGET_DBHDT_RAW_COL] = df[dbhdt_col].astype(np.float32)
        if USE_TARGET_CLIP:
            df[TARGET_DBHDT_COL] = df[TARGET_DBHDT_RAW_COL].abs().clip(0, TARGET_MAX_CLIP)
        else:
            df[TARGET_DBHDT_COL] = df[TARGET_DBHDT_RAW_COL].abs()
        df[TARGET_DBHDT_FEATURE_COL] = df[TARGET_DBHDT_RAW_COL].abs().astype(np.float32)

    for col in TARGET_COLUMNS:
        df[col] = df[col].astype(np.float32)

    # 兼容旧流程默认风暴筛选列
    df["gic_abs"] = df[TARGET_VYK_COL]

    print("\n[目标处理]")
    for col in TARGET_COLUMNS:
        print(
            f"  {col}: mean={df[col].mean():.4f}, "
            f"max={df[col].max():.4f}, q95={df[col].quantile(0.95):.4f}"
        )
    return df


def split_by_time(
    df: pd.DataFrame,
    train_end: str = TRAIN_END,
    val_end: str = VAL_END,
    use_ratio_split: bool = USE_RATIO_SPLIT,
    split_ratios: Tuple[float, float, float] = SPLIT_RATIOS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def _safe_ratio_split(
        src_df: pd.DataFrame, ratios: Tuple[float, float, float]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        n = len(src_df)
        if n == 0:
            return src_df.copy(), src_df.copy(), src_df.copy()
        if n < 3:
            return src_df.copy(), src_df.iloc[0:0].copy(), src_df.iloc[0:0].copy()

        r_train, r_val, r_test = [max(float(x), 0.0) for x in ratios]
        total = r_train + r_val + r_test
        if total <= 0:
            r_train, r_val, r_test = 0.7, 0.15, 0.15
            total = 1.0
        r_train, r_val, r_test = r_train / total, r_val / total, r_test / total

        i_train = int(n * r_train)
        i_val = int(n * (r_train + r_val))
        i_train = min(max(i_train, 1), n - 2)
        i_val = min(max(i_val, i_train + 1), n - 1)

        return (
            src_df.iloc[:i_train].copy(),
            src_df.iloc[i_train:i_val].copy(),
            src_df.iloc[i_val:].copy(),
        )

    if use_ratio_split:
        train_df, val_df, test_df = _safe_ratio_split(df, split_ratios)
        split_mode = f"ratio={split_ratios}"
    else:
        train_df = df[df.index <= train_end].copy()
        val_df = df[(df.index > train_end) & (df.index <= val_end)].copy()
        test_df = df[df.index > val_end].copy()
        split_mode = f"time(train_end={train_end}, val_end={val_end})"

        # Fallback for short date windows where fixed dates may create empty splits.
        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            print("[数据划分] 固定日期切分出现空集合，自动退回比例切分。")
            train_df, val_df, test_df = _safe_ratio_split(df, split_ratios)
            split_mode = f"ratio_fallback={split_ratios}"

    def _range_text(part: pd.DataFrame) -> str:
        if len(part) == 0:
            return "EMPTY"
        return f"{part.index.min()} ~ {part.index.max()}"

    print("\n[数据划分]")
    print(f"  mode: {split_mode}")
    print(
        f"  训练集: {_range_text(train_df)}, "
        f"共 {len(train_df):,} 条"
    )
    print(
        f"  验证集: {_range_text(val_df)}, "
        f"共 {len(val_df):,} 条"
    )
    print(
        f"  测试集: {_range_text(test_df)}, "
        f"共 {len(test_df):,} 条"
    )
    return train_df, val_df, test_df


def _bool_mask_to_intervals(
    index: pd.DatetimeIndex, mask: np.ndarray
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if len(index) == 0 or len(mask) == 0:
        return []
    arr = np.asarray(mask, dtype=bool)
    if len(arr) != len(index):
        raise ValueError("mask length must equal index length")
    if not arr.any():
        return []

    changes = np.diff(arr.astype(np.int8))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0]
    if arr[0]:
        starts = np.r_[0, starts]
    if arr[-1]:
        ends = np.r_[ends, len(arr) - 1]

    return [(index[int(s)], index[int(e)]) for s, e in zip(starts, ends)]


def _merge_intervals(
    intervals: List[Tuple[pd.Timestamp, pd.Timestamp]],
    max_gap: pd.Timedelta,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        gap = start - prev_end
        if gap <= max_gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _subtract_intervals(
    intervals: List[Tuple[pd.Timestamp, pd.Timestamp]],
    blocked: List[Tuple[pd.Timestamp, pd.Timestamp]],
    step: pd.Timedelta = pd.Timedelta(minutes=1),
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Subtract blocked intervals from intervals and return remaining disjoint pieces.
    """
    if not intervals:
        return []
    if not blocked:
        return _merge_intervals(intervals, max_gap=pd.Timedelta(minutes=0))

    blocked_sorted = _merge_intervals(blocked, max_gap=pd.Timedelta(minutes=0))
    remaining: List[Tuple[pd.Timestamp, pd.Timestamp]] = []

    for start, end in intervals:
        parts = [(start, end)]
        for b_start, b_end in blocked_sorted:
            next_parts: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
            for p_start, p_end in parts:
                if b_end < p_start or b_start > p_end:
                    next_parts.append((p_start, p_end))
                    continue
                # overlap exists
                if p_start < b_start:
                    left_end = b_start - step
                    if left_end >= p_start:
                        next_parts.append((p_start, left_end))
                if p_end > b_end:
                    right_start = b_end + step
                    if right_start <= p_end:
                        next_parts.append((right_start, p_end))
            parts = next_parts
            if not parts:
                break
        remaining.extend(parts)

    return _merge_intervals(remaining, max_gap=pd.Timedelta(minutes=0))


def _sum_interval_rows(
    index: pd.DatetimeIndex,
    intervals: List[Tuple[pd.Timestamp, pd.Timestamp]],
) -> int:
    total = 0
    for start, end in intervals:
        sl = index.slice_indexer(start, end)
        start_i = 0 if sl.start is None else int(sl.start)
        stop_i = len(index) if sl.stop is None else int(sl.stop)
        total += max(stop_i - start_i, 0)
    return total


def extract_driver_chain_periods(
    df: pd.DataFrame,
    sw_roll_min: int = DRIVER_SW_ROLL_MIN,
    gm_roll_min: int = DRIVER_GM_ROLL_MIN,
    propagation_min: int = DRIVER_PROPAGATION_MIN,
    bz_threshold: float = DRIVER_BZ_THRESHOLD,
    ey_threshold: float = DRIVER_EY_THRESHOLD,
    high_quantile: float = DRIVER_HIGH_QUANTILE,
    dbhdt_threshold: float = DRIVER_DBHDT_THRESHOLD,
    merge_gap_min: int = DRIVER_MERGE_GAP_MIN,
    pre_context_min: int = DRIVER_PRE_CONTEXT_MIN,
    post_context_min: int = DRIVER_POST_CONTEXT_MIN,
    min_event_min: int = DRIVER_MIN_EVENT_MIN,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    事件驱动筛选(无 SYM-H):
      SW(t - tau) AND GM(t)
    其中 SW 基于 Bz/Ey/Newell/epsilon(或代理耦合)，GM 基于 |dH/dt|。
    """
    if len(df) == 0:
        return []

    idx = df.index
    sw_conditions = {}

    # SW condition 1: Bz rolling mean <= threshold.
    if "Bz_gse" in df.columns:
        bz = pd.to_numeric(df["Bz_gse"], errors="coerce")
        bz_roll = bz.rolling(sw_roll_min, min_periods=1).mean()
        sw_conditions["Bz"] = (bz_roll <= bz_threshold).fillna(False)

    # SW condition 2: Ey rolling mean >= threshold.
    if "Ey_mV/m" in df.columns:
        ey = pd.to_numeric(df["Ey_mV/m"], errors="coerce")
        ey_roll = ey.rolling(sw_roll_min, min_periods=1).mean()
        sw_conditions["Ey"] = (ey_roll >= ey_threshold).fillna(False)

    # SW condition 3: coupling quantile (Newell / epsilon / fallback proxy).
    used_coupling = False
    for col in ("Newell", "epsilon_norm"):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            s_roll = s.rolling(sw_roll_min, min_periods=1).mean()
            valid = s_roll.dropna()
            if len(valid) > 0:
                thr = float(valid.quantile(high_quantile))
                sw_conditions[col] = (s_roll >= thr).fillna(False)
                used_coupling = True

    if not used_coupling and "Vp" in df.columns and "Bz_gse" in df.columns:
        vp = pd.to_numeric(df["Vp"], errors="coerce")
        bz = pd.to_numeric(df["Bz_gse"], errors="coerce")
        coupling_proxy = vp * (-bz.clip(upper=0))
        cp_roll = coupling_proxy.rolling(sw_roll_min, min_periods=1).mean()
        valid = cp_roll.dropna()
        if len(valid) > 0:
            thr = float(valid.quantile(high_quantile))
            sw_conditions["Vp*Bz_south"] = (cp_roll >= thr).fillna(False)

    if not sw_conditions:
        print("[驱动链筛选] 缺少可用太阳风驱动列，回退为全时段。")
        return [(idx.min(), idx.max())]

    sw_flag = np.zeros(len(df), dtype=bool)
    for cond in sw_conditions.values():
        sw_flag |= cond.to_numpy(dtype=bool)
    sw_lagged = pd.Series(sw_flag, index=idx).shift(
        int(propagation_min), fill_value=False
    ).to_numpy(dtype=bool)

    gm_col = None
    for c in DBHDT_SOURCE_CANDIDATES:
        if c in df.columns:
            gm_col = c
            break
    if gm_col is None:
        if TARGET_DBHDT_RAW_COL in df.columns:
            gm_col = TARGET_DBHDT_RAW_COL
        else:
            print("[驱动链筛选] 缺少 dH/dt 列，回退为全时段。")
            return [(idx.min(), idx.max())]

    gm = pd.to_numeric(df[gm_col], errors="coerce").abs()
    gm_roll = gm.rolling(int(gm_roll_min), min_periods=1).max()
    gm_flag = (gm_roll >= dbhdt_threshold).fillna(False).to_numpy(dtype=bool)

    event_flag = sw_lagged & gm_flag
    intervals = _bool_mask_to_intervals(idx, event_flag)
    if not intervals:
        print("[驱动链筛选] 未提取到事件时段，回退为全时段。")
        return [(idx.min(), idx.max())]

    merged = _merge_intervals(
        intervals, max_gap=pd.Timedelta(minutes=int(merge_gap_min))
    )
    pre_td = pd.Timedelta(minutes=int(pre_context_min))
    post_td = pd.Timedelta(minutes=int(post_context_min))
    min_td = pd.Timedelta(minutes=int(min_event_min))

    expanded = []
    data_start = idx.min()
    data_end = idx.max()
    for start, end in merged:
        s = max(data_start, start - pre_td)
        e = min(data_end, end + post_td)
        if e - s >= min_td:
            expanded.append((s, e))

    if not expanded:
        print("[驱动链筛选] 事件时段扩展后为空，回退为全时段。")
        return [(idx.min(), idx.max())]

    final_intervals = _merge_intervals(expanded, max_gap=pd.Timedelta(minutes=0))

    total_rows = 0
    for start, end in final_intervals:
        sl = idx.slice_indexer(start, end)
        start_i = 0 if sl.start is None else sl.start
        stop_i = len(df) if sl.stop is None else sl.stop
        total_rows += max(stop_i - start_i, 0)

    sw_ratio = float(sw_flag.mean())
    gm_ratio = float(gm_flag.mean())
    event_ratio = float(event_flag.mean())
    print(
        f"[驱动链筛选] SW占比={sw_ratio:.2%}, GM占比={gm_ratio:.2%}, "
        f"SW(t-tau)&GM占比={event_ratio:.2%}, 时段数={len(final_intervals)}, rows={total_rows:,}"
    )
    return final_intervals


def extract_storm_periods(
    df: pd.DataFrame,
    gic_threshold: float = 10.0,
    context_hours: int = 48,
    target_col: str = DEFAULT_STORM_TARGET_COL,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    gic = df[target_col] if target_col in df.columns else df[TARGET_COL].abs()
    context_td = pd.Timedelta(hours=context_hours)

    peak_times = df.index[gic >= gic_threshold]
    if len(peak_times) == 0:
        print(f"[风暴筛选] 警告: 无 {target_col} >= {gic_threshold}A 的数据点")
        if len(df) == 0:
            return []
        return [(df.index.min(), df.index.max())]

    intervals = list(zip(peak_times - context_td, peak_times + context_td))
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    total_rows = 0
    for start, end in merged:
        sl = df.index.slice_indexer(start, end)
        start_i = 0 if sl.start is None else sl.start
        stop_i = len(df) if sl.stop is None else sl.stop
        total_rows += max(stop_i - start_i, 0)

    print(
        f"[风暴筛选] target={target_col}, 阈值={gic_threshold}A, "
        f"时段数={len(merged)}, 行数={total_rows:,}"
    )
    return merged


def build_time_index_from_intervals(
    df: pd.DataFrame,
    intervals: List[Tuple[pd.Timestamp, pd.Timestamp]],
) -> pd.DatetimeIndex:
    if not intervals:
        return df.index[:0].copy()

    parts = []
    for start, end in intervals:
        sl = df.index.slice_indexer(start, end)
        start_i = 0 if sl.start is None else sl.start
        stop_i = len(df) if sl.stop is None else sl.stop
        if stop_i > start_i:
            parts.append(df.index[start_i:stop_i])
    if not parts:
        return df.index[:0].copy()
    if len(parts) == 1:
        return parts[0].copy()
    return parts[0].append(parts[1:])


def _compute_driver_chain_flags_for_report(
    df: pd.DataFrame,
    sw_roll_min: int,
    gm_roll_min: int,
    propagation_min: int,
    bz_threshold: float,
    ey_threshold: float,
    high_quantile: float,
    dbhdt_threshold: float,
) -> Dict[str, np.ndarray]:
    idx = df.index
    n = len(df)
    if n == 0:
        return {
            "sw_lagged": np.zeros(0, dtype=bool),
            "gm_flag": np.zeros(0, dtype=bool),
            "chain_flag": np.zeros(0, dtype=bool),
            "gm_abs": np.zeros(0, dtype=np.float32),
        }

    sw_conditions = {}
    if "Bz_gse" in df.columns:
        bz = pd.to_numeric(df["Bz_gse"], errors="coerce")
        bz_roll = bz.rolling(int(sw_roll_min), min_periods=1).mean()
        sw_conditions["Bz"] = (bz_roll <= bz_threshold).fillna(False)
    if "Ey_mV/m" in df.columns:
        ey = pd.to_numeric(df["Ey_mV/m"], errors="coerce")
        ey_roll = ey.rolling(int(sw_roll_min), min_periods=1).mean()
        sw_conditions["Ey"] = (ey_roll >= ey_threshold).fillna(False)

    used_coupling = False
    for col in ("Newell", "epsilon_norm"):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            s_roll = s.rolling(int(sw_roll_min), min_periods=1).mean()
            valid = s_roll.dropna()
            if len(valid) > 0:
                thr = float(valid.quantile(high_quantile))
                sw_conditions[col] = (s_roll >= thr).fillna(False)
                used_coupling = True

    if not used_coupling and "Vp" in df.columns and "Bz_gse" in df.columns:
        vp = pd.to_numeric(df["Vp"], errors="coerce")
        bz = pd.to_numeric(df["Bz_gse"], errors="coerce")
        coupling_proxy = vp * (-bz.clip(upper=0))
        cp_roll = coupling_proxy.rolling(int(sw_roll_min), min_periods=1).mean()
        valid = cp_roll.dropna()
        if len(valid) > 0:
            thr = float(valid.quantile(high_quantile))
            sw_conditions["Vp*Bz_south"] = (cp_roll >= thr).fillna(False)

    sw_flag = np.zeros(n, dtype=bool)
    for cond in sw_conditions.values():
        sw_flag |= cond.to_numpy(dtype=bool)
    sw_lagged = pd.Series(sw_flag, index=idx).shift(
        int(propagation_min), fill_value=False
    ).to_numpy(dtype=bool)

    gm_col = None
    for c in DBHDT_SOURCE_CANDIDATES:
        if c in df.columns:
            gm_col = c
            break
    if gm_col is None and TARGET_DBHDT_RAW_COL in df.columns:
        gm_col = TARGET_DBHDT_RAW_COL

    if gm_col is None:
        gm_abs = np.zeros(n, dtype=np.float32)
        gm_flag = np.zeros(n, dtype=bool)
    else:
        gm_abs_series = pd.to_numeric(df[gm_col], errors="coerce").abs()
        gm_abs = gm_abs_series.to_numpy(dtype=np.float32)
        gm_roll = gm_abs_series.rolling(int(gm_roll_min), min_periods=1).max()
        gm_flag = (gm_roll >= dbhdt_threshold).fillna(False).to_numpy(dtype=bool)

    chain_flag = sw_lagged & gm_flag
    return {
        "sw_lagged": sw_lagged,
        "gm_flag": gm_flag,
        "chain_flag": chain_flag,
        "gm_abs": gm_abs,
    }


def _split_counts_by_ratio(
    n_events: int,
    split_ratios: Tuple[float, float, float] = SPLIT_RATIOS,
) -> Tuple[int, int, int]:
    if n_events <= 0:
        return 0, 0, 0

    ratios = np.asarray(split_ratios, dtype=np.float64)
    ratios = np.clip(ratios, a_min=0.0, a_max=None)
    if ratios.sum() <= 0:
        ratios = np.array([0.8, 0.1, 0.1], dtype=np.float64)
    ratios = ratios / ratios.sum()

    raw = ratios * float(n_events)
    counts = np.floor(raw).astype(int)
    remainder = int(n_events - counts.sum())
    if remainder > 0:
        frac = raw - counts
        order = list(np.argsort(-frac))
        for i in range(remainder):
            counts[order[i % len(order)]] += 1

    if n_events >= 1 and counts[0] == 0:
        give_from = int(np.argmax(counts))
        if counts[give_from] > 1:
            counts[give_from] -= 1
        counts[0] += 1

    if n_events >= 3:
        for dst in (1, 2):
            if counts[dst] == 0:
                give_from = int(np.argmax(counts))
                if counts[give_from] > 1:
                    counts[give_from] -= 1
                    counts[dst] += 1

    c_train, c_val, c_test = [int(x) for x in counts]
    adjust = n_events - (c_train + c_val + c_test)
    if adjust != 0:
        c_train += adjust
    return c_train, c_val, c_test


def split_driver_chain_events(
    df: pd.DataFrame,
    split_ratios: Tuple[float, float, float] = SPLIT_RATIOS,
    target_col: str = DEFAULT_STORM_TARGET_COL,
    sw_roll_min: int = DRIVER_SW_ROLL_MIN,
    gm_roll_min: int = DRIVER_GM_ROLL_MIN,
    propagation_min: int = DRIVER_PROPAGATION_MIN,
    bz_threshold: float = DRIVER_BZ_THRESHOLD,
    ey_threshold: float = DRIVER_EY_THRESHOLD,
    high_quantile: float = DRIVER_HIGH_QUANTILE,
    dbhdt_threshold: float = DRIVER_DBHDT_THRESHOLD,
    merge_gap_min: int = DRIVER_MERGE_GAP_MIN,
    pre_context_min: int = DRIVER_PRE_CONTEXT_MIN,
    post_context_min: int = DRIVER_POST_CONTEXT_MIN,
    min_event_min: int = DRIVER_MIN_EVENT_MIN,
) -> Tuple[
    List[Tuple[pd.Timestamp, pd.Timestamp]],
    List[Tuple[pd.Timestamp, pd.Timestamp]],
    List[Tuple[pd.Timestamp, pd.Timestamp]],
    pd.DataFrame,
]:
    if len(df) == 0:
        empty_report = pd.DataFrame(
            columns=[
                "event_id", "start", "end", "duration_min", "n_rows",
                "sw_hit_ratio", "gm_hit_ratio", "chain_hit_ratio",
                "max_abs_dbhdt", "max_target_abs", "split",
            ]
        )
        return [], [], [], empty_report

    intervals = extract_driver_chain_periods(
        df=df,
        sw_roll_min=sw_roll_min,
        gm_roll_min=gm_roll_min,
        propagation_min=propagation_min,
        bz_threshold=bz_threshold,
        ey_threshold=ey_threshold,
        high_quantile=high_quantile,
        dbhdt_threshold=dbhdt_threshold,
        merge_gap_min=merge_gap_min,
        pre_context_min=pre_context_min,
        post_context_min=post_context_min,
        min_event_min=min_event_min,
    )
    intervals = sorted(intervals, key=lambda x: x[0])

    n_events = len(intervals)
    c_train, c_val, c_test = _split_counts_by_ratio(n_events, split_ratios=split_ratios)
    train_intervals = intervals[:c_train]
    val_intervals = intervals[c_train:c_train + c_val]
    test_intervals = intervals[c_train + c_val:c_train + c_val + c_test]

    split_labels = (
        ["train"] * len(train_intervals) +
        ["val"] * len(val_intervals) +
        ["test"] * len(test_intervals)
    )
    split_intervals = train_intervals + val_intervals + test_intervals

    flags = _compute_driver_chain_flags_for_report(
        df=df,
        sw_roll_min=sw_roll_min,
        gm_roll_min=gm_roll_min,
        propagation_min=propagation_min,
        bz_threshold=bz_threshold,
        ey_threshold=ey_threshold,
        high_quantile=high_quantile,
        dbhdt_threshold=dbhdt_threshold,
    )
    sw_lagged = flags["sw_lagged"]
    gm_flag = flags["gm_flag"]
    chain_flag = flags["chain_flag"]
    gm_abs = flags["gm_abs"]

    records = []
    for i, ((start, end), split_name) in enumerate(zip(split_intervals, split_labels), start=1):
        sl = df.index.slice_indexer(start, end)
        start_i = 0 if sl.start is None else int(sl.start)
        stop_i = len(df) if sl.stop is None else int(sl.stop)
        n_rows = max(stop_i - start_i, 0)

        if n_rows > 0:
            sw_hit_ratio = float(np.mean(sw_lagged[start_i:stop_i]))
            gm_hit_ratio = float(np.mean(gm_flag[start_i:stop_i]))
            chain_hit_ratio = float(np.mean(chain_flag[start_i:stop_i]))
            max_abs_dbhdt = float(np.nanmax(gm_abs[start_i:stop_i]))
        else:
            sw_hit_ratio = float("nan")
            gm_hit_ratio = float("nan")
            chain_hit_ratio = float("nan")
            max_abs_dbhdt = float("nan")

        if n_rows > 0 and target_col in df.columns:
            target_slice = pd.to_numeric(df[target_col].iloc[start_i:stop_i], errors="coerce")
            max_target_abs = float(np.nanmax(np.abs(target_slice.to_numpy(dtype=np.float32))))
        else:
            max_target_abs = float("nan")

        duration_min = float(max((end - start) / pd.Timedelta(minutes=1), 0.0))
        records.append(
            {
                "event_id": i,
                "start": start,
                "end": end,
                "duration_min": duration_min,
                "n_rows": n_rows,
                "sw_hit_ratio": sw_hit_ratio,
                "gm_hit_ratio": gm_hit_ratio,
                "chain_hit_ratio": chain_hit_ratio,
                "max_abs_dbhdt": max_abs_dbhdt,
                "max_target_abs": max_target_abs,
                "split": split_name,
            }
        )

    events_report = pd.DataFrame.from_records(records)
    print(
        f"[DriverChainSplit] events={n_events}, "
        f"train/val/test={len(train_intervals)}/{len(val_intervals)}/{len(test_intervals)}"
    )
    return train_intervals, val_intervals, test_intervals, events_report


def save_events_report(events_report: pd.DataFrame, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    events_report.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"[DriverChainSplit] events_report saved: {save_path}")


VKH_92_EVENT_PEAKS: List[Tuple[int, str]] = [
    (1, "2012-03-09 05:47:00"), (2, "2012-03-09 13:55:00"), (3, "2012-04-24 02:47:00"),
    (4, "2012-06-05 21:20:00"), (5, "2012-06-12 01:31:00"), (6, "2012-09-05 06:49:00"),
    (7, "2012-10-09 01:19:00"), (8, "2012-10-14 20:51:00"), (9, "2012-11-14 01:01:00"),
    (10, "2013-02-23 20:31:00"), (11, "2013-03-17 15:57:00"), (12, "2013-03-21 03:41:00"),
    (13, "2013-03-27 17:42:00"), (14, "2013-05-01 19:24:00"), (15, "2013-05-18 02:10:00"),
    (16, "2013-06-01 03:49:00"), (17, "2013-06-07 02:21:00"), (18, "2013-06-29 02:11:00"),
    (19, "2013-07-06 03:00:00"), (20, "2013-07-11 14:11:00"), (21, "2013-10-02 04:36:00"),
    (22, "2013-10-02 20:22:00"), (23, "2013-10-09 06:31:00"), (24, "2013-11-15 21:58:00"),
    (25, "2013-12-14 19:03:00"), (26, "2014-02-20 04:10:00"), (27, "2014-02-27 22:27:00"),
    (28, "2014-09-13 06:27:00"), (29, "2014-09-26 16:04:00"), (30, "2014-10-14 22:40:00"),
    (31, "2014-10-20 16:35:00"), (32, "2014-11-04 20:44:00"), (33, "2014-12-29 21:29:00"),
    (34, "2015-01-04 03:15:00"), (35, "2015-03-17 04:46:00"), (36, "2015-03-17 23:13:00"),
    (37, "2015-03-18 19:13:00"), (38, "2015-03-22 08:27:00"), (39, "2015-04-02 21:28:00"),
    (40, "2015-04-15 13:28:00"), (41, "2015-05-11 01:54:00"), (42, "2015-06-09 03:22:00"),
    (43, "2015-09-08 01:31:00"), (44, "2015-10-07 18:55:00"), (45, "2015-12-14 21:35:00"),
    (46, "2015-12-20 16:14:00"), (47, "2016-01-01 05:02:00"), (48, "2016-01-21 20:28:00"),
    (49, "2016-01-24 19:01:00"), (50, "2016-03-15 20:18:00"), (51, "2016-04-14 16:13:00"),
    (52, "2016-05-08 03:10:00"), (53, "2016-05-30 21:59:00"), (54, "2016-06-14 22:11:00"),
    (55, "2016-06-24 01:25:00"), (56, "2016-07-20 07:29:00"), (57, "2016-08-02 23:05:00"),
    (58, "2016-09-01 19:44:00"), (59, "2016-09-02 01:06:00"), (60, "2016-09-27 14:57:00"),
    (61, "2016-10-13 16:02:00"), (62, "2016-10-25 17:26:00"), (63, "2016-10-27 16:02:00"),
    (64, "2016-12-21 16:04:00"), (65, "2016-12-23 18:10:00"), (66, "2016-12-25 17:58:00"),
    (67, "2017-03-27 19:26:00"), (68, "2017-05-28 02:21:00"), (69, "2017-09-07 23:22:00"),
    (70, "2017-09-08 15:47:00"), (71, "2017-09-13 00:07:00"), (72, "2017-09-15 19:11:00"),
    (73, "2017-09-16 17:45:00"), (74, "2017-09-28 02:47:00"), (75, "2017-10-13 18:45:00"),
    (76, "2017-10-26 15:05:00"), (77, "2017-11-21 00:35:00"), (78, "2017-11-21 17:11:00"),
    (79, "2017-11-22 18:30:00"), (80, "2017-12-17 18:53:00"), (81, "2018-04-13 21:14:00"),
    (82, "2018-04-20 19:23:00"), (83, "2018-05-06 01:13:00"), (84, "2018-05-10 17:17:00"),
    (85, "2018-06-01 15:56:00"), (86, "2018-08-26 03:13:00"), (87, "2019-01-24 21:15:00"),
    (88, "2021-03-01 19:41:00"), (89, "2021-10-12 03:21:00"), (90, "2022-06-15 14:42:00"),
    (91, "2022-11-27 19:12:00"), (92, "2022-11-30 18:48:00"),
]

VKH_92_EVENT_STORM_TYPES: Dict[int, str] = {
    1: "CME, MC", 2: "CME, MC", 3: "CME, MC", 4: "CIR", 5: "CIR",
    6: "CME, EJ", 7: "CME, MC", 8: "CME, EJ", 9: "CME, MC", 10: "No",
    11: "CME, EJ", 12: "CME", 13: "CIR", 14: "CME, EJ", 15: "CME, EJ",
    16: "CME, EJ", 17: "CME, MC", 18: "CME, EJ", 19: "CME, EJ", 20: "CME, EJ",
    21: "SC, CME, MC", 22: "CME, MC", 23: "CME, EJ", 24: "CIR", 25: "CME, EJ",
    26: "CME, EJ", 27: "CME, EJ", 28: "CME, MC", 29: "No", 30: "CIR",
    31: "CIR", 32: "CIR", 33: "No", 34: "CME, EJ", 35: "SC, CME, MC",
    36: "CME, MC", 37: "CME, MC", 38: "CME, EJ, SI", 39: "No", 40: "CIR",
    41: "CME, MC", 42: "CME, EJ", 43: "CME, MC", 44: "CME", 45: "CME, EJ",
    46: "CME, MC", 47: "CME, EJ", 48: "CME, EJ", 49: "CME, EJ", 50: "CIR",
    51: "CME, EJ", 52: "CME", 53: "No", 54: "No", 55: "CIR",
    56: "SI", 57: "CME, MC", 58: "CIR", 59: "CIR", 60: "CIR",
    61: "CME, MC", 62: "CIR", 63: "CIR", 64: "CIR", 65: "CIR, EJ",
    66: "CIR, EJ", 67: "CIR", 68: "CME, MC", 69: "CME, MC", 70: "CME, EJ",
    71: "CME", 72: "CIR", 73: "CIR", 74: "CIR", 75: "CIR",
    76: "CIR", 77: "CIR", 78: "CIR", 79: "CIR", 80: "No",
    81: "CIR", 82: "CME", 83: "CIR", 84: "CIR", 85: "CIR",
    86: "CME, MC", 87: "CIR", 88: "CME", 89: "CIR", 90: "No",
    91: "CIR", 92: "CME",
}


def coarse_vkh_event_type(storm_type: str) -> str:
    s = str(storm_type or "").upper()
    if "SC" in s or "SI" in s:
        return "SC_SI"
    if "CME" in s:
        return "CME"
    if "CIR" in s:
        return "CIR"
    if "NO" in s:
        return "NO_WEAK"
    return "OTHER"

VKH_CME_SPLIT_IDS: Dict[str, List[int]] = {
    "train": [
        1, 2, 3, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18, 19, 20, 22, 23, 25, 26,
        27, 28, 34, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 57, 61,
    ],
    "val": [65, 66, 68, 69, 70],
    "test": [71, 82, 86, 88, 92],
}

_VKH_PEAK_MAP: Dict[int, str] = {event_id: peak for event_id, peak in VKH_92_EVENT_PEAKS}
VKH_CME_EVENT_SPLIT: Dict[str, List[Tuple[int, str]]] = {
    split: [(event_id, _VKH_PEAK_MAP[event_id]) for event_id in ids]
    for split, ids in VKH_CME_SPLIT_IDS.items()
}


def split_fixed_vkh_cme_events(
    df: pd.DataFrame,
    pre_context_min: Optional[int] = None,
    post_context_min: Optional[int] = None,
    merge_gap_min: int = 0,
    split_ratios: Tuple[float, float, float] = SPLIT_RATIOS,
    quality_filter: bool = FIXED_VKH_QUALITY_FILTER,
    quality_core_hours: float = FIXED_VKH_QUALITY_CORE_HOURS,
    min_core_peak_a: float = FIXED_VKH_MIN_CORE_PEAK_A,
    max_core_density_missing: float = FIXED_VKH_MAX_CORE_DENSITY_MISSING,
    max_core_filled_ratio: float = FIXED_VKH_MAX_CORE_FILLED_RATIO,
) -> Tuple[
    List[Tuple[pd.Timestamp, pd.Timestamp]],
    List[Tuple[pd.Timestamp, pd.Timestamp]],
    List[Tuple[pd.Timestamp, pd.Timestamp]],
    pd.DataFrame,
]:
    if len(df) == 0:
        empty = pd.DataFrame(
            columns=[
                "event_id", "paper_event_id", "split", "peak_time",
                "start", "end", "in_data_range", "n_rows",
                "quality_keep", "quality_reason",
                "core_peak_abs", "core_density_missing", "core_filled_by_model",
            ]
        )
        return [], [], [], empty

    if pre_context_min is None:
        pre_context_min = int(FIXED_VKH_PRE_DAYS * 24 * 60)
    if post_context_min is None:
        post_context_min = int(FIXED_VKH_POST_DAYS * 24 * 60)

    data_start = df.index.min()
    data_end = df.index.max()
    pre_td = pd.Timedelta(minutes=int(pre_context_min))
    post_td = pd.Timedelta(minutes=int(post_context_min))
    core_td = pd.Timedelta(hours=float(max(quality_core_hours, 0.0)))
    # Keep merge gap conservative by default in fixed-event mode.
    max_gap = pd.Timedelta(minutes=max(int(merge_gap_min), 0))

    records = []
    kept_events = []
    quality_target_col = (
        TARGET_VYK_COL
        if TARGET_VYK_COL in df.columns
        else ("gic_abs" if "gic_abs" in df.columns else TARGET_COL)
    )
    density_col = "density_missing" if "density_missing" in df.columns else None
    filled_col = "filled_by_model" if "filled_by_model" in df.columns else None

    for event_id, (paper_event_id, peak_str) in enumerate(VKH_92_EVENT_PEAKS, start=1):
        peak = pd.Timestamp(peak_str)
        in_data_range = bool(data_start <= peak <= data_end)
        quality_keep = bool(in_data_range)
        quality_reason = ""
        core_peak_abs = float("nan")
        core_density_missing = float("nan")
        core_filled_by_model = float("nan")
        split = "out_of_range"

        if in_data_range:
            start = max(data_start, peak - pre_td)
            end = min(data_end, peak + post_td)
            core_start = max(data_start, peak - core_td)
            core_end = min(data_end, peak + core_td)
            core_df = df.loc[core_start:core_end]

            if len(core_df) > 0:
                target_vals = pd.to_numeric(
                    core_df[quality_target_col], errors="coerce"
                ).to_numpy(dtype=np.float32)
                if target_vals.size > 0:
                    core_peak_abs = float(np.nanmax(np.abs(target_vals)))
                if density_col is not None:
                    core_density_missing = float(
                        pd.to_numeric(core_df[density_col], errors="coerce")
                        .fillna(0.0)
                        .mean()
                    )
                if filled_col is not None:
                    core_filled_by_model = float(
                        pd.to_numeric(core_df[filled_col], errors="coerce")
                        .fillna(0.0)
                        .mean()
                    )

            if quality_filter:
                reasons: List[str] = []
                if np.isfinite(core_peak_abs) and core_peak_abs < float(min_core_peak_a):
                    reasons.append(f"core_peak<{float(min_core_peak_a):.2f}A")
                if (
                    np.isfinite(core_density_missing)
                    and core_density_missing > float(max_core_density_missing)
                ):
                    reasons.append(
                        f"core_density_missing>{float(max_core_density_missing):.2f}"
                    )
                if (
                    np.isfinite(core_filled_by_model)
                    and core_filled_by_model > float(max_core_filled_ratio)
                ):
                    reasons.append(
                        f"core_filled_by_model>{float(max_core_filled_ratio):.2f}"
                    )
                quality_keep = len(reasons) == 0
                quality_reason = ";".join(reasons)

            if quality_keep:
                split = "pending"
                kept_events.append(
                    {
                        "paper_event_id": int(paper_event_id),
                        "peak_time": peak,
                        "start": start,
                        "end": end,
                    }
                )
            else:
                split = "dropped"

            sl = df.index.slice_indexer(start, end)
            start_i = 0 if sl.start is None else int(sl.start)
            stop_i = len(df) if sl.stop is None else int(sl.stop)
            n_rows = max(stop_i - start_i, 0)
        else:
            start = pd.NaT
            end = pd.NaT
            n_rows = 0

        records.append(
            {
                "event_id": event_id,
                "paper_event_id": int(paper_event_id),
                "split": split,
                "peak_time": peak,
                "start": start,
                "end": end,
                "in_data_range": in_data_range,
                "n_rows": n_rows,
                "quality_keep": bool(quality_keep),
                "quality_reason": quality_reason,
                "core_peak_abs": core_peak_abs,
                "core_density_missing": core_density_missing,
                "core_filled_by_model": core_filled_by_model,
            }
        )

    kept_events = sorted(kept_events, key=lambda x: x["peak_time"])
    n_events = len(kept_events)
    c_train, c_val, c_test = _split_counts_by_ratio(
        n_events, split_ratios=split_ratios
    )

    train_events = kept_events[:c_train]
    val_events = kept_events[c_train:c_train + c_val]
    test_events = kept_events[c_train + c_val:c_train + c_val + c_test]

    split_map = {}
    for item in train_events:
        split_map[item["paper_event_id"]] = "train"
    for item in val_events:
        split_map[item["paper_event_id"]] = "val"
    for item in test_events:
        split_map[item["paper_event_id"]] = "test"

    for rec in records:
        if rec["split"] == "pending":
            rec["split"] = split_map.get(int(rec["paper_event_id"]), "pending")

    train_raw = _merge_intervals(
        [(e["start"], e["end"]) for e in train_events], max_gap=max_gap
    )
    val_raw = _merge_intervals(
        [(e["start"], e["end"]) for e in val_events], max_gap=max_gap
    )
    test_raw = _merge_intervals(
        [(e["start"], e["end"]) for e in test_events], max_gap=max_gap
    )

    # Ensure split disjointness under long windows:
    # keep test fully, then trim val by test, then trim train by (val + test).
    test_intervals = test_raw
    val_intervals = _subtract_intervals(val_raw, test_intervals)
    train_intervals = _subtract_intervals(train_raw, val_intervals + test_intervals)

    events_report = pd.DataFrame.from_records(records)
    matched = int(events_report["in_data_range"].sum())
    total = int(len(events_report))
    quality_kept = int(events_report["quality_keep"].sum())
    quality_dropped = int(max(matched - quality_kept, 0))
    rows_train = _sum_interval_rows(df.index, train_intervals)
    rows_val = _sum_interval_rows(df.index, val_intervals)
    rows_test = _sum_interval_rows(df.index, test_intervals)
    print(
        "[FixedCME] paper events="
        f"{total}, in-range={matched}, kept={n_events}, split events="
        f"{len(train_events)}/{len(val_events)}/{len(test_events)}, split intervals="
        f"{len(train_intervals)}/{len(val_intervals)}/{len(test_intervals)}"
        f", rows={rows_train}/{rows_val}/{rows_test}"
        f", context(min)={pre_context_min}/{post_context_min}"
        f", quality_filter={quality_filter}, kept={quality_kept}, dropped={quality_dropped}"
    )
    if matched < total:
        print(
            "[FixedCME] warning: some paper events are out of current DATA_START/DATA_END window."
        )

    return train_intervals, val_intervals, test_intervals, events_report


def build_vkh_event_type_report(
    df: pd.DataFrame,
    pre_context_min: Optional[int] = None,
    post_context_min: Optional[int] = None,
    quality_filter: bool = FIXED_VKH_QUALITY_FILTER,
    quality_core_hours: float = FIXED_VKH_QUALITY_CORE_HOURS,
    min_core_peak_a: float = FIXED_VKH_MIN_CORE_PEAK_A,
    max_core_density_missing: float = FIXED_VKH_MAX_CORE_DENSITY_MISSING,
    max_core_filled_ratio: float = FIXED_VKH_MAX_CORE_FILLED_RATIO,
) -> pd.DataFrame:
    if pre_context_min is None:
        pre_context_min = int(FIXED_VKH_PRE_DAYS * 24 * 60)
    if post_context_min is None:
        post_context_min = int(FIXED_VKH_POST_DAYS * 24 * 60)

    if len(df) == 0:
        return pd.DataFrame()

    data_start = df.index.min()
    data_end = df.index.max()
    pre_td = pd.Timedelta(minutes=int(pre_context_min))
    post_td = pd.Timedelta(minutes=int(post_context_min))
    core_td = pd.Timedelta(hours=float(max(quality_core_hours, 0.0)))
    quality_target_col = (
        TARGET_VYK_COL
        if TARGET_VYK_COL in df.columns
        else ("gic_abs" if "gic_abs" in df.columns else TARGET_COL)
    )
    density_col = "density_missing" if "density_missing" in df.columns else None
    filled_col = "filled_by_model" if "filled_by_model" in df.columns else None

    rows = []
    for paper_event_id, peak_str in VKH_92_EVENT_PEAKS:
        peak = pd.Timestamp(peak_str)
        raw_type = VKH_92_EVENT_STORM_TYPES.get(int(paper_event_id), "")
        event_type = coarse_vkh_event_type(raw_type)
        in_data_range = bool(data_start <= peak <= data_end)
        start = pd.NaT
        end = pd.NaT
        n_rows = 0
        quality_keep = bool(in_data_range)
        quality_reason = ""
        core_peak_abs = float("nan")
        core_density_missing = float("nan")
        core_filled_by_model = float("nan")

        if in_data_range:
            start = max(data_start, peak - pre_td)
            end = min(data_end, peak + post_td)
            sl = df.index.slice_indexer(start, end)
            start_i = 0 if sl.start is None else int(sl.start)
            stop_i = len(df) if sl.stop is None else int(sl.stop)
            n_rows = max(stop_i - start_i, 0)

            core_df = df.loc[max(data_start, peak - core_td):min(data_end, peak + core_td)]
            if len(core_df) > 0:
                target_vals = pd.to_numeric(
                    core_df[quality_target_col], errors="coerce"
                ).to_numpy(dtype=np.float32)
                if target_vals.size > 0:
                    core_peak_abs = float(np.nanmax(np.abs(target_vals)))
                if density_col is not None:
                    core_density_missing = float(
                        pd.to_numeric(core_df[density_col], errors="coerce")
                        .fillna(0.0)
                        .mean()
                    )
                if filled_col is not None:
                    core_filled_by_model = float(
                        pd.to_numeric(core_df[filled_col], errors="coerce")
                        .fillna(0.0)
                        .mean()
                    )

            if quality_filter:
                reasons: List[str] = []
                if np.isfinite(core_peak_abs) and core_peak_abs < float(min_core_peak_a):
                    reasons.append(f"core_peak<{float(min_core_peak_a):.2f}A")
                if (
                    np.isfinite(core_density_missing)
                    and core_density_missing > float(max_core_density_missing)
                ):
                    reasons.append(
                        f"core_density_missing>{float(max_core_density_missing):.2f}"
                    )
                if (
                    np.isfinite(core_filled_by_model)
                    and core_filled_by_model > float(max_core_filled_ratio)
                ):
                    reasons.append(
                        f"core_filled_by_model>{float(max_core_filled_ratio):.2f}"
                    )
                quality_keep = len(reasons) == 0
                quality_reason = ";".join(reasons)

        rows.append(
            {
                "paper_event_id": int(paper_event_id),
                "peak_time": peak,
                "start": start,
                "end": end,
                "in_data_range": in_data_range,
                "n_rows": int(n_rows),
                EVENT_TYPE_RAW_COL: raw_type,
                EVENT_TYPE_COL: event_type,
                "quality_keep": bool(quality_keep),
                "quality_reason": quality_reason,
                "core_peak_abs": core_peak_abs,
                "core_density_missing": core_density_missing,
                "core_filled_by_model": core_filled_by_model,
            }
        )

    return pd.DataFrame.from_records(rows)


def add_event_type_features(
    df: pd.DataFrame,
    events_report: pd.DataFrame,
) -> pd.DataFrame:
    out = df.copy()
    for col in EVENT_TYPE_FEATURE_COLS:
        out[col] = np.float32(0.0)

    if events_report is None or events_report.empty:
        return out

    type_to_col = {
        "CME": "event_type_CME",
        "CIR": "event_type_CIR",
        "NO_WEAK": "event_type_NO_WEAK",
        "SC_SI": "event_type_SC_SI",
    }
    keep = events_report[events_report.get("quality_keep", False).astype(bool)]
    for _, row in keep.iterrows():
        col = type_to_col.get(str(row.get(EVENT_TYPE_COL, "")))
        if col is None:
            continue
        start = row.get("start")
        end = row.get("end")
        if pd.isna(start) or pd.isna(end):
            continue
        out.loc[pd.Timestamp(start):pd.Timestamp(end), col] = np.float32(1.0)
    return out


def split_vkh_event_type_events(
    df: pd.DataFrame,
    include_types: Optional[List[str]] = None,
    split_ratios: Tuple[float, float] = EVENT_TYPE_SPLIT_RATIOS,
    pre_context_min: Optional[int] = None,
    post_context_min: Optional[int] = None,
    quality_filter: bool = FIXED_VKH_QUALITY_FILTER,
) -> Tuple[
    List[Tuple[pd.Timestamp, pd.Timestamp]],
    List[Tuple[pd.Timestamp, pd.Timestamp]],
    pd.DataFrame,
]:
    include = set(include_types or EVENT_TYPE_TRAIN_TYPES)
    report = build_vkh_event_type_report(
        df=df,
        pre_context_min=pre_context_min,
        post_context_min=post_context_min,
        quality_filter=quality_filter,
    )
    if report.empty:
        return [], [], report

    eligible = report[
        report["in_data_range"].astype(bool)
        & report["quality_keep"].astype(bool)
        & report[EVENT_TYPE_COL].isin(include)
    ].sort_values("peak_time")

    n_events = len(eligible)
    if n_events == 0:
        report["split"] = "dropped"
        return [], [], report

    train_ratio = float(split_ratios[0])
    total_ratio = max(float(split_ratios[0]) + float(split_ratios[1]), 1e-9)
    n_train = int(round(n_events * train_ratio / total_ratio))
    if n_events >= 2:
        n_train = min(max(n_train, 1), n_events - 1)
    else:
        n_train = 1

    train_ids = set(eligible.iloc[:n_train]["paper_event_id"].astype(int).tolist())
    test_ids = set(eligible.iloc[n_train:]["paper_event_id"].astype(int).tolist())

    split_values = []
    for _, row in report.iterrows():
        event_id = int(row["paper_event_id"])
        if event_id in train_ids:
            split_values.append("train")
        elif event_id in test_ids:
            split_values.append("test")
        elif bool(row.get("quality_keep", False)):
            split_values.append("unused_type")
        else:
            split_values.append("dropped")
    report = report.copy()
    report["split"] = split_values

    def _intervals_for(ids: set) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        rows = report[report["paper_event_id"].astype(int).isin(ids)]
        intervals = [
            (pd.Timestamp(r["start"]), pd.Timestamp(r["end"]))
            for _, r in rows.iterrows()
            if pd.notna(r["start"]) and pd.notna(r["end"])
        ]
        return _merge_intervals(intervals, max_gap=pd.Timedelta(minutes=0))

    test_intervals = _intervals_for(test_ids)
    train_intervals = _subtract_intervals(_intervals_for(train_ids), test_intervals)
    print(
        "[EventTypeSplit] "
        f"types={sorted(include)}, events={n_events}, "
        f"train/test={len(train_ids)}/{len(test_ids)}, "
        f"intervals={len(train_intervals)}/{len(test_intervals)}"
    )
    return train_intervals, test_intervals, report


def build_vkh_92_event_windows(
    df: pd.DataFrame,
    pre_context_min: Optional[int] = None,
    post_context_min: Optional[int] = None,
) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame(
            columns=[
                "paper_event_id", "peak_time", "start", "end", "in_data_range", "n_rows",
            ]
        )

    if pre_context_min is None:
        pre_context_min = int(FIXED_VKH_PRE_DAYS * 24 * 60)
    if post_context_min is None:
        post_context_min = int(FIXED_VKH_POST_DAYS * 24 * 60)

    data_start = df.index.min()
    data_end = df.index.max()
    pre_td = pd.Timedelta(minutes=int(pre_context_min))
    post_td = pd.Timedelta(minutes=int(post_context_min))

    rows = []
    for paper_event_id, peak_str in VKH_92_EVENT_PEAKS:
        peak = pd.Timestamp(peak_str)
        in_data_range = bool(data_start <= peak <= data_end)

        if in_data_range:
            start = max(data_start, peak - pre_td)
            end = min(data_end, peak + post_td)
            sl = df.index.slice_indexer(start, end)
            start_i = 0 if sl.start is None else int(sl.start)
            stop_i = len(df) if sl.stop is None else int(sl.stop)
            n_rows = max(stop_i - start_i, 0)
        else:
            start = pd.NaT
            end = pd.NaT
            n_rows = 0

        rows.append(
            {
                "paper_event_id": paper_event_id,
                "peak_time": peak,
                "start": start,
                "end": end,
                "in_data_range": in_data_range,
                "n_rows": n_rows,
            }
        )

    out = pd.DataFrame(rows)
    print(
        f"[VKH92] total={len(out)}, in_range={int(out['in_data_range'].sum())}, "
        f"out_of_range={int((~out['in_data_range']).sum())}"
    )
    return out


def load_and_preprocess(filepath: Optional[str] = None) -> pd.DataFrame:
    df = load_raw_data(filepath)
    print_missing_report(df)
    df = handle_missing_values(df)
    df = prepare_targets(df)
    return df
