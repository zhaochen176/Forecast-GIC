"""
GIC 预测项目 - PyTorch 数据集模块 (v5)

v5 核心:
  ★ 去掉 log1p 变换, 在原始空间训练
  ★ 对 target 做 RobustScaler (中位数/IQR)
  ★ 3-tuple 输出 (X, y, w), 简化接口
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Tuple, List, Optional
import joblib
import os
import gc

from src.config import (
    SEQ_LEN, PRED_HORIZON, TRAIN_STRIDE, EVAL_STRIDE,
    BATCH_SIZE, PEAK_THRESHOLD_QUANTILE, SEED, MODEL_DIR,
    LOG_TRANSFORM_TARGET, PEAK_STRIDE, TARGET_ROBUST_SCALE,
    NUM_WORKERS, PREFETCH_FACTOR, PEAK_WEIGHT_THRESHOLD, GIC_MAX_CLIP,
    USE_TARGET_CLIP,
    PEAK_LABEL_STRATEGY, PEAK_LABEL_THRESHOLD_A, PEAK_SAMPLER_POWER,
    PEAK_WEIGHT, NORMAL_WEIGHT,
    QUANTILE_EVENT_CANDIDATES, QUANTILE_EVENT_LEVELS,
    QUANTILE_EVENT_MIN_RATIO, QUANTILE_EVENT_MAX_LEVELS,
)

# memmap 临时文件目录
_MEMMAP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "_memmap_cache",
)


def _df_to_memmap(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    target_raw_col: Optional[str],
    name: str,
    intervals: Optional[List[Tuple[pd.Timestamp, pd.Timestamp]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """
    将 DataFrame 逐列写入 memmap 文件，避免一次性分配大数组。

    返回:
        X_mmap: np.memmap (n_rows, n_features) float32，可读写
        y_arr:      np.ndarray (n_rows,) float32，训练标签
        y_raw_arr:  np.ndarray (n_rows,) float32，原始标签（未clip）
        segments:   List[(start, stop)]，memmap 空间中的区间（stop 不含）
    """
    os.makedirs(_MEMMAP_DIR, exist_ok=True)
    n_features = len(feature_cols)

    interval_slices = []
    if intervals:
        for start, end in intervals:
            sl = df.index.slice_indexer(start, end)
            start_i = 0 if sl.start is None else sl.start
            stop_i = len(df) if sl.stop is None else sl.stop
            if stop_i > start_i:
                interval_slices.append((start_i, stop_i))
    else:
        interval_slices = [(0, len(df))]

    n_rows = sum(stop_i - start_i for start_i, stop_i in interval_slices)

    # 创建 memmap 文件
    x_path = os.path.join(_MEMMAP_DIR, f"{name}_X.dat")
    X_mmap = np.memmap(x_path, dtype=np.float32, mode="w+",
                       shape=(n_rows, n_features))
    y_arr = np.empty(n_rows, dtype=np.float32)
    y_raw_arr = np.empty(n_rows, dtype=np.float32)

    # 分区间逐列写入，避免一次性复制大 DataFrame
    col_batch = 10  # 每次处理 10 列
    cursor = 0
    segments: List[Tuple[int, int]] = []
    for start_i, stop_i in interval_slices:
        seg_len = stop_i - start_i
        seg_start = cursor
        seg_stop = cursor + seg_len
        seg_df = df.iloc[start_i:stop_i]
        for i in range(0, n_features, col_batch):
            batch_cols = feature_cols[i:i + col_batch]
            chunk = seg_df[batch_cols].to_numpy(dtype=np.float32, copy=True)
            X_mmap[cursor:cursor + seg_len, i:i + len(batch_cols)] = chunk
            del chunk

        y_arr[cursor:cursor + seg_len] = seg_df[target_col].to_numpy(dtype=np.float32, copy=True)
        if target_raw_col is not None and target_raw_col in seg_df.columns:
            y_raw_arr[cursor:cursor + seg_len] = seg_df[target_raw_col].to_numpy(
                dtype=np.float32, copy=True
            )
        else:
            y_raw_arr[cursor:cursor + seg_len] = seg_df[target_col].to_numpy(
                dtype=np.float32, copy=True
            )
        cursor += seg_len
        segments.append((int(seg_start), int(seg_stop)))
        del seg_df

    X_mmap.flush()

    if USE_TARGET_CLIP:
        y_arr = np.clip(y_arr, 0, GIC_MAX_CLIP)

    # v4: 仅当显式启用时才做 log1p (默认关闭)
    if LOG_TRANSFORM_TARGET:
        y_arr = np.log1p(y_arr)

    return X_mmap, y_arr, y_raw_arr, segments


def _forward_window_max(arr: np.ndarray, window: int) -> np.ndarray:
    """
    前向窗口最大值:
    out[i] = max(arr[i : i+window])
    """
    if window <= 1:
        return arr.astype(np.float32, copy=False)
    rev = pd.Series(arr[::-1])
    out = rev.rolling(window=window, min_periods=1).max().to_numpy()[::-1]
    return out.astype(np.float32, copy=False)


def _forward_window_extreme_signed(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Forward window signed extreme by absolute magnitude:
    out[i] = argmax_{j in [i, i+window-1]} |arr[j]|, keep original sign.
    """
    if window <= 1:
        return arr.astype(np.float32, copy=False)
    rev = pd.Series(arr[::-1])
    fmax = rev.rolling(window=window, min_periods=1).max().to_numpy()[::-1]
    fmin = rev.rolling(window=window, min_periods=1).min().to_numpy()[::-1]
    choose_max = np.abs(fmax) >= np.abs(fmin)
    out = np.where(choose_max, fmax, fmin)
    return out.astype(np.float32, copy=False)


def _quantile_label(q: float) -> str:
    return f"Q{int(round(q * 100))}"


class GICTimeSeriesDataset(Dataset):
    """
    GIC 时间序列滑动窗口数据集 (v5)。
    支持 memmap 数组，按需读取。

    每个样本:
        X: (seq_len, n_features) 过去 seq_len 分钟的特征
        y: (1,)                  当前时刻的 |GIC| (scaled)
        weight: (1,)             样本权重（峰值样本权重更高）
    """

    def __init__(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        targets_raw: np.ndarray,     # 原始空间的 target (未 scale)
        segments: Optional[List[Tuple[int, int]]] = None,
        preserve_signed_window_extreme: bool = False,
        seq_len: int = SEQ_LEN,
        pred_horizon: int = PRED_HORIZON,
        future_window: int = 1,
        stride: int = TRAIN_STRIDE,
        peak_threshold: float = None,
        peak_weight: float = 5.0,
        normal_weight: float = 1.0,
        peak_stride: int = None,
        is_train: bool = False,
    ):
        self.data = data
        self.targets = targets
        self.targets_raw = targets_raw
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon
        self.future_window = max(int(future_window), 1)
        self.stride = stride
        self.is_train = is_train
        self.preserve_signed_window_extreme = bool(preserve_signed_window_extreme)
        if segments:
            self.segments = [(int(s), int(e)) for s, e in segments if int(e) > int(s)]
        else:
            self.segments = [(0, len(data))]

        # Future-window label construction.
        if self.preserve_signed_window_extreme:
            self.targets_window = _forward_window_extreme_signed(
                self.targets, self.future_window
            )
            self.targets_raw_window = _forward_window_extreme_signed(
                self.targets_raw, self.future_window
            )
        else:
            self.targets_window = _forward_window_max(self.targets, self.future_window)
            self.targets_raw_window = _forward_window_max(
                self.targets_raw, self.future_window
            )

        # 有效索引：按区间逐段构建，避免跨事件区间滑窗。
        candidate_chunks: List[np.ndarray] = []
        for seg_start, seg_stop in self.segments:
            seg_len = int(seg_stop - seg_start)
            seg_max_start = seg_len - seq_len - pred_horizon - self.future_window + 1
            if seg_max_start <= 0:
                continue

            if peak_threshold is not None and peak_stride is not None and peak_stride < stride:
                target_lo = seg_start + seq_len + pred_horizon - 1
                target_hi_inclusive = (
                    seg_start + seg_max_start + seq_len + pred_horizon - 2
                )
                target_indices = np.arange(
                    target_lo, target_hi_inclusive + 1, dtype=np.int64
                )
                start_indices = target_indices - seq_len - pred_horizon + 1
                is_peak = self.targets_raw_window[target_indices] >= PEAK_WEIGHT_THRESHOLD

                peak_starts = start_indices[is_peak][::peak_stride]
                normal_starts = start_indices[~is_peak][::stride]
                if len(peak_starts) == 0 and len(normal_starts) == 0:
                    continue
                seg_candidates = np.sort(
                    np.concatenate([peak_starts, normal_starts])
                ).astype(np.int64, copy=False)
            else:
                # seg_max_start is count of valid starts in this segment.
                seg_candidates = np.arange(
                    seg_start, seg_start + seg_max_start, stride, dtype=np.int64
                )
                last_start = seg_start + seg_max_start - 1
                if len(seg_candidates) == 0 or int(seg_candidates[-1]) != int(last_start):
                    seg_candidates = np.r_[seg_candidates, np.array([last_start], dtype=np.int64)]

            candidate_chunks.append(seg_candidates)

        if candidate_chunks:
            candidate_indices = np.concatenate(candidate_chunks)
        else:
            candidate_indices = np.array([], dtype=np.int64)

        # Drop windows with invalid target labels (e.g., Loukhi sentinel -> NaN).
        if len(candidate_indices) > 0:
            target_indices = candidate_indices + self.seq_len + self.pred_horizon - 1
            valid_mask = (
                np.isfinite(self.targets_window[target_indices]) &
                np.isfinite(self.targets_raw_window[target_indices])
            )
            self.indices = candidate_indices[valid_mask].tolist()
        else:
            self.indices = []

        # ★ 幅度缩放权重 — 在原始空间计算, 用 sqrt 压缩避免极端权重
        self.peak_threshold = peak_threshold
        if peak_threshold is not None:
            safe_raw_window = np.nan_to_num(self.targets_raw_window, nan=0.0, posinf=0.0, neginf=0.0)
            excess = np.maximum(
                safe_raw_window - PEAK_WEIGHT_THRESHOLD, 0
            ) / (PEAK_WEIGHT_THRESHOLD + 1e-6)
            self.weights = (normal_weight + peak_weight *
                            np.sqrt(excess)).astype(np.float32)
        else:
            self.weights = np.ones(len(targets), dtype=np.float32)

    def get_sample_weights(self):
        """返回每个窗口的采样权重，用于 WeightedRandomSampler。"""
        sample_w = np.empty(len(self.indices), dtype=np.float32)
        for i, start in enumerate(self.indices):
            target_idx = start + self.seq_len + self.pred_horizon - 1
            sample_w[i] = self.weights[target_idx]
        return sample_w

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        end = start + self.seq_len
        target_idx = end + self.pred_horizon - 1

        X = torch.from_numpy(np.array(self.data[start:end], dtype=np.float32))
        y = torch.tensor(self.targets_window[target_idx])
        w = torch.tensor(self.weights[target_idx])

        return X, y, w

    def get_target_distribution(self):
        """返回当前数据集（按窗口标签）的原始空间分布值。"""
        if not self.indices:
            return np.array([], dtype=np.float32)
        vals = []
        for start in self.indices:
            idx = start + self.seq_len + self.pred_horizon - 1
            vals.append(self.targets_raw_window[idx])
        return np.asarray(vals, dtype=np.float32)

    def get_aligned_raw_targets(self) -> np.ndarray:
        """Return raw window-label sequence aligned with dataset iteration order."""
        if not self.indices:
            return np.array([], dtype=np.float32)
        vals = np.empty(len(self.indices), dtype=np.float32)
        for i, start in enumerate(self.indices):
            idx = start + self.seq_len + self.pred_horizon - 1
            vals[i] = self.targets_raw_window[idx]
        return vals


class CombinedGICTimeSeriesDataset(Dataset):
    """Concatenate multiple GICTimeSeriesDataset objects and expose weighted sampling."""

    def __init__(self, datasets: List[GICTimeSeriesDataset], dataset_weights: Optional[List[float]] = None):
        if not datasets:
            raise ValueError("datasets must not be empty")
        self.datasets = datasets
        self.cum_sizes = np.cumsum([len(ds) for ds in datasets]).tolist()
        self.peak_threshold = datasets[0].peak_threshold
        self.quantile_thresholds = getattr(datasets[0], "quantile_thresholds", {})
        self.quantile_threshold_stats = getattr(datasets[0], "quantile_threshold_stats", [])
        self.target_scaler = getattr(datasets[0], "target_scaler", None)
        self.dataset_weights = dataset_weights or [1.0] * len(datasets)
        if len(self.dataset_weights) != len(datasets):
            raise ValueError("dataset_weights length must match datasets")

    def __len__(self):
        return self.cum_sizes[-1]

    def _locate(self, idx: int):
        ds_idx = int(np.searchsorted(self.cum_sizes, idx, side="right"))
        prev = 0 if ds_idx == 0 else self.cum_sizes[ds_idx - 1]
        return ds_idx, idx - prev

    def __getitem__(self, idx):
        ds_idx, local_idx = self._locate(int(idx))
        return self.datasets[ds_idx][local_idx]

    def get_sample_weights(self):
        weights = []
        for ds, ds_weight in zip(self.datasets, self.dataset_weights):
            if hasattr(ds, "get_sample_weights"):
                base = ds.get_sample_weights()
                weights.append(np.asarray(base, dtype=np.float32) * float(ds_weight))
            else:
                weights.append(np.full(len(ds), float(ds_weight), dtype=np.float32))
        return np.concatenate(weights) if weights else np.array([], dtype=np.float32)


def subset_dataset_by_absolute_intervals(
    dataset: GICTimeSeriesDataset,
    source_index: pd.DatetimeIndex,
    intervals: Optional[List[Tuple[pd.Timestamp, pd.Timestamp]]],
    name: str = "event",
) -> GICTimeSeriesDataset:
    """Create a dataset view restricted to absolute-time intervals.

    The new dataset reuses the already-scaled memmap arrays from ``dataset`` so
    two-stage fine-tuning keeps exactly the same feature and target scalers as
    background-dominant pretraining.
    """
    if not intervals:
        print(f"[TwoStage] {name}: no intervals; reuse original dataset")
        return dataset

    segments: List[Tuple[int, int]] = []
    for start, end in intervals:
        sl = source_index.slice_indexer(pd.Timestamp(start), pd.Timestamp(end))
        start_i = 0 if sl.start is None else int(sl.start)
        stop_i = len(source_index) if sl.stop is None else int(sl.stop)
        if stop_i > start_i:
            segments.append((start_i, stop_i))

    if not segments:
        print(f"[TwoStage] {name}: empty interval slice; reuse original dataset")
        return dataset

    subset = GICTimeSeriesDataset(
        dataset.data,
        dataset.targets,
        dataset.targets_raw,
        segments=segments,
        preserve_signed_window_extreme=dataset.preserve_signed_window_extreme,
        seq_len=dataset.seq_len,
        pred_horizon=dataset.pred_horizon,
        future_window=dataset.future_window,
        stride=dataset.stride,
        peak_threshold=dataset.peak_threshold,
        peak_weight=PEAK_WEIGHT,
        normal_weight=NORMAL_WEIGHT,
        peak_stride=PEAK_STRIDE if dataset.is_train else None,
        is_train=dataset.is_train,
    )
    subset.target_scaler = getattr(dataset, "target_scaler", None)
    print(
        f"[TwoStage] {name}: intervals={len(segments)}, "
        f"windows={len(subset)} / base={len(dataset)}"
    )
    return subset


def split_background_event_intervals(
    full_intervals: List[Tuple[pd.Timestamp, pd.Timestamp]],
    event_intervals: List[Tuple[pd.Timestamp, pd.Timestamp]],
) -> Tuple[List[Tuple[pd.Timestamp, pd.Timestamp]], List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    """Return background complements and normalized event intervals.

    Background intervals are the parts of ``full_intervals`` not covered by
    ``event_intervals``. This is used for two-stage training: background-heavy
    pretraining first, event-focused fine-tuning second.
    """
    full = [
        (pd.Timestamp(start), pd.Timestamp(end))
        for start, end in full_intervals
        if pd.notna(start) and pd.notna(end)
    ]
    ev = [
        (pd.Timestamp(start), pd.Timestamp(end))
        for start, end in event_intervals
        if pd.notna(start) and pd.notna(end)
    ]
    full = sorted([(s, e) for s, e in full if e >= s], key=lambda x: x[0])
    ev = sorted([(s, e) for s, e in ev if e >= s], key=lambda x: x[0])

    bg: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for full_start, full_end in full:
        cursor = full_start
        for ev_start, ev_end in ev:
            if ev_end < full_start or ev_start > full_end:
                continue
            clipped_start = max(ev_start, full_start)
            clipped_end = min(ev_end, full_end)
            if clipped_start > cursor:
                bg.append((cursor, clipped_start - pd.Timedelta(minutes=1)))
            cursor = max(cursor, clipped_end + pd.Timedelta(minutes=1))
            if cursor > full_end:
                break
        if cursor <= full_end:
            bg.append((cursor, full_end))
    return bg, ev


def _fit_scaler_from_memmap(
    X_mmap: np.ndarray,
    save_path: Optional[str] = None,
    chunk_size: int = 200000,
) -> StandardScaler:
    """
    在 memmap 上分块拟合 StandardScaler（Welford 在线算法）。
    """
    n_samples, n_features = X_mmap.shape
    mean = np.zeros(n_features, dtype=np.float64)
    m2 = np.zeros(n_features, dtype=np.float64)
    count = 0

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        chunk = np.array(X_mmap[start:end], dtype=np.float64)
        batch_n = len(chunk)
        batch_mean = chunk.mean(axis=0)
        batch_var = chunk.var(axis=0)

        new_count = count + batch_n
        delta = batch_mean - mean
        mean = mean + delta * batch_n / new_count
        m2 = m2 + batch_var * batch_n + delta ** 2 * count * batch_n / new_count
        count = new_count
        del chunk

    variance = m2 / count
    std = np.sqrt(variance)
    std[std < 1e-8] = 1.0

    scaler = StandardScaler()
    scaler.mean_ = mean
    scaler.scale_ = std
    scaler.var_ = variance
    scaler.n_samples_seen_ = n_samples
    scaler.n_features_in_ = n_features

    if save_path:
        joblib.dump(scaler, save_path)
        print(f"[Scaler] 已保存: {save_path}")
    return scaler


def _scale_memmap_inplace(X_mmap: np.ndarray, mean32, scale32, chunk=200000):
    """就地在 memmap 上分块标准化。"""
    n = len(X_mmap)
    for i in range(0, n, chunk):
        end = min(i + chunk, n)
        X_mmap[i:end] -= mean32
        X_mmap[i:end] /= scale32
    X_mmap.flush() if hasattr(X_mmap, 'flush') else None


def _fit_target_robust_scaler(y_arr: np.ndarray, save_path: Optional[str] = None):
    """
    ★ 对 target 做 RobustScaler: (y - median) / IQR。
    比 StandardScaler 更抗离群值。
    """
    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    scaler.fit(y_arr.reshape(-1, 1))
    if save_path:
        joblib.dump(scaler, save_path)
        print(f"[Target Scaler] center={scaler.center_[0]:.4f}, "
              f"scale={scaler.scale_[0]:.4f}, saved: {save_path}")
    return scaler


def create_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "gic_abs",
    target_raw_col: Optional[str] = None,
    seq_len: int = SEQ_LEN,
    pred_horizon: int = PRED_HORIZON,
    future_window: int = 1,
    train_stride: int = TRAIN_STRIDE,
    eval_stride: int = EVAL_STRIDE,
    train_intervals: Optional[List[Tuple[pd.Timestamp, pd.Timestamp]]] = None,
    val_intervals: Optional[List[Tuple[pd.Timestamp, pd.Timestamp]]] = None,
    test_intervals: Optional[List[Tuple[pd.Timestamp, pd.Timestamp]]] = None,
    cache_prefix: str = "",
    scaler_save_path: Optional[str] = None,
    target_scaler_save_path: Optional[str] = None,
) -> Tuple:
    """
    创建训练/验证/测试数据集 (v4)。
    返回: train_dataset, val_dataset, test_dataset, scaler, target_scaler,
          quantile_thresholds, quantile_threshold_stats
    """
    print("\n[数据集] 创建 PyTorch 数据集 (v5: 原始空间 + 峰值权重)...")
    preserve_signed_window_extreme = (
        (not str(target_col).endswith("_abs")) and int(future_window) > 1
    )
    if preserve_signed_window_extreme:
        print(
            "[数据集] Signed target + future window: "
            "使用窗口内|值|最大且保留符号的标签构造。"
        )

    # --- 逐个 split 提取到 memmap ---
    print("[数据集] 写入训练集 memmap...")
    name_train = f"{cache_prefix}_train" if cache_prefix else "train"
    name_val = f"{cache_prefix}_val" if cache_prefix else "val"
    name_test = f"{cache_prefix}_test" if cache_prefix else "test"

    train_X, train_y, train_y_raw, train_segments = _df_to_memmap(
        train_df, feature_cols, target_col, target_raw_col, name_train, train_intervals
    )
    del train_df
    gc.collect()

    print("[数据集] 写入验证集 memmap...")
    val_X, val_y, val_y_raw, val_segments = _df_to_memmap(
        val_df, feature_cols, target_col, target_raw_col, name_val, val_intervals
    )
    del val_df
    gc.collect()

    print("[数据集] 写入测试集 memmap...")
    test_X, test_y, test_y_raw, test_segments = _df_to_memmap(
        test_df, feature_cols, target_col, target_raw_col, name_test, test_intervals
    )
    del test_df
    gc.collect()

    print(f"[数据集] memmap 形状: 训练={train_X.shape}, "
          f"验证={val_X.shape}, 测试={test_X.shape}")

    # Keep raw labels uncapped for analysis/plotting, but align sign with abs-target tasks.
    if target_col.endswith("_abs"):
        train_y_raw = np.abs(train_y_raw)
        val_y_raw = np.abs(val_y_raw)
        test_y_raw = np.abs(test_y_raw)

    # --- 拟合 Feature Scaler ---
    scaler_path = scaler_save_path or os.path.join(MODEL_DIR, "scaler.pkl")
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    scaler = _fit_scaler_from_memmap(train_X, save_path=scaler_path)

    # --- 就地标准化特征 ---
    mean32 = scaler.mean_.astype(np.float32)
    scale32 = scaler.scale_.astype(np.float32)

    _scale_memmap_inplace(train_X, mean32, scale32)
    _scale_memmap_inplace(val_X, mean32, scale32)
    _scale_memmap_inplace(test_X, mean32, scale32)
    gc.collect()

    # --- ★ Target RobustScaler ---
    target_scaler = None
    if TARGET_ROBUST_SCALE and not LOG_TRANSFORM_TARGET:
        target_scaler_path = (
            target_scaler_save_path or os.path.join(MODEL_DIR, "target_scaler.pkl")
        )
        os.makedirs(os.path.dirname(target_scaler_path), exist_ok=True)
        target_scaler = _fit_target_robust_scaler(
            train_y_raw, save_path=target_scaler_path)
        train_y = target_scaler.transform(
            train_y.reshape(-1, 1)).ravel().astype(np.float32)
        val_y = target_scaler.transform(
            val_y.reshape(-1, 1)).ravel().astype(np.float32)
        test_y = target_scaler.transform(
            test_y.reshape(-1, 1)).ravel().astype(np.float32)
        print(f"[数据集] Target RobustScaler 已应用")

    # --- 峰值阈值/分位数阈值: 使用“有效且不跨区间”的窗口标签分布 ---
    if preserve_signed_window_extreme:
        train_y_raw_window = _forward_window_extreme_signed(train_y_raw, future_window)
    else:
        train_y_raw_window = _forward_window_max(train_y_raw, future_window)
    train_dist_chunks: List[np.ndarray] = []
    for seg_start, seg_stop in train_segments:
        seg_len = int(seg_stop - seg_start)
        seg_max_start = seg_len - seq_len - pred_horizon - future_window + 1
        if seg_max_start <= 0:
            continue
        target_lo = seg_start + seq_len + pred_horizon - 1
        target_hi_inclusive = seg_start + seg_max_start + seq_len + pred_horizon - 2
        train_dist_chunks.append(train_y_raw_window[target_lo:target_hi_inclusive + 1])

    if train_dist_chunks:
        train_dist_raw = np.concatenate(train_dist_chunks)
        train_dist_raw = train_dist_raw[np.isfinite(train_dist_raw)]
    else:
        train_dist_raw = np.array([], dtype=np.float32)
    if PEAK_LABEL_STRATEGY == "fixed":
        peak_threshold_raw = float(PEAK_LABEL_THRESHOLD_A)
        print(f"[数据集] 峰值阈值 (原始空间, fixed): {peak_threshold_raw:.4f} A")
    elif PEAK_LABEL_STRATEGY == "quantile":
        if len(train_dist_raw) == 0:
            valid_raw_window = train_y_raw_window[np.isfinite(train_y_raw_window)]
            if len(valid_raw_window) == 0:
                peak_threshold_raw = float(PEAK_WEIGHT_THRESHOLD)
            else:
                peak_threshold_raw = float(np.quantile(valid_raw_window, PEAK_THRESHOLD_QUANTILE))
        else:
            peak_threshold_raw = float(np.quantile(train_dist_raw, PEAK_THRESHOLD_QUANTILE))
        print(f"[数据集] 峰值阈值 (原始空间, {PEAK_THRESHOLD_QUANTILE*100:.0f}%分位): "
              f"{peak_threshold_raw:.4f} A")
    else:
        raise ValueError(
            f"Unsupported PEAK_LABEL_STRATEGY={PEAK_LABEL_STRATEGY}, "
            "expected one of {'fixed', 'quantile'}."
        )

    if target_scaler is not None:
        peak_threshold = target_scaler.transform([[peak_threshold_raw]])[0, 0]
    elif LOG_TRANSFORM_TARGET:
        peak_threshold = np.log1p(peak_threshold_raw)
    else:
        peak_threshold = peak_threshold_raw

    # --- 创建 Dataset ---
    from src.config import PEAK_WEIGHT, NORMAL_WEIGHT
    train_dataset = GICTimeSeriesDataset(
        train_X, train_y, train_y_raw,
        segments=train_segments,
        preserve_signed_window_extreme=preserve_signed_window_extreme,
        seq_len=seq_len, pred_horizon=pred_horizon,
        future_window=future_window,
        stride=train_stride,
        peak_threshold=peak_threshold,
        peak_weight=PEAK_WEIGHT, normal_weight=NORMAL_WEIGHT,
        peak_stride=PEAK_STRIDE,
        is_train=True,
    )
    val_dataset = GICTimeSeriesDataset(
        val_X, val_y, val_y_raw,
        segments=val_segments,
        preserve_signed_window_extreme=preserve_signed_window_extreme,
        seq_len=seq_len, pred_horizon=pred_horizon,
        future_window=future_window,
        stride=eval_stride,
        peak_threshold=peak_threshold,
        peak_weight=PEAK_WEIGHT, normal_weight=NORMAL_WEIGHT,
    )
    test_dataset = GICTimeSeriesDataset(
        test_X, test_y, test_y_raw,
        segments=test_segments,
        preserve_signed_window_extreme=preserve_signed_window_extreme,
        seq_len=seq_len, pred_horizon=pred_horizon,
        future_window=future_window,
        stride=eval_stride,
        peak_threshold=peak_threshold,
        peak_weight=PEAK_WEIGHT, normal_weight=NORMAL_WEIGHT,
    )

    print(f"[数据集] 训练集样本数: {len(train_dataset)}")
    print(f"[数据集] 验证集样本数: {len(val_dataset)}")
    print(f"[数据集] 测试集样本数: {len(test_dataset)}")
    print(f"[数据集] 特征维度: {train_X.shape[1]}")

    # 保存 target_scaler 到 dataset 属性
    train_dataset.target_scaler = target_scaler
    val_dataset.target_scaler = target_scaler
    test_dataset.target_scaler = target_scaler

    # Quantile threshold candidates + adaptive selection (avoid too-sparse events).
    quantile_threshold_stats = []
    quantile_thresholds = {}
    if len(train_dist_raw) > 0:
        n_dist = len(train_dist_raw)
        for q in QUANTILE_EVENT_CANDIDATES:
            thr = float(np.quantile(train_dist_raw, q))
            event_count = int((train_dist_raw >= thr).sum())
            event_ratio = float(event_count / max(n_dist, 1))
            quantile_threshold_stats.append({
                "quantile": float(q),
                "label": _quantile_label(q),
                "threshold_value": thr,
                "event_count": event_count,
                "event_ratio": event_ratio,
                "selected": False,
            })

        stats_by_q = {float(item["quantile"]): item for item in quantile_threshold_stats}
        valid_q = [
            float(item["quantile"])
            for item in quantile_threshold_stats
            if item["event_ratio"] >= QUANTILE_EVENT_MIN_RATIO
        ]
        max_levels = max(int(QUANTILE_EVENT_MAX_LEVELS), 1)
        selected_q = []

        # Prefer configured levels first.
        for q in QUANTILE_EVENT_LEVELS:
            q = float(q)
            if q in valid_q and q not in selected_q:
                selected_q.append(q)

        # Then fill with other valid candidates from lower to higher quantile.
        for q in valid_q:
            if q not in selected_q:
                selected_q.append(q)
            if len(selected_q) >= max_levels:
                break

        # Fallback: if all candidates are too sparse, still keep the lowest levels.
        if not selected_q:
            selected_q = [float(item["quantile"]) for item in quantile_threshold_stats[:max_levels]]

        selected_q = selected_q[:max_levels]
        quantile_thresholds = {
            stats_by_q[q]["label"]: stats_by_q[q]["threshold_value"]
            for q in selected_q
        }

        for item in quantile_threshold_stats:
            if float(item["quantile"]) in selected_q:
                item["selected"] = True

        picked = ", ".join([
            f"{stats_by_q[q]['label']}={stats_by_q[q]['threshold_value']:.4f}"
            for q in selected_q
        ])
        print(f"[数据集] 分位数阈值(自适应选择): {picked}")
    else:
        print("[数据集] 分位数阈值统计为空：训练分布样本不足。")

    train_dataset.quantile_thresholds = quantile_thresholds
    val_dataset.quantile_thresholds = quantile_thresholds
    test_dataset.quantile_thresholds = quantile_thresholds
    train_dataset.quantile_threshold_stats = quantile_threshold_stats
    val_dataset.quantile_threshold_stats = quantile_threshold_stats
    test_dataset.quantile_threshold_stats = quantile_threshold_stats

    return (
        train_dataset, val_dataset, test_dataset,
        scaler, target_scaler, quantile_thresholds, quantile_threshold_stats
    )


def create_dataloaders(
    train_dataset: GICTimeSeriesDataset,
    val_dataset: GICTimeSeriesDataset,
    test_dataset: GICTimeSeriesDataset,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建 DataLoader (WeightedRandomSampler + 多进程预取)。"""
    import torch
    from torch.utils.data import WeightedRandomSampler
    from src.config import USE_WEIGHTED_SAMPLER

    use_pin = torch.cuda.is_available()
    persist = num_workers > 0
    prefetch = PREFETCH_FACTOR if num_workers > 0 else None

    common = dict(
        num_workers=num_workers, pin_memory=use_pin,
        persistent_workers=persist, prefetch_factor=prefetch,
    )

    # 训练集: WeightedRandomSampler 确保每 batch 含足够峰值样本
    if USE_WEIGHTED_SAMPLER:
        sw = train_dataset.get_sample_weights()
        sw_boosted = sw ** PEAK_SAMPLER_POWER
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sw_boosted).double(),
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler,
            drop_last=True, **common,
        )
        peak_frac = (sw > sw.min() + 0.1).sum() / len(sw)
        print(f"[DataLoader] WeightedRandomSampler 已启用, "
              f"峰值样本占比: {peak_frac:.1%} | power={PEAK_SAMPLER_POWER:.2f}")
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            drop_last=True, **common,
        )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, **common,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, **common,
    )
    print(f"[DataLoader] batch_size={batch_size}, workers={num_workers}, "
          f"训练 {len(train_loader)} batches, "
          f"验证 {len(val_loader)} batches, "
          f"测试 {len(test_loader)} batches")
    return train_loader, val_loader, test_loader
