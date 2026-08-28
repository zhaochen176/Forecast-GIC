"""Train a compact CNN-BiLSTM model for 30-minute VKH GIC forecasts.

This is the primary sequence model, evaluated under exactly the same data,
forward-validation, operational threshold, and event-level protocol as 04.
It consumes the CME/CIR event-window timeline from 03, keeps the physical
lag window [t-L-W+1, t-L] inside each event block, and predicts four
independent probabilities for the future 30-minute absolute-GIC peak
exceeding 3, 5, 10, and 20 A.  Four onset outputs are trained only as an
auxiliary task; operational scoring remains on the peak outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Avoid cuDNN v8 Conv1d plan-selection failures seen with short sequences on
# some CUDA/cuDNN image combinations. This must precede the PyTorch import.
os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
# The 6:2:2 chronological event-block split is the active evaluation
# protocol.  The legacy 7:2:1 output remains on disk for reproducibility but
# is no longer the implicit source for new runs.
DATASET_DIR = ROOT / "data" / "prediction_dataset_6020"
TIMELINE_PATH = DATASET_DIR / "prediction_timeline.csv"
EVENT_CATALOG_PATH = DATASET_DIR / "vkh_event_catalog_selected.csv"
WINDOW_STATISTICS_TEMPLATE = DATASET_DIR / "window_statistics_W{window}.csv"
OUTPUT_ROOT = ROOT / "outputs" / "cnn_bilstm"
THRESHOLDS = (3, 5, 10, 20)
PRIMARY_OUTPUT_COUNT = len(THRESHOLDS)
TOTAL_OUTPUT_COUNT = PRIMARY_OUTPUT_COUNT * 2
EVENT_CUTOFF_QUANTILE_COUNT = 31
EVENT_POSTPROCESS_SETTINGS = (
    # (minimum consecutive, merge gap, minimum final length), in minutes.
    # These settings are selected on forward-validation OOF predictions only.
    (1, 0, 1),
    (2, 2, 2),
    (3, 5, 3),
    (3, 10, 3),
    (5, 5, 5),
    (5, 10, 5),
)
DEFAULT_POSTPROCESS = (3, 5, 3)
DEFAULT_SELECTION_WEIGHTS = (0.45, 0.45, 0.10, 0.0)

RAW_FEATURES = (
    "Btot", "Bx_gsm", "By_gsm", "Bz_gsm", "V", "Np", "Psw", "Ma", "Mms",
    "Epsilon", "VBs", "Bt_gsm", "Bz_south", "Newell_coupling",
    "Borovsky_Rquick_mV_m", "clock_angle_rad",
)
LOG1P_FEATURES = {
    "Btot", "V", "Np", "Psw", "Ma", "Mms", "Epsilon", "VBs", "Bt_gsm",
    "Bz_south", "Newell_coupling", "Borovsky_Rquick_mV_m",
}
TEMPORAL_DELTA_FEATURES = (
    "Bz_gsm", "log1p_V", "log1p_Np", "log1p_Psw", "log1p_Epsilon", "log1p_Newell_coupling",
)
TEMPORAL_FEATURE_GROUPS = frozenset(("delta", "statistics", "accumulation"))
FEATURE_GROUPS = {
    "instantaneous_solar_wind": (
        "log1p_Btot", "Bx_gsm", "By_gsm", "Bz_gsm", "log1p_V", "log1p_Np", "log1p_Psw",
        "log1p_Ma", "log1p_Mms", "log1p_Epsilon", "log1p_VBs", "log1p_Bt_gsm",
        "log1p_Bz_south", "sin_clock_angle", "cos_clock_angle",
    ),
    "coupling_functions": ("log1p_Newell_coupling", "log1p_Borovsky_Rquick_mV_m"),
    "temporal_change": tuple(
        f"delta_{minutes}min_{name}" for name in TEMPORAL_DELTA_FEATURES for minutes in (1, 10)
    ),
    "rolling_statistics": (),  # Populated from W-specific files: moments and physical accumulations.
}


def configure_dataset_paths(dataset_dir: Path) -> Path:
    """Point the shared training protocol at one complete 03 output directory."""
    global DATASET_DIR, TIMELINE_PATH, EVENT_CATALOG_PATH, WINDOW_STATISTICS_TEMPLATE
    DATASET_DIR = dataset_dir if dataset_dir.is_absolute() else ROOT / dataset_dir
    DATASET_DIR = DATASET_DIR.resolve()
    TIMELINE_PATH = DATASET_DIR / "prediction_timeline.csv"
    EVENT_CATALOG_PATH = DATASET_DIR / "vkh_event_catalog_selected.csv"
    WINDOW_STATISTICS_TEMPLATE = DATASET_DIR / "window_statistics_W{window}.csv"
    return DATASET_DIR


def parse_temporal_feature_groups(value: str | None) -> frozenset[str]:
    """Parse a comma-separated temporal feature selection.

    ``accumulation`` is the default because it summarizes the duration and
    integrated strength of the upstream driver without expanding the causal
    lookback beyond the selected sequence window.  ``all`` remains available
    for ablation and backwards-compatible exploratory runs.
    """
    if value is None or not str(value).strip():
        return frozenset(("accumulation",))
    tokens = {token.strip().lower() for token in str(value).split(",") if token.strip()}
    if "all" in tokens:
        tokens.remove("all")
        tokens.update(TEMPORAL_FEATURE_GROUPS)
    if "none" in tokens:
        tokens.remove("none")
        if tokens:
            raise ValueError("temporal feature group 'none' cannot be combined with another group")
    unknown = tokens - TEMPORAL_FEATURE_GROUPS
    if unknown:
        raise ValueError(f"Unknown temporal feature groups: {', '.join(sorted(unknown))}")
    return frozenset(tokens)


def parse_selection_weights(value: str | None) -> tuple[float, ...]:
    """Parse per-threshold validation-selection weights in threshold order."""
    if value is None or not str(value).strip():
        return DEFAULT_SELECTION_WEIGHTS
    try:
        weights = tuple(float(token.strip()) for token in str(value).split(","))
    except ValueError as exc:
        raise ValueError("--selection-weights must be four comma-separated numbers") from exc
    if len(weights) != PRIMARY_OUTPUT_COUNT:
        raise ValueError(
            f"--selection-weights must contain {PRIMARY_OUTPUT_COUNT} values for thresholds {THRESHOLDS}"
        )
    if any(not np.isfinite(weight) or weight < 0.0 for weight in weights) or not any(weights):
        raise ValueError("--selection-weights must be finite, non-negative, and not all zero")
    total = float(sum(weights))
    return tuple(weight / total for weight in weights)


def temporal_feature_group(name: str) -> str:
    """Classify a derived W-specific feature by its physical role."""
    if name.startswith("delta_"):
        return "delta"
    if re.search(r"(?:_sum(?:_nT_min)?|_duration_min)_W\d+$", name):
        return "accumulation"
    return "statistics"


def feature_role_indices(feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Split minute-varying inputs from end-of-window context features."""
    context = np.asarray(
        [
            index for index, name in enumerate(feature_names)
            if re.search(r"(?:_(?:mean|max|std)|_sum(?:_nT_min)?|_duration_min)_W\d+$", name)
        ],
        dtype=np.int64,
    )
    context_set = set(context.tolist())
    sequence = np.asarray(
        [index for index in range(len(feature_names)) if index not in context_set],
        dtype=np.int64,
    )
    return sequence, context


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_cuda(device: torch.device) -> None:
    """Enable fast kernels for the fixed-shape batches used by this script."""
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, args: argparse.Namespace) -> DataLoader:
    options: dict[str, object] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }
    if args.num_workers > 0:
        options["persistent_workers"] = True
        options["prefetch_factor"] = args.prefetch_factor
    return DataLoader(dataset, **options)


def transform_features(
    frame: pd.DataFrame,
    temporal_feature_groups: frozenset[str] = frozenset(("accumulation",)),
) -> tuple[np.ndarray, list[str]]:
    columns = []
    names = []
    for name in RAW_FEATURES:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float32)
        if name in LOG1P_FEATURES:
            values = np.where(values >= 0.0, np.log1p(values), np.nan)
            names.append(f"log1p_{name}")
        elif name != "clock_angle_rad":
            names.append(name)
        else:
            continue
        columns.append(values)
    angle = pd.to_numeric(frame["clock_angle_rad"], errors="coerce").to_numpy(dtype=np.float32)
    columns.extend([np.sin(angle), np.cos(angle)])
    names.extend(["sin_clock_angle", "cos_clock_angle"])
    features = pd.DataFrame(np.column_stack(columns), columns=names, index=frame.index)
    if "delta" in temporal_feature_groups:
        # All temporal summaries are causal and isolated to an event block.
        # Zeros at a block's first rows represent no available prior change.
        groups = frame["event_group"]
        for name in TEMPORAL_DELTA_FEATURES:
            values = features[name].groupby(groups, sort=False)
            features[f"delta_1min_{name}"] = values.diff().fillna(0.0)
            features[f"delta_10min_{name}"] = values.diff(10).fillna(0.0)
    return features.to_numpy(dtype=np.float32, copy=False), list(features.columns)


def load_timeline(
    lag: int,
    window: int,
    temporal_feature_groups: frozenset[str] = frozenset(("accumulation",)),
    window_statistics: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    valid_col = f"sample_valid_L{lag}_W{window}"
    label_cols = [f"target_exceeds_{threshold}A_30min" for threshold in THRESHOLDS]
    onset_label_cols = [f"target_onset_{threshold}A_30min" for threshold in THRESHOLDS]
    needed = [
        "time", "GIC_abs", "event_group", "split",
        *RAW_FEATURES, valid_col, *label_cols, *onset_label_cols,
    ]
    if not TIMELINE_PATH.exists():
        raise FileNotFoundError(f"03 output not found: {TIMELINE_PATH}")
    if not EVENT_CATALOG_PATH.exists():
        raise FileNotFoundError(f"03 event catalog not found: {EVENT_CATALOG_PATH}")
    header = pd.read_csv(TIMELINE_PATH, nrows=0)
    missing = [column for column in needed if column not in header.columns]
    if missing:
        raise ValueError(f"prediction_timeline.csv is missing: {', '.join(missing)}")
    frame = pd.read_csv(TIMELINE_PATH, usecols=needed, parse_dates=["time"])
    statistic_names: list[str] = []
    if window_statistics:
        statistics_path = Path(str(WINDOW_STATISTICS_TEMPLATE).format(window=window))
        if not statistics_path.exists():
            raise FileNotFoundError(
                f"03 window statistics not found: {statistics_path}. Run 03_build_prediction_dataset.py first."
            )
        statistics = pd.read_csv(statistics_path, parse_dates=["time"])
        statistic_names = [name for name in statistics.columns if name != "time"]
        if not statistic_names:
            raise ValueError(f"{statistics_path.name} contains no feature columns.")
        if statistics["time"].duplicated().any():
            raise ValueError(f"{statistics_path.name} contains duplicate timestamps.")
        frame = frame.merge(statistics, on="time", how="left", validate="one_to_one")
    if frame["time"].duplicated().any() or not frame["time"].is_monotonic_increasing:
        raise ValueError("03 event timeline must be sorted and contain unique timestamps.")
    if frame["split"].isna().any() or frame["event_group"].isna().any():
        raise ValueError("03 event timeline contains rows without an event split/group.")
    unknown_splits = sorted(set(frame["split"].astype(str)) - {"train", "validation", "test"})
    if unknown_splits:
        raise ValueError(f"03 event timeline has unknown split names: {', '.join(unknown_splits)}")
    features, feature_names = transform_features(frame, temporal_feature_groups=temporal_feature_groups)
    if statistic_names:
        selected_statistic_names = [
            name for name in statistic_names if temporal_feature_group(name) in temporal_feature_groups
        ]
        if selected_statistic_names:
            statistic_values = frame[selected_statistic_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
            features = np.column_stack([features, statistic_values]).astype(np.float32, copy=False)
            feature_names.extend(selected_statistic_names)
    labels = frame[label_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    onset_labels = frame[onset_label_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    source_valid = pd.to_numeric(frame[valid_col], errors="coerce").fillna(0).to_numpy(dtype=np.uint8)
    return frame, features, labels, onset_labels, source_valid, feature_names


def build_indices(
    features: np.ndarray,
    labels: np.ndarray,
    source_valid: np.ndarray,
    split_labels: np.ndarray,
    lag: int,
    window: int,
    split: str,
    stride: int,
    sequence_feature_indices: np.ndarray | None = None,
    context_feature_indices: np.ndarray | None = None,
) -> np.ndarray:
    candidates = np.flatnonzero(split_labels == split)[::max(1, stride)]
    if not len(candidates):
        return np.empty(0, dtype=np.int64)
    if sequence_feature_indices is None:
        sequence_feature_indices = np.arange(features.shape[1], dtype=np.int64)
    if context_feature_indices is None:
        context_feature_indices = np.empty(0, dtype=np.int64)
    row_valid = np.isfinite(features[:, sequence_feature_indices]).all(axis=1)
    bad_prefix = np.r_[0, np.cumsum(~row_valid, dtype=np.int64)]
    input_end = candidates - lag
    input_start = input_end - window + 1
    in_bounds = (input_start >= 0) & (input_end < len(features))
    sequence_valid = np.zeros(len(candidates), dtype=bool)
    valid_positions = np.flatnonzero(in_bounds)
    if len(valid_positions):
        valid_start = input_start[valid_positions]
        valid_end = input_end[valid_positions]
        sequence_valid[valid_positions] = (bad_prefix[valid_end + 1] - bad_prefix[valid_start]) == 0
    context_valid = np.ones(len(candidates), dtype=bool)
    if len(context_feature_indices):
        context_valid[valid_positions] = np.isfinite(
            features[input_end[valid_positions]][:, context_feature_indices]
        ).all(axis=1)
    label_valid = np.isfinite(labels[candidates]).all(axis=1)
    keep = in_bounds & source_valid[candidates].astype(bool) & sequence_valid & context_valid & label_valid
    return candidates[keep]


def select_nonredundant_features(
    raw_features: np.ndarray,
    feature_names: list[str],
    training_rows: np.ndarray,
    correlation_threshold: float,
) -> np.ndarray:
    """Keep all useful feature families while removing only near-duplicate columns.

    Selection is fit on training rows of each forward fold.  Features are
    considered in their established physical order, so a derived rolling
    summary is removed only when an earlier feature already carries nearly
    identical information.
    """
    training = raw_features[training_rows]
    finite_count = np.isfinite(training).sum(axis=0)
    variance = np.full(training.shape[1], np.nan, dtype=np.float64)
    variance_candidates = np.flatnonzero(finite_count >= 2)
    if len(variance_candidates):
        variance[variance_candidates] = np.nanvar(training[:, variance_candidates], axis=0)
    valid = (finite_count >= 2) & (variance > 1e-10)
    candidates = np.flatnonzero(valid)
    if not len(candidates):
        raise RuntimeError("Feature selection removed every input feature.")
    selected: list[int] = []
    for feature_index in candidates:
        if not selected:
            selected.append(int(feature_index))
            continue
        redundant = False
        values = training[:, feature_index]
        for prior in selected:
            prior_values = training[:, prior]
            shared = np.isfinite(values) & np.isfinite(prior_values)
            if shared.sum() < 2:
                continue
            correlation = np.corrcoef(values[shared], prior_values[shared])[0, 1]
            if np.isfinite(correlation) and abs(correlation) >= correlation_threshold:
                redundant = True
                break
        if not redundant:
            selected.append(int(feature_index))
    if len(selected) < 2:
        raise RuntimeError("Feature selection retained fewer than two non-redundant features.")
    return np.asarray(selected, dtype=np.int64)


class LagWindowDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        times: np.ndarray,
        lag: int,
        window: int,
        context_feature_indices: np.ndarray | None = None,
    ):
        self.features = features
        self.labels = labels
        self.times = times
        self.lag = lag
        self.window = window
        self.context_feature_indices = (
            np.asarray(context_feature_indices, dtype=np.int64)
            if context_feature_indices is not None else np.empty(0, dtype=np.int64)
        )

    def __len__(self) -> int:
        return len(self.times)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = int(self.times[index])
        end = t - self.lag
        start = end - self.window + 1
        x = np.array(self.features[start:end + 1], dtype=np.float32, copy=True)
        if len(self.context_feature_indices):
            # Window summaries are defined at t-L over this exact input window.
            # Repeat that endpoint context over time so the sequence encoder can
            # use it without requiring a complete rolling history at every row.
            x[:, self.context_feature_indices] = x[-1, self.context_feature_indices]
        y = self.labels[t].astype(np.float32, copy=False)
        return torch.from_numpy(x), torch.from_numpy(y)


class CNNBiLSTMClassifier(nn.Module):
    """Fuse direct CNN peak evidence with attentive BiLSTM sequence context.

    The CNN branch pools its local and dilated responses directly with mean
    and max pooling.  In parallel, the BiLSTM branch summarizes the same CNN
    sequence with terminal states and learned temporal attention.  A learned
    element-wise gate fuses the two representations before classification,
    allowing the model to retain short-duration CNN responses rather than
    forcing every decision through the recurrent encoder.
    """

    encoder_type = "multi_scale_residual_cnn_direct_pool_gated_attentive_bilstm"
    pooling_type = "cnn_temporal_mean_max_plus_bilstm_terminal_attention_gated_fusion"

    def __init__(
        self,
        feature_count: int,
        channels: int = 24,
        lstm_hidden_size: int = 64,
        lstm_layers: int = 2,
        dropout: float = 0.10,
        attention_hidden_size: int = 32,
        output_count: int = TOTAL_OUTPUT_COUNT,
    ):
        super().__init__()
        self.input_projection = nn.Linear(feature_count, channels) if feature_count != channels else nn.Identity()
        self.input_norm = nn.LayerNorm(channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, dilation=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=4, dilation=4)
        self.bn2 = nn.BatchNorm1d(channels)
        self.activation = nn.GELU()
        self.conv_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        cnn_representation_size = channels * 2
        lstm_representation_size = lstm_hidden_size * 4
        self.sequence_norm = nn.LayerNorm(lstm_hidden_size * 2)
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, attention_hidden_size), nn.Tanh(),
            nn.Linear(attention_hidden_size, 1),
        )
        self.cnn_pool_norm = nn.LayerNorm(cnn_representation_size)
        self.lstm_pool_norm = nn.LayerNorm(lstm_representation_size)
        self.lstm_projection = nn.Sequential(
            nn.Linear(lstm_representation_size, cnn_representation_size),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(cnn_representation_size * 2, cnn_representation_size),
            nn.Sigmoid(),
        )
        self.fusion_norm = nn.LayerNorm(cnn_representation_size)
        self.classification_head = nn.Sequential(
            nn.Linear(cnn_representation_size, cnn_representation_size), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(cnn_representation_size, output_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.input_norm(self.input_projection(x)).transpose(1, 2)
        residual = encoded
        local = self.activation(self.bn1(self.conv1(encoded)))
        wide = self.activation(self.bn2(self.conv2(local)))
        encoded = self.conv_dropout(local + wide) + residual
        cnn_representation = self.cnn_pool_norm(
            torch.cat([encoded.mean(dim=2), encoded.amax(dim=2)], dim=1)
        )
        sequence, (hidden, _) = self.lstm(encoded.transpose(1, 2))
        sequence = self.sequence_norm(sequence)
        attention_weights = torch.softmax(self.attention(sequence).squeeze(-1), dim=1)
        attended = torch.sum(sequence * attention_weights.unsqueeze(-1), dim=1)
        terminal = torch.cat([hidden[-2], hidden[-1]], dim=1)
        lstm_representation = self.lstm_projection(
            self.lstm_pool_norm(torch.cat([terminal, attended], dim=1))
        )
        cnn_gate = self.fusion_gate(torch.cat([cnn_representation, lstm_representation], dim=1))
        representation = cnn_gate * cnn_representation + (1.0 - cnn_gate) * lstm_representation
        return self.classification_head(self.fusion_norm(representation))


def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            all_logits.append(model(x.to(device, non_blocking=True)).cpu().numpy())
            all_labels.append(y.numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


def multilabel_forecast_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
    monotonicity_weight: float,
    onset_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Peak BCE plus a weighted onset auxiliary BCE and consistency penalty.

    Since ``|GIC| > 20`` implies ``|GIC| > 10 > 5 > 3``, a peak prediction
    with a higher-threshold probability above a lower-threshold probability
    is physically incoherent. The penalty is deliberately soft: it preserves
    four task-specific logits and therefore does not force a potentially
    misspecified shared ordinal score. Onset labels are not nested (a series
    already above 10 A can later cross 20 A), so they are intentionally not
    constrained this way.
    """
    peak_logits, onset_logits = logits[:, :PRIMARY_OUTPUT_COUNT], logits[:, PRIMARY_OUTPUT_COUNT:]
    peak_targets, onset_targets = targets[:, :PRIMARY_OUTPUT_COUNT], targets[:, PRIMARY_OUTPUT_COUNT:]
    peak_bce = F.binary_cross_entropy_with_logits(
        peak_logits, peak_targets, pos_weight=pos_weight[:PRIMARY_OUTPUT_COUNT],
    )
    onset_bce = F.binary_cross_entropy_with_logits(
        onset_logits, onset_targets, pos_weight=pos_weight[PRIMARY_OUTPUT_COUNT:],
    )
    weighted_bce = peak_bce + onset_loss_weight * onset_bce
    probabilities = torch.sigmoid(logits)
    peak_probabilities = probabilities[:, :PRIMARY_OUTPUT_COUNT]
    violations = F.relu(peak_probabilities[:, 1:] - peak_probabilities[:, :-1])
    consistency = violations.square().mean()
    return weighted_bce + monotonicity_weight * consistency, peak_bce, onset_bce, consistency


def metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    cutoff: float,
    include_ranking: bool = True,
) -> dict[str, float | int]:
    pred = probability >= cutoff
    truth = y_true.astype(bool)
    tp = int(np.sum(pred & truth)); fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth)); tn = int(np.sum(~pred & ~truth))
    pod = tp / (tp + fn) if tp + fn else np.nan
    far = fp / (tp + fp) if tp + fp else np.nan
    csi = tp / (tp + fp + fn) if tp + fp + fn else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
    bias = (tp + fp) / (tp + fn) if tp + fn else np.nan
    if include_ranking:
        try: auc = float(roc_auc_score(truth, probability))
        except ValueError: auc = np.nan
        try: pr_auc = float(average_precision_score(truth, probability))
        except ValueError: pr_auc = np.nan
    else:
        auc, pr_auc = np.nan, np.nan
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn, "POD": pod, "FAR": far,
        "CSI": csi, "F1": f1, "Bias": bias, "ROC_AUC": auc, "PR_AUC": pr_auc,
    }


def _segments_from_mask(mask: np.ndarray, times: pd.DatetimeIndex, groups: np.ndarray) -> list[tuple[int, int]]:
    """Return positive segments without crossing event blocks or time gaps."""
    active = np.asarray(mask, dtype=bool)
    if not active.any():
        return []
    previous_contiguous = np.r_[
        False,
        (groups[1:] == groups[:-1])
        & (np.diff(times.asi8) == pd.Timedelta(minutes=1).value),
    ]
    next_contiguous = np.r_[previous_contiguous[1:], False]
    starts = np.flatnonzero(active & ~(np.r_[False, active[:-1]] & previous_contiguous))
    ends = np.flatnonzero(active & ~(np.r_[active[1:], False] & next_contiguous))
    if len(starts) != len(ends):
        raise RuntimeError("Could not pair alarm-segment starts and ends.")
    return list(zip(starts.astype(int).tolist(), ends.astype(int).tolist()))


def _postprocess_mask(
    raw_mask: np.ndarray,
    times: pd.DatetimeIndex,
    groups: np.ndarray,
    min_consecutive: int,
    merge_gap: int,
    min_event_length: int,
) -> np.ndarray:
    """Remove short alarms, bridge short gaps, then remove short final episodes."""
    segments = [
        segment for segment in _segments_from_mask(raw_mask, times, groups)
        if segment[1] - segment[0] + 1 >= min_consecutive
    ]
    merged: list[tuple[int, int]] = []
    for segment in segments:
        if not merged:
            merged.append(segment)
            continue
        previous = merged[-1]
        gap_minutes = (times[segment[0]] - times[previous[1]]) / pd.Timedelta(minutes=1) - 1
        same_block = groups[segment[0]] == groups[previous[1]]
        if same_block and gap_minutes <= merge_gap:
            merged[-1] = (previous[0], segment[1])
        else:
            merged.append(segment)
    result = np.zeros(len(raw_mask), dtype=bool)
    for start, end in merged:
        duration = (times[end] - times[start]) / pd.Timedelta(minutes=1) + 1
        if duration >= min_event_length:
            result[start:end + 1] = True
    return result


def _event_metrics_from_masks(
    truth: np.ndarray,
    prediction: np.ndarray,
    times: pd.DatetimeIndex,
    groups: np.ndarray,
) -> dict[str, float | int]:
    """Compute point POFD plus one-to-one event matching metrics."""
    truth = np.asarray(truth, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    point_tp = int((prediction & truth).sum())
    point_fp = int((prediction & ~truth).sum())
    point_fn = int((~prediction & truth).sum())
    point_tn = int((~prediction & ~truth).sum())
    true_segments = _segments_from_mask(truth, times, groups)
    predicted_segments = _segments_from_mask(prediction, times, groups)
    matched_true: set[int] = set()
    matched_pred: set[int] = set()
    for pred_id, (pred_start, pred_end) in enumerate(predicted_segments):
        candidates = [
            true_id for true_id, (true_start, true_end) in enumerate(true_segments)
            if true_id not in matched_true and pred_start <= true_end and pred_end >= true_start
        ]
        if candidates:
            true_id = min(candidates, key=lambda index: abs(true_segments[index][0] - pred_start))
            matched_true.add(true_id)
            matched_pred.add(pred_id)
    event_tp = len(matched_pred)
    event_fp = len(predicted_segments) - event_tp
    event_fn = len(true_segments) - len(matched_true)
    event_pod = event_tp / (event_tp + event_fn) if event_tp + event_fn else np.nan
    event_far = event_fp / (event_tp + event_fp) if event_tp + event_fp else np.nan
    event_csi = event_tp / (event_tp + event_fp + event_fn) if event_tp + event_fp + event_fn else np.nan
    event_f1 = 2 * event_tp / (2 * event_tp + event_fp + event_fn) if 2 * event_tp + event_fp + event_fn else np.nan
    return {
        "point_POD": point_tp / (point_tp + point_fn) if point_tp + point_fn else np.nan,
        "point_POFD": point_fp / (point_fp + point_tn) if point_fp + point_tn else np.nan,
        "point_FAR": point_fp / (point_tp + point_fp) if point_tp + point_fp else np.nan,
        "point_CSI": point_tp / (point_tp + point_fp + point_fn) if point_tp + point_fp + point_fn else np.nan,
        "point_F1": 2 * point_tp / (2 * point_tp + point_fp + point_fn) if 2 * point_tp + point_fp + point_fn else np.nan,
        "event_TP": int(event_tp), "event_FP": int(event_fp), "event_FN": int(event_fn),
        "event_POD": event_pod, "event_FAR": event_far, "event_CSI": event_csi, "event_F1": event_f1,
        "event_Bias": (event_tp + event_fp) / (event_tp + event_fn) if event_tp + event_fn else np.nan,
        "event_POFD": point_fp / (point_fp + point_tn) if point_fp + point_tn else np.nan,
        "event_alarm_count": int(len(predicted_segments)),
        "event_true_count": int(len(true_segments)),
    }


def select_cutoff(
    y_true: np.ndarray,
    probability: np.ndarray,
    policy: str,
    max_far: float,
) -> tuple[float, dict[str, float | int], str, pd.DataFrame]:
    """Choose a validation cutoff with an operational false-alarm policy."""
    candidates = np.unique(np.r_[0.0, np.quantile(probability, np.linspace(0.0, 1.0, 1001)), 1.0])
    truth = y_true.astype(bool)
    order = np.argsort(probability, kind="stable")
    sorted_probability = probability[order]
    sorted_truth = truth[order].astype(np.int64)
    positive_prefix = np.r_[0, np.cumsum(sorted_truth)]
    total_positive = int(positive_prefix[-1])
    rows = []
    for cutoff in candidates:
        first_positive = int(np.searchsorted(sorted_probability, cutoff, side="left"))
        predicted_positive = len(probability) - first_positive
        tp = total_positive - int(positive_prefix[first_positive])
        fp = predicted_positive - tp
        fn = total_positive - tp
        tn = len(probability) - tp - fp - fn
        rows.append({
            "decision_threshold": float(cutoff), "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "POD": tp / (tp + fn) if tp + fn else np.nan,
            "FAR": fp / (tp + fp) if tp + fp else np.nan,
            "CSI": tp / (tp + fp + fn) if tp + fp + fn else np.nan,
            "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan,
            "Bias": (tp + fp) / (tp + fn) if tp + fn else np.nan,
        })
    tradeoff = pd.DataFrame(rows)
    alarms = tradeoff.loc[(tradeoff["TP"] + tradeoff["FP"]).gt(0)].copy()
    if alarms.empty:
        raise RuntimeError("Validation probabilities produced no alarm candidates.")

    if policy == "csi":
        selected = alarms.sort_values(["CSI", "FAR"], ascending=[False, True], na_position="last").iloc[0]
        status = "maximum_CSI"
    elif policy == "bias_one":
        alarms["bias_distance"] = (alarms["Bias"] - 1.0).abs()
        selected = alarms.sort_values(["bias_distance", "CSI", "FAR"], ascending=[True, False, True], na_position="last").iloc[0]
        status = "closest_validation_Bias_to_1"
    else:
        constrained = alarms.loc[alarms["FAR"].le(max_far)]
        if not constrained.empty:
            constrained = constrained.copy()
            constrained["bias_distance"] = (constrained["Bias"] - 1.0).abs()
            selected = constrained.sort_values(["CSI", "bias_distance"], ascending=[False, True], na_position="last").iloc[0]
            status = f"maximum_CSI_with_validation_FAR_at_most_{max_far:.2f}"
        else:
            selected = alarms.sort_values(["FAR", "CSI"], ascending=[True, False], na_position="last").iloc[0]
            status = f"validation_FAR_cap_{max_far:.2f}_unattainable_used_minimum_FAR"
    cutoff = float(selected["decision_threshold"])
    return cutoff, metrics(y_true, probability, cutoff), status, tradeoff


def select_event_operating_point(
    y_true: np.ndarray,
    probability: np.ndarray,
    times: pd.DatetimeIndex,
    groups: np.ndarray,
    max_pofd: float,
    pod_floor: float,
    settings: tuple[tuple[int, int, int], ...] = EVENT_POSTPROCESS_SETTINGS,
    verbose: bool = True,
) -> tuple[float, dict[str, float | int], str, dict[str, int], pd.DataFrame]:
    """Select cutoff and alarm post-processing using OOF event metrics only."""
    candidates = np.unique(np.r_[
        0.0, np.quantile(probability, np.linspace(0.0, 1.0, EVENT_CUTOFF_QUANTILE_COUNT)), 1.0,
    ])
    if verbose:
        print(
            f"Event operating-point search: {len(candidates)} probability cutoffs x "
            f"{len(settings)} post-processing settings",
            flush=True,
        )
    rows: list[dict[str, float | int]] = []
    progress_step = max(1, len(candidates) // 5)
    for candidate_index, cutoff in enumerate(candidates, start=1):
        raw = probability >= cutoff
        for min_consecutive, merge_gap, min_event_length in settings:
            processed = _postprocess_mask(
                raw, times, groups, min_consecutive, merge_gap, min_event_length,
            )
            row = {
                "decision_threshold": float(cutoff),
                "min_consecutive": min_consecutive,
                "merge_gap": merge_gap,
                "min_event_length": min_event_length,
            }
            row.update(_event_metrics_from_masks(y_true, processed, times, groups))
            rows.append(row)
        if verbose and (candidate_index % progress_step == 0 or candidate_index == len(candidates)):
            print(f"  event search cutoffs: {candidate_index}/{len(candidates)}", flush=True)
    table = pd.DataFrame(rows)
    feasible = table.loc[table["event_POFD"].le(max_pofd)].copy()
    if not feasible.empty:
        selected = feasible.sort_values(
            ["event_CSI", "event_POD", "event_POFD"],
            ascending=[False, False, True], na_position="last",
        ).iloc[0]
        status = f"maximum_event_CSI_with_event_POFD_at_most_{max_pofd:.2f}"
    else:
        floor_rows = table.loc[table["event_POD"].ge(pod_floor)].copy()
        if not floor_rows.empty:
            selected = floor_rows.sort_values(
            ["event_POFD", "event_F1", "event_POD"],
            ascending=[True, False, False], na_position="last",
            ).iloc[0]
            status = f"minimum_event_POFD_with_event_POD_at_least_{pod_floor:.2f}"
        else:
            selected = table.sort_values(
            ["event_POFD", "event_F1", "event_POD"],
            ascending=[True, False, False], na_position="last",
            ).iloc[0]
            status = "minimum_event_POFD_no_POD_floor_attainable"
    params = {
        "min_consecutive": int(selected["min_consecutive"]),
        "merge_gap": int(selected["merge_gap"]),
        "min_event_length": int(selected["min_event_length"]),
    }
    return (
        float(selected["decision_threshold"]),
        {key: selected[key] for key in (
            "event_TP", "event_FP", "event_FN", "event_POD", "event_FAR", "event_CSI", "event_F1",
            "event_Bias", "event_POFD", "event_alarm_count", "event_true_count",
        )},
        status,
        params,
        table,
    )


def validation_event_checkpoint_score(
    logits: np.ndarray,
    labels: np.ndarray,
    times: pd.DatetimeIndex,
    groups: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, float]:
    """Score a checkpoint with weighted validation event CSI and a POFD guard."""
    logits = logits[:, :PRIMARY_OUTPUT_COUNT]
    labels = labels[:, :PRIMARY_OUTPUT_COUNT]
    probability = 1.0 / (1.0 + np.exp(-logits))
    csi_values, pofd_values = [], []
    for column in range(len(THRESHOLDS)):
        _, event_metrics, _, _, _ = select_event_operating_point(
            labels[:, column], probability[:, column], times, groups,
            args.event_pofd_cap, args.event_pod_floor, settings=(DEFAULT_POSTPROCESS,), verbose=False,
        )
        csi_values.append(float(event_metrics["event_CSI"]))
        pofd_values.append(float(event_metrics["event_POFD"]))
    weights = np.asarray(args.selection_weights, dtype=np.float64)
    csi_array = np.asarray(csi_values, dtype=np.float64)
    finite = np.isfinite(csi_array) & (weights > 0.0)
    weighted_csi = float(np.sum(csi_array[finite] * weights[finite]) / np.sum(weights[finite])) if finite.any() else np.nan
    return weighted_csi, float(np.nanmax(pofd_values))


def plot_validation_threshold_tradeoffs(tradeoff: pd.DataFrame, output_path: Path) -> None:
    """Show the validation POD/FAR/CSI/Bias trade-off for all GIC thresholds."""
    figure, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    settings = (("POD", "POD"), ("FAR", "FAR"), ("CSI", "CSI"), ("Bias", "Bias"))
    for axis, (column, label) in zip(axes.ravel()[:4], settings):
        for threshold, subset in tradeoff.groupby("threshold_A", sort=True):
            axis.plot(subset["decision_threshold"], subset[column], linewidth=1.1, label=f"> {threshold} A")
            chosen = subset.loc[subset["selected"].eq(1)]
            if not chosen.empty:
                axis.scatter(chosen["decision_threshold"], chosen[column], s=24, zorder=3)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25, linewidth=0.4)
    for axis in axes.ravel()[:4]:
        axis.set_xlabel("Decision threshold")
    axes[0, 0].legend(loc="best", fontsize=8, frameon=False)
    pod_far_axis, pr_axis = axes[1, 1], axes[1, 2]
    # A performance diagram puts probability-of-detection, false-alarm ratio,
    # CSI, and frequency bias in one operational view.
    success_ratio = np.linspace(0.01, 1.0, 200)
    pod = np.linspace(0.01, 1.0, 200)
    success_grid, pod_grid = np.meshgrid(success_ratio, pod)
    csi_grid = 1.0 / (1.0 / success_grid + 1.0 / pod_grid - 1.0)
    contours = pod_far_axis.contour(
        1.0 - success_grid,
        pod_grid,
        csi_grid,
        levels=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
        colors="0.55",
        linewidths=0.55,
    )
    pod_far_axis.clabel(contours, inline=True, fontsize=7, fmt="CSI %.1f")
    for bias in (0.5, 1.0, 2.0, 3.0):
        line_pod = bias * success_ratio
        keep = line_pod <= 1.0
        pod_far_axis.plot(
            1.0 - success_ratio[keep],
            line_pod[keep],
            color="0.4",
            linestyle="--",
            linewidth=0.65,
            zorder=1,
        )
        label_index = max(0, min(np.count_nonzero(keep) - 1, 125))
        if np.count_nonzero(keep):
            pod_far_axis.text(
                1.0 - success_ratio[keep][label_index],
                line_pod[keep][label_index],
                f"Bias={bias:g}",
                fontsize=6.5,
                color="0.35",
                ha="left",
                va="bottom",
            )
    for threshold, subset in tradeoff.groupby("threshold_A", sort=True):
        chosen = subset.loc[subset["selected"].eq(1)]
        pod_far_axis.plot(subset["FAR"], subset["POD"], linewidth=1.1, label=f"> {threshold} A")
        pr_axis.plot(subset["POD"], 1.0 - subset["FAR"], linewidth=1.1, label=f"> {threshold} A")
        if not chosen.empty:
            pod_far_axis.scatter(chosen["FAR"], chosen["POD"], s=24, zorder=3)
            pr_axis.scatter(chosen["POD"], 1.0 - chosen["FAR"], s=24, zorder=3)
    pod_far_axis.set(
        xlabel="FAR (lower is better)",
        ylabel="POD (higher is better)",
        xlim=(0, 1),
        ylim=(0, 1),
        title="Performance diagram (CSI contours and Bias lines)",
    )
    pr_axis.set(xlabel="Recall (POD)", ylabel="Precision (1-FAR)", xlim=(0, 1), ylim=(0, 1), title="Precision-recall operating curve")
    for axis in (pod_far_axis, pr_axis):
        axis.grid(alpha=0.25, linewidth=0.4)
    figure.suptitle("Validation-set decision-threshold trade-offs", fontsize=13)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def build_alarm_episodes(
    times: pd.DatetimeIndex,
    event_groups: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    threshold: int,
) -> list[dict[str, object]]:
    """Merge consecutive positive minutes into operational alarm episodes."""
    episodes: list[dict[str, object]] = []
    start: int | None = None
    previous: int | None = None

    def close_episode(begin: int, end: int) -> None:
        matched_target = bool(truth[begin:end + 1].any())
        episodes.append({
            "threshold_A": threshold,
            "event_group": int(event_groups[begin]),
            "alarm_start": times[begin],
            "alarm_end": times[end],
            "duration_min": end - begin + 1,
            "target_positive_minutes": int(truth[begin:end + 1].sum()),
            "target_matched": int(matched_target),
            "false_alarm_event": int(not matched_target),
        })

    for position, active in enumerate(prediction):
        contiguous = (
            previous is not None
            and event_groups[position] == event_groups[previous]
            and times[position] - times[previous] == pd.Timedelta(minutes=1)
        )
        if active and start is None:
            start = position
        elif active and not contiguous:
            close_episode(start, previous)  # type: ignore[arg-type]
            start = position
        elif not active and start is not None:
            close_episode(start, previous)  # type: ignore[arg-type]
            start = None
        previous = position
    if start is not None:
        close_episode(start, previous)  # type: ignore[arg-type]
    return episodes


def find_event_onset(
    times: pd.DatetimeIndex,
    gic_abs: np.ndarray,
    event: object,
    threshold: int,
) -> pd.Timestamp | pd.NaT:
    """Find the threshold crossing belonging to this catalogued event peak."""
    event_rows = np.flatnonzero(
        (times >= pd.Timestamp(event.event_window_start))
        & (times < pd.Timestamp(event.event_window_end))
        & np.isfinite(gic_abs)
        & (gic_abs >= threshold)
    )
    if not len(event_rows):
        return pd.NaT
    peak_time = pd.Timestamp(event.peak_time)
    anchor = event_rows[np.argmin(np.abs((times[event_rows] - peak_time).asi8))]
    onset = int(anchor)
    event_start = int(event_rows.min())
    while (
        onset > event_start
        and times[onset] - times[onset - 1] == pd.Timedelta(minutes=1)
        and np.isfinite(gic_abs[onset - 1])
        and gic_abs[onset - 1] >= threshold
    ):
        onset -= 1
    return times[onset]


def evaluate_test_events(
    all_times: pd.DatetimeIndex,
    all_gic_abs: np.ndarray,
    prediction_times: pd.DatetimeIndex,
    prediction_groups: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    cutoffs: np.ndarray,
    postprocess_params: list[dict[str, int]],
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Report alarm episodes and per-event warning success inside the 30-min horizon."""
    episodes: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    test_events = events.loc[events["split"].eq("test")]
    for column, threshold in enumerate(THRESHOLDS):
        raw_predicted = probabilities[:, column] >= cutoffs[column]
        predicted = _postprocess_mask(
            raw_predicted, prediction_times, prediction_groups, **postprocess_params[column],
        )
        episodes.extend(build_alarm_episodes(prediction_times, prediction_groups, labels[:, column].astype(bool), predicted, threshold))
        for event in test_events.itertuples(index=False):
            onset = find_event_onset(all_times, all_gic_abs, event, threshold)
            if pd.isna(onset):
                continue
            warning_detected = False
            first_warning = pd.NaT
            lead_time = np.nan
            if pd.notna(onset):
                in_horizon = (prediction_times >= onset - pd.Timedelta(minutes=30)) & (prediction_times < onset) & predicted
                if in_horizon.any():
                    first_warning = prediction_times[np.flatnonzero(in_horizon)[0]]
                    warning_detected = True
                    lead_time = float((onset - first_warning) / pd.Timedelta(minutes=1))
            event_rows.append({
                "threshold_A": threshold,
                "event_id": int(event.event_id),
                "driver": event.driver,
                "peak_time": event.peak_time,
                "gic_peak_abs_A": event.gic_peak_abs_A,
                "threshold_onset": onset,
                "warning_detected_30min": int(warning_detected),
                "first_warning_time": first_warning,
                "lead_time_min": lead_time,
            })
    # A stringent validation-derived cutoff may legitimately yield no test
    # alarms.  Preserve the schema so per-threshold summaries still work.
    episode_table = pd.DataFrame(
        episodes,
        columns=[
            "threshold_A", "event_group", "alarm_start", "alarm_end", "duration_min",
            "target_positive_minutes", "target_matched", "false_alarm_event",
        ],
    )
    event_table = pd.DataFrame(
        event_rows,
        columns=[
            "threshold_A", "event_id", "driver", "peak_time", "gic_peak_abs_A",
            "threshold_onset", "warning_detected_30min", "first_warning_time", "lead_time_min",
        ],
    )
    summaries = []
    for threshold in THRESHOLDS:
        event_subset = event_table.loc[event_table["threshold_A"].eq(threshold)]
        episode_subset = episode_table.loc[episode_table["threshold_A"].eq(threshold)]
        detected = int(event_subset["warning_detected_30min"].sum())
        event_count = len(event_subset)
        summaries.append({
            "threshold_A": threshold,
            "eligible_event_count": event_count,
            "test_event_count": event_count,
            "detected_events_30min": detected,
            "missed_events_30min": event_count - detected,
            "event_POD_30min": detected / event_count if event_count else np.nan,
            "median_lead_time_min": float(event_subset.loc[event_subset["warning_detected_30min"].eq(1), "lead_time_min"].median()),
            "alarm_episode_count": len(episode_subset),
            "true_alarm_episode_count": int(episode_subset["target_matched"].sum()),
            "false_alarm_event_count": int(episode_subset["false_alarm_event"].sum()),
            "false_alarm_event_ratio": float(episode_subset["false_alarm_event"].mean()) if len(episode_subset) else np.nan,
        })
    return episode_table, event_table, pd.DataFrame(summaries)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, device: torch.device) -> np.ndarray:
    """Per-threshold temperature scaling on validation logits."""
    values = []
    for column in range(logits.shape[1]):
        x = torch.tensor(logits[:, column], dtype=torch.float32, device=device)
        y = torch.tensor(labels[:, column], dtype=torch.float32, device=device)
        if len(torch.unique(y)) < 2:
            values.append(1.0); continue
        log_t = torch.zeros(1, device=device, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=50)
        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(x / log_t.exp().clamp(0.05, 20.0), y)
            loss.backward()
            return loss
        optimizer.step(closure)
        values.append(float(log_t.exp().detach().cpu().clamp(0.05, 20.0)))
    return np.asarray(values, dtype=np.float32)


def build_forward_folds(
    frame: pd.DataFrame,
    event_groups: np.ndarray,
    fold_count: int,
    initial_train_fraction: float,
) -> list[dict[str, np.ndarray]]:
    """Create expanding chronological folds from the pre-test event blocks."""
    group_frame = pd.DataFrame({"event_group": event_groups, "time": frame["time"], "split": frame["split"]})
    group_info = group_frame.groupby("event_group", sort=False).agg(
        first_time=("time", "min"),
        split_count=("split", "nunique"),
        split=("split", "first"),
    ).reset_index().sort_values("first_time")
    if (group_info["split_count"] != 1).any():
        raise ValueError("An event_group belongs to more than one original split.")
    development = group_info.loc[~group_info["split"].eq("test"), "event_group"].to_numpy(dtype=np.int16)
    if len(development) < fold_count + 2:
        raise ValueError("Too few development event blocks for forward validation.")
    initial_count = int(np.ceil(len(development) * initial_train_fraction))
    initial_count = max(1, min(initial_count, len(development) - fold_count))
    validation_chunks = [chunk.astype(np.int16) for chunk in np.array_split(development[initial_count:], fold_count)]
    if any(len(chunk) == 0 for chunk in validation_chunks):
        raise ValueError("A forward-validation fold would have no validation event blocks.")
    folds = []
    for fold, validation_groups in enumerate(validation_chunks, start=1):
        validation_start = initial_count + sum(len(chunk) for chunk in validation_chunks[:fold - 1])
        folds.append({
            "fold": np.asarray([fold], dtype=np.int16),
            "train_groups": development[:validation_start],
            "validation_groups": validation_groups,
        })
    return folds


def train_forward_fold(
    raw_features: np.ndarray,
    labels: np.ndarray,
    source_valid: np.ndarray,
    event_groups: np.ndarray,
    all_times: pd.DatetimeIndex,
    test_indices: np.ndarray,
    fold_definition: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    run_dir: Path,
    feature_names: list[str],
) -> dict[str, object]:
    """Fit one expanding-window fold and return OOF and test probabilities."""
    fold = int(fold_definition["fold"][0])
    fold_labels = np.full(len(event_groups), "unused", dtype=object)
    fold_labels[np.isin(event_groups, fold_definition["train_groups"])] = "train"
    fold_labels[np.isin(event_groups, fold_definition["validation_groups"])] = "validation"
    source_sequence_indices, _ = feature_role_indices(feature_names)
    # Rolling summaries are only consumed at the input endpoint t-L.  Do not
    # discard the first W-1 rows of every event block while fitting feature
    # selection/scaling merely because their rolling context is undefined.
    scaler_rows = (
        (fold_labels == "train")
        & np.isfinite(raw_features[:, source_sequence_indices]).all(axis=1)
    )
    if not scaler_rows.any():
        raise RuntimeError(f"Forward fold {fold} has no finite rows for feature scaling.")
    selected_features = select_nonredundant_features(
        raw_features, feature_names, scaler_rows, args.feature_correlation_threshold,
    )
    scaler = StandardScaler().fit(raw_features[scaler_rows][:, selected_features])
    features = scaler.transform(raw_features[:, selected_features]).astype(np.float32)
    selected_feature_names = [feature_names[index] for index in selected_features]
    sequence_feature_indices, context_feature_indices = feature_role_indices(selected_feature_names)
    train_idx = build_indices(
        features, labels, source_valid, fold_labels, args.lag, args.window, "train", args.train_stride,
        sequence_feature_indices, context_feature_indices,
    )
    val_idx = build_indices(
        features, labels, source_valid, fold_labels, args.lag, args.window, "validation", args.eval_stride,
        sequence_feature_indices, context_feature_indices,
    )
    if not len(train_idx) or not len(val_idx):
        raise RuntimeError(f"Forward fold {fold} has no valid training or validation samples.")
    positive = labels[train_idx].sum(axis=0)
    primary_positive = positive[:PRIMARY_OUTPUT_COUNT]
    if np.any(primary_positive == 0):
        missing = [str(THRESHOLDS[i]) for i in np.flatnonzero(primary_positive == 0)]
        raise RuntimeError(f"Forward fold {fold} has no training positives for: {', '.join(missing)} A")
    train_ds = LagWindowDataset(features, labels, train_idx, args.lag, args.window, context_feature_indices)
    val_ds = LagWindowDataset(features, labels, val_idx, args.lag, args.window, context_feature_indices)
    test_ds = LagWindowDataset(features, labels, test_indices, args.lag, args.window, context_feature_indices)
    loaders = {
        "train": make_loader(train_ds, args.batch_size, True, args),
        "validation": make_loader(val_ds, args.batch_size, False, args),
        "test": make_loader(test_ds, args.batch_size, False, args),
    }
    set_seed(args.seed + fold)
    pos_weight = torch.tensor((len(train_idx) - positive) / np.maximum(positive, 1.0), dtype=torch.float32, device=device)
    model = CNNBiLSTMClassifier(
        features.shape[1], channels=args.channels, lstm_hidden_size=args.lstm_hidden_size,
        lstm_layers=args.lstm_layers, dropout=args.dropout,
        attention_hidden_size=args.attention_hidden_size, output_count=TOTAL_OUTPUT_COUNT,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr,
    )
    history: list[dict[str, float | int]] = []
    best_event_csi, best_loss, stale, best_epoch = -np.inf, np.inf, 0, 0
    fallback_loss, checkpoint_feasible = np.inf, False
    checkpoint_path = run_dir / f"fold_{fold:02d}_best_model.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_weighted_bce = 0.0
        train_peak_bce = 0.0
        train_onset_bce = 0.0
        train_consistency = 0.0
        for x, y in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x.to(device, non_blocking=True))
            loss, peak_bce, onset_bce, consistency = multilabel_forecast_loss(
                logits, y.to(device, non_blocking=True), pos_weight, args.monotonicity_weight,
                args.onset_loss_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.item()) * len(x)
            train_weighted_bce += float((peak_bce + args.onset_loss_weight * onset_bce).item()) * len(x)
            train_peak_bce += float(peak_bce.item()) * len(x)
            train_onset_bce += float(onset_bce.item()) * len(x)
            train_consistency += float(consistency.item()) * len(x)
        val_logits, val_y = predict_logits(model, loaders["validation"], device)
        val_logits_tensor = torch.as_tensor(val_logits, dtype=torch.float32, device=device)
        val_y_tensor = torch.as_tensor(val_y, dtype=torch.float32, device=device)
        primary_logits = val_logits_tensor[:, :PRIMARY_OUTPUT_COUNT]
        primary_targets = val_y_tensor[:, :PRIMARY_OUTPUT_COUNT]
        onset_logits = val_logits_tensor[:, PRIMARY_OUTPUT_COUNT:]
        onset_targets = val_y_tensor[:, PRIMARY_OUTPUT_COUNT:]
        # Model selection and the learning-rate schedule remain governed by
        # the four operational future-peak outputs, not the auxiliary head.
        val_loss = float(F.binary_cross_entropy_with_logits(primary_logits, primary_targets).item())
        val_weighted_loss = float(F.binary_cross_entropy_with_logits(
            primary_logits, primary_targets, pos_weight=pos_weight[:PRIMARY_OUTPUT_COUNT],
        ).item())
        val_onset_bce = float(F.binary_cross_entropy_with_logits(onset_logits, onset_targets).item())
        val_onset_weighted_bce = float(F.binary_cross_entropy_with_logits(
            onset_logits, onset_targets, pos_weight=pos_weight[PRIMARY_OUTPUT_COUNT:],
        ).item())
        event_csi, event_pofd = validation_event_checkpoint_score(
            val_logits, val_y, all_times[val_idx], event_groups[val_idx], args,
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append({
            "fold": fold,
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_loss / len(train_ds),
            "train_monotonicity_penalty": train_consistency / len(train_ds),
            "train_weighted_bce": train_weighted_bce / len(train_ds),
            "train_peak_weighted_bce": train_peak_bce / len(train_ds),
            "train_onset_weighted_bce": train_onset_bce / len(train_ds),
            "validation_bce": val_loss,
            "validation_weighted_bce": val_weighted_loss,
            "validation_onset_bce": val_onset_bce,
            "validation_onset_weighted_bce": val_onset_weighted_bce,
            "validation_event_CSI_checkpoint": event_csi,
            "validation_event_POFD_checkpoint": event_pofd,
            "checkpoint_selected": 0,
        })
        print(
            f"fold={fold} epoch={epoch:03d} lr={learning_rate:.2e} "
            f"train_w={history[-1]['train_weighted_bce']:.5f} val={val_loss:.5f} "
            f"onset_val={val_onset_bce:.5f} "
            f"event_csi={event_csi:.5f} event_pofd={event_pofd:.5f}"
        )
        scheduler.step(val_loss)
        loss_improved = val_loss < fallback_loss - 1e-5
        if loss_improved:
            fallback_loss, stale = val_loss, 0
        else:
            stale += 1
        feasible_event_score = event_pofd <= args.event_pofd_cap
        event_score_improved = feasible_event_score and event_csi > best_event_csi + 1e-5
        if event_score_improved:
            best_event_csi = event_csi
        checkpoint_selected = False
        if args.checkpoint_metric == "validation_bce":
            # BCE is a dense, stable score on every validation minute.  The
            # event CSI remains useful for monitoring and final OOF cutoff
            # selection, but is too discontinuous to select an epoch here.
            if loss_improved:
                best_loss, best_epoch = val_loss, epoch
                checkpoint_feasible = feasible_event_score
                checkpoint_selected = True
                selection_metric = "validation_BCE"
        elif event_score_improved:
            # Retain the previous event-level policy as an explicit ablation.
            best_loss, best_epoch = val_loss, epoch
            checkpoint_feasible = True
            checkpoint_selected = True
            selection_metric = "validation_macro_event_CSI"
        elif not checkpoint_feasible and loss_improved:
            # A valid fallback is still needed if no epoch meets the POFD cap.
            best_loss, best_epoch = val_loss, epoch
            checkpoint_selected = True
            selection_metric = "validation_BCE_fallback"

        if checkpoint_selected:
            history[-1]["checkpoint_selected"] = 1
            torch.save({
                "model_state": model.state_dict(), "feature_names": feature_names,
                "args": vars(args), "fold": fold, "selection_metric": selection_metric,
                "selection_validation_BCE": val_loss,
                "selection_validation_onset_BCE": val_onset_bce,
                "selection_event_CSI": event_csi, "selection_event_POFD": event_pofd,
            }, checkpoint_path)
        if stale >= args.patience:
            break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    val_logits, val_y = predict_logits(model, loaders["validation"], device)
    temperature = fit_temperature(val_logits, val_y, device)
    val_prob = 1.0 / (1.0 + np.exp(-(val_logits / temperature)))
    test_logits, test_y = predict_logits(model, loaders["test"], device)
    test_prob = 1.0 / (1.0 + np.exp(-(test_logits / temperature)))
    summary: dict[str, float | int] = {
        "fold": fold,
        "train_event_group_count": int(len(fold_definition["train_groups"])),
        "validation_event_group_count": int(len(fold_definition["validation_groups"])),
        "train_sample_count": int(len(train_idx)),
        "validation_sample_count": int(len(val_idx)),
        "best_epoch": best_epoch,
        "best_validation_bce": float(best_loss),
        "checkpoint_metric": checkpoint["selection_metric"],
        "checkpoint_validation_macro_event_CSI": float(checkpoint["selection_event_CSI"]),
        "maximum_validation_macro_event_CSI": float(best_event_csi),
        "checkpoint_event_pofd_feasible": int(checkpoint_feasible),
        "selected_feature_count": int(len(selected_features)),
    }
    summary.update({f"temperature_{threshold}A": float(temperature[index]) for index, threshold in enumerate(THRESHOLDS)})
    summary.update({
        f"temperature_onset_{threshold}A": float(temperature[PRIMARY_OUTPUT_COUNT + index])
        for index, threshold in enumerate(THRESHOLDS)
    })
    return {
        "fold": fold,
        "scaler": scaler,
        "selected_feature_indices": selected_features,
        "selected_feature_names": selected_feature_names,
        "sequence_feature_indices": sequence_feature_indices,
        "context_feature_indices": context_feature_indices,
        "history": history,
        "summary": summary,
        "validation_indices": val_idx,
        "validation_labels": val_y,
        "validation_probability": val_prob,
        "test_labels": test_y,
        "test_probability": test_prob,
    }


def plot_test_event_probability_curves(
    all_times: pd.DatetimeIndex,
    all_gic_abs: np.ndarray,
    prediction_times: pd.DatetimeIndex,
    labels: np.ndarray,
    probabilities: np.ndarray,
    cutoffs: np.ndarray,
    events: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write one five-panel absolute-GIC/probability plot for every test event."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for event in events.loc[events["split"].eq("test")].itertuples(index=False):
        event_rows = (all_times >= pd.Timestamp(event.event_window_start)) & (all_times < pd.Timestamp(event.event_window_end))
        prediction_rows = (prediction_times >= pd.Timestamp(event.event_window_start)) & (prediction_times < pd.Timestamp(event.event_window_end))
        if not event_rows.any() or not prediction_rows.any():
            continue
        figure, axes = plt.subplots(len(THRESHOLDS) + 1, 1, figsize=(18, 12), sharex=True, constrained_layout=True)
        gic_axis = axes[0]
        gic_axis.plot(all_times[event_rows], all_gic_abs[event_rows], color="#111111", linewidth=0.55, label="Observed |GIC|")
        gic_axis.set_ylim(bottom=0.0)
        gic_axis.set_ylabel("|GIC| [A]")
        gic_axis.grid(alpha=0.25, linewidth=0.4)
        gic_axis.legend(loc="upper right", fontsize=8, frameon=False)
        for column, (axis, threshold) in enumerate(zip(axes[1:], THRESHOLDS)):
            axis.plot(prediction_times[prediction_rows], probabilities[prediction_rows, column], color="#1f4e79", linewidth=0.65, label="Predicted probability")
            axis.axhline(cutoffs[column], color="#d95f02", linestyle="--", linewidth=0.9, label="Decision cutoff")
            observed = labels[prediction_rows, column].astype(bool)
            axis.scatter(prediction_times[prediction_rows][observed], np.ones(observed.sum()), s=7, color="#b2182b", alpha=0.8, label="Observed target")
            axis.set_ylim(-0.02, 1.02)
            axis.set_ylabel(f"> {threshold} A")
            axis.grid(alpha=0.25, linewidth=0.4)
            axis.legend(loc="upper right", fontsize=8, frameon=False)
        axes[-1].set_xlabel("UTC time")
        peak = pd.Timestamp(event.peak_time).strftime("%Y-%m-%d %H:%M UTC")
        figure.suptitle(f"Test event {int(event.event_id)} ({event.driver}, peak {peak})", fontsize=14)
        filename = f"event_{int(event.event_id):03d}_{pd.Timestamp(event.peak_time):%Y%m%dT%H%M}.png"
        figure.savefig(output_dir / filename, dpi=180)
        plt.close(figure)


def write_lag_window_comparison(output_root: Path, event_pofd_cap: float = 0.20) -> Path:
    """Aggregate all completed L/W runs without using test metrics for selection."""
    rows = []
    for lag in (30, 45, 60, 90):
        for window in (30, 60, 120):
            run_dir = output_root / f"L{lag}_W{window}"
            test_path = run_dir / "test_metrics.csv"
            validation_path = run_dir / "validation_decision_thresholds.csv"
            history_path = run_dir / "training_history.csv"
            event_path = run_dir / "test_event_metrics.csv"
            if not all(path.exists() for path in (test_path, validation_path, history_path, event_path)):
                raise FileNotFoundError(f"Missing completed outputs for L={lag}, W={window}: {run_dir}")
            test = pd.read_csv(test_path)
            validation = pd.read_csv(validation_path).rename(
                columns=lambda name: name if name == "threshold_A" or name.startswith("validation_") else f"validation_{name}"
            )
            event_metrics = pd.read_csv(event_path).rename(
                columns=lambda name: name if name == "threshold_A" else f"event_{name}"
            )
            history = pd.read_csv(history_path)
            fold_summary_path = run_dir / "forward_fold_summary.csv"
            if fold_summary_path.exists():
                fold_summary = pd.read_csv(fold_summary_path)
                best_epoch = int(round(fold_summary["best_epoch"].mean()))
                best_validation_bce = float(fold_summary["best_validation_bce"].mean())
            else:
                best = history.loc[history["validation_bce"].idxmin()]
                best_epoch = int(best["epoch"])
                best_validation_bce = float(best["validation_bce"])
            merged = test.merge(validation, on="threshold_A", how="left").merge(event_metrics, on="threshold_A", how="left")
            merged.insert(0, "window", window)
            merged.insert(0, "lag", lag)
            merged["best_validation_epoch"] = best_epoch
            merged["best_validation_bce"] = best_validation_bce
            rows.append(merged)
    comparison = pd.concat(rows, ignore_index=True)
    output_path = output_root / "all_lag_window_metrics.csv"
    comparison.to_csv(output_path, index=False)

    # The rank deliberately uses validation statistics only.  Test statistics
    # remain in the comprehensive table for the one-time final comparison.
    selection_columns = [
        "lag", "window", "threshold_A", "best_validation_epoch", "best_validation_bce",
        "validation_decision_threshold", "validation_selection_status",
        "validation_POD", "validation_FAR", "validation_CSI", "validation_F1",
        "validation_Bias", "validation_ROC_AUC", "validation_PR_AUC",
    ]
    selection = comparison.loc[:, selection_columns].copy()
    selection["validation_bias_distance"] = (selection["validation_Bias"] - 1.0).abs()
    selection = selection.sort_values(
        ["threshold_A", "validation_CSI", "validation_FAR", "validation_bias_distance"],
        ascending=[True, False, True, True],
        na_position="last",
    )
    selection["validation_selection_rank"] = selection.groupby("threshold_A").cumcount() + 1
    selection.to_csv(output_root / "all_lag_window_validation_ranking.csv", index=False)

    # Select one global W from validation event metrics, then compare L at W*.
    window_summary = (
        comparison.groupby("window", as_index=False)
        .agg(
            validation_event_F1=("validation_event_F1", "mean"),
            validation_event_CSI=("validation_event_CSI", "mean"),
            validation_event_POD=("validation_event_POD", "mean"),
            validation_event_POFD=("validation_event_POFD", "mean"),
            validation_event_FAR=("validation_event_FAR", "mean"),
        )
    )
    feasible_windows = window_summary.loc[window_summary["validation_event_POFD"].le(event_pofd_cap)]
    if feasible_windows.empty:
        selected_window = int(window_summary.sort_values(
            ["validation_event_POFD", "validation_event_F1"], ascending=[True, False], na_position="last"
        ).iloc[0]["window"])
        window_selection_status = "minimum_validation_event_POFD_no_window_under_cap"
    else:
        selected_window = int(feasible_windows.sort_values(
            ["validation_event_CSI", "validation_event_POD", "validation_event_POFD"],
            ascending=[False, False, True], na_position="last",
        ).iloc[0]["window"])
        window_selection_status = f"maximum_validation_event_CSI_with_event_POFD_at_most_{event_pofd_cap:.2f}"
    window_summary["selected_window"] = (window_summary["window"] == selected_window).astype("int8")
    window_summary["selection_status"] = window_selection_status
    window_summary.to_csv(output_root / "validation_window_selection.csv", index=False)
    comparison.loc[comparison["window"].eq(selected_window)].sort_values(["lag", "threshold_A"]).to_csv(
        output_root / "fixed_window_lag_comparison.csv", index=False,
    )
    (output_root / "selected_window.json").write_text(
        json.dumps({"selected_window": selected_window, "event_pofd_cap": event_pofd_cap,
                    "selection_status": window_selection_status}, indent=2),
        encoding="utf-8",
    )

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    lags, windows = (30, 45, 60, 90), (30, 60, 120)
    for axis, threshold in zip(axes.ravel(), THRESHOLDS):
        values = np.full((len(lags), len(windows)), np.nan, dtype=np.float64)
        subset = comparison.loc[comparison["threshold_A"].eq(threshold)]
        for row in subset.itertuples(index=False):
            values[lags.index(int(row.lag)), windows.index(int(row.window))] = row.CSI
        image = axis.imshow(values, vmin=0.0, vmax=max(0.1, np.nanmax(values)), cmap="YlGnBu", aspect="auto")
        axis.set_xticks(range(len(windows)), [f"W={value}" for value in windows])
        axis.set_yticks(range(len(lags)), [f"L={value}" for value in lags])
        axis.set_title(f"Test CSI: |GIC| > {threshold} A")
        for row in range(len(lags)):
            for column in range(len(windows)):
                axis.text(column, row, f"{values[row, column]:.3f}", ha="center", va="center", fontsize=8)
        figure.colorbar(image, ax=axis, shrink=0.82)
    figure.savefig(output_root / "all_lag_window_test_csi.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, threshold in zip(axes.ravel(), THRESHOLDS):
        values = np.full((len(lags), len(windows)), np.nan, dtype=np.float64)
        subset = comparison.loc[comparison["threshold_A"].eq(threshold)]
        for row in subset.itertuples(index=False):
            values[lags.index(int(row.lag)), windows.index(int(row.window))] = row.validation_CSI
        image = axis.imshow(values, vmin=0.0, vmax=max(0.1, np.nanmax(values)), cmap="YlGnBu", aspect="auto")
        axis.set_xticks(range(len(windows)), [f"W={value}" for value in windows])
        axis.set_yticks(range(len(lags)), [f"L={value}" for value in lags])
        axis.set_title(f"Validation CSI: |GIC| > {threshold} A")
        for row in range(len(lags)):
            for column in range(len(windows)):
                axis.text(column, row, f"{values[row, column]:.3f}", ha="center", va="center", fontsize=8)
        figure.colorbar(image, ax=axis, shrink=0.82)
    figure.savefig(output_root / "all_lag_window_validation_csi.png", dpi=180)
    plt.close(figure)
    return output_path


def run_forward_validation(args: argparse.Namespace) -> None:
    """Train expanding chronological folds and evaluate their ensemble once on test."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_cuda(device)
    run_dir = args.output_root / f"L{args.lag}_W{args.window}"
    run_dir.mkdir(parents=True, exist_ok=True)
    frame, raw_features, peak_labels, onset_labels, source_valid, feature_names = load_timeline(
        args.lag, args.window, temporal_feature_groups=args.temporal_feature_groups,
        window_statistics=args.window_statistics,
    )
    labels = np.column_stack([peak_labels, onset_labels]).astype(np.float32, copy=False)
    if labels.shape[1] != TOTAL_OUTPUT_COUNT:
        raise RuntimeError(f"Expected {TOTAL_OUTPUT_COUNT} target columns, received {labels.shape[1]}.")
    original_split = frame["split"].astype(str).to_numpy()
    event_groups = pd.to_numeric(frame["event_group"], errors="coerce").to_numpy(dtype=np.int16)
    all_times = pd.DatetimeIndex(frame["time"])
    all_gic_abs = pd.to_numeric(frame["GIC_abs"], errors="coerce").to_numpy(dtype=np.float32)
    event_catalog = pd.read_csv(EVENT_CATALOG_PATH, parse_dates=["peak_time", "event_window_start", "event_window_end"])
    test_labels = np.where(original_split == "test", "test", "unused")
    sequence_feature_indices, context_feature_indices = feature_role_indices(feature_names)
    test_indices = build_indices(
        raw_features, labels, source_valid, test_labels, args.lag, args.window, "test", args.eval_stride,
        sequence_feature_indices, context_feature_indices,
    )
    if not len(test_indices):
        raise RuntimeError("The fixed chronological test split has no valid samples.")
    fold_definitions = build_forward_folds(frame, event_groups, args.forward_folds, args.initial_train_fraction)
    group_start = frame.groupby("event_group", sort=False)["time"].min().to_dict()
    fold_definition_rows = []
    results = []
    for definition in fold_definitions:
        fold = int(definition["fold"][0])
        train_groups = definition["train_groups"]
        validation_groups = definition["validation_groups"]
        fold_definition_rows.append({
            "fold": fold,
            "train_event_group_count": len(train_groups),
            "validation_event_group_count": len(validation_groups),
            "train_last_group": int(train_groups[-1]),
            "train_last_time": group_start[int(train_groups[-1])],
            "validation_first_group": int(validation_groups[0]),
            "validation_first_time": group_start[int(validation_groups[0])],
            "validation_last_group": int(validation_groups[-1]),
            "validation_last_time": group_start[int(validation_groups[-1])],
        })
        results.append(train_forward_fold(
            raw_features, labels, source_valid, event_groups, all_times, test_indices, definition,
            args, device, run_dir, feature_names,
        ))
    histories = pd.DataFrame([row for result in results for row in result["history"]])
    fold_summary = pd.DataFrame([result["summary"] for result in results])
    histories.to_csv(run_dir / "training_history.csv", index=False)
    fold_summary.to_csv(run_dir / "forward_fold_summary.csv", index=False)
    pd.DataFrame(fold_definition_rows).to_csv(run_dir / "forward_fold_definitions.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    joblib.dump(
        {
            "fold_scalers": [result["scaler"] for result in results],
            "selected_feature_indices": [result["selected_feature_indices"] for result in results],
            "selected_feature_names": [result["selected_feature_names"] for result in results],
            "sequence_feature_indices": [result["sequence_feature_indices"] for result in results],
            "context_feature_indices": [result["context_feature_indices"] for result in results],
            "source_feature_names": feature_names, "log1p_features": sorted(LOG1P_FEATURES),
        },
        run_dir / "feature_scalers.joblib",
    )

    oof_indices = np.concatenate([result["validation_indices"] for result in results]).astype(np.int64)
    oof_labels = np.concatenate([result["validation_labels"] for result in results])
    oof_probability = np.concatenate([result["validation_probability"] for result in results])
    oof_folds = np.concatenate([
        np.full(len(result["validation_indices"]), int(result["fold"]), dtype=np.int16) for result in results
    ])
    if len(np.unique(oof_indices)) != len(oof_indices):
        raise RuntimeError("Forward-validation folds overlap at the sample level.")
    oof_order = np.argsort(oof_indices)
    oof_indices, oof_labels, oof_probability, oof_folds = (
        oof_indices[oof_order], oof_labels[oof_order], oof_probability[oof_order], oof_folds[oof_order]
    )
    oof_times = all_times[oof_indices]
    oof_groups = event_groups[oof_indices]
    test_labels_from_folds = [result["test_labels"] for result in results]
    if any(not np.array_equal(test_labels_from_folds[0], values) for values in test_labels_from_folds[1:]):
        raise RuntimeError("Test labels differ across forward-validation folds.")
    test_y = test_labels_from_folds[0]
    test_prob = np.mean(np.stack([result["test_probability"] for result in results]), axis=0)

    cutoff_rows, test_rows, tradeoff_tables, event_search_tables, selected_postprocess = [], [], [], [], []
    for column, threshold in enumerate(THRESHOLDS):
        if args.cutoff_policy == "event_pofd":
            cutoff, validation_event_metrics, selection_status, postprocess_params, event_search = select_event_operating_point(
                oof_labels[:, column], oof_probability[:, column], oof_times, oof_groups,
                args.event_pofd_cap, args.event_pod_floor, settings=EVENT_POSTPROCESS_SETTINGS,
            )
            validation_metrics = metrics(oof_labels[:, column], oof_probability[:, column], cutoff)
            event_search["threshold_A"] = threshold
            event_search["selected"] = (
                np.isclose(event_search["decision_threshold"], cutoff)
                & event_search["min_consecutive"].eq(postprocess_params["min_consecutive"])
                & event_search["merge_gap"].eq(postprocess_params["merge_gap"])
                & event_search["min_event_length"].eq(postprocess_params["min_event_length"])
            ).astype("int8")
            event_search_tables.append(event_search)
        else:
            cutoff, validation_metrics, selection_status, _ = select_cutoff(
                oof_labels[:, column], oof_probability[:, column], args.cutoff_policy, args.max_far,
            )
            postprocess_params = {"min_consecutive": DEFAULT_POSTPROCESS[0], "merge_gap": DEFAULT_POSTPROCESS[1], "min_event_length": DEFAULT_POSTPROCESS[2]}
            validation_event_metrics = _event_metrics_from_masks(
                oof_labels[:, column],
                _postprocess_mask(oof_probability[:, column] >= cutoff, oof_times, oof_groups, **postprocess_params),
                oof_times, oof_groups,
            )
        _, _, _, tradeoff = select_cutoff(
            oof_labels[:, column], oof_probability[:, column], "csi", args.max_far,
        )
        tradeoff["threshold_A"] = threshold
        tradeoff["cutoff_policy"] = args.cutoff_policy
        tradeoff["max_far"] = args.max_far
        tradeoff["selected"] = np.isclose(tradeoff["decision_threshold"], cutoff).astype("int8")
        tradeoff_tables.append(tradeoff)
        mean_temperature = float(np.mean([result["summary"][f"temperature_{threshold}A"] for result in results]))
        cutoff_rows.append({
            "threshold_A": threshold,
            "temperature_mean": mean_temperature,
            "cutoff_policy": args.cutoff_policy,
            "max_far": args.max_far,
            "selection_status": selection_status,
            "decision_threshold": cutoff,
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_event_metrics.items()},
            **postprocess_params,
        })
        selected_postprocess.append(postprocess_params)
        raw_test_metrics = metrics(test_y[:, column], test_prob[:, column], cutoff)
        processed_test_prediction = _postprocess_mask(
            test_prob[:, column] >= cutoff, all_times[test_indices], event_groups[test_indices], **postprocess_params,
        )
        processed_test_metrics = _event_metrics_from_masks(
            test_y[:, column], processed_test_prediction, all_times[test_indices], event_groups[test_indices],
        )
        test_rows.append({
            "threshold_A": threshold,
            "cutoff_policy": args.cutoff_policy,
            "max_far": args.max_far,
            "selection_status": selection_status,
            "decision_threshold": cutoff,
            **raw_test_metrics,
            **{f"postprocessed_{key}": value for key, value in processed_test_metrics.items()},
            **postprocess_params,
        })
    tradeoff_table = pd.concat(tradeoff_tables, ignore_index=True)
    pd.DataFrame(cutoff_rows).to_csv(run_dir / "validation_decision_thresholds.csv", index=False)
    pd.DataFrame(test_rows).to_csv(run_dir / "test_metrics.csv", index=False)
    tradeoff_table.to_csv(run_dir / "validation_threshold_tradeoffs.csv", index=False)
    if event_search_tables:
        pd.concat(event_search_tables, ignore_index=True).to_csv(
            run_dir / "validation_event_operating_point_search.csv", index=False,
        )
    plot_validation_threshold_tradeoffs(tradeoff_table, run_dir / "validation_threshold_tradeoffs.png")

    oof_prediction = pd.DataFrame({
        "time": all_times[oof_indices], "event_group": event_groups[oof_indices], "forward_fold": oof_folds,
    })
    for column, threshold in enumerate(THRESHOLDS):
        oof_prediction[f"target_{threshold}A"] = oof_labels[:, column].astype("int8")
        oof_prediction[f"probability_{threshold}A"] = oof_probability[:, column]
        onset_column = PRIMARY_OUTPUT_COUNT + column
        oof_prediction[f"target_onset_{threshold}A"] = oof_labels[:, onset_column].astype("int8")
        oof_prediction[f"probability_onset_{threshold}A"] = oof_probability[:, onset_column]
    oof_prediction.to_csv(run_dir / "forward_validation_oof_probabilities.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")

    test_times = all_times[test_indices]
    test_events = event_catalog.loc[event_catalog["split"].eq("test")]
    event_ids = np.full(len(test_times), "", dtype=object)
    for event in test_events.itertuples(index=False):
        event_rows = (test_times >= event.event_window_start) & (test_times < event.event_window_end)
        event_ids[event_rows] = np.where(event_ids[event_rows] == "", str(event.event_id), event_ids[event_rows] + "," + str(event.event_id))
    prediction = pd.DataFrame({"time": test_times, "GIC_abs": all_gic_abs[test_indices], "event_group": event_groups[test_indices], "event_ids": event_ids})
    for column, threshold in enumerate(THRESHOLDS):
        cutoff = float(test_rows[column]["decision_threshold"])
        prediction[f"target_{threshold}A"] = test_y[:, column].astype("int8")
        prediction[f"probability_{threshold}A"] = test_prob[:, column]
        onset_column = PRIMARY_OUTPUT_COUNT + column
        prediction[f"target_onset_{threshold}A"] = test_y[:, onset_column].astype("int8")
        prediction[f"probability_onset_{threshold}A"] = test_prob[:, onset_column]
        prediction[f"decision_threshold_{threshold}A"] = cutoff
        prediction[f"raw_prediction_{threshold}A"] = (test_prob[:, column] >= cutoff).astype("int8")
        prediction[f"prediction_{threshold}A"] = _postprocess_mask(
            test_prob[:, column] >= cutoff, test_times, event_groups[test_indices], **selected_postprocess[column],
        ).astype("int8")
    prediction.to_csv(run_dir / "test_probabilities.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    cutoffs = np.asarray([row["decision_threshold"] for row in test_rows], dtype=np.float32)
    alarm_episodes, event_warnings, event_metrics = evaluate_test_events(
        all_times, all_gic_abs, test_times, event_groups[test_indices], test_y, test_prob, cutoffs,
        selected_postprocess, event_catalog,
    )
    cutoff_by_threshold = {int(row["threshold_A"]): float(row["decision_threshold"]) for row in test_rows}
    for table in (alarm_episodes, event_warnings, event_metrics):
        table["decision_threshold"] = table["threshold_A"].map(cutoff_by_threshold)
        table["cutoff_policy"] = args.cutoff_policy
        table["max_far"] = args.max_far
    postprocess_by_threshold = {int(threshold): params for threshold, params in zip(THRESHOLDS, selected_postprocess)}
    for table in (alarm_episodes, event_warnings, event_metrics):
        table["min_consecutive"] = table["threshold_A"].map(lambda value: postprocess_by_threshold[int(value)]["min_consecutive"])
        table["merge_gap"] = table["threshold_A"].map(lambda value: postprocess_by_threshold[int(value)]["merge_gap"])
        table["min_event_length"] = table["threshold_A"].map(lambda value: postprocess_by_threshold[int(value)]["min_event_length"])
    alarm_episodes.to_csv(run_dir / "test_alarm_episodes.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    event_warnings.to_csv(run_dir / "test_event_warnings.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    event_metrics.to_csv(run_dir / "test_event_metrics.csv", index=False)
    plot_test_event_probability_curves(
        all_times, all_gic_abs, test_times, test_y, test_prob, cutoffs, event_catalog,
        run_dir / "test_event_probability_curves",
    )
    default_encoder_metadata = {
        "type": getattr(CNNBiLSTMClassifier, "encoder_type", "multi_scale_residual_cnn_then_attentive_bilstm"),
        "cnn_kernels": [3, 3],
        "cnn_dilations": [1, 4],
        "cnn_effective_receptive_field_minutes": 11,
        "cnn_channels": args.channels,
        "attention_hidden_size": args.attention_hidden_size,
        "pooling": getattr(CNNBiLSTMClassifier, "pooling_type", "last_layer_bidirectional_terminal_state_plus_learned_attention"),
        "fusion": "elementwise_gate(cnn_mean_max, bilstm_terminal_attention)",
        "fusion_representation_size": args.channels * 2,
    }
    describe_encoder = getattr(CNNBiLSTMClassifier, "describe_encoder", None)
    encoder_metadata = describe_encoder(args) if callable(describe_encoder) else default_encoder_metadata
    metadata = {
        "args": vars(args),
        "device": str(device),
        "source_feature_names": feature_names,
        "feature_selection": {
            "method": "training_fold_pairwise_absolute_correlation_filter",
            "correlation_threshold": args.feature_correlation_threshold,
            "selected_feature_names_by_fold": [result["selected_feature_names"] for result in results],
            "sequence_feature_names": [feature_names[index] for index in sequence_feature_indices],
            "context_feature_names": [feature_names[index] for index in context_feature_indices],
        },
        "encoder": encoder_metadata,
        "loss": {
            "type": "primary_peak_weighted_bce_plus_weighted_onset_auxiliary_bce_plus_soft_monotonicity",
            "monotonicity_weight": args.monotonicity_weight,
            "onset_loss_weight": args.onset_loss_weight,
            "threshold_order": list(THRESHOLDS),
            "validation_event_selection_weights": list(args.selection_weights),
            "operational_output_count": PRIMARY_OUTPUT_COUNT,
            "auxiliary_output_count": PRIMARY_OUTPUT_COUNT,
        },
        "dataset": "CME/CIR event-window blocks from 03",
        "dataset_directory": str(DATASET_DIR),
        "validation": "expanding chronological forward folds over the pre-test event blocks",
        "fixed_test_event_count": int(test_events.shape[0]),
        "forward_fold_count": len(results),
        "forward_fold_sample_counts": fold_summary.to_dict(orient="records"),
        "test_sample_count": int(len(test_indices)),
        "oof_validation_sample_count": int(len(oof_indices)),
        "outputs": [
            "fold_XX_best_model.pt", "feature_scalers.joblib", "training_history.csv",
            "forward_fold_definitions.csv", "forward_fold_summary.csv", "forward_validation_oof_probabilities.csv",
            "validation_decision_thresholds.csv", "validation_threshold_tradeoffs.csv",
            "validation_event_operating_point_search.csv",
            "validation_threshold_tradeoffs.png", "test_metrics.csv", "test_probabilities.csv",
            "test_alarm_episodes.csv", "test_event_warnings.csv", "test_event_metrics.csv",
            "test_event_probability_curves/*.png",
        ],
        "note": "Each validation fold follows its training events in time; fixed test event blocks are never used for fitting, calibration, cutoff selection, or model selection.",
        "postprocessing": {
            "primary_setting_minutes": {
                "min_consecutive": DEFAULT_POSTPROCESS[0],
                "merge_gap": DEFAULT_POSTPROCESS[1],
                "min_event_length": DEFAULT_POSTPROCESS[2],
            },
            "candidate_settings_minutes": [list(setting) for setting in EVENT_POSTPROCESS_SETTINGS],
            "selection": "maximum event CSI subject to validation event POFD cap",
        },
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(f"CNN-BiLSTM forward-validation complete: {run_dir}")


def run_all_configurations(args: argparse.Namespace) -> Path:
    """Launch one isolated training process per lag/window setting."""
    for lag in (30, 45, 60, 90):
        for window in (30, 60, 120):
            run_dir = args.output_root / f"L{lag}_W{window}"
            if args.skip_completed and (run_dir / "run_metadata.json").exists():
                print(f"Skipping completed configuration: L={lag}, W={window}")
                continue
            command = [
                sys.executable, str(Path(__file__).resolve()), "--lag", str(lag), "--window", str(window),
                "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                "--num-workers", str(args.num_workers), "--prefetch-factor", str(args.prefetch_factor),
                "--train-stride", str(args.train_stride), "--eval-stride", str(args.eval_stride),
                "--lr", str(args.lr), "--weight-decay", str(args.weight_decay),
                "--channels", str(args.channels), "--dropout", str(args.dropout),
                "--lstm-hidden-size", str(args.lstm_hidden_size), "--lstm-layers", str(args.lstm_layers),
                "--attention-hidden-size", str(args.attention_hidden_size),
                "--feature-correlation-threshold", str(args.feature_correlation_threshold),
                "--monotonicity-weight", str(args.monotonicity_weight),
                "--onset-loss-weight", str(args.onset_loss_weight),
                "--selection-weights", ",".join(str(value) for value in args.selection_weights),
                "--checkpoint-metric", args.checkpoint_metric,
                "--patience", str(args.patience), "--lr-patience", str(args.lr_patience),
                "--lr-factor", str(args.lr_factor), "--min-lr", str(args.min_lr), "--seed", str(args.seed),
                "--forward-folds", str(args.forward_folds),
                "--initial-train-fraction", str(args.initial_train_fraction),
                "--dataset-dir", str(args.dataset_dir),
                "--cutoff-policy", args.cutoff_policy, "--max-far", str(args.max_far),
                "--event-pofd-cap", str(args.event_pofd_cap), "--event-pod-floor", str(args.event_pod_floor),
            ]
            if args.experiment_name:
                command.extend(["--experiment-name", args.experiment_name])
            if args.pin_memory:
                command.append("--pin-memory")
            command.extend(["--temporal-feature-groups", args.temporal_feature_groups_arg])
            if not args.window_statistics:
                command.append("--disable-window-statistics")
            subprocess.run(command, check=True)
    return write_lag_window_comparison(args.output_root, args.event_pofd_cap)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lag", type=int, default=45, choices=(30, 45, 60, 90))
    parser.add_argument("--window", type=int, default=60, choices=(30, 60, 120))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--experiment-name", default="context_onset_v1",
        help="Optional result subdirectory below outputs/cnn_bilstm (letters, digits, _ and - only).",
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=DATASET_DIR,
        help="Complete 03 output directory, relative to this script unless an absolute path is supplied.",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers; use 0 only for debugging.")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="Batches prefetched per DataLoader worker.")
    parser.add_argument("--pin-memory", action="store_true", default=torch.cuda.is_available(), help="Use pinned host memory for faster CUDA transfers.")
    parser.add_argument(
        "--temporal-feature-groups", default="accumulation",
        help="Comma-separated selection: accumulation (default), delta, statistics, all, or none.",
    )
    parser.add_argument(
        "--disable-temporal-features", action="store_true",
        help="Legacy alias for --temporal-feature-groups none.",
    )
    parser.add_argument(
        "--disable-window-statistics", dest="window_statistics", action="store_false", default=True,
        help="Disable 03 rolling-statistic features for an ablation run.",
    )
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--eval-stride", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--channels", type=int, default=32, help="CNN latent channel count before the BiLSTM.")
    parser.add_argument("--lstm-hidden-size", type=int, default=64, help="Per-direction BiLSTM hidden size.")
    parser.add_argument("--lstm-layers", type=int, default=2, help="Number of stacked bidirectional LSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.35, help="Dropout used in the CNN, BiLSTM, and classification head.")
    parser.add_argument("--attention-hidden-size", type=int, default=32, help="Hidden width of the temporal attention scorer.")
    parser.add_argument("--feature-correlation-threshold", type=float, default=0.95, help="Training-fold absolute-correlation cutoff used only to remove redundant inputs.")
    parser.add_argument("--monotonicity-weight", type=float, default=0.05, help="Soft penalty weight for p(20A) <= p(10A) <= p(5A) <= p(3A).")
    parser.add_argument(
        "--onset-loss-weight", type=float, default=0.5,
        help="Relative BCE weight for the four onset auxiliary outputs.",
    )
    parser.add_argument(
        "--selection-weights", default=",".join(str(value) for value in DEFAULT_SELECTION_WEIGHTS),
        help="Validation event-CSI weights for 3,5,10,20 A; defaults to 0.45,0.45,0.10,0.0.",
    )
    parser.add_argument(
        "--checkpoint-metric", choices=("validation_bce", "event_csi"), default="validation_bce",
        help="Epoch-selection metric. validation_bce is stable; event_csi uses weighted 3/5/10 A event CSI.",
    )
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr-patience", type=int, default=2, help="Validation-loss plateaus before halving the learning rate.")
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--forward-folds", type=int, default=3, help="Number of expanding chronological validation folds inside the pre-test events.")
    parser.add_argument("--initial-train-fraction", type=float, default=0.50, help="Earliest development event-block fraction used as the first fold's training period.")
    parser.add_argument("--all-configurations", action="store_true", help="Train all 4 lag x 3 window combinations and write a comparison summary.")
    parser.add_argument("--skip-completed", action="store_true", help="With --all-configurations, retain configurations that already have run_metadata.json.")
    parser.add_argument("--aggregate-only", action="store_true", help="Write the all-L/W comparison from completed run directories without training.")
    parser.add_argument(
        "--cutoff-policy",
        choices=("event_pofd", "far_cap", "bias_one", "csi"),
        default="event_pofd",
        help="Validation operating-point rule; event_pofd selects cutoff plus alarm post-processing from OOF event metrics.",
    )
    parser.add_argument("--max-far", type=float, default=0.30, help="Validation FAR cap used by --cutoff-policy far_cap.")
    parser.add_argument("--event-pofd-cap", type=float, default=0.20, help="Validation event POFD cap for event-level selection.")
    parser.add_argument("--event-pod-floor", type=float, default=0.50, help="Fallback validation event POD floor when the POFD cap is unattainable.")
    args = parser.parse_args()
    if args.experiment_name and not re.fullmatch(r"[A-Za-z0-9_-]+", args.experiment_name):
        raise ValueError("--experiment-name may contain only letters, digits, underscores, and hyphens.")
    args.dataset_dir = configure_dataset_paths(args.dataset_dir)
    args.temporal_feature_groups_arg = "none" if args.disable_temporal_features else args.temporal_feature_groups
    args.temporal_feature_groups = parse_temporal_feature_groups(args.temporal_feature_groups_arg)
    args.selection_weights = parse_selection_weights(args.selection_weights)
    args.output_root = OUTPUT_ROOT / args.experiment_name if args.experiment_name else OUTPUT_ROOT
    if not 0.0 <= args.max_far <= 1.0:
        raise ValueError("--max-far must be between 0 and 1.")
    if not 0.0 <= args.event_pofd_cap <= 1.0 or not 0.0 <= args.event_pod_floor <= 1.0:
        raise ValueError("--event-pofd-cap and --event-pod-floor must be between 0 and 1.")
    if args.channels < 1 or args.lstm_hidden_size < 1 or args.lstm_layers < 1 or args.attention_hidden_size < 1:
        raise ValueError("--channels, --lstm-hidden-size, --lstm-layers, and --attention-hidden-size must be positive.")
    if args.num_workers < 0 or args.prefetch_factor < 1:
        raise ValueError("--num-workers must be non-negative and --prefetch-factor must be positive.")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1).")
    if args.weight_decay < 0.0 or args.min_lr <= 0.0:
        raise ValueError("--weight-decay must be non-negative and --min-lr must be positive.")
    if not 0.0 < args.feature_correlation_threshold <= 1.0:
        raise ValueError("--feature-correlation-threshold must be in (0, 1].")
    if args.monotonicity_weight < 0.0:
        raise ValueError("--monotonicity-weight must be non-negative.")
    if args.onset_loss_weight < 0.0:
        raise ValueError("--onset-loss-weight must be non-negative.")
    if args.temporal_feature_groups and not args.window_statistics:
        raise ValueError("Temporal feature groups require window statistics; omit --disable-window-statistics.")
    if args.lr_patience < 0 or not 0.0 < args.lr_factor < 1.0:
        raise ValueError("--lr-patience must be non-negative and --lr-factor must be in (0, 1).")
    if args.forward_folds < 2 or not 0.0 < args.initial_train_fraction < 1.0:
        raise ValueError("--forward-folds must be at least 2 and --initial-train-fraction must be in (0, 1).")
    if args.aggregate_only:
        summary_path = write_lag_window_comparison(args.output_root, args.event_pofd_cap)
        print(f"Lag/window comparison complete: {summary_path}")
        return
    if args.all_configurations:
        summary_path = run_all_configurations(args)
        print(f"All lag/window configurations complete: {summary_path}")
        return
    run_forward_validation(args)
    return


if __name__ == "__main__":
    main()
