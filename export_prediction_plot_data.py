"""Export CSV data used by existing prediction comparison plots.

This is a post-processing utility. It reads cached model predictions and the
existing plot index CSVs under ``prediction_plots``; it does not retrain models
or regenerate figures.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_RUN_DIR = Path("outputs/experiments/paper_vkh_drivers_all/gic_vyk_abs_H30")
PLOT_COLUMNS = [
    "datetime",
    "y_true",
    "gic_true",
    "prob_ge_3A",
    "prob_ge_5A",
    "prob_ge_10A",
    "prob_ge_20A",
    "Q80",
    "Q90",
    "Q95",
    "final_pred",
    "final_pred_quantile",
]


def _read_predictions(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    missing = [col for col in PLOT_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    return frame[PLOT_COLUMNS].copy()


def _prediction_cache_files(cache_dir: Path) -> List[Path]:
    csv_files = sorted(cache_dir.glob("*_predictions.csv"))
    if csv_files:
        return csv_files
    return sorted(cache_dir.glob("*_predictions.parquet"))


def _group_from_cache_path(path: Path) -> str:
    suffix = "_predictions"
    stem = path.stem
    if not stem.endswith(suffix):
        raise ValueError(f"Unexpected prediction cache name: {path.name}")
    return stem[: -len(suffix)]


def _safe_int(value: object) -> int:
    if pd.isna(value):
        raise ValueError("row index is NaN")
    return int(value)


def _with_plot_x(frame: pd.DataFrame, start_row: int) -> pd.DataFrame:
    out = frame.copy()
    out.insert(0, "row", np.arange(start_row, start_row + len(out), dtype=int))
    out.insert(1, "plot_x", np.arange(len(out), dtype=int))
    return out


def _slice_rows(
    predictions: pd.DataFrame,
    start_row: int,
    end_row_exclusive: int,
    clip_to_available: bool = False,
) -> pd.DataFrame:
    if clip_to_available:
        start_row = max(0, min(start_row, len(predictions)))
        end_row_exclusive = max(start_row, min(end_row_exclusive, len(predictions)))
    if start_row < 0 or end_row_exclusive < start_row or end_row_exclusive > len(predictions):
        raise ValueError(
            f"Invalid row slice [{start_row}, {end_row_exclusive}) for predictions length {len(predictions)}"
        )
    return _with_plot_x(predictions.iloc[start_row:end_row_exclusive].reset_index(drop=True), start_row)


def _overview_window(predictions: pd.DataFrame, max_points: int) -> Tuple[int, int]:
    n_rows = len(predictions)
    if n_rows <= max_points:
        return 0, n_rows
    y_true = pd.to_numeric(predictions["y_true"], errors="coerce").to_numpy(dtype=float)
    if np.all(np.isnan(y_true)):
        center = n_rows // 2
    else:
        center = int(np.nanargmax(y_true))
    start = max(0, center - max_points // 2)
    end = min(n_rows, start + max_points)
    start = max(0, end - max_points)
    return start, end


def _export_overview_data(group: str, predictions: pd.DataFrame, plots_dir: Path, max_points: int) -> int:
    png_path = plots_dir / f"{group}_prediction_comparison.png"
    if not png_path.exists():
        return 0
    start, end = _overview_window(predictions, max_points)
    data = _slice_rows(predictions, start, end)
    data.insert(0, "plot_name", png_path.name)
    data.insert(1, "plot_type", "prediction_comparison")
    data.to_csv(plots_dir / f"{group}_prediction_comparison_data.csv", index=False, encoding="utf-8-sig")
    return 1


def _top_event_windows(predictions: pd.DataFrame, top_k: int, left: int, right: int, min_gap: int) -> List[Tuple[int, int, int]]:
    y_true = pd.to_numeric(predictions["y_true"], errors="coerce").to_numpy(dtype=float)
    if len(y_true) == 0 or np.all(np.isnan(y_true)):
        return []

    selected_centers: List[int] = []
    for center in np.argsort(np.nan_to_num(y_true, nan=-np.inf))[::-1]:
        center = int(center)
        if all(abs(center - existing) >= min_gap for existing in selected_centers):
            selected_centers.append(center)
        if len(selected_centers) >= top_k:
            break

    windows: List[Tuple[int, int, int]] = []
    for center in selected_centers:
        start = max(0, center - left)
        end = min(len(predictions), center + right)
        windows.append((center, start, end))
    return windows


def _export_top_events_data(group: str, predictions: pd.DataFrame, plots_dir: Path) -> int:
    png_path = plots_dir / f"{group}_top_events_quantile_detail.png"
    if not png_path.exists():
        return 0

    pieces: List[pd.DataFrame] = []
    for panel_no, (center, start, end) in enumerate(_top_event_windows(predictions, 6, 180, 240, 180), start=1):
        panel = _slice_rows(predictions, start, end)
        panel.insert(0, "plot_name", png_path.name)
        panel.insert(1, "plot_type", "top_events_quantile_detail")
        panel.insert(2, "panel_no", panel_no)
        panel.insert(3, "center_row", center)
        panel.insert(4, "panel_x", np.arange(len(panel), dtype=int))
        pieces.append(panel)

    if not pieces:
        return 0
    pd.concat(pieces, ignore_index=True).to_csv(
        plots_dir / f"{group}_top_events_quantile_detail_data.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return 1


def _event_png_by_key(event_dir: Path, group: str, event_kind: str) -> Dict[str, Path]:
    pattern = f"{group}_{event_kind}_event_*.png"
    pngs = sorted(event_dir.glob(pattern))
    by_key: Dict[str, Path] = {}
    if event_kind == "test":
        regex = re.compile(rf"^{re.escape(group)}_test_event_(\d+)_")
    else:
        regex = re.compile(rf"^{re.escape(group)}_paper_event_([^_]+)_")
    for path in pngs:
        match = regex.match(path.name)
        if match:
            by_key[str(match.group(1))] = path
    return by_key


def _fallback_event_png(event_dir: Path, group: str, event_kind: str, row: pd.Series) -> Path:
    if event_kind == "test":
        event_no = _safe_int(row["test_event_no"])
        return event_dir / f"{group}_test_event_{event_no:02d}_data.csv"
    event_id = str(row["paper_event_id"])
    return event_dir / f"{group}_paper_event_{event_id}_data.csv"


def _export_event_data(group: str, predictions: pd.DataFrame, plots_dir: Path, event_kind: str) -> int:
    event_dir = plots_dir / f"{group}_{event_kind}_events"
    index_path = event_dir / f"{group}_{event_kind}_events.csv"
    if not index_path.exists():
        return 0

    index_frame = pd.read_csv(index_path)
    key_col = "test_event_no" if event_kind == "test" else "paper_event_id"
    if key_col not in index_frame.columns:
        raise ValueError(f"{index_path} missing {key_col}")
    for required in ["start_row", "end_row_exclusive"]:
        if required not in index_frame.columns:
            raise ValueError(f"{index_path} missing {required}")

    png_by_key = _event_png_by_key(event_dir, group, event_kind)
    written = 0
    for _, row in index_frame.iterrows():
        start = _safe_int(row["start_row"])
        end = _safe_int(row["end_row_exclusive"])
        clipped_start = max(0, min(start, len(predictions)))
        clipped_end = max(clipped_start, min(end, len(predictions)))
        data = _slice_rows(predictions, start, end, clip_to_available=True)
        key_value = row[key_col]
        key = f"{int(key_value):02d}" if event_kind == "test" else str(key_value)
        png_path = png_by_key.get(key)
        if png_path is not None:
            out_path = png_path.with_name(f"{png_path.stem}_data.csv")
            plot_name = png_path.name
        else:
            out_path = _fallback_event_png(event_dir, group, event_kind, row)
            plot_name = out_path.name.replace("_data.csv", ".png")

        data.insert(0, "plot_name", plot_name)
        data.insert(1, "plot_type", f"{event_kind}_event")
        data.insert(2, "slice_was_clipped", bool(clipped_start != start or clipped_end != end))
        data.insert(3, "requested_start_row", start)
        data.insert(4, "requested_end_row_exclusive", end)
        for col in reversed(list(index_frame.columns)):
            data.insert(5, col, row[col])
        data.to_csv(out_path, index=False, encoding="utf-8-sig")
        written += 1
    return written


def export_plot_data(run_dir: Path, max_overview_points: int) -> pd.DataFrame:
    cache_dir = run_dir / "prediction_cache"
    plots_dir = run_dir / "prediction_plots"
    if not cache_dir.exists():
        raise FileNotFoundError(f"Prediction cache directory not found: {cache_dir}")
    if not plots_dir.exists():
        raise FileNotFoundError(f"Prediction plots directory not found: {plots_dir}")

    rows = []
    for cache_path in _prediction_cache_files(cache_dir):
        group = _group_from_cache_path(cache_path)
        predictions = _read_predictions(cache_path)
        overview_count = _export_overview_data(group, predictions, plots_dir, max_overview_points)
        top_count = _export_top_events_data(group, predictions, plots_dir)
        paper_count = _export_event_data(group, predictions, plots_dir, "paper")
        test_count = _export_event_data(group, predictions, plots_dir, "test")
        rows.append(
            {
                "group": group,
                "prediction_rows": len(predictions),
                "overview_data_files": overview_count,
                "top_event_detail_data_files": top_count,
                "paper_event_data_files": paper_count,
                "test_event_data_files": test_count,
                "total_data_files": overview_count + top_count + paper_count + test_count,
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(plots_dir / "prediction_plot_data_export_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--max-overview-points", type=int, default=6000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_plot_data(args.run_dir, args.max_overview_points)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
