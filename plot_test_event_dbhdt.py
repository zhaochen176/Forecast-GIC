"""Plot dBH/dt for all test events and export plotting data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RUN_DIR = Path("outputs/experiments/geomag_short_term_model3/paper_vkh_drivers_all/gic_vyk_abs_tplus1")
DEFAULT_DATA = Path("data/merged_2012_2022_processed.parquet")


def _load_source(data_path: Path, dbhdt_col: str) -> pd.DataFrame:
    cols = [dbhdt_col]
    optional = ["gic", "X", "Y", "Z", "H_pert"]
    try:
        df = pd.read_parquet(data_path, columns=list(dict.fromkeys(cols + optional)))
    except Exception:
        df = pd.read_parquet(data_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ("datetime", "time", "timestamp"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df = df.set_index(col)
                break
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Input data must have a DatetimeIndex or datetime column.")
    if dbhdt_col not in df.columns:
        raise KeyError(f"Missing dBH/dt column: {dbhdt_col}. Available columns include: {list(df.columns)[:30]}")
    return df.sort_index()


def _event_filename(row: pd.Series) -> str:
    event_id = int(row["paper_event_id"])
    start = pd.Timestamp(row["start"]).strftime("%Y%m%d_%H%M")
    end = pd.Timestamp(row["end"]).strftime("%Y%m%d_%H%M")
    event_type = str(row.get("event_type", "event")).replace("/", "_").replace(" ", "_")
    return f"test_event_{event_id:03d}_{event_type}_{start}_{end}"


def export_test_event_dbhdt(
    run_dir: Path,
    data_path: Path,
    dbhdt_col: str,
    include_gic: bool,
    dpi: int,
) -> pd.DataFrame:
    events_path = run_dir / "events_report.csv"
    if not events_path.exists():
        raise FileNotFoundError(f"Missing events_report.csv: {events_path}")
    events = pd.read_csv(events_path, parse_dates=["peak_time", "start", "end"])
    test_events = events[events["split"].astype(str).str.lower().eq("test")].copy()
    if test_events.empty:
        raise RuntimeError(f"No test events found in {events_path}")

    df = _load_source(data_path, dbhdt_col)
    out_dir = run_dir / "dbhdt_test_event_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for _, row in test_events.iterrows():
        start = pd.Timestamp(row["start"])
        end = pd.Timestamp(row["end"])
        peak = pd.Timestamp(row["peak_time"])
        sub = df.loc[(df.index >= start) & (df.index <= end)].copy()
        if sub.empty:
            summary_rows.append(
                {
                    "paper_event_id": int(row["paper_event_id"]),
                    "event_type": row.get("event_type", ""),
                    "start": start,
                    "end": end,
                    "n_points": 0,
                    "plot_path": "",
                    "data_path": "",
                    "status": "empty_window",
                }
            )
            continue

        plot_data = pd.DataFrame(
            {
                "datetime": sub.index,
                "paper_event_id": int(row["paper_event_id"]),
                "event_type": row.get("event_type", ""),
                "event_type_raw": row.get("event_type_raw", ""),
                "event_peak_time": peak,
                "minutes_from_peak": (sub.index - peak) / pd.Timedelta(minutes=1),
                "dBHdt": pd.to_numeric(sub[dbhdt_col], errors="coerce").to_numpy(dtype=float),
            }
        )
        plot_data["abs_dBHdt"] = np.abs(plot_data["dBHdt"].to_numpy(dtype=float))
        if include_gic and "gic" in sub.columns:
            plot_data["gic"] = pd.to_numeric(sub["gic"], errors="coerce").to_numpy(dtype=float)

        stem = _event_filename(row)
        data_path_out = out_dir / f"{stem}_dbhdt_data.csv"
        plot_path_out = out_dir / f"{stem}_dbhdt.png"
        plot_data.to_csv(data_path_out, index=False, encoding="utf-8-sig")

        fig, ax = plt.subplots(figsize=(14, 4.8), dpi=dpi)
        ax.plot(plot_data["datetime"], plot_data["dBHdt"], color="#1f77b4", lw=1.0, label=f"dBH/dt ({dbhdt_col})")
        ax.plot(plot_data["datetime"], plot_data["abs_dBHdt"], color="#d62728", lw=0.9, alpha=0.75, label="|dBH/dt|")
        ax.axvline(peak, color="black", ls="--", lw=1.0, alpha=0.65, label="paper GIC peak")
        ax.axhline(0.0, color="gray", lw=0.8, alpha=0.5)
        title = (
            f"Test event {int(row['paper_event_id'])} | {row.get('event_type', '')} | "
            f"{start:%Y-%m-%d %H:%M} ~ {end:%Y-%m-%d %H:%M}"
        )
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("dBH/dt")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(plot_path_out, bbox_inches="tight")
        plt.close(fig)

        summary_rows.append(
            {
                "paper_event_id": int(row["paper_event_id"]),
                "event_type": row.get("event_type", ""),
                "event_type_raw": row.get("event_type_raw", ""),
                "peak_time": peak,
                "start": start,
                "end": end,
                "n_points": int(len(plot_data)),
                "dBHdt_max": float(np.nanmax(plot_data["dBHdt"])),
                "dBHdt_min": float(np.nanmin(plot_data["dBHdt"])),
                "abs_dBHdt_max": float(np.nanmax(plot_data["abs_dBHdt"])),
                "plot_path": str(plot_path_out),
                "data_path": str(data_path_out),
                "status": "ok",
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "test_event_dbhdt_plot_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--dbhdt-col", default="dH_pert_dt")
    parser.add_argument("--include-gic", action="store_true")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_test_event_dbhdt(
        run_dir=args.run_dir,
        data_path=args.data,
        dbhdt_col=args.dbhdt_col,
        include_gic=args.include_gic,
        dpi=args.dpi,
    )
    print(summary[["paper_event_id", "event_type", "n_points", "abs_dBHdt_max", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
