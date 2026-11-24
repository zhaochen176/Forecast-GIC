"""Stage 03: construct the revision's 49 features, segments and future targets.

Target: maximum absolute GIC in [s+h, s+h+30 min); no missing-value imputation.
FA contains seven upstream quantities; FB contains 49 derived input features.
"""
from pathlib import Path
import argparse, importlib.util, json, sys
import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

def load_stage(number):
    """Load a numbered sibling program without relying on the current directory."""
    paths = {
        2: "02_data_preprocessing/02_data_preprocessing.py",
        3: "03_build_prediction_dataset/03_build_prediction_dataset.py",
        4: "04_train_cnn_bilstm/04_train_cnn_bilstm.py",
        5: "05_model_evaluation/05_model_evaluation.py",
    }
    name = f"_forecast_gic_stage_{number:02d}"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REPOSITORY_ROOT / paths[number])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]

preprocessing = load_stage(2)
ROOT = REPOSITORY_ROOT
DATA = ROOT / "data"
FA = ["Btot", "Bx_gsm", "By_gsm", "Bz_gsm", "V", "Np", "Ma"]
BASE = FA + ["Pdyn_proton", "Ey_south", "Epsilon", "Newell", "VBt_clock_proxy"]
FEATURES = BASE + [f"{name}_{stat}_30min" for name in BASE for stat in ("mean", "max", "std")] + ["Bz_south_integral_30min"]
THRESHOLDS = np.array([3, 5, 10, 20])

def configure_workspace(workspace=None):
    global ROOT, DATA
    ROOT = Path(workspace).resolve() if workspace is not None else REPOSITORY_ROOT
    DATA = ROOT / "data"
    preprocessing.configure_workspace(ROOT)

monthly_wind = preprocessing.monthly_wind
align_prior = preprocessing.align_prior
gic_frame = preprocessing.gic_frame

def solar_frame(issue_times):
    months = sorted({(t.year,t.month) for t in issue_times})
    # One source day before the segment is already included in issue_times.
    mfi = pd.concat([monthly_wind(y,m,"MFI") for y,m in months])
    swe = pd.concat([monthly_wind(y,m,"SWE") for y,m in months])
    mfi = mfi.loc[~mfi.index.duplicated()]
    swe = swe.loc[~swe.index.duplicated()]
    # SWE is an accumulated spectrum, not an instantaneous sample. The ASCII
    # service does not expose acquisition-start/end metadata. Impose a fixed
    # 120 s availability guard beyond its reported time before the as-of join.
    # This covers the nominal ~92 s accumulation; it is not telemetry latency.
    swe = swe.copy()
    swe.index = swe.index + pd.Timedelta(seconds=120)
    a, age_a = align_prior(mfi,issue_times,60)
    b, age_b = align_prior(swe,issue_times,120)
    a = a.rename(columns={"B":"Btot","Bx":"Bx_gsm","By":"By_gsm","Bz":"Bz_gsm"})
    x = pd.concat([a, b], axis=1)
    for name in ["Btot","V","Np","Ma"]:
        x.loc[x[name] <= 0, name] = np.nan
    x.loc[x.Epsilon < 0,"Epsilon"] = np.nan
    bt = np.hypot(x.By_gsm, x.Bz_gsm)
    theta = np.arctan2(x.By_gsm, x.Bz_gsm)
    south = (-x.Bz_gsm).clip(lower=0)
    x["Pdyn_proton"] = 1.67262192369e-6 * x.Np * x.V**2
    x["Ey_south"] = 1e-3 * x.V * south
    x["Newell"] = x.V**(4/3) * bt**(2/3) * np.abs(np.sin(theta/2))**(8/3)
    # Speed-transverse-field coupling weighted by the IMF clock angle.
    x["VBt_clock_proxy"] = 1e-3 * x.V * bt * np.sin(theta/2)**2
    x = x[BASE].copy()
    for name in BASE:
        r = x[name].rolling(30,min_periods=30)
        x[f"{name}_mean_30min"] = r.mean()
        x[f"{name}_max_30min"] = r.max()
        x[f"{name}_std_30min"] = r.std(ddof=0)
    # Integral in nT min, assuming the stated one-minute analysis grid.
    x["Bz_south_integral_30min"] = south.rolling(30,min_periods=30).sum()
    return x[FEATURES], dict(mfi_max_age_seconds=float(age_a.max()),
                             swe_max_age_seconds=float(age_b.max()+120),
                             swe_min_age_seconds=float(age_b.min()+120),
                             swe_availability_guard_seconds=120,
                             mfi_missing_minutes=int(age_a.isna().sum()),
                             swe_missing_minutes=int(age_b.isna().sum()))


def future_peak(gic_abs, horizon=0, width=30):
    """At s use bins s+h,...,s+h+width-1. Never impute a target."""
    series = pd.Series(gic_abs)
    return series.rolling(width,min_periods=width).max().shift(-(width-1+horizon)).to_numpy(np.float32)


def prepare_segment(name, start, end, split, block, force=False):
    dest = DATA / "segments" / f"{name}.npz"
    if dest.exists() and dest.with_suffix(".json").exists() and not force:
        return
    begin = pd.Timestamp(start).floor("D") - pd.Timedelta(days=1)
    finish = pd.Timestamp(end).ceil("D")
    # A continuous UTC grid prevents rolling across removed time gaps.
    index = pd.date_range(begin,finish,freq="min",inclusive="left")
    for day in pd.date_range(begin,finish,freq="D",inclusive="left"):
        path=DATA/"GIC_minute"/f"{day:%Y%m%d}.parquet"
        if not path.exists() or not path.with_suffix(".json").exists():
            raise FileNotFoundError(f"Not yet recovered: {path}")
    x, audit = solar_frame(index)
    gic = gic_frame(index)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    in_block = (index >= start) & (index < end)
    raw_x = x.to_numpy(np.float32)
    raw_gic = gic.GIC.to_numpy(np.float32)
    dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest,time=index.to_numpy("datetime64[ns]").astype("int64"),
                        x=raw_x,gic=raw_gic,valid_native_samples=gic.valid_native_samples.to_numpy(np.int16),
                        in_block=in_block)
    summary = dict(name=name,start=str(start),end=str(end),split=split,block=block,
                   total_minutes=int(in_block.sum()),valid_gic_minutes=int(np.isfinite(raw_gic[in_block]).sum()),
                   complete_feature_minutes=int(np.isfinite(raw_x[in_block]).all(axis=1).sum()),**audit)
    dest.with_suffix(".json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, help="Local data/run root; default is the repository root.")
    parser.add_argument("--available-only", action="store_true", help="Report unavailable sources and prepare the remaining segments.")
    parser.add_argument("--force", action="store_true", help="Rebuild prepared segments instead of reusing them.")
    args = parser.parse_args()
    configure_workspace(args.workspace)
    blocks = pd.read_csv(DATA / "supplied_blocks.csv")
    unavailable = []
    for block in blocks.itertuples():
        try:
            prepare_segment(f"block_{block.block:02d}", block.start, block.end, block.split, int(block.block), args.force)
        except FileNotFoundError as exc:
            unavailable.append(str(exc))
            if not args.available_only:
                raise
    try:
        prepare_segment("continuous_2022", "2022-01-01", "2023-01-01", "continuous", 100, args.force)
    except FileNotFoundError as exc:
        unavailable.append(str(exc))
        if not args.available_only:
            raise
    (DATA / "feature_names.json").write_text(json.dumps(FEATURES, indent=2), encoding="utf-8")
    print(f"Not yet available: {len(unavailable)}", flush=True)


if __name__ == "__main__":
    main()
