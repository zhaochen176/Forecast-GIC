# Stage 03: Build the Prediction Dataset

## Purpose

`03_build_prediction_dataset.py` creates the canonical modelling timeline from the Stage 02 one-minute tables and the supporting-information CME/CIR event catalogue. It aligns solar-wind predictors with the GIC response, computes rolling statistics and physical coupling features, and represents the propagation delay explicitly.

For each prediction time, the input sequence ends at `t - L`, where `L` is one of the candidate lags 30, 45, 60, or 90 minutes. The target is whether the maximum absolute GIC in `(t, t + 30 minutes]` exceeds 3, 5, 10, or 20 A. The script records target validity, excludes windows crossing unacceptable data gaps, and assigns complete event blocks to chronological train, validation, and test partitions.

## Usage and required inputs

Place the exact supporting-information document at the path described in `docs/DATA.md`, then run:

```powershell
python 03_build_prediction_dataset\03_build_prediction_dataset.py --split-ratios 0.6 0.2 0.2 --output-dir data\prediction_dataset_6020
```

The command reads `data/processed/GIC_1min_2012_2022.csv` and `data/processed/Wind_L1_1min_2012_2022.csv`. It writes the event timeline, feature metadata, split manifest, alignment reports, and optional yearly plots below the selected output directory. This is derived data, not source code, and must remain local.

## Handoff to training

Stage 04 consumes the generated `prediction_timeline.csv` and associated feature files. Keep the output directory intact for a training run; rebuilding it with different preprocessing or split arguments creates a different experiment.