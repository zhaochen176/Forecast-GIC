# GIC Advance Prediction for VKH Station

Code for the paper experiment on lead-time forecasting of intense
geomagnetically induced current (GIC) events at the Vykhodnoy (VKH) 220 kV
substation.

The workflow predicts whether the future 30-minute peak of `|GIC|` exceeds
risk thresholds using L1 solar-wind observations, physics-driven propagation
lag features, CNN-BiLSTM models, event-level metrics, and SHAP-style feature
attribution.

## Repository Structure

```text
.
├── src/                                # shared data, feature, model, training, evaluation utilities
├── data/                               # local data placement only; large data are not tracked
├── outputs/                            # generated reports, figures, caches, and checkpoints
├── nonlinear_feature_gic_analysis.py   # Section 2.3 nonlinear dependence analysis
├── solar_wind_lagged_ablation.py       # main lead-time solar-wind lag experiment
├── threshold_quantile_ablation.py      # model backends, threshold and quantile training utilities
├── evaluate_contiguous_event_metrics.py# event-level alarm matching metrics
├── plot_event_threshold_tradeoff.py    # probability-threshold tradeoff plots
├── export_final_prediction_shap.py     # gradient-SHAP style feature attribution
└── export_prediction_plot_data.py      # export plot-ready prediction series
```

## Environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For GPU training, install a PyTorch build that matches your CUDA version before
running the scripts.

## Data Preparation

This repository does not include experimental results, model weights, caches,
or large processed data files.

Put the processed 1-minute data file here:

```text
data/merged_2012_2022_processed.parquet
```

See `data/README.md` for the public source links and expected local placement.

## Main Paper Workflow

Run the nonlinear dependence analysis:

```bash
python nonlinear_feature_gic_analysis.py --run-presets --scope event-type --event-types CME CIR --horizon 30
```

Run the main VKH CME/CIR lead-time prediction experiment:

```bash
python solar_wind_lagged_ablation.py ^
  --scope paper-vkh-drivers ^
  --event-types CME CIR ^
  --target gic_vyk_abs ^
  --horizon 30 ^
  --lags 30 45 60 90 ^
  --rolling-windows 15 30 60 ^
  --thresholds 3 5 10 20 ^
  --backend cnn_bilstm_attention ^
  --source-window-only
```

Evaluate contiguous event-level alarm performance from the generated prediction
caches:

```bash
python evaluate_contiguous_event_metrics.py ^
  --experiment-dir outputs/experiments/paper_vkh_drivers_all/gic_vyk_abs_H30 ^
  --thresholds 3 5 10 20 ^
  --probability-threshold 0.5
```

Export SHAP-style interpretation for the final model:

```bash
python export_final_prediction_shap.py ^
  --run-dir outputs/experiments/paper_vkh_drivers_all/gic_vyk_abs_H30 ^
  --data data/merged_2012_2022_processed.parquet ^
  --groups solar_lag45_source_window_plus_time
```

All generated files are written under `outputs/` and are excluded from version
control.

## Data Availability Statement

Solar wind and interplanetary magnetic field data are available from NASA
CDAWeb: https://cdaweb.gsfc.nasa.gov/. VKH GIC observations are available from
the public Vykhodnoy 220 kV substation database: http://gic.en51.ru/. The code
for reproducing the lead-time GIC prediction workflow is provided in this
repository.
