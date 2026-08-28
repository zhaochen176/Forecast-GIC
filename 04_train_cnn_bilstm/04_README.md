# Stage 04: Train the CNN-BiLSTM Model

## Purpose

`04_train_cnn_bilstm.py` is the main modelling program. It loads the event-window dataset from Stage 03 and learns four calibrated-looking exceedance scores for the 3, 5, 10, and 20 A thresholds. The network first extracts local and multi-scale patterns with residual dilated 1-D CNN blocks, models longer temporal dependencies with a stacked bidirectional LSTM, and applies temporal attention before gated CNN/LSTM fusion.

Training uses a peak-weighted binary cross-entropy objective, an auxiliary onset objective, and a soft penalty for violations of the expected threshold ordering. Feature selection and scaling are fitted within each training fold. Expanding chronological forward validation selects model states and operating points while fixed test events are kept out of fitting, calibration, and selection.

## Usage

A single configuration can be trained with:

```powershell
python 04_train_cnn_bilstm\04_train_cnn_bilstm.py --dataset-dir data\prediction_dataset_6020 --lag 45 --window 60 --num-workers 0
```

The supported lag values are 30, 45, 60, and 90 minutes; supported context windows are 30, 60, and 120 minutes. Add `--all-configurations` to run all twelve combinations, `--skip-completed` to resume an interrupted grid, or `--aggregate-only` to rebuild the comparison from existing runs.

## Outputs and handoff

Runs are written below `outputs/cnn_bilstm/` by default and contain fold checkpoints, scalers, histories, probabilities, selected decision thresholds, event metrics, and run metadata. These are generated experiment artefacts and are excluded from this repository. Pass one completed run directory to Stage 05 for independent test metrics and plots.