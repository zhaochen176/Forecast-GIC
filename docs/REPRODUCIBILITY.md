# Reproducibility Guide

Run the five scripts from the repository root in numeric order. Step 2 produces an explicit UTC one-minute grid and records missingness rather than interpolating source gaps. Step 3 creates the canonical event-window timeline, 3/5/10/20 A future-exceedance targets, validity masks, and chronological whole-event splits.

Step 4 uses expanding chronological forward validation on development events. Held-out test events are not used for fitting, cutoff selection, calibration, or model selection. The default random seed is `42`.

The built-in configuration grid contains lags of 30, 45, 60, and 90 minutes crossed with look-back windows of 30, 60, and 120 minutes:

```powershell
python 04_train_cnn_bilstm.py --dataset-dir data/prediction_dataset_6020 --all-configurations --num-workers 0
```

Use `--skip-completed` to resume a grid run and `--aggregate-only` to rebuild a comparison from completed runs. On Windows, start with `--num-workers 0` if PyTorch DataLoader workers fail.

Before publishing changes:

```powershell
python -m compileall -q 01_read_and_save_data.py 02_data_preprocessing.py 03_build_prediction_dataset.py 04_train_cnn_bilstm.py 05_model_evaluation.py
python 01_read_and_save_data.py --help
python 02_data_preprocessing.py --help
python 03_build_prediction_dataset.py --help
python 04_train_cnn_bilstm.py --help
python 05_model_evaluation.py --help
git status --ignored
```