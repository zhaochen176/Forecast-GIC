# Probabilistic Forecasting of Geomagnetically Induced Current Threshold Exceedance

## Purpose of This Repository

This repository provides the complete, reproducible source code for the study **“Probabilistic Forecasting of Geomagnetically Induced Current Threshold Exceedance from L1 Solar Wind Observations.”** The project develops a probabilistic early-warning method for geomagnetically induced current (GIC) at the Vykhodnoy (VKH) 330 kV substation. Given measurements observed at the L1 solar-wind monitor, the method estimates the probability that the maximum absolute GIC during the following 30 minutes will exceed 3, 5, 10, or 20 amperes.

The scientific workflow joins two time series: high-frequency VKH GIC observations and Wind spacecraft L1 magnetic-field and plasma observations. The source records are cleaned and converted to a common one-minute UTC timeline. Solar-wind features and physically motivated coupling functions are then calculated over several look-back windows. The resulting event-window sequences are passed to a CNN-BiLSTM model. Dilated residual one-dimensional convolution blocks learn short-term patterns, a bidirectional LSTM models temporal context, temporal attention highlights informative minutes, and gated fusion combines the learned representations. The model produces four exceedance probabilities and is trained with peak-event weighting, an auxiliary onset task, and a soft monotonicity penalty so that higher thresholds do not receive implausibly larger probabilities.

The evaluation protocol is chronological and event based. Candidate propagation lags are 30, 45, 60, and 90 minutes. Whole CME/CIR event blocks are assigned to development and test periods without mixing samples from the same event across splits. Expanding forward validation is used for model selection and operating-point calibration; the final test events remain isolated until the final evaluation step.

Only source code, documentation, and small example input files are included. Raw observations, the supporting-information event catalogue, processed tables, labels, checkpoints, plots, and reports are generated locally and are excluded from version control.

## Running the Pipeline

Run the five stages from the repository root and keep this order. Stage 01 downloads or reads the original observations. Stage 02 cleans and aligns them. Stage 03 builds the labelled prediction timeline. Stage 04 trains the model. Stage 05 evaluates one completed training run.

Install Python 3.10 or newer and the packages in `requirements.txt`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The standard command sequence is:

```powershell
python 01_read_and_save_data\01_read_and_save_data.py --download
python 02_data_preprocessing\02_data_preprocessing.py --force-rebuild
python 03_build_prediction_dataset\03_build_prediction_dataset.py --split-ratios 0.6 0.2 0.2 --output-dir data\prediction_dataset_6020
python 04_train_cnn_bilstm\04_train_cnn_bilstm.py --dataset-dir data\prediction_dataset_6020 --lag 30 --window 30 --num-workers 0
python 05_model_evaluation\05_model_evaluation.py --run-dir outputs\cnn_bilstm\
```

Use `--help` on any stage to see all options. Stage 04 can train all twelve lag/window configurations with `--all-configurations`. On Windows, `--num-workers 0` is the most portable setting.

## Stage Documentation

The detailed purpose, inputs, processing, outputs, and command examples for each program are documented in `01_read_and_save_data/01_README.md`, `02_data_preprocessing/02_README.md`, `03_build_prediction_dataset/03_README.md`, `04_train_cnn_bilstm/04_README.md`, and `05_model_evaluation/05_README.md`.

## Data and Reproducibility

Data access requirements and provider links are described in [docs/DATA.md](docs/DATA.md). The chronological split policy, feature windows, target definition, and validation procedure are described in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). The `data example/` directory contains a small format example only.

The original VKH GIC records, Wind observations, event catalogue, and any derived products remain subject to their providers' licences. Cite the associated manuscript and this software using `CITATION.cff`. Source code is released under the MIT License.

## Repository Structure

The five numbered directories contain the executable research stages. `data example/` documents the expected raw-file layout with small examples. `docs/` contains data-access and reproducibility notes. The root configuration files define dependencies, licensing, contribution rules, and automated interface checks.
