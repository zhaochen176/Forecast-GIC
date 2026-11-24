# Probabilistic Forecasting of Geomagnetically Induced Current Threshold Exceedance from L1 Solar Wind Observations

Source code for station-specific forecasts of 30-minute GIC exceedances at 3, 5, 10 and 20 A. The software consists of five Python programs covering data acquisition, preprocessing, feature construction, model training and evaluation.

| Program | Responsibility |
| --- | --- |
| [01](01_read_and_save_data/01_read_and_save_data.py) | Native GIC/Wind acquisition |
| [02](02_data_preprocessing/02_data_preprocessing.py) | Input catalogue, source validation/recovery and native GIC aggregation |
| [03](03_build_prediction_dataset/03_build_prediction_dataset.py) | Causal 49-feature construction, segments and future labels |
| [04](04_train_cnn_bilstm/04_train_cnn_bilstm.py) | Models, training, selection, calibration, baseline and prediction |
| [05](05_model_evaluation/05_model_evaluation.py) | Scores, bootstrap, warnings, coverage, interpretation and figures |

The package includes source and documentation only. Observations, article documents, input catalogues, arrays, model weights, tables, figures and logs must be supplied or generated locally. Retain all five stage directories together: later programs load definitions from their numbered siblings.

## Environment

Use Python 3.12 or 3.13 in an isolated environment. Install a suitable CPU/CUDA PyTorch build and then:

```powershell
python -m pip install -r requirements.txt
```

Verification used Python 3.13.5, NumPy 2.1.3, pandas 2.2.3, SciPy 1.15.3, scikit-learn 1.7.1, PyTorch 2.7.0+cpu, Matplotlib 3.10.0, PyArrow 19.0.0 and SHAP 0.52.0. Archived neural-network training runs record PyTorch 2.6.0 with CUDA 12.4. Online recovery in Stage 02 also requires `curl` (`curl.exe` on Windows).

## Run the pipeline

Run from the repository root. First make the revised Supporting Information available locally. Only its input event/block tables S2/S3 are read; its reported scores are never used to fit or evaluate a model. Use an absolute path below in place of the example path.

```powershell
python -B 01_read_and_save_data/01_read_and_save_data.py --download
python -B 02_data_preprocessing/02_data_preprocessing.py --catalogue "D:/local/Supporting Information.docx" --kind plan
python -B 02_data_preprocessing/02_data_preprocessing.py --catalogue "D:/local/Supporting Information.docx" --kind all --offline
python -B 03_build_prediction_dataset/03_build_prediction_dataset.py
python -B 04_train_cnn_bilstm/04_train_cnn_bilstm.py --phase search
python -B 04_train_cnn_bilstm/04_train_cnn_bilstm.py --phase selected
python -B 04_train_cnn_bilstm/04_train_cnn_bilstm.py --phase infer
python -B 05_model_evaluation/05_model_evaluation.py --mode all
```

Stage 01 can be skipped when native observations are already available. Stage 02 validates/reuses its files; remove `--offline` to recover missing revision sources from the providers. Alternatively, supply both extracted input catalogue CSVs as described in [Stage 02](02_data_preprocessing/02_README.md). A source failure terminates Stage 02 with a nonzero exit code. Resolve it before Stage 03.

Stages 02–05 accept `--workspace "D:/local/gic_work"` to put local inputs and outputs elsewhere. Pass the same workspace to every stage. The unchanged Stage 01 always uses the repository's own `data/` directory; its sources must be placed under the chosen workspace before Stage 02 if using another workspace.

Independent boundary checks and a short training check:

```powershell
python -B 05_model_evaluation/05_model_evaluation.py --mode check
python -B 04_train_cnn_bilstm/04_train_cnn_bilstm.py --phase smoke --device cpu
```

The smoke phase needs at least one prepared training segment containing all four positive classes. Its two-epoch check deliberately reuses a small training subset for internal validation/calibration and is unsuitable for scientific reporting. Full experiments use the disjoint partitions below. See the numbered READMEs for inputs, command options and prediction-only use.

## Experimental contract

All times are UTC. Inputs are upstream observations available strictly before issue time; historical GIC is used only for training/evaluation targets. The target is the maximum absolute native-minute GIC in `[s+h, s+h+30 min)`, thresholded using `>=`. Missing inputs or target minutes exclude a forecast; gaps are not interpolated.

Blocks 1–30 train the models and preprocessing; 31–35 select delayed context using common eligible issue times; 36–40 calibrate probabilities; 41–49 form the event test. Calendar year 2022 is the complementary continuous evaluation. Training stride is five minutes and evaluation stride is one minute. Context delay and forecast horizon vary separately. Primary decisions use calibrated, ordered probabilities at a fixed cutoff of 0.5; calibration-selected CSI cutoffs are secondary.

The architecture, Adam settings, early stopping, four-output loss and experiment grid are implemented in Stage 04. `experiment_protocol()` records these settings in each new neural run with a SHA-256 fingerprint. The embedded protocol records the executable settings used by each new run.

Stage 05 generates Figures 1–7 and S1–S2 with the final submission's numbering, including the full 2022 forecast example. SHAP targets each calibrated, ordered output; attribution targets and association methods are described in the Stage 05 README.

## Software archive and citation

This software can be deposited directly in Zenodo as a software record. Cite the published record's specific version DOI and include the software citation in the associated article's reference list. The observation providers and local-input requirements are described in Stage 02; observations and trained model weights are not bundled with this source archive.

MIT source license: [LICENSE](LICENSE). Observations and other external materials retain their providers' terms: [LICENSES.md](LICENSES.md).
