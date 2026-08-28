# Stage 05: Evaluate a Completed Run

## Purpose

`05_model_evaluation.py` is the final reporting stage. It reads the `test_probabilities.csv` file produced by Stage 04 and evaluates the four operational threshold probabilities without retraining the network. For a chosen decision cutoff it calculates ROC-AUC, PR-AUC, probability of detection (POD), probability of false detection (POFD), false alarm ratio (FAR), critical success index (CSI), F1, Bias, and the underlying confusion counts.

The script also creates ROC and precision-recall curves, a training-history plot when history is available, and a compact probability time series containing identifiers and the four model probabilities. These diagnostics make it possible to inspect discrimination and operational trade-offs for each threshold.

## Usage

Run from the repository root and point `--run-dir` to one completed Stage 04 configuration:

```powershell
python 05_model_evaluation\05_model_evaluation.py --run-dir outputs\cnn_bilstm\context_onset_v1\L45_W60 --threshold 0.5
```

Results are written to `final_evaluation/` inside that run directory. They are local reports and must not be added to Git. The cutoff supplied here is a reporting choice; use the validation-selected operating point recorded by Stage 04 when reproducing the paper protocol.