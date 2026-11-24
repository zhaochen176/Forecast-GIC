# Stage 04: Training, calibration and prediction

The actual CNN–BiLSTM implementation, LSTM comparison, logistic baseline and inference are all in this program. It loads only the numbered sibling stages. Prepared inputs are required; trained weights are not distributed.

Run the full experiment in order:

```powershell
python -B 04_train_cnn_bilstm/04_train_cnn_bilstm.py --phase search
python -B 04_train_cnn_bilstm/04_train_cnn_bilstm.py --phase selected
python -B 04_train_cnn_bilstm/04_train_cnn_bilstm.py --phase infer
```

`search` fits nine FB settings: horizons 0/30/45/60/90 with zero context delay, plus delays 30/45/60/90 with zero horizon. Delay selection uses mean unweighted BCE on the common eligible selection samples; a smaller delay breaks a tie. `selected` trains FA, LSTM and the retrained southward-integral ablation at the selected context, then FB repeats with seeds 7 and 2026. `infer` evaluates the 14 neural fits and fits/evaluates the independent four-output logistic baseline. Selection uses no test or continuous-year scores.

The two context streams each contain 30 minutes; delta zero duplicates the latest stream. Signed log1p normalization and its means/standard deviations use training blocks only. The CNN has 32 channels and dilations 1/4; the recurrent branch uses two bidirectional LSTM layers with 64 hidden units, temporal attention and a learned fusion gate. Adam uses learning rate/weight decay 0.0005, batch size 1024, at most 30 epochs, gradient norm clipping at 1, early stopping patience 8 and ReduceLROnPlateau factor 0.5/patience 2/minimum 1e-5. The loss is positive-weighted BCE plus a 0.05 probability-order penalty. There is no onset head.

Blocks 36–40 fit positive-slope sigmoid calibrators; exact decreasing isotonic projection enforces ordered threshold probabilities. Cutoff 0.5 is primary; calibration-CSI cutoffs are secondary. The logistic baseline uses trailing southward-IMF integral values with fixed C=1 and the same partitions/eligibility/calibration.

Use `--workspace PATH` consistently across stages and `--device auto|cpu|cuda` as appropriate. A two-epoch software check is `--phase smoke --device cpu`; it writes `smoke-runs/` and cannot replace the formal experiment. Formal outputs go to ignored `runs/` and `selection.json`. Completed runs are reused, so use a fresh workspace after changing source settings or input data. `experiment_protocol()` and its hash are saved in new neural-run metadata.

## Upstream-only deployment

```powershell
python -B 04_train_cnn_bilstm/04_train_cnn_bilstm.py --phase predict --checkpoint runs/fb_h0_d90_s42/best.pt --features data/upstream_only.npz --output predictions.csv
```

Replace the example run name with the selected run. Keep its `development_complete.json` next to `best.pt`. The input NPZ needs only increasing UTC nanosecond `time` and N-by-49 float32 `x`, in Stage 03 feature order; it must already use Stage 03 causal alignment/feature construction. No GIC or future labels are read. The output contains issue time, exclusive-end target interval, availability and four calibrated ordered probabilities. Incomplete or interrupted histories remain unavailable. Load only trusted local checkpoints.
