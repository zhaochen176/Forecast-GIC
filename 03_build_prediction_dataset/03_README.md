# Stage 03: Features, segments and targets

```powershell
python -B 03_build_prediction_dataset/03_build_prediction_dataset.py
```

Requires the Stage 02 catalogues, validated monthly Wind observations and daily GIC minute files. `--workspace PATH` uses another local data root. `--force` rebuilds existing segments; `--available-only` permits preparing an incomplete local subset and prints its unavailable segment count. Complete formal experiments require every segment.

FA contains seven instantaneous inputs: `Btot, Bx_gsm, By_gsm, Bz_gsm, V, Np, Ma`. FB contains these plus proton dynamic pressure, southward electric-field proxy, Wind Epsilon, Newell coupling and `VBt_clock_proxy`; each of the 12 base features also has its trailing 30-minute mean, maximum and population standard deviation. The trailing southward-Bz integral adds the 49th feature. The exact order is `FEATURES` in this program and the locally written `data/feature_names.json`.

`VBt_clock_proxy = 1e-3 * V * hypot(By,Bz) * sin(theta/2)^2`, with `theta=atan2(By,Bz)`. This quantity is a solar-wind speed, transverse-field and clock-angle coupling proxy. Its definition and feature order are fixed in the program.

All rolling windows are causal. Continuous one-minute UTC grids retain source gaps, and complete-window statistics require all 30 minutes. The documented problematic GIC dates are excluded by the reader. Target labels use the maximum absolute GIC in `[s+h,s+h+30 min)` and thresholds `>=3,5,10,20 A`; no target-minute interpolation is used. Eligibility and horizon-specific labels are formed by Stage 04.

Outputs are ignored local `data/segments/block_01.npz` through `block_49.npz`, `continuous_2022.npz` and their JSON metadata. Arrays are `time` (UTC int64 nanoseconds), `x` (N by 49 float32), `gic`, `valid_native_samples`, and `in_block`. The preceding day supplies causal history; labels cannot cross a block's end. Keep matching NPZ and JSON files together. Feature-only prediction needs only `time` and `x` in the same order.
