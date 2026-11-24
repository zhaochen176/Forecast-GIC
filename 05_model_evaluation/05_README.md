# Stage 05: Evaluation, interpretation and figures

```powershell
python -B 05_model_evaluation/05_model_evaluation.py --mode evaluate
python -B 05_model_evaluation/05_model_evaluation.py --mode explain
python -B 05_model_evaluation/05_model_evaluation.py --mode figures
```

`--mode all` runs these in order. `--workspace PATH` must match the preceding stages. Evaluation first requires complete search, selected-comparison and inference outputs, including the logistic baseline. Missing inputs are reported rather than replaced with published scores. This stage does not retrain models.

Evaluation calculates event and continuous-year scores, uncalibrated/calibrated reliability, primary and secondary decisions, paired block/day bootstrap intervals (2000 replicates, seed 617), repeated seeds, common-sample horizon/context comparisons, causal warning rules, catalogue matching and coverage/eligibility. Continuous unavailable periods remain gaps. Tables are computed from local predictions and observations, never read from article result tables.

Interpretation uses GradientExplainer with 64 training background sequences, up to 64 event-test sequences from each of five exclusive threshold strata, 200 draws and seed 628. Actual stratum proportions weight the grouped absolute contributions over time and context streams. Each output is a calibrated, ordered threshold probability. A gradient cache is reused only if model hash, sample indices, weights, native logits and baseline agree. MI (three neighbours) and Spearman associations use training data only. Explanations are associations, not causal effects. Full SHAP execution can be computationally expensive.

Figure code corresponds to the final submission: 1 block chronology/durations and exceedance; 2 low-threshold ROC/PR; 3 the whole 2022 year, issue-time probabilities, targets, both cutoffs and missing gaps; 4 association/model comparisons including retrained ablation; 5 horizon/context sensitivity; 6 four-output SHAP; 7 event reliability; S1 high-threshold ROC/PR; S2 continuous reliability. Figures require both evaluation and explanation outputs.

`--mode check` independently verifies causal alignment, gap preservation, future targets, ordered projection, warning causality, matching and standard metrics. Add `--with-catalogue` to validate local input counts (92 catalogue events/71 retained/49 blocks).

All generated CSV, JSON, NPZ, PNG and PDF products go to ignored local `evaluation/` or `figures/`. They are not part of the source release.
