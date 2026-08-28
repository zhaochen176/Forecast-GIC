# Stage 02: Data Preprocessing

## Purpose

`02_data_preprocessing.py` converts heterogeneous source records into the clean, aligned feature series used by dataset construction. It parses VKH timestamps and values, identifies provider missing-value sentinels and physically invalid amplitudes, reads Wind MFI and SWE ASCII responses, and aligns all variables to an explicit one-minute UTC grid from 2012-01-01 through 2022-12-31.

Missing observations are retained as missing flags. The script does not silently interpolate long source gaps, which is important because later target validity masks must distinguish an observed quiet interval from an unavailable interval. The preprocessing stage also writes time-coverage and missingness audits and can produce selected-event diagnostics for quality control.

## Usage

Run from the repository root:

```powershell
python 02_data_preprocessing\02_data_preprocessing.py --force-rebuild
```

Use `--audit-only` when the two one-minute tables already exist and only the combined missingness audit should be regenerated. The script expects Stage 01 files under `data/raw/` and creates its generated tables and audit products below `data/processed/`. Those products are required locally by Stage 03 but are intentionally ignored by Git.

## Quality checks

Review the generated missingness summaries and time-audit files before building labels. A preprocessing run should preserve the common UTC index, retain the `missing_flag` columns, and report any periods that cannot support a valid prediction. Do not commit the resulting CSV or PNG files.