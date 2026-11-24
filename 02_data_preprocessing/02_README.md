# Stage 02: Input validation and native preprocessing

This program contains source recovery, revision-catalogue extraction, native-minute GIC aggregation and the causal readers used by Stage 03. Stage 01 is unchanged.

From the repository root:

```powershell
python -B 02_data_preprocessing/02_data_preprocessing.py --kind plan --catalogue "D:/local/Supporting Information.docx"
python -B 02_data_preprocessing/02_data_preprocessing.py --kind all --catalogue "D:/local/Supporting Information.docx" --offline
```

`plan` extracts/checks the input catalogues and reports the required observations without downloading them. Choose `gic`, `wind` or `all` to process sources. `--offline` prohibits recovery requests; omit it to fetch missing observations. Wind requests are serial to avoid the legacy CGI mixing concurrent responses. Inspect local `data/recovery_*.jsonl` if the command exits with a source failure. `--workspace PATH` changes the data root. `--catalogue PATH` selects a local DOCX; the default is `data/external/supporting_information.docx`.

## Input catalogue

The revised SI must contain 92 catalogue events, 71 retained events and 49 blocks. Only event and interval definitions are read from Tables S2/S3. Alternatively provide both local CSVs:

* `data/supplied_events.csv`: `event_id,peak_time,gic_peak_text,driver,retained`. `retained` is a Boolean.
* `data/supplied_blocks.csv`: `split,block,event_ids,start,end`. Use `train` for blocks 1–30, `val` for 31–40 and `test` for 41–49. Quote comma-separated event IDs. UTC intervals are `[start,end)`.

Stage 04 separates `val` into selection blocks 31–35 and calibration blocks 36–40. Preserve the submitted interval definitions and numbering. No catalogue or submitted document is bundled.

## Observations and local products

GIC provider: `http://gic.en51.ru/data/vkh_gic/`. Stage 01 native files are reused from `data/raw/GIC/YYYY/YYYY-MM/YYYYMMDD.txt`; Stage 02 can also read `data/raw/GIC/YYYYMMDD.txt` or `.txt.gz`. Each minute retains the signed valid native sample having the largest absolute value and its native valid-sample count. Sentinels and out-of-range values are masked. A confirmed HTTP 404 creates an explicitly missing day; other source failures remain errors. Products are local Parquet and provenance JSON files in `data/GIC_minute/`.

Wind provider: `https://wind.nasa.gov/mfi_swe_display_results.php`. Stage 01 monthly files in `data/raw/Wind_L1/MFI/` and `SWE/` are validated for product, year, month and columns before reuse. The validated monthly products and source hashes are stored under `data/raw/Wind/`. Incomplete daily fallbacks are not silently joined into a complete monthly response.

Recovery covers the event blocks, calendar year 2022 and preceding causal-history days. Readers preserve missing source values. MFI is aligned strictly earlier with age at most 60 seconds; SWE receives a 120-second availability guard and is then aligned strictly earlier with guarded age at most 120 seconds. The guard is a conservative accumulation allowance, not a measured telemetry delay.

All downloaded inputs, derived data and logs are local ignored files. The original Stage 01 README's historical `docs/DATA.md` reference is superseded by this data-access description.
