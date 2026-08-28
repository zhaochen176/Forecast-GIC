# Stage 01: Read and Save Source Data

## Purpose

`01_read_and_save_data.py` is the input stage of the study. It obtains the official daily VKH GIC records and the Wind spacecraft L1 data required by later stages. The script deliberately keeps this stage close to the providers' original format: it downloads and stores source files, and its reader can inspect one GIC file at a time without merging, resampling, interpolation, or feature engineering.

## Inputs and acquisition

The GIC downloader requests the VKH Vykhodnoy product for 1 January 2012 through 31 December 2022. The source has approximately 2 Hz sampling and a nominal dynamic range of -120 to 120 A. Wind requests retrieve GSM magnetic-field variables through the MFI service and plasma/coupling variables through the SWE service for the same period. Provider URLs, attribution, and the event-catalogue requirement are documented in `docs/DATA.md`.

## Usage and local files

From the repository root, run:

```powershell
python 01_read_and_save_data\01_read_and_save_data.py --download
```

GIC files are placed below `data/raw/GIC/YYYY/YYYY-MM/`. Wind responses are placed in `data/raw/Wind_L1/MFI/` and `data/raw/Wind_L1/SWE/`. Existing non-empty GIC files are skipped, and failed monthly Wind requests fall back to daily requests. These directories are local inputs and must not be committed.

## Next stage

After acquisition, run Stage 02. A successful Stage 01 run is indicated by source files in both Wind subdirectories and by readable GIC daily files. Missing days should be investigated before preprocessing rather than silently filled.