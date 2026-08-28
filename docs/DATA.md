# Data Access

This project does not redistribute observations or derived data. Download inputs from their original providers and keep them in ignored local directories.

| Input | Source | Local location | Used by |
| --- | --- | --- | --- |
| VKH GIC daily records | `http://gic.en51.ru/data/vkh_gic/` | `data/raw/GIC/YYYY/YYYY-MM/YYYYMMDD.txt` | Steps 1-3 |
| Wind MFI magnetic-field data | Wind Science Data Center | `data/raw/Wind_L1/MFI/` | Steps 1-3 |
| Wind SWE plasma data | Wind Science Data Center | `data/raw/Wind_L1/SWE/` | Steps 1-3 |
| VKH event catalogue | Supporting information of the associated study | `data/external/supporting_information.docx` | Step 3 |

Step 1 downloads VKH daily records and requests Wind ASCII responses for 2012-01-01 through 2022-12-31. It may be re-run safely: existing non-empty daily GIC files are skipped and failed monthly Wind requests fall back to daily requests.

The event catalogue is not downloaded automatically. Obtain the exact supporting-information document used by the study and place it at the location above. Do not redistribute source, derived, or labelled data unless the provider explicitly permits redistribution and sublicensing.