# Data Preparation

This directory contains the code and local placeholders for building the final
1-minute dataset:

```text
data/merged_2012_2022_processed.parquet
```

Large raw files, intermediate parquet files, and generated datasets are not
tracked by Git.

The paper workflow in this repository uses ACE/L1 solar-wind variables and VKH
GIC observations only. Geomagnetic observation data are not required.

## Public Sources

- Solar wind and interplanetary magnetic field data: NASA CDAWeb, ACE
  observations at L1, https://cdaweb.gsfc.nasa.gov/
- VKH GIC observations: Vykhodnoy 220 kV substation public database,
  http://gic.en51.ru/

## Expected Inputs

Download/export the ACE solar-wind table to:

```text
data/raw/ace_solar_wind_2012_2022.csv
```

The scripts accept CSV or parquet files with a datetime/timestamp column.
Column aliases are handled in `scripts/build_merged_dataset.py`. The required
standard variables are:

```text
Solar wind: Btot, Bx_gse, By_gse, Bz_gse, Vp, Np
GIC: gic
```

## Build Steps

Download and aggregate the public VKH GIC text files:

```bash
python data/scripts/download_vkh_gic.py
```

This creates:

```text
data/interim/vkh_gic_2012_2022_1min.parquet
```

Merge solar-wind and GIC data, then compute solar-wind coupling functions:

```bash
python data/scripts/build_merged_dataset.py
```

This creates the final modeling input:

```text
data/merged_2012_2022_processed.parquet
```

The output schema is:

```text
Btot, Bx_gse, By_gse, Bz_gse, Vp, Np_filled, P_dyn_nPa, Ey_mV/m, Ma,
epsilon_norm, Newell, Borovsky, density_missing, filled_by_model,
gic
```
