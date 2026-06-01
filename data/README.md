# Data

This repository does not include large processed data files or experiment
outputs.

Place the processed 1-minute data table at:

```text
data/merged_2012_2022_processed.parquet
```

The code expects a `DatetimeIndex` and the columns used in `src/config.py`,
including VKH GIC, solar-wind, geomagnetic, and coupling-function variables.

Original public data sources:

- Solar wind and interplanetary magnetic field data: NASA CDAWeb, ACE
  observations at L1, https://cdaweb.gsfc.nasa.gov/
- VKH GIC observations: Vykhodnoy 220 kV substation public database,
  http://gic.en51.ru/

Generated feature caches and experiment outputs are intentionally ignored by
Git. They are recreated by the scripts and written under `data/` and
`outputs/`.
