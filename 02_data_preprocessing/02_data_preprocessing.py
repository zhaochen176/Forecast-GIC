"""Stage 02: validate sources, clean native GIC and provide causal UTC readers.

The original Stage 01 remains unchanged. This stage reuses its downloaded
files and recovers missing revision observations only when not --offline.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from zipfile import ZipFile
from xml.etree import ElementTree as ET
import argparse, calendar, gzip, hashlib, io, json, re, shutil, subprocess
import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPOSITORY_ROOT
DATA = ROOT / "data"
CATALOGUE = DATA / "external/supporting_information.docx"
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
OFFLINE = False

def configure_workspace(workspace=None, catalogue=None):
    global ROOT, DATA, CATALOGUE
    ROOT = Path(workspace).resolve() if workspace is not None else REPOSITORY_ROOT
    DATA = ROOT / "data"
    CATALOGUE = Path(catalogue).resolve() if catalogue is not None else DATA / "external/supporting_information.docx"

def supplied_tables():
    source = CATALOGUE
    if not source.exists():
        # Input catalogues are supplied locally, never bundled as results.
        required = [DATA / "supplied_events.csv", DATA / "supplied_blocks.csv"]
        if not all(path.is_file() for path in required):
            raise FileNotFoundError(
                "Provide data/external/supporting_information.docx (the revision "
                "catalogue with 92 events and 49 blocks), or both "
                "data/supplied_events.csv and data/supplied_blocks.csv. See 02_data_preprocessing/02_README.md."
            )
        events=pd.read_csv(DATA/"supplied_events.csv").to_dict("records")
        blocks=pd.read_csv(DATA/"supplied_blocks.csv").to_dict("records")
        assert len(events)==92 and sum(bool(e["retained"]) for e in events)==71
        assert len(blocks)==49 and len({b["block"] for b in blocks})==49
        assert [sum(b["split"] == s for b in blocks) for s in ["train", "val", "test"]] == [30, 10, 9]
        assert set(b["block"] for b in blocks) == set(range(1, 50))
        return events,blocks
    with ZipFile(source) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    tables = []
    for table in root.findall(".//w:tbl", NS):
        rows = []
        previous_split = ""
        for row in table.findall("./w:tr", NS):
            cells = ["".join(t.text for t in cell.findall(".//w:t", NS) if t.text).strip()
                     for cell in row.findall("./w:tc", NS)]
            if len(cells) == 11:
                if cells[0] in ("Train","Val","Test"): previous_split = cells[0]
                elif not cells[0] and cells[1].isdigit(): cells[0] = previous_split
            rows.append(cells)
        tables.append(rows)
    events, blocks = [], []
    for rows in tables:
        for cells in rows:
            if len(cells) == 9 and re.fullmatch(r"\d+\*?", cells[0]):
                events.append(dict(event_id=int(cells[0].rstrip("*")),
                                   peak_time=str(pd.Timestamp(cells[1])),
                                   gic_peak_text=cells[2], driver=cells[6],
                                   retained=cells[8].lower() == "retained"))
            if len(cells) == 11 and cells[0] in ("Train", "Val", "Test") and cells[1].isdigit():
                blocks.append(dict(split=cells[0].lower(), block=int(cells[1]),
                                   event_ids=cells[2], start=str(pd.Timestamp(cells[3])),
                                   end=str(pd.Timestamp(cells[4]))))
    assert len(events) == 92 and sum(e["retained"] for e in events) == 71
    assert len(blocks) == 49 and len({b["block"] for b in blocks}) == 49
    assert [sum(b["split"] == s for b in blocks) for s in ["train","val","test"]] == [30,10,9]
    DATA.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(events).to_csv(DATA / "supplied_events.csv", index=False)
    pd.DataFrame(blocks).to_csv(DATA / "supplied_blocks.csv", index=False)
    return events, blocks


def requested_days(blocks):
    # One day before each block supplies causal histories, never extra labels.
    days = set(pd.date_range("2021-12-31", "2022-12-31", freq="D"))
    for b in blocks:
        days.update(pd.date_range(pd.Timestamp(b["start"]).floor("D") - pd.Timedelta(days=1),
                                  pd.Timestamp(b["end"]).floor("D"), freq="D"))
    return sorted(days)


def curl(url, target, payload=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    cmd = [shutil.which("curl.exe") or shutil.which("curl") or "curl", "-sS", "--compressed", "-L", "--connect-timeout", "15",
           "--max-time", "180", "--retry", "2", "--retry-delay", "2",
           "--fail", url, "-o", str(part)]
    if payload is not None:
        cmd += ["--data", payload]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip()[-500:])
    data = part.read_bytes()
    if not data or b"<html" in data[:500].lower():
        raise ValueError("Empty or HTML response")
    return data, part


def gic(day):
    key = day.strftime("%Y%m%d")
    out = DATA / "GIC_minute" / f"{key}.parquet"
    meta_path = out.with_suffix(".json")
    if out.exists() and meta_path.exists():
        return {"kind": "GIC", "key": key, "status": "cached"}
    url = f"http://gic.en51.ru/data/vkh_gic/{day.year}/{day:%Y-%m}/{key}.txt"
    rawpath = DATA / "raw/GIC" / f"{key}.txt"
    local = DATA / "raw/GIC" / str(day.year) / f"{day:%Y-%m}" / f"{key}.txt"
    if not local.exists():
        local = rawpath
    part = None
    try:
        if local.exists() and local.stat().st_size:
            data = local.read_bytes()
        elif rawpath.with_suffix(".txt.gz").exists():
            data = gzip.decompress(rawpath.with_suffix(".txt.gz").read_bytes())
        elif OFFLINE:
            raise FileNotFoundError(f"Missing local GIC source for {key}; offline mode forbids download")
        else:
            data, part = curl(url, rawpath)
    except RuntimeError as exc:
        if "404" not in str(exc):
            raise
        # A source-confirmed absent day is different from a failed transfer.
        index = pd.date_range(day, day + pd.Timedelta(days=1), freq="min", inclusive="left")
        missing = pd.DataFrame({"GIC": np.nan, "valid_native_samples": 0}, index=index)
        missing.index.name = "time"
        out.parent.mkdir(exist_ok=True, parents=True)
        missing.to_parquet(out)
        meta_path.write_text(json.dumps(dict(source=url, http_status=404,
                                             minute_rows=1440, valid_minutes=0), indent=2), encoding="utf-8")
        return dict(kind="GIC", key=key, status="source_missing_404")
    frame = pd.read_csv(io.BytesIO(data), sep=r"\s+", header=None,
                        names=["year","month","day","hour","minute","second","gic"])
    if frame.shape[1] != 7 or frame.empty:
        raise ValueError("Invalid GIC columns")
    # Strict parsing: no silent skipping of malformed records.
    frame = frame.apply(pd.to_numeric, errors="raise")
    actual_day = pd.to_datetime(frame[["year","month","day"]])
    assert actual_day.eq(day).all()
    timestamp = actual_day + pd.to_timedelta(frame.hour, unit="h") + pd.to_timedelta(frame.minute, unit="m") + pd.to_timedelta(frame.second, unit="s")
    assert timestamp.is_monotonic_increasing and timestamp.is_unique
    values = frame.gic.to_numpy(float, copy=True)
    bad = ~np.isfinite(values) | (np.abs(values) > 120)
    # Match the documented legacy preprocessing; preserve exclusion counts.
    for sentinel in (9.99,99.99,999.99,9999.99,999.9,9999.9,99999.9):
        bad |= np.isclose(np.abs(values), sentinel, atol=1e-6, rtol=0)
    values[bad] = np.nan
    minute = timestamp.dt.floor("min")
    valid = pd.DataFrame({"time": minute, "GIC": values}).dropna()
    if len(valid):
        ix = valid.GIC.abs().groupby(valid.time).idxmax()
        series = valid.loc[ix].set_index("time").GIC
        counts = valid.groupby("time").size()
    else:
        series, counts = pd.Series(dtype=float), pd.Series(dtype=int)
    index = pd.date_range(day, day + pd.Timedelta(days=1), freq="min", inclusive="left")
    result = pd.DataFrame({"GIC": series.reindex(index), "valid_native_samples": counts.reindex(index, fill_value=0)})
    result.index.name = "time"
    out.parent.mkdir(exist_ok=True, parents=True)
    result.to_parquet(out)
    compressed = rawpath.with_suffix(".txt.gz")
    with gzip.open(compressed, "wb", compresslevel=3) as handle:
        handle.write(data)
    meta = dict(kind="GIC", key=key, source=url, sha256=hashlib.sha256(data).hexdigest(),
                response_bytes=len(data), rows=len(frame), valid_rows=int((~bad).sum()),
                minute_rows=len(result), valid_minutes=int(result.GIC.notna().sum()),
                first_time=str(timestamp.iloc[0]), last_time=str(timestamp.iloc[-1]),
                aggregation="Signed native sample with largest absolute value in [minute,minute+1min)")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if part is not None:
        part.unlink()
    return dict(kind="GIC", key=key, status="downloaded" if part is not None else "local_validated")


def wind(job):
    year, month, product = job
    key = f"{year}{month:02d}_{product}"
    out = DATA / "raw/Wind" / f"{key}.txt"
    expected_columns = (["Year","DOY","millisecs","B","Bx","By","Bz"] if product == "MFI" else
                        ["Year","DOY","millisecs","V","Np","Psw","Ma","Mms","Epsilon","VBs"])
    if out.exists() and out.with_suffix(".json").exists():
        cached = out.read_bytes()
        cached_dates = [line.split()[:2] for line in cached.splitlines() if re.match(rb"^\d{4}\s+\d",line)]
        month_start = pd.Timestamp(year,month,1).dayofyear
        month_end = pd.Timestamp(year,month,calendar.monthrange(year,month)[1]).dayofyear
        dates_valid = bool(cached_dates) and all(int(y)==year and month_start <= int(d) <= month_end for y,d in cached_dates)
        header_valid = len(cached.splitlines()) > 1 and set(expected_columns).issubset(cached.splitlines()[1].decode().split())
        metadata = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
        hash_valid = hashlib.sha256(cached).hexdigest() == metadata.get("sha256")
        if header_valid and dates_valid and hash_valid:
            return dict(kind="Wind", key=key, status="cached")
        # Preserve rejected responses rather than silently overwriting them.
        out.rename(out.with_suffix(".rejected.txt"))
        out.with_suffix(".json").rename(out.with_suffix(".rejected.json"))
    end = calendar.monthrange(year, month)[1]
    params = ("p01=b&p02=bx&p03=by&p04=bz" if product=="MFI" else
              "p07=v&p13=np&p15=psw&p20=ma&p21=mms&p22=epsilon&p23=vbs")
    payload = (f"output=ascii&system=gsm&time1=1_min&time2=92_sec&{params}"
               f"&startdate={year}{month:02d}01&starttime=000000"
               f"&enddate={year}{month:02d}{end:02d}&endtime=235959")
    url = "https://wind.nasa.gov/mfi_swe_display_results.php"
    local = DATA / "raw/Wind_L1" / product / f"{key}.txt"
    part = None
    if local.exists() and local.stat().st_size:
        data = local.read_bytes()
    elif out.exists() and not out.with_suffix(".json").exists():
        data = out.read_bytes()
        local = out
    elif OFFLINE:
        raise FileNotFoundError(f"Missing validated local Wind source for {key}; offline mode forbids download")
    else:
        data, part = curl(url, out, payload)
    assert b"Year" in data[:300] and b"millisecs" in data[:300], data[:300]
    actual_columns = data.splitlines()[1].decode().split()
    # Stage 01 SWE requests include extra plasma variables. Keep native bytes
    # and use the same required columns; Stage 03 selects the revision features.
    if part is None:
        assert actual_columns[:3] == expected_columns[:3] and set(expected_columns).issubset(actual_columns), "Wind response product mismatch"
    else:
        assert actual_columns == expected_columns, "Wind response product mismatch"
    numeric_lines = [line for line in data.splitlines() if re.match(rb"^\d{4}\s+\d",line)]
    if not numeric_lines:
        raise ValueError("No Wind observations in response")
    assert all(len(line.split()) == len(actual_columns) for line in numeric_lines), "Wind row width mismatch"
    years = {int(line[:4]) for line in numeric_lines}
    assert years == {year}, years
    doys = [int(line.split()[1]) for line in numeric_lines]
    lower = pd.Timestamp(year,month,1).dayofyear
    upper = pd.Timestamp(year,month,end).dayofyear
    assert min(doys) >= lower and max(doys) <= upper, "Wind response month mismatch"
    if part is not None:
        part.replace(out)
    elif local != out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
    out.with_suffix(".json").write_text(json.dumps(dict(
        source=url, parameters=payload if part is not None else None,
        acquisition="download" if part is not None else "local_source_validated",
        local_source=str(local.relative_to(ROOT)) if part is None else None,
        sha256=hashlib.sha256(data).hexdigest(),
        bytes=len(data), records=len(numeric_lines), header=data.splitlines()[0].decode(),
        first_record=numeric_lines[0].decode(), last_record=numeric_lines[-1].decode()
    ), indent=2), encoding="utf-8")
    return dict(kind="Wind", key=key, status="downloaded" if part is not None else "local_validated")


def run(jobs, fn, workers, logname):
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool, (DATA / logname).open("a", encoding="utf-8") as log:
        futures = {pool.submit(fn, j): j for j in jobs}
        for n, future in enumerate(as_completed(futures), 1):
            job = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = dict(job=str(job), error=str(exc), status="failed")
                failures.append(row)
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()
            if n % 10 == 0 or row["status"] == "failed" or n == len(jobs):
                print(f"{logname}: {n}/{len(jobs)}; failures={len(failures)}", flush=True)
    return failures


def read_wind(path):
    columns = path.read_text(encoding="utf-8").splitlines()[1].split()
    raw = pd.read_csv(path, sep=r"\s+", names=columns, skiprows=3, dtype=str)
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    times = pd.to_datetime(numeric.Year.astype("Int64").astype(str), format="%Y") + pd.to_timedelta(numeric.DOY-1, unit="D") + pd.to_timedelta(numeric.millisecs, unit="ms")
    valid = times.notna()
    out = pd.DataFrame(index=pd.DatetimeIndex(times[valid], name="source_time"))
    for col in columns[3:]:
        values = numeric.loc[valid, col].to_numpy(float, copy=True)
        text = raw.loc[valid,col]
        # Match explicit ASCII/CDF fill conventions, not general outliers.
        sentinel = text.str.fullmatch(r"[+-]?9{3,}(?:\.9+)?(?:[eE][+-]?\d+)?", na=False).to_numpy()
        values[sentinel | (np.abs(values) >= 1e30)] = np.nan
        out[col] = values
    if out.index.has_duplicates:
        raise ValueError(f"Duplicate Wind time in {path}")
    return out.sort_index()


def monthly_wind(year, month, product):
    source = DATA / "raw/Wind" / f"{year}{month:02d}_{product}.txt"
    meta = source.with_suffix(".json")
    if not source.exists() or not meta.exists():
        raise FileNotFoundError(f"Not yet recovered: {source}")
    cache = DATA / "Wind_parquet" / f"{year}{month:02d}_{product}.parquet"
    if not cache.exists() or cache.stat().st_mtime < source.stat().st_mtime:
        frame = read_wind(source)
        cache.parent.mkdir(exist_ok=True, parents=True)
        frame.to_parquet(cache)
    return pd.read_parquet(cache)


def align_prior(source, issue_times, max_age_seconds):
    """Only strictly earlier observations; preserve invalid latest records."""
    source = source.sort_index()
    aligned = pd.merge_asof(
        pd.DataFrame({"issue_time": issue_times}),
        source.reset_index(),
        left_on="issue_time", right_on="source_time",
        direction="backward", allow_exact_matches=False,
        tolerance=pd.Timedelta(seconds=max_age_seconds),
    ).set_index("issue_time")
    age = (aligned.index.to_series() - aligned.source_time).dt.total_seconds()
    assert age.dropna().gt(0).all() and age.dropna().le(max_age_seconds).all()
    return aligned.drop(columns="source_time"), age


def gic_frame(index):
    days = sorted(set(index.floor("D")))
    parts = []
    for day in days:
        path = DATA / "GIC_minute" / f"{day:%Y%m%d}.parquet"
        if not path.exists() or not path.with_suffix(".json").exists():
            raise FileNotFoundError(f"Not yet recovered: {path}")
        parts.append(pd.read_parquet(path))
    frame = pd.concat(parts).reindex(index)
    for day in ["2013-06-29","2013-07-06"]:
        frame.loc[frame.index.floor("D") == pd.Timestamp(day),"GIC"] = np.nan
    return frame


def main():
    global OFFLINE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["plan", "gic", "wind", "all"], default="plan")
    parser.add_argument("--workspace", type=Path, help="Local data/run root; default is the repository root.")
    parser.add_argument("--catalogue", type=Path, help="Revision Supporting Information DOCX; only input tables S2/S3 are read.")
    parser.add_argument("--offline", action="store_true", help="Use existing sources only; do not download missing files.")
    args = parser.parse_args()
    configure_workspace(args.workspace, args.catalogue)
    OFFLINE = args.offline
    _, blocks = supplied_tables()
    days = requested_days(blocks)
    months = sorted({(d.year, d.month) for d in days})
    jobs = [(y, m, product) for y, m in months for product in ["MFI", "SWE"]]
    print(json.dumps(dict(days=len(days), wind_month_products=len(jobs)), indent=2), flush=True)
    failures = []
    if args.kind in {"gic", "all"}:
        failures.extend(run(days, gic, 6, "recovery_GIC.jsonl"))
    if args.kind in {"wind", "all"}:
        # The legacy CGI may cross concurrent responses: always serial.
        failures.extend(run(jobs, wind, 1, "recovery_Wind.jsonl"))
    if failures:
        raise SystemExit(f"Recovery failed for {len(failures)} inputs; inspect data/recovery_*.jsonl before Stage 03.")


if __name__ == "__main__":
    main()
