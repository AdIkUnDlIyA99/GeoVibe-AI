"""Build geographically disjoint, time-separated SPEI-3 drought sequences."""

import json
import math
import os
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr


SOURCE = Path(os.environ.get("SPEI_SOURCE", "/content/drive/MyDrive/era5-spei3-2000-latest"))
OUTPUT = Path(os.environ.get("DROUGHT_OUTPUT", "/content/drive/MyDrive/drought-sequences-era5-spei.jsonl"))
SEED = int(os.environ.get("DROUGHT_SEED", "20260915"))
TARGETS = {"train": 100_000, "calibration": 15_000, "test": 20_000}
CELL_COUNTS = {"train": 1_000, "calibration": 500, "test": 500}
ORIGIN_RANGES = {
    "train": ("2001-01", "2018-06"),
    "calibration": ("2019-01", "2021-06"),
    "test": ("2022-01", "2025-12"),
}


def month_key(year, month, offset=0):
    absolute = year * 12 + month - 1 + offset
    return f"{absolute // 12:04d}-{absolute % 12 + 1:02d}"


def months_between(start, end):
    current = datetime.strptime(start, "%Y-%m")
    finish = datetime.strptime(end, "%Y-%m")
    months = []
    while current <= finish:
        months.append(current.strftime("%Y-%m"))
        current = datetime(current.year + (current.month == 12), current.month % 12 + 1, 1)
    return months


def dataset_parts(dataset):
    variable_name = next((name for name in dataset.data_vars if "spei" in name.lower()), None)
    if not variable_name:
        raise RuntimeError("No SPEI variable found")
    variable = dataset[variable_name]
    latitude_name = next(name for name in variable.dims if name.lower() in {"lat", "latitude"})
    longitude_name = next(name for name in variable.dims if name.lower() in {"lon", "longitude"})
    time_name = next(name for name in variable.dims if name.lower() == "time")
    values = variable.isel({time_name: 0}).transpose(latitude_name, longitude_name).values
    date = np.datetime_as_string(dataset[time_name].values[0], unit="M")
    return date, dataset[latitude_name].values, dataset[longitude_name].values, values


def select_cells(files, rng):
    with xr.open_dataset(files[len(files) // 2]) as dataset:
        _, latitudes, longitudes, values = dataset_parts(dataset)
    valid = np.argwhere(np.isfinite(values) & (np.abs(values) < 20))
    valid = valid[np.abs(latitudes[valid[:, 0]]) <= 72]
    if len(valid) < sum(CELL_COUNTS.values()):
        raise RuntimeError("Not enough valid SPEI land cells")

    # Sampling latitude bands separately prevents dense mid-latitude grids dominating.
    bands = defaultdict(list)
    for lat_index, lon_index in valid:
        band = int((float(latitudes[lat_index]) + 72) // 12)
        bands[band].append((int(lat_index), int(lon_index)))
    for cells in bands.values():
        rng.shuffle(cells)
    selected = []
    band_ids = sorted(bands)
    while len(selected) < sum(CELL_COUNTS.values()):
        progressed = False
        for band in band_ids:
            if bands[band]:
                selected.append(bands[band].pop())
                progressed = True
                if len(selected) == sum(CELL_COUNTS.values()):
                    break
        if not progressed:
            break
    rng.shuffle(selected)
    return selected, latitudes, longitudes


def load_series(files, cells):
    series = {}
    for number, file in enumerate(files, 1):
        with xr.open_dataset(file) as dataset:
            date, _, _, values = dataset_parts(dataset)
            series[date] = np.asarray([values[lat, lon] for lat, lon in cells], dtype=np.float32)
        if number % 24 == 0 or number == len(files):
            print(f"Loaded {number}/{len(files)} monthly SPEI files", flush=True)
    return series


def balanced_sample(groups, target, rng):
    for rows in groups.values():
        rng.shuffle(rows)
    selected = []
    keys = sorted(groups)
    while len(selected) < target:
        progressed = False
        for key in keys:
            if groups[key]:
                selected.append(groups[key].pop())
                progressed = True
                if len(selected) == target:
                    break
        if not progressed:
            break
    return selected


def build_rows(split, cells, cell_offset, latitudes, longitudes, series, rng):
    groups = defaultdict(list)
    start, end = ORIGIN_RANGES[split]
    for origin_key in months_between(start, end):
        origin = datetime.strptime(origin_key, "%Y-%m")
        required = [month_key(origin.year, origin.month, offset) for offset in range(-11, 1)]
        targets = [month_key(origin.year, origin.month, horizon) for horizon in (1, 3, 6)]
        if any(month not in series for month in required + targets):
            continue
        for local_index, (lat_index, lon_index) in enumerate(cells):
            index = cell_offset + local_index
            history = [float(series[month][index]) for month in required]
            future = [float(series[month][index]) for month in targets]
            if not all(math.isfinite(value) and abs(value) < 20 for value in history + future):
                continue
            labels = [int(value <= -1) for value in future]
            pattern = "".join(map(str, labels))
            groups[pattern].append({
                "originDate": f"{origin_key}-15",
                "region": f"spei-cell-{lat_index}-{lon_index}",
                "split": split,
                "latitude": round(float(latitudes[lat_index]), 4),
                "longitude": round(float(longitudes[lon_index]), 4),
                "input": [{"spei": round(value, 4)} for value in history],
                "targets": {"spei": [round(value, 4) for value in future], "drought": labels},
                "source": "Copernicus ERA5-Drought SPEI-3 v1.0",
            })
    rows = balanced_sample(groups, TARGETS[split], rng)
    if len(rows) < TARGETS[split]:
        raise RuntimeError(f"Only {len(rows)} valid {split} rows; required {TARGETS[split]}")
    rng.shuffle(rows)
    patterns = defaultdict(int)
    for row in rows:
        patterns["".join(map(str, row["targets"]["drought"]))] += 1
    print(f"{split}: {len(rows)} rows, label patterns {dict(patterns)}", flush=True)
    return rows


def main():
    if not SOURCE.exists():
        raise RuntimeError(f"SPEI source directory not found: {SOURCE}")
    files = sorted(SOURCE.rglob("*.nc"))
    if len(files) < 300:
        raise RuntimeError(f"Expected at least 300 monthly SPEI files, found {len(files)}")
    rng = random.Random(SEED)
    cells, latitudes, longitudes = select_cells(files, rng)
    series = load_series(files, cells)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    offset = 0
    counts = {}
    with temporary.open("w", encoding="utf-8") as target:
        for split in ("train", "calibration", "test"):
            count = CELL_COUNTS[split]
            split_cells = cells[offset:offset + count]
            rows = build_rows(split, split_cells, offset, latitudes, longitudes, series, rng)
            for row in rows:
                target.write(json.dumps(row, separators=(",", ":")) + "\n")
            counts[split] = len(rows)
            offset += count
    temporary.replace(OUTPUT)
    print(f"Wrote {sum(counts.values())} rows to {OUTPUT}: {counts}", flush=True)


if __name__ == "__main__":
    main()
