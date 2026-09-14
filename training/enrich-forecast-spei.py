"""Add twelve historical SPEI values to an existing forecast JSONL dataset."""

import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

SPEI_FILE = Path(os.environ.get("SPEI_FILE", "/tmp/spei03.nc"))
SOURCE = Path(os.environ.get("FORECAST_DATASET", "/content/drive/MyDrive/forecast-sequences.jsonl"))
OUTPUT = Path(os.environ.get("FORECAST_ENRICHED_OUTPUT", "/content/drive/MyDrive/forecast-sequences-spei.jsonl"))


def month_key(year, month, offset):
    value = year * 12 + month - 1 + offset
    return f"{value // 12:04d}-{value % 12 + 1:02d}"


def main():
    if not SPEI_FILE.exists() or SPEI_FILE.stat().st_size < 300_000_000:
        raise RuntimeError(f"A complete SPEI NetCDF is required at {SPEI_FILE}")
    if not SOURCE.exists():
        raise RuntimeError(f"Forecast dataset not found at {SOURCE}")
    if SOURCE.resolve() == OUTPUT.resolve():
        raise RuntimeError("Use a different FORECAST_ENRICHED_OUTPUT so the source remains recoverable")

    with xr.open_dataset(SPEI_FILE) as dataset:
        variable = next(name for name in dataset.data_vars if "spei" in name.lower())
        lat_name = next(name for name in dataset.coords if name.lower() in ("lat", "latitude"))
        lon_name = next(name for name in dataset.coords if name.lower() in ("lon", "longitude"))
        time_name = next(name for name in dataset.coords if name.lower() == "time")
        latitudes = np.asarray(dataset[lat_name].values, dtype=float)
        longitudes = np.asarray(dataset[lon_name].values, dtype=float)
        time_lookup = {np.datetime_as_string(value, unit="M"): index for index, value in enumerate(dataset[time_name].values)}
        print("Loading SPEI values into memory...", flush=True)
        values = dataset[variable].transpose(time_name, lat_name, lon_name).values

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    written = 0
    with SOURCE.open("r", encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            origin = datetime.fromisoformat(row["originDate"])
            latitude = float(row["latitude"])
            longitude = float(row["longitude"])
            if longitudes.min() >= 0 and longitude < 0:
                longitude %= 360
            lat_index = int(np.abs(latitudes - latitude).argmin())
            lon_index = int(np.abs(longitudes - longitude).argmin())
            history = []
            for offset in range(-11, 1):
                index = time_lookup.get(month_key(origin.year, origin.month, offset))
                value = float(values[index, lat_index, lon_index]) if index is not None else math.nan
                if not math.isfinite(value) or abs(value) >= 20:
                    raise RuntimeError(f"Missing historical SPEI at source row {line_number}, offset {offset}")
                history.append(round(value, 4))
            if len(row.get("input", [])) != 12:
                raise RuntimeError(f"Source row {line_number} does not contain twelve input months")
            for item, value in zip(row["input"], history):
                item["spei"] = value
            row.setdefault("sources", []).append("SPEIbase lagged 12-month climate history")
            target.write(json.dumps(row, separators=(",", ":")) + "\n")
            written += 1
            if written % 250 == 0:
                print(f"Enriched {written} rows", flush=True)
    temporary.replace(OUTPUT)
    print(f"Enriched {written} rows and wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
