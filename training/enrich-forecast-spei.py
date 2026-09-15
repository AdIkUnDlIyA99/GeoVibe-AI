"""Add twelve historical SPEI values to an existing forecast JSONL dataset."""

import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

SPEI_SOURCE = Path(os.environ.get("SPEI_SOURCE", os.environ.get("SPEI_FILE", "/tmp/spei03.nc")))
SOURCE = Path(os.environ.get("FORECAST_DATASET", "/content/drive/MyDrive/forecast-sequences.jsonl"))
OUTPUT = Path(os.environ.get("FORECAST_ENRICHED_OUTPUT", "/content/drive/MyDrive/forecast-sequences-spei.jsonl"))


def month_key(year, month, offset):
    value = year * 12 + month - 1 + offset
    return f"{value // 12:04d}-{value % 12 + 1:02d}"


def coordinate_names(dataset):
    variable = next(name for name in dataset.data_vars if "spei" in name.lower())
    lat_name = next(name for name in dataset.coords if name.lower() in ("lat", "latitude"))
    lon_name = next(name for name in dataset.coords if name.lower() in ("lon", "longitude"))
    time_name = next(name for name in dataset.coords if name.lower() == "time")
    return variable, lat_name, lon_name, time_name


def load_spei_source(path):
    files = sorted(path.glob("*.nc")) if path.is_dir() else [path]
    if not files or not all(file.is_file() for file in files):
        raise RuntimeError(f"SPEI NetCDF source not found at {path}")

    values_by_month = {}
    latitudes = None
    longitudes = None
    title = "SPEI climate history"
    for file_number, file in enumerate(files, 1):
        with xr.open_dataset(file) as dataset:
            variable, lat_name, lon_name, time_name = coordinate_names(dataset)
            file_latitudes = np.asarray(dataset[lat_name].values, dtype=float)
            file_longitudes = np.asarray(dataset[lon_name].values, dtype=float)
            if latitudes is None:
                latitudes = file_latitudes
                longitudes = file_longitudes
                title = str(dataset.attrs.get("title", title))
            elif not np.array_equal(latitudes, file_latitudes) or not np.array_equal(longitudes, file_longitudes):
                raise RuntimeError(f"SPEI grid changed in {file.name}")

            times = np.asarray(dataset[time_name].values)
            values = np.asarray(dataset[variable].transpose(time_name, lat_name, lon_name).values, dtype=np.float32)
            for index, timestamp in enumerate(times):
                key = np.datetime_as_string(timestamp, unit="M")
                if key in values_by_month:
                    raise RuntimeError(f"Duplicate SPEI month {key} in {file.name}")
                values_by_month[key] = values[index]
        if len(files) > 1 and file_number % 12 == 0:
            print(f"Loaded {file_number}/{len(files)} monthly SPEI files", flush=True)

    if not values_by_month:
        raise RuntimeError(f"No monthly SPEI values found at {path}")
    print(f"Loaded {len(values_by_month)} SPEI months from {min(values_by_month)} through {max(values_by_month)}", flush=True)
    return latitudes, longitudes, values_by_month, title


def nearest_valid_value(values, lat_index, lon_index, max_radius=4):
    """Return the nearest finite land cell, searching no farther than about one degree."""
    latitude_count, longitude_count = values.shape
    for radius in range(max_radius + 1):
        candidates = []
        for lat_offset in range(-radius, radius + 1):
            for lon_offset in range(-radius, radius + 1):
                if radius and max(abs(lat_offset), abs(lon_offset)) != radius:
                    continue
                candidate_lat = lat_index + lat_offset
                if candidate_lat < 0 or candidate_lat >= latitude_count:
                    continue
                candidate_lon = (lon_index + lon_offset) % longitude_count
                value = float(values[candidate_lat, candidate_lon])
                if math.isfinite(value) and abs(value) < 20:
                    distance = lat_offset * lat_offset + lon_offset * lon_offset
                    candidates.append((distance, value))
        if candidates:
            candidates.sort(key=lambda item: item[0])
            return candidates[0][1], radius
    return math.nan, None


def main():
    if not SPEI_SOURCE.exists():
        raise RuntimeError(f"SPEI NetCDF source not found at {SPEI_SOURCE}")
    if not SOURCE.exists():
        raise RuntimeError(f"Forecast dataset not found at {SOURCE}")
    if SOURCE.resolve() == OUTPUT.resolve():
        raise RuntimeError("Use a different FORECAST_ENRICHED_OUTPUT so the source remains recoverable")

    latitudes, longitudes, values_by_month, source_title = load_spei_source(SPEI_SOURCE)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    written = 0
    skipped = 0
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
            fallback_radii = []
            missing_reason = None
            for offset in range(-11, 1):
                month = month_key(origin.year, origin.month, offset)
                values = values_by_month.get(month)
                value, radius = nearest_valid_value(values, lat_index, lon_index) if values is not None else (math.nan, None)
                if not math.isfinite(value) or abs(value) >= 20:
                    missing_reason = f"historical offset {offset}"
                    break
                fallback_radii.append(radius)
                history.append(round(value, 4))
            if missing_reason:
                skipped += 1
                print(
                    f"Skipped source row {line_number}: no nearby valid SPEI for {missing_reason}",
                    flush=True,
                )
                continue
            targets = []
            for horizon in (1, 3, 6):
                month = month_key(origin.year, origin.month, horizon)
                values = values_by_month.get(month)
                value, radius = nearest_valid_value(values, lat_index, lon_index) if values is not None else (math.nan, None)
                if not math.isfinite(value) or abs(value) >= 20:
                    missing_reason = f"target +{horizon} month ({month})"
                    break
                fallback_radii.append(radius)
                targets.append(round(value, 4))
            if missing_reason:
                skipped += 1
                print(
                    f"Skipped source row {line_number}: no nearby valid SPEI for {missing_reason}",
                    flush=True,
                )
                continue
            if len(row.get("input", [])) != 12:
                raise RuntimeError(f"Source row {line_number} does not contain twelve input months")
            for item, value in zip(row["input"], history):
                item["spei"] = value
            row["sources"] = [source for source in row.get("sources", []) if "spei" not in source.lower()]
            row["sources"].append(f"{source_title}; lagged 12-month SPEI-3 climate history")
            row["targets"]["spei"] = targets
            row["targets"]["drought"] = [int(value <= -1) for value in targets]
            row["speiGridFallbackCells"] = max(fallback_radii)
            target.write(json.dumps(row, separators=(",", ":")) + "\n")
            written += 1
            if written % 250 == 0:
                print(f"Enriched {written} rows", flush=True)
    temporary.replace(OUTPUT)
    print(f"Enriched {written} rows, skipped {skipped}, and wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
