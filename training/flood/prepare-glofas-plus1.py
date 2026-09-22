"""Build a +1 month flood-training dataset from GloFAS seasonal reforecasts."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


SEASONAL_SOURCE = Path(os.environ.get("GLOFAS_SEASONAL_SOURCE", "/content/drive/MyDrive/glofas-seasonal-1981-2025"))
HISTORICAL_SOURCE = Path(os.environ.get("GLOFAS_HISTORICAL_SOURCE", "/content/drive/MyDrive/glofas-historical-1981-2025"))
OUTPUT = Path(os.environ.get("FLOOD_PLUS1_OUTPUT", "/content/drive/MyDrive/glofas-flood-plus1.jsonl"))
GRID_STRIDE = max(1, int(os.environ.get("GLOFAS_GRID_STRIDE", "4")))
BASELINE_END = np.datetime64(os.environ.get("GLOFAS_BASELINE_END", "2016-12-31"))
TARGET_DAYS = 30


def coordinate_name(dataset, candidates):
    for name in candidates:
        if name in dataset.coords or name in dataset.dims:
            return name
    raise RuntimeError(f"Missing coordinate; expected one of {candidates}")


def discharge_name(dataset):
    for name in ("dis24", "avg_dis", "average_river_discharge_in_the_last_24_hours", "river_discharge_in_the_last_24_hours", "discharge"):
        if name in dataset.data_vars:
            return name
    raise RuntimeError(f"No discharge variable found; available: {list(dataset.data_vars)}")


def normalize_longitudes(longitudes, coordinate):
    values = np.asarray(coordinate.values)
    return np.mod(longitudes, 360) if values.min() >= 0 and np.any(longitudes < 0) else longitudes


def split_for(latitude, longitude, year):
    region = f"glofas-{round(latitude, 1):.1f}-{round(longitude, 1):.1f}"
    bucket = int(hashlib.sha256(region.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 70 and year <= 2017:
        return region, "train"
    if 70 <= bucket < 85 and 2018 <= year <= 2020:
        return region, "calibration"
    if bucket >= 85 and year >= 2021:
        return region, "test"
    return region, None


def origin_time(dataset):
    for name in ("forecast_reference_time", "time"):
        if name in dataset.coords:
            return pd.Timestamp(np.asarray(dataset[name].values).reshape(-1)[0])
    raise RuntimeError("Seasonal file is missing forecast_reference_time/time")


def plus_one_index(dataset):
    lead_name = coordinate_name(dataset, ("forecast_period", "step", "leadtime"))
    values = np.asarray(dataset[lead_name].values).reshape(-1)
    days = values / np.timedelta64(1, "D") if np.issubdtype(values.dtype, np.timedelta64) else values.astype(float)
    index = int(np.argmin(np.abs(days - TARGET_DAYS)))
    if abs(float(days[index]) - TARGET_DAYS) > 20:
        raise RuntimeError(f"Missing usable +1 month lead; nearest lead is {days[index]} days")
    return lead_name, index


def main():
    seasonal_files = sorted(path for path in SEASONAL_SOURCE.glob("seasonal-????-??.nc") if path.stat().st_size > 10_000)
    historical_files = sorted(HISTORICAL_SOURCE.glob("historical-*.nc")) if HISTORICAL_SOURCE.is_dir() else [HISTORICAL_SOURCE]
    if not seasonal_files:
        raise RuntimeError(f"No individual seasonal NetCDF files found in {SEASONAL_SOURCE}")
    if not historical_files:
        raise RuntimeError(f"No historical NetCDF files found in {HISTORICAL_SOURCE}")

    with xr.open_dataset(seasonal_files[0]) as sample:
        seasonal_lat = coordinate_name(sample, ("latitude", "lat"))
        seasonal_lon = coordinate_name(sample, ("longitude", "lon"))
        latitudes = np.asarray(sample[seasonal_lat].values)[::GRID_STRIDE]
        longitudes = np.asarray(sample[seasonal_lon].values)[::GRID_STRIDE]

    history = xr.open_mfdataset(historical_files, combine="by_coords", chunks="auto")
    history_variable = discharge_name(history)
    history_lat = coordinate_name(history, ("latitude", "lat"))
    history_lon = coordinate_name(history, ("longitude", "lon"))
    history_time = coordinate_name(history, ("time", "valid_time"))
    history_grid = history[history_variable].sel(
        {history_lat: latitudes, history_lon: normalize_longitudes(longitudes, history[history_lon])}, method="nearest"
    )
    for dimension in list(history_grid.dims):
        if dimension not in (history_time, history_lat, history_lon):
            history_grid = history_grid.mean(dimension, skipna=True)
    history_grid = history_grid.load()
    thresholds = history_grid.sel({history_time: slice(None, BASELINE_END)}).quantile(0.95, dim=history_time, skipna=True).load()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    rows_written = 0
    rows_skipped = 0

    with temporary.open("w", encoding="utf-8") as target:
        for file_number, file in enumerate(seasonal_files, 1):
            with xr.open_dataset(file) as dataset:
                variable = discharge_name(dataset)
                lat_name = coordinate_name(dataset, ("latitude", "lat"))
                lon_name = coordinate_name(dataset, ("longitude", "lon"))
                member_name = coordinate_name(dataset, ("number", "realization", "member"))
                lead_name, lead_index = plus_one_index(dataset)
                origin = origin_time(dataset)
                valid_start = np.datetime64((origin + pd.DateOffset(months=1)).replace(day=1).date())
                valid_end = np.datetime64((pd.Timestamp(valid_start) + pd.offsets.MonthEnd(1)).date())
                observed = history_grid.sel({history_time: slice(valid_start, valid_end)}).max(history_time, skipna=True)
                lead = dataset[variable].isel({lead_name: lead_index}).sel(
                    {lat_name: latitudes, lon_name: normalize_longitudes(longitudes, dataset[lon_name])}, method="nearest"
                )
                for dimension in list(lead.dims):
                    if dimension not in (member_name, lat_name, lon_name):
                        lead = lead.isel({dimension: 0})
                lead = lead.transpose(member_name, lat_name, lon_name).load()
                members = np.asarray(lead.values, dtype=float)
                means = np.nanmean(members, axis=0)
                spreads = np.nanstd(members, axis=0)
                p10, p50, p90 = np.nanquantile(members, (0.1, 0.5, 0.9), axis=0)
                observed_values = np.asarray(observed.values, dtype=float)
                threshold_values = np.asarray(thresholds.values, dtype=float)

                for lat_index, latitude in enumerate(latitudes):
                    for lon_index, longitude in enumerate(longitudes):
                        region, split = split_for(float(latitude), float(longitude), origin.year)
                        threshold = threshold_values[lat_index, lon_index]
                        member_values = members[:, lat_index, lon_index]
                        if split is None or not np.isfinite(threshold) or threshold <= 0 or not np.isfinite(observed_values[lat_index, lon_index]):
                            rows_skipped += 1
                            continue
                        values = (means[lat_index, lon_index], spreads[lat_index, lon_index], p10[lat_index, lon_index], p50[lat_index, lon_index], p90[lat_index, lon_index])
                        if not np.isfinite(values).all():
                            rows_skipped += 1
                            continue
                        row = {
                            "schemaVersion": 1,
                            "originDate": origin.isoformat(),
                            "latitude": float(latitude),
                            "longitude": float(longitude),
                            "region": region,
                            "split": split,
                            "floodThreshold": float(threshold),
                            "forecast": [{
                                "leadMonth": 1,
                                "mean": float(values[0]), "spread": float(values[1]), "p10": float(values[2]),
                                "p50": float(values[3]), "p90": float(values[4]),
                                "exceedanceProbability": float(np.mean(member_values >= threshold)),
                                "thresholdRatio": float(values[0] / threshold),
                            }],
                            "targets": {"flood": [int(observed_values[lat_index, lon_index] >= threshold)]},
                        }
                        target.write(json.dumps(row, separators=(",", ":")) + "\n")
                        rows_written += 1
            if file_number % 12 == 0 or file_number == len(seasonal_files):
                print(f"Prepared {file_number}/{len(seasonal_files)} origins; rows={rows_written}; skipped={rows_skipped}", flush=True)

    history.close()
    temporary.replace(OUTPUT)
    print(f"Wrote {rows_written} +1 month GloFAS flood rows to {OUTPUT}")


if __name__ == "__main__":
    main()
