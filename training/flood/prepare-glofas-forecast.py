"""Build supervised +1/+3/+6 month flood rows from GloFAS seasonal and historical discharge."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


SEASONAL_SOURCE = Path(os.environ.get("GLOFAS_SEASONAL_SOURCE", "/content/drive/MyDrive/glofas-seasonal-1981-2025"))
HISTORICAL_SOURCE = Path(os.environ.get("GLOFAS_HISTORICAL_SOURCE", "/content/drive/MyDrive/glofas-historical-1981-2025"))
OUTPUT = Path(os.environ.get("FLOOD_FORECAST_OUTPUT", "/content/drive/MyDrive/glofas-flood-sequences.jsonl"))
GRID_STRIDE = max(1, int(os.environ.get("GLOFAS_GRID_STRIDE", "4")))
BASELINE_END = np.datetime64(os.environ.get("GLOFAS_BASELINE_END", "2016-12-31"))
HORIZONS = (1, 3, 6)
TARGET_DAYS = np.asarray((30, 90, 180))


def coordinate_name(dataset, candidates):
    for name in candidates:
        if name in dataset.coords or name in dataset.dims:
            return name
    raise RuntimeError(f"Missing coordinate; expected one of {candidates}")


def discharge_name(dataset):
    for name in (
        "dis24",
        "average_river_discharge_in_the_last_24_hours",
        "river_discharge_in_the_last_24_hours",
        "discharge",
    ):
        if name in dataset.data_vars:
            return name
    raise RuntimeError(f"No discharge variable found; available: {list(dataset.data_vars)}")


def normalize_longitude(longitude, coordinate):
    values = np.asarray(coordinate.values)
    return longitude % 360 if values.min() >= 0 and longitude < 0 else longitude


def split_for(latitude, longitude, year):
    region = f"glofas-{round(latitude, 1):.1f}-{round(longitude, 1):.1f}"
    bucket = int(hashlib.sha256(region.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 70 and year <= 2018:
        return region, "train"
    if 70 <= bucket < 85 and 2019 <= year <= 2021:
        return region, "calibration"
    if bucket >= 85 and year >= 2022:
        return region, "test"
    return region, None


def lead_indices(dataset):
    if "valid_time" in dataset.coords:
        valid = np.asarray(dataset["valid_time"].values).reshape(-1)
        origin = np.asarray(dataset["forecast_reference_time"].values).reshape(-1)[0] if "forecast_reference_time" in dataset.coords else np.asarray(dataset["time"].values).reshape(-1)[0]
        days = (valid - origin) / np.timedelta64(1, "D")
    else:
        lead_name = coordinate_name(dataset, ("forecast_period", "step", "leadtime"))
        values = np.asarray(dataset[lead_name].values).reshape(-1)
        days = values / np.timedelta64(1, "D") if np.issubdtype(values.dtype, np.timedelta64) else values.astype(float)
    return [int(np.argmin(np.abs(days - target))) for target in TARGET_DAYS]


def origin_time(dataset):
    for name in ("forecast_reference_time", "time"):
        if name in dataset.coords:
            return pd.Timestamp(np.asarray(dataset[name].values).reshape(-1)[0])
    raise RuntimeError("Seasonal file is missing forecast_reference_time/time")


def historical_series(history, variable, latitude, longitude, lat_name, lon_name):
    selected_lon = normalize_longitude(longitude, history[lon_name])
    point = history[variable].sel({lat_name: latitude, lon_name: selected_lon}, method="nearest")
    for dimension in list(point.dims):
        if dimension not in ("time", "valid_time"):
            point = point.mean(dimension, skipna=True)
    time_name = "time" if "time" in point.coords else "valid_time"
    return point, time_name


def observed_peak(point, time_name, valid_date):
    start = np.datetime64(pd.Timestamp(valid_date).replace(day=1).date())
    end = np.datetime64((pd.Timestamp(start) + pd.offsets.MonthEnd(1)).date())
    values = np.asarray(point.sel({time_name: slice(start, end)}).values, dtype=float)
    return float(np.nanmax(values)) if np.isfinite(values).any() else None


def main():
    seasonal_files = sorted(SEASONAL_SOURCE.glob("*.nc"))
    historical_files = sorted(HISTORICAL_SOURCE.glob("*.nc")) if HISTORICAL_SOURCE.is_dir() else [HISTORICAL_SOURCE]
    if not seasonal_files:
        raise RuntimeError(f"No seasonal NetCDF files found in {SEASONAL_SOURCE}")
    if not historical_files or not all(path.exists() for path in historical_files):
        raise RuntimeError(f"No historical NetCDF files found at {HISTORICAL_SOURCE}")

    history = xr.open_mfdataset(historical_files, combine="by_coords", chunks="auto")
    history_variable = discharge_name(history)
    history_lat = coordinate_name(history, ("latitude", "lat"))
    history_lon = coordinate_name(history, ("longitude", "lon"))
    rows_written = 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8") as target:
        for file_number, file in enumerate(seasonal_files, 1):
            with xr.open_dataset(file) as dataset:
                variable = discharge_name(dataset)
                lat_name = coordinate_name(dataset, ("latitude", "lat"))
                lon_name = coordinate_name(dataset, ("longitude", "lon"))
                member_name = coordinate_name(dataset, ("number", "realization", "member"))
                lead_name = coordinate_name(dataset, ("forecast_period", "step", "leadtime"))
                origin = origin_time(dataset)
                indices = lead_indices(dataset)
                latitudes = np.asarray(dataset[lat_name].values)[::GRID_STRIDE]
                longitudes = np.asarray(dataset[lon_name].values)[::GRID_STRIDE]
                discharge = dataset[variable]
                for latitude in latitudes:
                    for longitude in longitudes:
                        region, split = split_for(float(latitude), float(longitude), origin.year)
                        if split is None:
                            continue
                        history_point, time_name = historical_series(history, history_variable, float(latitude), float(longitude), history_lat, history_lon)
                        baseline = history_point.sel({time_name: slice(None, BASELINE_END)})
                        threshold = float(baseline.quantile(0.95, dim=time_name, skipna=True).compute())
                        if not np.isfinite(threshold) or threshold <= 0:
                            continue
                        forecasts, targets = [], []
                        valid = True
                        for horizon, lead_index in zip(HORIZONS, indices):
                            lead = discharge.isel({lead_name: lead_index}).sel({lat_name: latitude, lon_name: longitude}, method="nearest")
                            for dimension in list(lead.dims):
                                if dimension not in (member_name,):
                                    lead = lead.isel({dimension: 0})
                            members = np.asarray(lead.values, dtype=float).reshape(-1)
                            members = members[np.isfinite(members)]
                            if not len(members):
                                valid = False
                                break
                            valid_date = origin + pd.DateOffset(months=horizon)
                            observed = observed_peak(history_point, time_name, valid_date)
                            if observed is None:
                                valid = False
                                break
                            mean, spread = float(np.mean(members)), float(np.std(members))
                            p10, p50, p90 = (float(value) for value in np.quantile(members, (0.1, 0.5, 0.9)))
                            forecasts.append({"leadMonth": horizon, "mean": mean, "spread": spread, "p10": p10, "p50": p50, "p90": p90, "exceedanceProbability": float(np.mean(members >= threshold)), "thresholdRatio": mean / threshold})
                            targets.append(int(observed >= threshold))
                        if valid:
                            row = {"schemaVersion": 1, "originDate": origin.isoformat(), "latitude": float(latitude), "longitude": float(longitude), "region": region, "split": split, "floodThreshold": threshold, "forecast": forecasts, "targets": {"flood": targets}}
                            target.write(json.dumps(row, separators=(",", ":")) + "\n")
                            rows_written += 1
            if file_number % 12 == 0 or file_number == len(seasonal_files):
                print(f"Prepared {file_number}/{len(seasonal_files)} forecast origins; rows={rows_written}", flush=True)

    history.close()
    temporary.replace(OUTPUT)
    print(f"Wrote {rows_written} GloFAS flood forecast rows to {OUTPUT}")


if __name__ == "__main__":
    main()
