"""Download one operational GloFAS +1-month forecast and build a runtime feature cache."""

import json
import os
import tempfile
import zipfile
from datetime import date
from pathlib import Path

import cdsapi
import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[2]
HISTORY = Path(os.environ.get("GLOFAS_HISTORICAL_SOURCE", "/content/drive/MyDrive/glofas-historical-1981-2025"))
OUTPUT = Path(os.environ.get("GLOFAS_LIVE_CACHE_OUTPUT", ROOT / "models/flood/glofas-current-plus1.json"))
DOWNLOAD = Path(os.environ.get("GLOFAS_LIVE_DOWNLOAD", "/content/drive/MyDrive/glofas-current-plus1.zip"))
YEAR = os.environ.get("GLOFAS_OPERATIONAL_YEAR", "2026")
MONTH = os.environ.get("GLOFAS_OPERATIONAL_MONTH", "08")
AREA = [27.5, 83.0, 24.0, 88.5]
BASELINE_END = np.datetime64("2016-12-31")


def coordinate_name(dataset, candidates):
    for name in candidates:
        if name in dataset.coords or name in dataset.dims:
            return name
    raise RuntimeError(f"Missing coordinate; expected one of {candidates}")


def discharge_name(dataset):
    for name in ("dis24", "avg_dis", "average_river_discharge_in_the_last_24_hours", "river_discharge_in_the_last_24_hours"):
        if name in dataset.data_vars:
            return name
    raise RuntimeError(f"No discharge variable found: {list(dataset.data_vars)}")


def lead_index(dataset):
    name = coordinate_name(dataset, ("forecast_period", "step", "leadtime"))
    values = np.asarray(dataset[name].values).reshape(-1)
    days = values / np.timedelta64(1, "D") if np.issubdtype(values.dtype, np.timedelta64) else values.astype(float)
    index = int(np.argmin(np.abs(days - 30)))
    if abs(float(days[index]) - 30) > 20:
        raise RuntimeError(f"No usable +1-month lead; nearest is {days[index]} days")
    return name, index


def main():
    token = os.environ.get("EWDS_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Set EWDS_TOKEN in the environment; never store it in this repository.")
    DOWNLOAD.parent.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client(url="https://ewds.climate.copernicus.eu/api", key=token)
    print(f"Requesting operational GloFAS {YEAR}-{MONTH} +1 month forecast...")
    client.retrieve("cems-glofas-seasonal", {
        "system_version": "operational",
        "hydrological_model": "lisflood",
        "variable": "river_discharge_in_the_last_24_hours",
        "year": YEAR,
        "month": MONTH,
        "leadtime_hour": "720",
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "zip",
    }, str(DOWNLOAD))

    with tempfile.TemporaryDirectory() as temporary:
        temporary_path = Path(temporary)
        with zipfile.ZipFile(DOWNLOAD) as archive:
            archive.extractall(temporary_path)
        forecast_files = list(temporary_path.rglob("*.nc"))
        if len(forecast_files) != 1:
            raise RuntimeError(f"Expected one NetCDF file, found {forecast_files}")
        with xr.open_dataset(forecast_files[0], engine="h5netcdf") as forecast:
            variable = discharge_name(forecast)
            lat_name = coordinate_name(forecast, ("latitude", "lat"))
            lon_name = coordinate_name(forecast, ("longitude", "lon"))
            member_name = coordinate_name(forecast, ("number", "realization", "member"))
            lead_name, index = lead_index(forecast)
            latitudes = np.asarray(forecast[lat_name].values)
            longitudes = np.asarray(forecast[lon_name].values)
            lead = forecast[variable].isel({lead_name: index})
            for dimension in list(lead.dims):
                if dimension not in (member_name, lat_name, lon_name):
                    lead = lead.isel({dimension: 0})
            members = np.asarray(lead.transpose(member_name, lat_name, lon_name).values, dtype=float)
            origin = str(np.asarray(forecast.get("forecast_reference_time", forecast.get("time")).values).reshape(-1)[0])

    history_files = sorted(HISTORY.glob("historical-*.nc"))
    if not history_files:
        raise RuntimeError(f"No historical NetCDF files found in {HISTORY}")
    history = xr.open_mfdataset(history_files, combine="by_coords", chunks="auto")
    history_variable = discharge_name(history)
    history_lat = coordinate_name(history, ("latitude", "lat"))
    history_lon = coordinate_name(history, ("longitude", "lon"))
    history_time = coordinate_name(history, ("time", "valid_time"))
    grid = history[history_variable].sel({history_lat: latitudes, history_lon: longitudes}, method="nearest")
    for dimension in list(grid.dims):
        if dimension not in (history_time, history_lat, history_lon):
            grid = grid.mean(dimension, skipna=True)
    thresholds = grid.sel({history_time: slice(None, BASELINE_END)}).quantile(0.95, dim=history_time, skipna=True).compute().values
    history.close()

    means, spreads = np.nanmean(members, axis=0), np.nanstd(members, axis=0)
    p10, p50, p90 = np.nanquantile(members, (0.1, 0.5, 0.9), axis=0)
    origin_month = int(origin[5:7]) - 1
    cells = []
    for lat_index, latitude in enumerate(latitudes):
        for lon_index, longitude in enumerate(longitudes):
            threshold = float(thresholds[lat_index, lon_index])
            values = [means[lat_index, lon_index], spreads[lat_index, lon_index], p10[lat_index, lon_index], p50[lat_index, lon_index], p90[lat_index, lon_index]]
            if not np.isfinite(values).all() or not np.isfinite(threshold) or threshold <= 0:
                continue
            cells.append([round(float(latitude), 5), round(float(longitude), 5), *[round(float(np.log1p(max(0, value))), 7) for value in values], round(float(np.mean(members[:, lat_index, lon_index] >= threshold)), 7), round(float(means[lat_index, lon_index] / threshold), 7), round(float(np.sin(2 * np.pi * origin_month / 12)), 7), round(float(np.cos(2 * np.pi * origin_month / 12)), 7), round(float(abs(latitude) / 90), 7)])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"schemaVersion": 1, "originDate": origin, "leadMonth": 1, "bounds": {"south": AREA[2], "north": AREA[0], "west": AREA[1], "east": AREA[3]}, "featureOrder": ["logMean", "logSpread", "logP10", "logP50", "logP90", "exceedanceProbability", "thresholdRatio", "monthSin", "monthCos", "absoluteLatitude"], "cells": cells}, separators=(",", ":")))
    print(f"Wrote {len(cells)} current GloFAS feature cells to {OUTPUT}")


if __name__ == "__main__":
    main()
