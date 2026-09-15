"""Join ECMWF seasonal ensemble summaries to SPEI drought sequences."""

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr


SOURCE = Path(os.environ.get("DROUGHT_DATASET", "/content/drive/MyDrive/drought-sequences-era5-spei.jsonl"))
SEASONAL_SOURCE = Path(os.environ.get("SEASONAL_SOURCE", "/content/drive/MyDrive/ecmwf-seasonal-drought-1981-2025"))
OUTPUT = Path(os.environ.get("DROUGHT_SEASONAL_OUTPUT", "/content/drive/MyDrive/drought-sequences-seasonal.jsonl"))
VARIABLES = ("t2m_mean", "t2m_spread", "tprate_mean", "tprate_spread")
MAX_FALLBACK_DEGREES = 7.5


def build_climatology():
    climatology = {}
    for month in range(1, 13):
        temperatures = []
        precipitation = []
        template = None
        for year in range(1981, 2017):
            file = SEASONAL_SOURCE / f"seasonal-{year}-{month:02d}.nc"
            if not file.exists():
                raise RuntimeError(f"Missing hindcast climatology file: {file}")
            with xr.open_dataset(file) as dataset:
                temperatures.append(np.asarray(dataset["t2m_mean"].values, dtype=np.float32))
                precipitation.append(np.asarray(dataset["tprate_mean"].values, dtype=np.float32))
                if template is None:
                    template = (np.asarray(dataset["latitude"].values), np.asarray(dataset["longitude"].values))
        temperature_stack = np.stack(temperatures)
        precipitation_stack = np.stack(precipitation)
        climatology[month] = xr.Dataset(
            {
                "t2m_climate_mean": (("step", "latitude", "longitude"), np.mean(temperature_stack, axis=0)),
                "t2m_climate_std": (("step", "latitude", "longitude"), np.maximum(np.std(temperature_stack, axis=0), 0.1)),
                "tprate_climate_mean": (("step", "latitude", "longitude"), np.mean(precipitation_stack, axis=0)),
                "tprate_climate_std": (("step", "latitude", "longitude"), np.maximum(np.std(precipitation_stack, axis=0), 1e-10)),
            },
            coords={"step": np.arange(6), "latitude": template[0], "longitude": template[1]},
        )
        print(f"Built hindcast climatology for initialization month {month:02d}", flush=True)
    return climatology


def nearest_valid_item(dataset, latitude, longitude, lead):
    planes = [np.asarray(dataset[name].isel(step=lead).values) for name in VARIABLES]
    valid = np.all(np.isfinite(np.stack(planes)), axis=0)
    valid_lat, valid_lon = np.where(valid)
    if not len(valid_lat):
        return None, None
    grid_latitudes = np.asarray(dataset["latitude"].values)[valid_lat]
    grid_longitudes = np.asarray(dataset["longitude"].values)[valid_lon]
    lon_delta = np.abs(grid_longitudes - longitude)
    lon_delta = np.minimum(lon_delta, 360 - lon_delta)
    distances = np.sqrt((grid_latitudes - latitude) ** 2 + (lon_delta * np.cos(np.radians(latitude))) ** 2)
    nearest = int(np.argmin(distances))
    if float(distances[nearest]) > MAX_FALLBACK_DEGREES:
        return None, None
    lat_index, lon_index = int(valid_lat[nearest]), int(valid_lon[nearest])
    item = {name: float(planes[index][lat_index, lon_index]) for index, name in enumerate(VARIABLES)}
    return item, float(distances[nearest])


def main():
    if not SOURCE.exists():
        raise RuntimeError(f"Drought dataset not found: {SOURCE}")
    if not SEASONAL_SOURCE.exists():
        raise RuntimeError(f"Seasonal source directory not found: {SEASONAL_SOURCE}")
    with SOURCE.open("r", encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    climatology = build_climatology()
    by_origin = defaultdict(list)
    for index, row in enumerate(rows):
        by_origin[row["originDate"][:7]].append(index)

    skipped = 0
    for number, (origin, indexes) in enumerate(sorted(by_origin.items()), 1):
        file = SEASONAL_SOURCE / f"seasonal-{origin}.nc"
        if not file.exists():
            raise RuntimeError(f"Missing seasonal forecast for origin {origin}: {file}")
        with xr.open_dataset(file) as dataset:
            missing = [name for name in VARIABLES if name not in dataset]
            if missing:
                raise RuntimeError(f"Missing variables in {file.name}: {missing}")
            latitudes = xr.DataArray([rows[index]["latitude"] for index in indexes], dims="points")
            longitudes = xr.DataArray([rows[index]["longitude"] for index in indexes], dims="points")
            selected = dataset[list(VARIABLES)].sel(latitude=latitudes, longitude=longitudes, method="nearest")
            selected_climate = climatology[int(origin[5:7])].sel(latitude=latitudes, longitude=longitudes, method="nearest")
            for point, row_index in enumerate(indexes):
                forecasts = []
                fallback_degrees = 0.0
                for lead in range(6):
                    item = {name: float(selected[name].isel(points=point, step=lead).values) for name in VARIABLES}
                    if not all(np.isfinite(value) for value in item.values()):
                        item, distance = nearest_valid_item(
                            dataset,
                            float(rows[row_index]["latitude"]),
                            float(rows[row_index]["longitude"]),
                            lead,
                        )
                        if item is None:
                            rows[row_index] = None
                            skipped += 1
                            break
                        fallback_degrees = max(fallback_degrees, distance)
                    climate = {name: float(selected_climate[name].isel(points=point, step=lead).values) for name in selected_climate.data_vars}
                    item["t2m_anomaly"] = (item["t2m_mean"] - climate["t2m_climate_mean"]) / climate["t2m_climate_std"]
                    item["tprate_anomaly"] = (item["tprate_mean"] - climate["tprate_climate_mean"]) / climate["tprate_climate_std"]
                    forecasts.append({"leadMonth": lead + 1, **{name: round(value, 8) for name, value in item.items()}})
                if rows[row_index] is not None:
                    rows[row_index]["seasonalForecast"] = forecasts
                    rows[row_index]["seasonalGridFallbackDegrees"] = round(fallback_degrees, 3)
        if number % 12 == 0 or number == len(by_origin):
            print(f"Enriched {number}/{len(by_origin)} forecast origins", flush=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        for row in rows:
            if row is not None:
                target.write(json.dumps(row, separators=(",", ":")) + "\n")
    temporary.replace(OUTPUT)
    print(f"Wrote {len(rows) - skipped} seasonally enriched rows, skipped {skipped}, to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
