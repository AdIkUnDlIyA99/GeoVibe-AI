"""Create global 12-month inputs and six-month future targets from real historical data."""

import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
import xarray as xr

SPEI_FILE = Path(os.environ.get("SPEI_FILE", "/content/drive/MyDrive/spei03.nc"))
OUTPUT = Path(os.environ.get("FORECAST_OUTPUT", "/content/drive/MyDrive/forecast-sequences.jsonl"))
TARGET = int(os.environ.get("FORECAST_SEQUENCES", "5000"))
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
POINT_URL = "https://planetarycomputer.microsoft.com/api/data/v1/item/point/{lon},{lat}"


def request_json(method, url, **kwargs):
    error = None
    for attempt in range(5):
        try:
            response = requests.request(method, url, timeout=60, **kwargs)
            if response.ok:
                return response.json()
            error = RuntimeError(f"HTTP {response.status_code}")
        except requests.RequestException as exc:
            error = exc
        time.sleep(min(20, 2**attempt))
    raise RuntimeError(f"Remote request failed: {error}")


def ratio(a, b):
    return 0.0 if abs(a + b) < 1e-9 else max(-1.0, min(1.0, (a - b) / (a + b)))


def month_targets(origin):
    targets = []
    for offset in range(-11, 7):
        month_index = origin.year * 12 + origin.month - 1 + offset
        targets.append(datetime(month_index // 12, month_index % 12 + 1, 15, tzinfo=timezone.utc))
    return targets


def sentinel_sequence(lat, lon, origin):
    targets = month_targets(origin)
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": [lon - 0.001, lat - 0.001, lon + 0.001, lat + 0.001],
        "datetime": f"{targets[0].strftime('%Y-%m-01T00:00:00Z')}/{targets[-1].strftime('%Y-%m-28T23:59:59Z')}",
        "limit": 250,
        "query": {"eo:cloud_cover": {"lt": 45}},
    }
    items = request_json("POST", STAC_URL, json=payload).get("features", [])
    sequence = []
    for target in targets:
        ranked = sorted(items, key=lambda item: abs((datetime.fromisoformat(item["properties"]["datetime"].replace("Z", "+00:00")) - target).days) + float(item["properties"].get("eo:cloud_cover", 100)))
        value = None
        for item in ranked[:3]:
            observed = datetime.fromisoformat(item["properties"]["datetime"].replace("Z", "+00:00"))
            if abs((observed - target).days) > 35:
                continue
            try:
                point = request_json("GET", POINT_URL.format(lon=lon, lat=lat), params=[("collection", "sentinel-2-l2a"), ("item", item["id"]), ("assets", "B03"), ("assets", "B04"), ("assets", "B08"), ("assets", "SCL")])
                bands = {name.split("_")[0]: float(point["values"][index]) for index, name in enumerate(point["band_names"])}
                if not all(math.isfinite(bands.get(name, math.nan)) and bands[name] > 0 for name in ("B03", "B04", "B08")):
                    continue
                if "SCL" in bands and round(bands["SCL"]) not in (2, 4, 5, 6, 7):
                    continue
                value = {"ndvi": round(ratio(bands["B08"], bands["B04"]), 4), "ndwi": round(ratio(bands["B03"], bands["B08"]), 4), "valid": True, "date": observed.date().isoformat()}
                break
            except (KeyError, TypeError, ValueError, RuntimeError):
                continue
        sequence.append(value)
    return sequence if all(sequence) else None


def split_for(region, year):
    bucket = int(hashlib.sha256(region.encode()).hexdigest()[:8], 16) % 10
    if bucket < 8 and year <= 2021:
        return "train"
    if bucket == 8 and year == 2022:
        return "calibration"
    if bucket == 9 and year >= 2023:
        return "test"
    return None


def candidates(dataset):
    variable = next(name for name in dataset.data_vars if "spei" in name.lower())
    lat_name = next(name for name in dataset.coords if name.lower() in ("lat", "latitude"))
    lon_name = next(name for name in dataset.coords if name.lower() in ("lon", "longitude"))
    time_name = next(name for name in dataset.coords if name.lower() == "time")
    times = dataset[time_name].values
    time_lookup = {np.datetime_as_string(value, unit="M"): index for index, value in enumerate(times)}
    latitudes, longitudes = dataset[lat_name].values, dataset[lon_name].values
    rows = []
    for lat_index in range(0, len(latitudes), max(1, len(latitudes) // 34)):
        lat = float(latitudes[lat_index])
        if abs(lat) > 72:
            continue
        for lon_index in range(0, len(longitudes), max(1, len(longitudes) // 68)):
            lon = float(longitudes[lon_index])
            region = f"spei-cell-{lat_index}-{lon_index}"
            for year in range(2018, 2025):
                split = split_for(region, year)
                if not split:
                    continue
                for month in (3, 6, 9, 12):
                    origin = datetime(year, month, 15, tzinfo=timezone.utc)
                    indices = []
                    for horizon in (1, 3, 6):
                        target = month_targets(origin)[11 + horizon]
                        index = time_lookup.get(target.strftime("%Y-%m"))
                        if index is None:
                            break
                        value = float(dataset[variable].isel({time_name: index, lat_name: lat_index, lon_name: lon_index}).values)
                        indices.append(value)
                    if len(indices) == 3 and all(math.isfinite(value) and abs(value) < 20 for value in indices):
                        rows.append({"region": region, "split": split, "lat": lat, "lon": lon, "origin": origin, "spei": indices})
    rows.sort(key=lambda row: hashlib.sha256(f'{row["region"]}:{row["origin"].isoformat()}'.encode()).hexdigest())
    per_split = {"train": int(TARGET * 0.72), "calibration": int(TARGET * 0.12), "test": TARGET - int(TARGET * 0.84)}
    selected = []
    for split, count in per_split.items():
        group = [row for row in rows if row["split"] == split]
        drought = [row for row in group if row["spei"][-1] <= -1]
        normal = [row for row in group if row["spei"][-1] > -0.5]
        selected.extend(drought[: count // 2] + normal[: count - count // 2])
    return selected


def main():
    if not SPEI_FILE.exists() or SPEI_FILE.stat().st_size < 300_000_000:
        raise RuntimeError(f"A complete SPEI NetCDF is required at {SPEI_FILE}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if OUTPUT.exists():
        for line in OUTPUT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed.add(f'{row["region"]}:{row["originDate"]}')
    with xr.open_dataset(SPEI_FILE) as dataset:
        selected = candidates(dataset)
    print(f"Selected {len(selected)} globally distributed spatiotemporal candidates; {len(completed)} already saved")
    with OUTPUT.open("a", encoding="utf-8") as target:
        for index, candidate in enumerate(selected, 1):
            origin_date = candidate["origin"].date().isoformat()
            key = f'{candidate["region"]}:{origin_date}'
            if key in completed:
                continue
            sequence = sentinel_sequence(candidate["lat"], candidate["lon"], candidate["origin"])
            if sequence:
                record = {
                    "region": candidate["region"], "split": candidate["split"], "originDate": origin_date,
                    "latitude": candidate["lat"], "longitude": candidate["lon"], "input": sequence[:12],
                    "targets": {"ndvi": [item["ndvi"] for item in sequence[12:]], "ndwi": [item["ndwi"] for item in sequence[12:]], "drought": [int(value <= -1) for value in candidate["spei"]], "spei": candidate["spei"]},
                    "sources": ["Copernicus Sentinel-2 L2A via Microsoft Planetary Computer", "SPEIbase local NetCDF"]
                }
                target.write(json.dumps(record, separators=(",", ":")) + "\n")
                target.flush()
            print(f"[{index}/{len(selected)}] {key} {'saved' if sequence else 'skipped: incomplete 18-month sequence'}")
    print(f"Forecast sequences written to {OUTPUT}")


if __name__ == "__main__":
    main()
