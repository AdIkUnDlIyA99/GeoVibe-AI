"""Build real SPEI-labelled Sentinel-2 NDVI/NDWI sequences without Earth Engine."""

import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import requests
import xarray as xr

SPEI_URL = "https://spei.csic.es/spei_database_2_11/nc/spei03.nc"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
POINT_URL = "https://planetarycomputer.microsoft.com/api/data/v1/item/point/{lon},{lat}"
TARGET = int(os.environ.get("DROUGHT_SEQUENCES", "600"))
OUTPUT = Path(os.environ.get("DROUGHT_OUTPUT", "training/data/drought-sequences.jsonl"))
SPEI_FILE = Path(os.environ.get("SPEI_FILE", "/tmp/spei03.nc"))
START_YEAR = max(2016, int(os.environ.get("DROUGHT_START_YEAR", "2016")))
END_YEAR = min(2024, int(os.environ.get("DROUGHT_END_YEAR", "2024")))


def request_json(method, url, **kwargs):
    error = None
    for attempt in range(4):
        try:
            response = requests.request(method, url, timeout=45, **kwargs)
            if response.ok:
                return response.json()
            error = RuntimeError(f"HTTP {response.status_code}")
        except requests.RequestException as exc:
            error = exc
        time.sleep(1.5 * (2**attempt))
    raise RuntimeError(f"Remote request failed: {error}")


def ensure_spei():
    if SPEI_FILE.exists() and SPEI_FILE.stat().st_size > 300_000_000:
        return
    SPEI_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading official SPEIbase v2.11 SPEI-03 to {SPEI_FILE} ...")
    urlretrieve(SPEI_URL, SPEI_FILE)


def split_for(region):
    bucket = int(hashlib.sha256(region.encode()).hexdigest()[:8], 16) % 10
    return "test" if bucket == 0 else "calibration" if bucket == 1 else "train"


def ratio(a, b):
    return 0.0 if abs(a + b) < 1e-9 else max(-1.0, min(1.0, (a - b) / (a + b)))


def month_targets(endpoint):
    year, month = endpoint.year, endpoint.month
    targets = []
    for offset in range(11, -1, -1):
        index = year * 12 + month - 1 - offset
        targets.append(datetime(index // 12, index % 12 + 1, 15, tzinfo=timezone.utc))
    return targets


def sentinel_sequence(lat, lon, endpoint):
    targets = month_targets(endpoint)
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": [lon - 0.001, lat - 0.001, lon + 0.001, lat + 0.001],
        "datetime": f"{targets[0].strftime('%Y-%m-01T00:00:00Z')}/{endpoint.strftime('%Y-%m-%dT23:59:59Z')}",
        "limit": 100,
        "query": {"eo:cloud_cover": {"lt": 40}},
    }
    items = request_json("POST", STAC_URL, json=payload).get("features", [])
    if not items:
        return None
    sequence = []
    for target in targets:
        ranked = sorted(items, key=lambda item: abs(datetime.fromisoformat(item["properties"]["datetime"].replace("Z", "+00:00")) - target).days + float(item["properties"].get("eo:cloud_cover", 100)) * 1.5)
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
                value = {"ndvi": round(ratio(bands["B08"], bands["B04"]), 4), "ndwi": round(ratio(bands["B03"], bands["B08"]), 4), "valid": True}
                break
            except (KeyError, TypeError, ValueError, RuntimeError):
                continue
        sequence.append(value or {"ndvi": 0, "ndwi": 0, "valid": False})
    return sequence if sum(item["valid"] for item in sequence) >= 8 else None


def candidate_rows(dataset):
    variable = next(name for name in dataset.data_vars if "spei" in name.lower())
    lat_name = next(name for name in dataset.coords if name.lower() in ("lat", "latitude"))
    lon_name = next(name for name in dataset.coords if name.lower() in ("lon", "longitude"))
    time_name = next(name for name in dataset.coords if name.lower() == "time")
    latitudes, longitudes = dataset[lat_name].values, dataset[lon_name].values
    times = dataset[time_name].values
    endpoints = [index for index, value in enumerate(times) if START_YEAR <= int(str(value)[:4]) <= END_YEAR and int(str(value)[5:7]) in (3, 6, 9, 12)]
    lat_step = max(1, len(latitudes) // 30)
    lon_step = max(1, len(longitudes) // 60)
    rows = []
    for lat_index in range(0, len(latitudes), lat_step):
        lat = float(latitudes[lat_index])
        if abs(lat) > 75:
            continue
        for lon_index in range(0, len(longitudes), lon_step):
            lon = float(longitudes[lon_index])
            region = f"spei-cell-{lat_index}-{lon_index}"
            for time_index in endpoints:
                spei = float(dataset[variable].isel({time_name: time_index, lat_name: lat_index, lon_name: lon_index}).values)
                if math.isfinite(spei) and abs(spei) < 20:
                    date = np.datetime_as_string(times[time_index], unit="D")
                    rows.append({"region": region, "split": split_for(region), "lat": lat, "lon": lon, "date": date, "spei": round(spei, 4)})
    drought = sorted((row for row in rows if row["spei"] <= -1), key=lambda row: hashlib.sha256((row["region"] + row["date"]).encode()).hexdigest())
    normal = sorted((row for row in rows if row["spei"] > -0.5), key=lambda row: hashlib.sha256((row["region"] + row["date"]).encode()).hexdigest())
    half = TARGET // 2
    return drought[:half] + normal[: TARGET - half]


def main():
    ensure_spei()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if OUTPUT.exists():
        for line in OUTPUT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                completed.add(f'{item["region"]}:{item["endpoint"]}')
    with xr.open_dataset(SPEI_FILE) as dataset:
        candidates = candidate_rows(dataset)
    print(f"Preparing {len(candidates)} globally distributed, class-balanced candidates; {len(completed)} already complete.")
    with OUTPUT.open("a", encoding="utf-8") as target:
        for index, candidate in enumerate(candidates, 1):
            key = f'{candidate["region"]}:{candidate["date"]}'
            if key in completed:
                continue
            endpoint = datetime.fromisoformat(candidate["date"]).replace(tzinfo=timezone.utc)
            sequence = sentinel_sequence(candidate["lat"], candidate["lon"], endpoint)
            if sequence:
                record = {"region": candidate["region"], "split": candidate["split"], "endpoint": candidate["date"], "latitude": candidate["lat"], "longitude": candidate["lon"], "spei": candidate["spei"], "sequence": sequence, "sources": ["CSIC SPEIbase v2.11", "Copernicus Sentinel-2 L2A via Microsoft Planetary Computer"]}
                target.write(json.dumps(record, separators=(",", ":")) + "\n")
                target.flush()
            print(f"[{index}/{len(candidates)}] {candidate['region']} {'saved' if sequence else 'skipped: insufficient clear observations'}")
    print(f"Drought sequences written to {OUTPUT}")


if __name__ == "__main__":
    main()
