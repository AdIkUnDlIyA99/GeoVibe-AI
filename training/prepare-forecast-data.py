"""Create global 12-month inputs and six-month future targets from real historical data."""

import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
import xarray as xr

SPEI_FILE = Path(os.environ.get("SPEI_FILE", "/content/drive/MyDrive/spei03.nc"))
OUTPUT = Path(os.environ.get("FORECAST_OUTPUT", "/content/drive/MyDrive/forecast-sequences.jsonl"))
TARGET = int(os.environ.get("FORECAST_SEQUENCES", "5000"))
START_INDEX = max(1, int(os.environ.get("FORECAST_START_INDEX", "1")))
WORKERS = max(1, min(8, int(os.environ.get("FORECAST_WORKERS", "4"))))
ATTEMPTS = Path(os.environ.get("FORECAST_ATTEMPTS", str(OUTPUT.with_suffix(".attempts.jsonl"))))
SPLITS = {
    value.strip()
    for value in os.environ.get("FORECAST_SPLITS", "train,calibration,test").split(",")
    if value.strip()
}
LATITUDE_SAMPLES = max(10, int(os.environ.get("FORECAST_LATITUDE_SAMPLES", "34")))
LONGITUDE_SAMPLES = max(20, int(os.environ.get("FORECAST_LONGITUDE_SAMPLES", "68")))
ORIGIN_MONTH_STEP = int(os.environ.get("FORECAST_ORIGIN_MONTH_STEP", "3"))
if ORIGIN_MONTH_STEP not in (1, 2, 3, 4, 6, 12):
    raise RuntimeError("FORECAST_ORIGIN_MONTH_STEP must divide twelve months")
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
    print("Loading SPEI values into memory for fast candidate generation...", flush=True)
    spei_values = dataset[variable].transpose(time_name, lat_name, lon_name).values
    years = []
    if "train" in SPLITS:
        years.extend(range(2018, 2022))
    if "calibration" in SPLITS:
        years.append(2022)
    if "test" in SPLITS:
        years.extend((2023, 2024))
    rows = []
    for lat_index in range(0, len(latitudes), max(1, len(latitudes) // LATITUDE_SAMPLES)):
        lat = float(latitudes[lat_index])
        if abs(lat) > 72:
            continue
        for lon_index in range(0, len(longitudes), max(1, len(longitudes) // LONGITUDE_SAMPLES)):
            lon = float(longitudes[lon_index])
            region = f"spei-cell-{lat_index}-{lon_index}"
            for year in years:
                split = split_for(region, year)
                if not split:
                    continue
                for month in range(ORIGIN_MONTH_STEP, 13, ORIGIN_MONTH_STEP):
                    origin = datetime(year, month, 15, tzinfo=timezone.utc)
                    indices = []
                    for horizon in (1, 3, 6):
                        target = month_targets(origin)[11 + horizon]
                        index = time_lookup.get(target.strftime("%Y-%m"))
                        if index is None:
                            break
                        value = float(spei_values[index, lat_index, lon_index])
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
    attempted = set()
    if ATTEMPTS.exists():
        for line in ATTEMPTS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                attempted.add(json.loads(line)["key"])
    with xr.open_dataset(SPEI_FILE) as dataset:
        selected = candidates(dataset)
    unknown_splits = SPLITS - {"train", "calibration", "test"}
    if unknown_splits:
        raise RuntimeError(f"Unknown FORECAST_SPLITS values: {sorted(unknown_splits)}")
    selected = [candidate for candidate in selected if candidate["split"] in SPLITS]
    pending = []
    for index, candidate in enumerate(selected, 1):
        if index < START_INDEX:
            continue
        key = f'{candidate["region"]}:{candidate["origin"].date().isoformat()}'
        if key not in completed and key not in attempted:
            pending.append((index, candidate, key))
    print(
        f"Selected {len(selected)} candidates; {len(completed)} saved, "
        f"{len(attempted)} previously attempted, {len(pending)} pending; "
        f"splits={','.join(sorted(SPLITS))}; starting at {START_INDEX} with {WORKERS} workers",
        f"grid={LATITUDE_SAMPLES}x{LONGITUDE_SAMPLES}, origin-step={ORIGIN_MONTH_STEP} month(s)",
        flush=True,
    )

    def process(entry):
        index, candidate, key = entry
        try:
            sequence = sentinel_sequence(candidate["lat"], candidate["lon"], candidate["origin"])
            return index, candidate, key, sequence, None
        except RuntimeError as error:
            return index, candidate, key, None, str(error)

    ATTEMPTS.parent.mkdir(parents=True, exist_ok=True)
    saved_this_run = 0
    processed_this_run = 0
    with OUTPUT.open("a", encoding="utf-8") as target, ATTEMPTS.open("a", encoding="utf-8") as attempts, ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(process, entry) for entry in pending]
        for future in as_completed(futures):
            index, candidate, key, sequence, error = future.result()
            origin_date = candidate["origin"].date().isoformat()
            if sequence:
                record = {
                    "region": candidate["region"], "split": candidate["split"], "originDate": origin_date,
                    "latitude": candidate["lat"], "longitude": candidate["lon"], "input": sequence[:12],
                    "targets": {"ndvi": [item["ndvi"] for item in sequence[12:]], "ndwi": [item["ndwi"] for item in sequence[12:]], "drought": [int(value <= -1) for value in candidate["spei"]], "spei": candidate["spei"]},
                    "sources": ["Copernicus Sentinel-2 L2A via Microsoft Planetary Computer", "SPEIbase local NetCDF"]
                }
                target.write(json.dumps(record, separators=(",", ":")) + "\n")
                target.flush()
                saved_this_run += 1
            attempts.write(json.dumps({"key": key, "saved": bool(sequence), "error": error}, separators=(",", ":")) + "\n")
            attempts.flush()
            processed_this_run += 1
            print(
                f"[{index}/{len(selected)}] {key} "
                f"{'saved' if sequence else ('skipped: ' + error if error else 'skipped: incomplete 18-month sequence')} "
                f"| run: {saved_this_run}/{processed_this_run} saved",
                flush=True,
            )
    print(f"Forecast sequences written to {OUTPUT}")


if __name__ == "__main__":
    main()
