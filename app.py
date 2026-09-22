"""Python backend for GeoVibe AI's satellite analysis and flood outlook API."""

from __future__ import annotations

import json
import math
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import joblib
import numpy as np
import requests
from flask import Flask, Response, jsonify, request, send_from_directory


ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / "public"
FLOOD_MODEL_PATH = ROOT / "models" / "flood" / "flood-cnn.json"
DROUGHT_MODEL_PATH = ROOT / "models" / "drought" / "drought-forecast.json"
FORECAST_MODEL_PATH = ROOT / "models" / "forecast" / "forecast-cnn.json"
GLOFAS_MODEL_PATH = Path(os.environ.get("GLOFAS_PLUS1_MODEL_PATH", ROOT / "models" / "flood" / "glofas-plus1-tree.joblib"))
GLOFAS_METRICS_PATH = Path(os.environ.get("GLOFAS_PLUS1_METRICS_PATH", ROOT / "models" / "flood" / "glofas-plus1-tree-metrics.json"))

STAC_SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
DATA_API = "https://planetarycomputer.microsoft.com/api/data/v1/item"
SENTINEL_COLLECTIONS = ("sentinel-2-l2a", "sentinel-2-l1c")
TARGET_OBSERVATIONS = 12
CAPTURE_TOLERANCE_DAYS = 65
SCENE_CANDIDATES_PER_TARGET = 6
SAMPLE_RADIUS_DEGREES = 0.018
IMAGE_RADIUS_DEGREES = 0.025
GLOFAS_BOUNDS = {"south": 24.0, "north": 27.5, "west": 83.0, "east": 88.5}

app = Flask(__name__, static_folder=None)
session = requests.Session()
geocode_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
analysis_cache: dict[str, tuple[float, dict[str, Any]]] = {}
cache_lock = Lock()
last_geocode_at = 0.0


def read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


FLOOD_MODEL = read_json(FLOOD_MODEL_PATH)
DROUGHT_MODEL = read_json(DROUGHT_MODEL_PATH)
FORECAST_MODEL = read_json(FORECAST_MODEL_PATH)
if FLOOD_MODEL is None:
    raise RuntimeError("Missing models/flood/flood-cnn.json")
if DROUGHT_MODEL is None:
    raise RuntimeError("Missing models/drought/drought-forecast.json")


def load_glofas_model() -> dict[str, Any] | None:
    if not GLOFAS_MODEL_PATH.exists():
        return None
    return joblib.load(GLOFAS_MODEL_PATH)


GLOFAS_MODEL = load_glofas_model()
GLOFAS_METRICS = read_json(GLOFAS_METRICS_PATH)


def clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-clamp(value, -30, 30)))


def parse_date(value: str | None, fallback: date) -> date:
    try:
        return date.fromisoformat(value or "")
    except ValueError:
        return fallback


def iso(value: datetime | date) -> str:
    if isinstance(value, date) and not isinstance(value, datetime):
        return f"{value.isoformat()}T00:00:00.000Z"
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def add_months(value: date, months: int) -> date:
    month = value.month - 1 + months
    year = value.year + month // 12
    month = month % 12 + 1
    day = min(value.day, (date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)).day)
    return date(year, month, day)


def fetch_with_retry(method: str, url: str, **kwargs: Any) -> requests.Response:
    error: Exception | None = None
    for attempt in range(3):
        try:
            response = session.request(method, url, timeout=25, **kwargs)
            if response.ok or response.status_code < 500 or attempt == 2:
                return response
        except requests.RequestException as exc:
            error = exc
            if attempt == 2:
                raise
        time.sleep(0.35 * (2**attempt))
    raise error or RuntimeError("Remote request failed")


def cached(cache: dict[str, tuple[float, Any]], key: str) -> Any | None:
    with cache_lock:
        item = cache.get(key)
        if item and item[0] > time.time():
            return item[1]
        cache.pop(key, None)
    return None


def cache_set(cache: dict[str, tuple[float, Any]], key: str, value: Any, seconds: int) -> None:
    with cache_lock:
        cache[key] = (time.time() + seconds, value)
        while len(cache) > 100:
            cache.pop(next(iter(cache)), None)


def geocode(query: str) -> list[dict[str, Any]]:
    global last_geocode_at
    key = query.strip().lower()
    if not key:
        return []
    hit = cached(geocode_cache, key)
    if hit is not None:
        return hit

    results: list[dict[str, Any]] = []
    try:
        response = fetch_with_retry("GET", "https://photon.komoot.io/api/", params={"q": query, "limit": 10, "lang": "en"})
        if response.ok:
            for feature in response.json().get("features", []):
                properties = feature.get("properties", {})
                coordinates = feature.get("geometry", {}).get("coordinates", [])
                parts = [properties.get(name) for name in ("name", "street", "city", "locality", "county", "state", "country")]
                if len(coordinates) >= 2:
                    results.append({"name": ", ".join(dict.fromkeys(part for part in parts if part)), "lat": float(coordinates[1]), "lon": float(coordinates[0]), "boundingBox": None, "provider": "Photon"})
    except requests.RequestException:
        pass

    try:
        pause = max(0, 1.1 - (time.time() - last_geocode_at))
        if pause:
            time.sleep(pause)
        response = fetch_with_retry("GET", "https://nominatim.openstreetmap.org/search", params={"q": query, "format": "jsonv2", "limit": 10, "dedupe": 0, "addressdetails": 1}, headers={"Accept-Language": "en", "User-Agent": "GeoVibe-AI/1.0 academic-environmental-monitoring-prototype"})
        last_geocode_at = time.time()
        if response.ok:
            for item in response.json():
                box = item.get("boundingbox") or []
                results.append({"name": item.get("display_name", ""), "lat": float(item["lat"]), "lon": float(item["lon"]), "boundingBox": [float(value) for value in box] if len(box) == 4 else None, "provider": "Nominatim"})
    except requests.RequestException:
        pass

    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in results:
        identity = f"{item['name'].lower()}|{item['lat']:.3f}|{item['lon']:.3f}"
        if item["name"] and identity not in seen and math.isfinite(item["lat"]) and math.isfinite(item["lon"]):
            seen.add(identity)
            unique.append(item)
    if not unique:
        raise RuntimeError("Location search providers are temporarily unavailable")
    cache_set(geocode_cache, key, unique[:12], 24 * 60 * 60)
    return unique[:12]


def search_sentinel_items(lat: float, lon: float, start: date, end: date) -> list[dict[str, Any]]:
    response = fetch_with_retry("POST", STAC_SEARCH, json={"collections": list(SENTINEL_COLLECTIONS), "bbox": [lon - 0.001, lat - 0.001, lon + 0.001, lat + 0.001], "datetime": f"{iso(start)}/{iso(end)}", "limit": 200, "query": {"eo:cloud_cover": {"lt": 70}}}, headers={"content-type": "application/json", "user-agent": "GeoVibe-AI academic-project"})
    if not response.ok:
        raise RuntimeError(f"Sentinel catalog request failed ({response.status_code})")
    return response.json().get("features", [])


def sample_sentinel_point(item: dict[str, Any], lat: float, lon: float) -> dict[str, float] | None:
    collection = item.get("collection") if item.get("collection") in SENTINEL_COLLECTIONS else "sentinel-2-l2a"
    assets = ["B03", "B04", "B08"] + (["SCL"] if collection == "sentinel-2-l2a" else [])
    try:
        response = fetch_with_retry("GET", f"{DATA_API}/point/{lon},{lat}", params=[("collection", collection), ("item", item["id"])] + [("assets", asset) for asset in assets])
        if not response.ok:
            return None
        payload = response.json()
        bands = {name.split("_")[0]: float(payload["values"][index]) for index, name in enumerate(payload["band_names"])}
        if not all(math.isfinite(bands.get(name, float("nan"))) and bands[name] > 0 for name in ("B03", "B04", "B08")):
            return None
        if "SCL" in bands and round(bands["SCL"]) not in (2, 4, 5, 6, 7):
            return None
        ratio = lambda left, right: 0.0 if abs(left + right) < 1e-9 else round((left - right) / (left + right), 3)
        return {"ndvi": ratio(bands["B08"], bands["B04"]), "ndwi": ratio(bands["B03"], bands["B08"])}
    except (requests.RequestException, KeyError, TypeError, ValueError):
        return None


def median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=float)))


def sample_sentinel_item(item: dict[str, Any], lat: float, lon: float) -> dict[str, Any] | None:
    lon_radius = SAMPLE_RADIUS_DEGREES / max(0.25, math.cos(math.radians(lat)))
    samples: list[dict[str, Any]] = []
    cell = 0
    for lat_offset in (-SAMPLE_RADIUS_DEGREES, 0, SAMPLE_RADIUS_DEGREES):
        for lon_offset in (-lon_radius, 0, lon_radius):
            values = sample_sentinel_point(item, lat + lat_offset, lon + lon_offset)
            if values:
                samples.append({**values, "cell": cell, "lat": lat + lat_offset, "lon": lon + lon_offset})
            cell += 1
    if len(samples) < 3:
        return None
    ndvi = [sample["ndvi"] for sample in samples]
    ndwi = [sample["ndwi"] for sample in samples]
    return {"id": item["id"], "collection": item.get("collection"), "date": item.get("properties", {}).get("datetime"), "cloudCover": round(float(item.get("properties", {}).get("eo:cloud_cover", 0)), 1), "indices": {"ndvi": round(median(ndvi), 3), "ndwi": round(median(ndwi), 3)}, "features": [round(median(ndvi), 3), round(median(ndwi), 3)], "sampleCount": len(samples), "dispersion": {"ndvi": round(max(ndvi) - min(ndvi), 3), "ndwi": round(max(ndwi) - min(ndwi), 3)}, "spatialSamples": samples}


def satellite_observations(lat: float, lon: float, start: date, end: date) -> list[dict[str, Any]]:
    items = search_sentinel_items(lat, lon, start, end)
    if not items:
        raise RuntimeError("No Sentinel-2 coverage was found for this location and date range")
    targets = [start + (end - start) * index / (TARGET_OBSERVATIONS - 1) for index in range(TARGET_OBSERVATIONS)]
    samples: list[dict[str, Any]] = []
    for target in targets:
        candidates = sorted(items, key=lambda item: abs(datetime.fromisoformat(item["properties"]["datetime"].replace("Z", "+00:00")).date() - target).days + float(item.get("properties", {}).get("eo:cloud_cover", 100)) * 1.25 + (0 if item.get("collection") == "sentinel-2-l2a" else 8))[:SCENE_CANDIDATES_PER_TARGET]
        selected = None
        for item in candidates:
            selected = sample_sentinel_item(item, lat, lon)
            if selected:
                captured = datetime.fromisoformat(selected["date"].replace("Z", "+00:00")).date()
                selected.update({"targetDate": iso(target), "targetOffsetDays": abs((captured - target).days)})
                break
        if selected:
            samples.append(selected)
    samples.sort(key=lambda item: item["targetDate"])
    if len(samples) < 2:
        raise RuntimeError("Fewer than two usable Sentinel-2 observations were returned. Check internet access or choose a nearby date/location.")
    return samples


def flood_cnn_inference(samples: list[dict[str, Any]]) -> dict[str, float] | None:
    if len(samples) != 9:
        return None
    ordered = sorted(samples, key=lambda item: item["cell"])
    ndvi, ndwi = [item["ndvi"] for item in ordered], [item["ndwi"] for item in ordered]
    probabilities = []
    for index in range(9):
        x, y = index % 3, index // 3
        features = []
        for channel in (ndvi, ndwi):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    px, py = clamp(x + dx, 0, 2), clamp(y + dy, 0, 2)
                    features.append(channel[int(py) * 3 + int(px)])
        raw = float(FLOOD_MODEL["weights"][0]) + sum(value * float(FLOOD_MODEL["weights"][position + 1]) for position, value in enumerate(features))
        calibration = FLOOD_MODEL.get("calibration", {"a": 1, "b": 0})
        probabilities.append(sigmoid(float(calibration["a"]) * raw + float(calibration["b"])))
    return {"temporaryWaterFraction": round(float(np.mean(probabilities)), 4), "peakProbability": round(float(max(probabilities)), 4)}


def forecast_sequence(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(observations) >= 12:
        return [{**item["indices"], "valid": True} for item in observations[-12:]]
    return [{**observations[round(index * (len(observations) - 1) / 11)]["indices"], "valid": False} for index in range(12)]


def predict_forecast_cnn(sequence: list[dict[str, Any]]) -> dict[str, list[float]]:
    """Port of the compact trained forecast CNN used by the former Node backend."""
    model = FORECAST_MODEL
    if model is None:
        raise RuntimeError("Forecast CNN artifact is unavailable")
    inputs = [[clamp(float(item.get("ndvi", 0))), clamp(float(item.get("ndwi", 0))), 0 if item.get("valid") is False else 1] for item in sequence]
    pooled: list[float] = []
    for filter_data in model["filters"]:
        for position in range(len(inputs) - 2):
            value = float(filter_data["bias"])
            for offset in range(3):
                for channel in range(3):
                    value += inputs[position + offset][channel] * float(filter_data["weights"][offset * 3 + channel])
            pooled.append(max(0.0, value))

    def linear(head: dict[str, Any]) -> float:
        return float(head["bias"]) + sum(float(value) * float(weight) for value, weight in zip(pooled, head["weights"]))

    output: dict[str, list[float]] = {}
    for key, channel in (("ndvi", 0), ("ndwi", 1)):
        values = []
        blends = model.get("indexBlend", {}).get(key, [])
        mixes = model.get("indexBaselineMix", {}).get(key, [])
        for horizon, head in enumerate(model["indexHeads"][key]):
            persistence = inputs[-1][channel]
            seasonal = persistence if horizon == 0 else inputs[horizon][channel]
            persistence_mix = float(mixes[horizon]) if horizon < len(mixes) else 0.0
            blend = float(blends[horizon]) if horizon < len(blends) else 1.0
            baseline = seasonal * (1 - persistence_mix) + persistence * persistence_mix
            values.append(clamp(baseline + blend * linear(head)))
        output[key] = values
    return output


def forecast_index(observations: list[dict[str, Any]]) -> dict[str, Any]:
    history_dates = [item["date"] for item in observations]
    history = {key: [item["indices"][key] for item in observations] for key in ("ndvi", "ndwi")}
    last_date = datetime.fromisoformat(history_dates[-1].replace("Z", "+00:00")).date()
    future_dates = [iso(add_months(last_date, index + 1)) for index in range(6)]
    accepted = bool(FORECAST_MODEL and str(FORECAST_MODEL.get("metadata", {}).get("status", "")).startswith("accepted"))
    future = predict_forecast_cnn(forecast_sequence(observations)) if accepted else {key: [round(history[key][-1], 3)] * 6 for key in ("ndvi", "ndwi")}
    trajectories = {}
    for key in ("ndvi", "ndwi"):
        values = future[key]
        rmse = [float(item.get("rmse", 0)) for item in (FORECAST_MODEL or {}).get("metrics", {}).get("index", {}).get(key, [])]
        trajectories[key] = {"history": history[key][-12:], "future": [round(value, 3) for value in values], "interval": [{"value": round(value, 3), "lower": round(clamp(value - 1.96 * (rmse[index] if index < len(rmse) else 0)), 3), "upper": round(clamp(value + 1.96 * (rmse[index] if index < len(rmse) else 0)), 3)} for index, value in enumerate(values)], "label": key.upper(), "description": "Satellite index"}
    return {"trajectories": trajectories, "historyDates": history_dates[-12:], "futureDates": future_dates, "droughtForecast": {1: None, 3: None, 6: None}, "modelStatus": "accepted-trained-cnn" if accepted else "python-backend-persistence-fallback", "inputCoverage": "complete" if len(observations) >= 12 else "partial-validity-masked", "inputObservations": len(observations), "uncertainty": "95% empirical error bands derived from held-out RMSE" if accepted else "Future index trajectory is retained as a conservative persistence fallback."}


def in_glofas_region(lat: float, lon: float) -> bool:
    return GLOFAS_BOUNDS["south"] <= lat <= GLOFAS_BOUNDS["north"] and GLOFAS_BOUNDS["west"] <= lon <= GLOFAS_BOUNDS["east"]


def glofas_probability(features: list[float], lat: float, lon: float) -> float | None:
    if GLOFAS_MODEL is None or not in_glofas_region(lat, lon) or len(features) != 10 or not all(math.isfinite(float(value)) for value in features):
        return None
    model = GLOFAS_MODEL["model"]
    return float(model.predict_proba(np.asarray([features], dtype=float))[0, 1])


def model_linked_hazards(observations: list[dict[str, Any]], lat: float, lon: float, live_features: list[float] | None) -> dict[str, Any]:
    current = flood_cnn_inference(observations[-1]["spatialSamples"])
    probability = glofas_probability(live_features or [], lat, lon)
    horizons = []
    for month in (1, 2, 3, 4, 5, 6):
        horizons.append({"month": month, "date": iso(add_months(datetime.now(timezone.utc).date(), month)), "ndvi": observations[-1]["indices"]["ndvi"], "ndwi": observations[-1]["indices"]["ndwi"], "flood": probability if month == 1 else None, "drought": None})
    return {"currentFlood": {**current, "status": "observed-flood-water-evidence"} if current else None, "horizons": horizons, "connection": {"ndviDelta": 0, "ndwiDelta": 0, "floodExplanation": "The validated +1-month GloFAS model is used only when a current operational ensemble feature vector is supplied for this regional coverage area.", "droughtExplanation": "No live drought climate input is configured in the Python backend."}, "validation": {"flood": {"metrics": (GLOFAS_METRICS or {}).get("testMetrics"), "scope": "Validated +1-month GloFAS regional flood outlook."}, "drought": {"metrics": DROUGHT_MODEL.get("metrics"), "deployment": DROUGHT_MODEL.get("deployment"), "status": DROUGHT_MODEL.get("metadata", {}).get("status")}}, "warning": None if probability is not None else "A current operational GloFAS ensemble input is required for the +1-month flood outlook."}


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    coordinates = payload.get("coordinates") or {}
    lat, lon = float(coordinates.get("lat", "nan")), float(coordinates.get("lon", "nan"))
    if not math.isfinite(lat) or not math.isfinite(lon) or abs(lat) > 90 or abs(lon) > 180:
        raise ValueError("A valid map location is required")
    end = parse_date(payload.get("analysisDate"), date.today())
    if end > date.today():
        raise ValueError("Analysis date cannot be in the future; select today or an earlier observation date")
    start = add_months(end, -11)
    key = f"{lat:.4f}:{lon:.4f}:{start}:{end}"
    hit = cached(analysis_cache, key)
    if hit is not None:
        return hit
    observations = satellite_observations(lat, lon, start, end)
    origin = min(observations, key=lambda item: abs((datetime.fromisoformat(item["date"].replace("Z", "+00:00")).date() - end).days))
    if abs((datetime.fromisoformat(origin["date"].replace("Z", "+00:00")).date() - end).days) > CAPTURE_TOLERANCE_DAYS:
        raise ValueError(f"No valid analysis-date capture was found within {CAPTURE_TOLERANCE_DAYS} days")
    index_forecast = forecast_index(observations)
    live_features = payload.get("glofasFeatures") if isinstance(payload.get("glofasFeatures"), list) else None
    hazards = model_linked_hazards(observations, lat, lon, live_features)
    data = {"indices": origin["indices"], "hazards": hazards, "modelValidation": {**FLOOD_MODEL.get("metrics", {}), **FLOOD_MODEL.get("metadata", {})}, "droughtValidation": {"metrics": DROUGHT_MODEL.get("metrics"), "deployment": DROUGHT_MODEL.get("deployment"), **DROUGHT_MODEL.get("metadata", {})}, "forecast": index_forecast, "imagery": {"observed": {"url": f"/api/satellite-image?collection={origin['collection']}&item={origin['id']}&lat={lat}&lon={lon}", "date": origin["date"], "requestedDate": iso(end), "offsetDays": abs((datetime.fromisoformat(origin["date"].replace("Z", "+00:00")).date() - end).days), "cloudCover": origin["cloudCover"]}}, "source": {"name": "Copernicus Sentinel-2", "provider": "Microsoft Planetary Computer", "observations": len(observations), "targetObservations": TARGET_OBSERVATIONS, "expectedSpatialSamples": 9, "minimumSpatialSamples": min(item["sampleCount"] for item in observations), "maximumSpatialSamples": max(item["sampleCount"] for item in observations), "aggregation": "median", "dispersion": origin["dispersion"], "analysisCoverage": round(sum(item["sampleCount"] for item in observations) / (TARGET_OBSERVATIONS * 9) * 100)}}
    cache_set(analysis_cache, key, data, 30 * 60)
    return data


@app.get("/api/geocode")
def api_geocode() -> Response:
    try:
        return jsonify({"results": geocode(request.args.get("q", ""))})
    except RuntimeError as error:
        return jsonify({"error": str(error)}), 503


@app.get("/api/overview")
def api_overview() -> Response:
    return jsonify({"modelValidation": {**FLOOD_MODEL.get("metrics", {}), **FLOOD_MODEL.get("metadata", {})}, "indices": ["NDVI", "NDWI"], "coverage": "global", "selectionModes": ["place search", "coordinates", "map click", "draggable pin"], "glofasFloodCoverage": GLOFAS_BOUNDS, "glofasModelLoaded": GLOFAS_MODEL is not None})


@app.post("/api/analyze")
def api_analyze() -> Response:
    try:
        return jsonify(analyze(request.get_json(silent=True) or {}))
    except ValueError as error:
        return jsonify({"error": str(error)}), 422
    except RuntimeError as error:
        return jsonify({"error": str(error)}), 502


@app.post("/api/flood-outlook")
def api_flood_outlook() -> Response:
    payload = request.get_json(silent=True) or {}
    coordinates = payload.get("coordinates") or {}
    try:
        lat, lon = float(coordinates.get("lat")), float(coordinates.get("lon"))
    except (TypeError, ValueError):
        return jsonify({"error": "A valid map location is required"}), 400
    if not in_glofas_region(lat, lon):
        return jsonify({"error": "The validated GloFAS flood model covers only the trained Bihar/eastern India region.", "coverage": GLOFAS_BOUNDS}), 422
    probability = glofas_probability(payload.get("features", []), lat, lon)
    if probability is None:
        return jsonify({"error": "Ten current GloFAS ensemble features are required."}), 400
    return jsonify({"month": 1, "probability": probability, "threshold": float(GLOFAS_MODEL["threshold"]), "status": "elevated" if probability >= float(GLOFAS_MODEL["threshold"]) else "lower", "metrics": (GLOFAS_METRICS or {}).get("testMetrics")})


@app.get("/api/satellite-image")
def satellite_image() -> Response:
    item = request.args.get("item", "")
    collection = request.args.get("collection", "sentinel-2-l2a")
    try:
        lat, lon = float(request.args.get("lat", "nan")), float(request.args.get("lon", "nan"))
    except ValueError:
        lat = lon = float("nan")
    if collection not in SENTINEL_COLLECTIONS or not item.replace("_", "").replace("-", "").isalnum() or not math.isfinite(lat) or not math.isfinite(lon):
        return jsonify({"error": "Invalid imagery request"}), 400
    lon_radius = IMAGE_RADIUS_DEGREES / max(0.25, math.cos(math.radians(lat)))
    endpoint = f"{DATA_API}/bbox/{lon-lon_radius},{lat-IMAGE_RADIUS_DEGREES},{lon+lon_radius},{lat+IMAGE_RADIUS_DEGREES}/720x480.png"
    response = fetch_with_retry("GET", endpoint, params={"collection": collection, "item": item, "assets": "visual", "asset_bidx": "visual|1,2,3", "nodata": "0"})
    if not response.ok:
        return jsonify({"error": "Satellite image unavailable"}), 502
    return Response(response.content, content_type=response.headers.get("content-type", "image/png"), headers={"Cache-Control": "public, max-age=1800"})


@app.get("/")
def index() -> Response:
    return send_from_directory(PUBLIC, "index.html")


@app.get("/<path:filename>")
def static_files(filename: str) -> Response:
    return send_from_directory(PUBLIC, filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "8501")), debug=False)
