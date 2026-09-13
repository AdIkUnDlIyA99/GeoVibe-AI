const http = require("node:http");
const fs = require("node:fs");
const path = require("node:path");
const { Readable } = require("node:stream");
const { buildPassport } = require("./src/analysis/passport");
const { forecastSeries } = require("./src/analysis/timeseries");
const { probability: droughtProbability } = require("./src/ml/drought-cnn");
const { inferGrid } = require("./src/ml/flood-cnn");
const { predictForecast } = require("./src/ml/forecast-cnn");

const ROOT = __dirname;
const PUBLIC = path.join(ROOT, "public");
const FLOOD_MODEL_PATH = path.join(ROOT, "models", "flood-cnn.json");
const DROUGHT_MODEL_PATH = path.join(ROOT, "models", "drought-cnn.json");
const FORECAST_MODEL_PATH = path.join(ROOT, "models", "forecast-cnn.json");
const geocodeCache = new Map();
const satelliteCache = new Map();
const STAC_SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search";
const DATA_API = "https://planetarycomputer.microsoft.com/api/data/v1/item";
const SENTINEL_COLLECTION = "sentinel-2-l2a";
const CACHE_LIMIT = 100;
const TARGET_OBSERVATIONS = 12;
const CAPTURE_TOLERANCE_DAYS = 40;
const IMAGE_RADIUS_DEGREES = 0.025;
const SAMPLE_RADIUS_DEGREES = 0.018;
const SAMPLE_GRID_SIZE = 9;
let lastGeocodeAt = 0;
if (!fs.existsSync(FLOOD_MODEL_PATH)) throw new Error("Missing real-data model. Run npm run train first.");
if (!fs.existsSync(DROUGHT_MODEL_PATH)) throw new Error("Missing drought model artifact.");
function loadFloodModel() { return JSON.parse(fs.readFileSync(FLOOD_MODEL_PATH, "utf8")); }
function loadDroughtModel() { return JSON.parse(fs.readFileSync(DROUGHT_MODEL_PATH, "utf8")); }
function loadForecastModel() {
  if (!fs.existsSync(FORECAST_MODEL_PATH)) return null;
  const model = JSON.parse(fs.readFileSync(FORECAST_MODEL_PATH, "utf8"));
  return model.metadata?.status === "accepted" ? model : null;
}
function json(res, status, payload) {
  res.writeHead(status, { "Content-Type": "application/json", "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer" });
  res.end(JSON.stringify(payload));
}
function body(req) {
  return new Promise((resolve, reject) => {
    let raw = "";
    let settled = false;
    req.on("data", (chunk) => {
      if (settled) return;
      raw += chunk;
      if (Buffer.byteLength(raw) > 1e6) {
        settled = true;
        reject(new Error("Payload too large"));
        req.destroy();
      }
    });
    req.on("end", () => {
      if (settled) return;
      try { resolve(raw ? JSON.parse(raw) : {}); } catch (error) { reject(error); }
    });
  });
}
function cacheSet(cache, key, value) {
  cache.delete(key);
  cache.set(key, value);
  while (cache.size > CACHE_LIMIT) cache.delete(cache.keys().next().value);
}
async function fetchWithRetry(url, options = {}, attempts = 3) {
  let lastError;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      const response = await fetch(url, { ...options, signal: options.signal || AbortSignal.timeout(25000) });
      if (response.ok || response.status < 500 || attempt === attempts - 1) return response;
      lastError = new Error(`Remote service returned ${response.status}`);
    } catch (error) {
      lastError = error;
      if (attempt === attempts - 1) throw error;
    }
    await new Promise((resolve) => setTimeout(resolve, 350 * 2 ** attempt));
  }
  throw lastError;
}
async function geocode(query) {
  const key = query.trim().toLowerCase();
  if (!key) return [];
  const cached = geocodeCache.get(key);
  if (cached?.expires > Date.now()) return cached.results;
  if (cached) geocodeCache.delete(key);
  const waitMs = Math.max(0, 1100 - (Date.now() - lastGeocodeAt));
  if (waitMs) await new Promise((resolve) => setTimeout(resolve, waitMs));
  const endpoint = new URL("https://nominatim.openstreetmap.org/search");
  endpoint.searchParams.set("q", query);
  endpoint.searchParams.set("format", "jsonv2");
  endpoint.searchParams.set("limit", "5");
  endpoint.searchParams.set("addressdetails", "1");
  lastGeocodeAt = Date.now();
  const response = await fetchWithRetry(endpoint, {
    headers: {
      "Accept-Language": "en",
      "User-Agent": "GeoVibe-AI/1.0 academic-environmental-monitoring-prototype"
    }
  });
  if (!response.ok) throw new Error(`Location search unavailable (${response.status})`);
  const results = (await response.json()).map((item) => ({
    name: item.display_name,
    lat: Number(item.lat),
    lon: Number(item.lon),
    boundingBox: item.boundingbox?.map(Number) || null
  }));
  cacheSet(geocodeCache, key, { expires: Date.now() + 24 * 60 * 60 * 1000, results });
  return results;
}
function clamp(value, min = -1, max = 1) { return Math.max(min, Math.min(max, value)); }
function addDays(date, days) { return new Date(date.getTime() + days * 86400000); }
function addMonths(date, months) { const result = new Date(date); result.setUTCMonth(result.getUTCMonth() + months); return result; }
function validDate(value, fallback) {
  const date = new Date(`${value || fallback}T12:00:00Z`);
  return Number.isNaN(date.getTime()) ? new Date(`${fallback}T12:00:00Z`) : date;
}
function targetDates(start, end, count = TARGET_OBSERVATIONS) {
  const span = Math.max(1, end.getTime() - start.getTime());
  return Array.from({ length: count }, (_, index) => new Date(start.getTime() + span * index / (count - 1)));
}
function forecast(observations) {
  const dates = observations.map((item) => item.date);
  const trainedModel = loadForecastModel();
  if (trainedModel && observations.length >= 12) {
    const sequence = observations.slice(-12).map((item) => ({ ...item.indices, valid: true }));
    const prediction = predictForecast(trainedModel, sequence);
    const lastDate = new Date(dates.at(-1));
    const futureDates = Array.from({ length: 6 }, (_, index) => addMonths(lastDate, index + 1).toISOString());
    const makeSeries = (key, label, description) => ({
      history: observations.slice(-12).map((item) => item.indices[key]),
      future: prediction.index[key].map((value) => +value.toFixed(3)),
      interval: prediction.index[key].map((value, index) => {
        const rmse = Number(trainedModel.metrics?.index?.[key]?.[index]?.rmse || 0);
        return { value: +value.toFixed(3), lower: +clamp(value - 1.96 * rmse).toFixed(3), upper: +clamp(value + 1.96 * rmse).toFixed(3) };
      }),
      label, description, method: trainedModel.architecture
    });
    return {
      trajectories: { ndvi: makeSeries("ndvi", "NDVI", "Vegetation health"), ndwi: makeSeries("ndwi", "NDWI", "Surface water") },
      historyDates: dates.slice(-12), futureDates,
      droughtForecast: { 1: prediction.drought[0], 3: prediction.drought[1], 6: prediction.drought[2] },
      modelStatus: "accepted-trained-cnn",
      uncertainty: "95% empirical error bands derived from geographically and temporally held-out RMSE"
    };
  }
  const makeSeries = (key, label, description) => {
    const history = observations.map((item) => item.indices[key]);
    const projected = forecastSeries(dates, history);
    return { history, future: projected.future.map((item) => item.value), interval: projected.future, label, description, method: projected.method };
  };
  const trajectories = {
    ndvi: makeSeries("ndvi", "NDVI", "Vegetation health"),
    ndwi: makeSeries("ndwi", "NDWI", "Surface water")
  };
  return {
    trajectories,
    historyDates: dates,
    futureDates: forecastSeries(dates, observations.map((item) => item.indices.ndvi)).futureDates,
    uncertainty: "95% prediction interval including fitted-parameter uncertainty; projection is not a weather forecast",
    modelStatus: "statistical-fallback",
    anomaly: Math.min(...trajectories.ndvi.history) < 0.2
  };
}

function modelLinkedHazards(observations, indexForecast, floodModel, droughtModel) {
  const origin = observations.at(-1);
  const observedSequence = observations.map((item) => ({ ...item.indices, valid: true })).slice(-12);
  while (observedSequence.length < 12) observedSequence.unshift({ ...observedSequence[0], valid: false });
  const projectedSequence = observedSequence.slice();
  const horizons = indexForecast.futureDates.map((date, index) => {
    const ndvi = indexForecast.trajectories.ndvi.future[index];
    const ndwi = indexForecast.trajectories.ndwi.future[index];
    projectedSequence.push({ ndvi, ndwi, valid: true });
    const shiftedGrid = origin.spatialSamples.map((cell) => ({
      ...cell,
      ndvi: clamp(cell.ndvi + ndvi - origin.indices.ndvi),
      ndwi: clamp(cell.ndwi + ndwi - origin.indices.ndwi)
    }));
    const flood = inferGrid(floodModel, shiftedGrid);
    return {
      month: index + 1,
      date,
      ndvi,
      ndwi,
      flood: flood ? +flood.temporaryWaterFraction.toFixed(4) : null,
      drought: indexForecast.droughtForecast?.[index + 1] ?? (droughtModel.metadata?.status === "trained" ? +droughtProbability(droughtModel, projectedSequence.slice(-12)).toFixed(4) : null)
    };
  });
  const final = horizons.at(-1);
  const ndviDelta = +(final.ndvi - origin.indices.ndvi).toFixed(3);
  const ndwiDelta = +(final.ndwi - origin.indices.ndwi).toFixed(3);
  return {
    horizons,
    connection: {
      ndviDelta,
      ndwiDelta,
      floodExplanation: `Projected NDWI ${ndwiDelta >= 0 ? "rises" : "falls"} by ${Math.abs(ndwiDelta).toFixed(2)}; the flood-water CNN evaluates that projected spectral state.`,
      droughtExplanation: `Projected NDVI ${ndviDelta >= 0 ? "rises" : "falls"} by ${Math.abs(ndviDelta).toFixed(2)} and NDWI ${ndwiDelta >= 0 ? "rises" : "falls"}; the temporal CNN evaluates the rolling 12-month sequence.`
    },
    validation: {
      flood: { ...floodModel.metrics, scope: "Current flood-water detection; future susceptibility is not separately validated." },
      drought: { ...droughtModel.metrics, status: droughtModel.metadata?.status, scope: "Current SPEI drought classification; future susceptibility is not separately validated." }
    },
    warning: indexForecast.modelStatus === "accepted-trained-cnn" ? "Research forecast from an accepted held-out temporal CNN; not an official event warning." : "Experimental six-month susceptibility generated from forecast spectral states; not an event or weather forecast."
  };
}
async function searchSentinelItems(lat, lon, start, end) {
  const response = await fetchWithRetry(STAC_SEARCH, {
    method: "POST",
    headers: { "content-type": "application/json", "user-agent": "GeoVibe-AI academic-project" },
    body: JSON.stringify({
      collections: [SENTINEL_COLLECTION],
      bbox: [lon - 0.001, lat - 0.001, lon + 0.001, lat + 0.001],
      datetime: `${addDays(start, -35).toISOString()}/${addDays(end, 35).toISOString()}`,
      limit: 100,
      query: { "eo:cloud_cover": { lt: 45 } }
    })
  });
  if (!response.ok) throw new Error(`Sentinel catalog request failed (${response.status})`);
  const data = await response.json();
  return data.features || [];
}
async function sampleSentinelPoint(item, lat, lon) {
  const endpoint = new URL(`${DATA_API}/point/${lon},${lat}`);
  endpoint.searchParams.set("collection", SENTINEL_COLLECTION);
  endpoint.searchParams.set("item", item.id);
  for (const asset of ["B03", "B04", "B08", "SCL"]) endpoint.searchParams.append("assets", asset);
  const response = await fetchWithRetry(endpoint, {}, 2);
  if (!response.ok) return null;
  const point = await response.json();
  const bands = Object.fromEntries(point.band_names.map((name, index) => [name.split("_")[0], Number(point.values[index])]));
  if (!["B03", "B04", "B08"].every((name) => Number.isFinite(bands[name]) && bands[name] > 0)) return null;
  if (Number.isFinite(bands.SCL) && ![2, 4, 5, 6, 7].includes(Math.round(bands.SCL))) return null;
  const ratio = (a, b) => +(Math.abs(a + b) < 1e-9 ? 0 : (a - b) / (a + b)).toFixed(3);
  const indices = {
    ndvi: ratio(bands.B08, bands.B04),
    ndwi: ratio(bands.B03, bands.B08)
  };
  return indices;
}
async function mapLimit(values, limit, worker) {
  const results = Array(values.length);
  let next = 0;
  async function run() {
    while (next < values.length) {
      const index = next++;
      results[index] = await worker(values[index], index);
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, values.length) }, run));
  return results;
}
function median(values) {
  const sorted = values.slice().sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}
async function sampleSentinelItem(item, lat, lon) {
  const lonRadius = SAMPLE_RADIUS_DEGREES / Math.max(0.25, Math.cos(lat * Math.PI / 180));
  const offsets = [-SAMPLE_RADIUS_DEGREES, 0, SAMPLE_RADIUS_DEGREES];
  const lonOffsets = [-lonRadius, 0, lonRadius];
  const points = offsets.flatMap((latOffset) => lonOffsets.map((lonOffset) => [lat + latOffset, lon + lonOffset]));
  const samples = (await mapLimit(points, 5, async ([pointLat, pointLon], cell) => {
    try {
      const sample = await sampleSentinelPoint(item, pointLat, pointLon);
      return sample ? { ...sample, cell, lat: pointLat, lon: pointLon } : null;
    } catch {
      return null;
    }
  })).filter(Boolean);
  if (samples.length < 5) return null;
  const indices = { ndvi: +median(samples.map((sample) => sample.ndvi)).toFixed(3), ndwi: +median(samples.map((sample) => sample.ndwi)).toFixed(3) };
  const dispersion = {
    ndvi: +(Math.max(...samples.map((sample) => sample.ndvi)) - Math.min(...samples.map((sample) => sample.ndvi))).toFixed(3),
    ndwi: +(Math.max(...samples.map((sample) => sample.ndwi)) - Math.min(...samples.map((sample) => sample.ndwi))).toFixed(3)
  };
  return { id: item.id, date: item.properties.datetime, cloudCover: +Number(item.properties["eo:cloud_cover"] || 0).toFixed(1), indices, features: [indices.ndvi, indices.ndwi], sampleCount: samples.length, dispersion, spatialSamples: samples };
}
async function satelliteObservations(lat, lon, start, end) {
  const items = await searchSentinelItems(lat, lon, start, end);
  if (!items.length) throw new Error("No Sentinel-2 coverage was found for this location and date range");
  const targets = targetDates(start, end);
  const choices = targets.map((target) => items.slice().sort((a, b) => {
    const score = (item) => Math.abs(new Date(item.properties.datetime) - target) / 86400000 + Number(item.properties["eo:cloud_cover"] || 100) * 2;
    return score(a) - score(b);
  }).slice(0, 3));
  const samples = await mapLimit(choices, 4, async (candidates, targetIndex) => {
    for (const item of candidates) {
      const sample = await sampleSentinelItem(item, lat, lon);
      if (sample) return { ...sample, targetDate: targets[targetIndex].toISOString(), targetOffsetDays: Math.round(Math.abs(new Date(sample.date) - targets[targetIndex]) / 86400000) };
    }
    return null;
  });
  const unique = [...new Map(samples.filter(Boolean).map((sample) => [sample.id, sample])).values()]
    .sort((a, b) => new Date(a.date) - new Date(b.date));
  if (unique.length < 2) throw new Error("Fewer than two usable Sentinel-2 observations were returned. Check internet access or choose a nearby date/location.");
  return unique;
}
function selectCapture(observations, requestedDate, label) {
  const ranked = observations.map((sample) => ({ sample, offsetDays: Math.round(Math.abs(new Date(sample.date) - requestedDate) / 86400000) }))
    .sort((a, b) => a.offsetDays - b.offsetDays);
  if (!ranked.length || ranked[0].offsetDays > CAPTURE_TOLERANCE_DAYS) {
    const error = new Error(`No valid ${label} capture was found within ${CAPTURE_TOLERANCE_DAYS} days of ${requestedDate.toISOString().slice(0, 10)}`);
    error.status = 422;
    throw error;
  }
  return { ...ranked[0].sample, requestedDate: requestedDate.toISOString(), offsetDays: ranked[0].offsetDays };
}
function imagePath(sample, lat, lon) {
  return `/api/satellite-image?item=${encodeURIComponent(sample.id)}&lat=${lat}&lon=${lon}`;
}
async function analyze(payload) {
  const lat = Number(payload.coordinates?.lat);
  const lon = Number(payload.coordinates?.lon);
  if (!Number.isFinite(lat) || !Number.isFinite(lon) || Math.abs(lat) > 90 || Math.abs(lon) > 180) {
    const error = new Error("A valid map location is required");
    error.status = 400;
    throw error;
  }
  const end = validDate(payload.analysisDate, new Date().toISOString().slice(0, 10));
  if (end.toISOString().slice(0, 10) > new Date().toISOString().slice(0, 10)) {
    const error = new Error("Analysis date cannot be in the future; select today or an earlier observation date");
    error.status = 422;
    throw error;
  }
  const start = addMonths(end, -11);
  const cacheKey = `${lat.toFixed(4)}:${lon.toFixed(4)}:${start.toISOString().slice(0, 10)}:${end.toISOString().slice(0, 10)}`;
  const cached = satelliteCache.get(cacheKey);
  if (cached && cached.expires > Date.now()) return cached.data;
  const observations = await satelliteObservations(lat, lon, start, end);
  const originSample = selectCapture(observations, end, "analysis-date");
  const floodModel = loadFloodModel();
  const droughtModel = loadDroughtModel();
  const analysisCoverage = Math.round(observations.reduce((sum, item) => sum + item.sampleCount, 0) / (TARGET_OBSERVATIONS * SAMPLE_GRID_SIZE) * 100);
  const indexForecast = forecast(observations);
  const hazardForecast = modelLinkedHazards(observations, indexForecast, floodModel, droughtModel);
  const data = {
    indices: originSample.indices,
    hazards: hazardForecast,
    modelValidation: { ...floodModel.metrics, ...floodModel.metadata },
    droughtValidation: { ...droughtModel.metrics, ...droughtModel.metadata },
    forecast: indexForecast,
    imagery: {
      observed: { url: imagePath(originSample, lat, lon), date: originSample.date, requestedDate: originSample.requestedDate, offsetDays: originSample.offsetDays, cloudCover: originSample.cloudCover }
    },
    source: {
      name: "Copernicus Sentinel-2 L2A",
      provider: "Microsoft Planetary Computer",
      observations: observations.length,
      targetObservations: TARGET_OBSERVATIONS,
      expectedSpatialSamples: SAMPLE_GRID_SIZE,
      minimumSpatialSamples: Math.min(...observations.map((item) => item.sampleCount)),
      maximumSpatialSamples: Math.max(...observations.map((item) => item.sampleCount)),
      aggregation: "median",
      dispersion: originSample.dispersion,
      analysisCoverage
    }
  };
  cacheSet(satelliteCache, cacheKey, { expires: Date.now() + 30 * 60 * 1000, data });
  return data;
}

const server = http.createServer(async (req, res) => {
  try {
    const requestUrl = new URL(req.url, "http://127.0.0.1");
    if (requestUrl.pathname === "/api/geocode" && req.method === "GET") {
      return json(res, 200, { results: await geocode(requestUrl.searchParams.get("q") || "") });
    }
    if (requestUrl.pathname === "/api/overview" && req.method === "GET") {
      const model = loadFloodModel();
      return json(res, 200, { modelValidation: { ...model.metrics, ...model.metadata }, indices: ["NDVI", "NDWI"], coverage: "global", selectionModes: ["place search", "coordinates", "map click", "draggable pin"] });
    }
    if (requestUrl.pathname === "/api/analyze" && req.method === "POST") return json(res, 200, await analyze(await body(req)));
    if (requestUrl.pathname === "/api/satellite-image" && req.method === "GET") {
      const item = requestUrl.searchParams.get("item") || "";
      const lat = Number(requestUrl.searchParams.get("lat"));
      const lon = Number(requestUrl.searchParams.get("lon"));
      if (!/^[A-Za-z0-9_-]+$/.test(item) || !Number.isFinite(lat) || !Number.isFinite(lon) || Math.abs(lat) > 90 || Math.abs(lon) > 180) return json(res, 400, { error: "Invalid imagery request" });
      const latRadius = IMAGE_RADIUS_DEGREES;
      const lonRadius = latRadius / Math.max(0.25, Math.cos(lat * Math.PI / 180));
      const endpoint = new URL(`${DATA_API}/bbox/${lon-lonRadius},${lat-latRadius},${lon+lonRadius},${lat+latRadius}/720x480.png`);
      endpoint.searchParams.set("collection", SENTINEL_COLLECTION);
      endpoint.searchParams.set("item", item);
      endpoint.searchParams.set("assets", "visual");
      endpoint.searchParams.set("asset_bidx", "visual|1,2,3");
      endpoint.searchParams.set("nodata", "0");
      const imageResponse = await fetchWithRetry(endpoint, {}, 2);
      if (!imageResponse.ok || !imageResponse.body) return json(res, 502, { error: "Satellite image unavailable" });
      res.writeHead(200, { "Content-Type": imageResponse.headers.get("content-type") || "image/png", "Cache-Control": "public, max-age=1800" });
      return Readable.fromWeb(imageResponse.body).pipe(res);
    }
    if (requestUrl.pathname === "/api/train") return json(res, 404, { error: "Model training is available only through the local npm run train command" });
    const requestPath = requestUrl.pathname === "/" ? "/index.html" : requestUrl.pathname;
    const filePath = path.normalize(path.join(PUBLIC, requestPath));
    const relativePath = path.relative(PUBLIC, filePath);
    if (relativePath.startsWith("..") || path.isAbsolute(relativePath) || !fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) return json(res, 404, { error: "Not found" });
    const types = { ".html": "text/html; charset=utf-8", ".css": "text/css", ".js": "text/javascript", ".svg": "image/svg+xml" };
    res.writeHead(200, { "Content-Type": types[path.extname(filePath)] || "application/octet-stream", "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer" });
    fs.createReadStream(filePath).pipe(res);
  } catch (error) { json(res, error.status || 500, { error: error.message }); }
});

if (require.main === module) {
  const port = Number(process.env.PORT || 8501);
  server.listen(port, "127.0.0.1", () => console.log(`GeoVibe AI running at http://127.0.0.1:${port}`));
}

module.exports = { forecast, modelLinkedHazards, selectCapture, server, targetDates };
