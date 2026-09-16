const $ = (selector) => document.querySelector(selector);
let worldMap;
let locationMarker;
let searchMarkers = [];
let selectedPoint = { lat: 28.6139, lon: 77.2090 };
let latestResult;
let searchTimer;
let searchRequest = 0;

const hasNumber = (value) => value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value));
const percentage = (value) => hasNumber(value) ? `${(Number(value) * 100).toFixed(1)}%` : "Unavailable";
const indexValue = (value) => hasNumber(value) ? Number(value).toFixed(2) : "--";

function coordinateLabel(lat, lon) {
  return `${Math.abs(lat).toFixed(4)}° ${lat >= 0 ? "N" : "S"} / ${Math.abs(lon).toFixed(4)}° ${lon >= 0 ? "E" : "W"}`;
}

function setMapSelection(lat, lon, label, move = true) {
  selectedPoint = { lat: Number(lat), lon: Number(lon) };
  $("#coordinates").textContent = coordinateLabel(selectedPoint.lat, selectedPoint.lon);
  if (label) $("#location").value = label;
  if (!worldMap) return;
  if (!locationMarker) {
    const icon = L.divIcon({ className: "", html: '<span class="pin-icon"></span>', iconSize: [28, 28], iconAnchor: [12, 26] });
    locationMarker = L.marker([lat, lon], { draggable: true, autoPan: true, icon }).addTo(worldMap);
    locationMarker.on("dragend", (event) => {
      const point = event.target.getLatLng();
      setMapSelection(point.lat, point.lng, `Pinned location (${point.lat.toFixed(5)}, ${point.lng.toFixed(5)})`, false);
    });
  } else locationMarker.setLatLng([lat, lon]);
  if (move) worldMap.flyTo([lat, lon], Math.max(7, worldMap.getZoom()), { duration: 0.65 });
}

function initMap() {
  if (!window.L) {
    $("#worldMap").textContent = "Map unavailable. Enter coordinates or search for a place.";
    return;
  }
  worldMap = L.map("worldMap", { worldCopyJump: true, minZoom: 2 }).setView([20, 0], 2);
  const tiles = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}", { maxZoom: 19, attribution: "Tiles © Esri" }).addTo(worldMap);
  let fallbackUsed = false;
  tiles.on("tileerror", () => {
    if (fallbackUsed) return;
    fallbackUsed = true;
    worldMap.removeLayer(tiles);
    L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", { maxZoom: 19, attribution: "Tiles © Esri" }).addTo(worldMap);
  });
  worldMap.on("click", ({ latlng }) => setMapSelection(latlng.lat, latlng.lng, `Pinned location (${latlng.lat.toFixed(5)}, ${latlng.lng.toFixed(5)})`, false));
  setMapSelection(selectedPoint.lat, selectedPoint.lon, null, false);
}

function showSearchResults(results, message = "") {
  const panel = $("#searchResults");
  panel.hidden = false;
  panel.replaceChildren();
  searchMarkers.forEach((marker) => worldMap?.removeLayer(marker));
  searchMarkers = [];
  if (!results.length) {
    const status = document.createElement("div");
    status.className = "search-result";
    status.textContent = message || "No matching location found.";
    panel.append(status);
    return;
  }
  const selectResult = (result) => {
    setMapSelection(result.lat, result.lon, result.name);
    if (result.boundingBox && worldMap) worldMap.fitBounds([[result.boundingBox[0], result.boundingBox[2]], [result.boundingBox[1], result.boundingBox[3]]]);
    panel.hidden = true;
    searchMarkers.forEach((marker) => worldMap?.removeLayer(marker));
    searchMarkers = [];
    $("#coverageHint").textContent = "Location selected. Adjust the pin if needed.";
  };
  for (const [index, result] of results.entries()) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "search-result";
    button.setAttribute("role", "option");
    button.textContent = result.name;
    button.addEventListener("click", () => selectResult(result));
    panel.append(button);
    if (worldMap) {
      const icon = L.divIcon({ className: "", html: `<span class="search-pin">${index + 1}</span>`, iconSize: [28, 28], iconAnchor: [14, 14] });
      const marker = L.marker([result.lat, result.lon], { icon, title: result.name }).addTo(worldMap);
      marker.bindTooltip(result.name, { direction: "top", offset: [0, -12] });
      marker.on("click", () => selectResult(result));
      searchMarkers.push(marker);
    }
  }
  if (worldMap && searchMarkers.length) {
    const bounds = L.latLngBounds(results.map((result) => [result.lat, result.lon]));
    worldMap.fitBounds(bounds, { padding: [35, 35], maxZoom: 10 });
  }
}

async function searchLocation() {
  const query = $("#location").value.trim();
  const requestId = ++searchRequest;
  if (!query) return $("#searchResults").hidden = true;
  const coordinates = query.match(/^\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)\s*$/);
  if (coordinates) {
    const lat = Number(coordinates[1]), lon = Number(coordinates[2]);
    if (Math.abs(lat) <= 90 && Math.abs(lon) <= 180) return showSearchResults([{ lat, lon, name: `Use coordinates ${lat.toFixed(5)}, ${lon.toFixed(5)}` }]);
  }
  if (query.length < 3) return showSearchResults([], "Keep typing to search worldwide.");
  showSearchResults([], "Searching worldwide...");
  try {
    const response = await fetch(`/api/geocode?q=${encodeURIComponent(query)}`);
    if (!response.ok) throw new Error();
    const data = await response.json();
    if (requestId === searchRequest) showSearchResults(data.results || []);
  } catch {
    if (requestId === searchRequest) showSearchResults([], "Location search is temporarily unavailable. You can still use coordinates or the map pin.");
  }
}

function indexState(type, value) {
  if (type === "ndvi") {
    if (value >= 0.5) return "Dense vegetation signal";
    if (value >= 0.2) return "Moderate vegetation signal";
    if (value >= 0) return "Sparse vegetation signal";
    return "Low vegetation response";
  }
  if (value >= 0.2) return "Strong surface-water signal";
  if (value >= 0) return "Moist or mixed surface";
  return "Low surface-water signal";
}

function riskState(value) {
  if (!hasNumber(value)) return "Model abstained";
  if (value >= 0.7) return "Elevated susceptibility";
  if (value >= 0.4) return "Moderate susceptibility";
  return "Lower susceptibility";
}

function renderFutureHazards(prefix, horizons, validation, selectedMonth) {
  for (const month of [1, 3, 6]) {
    const horizon = horizons.find((item) => item.month === month);
    const probability = horizon?.[prefix];
    const projection = horizon?.[`${prefix}Projection`];
    const value = $(`#${prefix}${month}Value`);
    const state = $(`#${prefix}${month}State`);
    value.closest("[data-horizon]").hidden = month !== selectedMonth;
    if (hasNumber(probability)) {
      value.textContent = percentage(probability);
      value.classList.remove("unavailable");
      state.textContent = projection
        ? `Projected from +${projection.sourceMonth}M · ${riskState(probability)} · not independently validated`
        : riskState(probability);
    } else {
      value.textContent = "NOT AVAILABLE";
      value.classList.add("unavailable");
      state.textContent = validation?.[month] || "No validated future model";
    }
  }
}

function trendChart(forecast) {
  if (!forecast?.trajectories) return;
  const selected = $("#trajectoryMetric").value;
  const keys = selected === "all" ? ["ndvi", "ndwi"] : [selected];
  const colors = { ndvi: "#b9ff35", ndwi: "#45e8d8" };
  const width = 900, height = 300, left = 52, right = 45, top = 24, bottom = 40;
  const historyCount = forecast.trajectories.ndvi.history.length;
  const lastObserved = historyCount - 1;
  const last = lastObserved + forecast.trajectories.ndvi.future.length;
  const values = keys.flatMap((key) => [...forecast.trajectories[key].history, ...forecast.trajectories[key].future]);
  const dataMin = Math.min(...values), dataMax = Math.max(...values), padding = Math.max(0.08, (dataMax - dataMin) * 0.2);
  const min = Math.max(-1, dataMin - padding), max = Math.min(1, dataMax + padding);
  const x = (index) => left + index * (width - left - right) / Math.max(1, last);
  const y = (value) => top + (max - value) * (height - top - bottom) / Math.max(0.1, max - min);
  const points = (array, offset = 0) => array.map((value, index) => `${x(index + offset)},${y(value)}`).join(" ");
  const grid = Array.from({ length: 5 }, (_, index) => {
    const value = min + (max - min) * index / 4;
    return `<line x1="${left}" y1="${y(value)}" x2="${width-right}" y2="${y(value)}" class="chart-grid"/><text x="${left-10}" y="${y(value)+3}" text-anchor="end" class="chart-axis">${value.toFixed(2)}</text>`;
  }).join("");
  const series = keys.map((key) => {
    const item = forecast.trajectories[key];
    const future = [item.history.at(-1), ...item.future];
    return `<polyline points="${points(item.history)}" class="history-line" style="stroke:${colors[key]}"/><polyline points="${points(future,lastObserved)}" class="future-line" style="stroke:${colors[key]}"/><text x="${x(last)-3}" y="${y(item.future.at(-1))-8}" text-anchor="end" class="end-label" style="fill:${colors[key]}">${key.toUpperCase()} ${item.future.at(-1).toFixed(2)}</text>`;
  }).join("");
  $("#trend").innerHTML = `${grid}<line x1="${x(lastObserved)}" y1="${top}" x2="${x(lastObserved)}" y2="${height-bottom}" class="forecast-divider"/>${series}`;
}

function renderResult(data, place) {
  latestResult = { ...data, requestedLocation: place, coordinates: selectedPoint, generatedAt: new Date().toISOString() };
  const selectedMonth = Number($("#forecastHorizon").value);
  const ndviFuture = data.forecast?.trajectories?.ndvi?.future?.[selectedMonth - 1];
  const ndwiFuture = data.forecast?.trajectories?.ndwi?.future?.[selectedMonth - 1];
  $("#resultPlace").textContent = place;
  $("#resultMeta").textContent = `${coordinateLabel(selectedPoint.lat, selectedPoint.lon)} · ${$("#analysisDate").value} · +${selectedMonth}M horizon · ${data.source.observations} observations`;
  $("#ndviValue").textContent = indexValue(data.indices.ndvi);
  $("#ndviState").textContent = indexState("ndvi", data.indices.ndvi);
  $("#ndviFuture").textContent = indexValue(ndviFuture);
  $("#ndviForecastLabel").textContent = `+${selectedMonth} MONTH FORECAST`;
  const trainedIndexForecast = data.forecast.modelStatus === "accepted-trained-cnn";
  const indexModelNote = trainedIndexForecast
    ? `Accepted trained CNN${data.forecast.inputCoverage === "complete" ? "" : ` · ${data.forecast.inputObservations} observed captures with validity masking`}`
    : "Statistical fallback; trained model unavailable";
  $("#ndviForecastNote").textContent = indexModelNote;
  $("#ndwiValue").textContent = indexValue(data.indices.ndwi);
  $("#ndwiState").textContent = indexState("ndwi", data.indices.ndwi);
  $("#ndwiFuture").textContent = indexValue(ndwiFuture);
  $("#ndwiForecastLabel").textContent = `+${selectedMonth} MONTH FORECAST`;
  $("#ndwiForecastNote").textContent = indexModelNote;
  renderFutureHazards("flood", data.hazards.horizons, {
    1: "Future flood model not deployed",
    3: "Future flood model not deployed",
    6: "Future flood model not deployed"
  }, selectedMonth);
  $("#floodMethod").textContent = "Awaiting validated hydrological forecast";
  $("#floodMethodNote").textContent = "The trained Sen1Floods11 CNN detects present water only, so it is not reused as a future forecast.";
  const droughtValidation = Object.fromEntries([1, 3, 6].map((month, index) => {
    const metric = data.droughtValidation.metrics?.[index];
    const deployed = data.droughtValidation.deployment?.[index]?.accepted === true;
    return [month, deployed
      ? "Live climate predictors unavailable"
      : `Horizon failed validation${metric ? ` · BA ${percentage(metric.balancedAccuracy)} · F1 ${percentage(metric.f1)}` : ""}`];
  }));
  renderFutureHazards("drought", data.hazards.horizons, droughtValidation, selectedMonth);
  $("#droughtMethod").textContent = "SPEI-3 seasonal drought forecast";
  $("#droughtMethodNote").textContent = "Historical SPEI-3 plus seasonal temperature and precipitation drive +1M. When available, +1M is carried to +6M as a clearly labeled constant-risk projection; it is not a separately validated forecast.";
  trendChart(data.forecast);
}

function setLoader(active) {
  $("#analysisLoader").hidden = !active;
  document.body.style.overflow = active ? "hidden" : "";
}

function toast(message) {
  const element = $("#toast");
  element.textContent = message;
  element.classList.add("show");
  setTimeout(() => element.classList.remove("show"), 3200);
}

async function analyze() {
  const place = $("#location").value.trim() || "Selected world area";
  const date = $("#analysisDate").value;
  if (!date) return toast("Choose an analysis date first.");
  setLoader(true);
  const stages = ["Locating the selected place", "Matching twelve clear observations", "Running four analytical lenses", "Writing plain-language explanations"];
  const stepElements = [...document.querySelectorAll(".loader-steps span")];
  let stage = 0;
  const timer = setInterval(() => {
    stage = Math.min(stage + 1, stages.length - 1);
    $("#loaderStage").textContent = stages[stage];
    stepElements.forEach((element, index) => element.classList.toggle("active", index <= stage));
  }, 2600);
  try {
    const response = await fetch("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ analysisDate: date, location: place, coordinates: selectedPoint }) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Analysis could not be completed");
    renderResult(data, place);
    $("#inputView").hidden = true;
    $("#resultsView").hidden = false;
    window.scrollTo({ top: 0, behavior: "smooth" });
  } catch (error) {
    toast(error.message);
  } finally {
    clearInterval(timer);
    setLoader(false);
    stepElements.forEach((element, index) => element.classList.toggle("active", index === 0));
    $("#loaderStage").textContent = stages[0];
  }
}

$("#analyzeBtn").addEventListener("click", analyze);
$("#newAnalysis").addEventListener("click", () => {
  $("#resultsView").hidden = true;
  $("#inputView").hidden = false;
  setTimeout(() => worldMap?.invalidateSize(), 50);
  window.scrollTo({ top: 0, behavior: "smooth" });
});
$("#trajectoryMetric").addEventListener("change", () => trendChart(latestResult?.forecast));
$("#location").addEventListener("input", () => {
  clearTimeout(searchTimer);
  $("#coverageHint").textContent = "Choose a suggestion to position the map.";
  searchTimer = setTimeout(searchLocation, 450);
});
$("#location").addEventListener("keydown", (event) => {
  if (event.key === "Escape") $("#searchResults").hidden = true;
  if (event.key === "Enter") {
    event.preventDefault();
    const first = $("#searchResults [role='option']");
    if (first) first.click(); else searchLocation();
  }
});
document.addEventListener("click", (event) => { if (!event.target.closest(".aoi-control")) $("#searchResults").hidden = true; });

const today = new Date().toISOString().slice(0, 10);
$("#analysisDate").value = today;
$("#analysisDate").max = today;
initMap();
