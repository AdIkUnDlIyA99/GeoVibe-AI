const $ = (selector) => document.querySelector(selector);
let worldMap;
let locationMarker;
let selectedPoint = { lat: 28.6139, lon: 77.2090 };
let latestForecast;
let latestPassport;
let latestShadow;
let shadowMapLayers = [];
let trajectoryZoom = 1;
let trajectoryChartSize = { width: 900, height: 300 };
let trajectoryCenter = { x: 0.5, y: 0.5 };
let trajectoryDrag;
let locationSearchTimer;
let locationSearchRequest = 0;

function coordinateLabel(lat, lon) {
  const ns = lat >= 0 ? "N" : "S";
  const ew = lon >= 0 ? "E" : "W";
  return `${Math.abs(lat).toFixed(4)}° ${ns} · ${Math.abs(lon).toFixed(4)}° ${ew}`;
}

function setMapSelection(lat, lon, label, moveMap = true) {
  selectedPoint = { lat: Number(lat), lon: Number(lon) };
  clearShadowMap();
  $("#coordinates").textContent = coordinateLabel(selectedPoint.lat, selectedPoint.lon);
  if (label) $("#location").value = label;
  if (!worldMap) return;
  if (!locationMarker) {
    const icon = L.divIcon({ className: "", html: '<span class="pin-icon"></span>', iconSize: [28, 28], iconAnchor: [12, 26] });
    locationMarker = L.marker([lat, lon], { draggable: true, autoPan: true, icon, title: "Drag to select area" }).addTo(worldMap);
    locationMarker.on("dragend", (event) => {
      const point = event.target.getLatLng();
      setMapSelection(point.lat, point.lng, `Pinned location (${point.lat.toFixed(5)}, ${point.lng.toFixed(5)})`, false);
    });
  } else {
    locationMarker.setLatLng([lat, lon]);
  }
  if (moveMap) worldMap.flyTo([lat, lon], Math.max(worldMap.getZoom(), 8), { duration: 0.7 });
}

function clearShadowMap() {
  shadowMapLayers.forEach((layer) => worldMap?.removeLayer(layer));
  shadowMapLayers = [];
  const status = $("#twinMapStatus");
  if (status) status.innerHTML = "Awaiting spatial reference";
}

function renderShadowMap(shadow) {
  clearShadowMap();
  if (shadow?.status === "insufficient-controls") {
    $("#twinMapStatus").innerHTML = "Spatial reference unavailable";
    return;
  }
  if (!worldMap || !shadow?.controls?.length) return;
  const matchedTwins=shadow.referenceMode==="matched-twins";
  shadow.controls.forEach((control, index) => {
    const line = L.polyline([[selectedPoint.lat, selectedPoint.lon], [control.lat, control.lon]], { color: "#9fe52f", weight: 1, opacity: 0.55, dashArray: "3 7", interactive: false }).addTo(worldMap);
    const marker = L.circleMarker([control.lat, control.lon], { radius: 6, color: "#caff55", weight: 1.5, fillColor: "#071008", fillOpacity: 0.9 })
      .bindTooltip(`${matchedTwins ? "Twin" : "Local reference"} ${index + 1} · ${(control.weight * 100).toFixed(0)}% weight · fit ${control.matchScore.toFixed(3)}`, { direction: "top" }).addTo(worldMap);
    shadowMapLayers.push(line, marker);
  });
  $("#twinMapStatus").innerHTML = matchedTwins
    ? `<b>${shadow.controls.length}</b> ecological twins · fit ${shadow.matchQuality}/100`
    : `<b>${shadow.controls.length}</b> adaptive local references`;
}

function initWorldMap() {
  if (!window.L) {
    $("#worldMap").innerHTML = '<p class="map-unavailable">Map tiles need an internet connection. Coordinate search still works.</p>';
    return;
  }
  worldMap = L.map("worldMap", { worldCopyJump: true, minZoom: 2 }).setView([20, 0], 2);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
  }).addTo(worldMap);
  worldMap.on("click", (event) => {
    const { lat, lng } = event.latlng;
    setMapSelection(lat, lng, `Pinned location (${lat.toFixed(5)}, ${lng.toFixed(5)})`, false);
  });
  setMapSelection(selectedPoint.lat, selectedPoint.lon, null, false);
}

function showSearchResults(results, message = "") {
  const panel = $("#searchResults");
  panel.hidden = false;
  panel.replaceChildren();
  if (!results.length) {
    const empty = document.createElement("div");
    empty.className = "search-result search-status";
    empty.textContent = message || "No matching place found. Try a fuller address or coordinates.";
    panel.append(empty);
    return;
  }
  results.forEach((result) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "search-result";
    button.setAttribute("role", "option");
    button.textContent = result.name;
    button.addEventListener("click", () => {
      setMapSelection(result.lat, result.lon, result.name);
      if (result.boundingBox && worldMap) worldMap.fitBounds([[result.boundingBox[0], result.boundingBox[2]], [result.boundingBox[1], result.boundingBox[3]]]);
      panel.hidden = true;
      $("#coverageHint").textContent = "Location selected · adjust the pin if needed";
    });
    panel.append(button);
  });
}

async function searchLocation() {
  const query = $("#location").value.trim();
  const requestId = ++locationSearchRequest;
  if (!query) {
    $("#searchResults").hidden = true;
    return;
  }
  const coordinateMatch = query.match(/^\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)\s*$/);
  if (coordinateMatch) {
    const lat = Number(coordinateMatch[1]);
    const lon = Number(coordinateMatch[2]);
    if (Math.abs(lat) <= 90 && Math.abs(lon) <= 180) {
      showSearchResults([{ lat, lon, name: `Use coordinates ${lat.toFixed(5)}, ${lon.toFixed(5)}` }]);
      return;
    }
  }
  if (query.length < 3) return showSearchResults([], "Keep typing to search worldwide");
  showSearchResults([], "Searching worldwide…");
  try {
    const response = await fetch(`/api/geocode?q=${encodeURIComponent(query)}`);
    if (!response.ok) throw new Error("Search failed");
    if (requestId !== locationSearchRequest) return;
    showSearchResults((await response.json()).results);
  } catch (error) {
    if (requestId === locationSearchRequest) showSearchResults([], "Location search is temporarily unavailable");
  }
}

function applyTrajectoryZoom() {
  const { width, height } = trajectoryChartSize;
  const viewWidth = width / trajectoryZoom;
  const viewHeight = height / trajectoryZoom;
  const centerX = Math.max(viewWidth/2,Math.min(width-viewWidth/2,trajectoryCenter.x*width));
  const centerY = Math.max(viewHeight/2,Math.min(height-viewHeight/2,trajectoryCenter.y*height));
  trajectoryCenter = { x: centerX/width, y: centerY/height };
  $("#trend").setAttribute("viewBox", `${centerX-viewWidth/2} ${centerY-viewHeight/2} ${viewWidth} ${viewHeight}`);
  $("#trend").classList.toggle("is-zoomed",trajectoryZoom>1);
  $("#trajectoryPanHint").classList.toggle("visible",trajectoryZoom>1);
  $("#trajectoryZoomOut").disabled = trajectoryZoom <= 1;
  $("#trajectoryZoomIn").disabled = trajectoryZoom >= 2.5;
}
function trendChart(forecast) {
  if (!forecast?.trajectories) return;
  latestForecast = forecast;
  const selected = $("#trajectoryMetric").value;
  const palette = { ndvi: "#c8ff45", ndwi: "#4ee6d3" };
  const keys = selected === "all" ? ["ndvi", "ndwi"] : [selected];
  const compact = window.innerWidth < 620;
  const w = compact ? 390 : 900, h = compact ? 280 : 300, left = compact ? 45 : 58, right = compact ? 34 : 58, top = 22, bottom = compact ? 40 : 42;
  const historyCount = forecast.trajectories.ndvi.history.length;
  const futureCount = forecast.trajectories.ndvi.future.length;
  const nowIndex = historyCount - 1;
  const lastIndex = nowIndex + futureCount;
  const visibleValues = keys.flatMap((key) => {
    const item = forecast.trajectories[key];
    return [...item.history, ...item.future, ...(item.interval || []).flatMap((entry) => [entry.lower, entry.upper])];
  });
  const dataMin = Math.min(...visibleValues);
  const dataMax = Math.max(...visibleValues);
  const padding = Math.max(0.08,(dataMax-dataMin)*0.22);
  let min = Math.max(-1,Math.floor((dataMin-padding)*10)/10);
  let max = Math.min(1,Math.ceil((dataMax+padding)*10)/10);
  if (max-min<0.3) {
    const midpoint=(max+min)/2;
    min=Math.max(-1,midpoint-0.15);
    max=Math.min(1,midpoint+0.15);
  }
  const x = (index) => left + index * (w - left - right) / Math.max(1, lastIndex);
  const y = (value) => top + (max - value) * (h - top - bottom) / (max - min);
  const points = (values, offset) => values.map((value, index) => `${x(index + offset).toFixed(1)},${y(value).toFixed(1)}`).join(" ");
  const tickCount = compact ? 4 : 6;
  const yTicks = Array.from({length:tickCount},(_,index)=>min+(max-min)*index/(tickCount-1));
  const dateLabel = (index) => new Date(forecast.historyDates[index]).toLocaleDateString("en", { month: "short", year: "2-digit", timeZone: "UTC" }).toUpperCase();
  const observedTicks = compact
    ? [[0, dateLabel(0)], [Math.floor(nowIndex / 2), dateLabel(Math.floor(nowIndex / 2))], [nowIndex, "LATEST"]]
    : [0, .25, .5, .75, 1].map((ratio) => { const index = Math.round(nowIndex * ratio); return [index, ratio === 1 ? "LATEST" : dateLabel(index)]; });
  const projectedEndLabel = forecast.futureDates?.length
    ? new Date(forecast.futureDates.at(-1)).toLocaleDateString("en", { month: "short", year: "2-digit", timeZone: "UTC" }).toUpperCase()
    : "+6M";
  const xTicks = [...new Map([...observedTicks, [lastIndex, projectedEndLabel]].map((entry) => [entry[0], entry])).values()];
  const tickPrecision = max-min<0.6 ? 2 : 1;
  const grid = yTicks.map((value) => `<line x1="${left}" y1="${y(value)}" x2="${w-right}" y2="${y(value)}" class="chart-grid"/><text x="${left-12}" y="${y(value)+4}" class="chart-axis" text-anchor="end">${value.toFixed(tickPrecision)}</text>`).join("");
  const labels = xTicks.map(([index, label]) => `<text x="${x(index)}" y="${h-13}" class="chart-axis" text-anchor="middle">${label}</text>`).join("");
  const series = keys.map((key) => {
    const item = forecast.trajectories[key];
    const history = points(item.history, 0);
    const future = points([item.history.at(-1), ...item.future], nowIndex);
    const interval = item.interval || [];
    const upper = points([item.history.at(-1), ...interval.map((entry) => entry.upper)], nowIndex);
    const lower = points([item.history.at(-1), ...interval.map((entry) => entry.lower)], nowIndex).split(" ").reverse().join(" ");
    const uncertainty = interval.length ? `<polygon points="${upper} ${lower}" class="uncertainty-band" style="fill:${palette[key]}"/>` : "";
    const last = item.future.at(-1);
    return `<g class="series series-${key}">${uncertainty}<polyline points="${history}" class="history-line" style="stroke:${palette[key]}"/><polyline points="${future}" class="future-line" style="stroke:${palette[key]}"/><circle cx="${x(nowIndex)}" cy="${y(item.history.at(-1))}" r="4" style="fill:${palette[key]}"/><text x="${x(lastIndex)-4}" y="${y(last)-9}" class="end-label" text-anchor="end" style="fill:${palette[key]}">${item.label} ${last.toFixed(2)}</text></g>`;
  }).join("");
  const shadowIsUsable = false;
  const shadowSeries = shadowIsUsable ? keys.map((key) => {
    const expected = latestShadow.expected[key];
    const actual = forecast.trajectories[key].history;
    if (!expected?.length || expected.length !== actual.length) return "";
    const expectedPoints = points(expected, 0);
    const area = `${points(actual, 0)} ${expectedPoints.split(" ").reverse().join(" ")}`;
    return `<g class="shadow-series shadow-${key}"><polygon points="${area}" class="counterfactual-gap" style="fill:${palette[key]}"/><polyline points="${expectedPoints}" class="shadow-line" style="stroke:${palette[key]}"/></g>`;
  }).join("") : "";
  trajectoryChartSize = { width: w, height: h };
  $("#trend").classList.toggle("compact-chart", compact);
  const accessibleTitle=shadowIsUsable?(selected==="all"?"Observed and ShadowEarth index trajectories":`${forecast.trajectories[selected].label} observed versus twin trajectory`):(selected==="all"?"Environmental index trajectories":forecast.trajectories[selected].description+" trajectory");
  const accessibleDescription=shadowIsUsable?`${historyCount} Sentinel-2 observations compared with accepted ecological twins, followed by a six-month projection.`:`${historyCount} Sentinel-2 observations followed by a six-month projection. Ecological controls were not accepted.`;
  $("#trend").innerHTML = `<title>${accessibleTitle}</title><desc>${accessibleDescription}</desc>${grid}<line x1="${x(nowIndex)}" y1="${top}" x2="${x(nowIndex)}" y2="${h-bottom}" class="forecast-divider"/><text x="${x(nowIndex)+8}" y="${top+12}" class="forecast-marker">PROJECTION →</text>${labels}${shadowSeries}${series}<text x="${compact ? 13 : 16}" y="${(top+h-bottom)/2}" class="axis-title" transform="rotate(-90 ${compact ? 13 : 16} ${(top+h-bottom)/2})" text-anchor="middle">NORMALIZED INDEX</text>`;
  applyTrajectoryZoom();
  $("#trajectoryTitle").textContent = shadowIsUsable
    ? (selected === "all" ? "Observed vs ShadowEarth" : `${forecast.trajectories[selected].label} · observed vs twin`)
    : (selected === "all" ? "Environmental index trajectory" : `${forecast.trajectories[selected].label} trajectory`);
  const referenceLabel=latestShadow?.referenceMode==="matched-twins"?"ECO TWIN":"LOCAL REF";
  $("#chartLegend").innerHTML = keys.map((key) => `<span><i style="background:${palette[key]}"></i>${forecast.trajectories[key].label}</span>`).join("") + (shadowIsUsable ? `<span class="twin-legend"><i></i>${referenceLabel}</span>` : "");
}
function percent(value){return `${(Number(value||0)*100).toFixed(2)}%`;}
function renderPassport(passport,validation){
  latestPassport={...passport,modelValidation:validation,generatedAt:new Date().toISOString(),location:$("#location").value,coordinates:selectedPoint};
  const verdict=$("#passportVerdict");
  verdict.className=`evidence-verdict ${passport.status}`;
  verdict.innerHTML=`<span>EVIDENCE DECISION · QUALITY ${passport.quality}%</span><h2>${passport.status.toUpperCase()}</h2><p>${passport.conclusion}</p><small>Spatial agreement ${passport.spatialStability}% · ΔNDVI ${passport.indexChange.ndvi} · ΔNDWI ${passport.indexChange.ndwi}</small>`;
  $("#passportState").textContent=passport.status.toUpperCase();
  const water=passport.floodWater;
  $("#waterChange").textContent=water ? `${water.change>=0?"+":""}${percent(water.change)} WATER` : "MODEL ABSTAINED";
  $("#cnnGrid").innerHTML=(water?.probabilities||Array(9).fill(0)).map((value)=>`<i title="${percent(value)}" style="--p:${value}"></i>`).join("");
  $("#droughtSignal").textContent=`${passport.drought.label} · ${passport.drought.score}/100`;
  $("#captureLedger").innerHTML=passport.captures.map((capture)=>`<div><b>${capture.role}</b><span>${new Date(capture.actualDate).toLocaleDateString("en-GB",{day:"2-digit",month:"short",year:"numeric",timeZone:"UTC"})}</span><small>Δ${capture.offsetDays}d · ${capture.cloudCover}% cloud · ${capture.spatialSamples}/9 cells</small></div>`).join("");
  $("#validationAccuracy").textContent=percent(validation.accuracy);
  $("#validationBalanced").textContent=percent(validation.balancedAccuracy);
  $("#validationF1").textContent=percent(validation.f1);
  $("#validationIou").textContent=percent(validation.iou);
  $("#validationBrier").textContent=Number(validation.brierScore).toFixed(3);
  $("#validationSource").textContent=`${validation.testPixels.toLocaleString()} held-out pixels · ${validation.testCountries.join(" + ")} · Sen1Floods11`;
  $("#downloadPassport").disabled=false;
}
function applyResult(data){
  $("#ndvi").textContent=data.indices.ndvi.toFixed(2);
  $("#ndwi").textContent=data.indices.ndwi.toFixed(2);
  $("#ndviGap").textContent="Vegetation health";
  $("#ndwiGap").textContent="Surface water";
  renderPassport(data.passport,data.modelValidation);
  trendChart(data.forecast);
}
function toast(message){const el=$("#toast");el.textContent=message;el.classList.add("show");setTimeout(()=>el.classList.remove("show"),2600)}
function drawStatus(canvas, message) { const ctx=canvas.getContext("2d");canvas.style.backgroundImage="none";ctx.fillStyle="#07110d";ctx.fillRect(0,0,canvas.width,canvas.height);ctx.fillStyle="#8f9c91";ctx.font="14px DM Mono";ctx.textAlign="center";ctx.fillText(message,canvas.width/2,canvas.height/2);ctx.textAlign="left"; }
function loadSatelliteCanvas(canvas, url) { return new Promise((resolve, reject) => { const img=new Image();img.onload=()=>{const ctx=canvas.getContext("2d");ctx.clearRect(0,0,canvas.width,canvas.height);canvas.style.backgroundImage=`url("${url}")`;canvas.style.backgroundSize="cover";canvas.style.backgroundPosition="center";resolve()};img.onerror=()=>reject(new Error("Satellite image could not be rendered"));img.src=url; }); }
async function analyze() {
  const place = $("#location").value.trim() || "Selected world area";
  const button = $("#analyzeBtn");
  document.body.classList.add("analyzing");
  button.disabled = true;
  $("#analyzeBtn span").textContent = "Running analysis…";
  $("#satelliteSource").textContent = "Searching cloud-filtered Copernicus Sentinel-2 observations…";
  drawStatus($("#beforeMap"), "LOADING SENTINEL-2 BASELINE");
  drawStatus($("#afterMap"), "LOADING SENTINEL-2 COMPARISON");
  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ beforeDate: $("#beforeDate").value, afterDate: $("#afterDate").value, location: place, coordinates: selectedPoint })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Satellite analysis failed");
    applyResult(data);
    await Promise.all([loadSatelliteCanvas($("#beforeMap"), data.imagery.before.url), loadSatelliteCanvas($("#afterMap"), data.imagery.after.url)]);
    const caption = (item) => `${place.toUpperCase()} · ${new Date(item.date).toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric", timeZone: "UTC" }).toUpperCase()} · Δ${item.offsetDays}D · ${item.cloudCover}% CLOUD`;
    $("#beforeCaption").textContent = caption(data.imagery.before);
    $("#afterCaption").textContent = caption(data.imagery.after);
    const gridCoverage = data.source.minimumSpatialSamples === data.source.maximumSpatialSamples
      ? `${data.source.minimumSpatialSamples}/${data.source.expectedSpatialSamples}`
      : `${data.source.minimumSpatialSamples}-${data.source.maximumSpatialSamples}/${data.source.expectedSpatialSamples}`;
    $("#satelliteSource").textContent = `${data.source.name} · ${data.source.observations}/${data.source.targetObservations} dates · ${gridCoverage} grid · ≤40d capture offset`;
    toast(`${place} · live Sentinel-2 analysis complete`);
  } catch (error) {
    drawStatus($("#beforeMap"), "SATELLITE DATA UNAVAILABLE");
    drawStatus($("#afterMap"), "TRY ANOTHER DATE OR LOCATION");
    $("#satelliteSource").textContent = error.message;
    toast(error.message);
  } finally {
    document.body.classList.remove("analyzing");
    button.disabled = false;
    $("#analyzeBtn span").textContent = "Run analysis";
  }
}
$("#analyzeBtn").addEventListener("click", analyze);
$("#downloadPassport").addEventListener("click",()=>{
  if(!latestPassport)return;
  const blob=new Blob([JSON.stringify(latestPassport,null,2)],{type:"application/json"});
  const link=document.createElement("a");link.href=URL.createObjectURL(blob);link.download=`geovibe-change-passport-${Date.now()}.json`;link.click();URL.revokeObjectURL(link.href);
});
$("#trajectoryMetric").addEventListener("change", () => trendChart(latestForecast));
$("#trajectoryZoomOut").addEventListener("click", () => { trajectoryZoom=Math.max(1,trajectoryZoom-.25);if(trajectoryZoom===1)trajectoryCenter={x:.5,y:.5};applyTrajectoryZoom(); });
$("#trajectoryZoomIn").addEventListener("click", () => { trajectoryZoom=Math.min(2.5,trajectoryZoom+.25);applyTrajectoryZoom(); });
$("#trend").addEventListener("pointerdown", (event) => {
  if (trajectoryZoom <= 1) return;
  event.preventDefault();
  $("#trend").setPointerCapture(event.pointerId);
  trajectoryDrag = { pointerId:event.pointerId, x:event.clientX, y:event.clientY, center:{...trajectoryCenter} };
  $("#trend").classList.add("is-dragging");
});
$("#trend").addEventListener("pointermove", (event) => {
  if (!trajectoryDrag || trajectoryDrag.pointerId !== event.pointerId) return;
  const rect = $("#trend").getBoundingClientRect();
  trajectoryCenter.x = trajectoryDrag.center.x - (event.clientX-trajectoryDrag.x)/(rect.width*trajectoryZoom);
  trajectoryCenter.y = trajectoryDrag.center.y - (event.clientY-trajectoryDrag.y)/(rect.height*trajectoryZoom);
  applyTrajectoryZoom();
});
const endTrajectoryDrag = (event) => {
  if (!trajectoryDrag || trajectoryDrag.pointerId !== event.pointerId) return;
  trajectoryDrag = null;
  $("#trend").classList.remove("is-dragging");
};
$("#trend").addEventListener("pointerup",endTrajectoryDrag);
$("#trend").addEventListener("pointercancel",endTrajectoryDrag);
window.addEventListener("resize", () => { if (latestForecast) trendChart(latestForecast); });
$("#location").addEventListener("input", () => {
  clearTimeout(locationSearchTimer);
  const query = $("#location").value.trim();
  $("#coverageHint").textContent = query ? "Choose a suggestion to position the map" : "Type any place worldwide or coordinates";
  if (!query) return $("#searchResults").hidden = true;
  locationSearchTimer = setTimeout(searchLocation, 450);
});
$("#location").addEventListener("keydown", (event) => {
  if (event.key === "Escape") $("#searchResults").hidden = true;
  if (event.key === "Enter") {
    event.preventDefault();
    const firstResult = $("#searchResults .search-result[role='option']");
    if (firstResult) firstResult.click();
    else searchLocation();
  }
});
document.addEventListener("click", (event) => {
  if (!event.target.closest(".aoi-control")) $("#searchResults").hidden = true;
});
initWorldMap();
analyze();
