const test = require("node:test");
const assert = require("node:assert/strict");
const { once } = require("node:events");
const { forecast, modelLinkedHazards, selectCapture, server } = require("../server");
const floodModel = require("../models/flood/flood-cnn.json");
const droughtModel = require("../models/drought/drought-forecast.json");

test("capture selection chooses the acquisition nearest the requested date", () => {
  const observations = [
    { id: "late", date: "2025-03-25T00:00:00Z" },
    { id: "near", date: "2025-03-17T00:00:00Z" },
    { id: "early", date: "2025-02-01T00:00:00Z" }
  ];
  const selected = selectCapture(observations, new Date("2025-03-15T12:00:00Z"), "baseline");
  assert.equal(selected.id, "near");
  assert.equal(selected.offsetDays, 2);
});

test("capture selection rejects acquisitions outside the tolerance", () => {
  assert.throws(() => selectCapture([{ date: "2024-01-01T00:00:00Z" }], new Date("2025-03-15T00:00:00Z"), "baseline"), /within 65 days/);
});

test("six-month outlook abstains from future flood and climate drought inference", () => {
  const observations = Array.from({ length: 12 }, (_, index) => ({
    date: new Date(Date.UTC(2025, index, 15)).toISOString(),
    indices: { ndvi: 0.45 - index * 0.01, ndwi: -0.12 + index * 0.005 },
    spatialSamples: Array.from({ length: 9 }, (__, cell) => ({ cell, ndvi: 0.34, ndwi: -0.065 }))
  }));
  const outlook = modelLinkedHazards(observations, forecast(observations), floodModel, droughtModel);
  assert.equal(outlook.horizons.length, 6);
  assert.ok(outlook.currentFlood.temporaryWaterFraction >= 0 && outlook.currentFlood.temporaryWaterFraction <= 1);
  assert.ok(outlook.horizons.every((item) => item.flood === null));
  assert.ok(outlook.horizons.filter((item) => [1, 3, 6].includes(item.month)).every((item) => item.drought === null));
});

test("runtime loads the accepted index forecast and exposes drought abstention", () => {
  const observations = Array.from({ length: 12 }, (_, index) => ({
    date: new Date(Date.UTC(2025, index, 15)).toISOString(),
    indices: { ndvi: 0.3 + index * 0.01, ndwi: -0.2 + index * 0.005 }
  }));
  const result = forecast(observations);
  assert.equal(result.modelStatus, "accepted-trained-cnn");
  assert.deepEqual(result.droughtForecast, { 1: null, 3: null, 6: null });
});

test("runtime keeps the trained index model for sparse satellite histories", () => {
  const observations = Array.from({ length: 7 }, (_, index) => ({
    date: new Date(Date.UTC(2025, index, 15)).toISOString(),
    indices: { ndvi: 0.3 + index * 0.01, ndwi: -0.2 + index * 0.005 }
  }));
  const result = forecast(observations);
  assert.equal(result.modelStatus, "accepted-trained-cnn");
  assert.equal(result.inputCoverage, "partial-validity-masked");
  assert.equal(result.inputObservations, 7);
  assert.equal(result.trajectories.ndvi.future.length, 6);
  assert.equal(result.trajectories.ndwi.future.length, 6);
});

test("six-month drought can explicitly project an available one-month probability", () => {
  const observations = Array.from({ length: 12 }, (_, index) => ({
    date: new Date(Date.UTC(2025, index, 15)).toISOString(),
    indices: { ndvi: 0.4, ndwi: -0.1 },
    spatialSamples: Array.from({ length: 9 }, (__, cell) => ({ cell, ndvi: 0.34, ndwi: -0.065 }))
  }));
  const indexForecast = forecast(observations);
  indexForecast.droughtForecast = { 1: 0.42, 3: null, 6: null };
  const outlook = modelLinkedHazards(observations, indexForecast, floodModel, droughtModel);
  const sixMonth = outlook.horizons.find((item) => item.month === 6);
  assert.equal(sixMonth.drought, 0.42);
  assert.deepEqual(sixMonth.droughtProjection, { sourceMonth: 1, method: "constant-risk projection", independentlyValidated: false });
});

test("API rejects future analysis dates and browser retraining", async (context) => {
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  context.after(() => server.close());
  const { port } = server.address();
  const origin = `http://127.0.0.1:${port}`;
  const future = await fetch(`${origin}/api/analyze`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ analysisDate: "2099-01-01", coordinates: { lat: 28.6, lon: 77.2 } })
  });
  assert.equal(future.status, 422);
  assert.match((await future.json()).error, /cannot be in the future/);
  assert.equal((await fetch(`${origin}/api/train`, { method: "POST" })).status, 404);
});
