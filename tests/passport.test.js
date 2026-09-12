const test = require("node:test");
const assert = require("node:assert/strict");
const { buildPassport, droughtSignal, spatialAgreement } = require("../src/analysis/passport");

const model = require("../models/flood-cnn.json");
const grid = (ndvi, ndwi, count = 9) => Array.from({ length: count }, (_, cell) => ({ cell, ndvi, ndwi }));
const sample = (ndvi, ndwi, count = 9) => ({
  indices: { ndvi, ndwi }, spatialSamples: grid(ndvi, ndwi, count), sampleCount: count,
  offsetDays: 2, cloudCover: 3, requestedDate: "2025-01-01T12:00:00.000Z", date: "2025-01-03T12:00:00.000Z"
});

test("spatial agreement rewards consistent directional change", () => {
  assert.equal(spatialAgreement([0.1, 0.2, 0.12, 0.08]), 1);
  assert.ok(spatialAgreement([0.1, -0.2, 0.12, -0.08]) < 1);
});

test("passport abstains when the CNN grid is incomplete", () => {
  const beforeSample = sample(0.4, -0.2, 8);
  const afterSample = sample(0.2, 0.2, 8);
  const passport = buildPassport({ model, observations: [beforeSample, afterSample], beforeSample, afterSample });
  assert.equal(passport.status, "insufficient");
  assert.equal(passport.floodWater, null);
});

test("drought signal rises when vegetation and water indices decline", () => {
  const observations = [sample(0.6, 0.2), sample(0.55, 0.1), sample(0.2, -0.3)];
  assert.equal(droughtSignal(observations).label, "Elevated");
});
