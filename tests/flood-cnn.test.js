const test = require("node:test");
const assert = require("node:assert/strict");
const {
  evaluateFloodCnn,
  fitPlatt,
  inferGrid,
  patchFeatures,
  probability,
  trainFloodCnn
} = require("../src/ml/flood-cnn");

const deployedModel = require("../models/flood-cnn.json");
const features = (ndvi, ndwi) => [...Array(9).fill(ndvi), ...Array(9).fill(ndwi)];

test("flood CNN extracts a centered 3x3 NDVI/NDWI patch", () => {
  const ndvi = Array.from({ length: 25 }, (_, index) => index);
  const ndwi = ndvi.map((value) => value + 100);
  assert.deepEqual(patchFeatures(ndvi, ndwi, 5, 5, 2, 2), [
    6, 7, 8, 11, 12, 13, 16, 17, 18,
    106, 107, 108, 111, 112, 113, 116, 117, 118
  ]);
});

test("deployed flood CNN always returns a bounded probability", () => {
  for (const row of [features(0.8, -0.5), features(0.1, 0.7), features(-1, 1)]) {
    const value = probability(deployedModel, row);
    assert.ok(Number.isFinite(value));
    assert.ok(value >= 0 && value <= 1);
  }
});

test("flood CNN learns separable water and non-water patches", () => {
  const rows = [];
  for (let index = 0; index < 30; index += 1) {
    rows.push({ features: features(0.65, -0.45), label: 0 });
    rows.push({ features: features(-0.15, 0.7), label: 1 });
  }
  const model = trainFloodCnn(rows, 80);
  const metrics = evaluateFloodCnn(model, rows);
  assert.equal(metrics.accuracy, 1);
  assert.equal(metrics.f1, 1);
  assert.deepEqual(metrics.confusionMatrix, { tp: 30, tn: 30, fp: 0, fn: 0 });
});

test("Platt calibration produces finite parameters and metrics", () => {
  const train = [
    ...Array.from({ length: 12 }, () => ({ features: features(0.7, -0.4), label: 0 })),
    ...Array.from({ length: 12 }, () => ({ features: features(-0.2, 0.65), label: 1 }))
  ];
  const model = fitPlatt(trainFloodCnn(train, 40), train, 40);
  const metrics = evaluateFloodCnn(model, train);
  assert.ok(Number.isFinite(model.calibration.a));
  assert.ok(Number.isFinite(model.calibration.b));
  assert.ok(metrics.brierScore >= 0 && metrics.brierScore <= 1);
  assert.equal(metrics.samples, 24);
});

test("grid inference requires exactly nine spatial cells", () => {
  const grid = Array.from({ length: 9 }, (_, cell) => ({ cell, ndvi: 0.2, ndwi: 0.3 }));
  const result = inferGrid(deployedModel, grid);
  assert.equal(result.probabilities.length, 9);
  assert.ok(result.temporaryWaterFraction >= 0 && result.temporaryWaterFraction <= 1);
  assert.ok(result.peakProbability >= 0 && result.peakProbability <= 1);
  assert.equal(inferGrid(deployedModel, grid.slice(0, 8)), null);
});
