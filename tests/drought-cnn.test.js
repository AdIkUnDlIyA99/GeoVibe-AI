const test = require("node:test");
const assert = require("node:assert/strict");
const { createDroughtCnn, evaluateDroughtCnn, normalizeSequence, probability, trainDroughtCnn } = require("../src/ml/drought-cnn");

const sequence = (ndvi, ndwi) => Array.from({ length: 12 }, () => ({ ndvi, ndwi, valid: true }));

test("drought CNN enforces twelve monthly observations", () => {
  assert.throws(() => normalizeSequence(sequence(0.4, 0.1).slice(0, 11)), /at least 12/);
  assert.equal(normalizeSequence(sequence(0.4, 0.1)).length, 12);
});

test("drought CNN produces bounded probabilities", () => {
  const value = probability(createDroughtCnn(), sequence(0.4, 0.1));
  assert.ok(value >= 0 && value <= 1);
});

test("drought CNN can learn contrasting real-valued sequences", () => {
  const rows = [];
  for (let index = 0; index < 20; index += 1) {
    rows.push({ sequence: sequence(0.65 - index * 0.002, 0.2), label: 0 });
    rows.push({ sequence: Array.from({ length: 12 }, (_, month) => ({ ndvi: 0.55 - month * 0.055, ndwi: 0.1 - month * 0.045, valid: true })), label: 1 });
  }
  const model = trainDroughtCnn(rows, 45);
  const metrics = evaluateDroughtCnn(model, rows);
  assert.ok(metrics.balancedAccuracy >= 0.9);
});
