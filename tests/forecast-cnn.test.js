const test = require("node:test");
const assert = require("node:assert/strict");
const { createForecastCnn, predictForecast, trainForecastCnn } = require("../src/ml/forecast-cnn");
const { acceptance, verifySplits } = require("../training/train-forecast");

function row(direction, split = "train", region = "region-a", originDate = "2021-01-01") {
  const input = Array.from({ length: 12 }, (_, month) => ({ ndvi: 0.2 + direction * month * 0.02, ndwi: -0.1 + direction * month * 0.015, spei: direction * (0.2 + month * 0.08), valid: true }));
  return {
    region, split, originDate, input,
    targets: {
      ndvi: Array.from({ length: 6 }, (_, month) => input.at(-1).ndvi + direction * (month + 1) * 0.02),
      ndwi: Array.from({ length: 6 }, (_, month) => input.at(-1).ndwi + direction * (month + 1) * 0.015),
      drought: direction < 0 ? [1, 1, 1] : [0, 0, 0]
    }
  };
}

test("forecast CNN emits six index values and three drought probabilities", () => {
  const prediction = predictForecast(createForecastCnn(), row(1).input);
  assert.equal(prediction.index.ndvi.length, 6);
  assert.equal(prediction.index.ndwi.length, 6);
  assert.equal(prediction.drought.length, 3);
  assert.ok(prediction.drought.every((value) => value >= 0 && value <= 1));
});

test("forecast CNN learns contrasting temporal sequences", () => {
  const rows = Array.from({ length: 20 }, (_, index) => row(index % 2 ? 1 : -1));
  const model = trainForecastCnn(rows, 50);
  const rising = predictForecast(model, row(1).input);
  const falling = predictForecast(model, row(-1).input);
  assert.ok(rising.index.ndvi.at(-1) > falling.index.ndvi.at(-1));
  assert.ok(falling.drought.at(-1) > rising.drought.at(-1));
});

test("split audit rejects geographic and temporal leakage", () => {
  assert.throws(() => verifySplits([row(1), row(1, "calibration", "region-a", "2022-01-01"), row(1, "test", "region-c", "2023-01-01")]), /Geographic leakage/);
  assert.throws(() => verifySplits([row(1, "train", "region-a", "2024-01-01"), row(1, "calibration", "region-b", "2022-01-01"), row(1, "test", "region-c", "2023-01-01")]), /Temporal leakage/);
});

test("acceptance gate rejects under-sized or unskilled candidates", () => {
  const weak = { index: { ndvi: [{ rmse: 1, persistenceRmse: 0.5 }], ndwi: [{ rmse: 1, persistenceRmse: 0.5 }] }, drought: [{ balancedAccuracy: 0.5, f1: 0.4 }] };
  assert.equal(acceptance(weak, { train: 10, calibration: 2, test: 2 }).passed, false);
});

test("acceptance gate deploys passing horizons and abstains on a failed middle horizon", () => {
  const skilled = {
    index: { ndvi: [{ rmse: 0.1, persistenceRmse: 0.2 }], ndwi: [{ rmse: 0.1, persistenceRmse: 0.2 }] },
    drought: [
      { balancedAccuracy: 0.7, f1: 0.6 },
      { balancedAccuracy: 0.63, f1: 0.48 },
      { balancedAccuracy: 0.65, f1: 0.64 }
    ]
  };
  const gate = acceptance(skilled, { train: 1598, calibration: 615, test: 654 });
  assert.equal(gate.passed, true);
  assert.equal(gate.fullPass, false);
  assert.deepEqual(gate.acceptedDroughtHorizons, [1, 6]);
});
