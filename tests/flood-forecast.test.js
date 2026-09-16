const test = require("node:test");
const assert = require("node:assert/strict");
const { features, fitFloodCalibration, predictFloodForecast, trainFloodForecast } = require("../src/ml/flood-forecast");

function row(high, flood) {
  return {
    originDate: "2020-06-01",
    latitude: 25,
    forecast: Array.from({ length: 6 }, () => high
      ? { mean: 900, spread: 120, p10: 700, p50: 880, p90: 1100, exceedanceProbability: 0.9, thresholdRatio: 1.8 }
      : { mean: 80, spread: 12, p10: 60, p50: 78, p90: 100, exceedanceProbability: 0.05, thresholdRatio: 0.3 }),
    targets: { flood: [flood, flood, flood] }
  };
}

test("flood forecast requires finite GloFAS ensemble predictors", () => {
  assert.throws(() => features({ ...row(true, 1), forecast: [] }, 0), /lead month 1/);
  const invalid = row(true, 1);
  invalid.forecast[0].mean = NaN;
  assert.throws(() => features(invalid, 0), /finite ensemble predictors/);
});

test("flood forecast learns bounded multi-horizon probabilities", () => {
  const train = [...Array(20).fill(0).map(() => row(true, 1)), ...Array(20).fill(0).map(() => row(false, 0))];
  const model = fitFloodCalibration(trainFloodForecast(train, 50), train, 40);
  const high = predictFloodForecast(model, row(true, 1));
  const low = predictFloodForecast(model, row(false, 0));
  assert.equal(high.length, 3);
  assert.ok(high.every((value, index) => value > low[index] && value >= 0 && value <= 1));
});
