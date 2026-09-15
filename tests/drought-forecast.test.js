const test = require("node:test");
const assert = require("node:assert/strict");
const { features, fitDroughtCalibration, predictDroughtForecast, trainDroughtForecast } = require("../src/ml/drought-forecast");

const row = (value, drought, split = "train") => ({
  originDate: "2021-06-15",
  region: `${split}-region`,
  split,
  input: Array.from({ length: 12 }, (_, index) => ({ spei: value + index * 0.01 })),
  targets: { drought: [drought, drought, drought] }
});

test("drought forecast requires twelve finite SPEI observations", () => {
  assert.throws(() => features([], "2021-01-15"), /exactly 12/);
  assert.throws(() => features(Array(12).fill({ spei: null }), "2021-01-15"), /finite SPEI/);
});

test("drought forecast learns and calibrates bounded multi-horizon probabilities", () => {
  const train = [...Array(15).fill(0).map(() => row(-2, 1)), ...Array(15).fill(0).map(() => row(1, 0))];
  const model = fitDroughtCalibration(trainDroughtForecast(train, 40), train, 30);
  const dry = predictDroughtForecast(model, row(-2, 1).input, "2021-06-15");
  const wet = predictDroughtForecast(model, row(1, 0).input, "2021-06-15");
  assert.equal(dry.length, 3);
  assert.ok(dry.every((value, index) => value > wet[index] && value >= 0 && value <= 1));
});
