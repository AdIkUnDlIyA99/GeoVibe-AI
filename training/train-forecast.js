const fs = require("node:fs");
const path = require("node:path");
const { fitForecastCalibration, predictForecast, trainForecastCnn } = require("../src/ml/forecast-cnn");

const DATASET = process.env.FORECAST_DATASET || path.join(__dirname, "data", "forecast-sequences.jsonl");
const ROOT = path.join(__dirname, "..");

function loadRows(file) {
  if (!fs.existsSync(file)) throw new Error(`Missing forecast dataset: ${file}`);
  return fs.readFileSync(file, "utf8").split(/\r?\n/).filter(Boolean).map((line, index) => {
    const row = JSON.parse(line);
    if (!row.region || !["train", "calibration", "test"].includes(row.split)) throw new Error(`Row ${index + 1} has an invalid geographic split`);
    if (!Array.isArray(row.input) || row.input.length !== 12) throw new Error(`Row ${index + 1} requires exactly 12 input months`);
    if (!["ndvi", "ndwi"].every((key) => Array.isArray(row.targets?.[key]) && row.targets[key].length === 6)) throw new Error(`Row ${index + 1} requires six future NDVI/NDWI targets`);
    if (!Array.isArray(row.targets.drought) || row.targets.drought.length !== 3) throw new Error(`Row ${index + 1} requires +1/+3/+6 drought labels`);
    return row;
  });
}

function verifySplits(rows) {
  const regions = (split) => new Set(rows.filter((row) => row.split === split).map((row) => row.region));
  const train = regions("train"), calibration = regions("calibration"), test = regions("test");
  if (!train.size || !calibration.size || !test.size) throw new Error("Train, calibration and test must all contain geographic regions");
  if ([...train].some((region) => calibration.has(region) || test.has(region)) || [...calibration].some((region) => test.has(region))) throw new Error("Geographic leakage: a region occurs in multiple splits");
  const dates = (split) => rows.filter((row) => row.split === split).map((row) => Date.parse(row.originDate));
  if (Math.max(...dates("train")) >= Math.min(...dates("test"))) throw new Error("Temporal leakage: training dates overlap the test period");
}

function binaryMetrics(probabilities, labels, threshold = 0.5) {
  let tp = 0, tn = 0, fp = 0, fn = 0, brier = 0;
  probabilities.forEach((probability, index) => {
    const predicted = probability >= threshold ? 1 : 0, label = labels[index];
    brier += (probability - label) ** 2;
    if (predicted && label) tp += 1;
    else if (!predicted && !label) tn += 1;
    else if (predicted) fp += 1;
    else fn += 1;
  });
  const precision = tp / Math.max(1, tp + fp), recall = tp / Math.max(1, tp + fn), specificity = tn / Math.max(1, tn + fp);
  return { threshold, balancedAccuracy: (recall + specificity) / 2, precision, recall, f1: 2 * precision * recall / Math.max(1e-9, precision + recall), brierScore: brier / labels.length, confusionMatrix: { tp, tn, fp, fn } };
}

function evaluate(model, rows) {
  const predictions = rows.map((row) => predictForecast(model, row.input));
  const index = {};
  for (const key of ["ndvi", "ndwi"]) {
    index[key] = Array.from({ length: 6 }, (_, horizon) => {
      const errors = predictions.map((prediction, i) => prediction.index[key][horizon] - rows[i].targets[key][horizon]);
      const persistence = rows.map((row) => row.input.at(-1)[key]);
      const seasonal = rows.map((row) => row.input[horizon][key]);
      const baselineRmse = (values) => Math.sqrt(values.reduce((sum, value, i) => sum + (value - rows[i].targets[key][horizon]) ** 2, 0) / rows.length);
      return {
        month: horizon + 1,
        mae: errors.reduce((sum, value) => sum + Math.abs(value), 0) / rows.length,
        rmse: Math.sqrt(errors.reduce((sum, value) => sum + value ** 2, 0) / rows.length),
        persistenceRmse: baselineRmse(persistence),
        seasonalNaiveRmse: baselineRmse(seasonal)
      };
    });
  }
  const drought = [0, 1, 2].map((horizon) => binaryMetrics(predictions.map((item) => item.drought[horizon]), rows.map((row) => row.targets.drought[horizon]), model.droughtThresholds?.[horizon] ?? 0.5));
  return { samples: rows.length, index, drought };
}

function acceptance(report, counts) {
  const forecastPoints = [...report.index.ndvi, ...report.index.ndwi];
  const beatsPersistence = forecastPoints.every((item) => item.rmse < item.persistenceRmse);
  const droughtPasses = report.drought.every((item) => item.balancedAccuracy >= 0.6 && item.f1 >= 0.5);
  const enoughData = counts.train >= 1000 && counts.calibration >= 150 && counts.test >= 300;
  return { passed: beatsPersistence && droughtPasses && enoughData, checks: { beatsPersistence, droughtPasses, enoughData }, requirements: { train: 1000, calibration: 150, test: 300, droughtBalancedAccuracy: 0.6, droughtF1: 0.5 } };
}

function main() {
  const rows = loadRows(DATASET);
  verifySplits(rows);
  const split = (name) => rows.filter((row) => row.split === name);
  const train = split("train"), calibration = split("calibration"), test = split("test");
  const model = fitForecastCalibration(trainForecastCnn(train), calibration);
  const metrics = evaluate(model, test);
  const counts = { train: train.length, calibration: calibration.length, test: test.length };
  const gate = acceptance(metrics, counts);
  const metadata = { status: gate.passed ? "accepted" : "rejected", trainedAt: new Date().toISOString(), dataset: path.resolve(DATASET), split: "disjoint geographic regions with training strictly earlier than testing", counts, gate };
  const artifact = { ...model, metrics, metadata };
  fs.mkdirSync(path.join(ROOT, "reports"), { recursive: true });
  fs.writeFileSync(path.join(ROOT, "reports", "forecast-evaluation.json"), JSON.stringify({ metadata, metrics }, null, 2));
  fs.writeFileSync(path.join(ROOT, "models", "forecast-cnn.candidate.json"), JSON.stringify(artifact));
  if (gate.passed) fs.writeFileSync(path.join(ROOT, "models", "forecast-cnn.json"), JSON.stringify(artifact));
  console.log(`Forecast candidate ${metadata.status.toUpperCase()} | train/cal/test ${counts.train}/${counts.calibration}/${counts.test}`);
  console.log(`Gate checks: ${JSON.stringify(gate.checks)}`);
  if (!gate.passed) process.exitCode = 2;
}

if (require.main === module) {
  try { main(); } catch (error) { console.error(error.message); process.exitCode = 1; }
}

module.exports = { acceptance, evaluate, loadRows, verifySplits };
