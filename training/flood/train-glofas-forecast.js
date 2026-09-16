const fs = require("node:fs");
const path = require("node:path");
const { HORIZONS, evaluateFloodForecast, fitFloodCalibration, trainFloodForecast } = require("../../src/ml/flood-forecast");

const DATASET = process.env.FLOOD_FORECAST_DATASET;
const ROOT = path.join(__dirname, "..", "..");

function loadRows(file) {
  if (!file || !fs.existsSync(file)) throw new Error(`Missing GloFAS dataset: ${file || "set FLOOD_FORECAST_DATASET"}`);
  return fs.readFileSync(file, "utf8").split(/\r?\n/).filter(Boolean).map((line, index) => {
    const row = JSON.parse(line);
    if (!Array.isArray(row.forecast) || row.forecast.length !== 3) throw new Error(`Row ${index + 1} requires +1/+3/+6 GloFAS predictors`);
    if (!Array.isArray(row.targets?.flood) || row.targets.flood.length !== 3 || !row.targets.flood.every((value) => value === 0 || value === 1)) throw new Error(`Row ${index + 1} requires binary +1/+3/+6 flood targets`);
    if (!Number.isFinite(Number(row.latitude)) || !Number.isFinite(Number(row.longitude)) || !row.originDate || !row.region || !["train", "calibration", "test"].includes(row.split)) throw new Error(`Row ${index + 1} has invalid metadata`);
    return row;
  });
}

function verifySplits(rows) {
  const regions = (split) => new Set(rows.filter((row) => row.split === split).map((row) => row.region));
  const train = regions("train"), calibration = regions("calibration"), test = regions("test");
  if (!train.size || !calibration.size || !test.size) throw new Error("Train, calibration and test splits are required");
  if ([...train].some((item) => calibration.has(item) || test.has(item)) || [...calibration].some((item) => test.has(item))) throw new Error("Geographic leakage detected");
}

function main() {
  const rows = loadRows(DATASET);
  verifySplits(rows);
  const split = (name) => rows.filter((row) => row.split === name);
  const train = split("train"), calibration = split("calibration"), test = split("test");
  const model = fitFloodCalibration(trainFloodForecast(train), calibration);
  const metrics = evaluateFloodForecast(model, test, calibration);
  const deployment = metrics.map((metric, index) => ({ month: HORIZONS[index], accepted: metric.balancedAccuracy >= 0.6 && metric.f1 >= 0.5 && metric.brierSkill > 0 }));
  const counts = { train: train.length, calibration: calibration.length, test: test.length };
  const enoughData = counts.train >= 5000 && counts.calibration >= 750 && counts.test >= 1500;
  const fullPass = enoughData && deployment.every((item) => item.accepted);
  const anyPass = enoughData && deployment.some((item) => item.accepted);
  const status = fullPass ? "accepted" : anyPass ? "accepted-with-abstention" : "rejected";
  const artifact = { ...model, deployment, metrics, metadata: { status, trainedAt: new Date().toISOString(), dataset: path.resolve(DATASET), counts, target: "observed GloFAS discharge exceeds the location-specific training-period 95th percentile", requirements: { train: 5000, calibration: 750, test: 1500, balancedAccuracy: 0.6, f1: 0.5, brierSkill: 0 } } };
  const directory = path.join(ROOT, "models", "flood");
  fs.mkdirSync(directory, { recursive: true });
  fs.writeFileSync(path.join(directory, "glofas-forecast.candidate.json"), JSON.stringify(artifact));
  if (anyPass) fs.writeFileSync(path.join(directory, "glofas-forecast.json"), JSON.stringify(artifact));
  console.log(`GloFAS flood forecast ${status.toUpperCase()} | train/cal/test ${counts.train}/${counts.calibration}/${counts.test}`);
  deployment.forEach((item, index) => console.log(`+${item.month}M ${item.accepted ? "PASS" : "ABSTAIN"} | BA=${metrics[index].balancedAccuracy.toFixed(3)} F1=${metrics[index].f1.toFixed(3)} BSS=${metrics[index].brierSkill.toFixed(3)} threshold=${metrics[index].threshold.toFixed(2)}`));
  if (!anyPass) process.exitCode = 2;
}

if (require.main === module) {
  try { main(); } catch (error) { console.error(error.message); process.exitCode = 1; }
}

module.exports = { loadRows, verifySplits };
