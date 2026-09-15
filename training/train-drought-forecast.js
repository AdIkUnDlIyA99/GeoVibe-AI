const fs = require("node:fs");
const path = require("node:path");
const { HORIZONS, evaluateDroughtForecast, fitDroughtCalibration, trainDroughtForecast } = require("../../src/ml/drought-forecast");

const DATASET = process.env.DROUGHT_DATASET || process.env.FORECAST_DATASET;
const ROOT = path.join(__dirname, "..", "..");

function loadRows(file) {
  if (!file || !fs.existsSync(file)) throw new Error(`Missing drought dataset: ${file || "set DROUGHT_DATASET"}`);
  return fs.readFileSync(file, "utf8").split(/\r?\n/).filter(Boolean).map((line, index) => {
    const row = JSON.parse(line);
    if (!Array.isArray(row.input) || row.input.length !== 12 || !row.input.every((item) => Number.isFinite(Number(item.spei)))) throw new Error(`Row ${index + 1} requires twelve SPEI observations`);
    if (!Array.isArray(row.targets?.drought) || row.targets.drought.length !== 3) throw new Error(`Row ${index + 1} requires +1/+3/+6 drought targets`);
    if (!["train", "calibration", "test"].includes(row.split) || !row.region || !row.originDate) throw new Error(`Row ${index + 1} has invalid split metadata`);
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
  const model = fitDroughtCalibration(trainDroughtForecast(train), calibration);
  const metrics = evaluateDroughtForecast(model, test);
  const deployment = metrics.map((metric, index) => ({ month: HORIZONS[index], accepted: metric.balancedAccuracy >= 0.6 && metric.f1 >= 0.5 }));
  const counts = { train: train.length, calibration: calibration.length, test: test.length };
  const enoughData = counts.train >= 1000 && counts.calibration >= 150 && counts.test >= 300;
  const fullPass = enoughData && deployment.every((item) => item.accepted);
  const anyPass = enoughData && deployment.some((item) => item.accepted);
  const status = fullPass ? "accepted" : anyPass ? "accepted-with-abstention" : "rejected";
  const artifact = { ...model, deployment, metrics, metadata: { status, trainedAt: new Date().toISOString(), dataset: path.resolve(DATASET), counts, requirements: { balancedAccuracy: 0.6, f1: 0.5 } } };
  const directory = path.join(ROOT, "models", "drought");
  fs.mkdirSync(directory, { recursive: true });
  fs.writeFileSync(path.join(directory, "drought-forecast.candidate.json"), JSON.stringify(artifact));
  if (anyPass) fs.writeFileSync(path.join(directory, "drought-forecast.json"), JSON.stringify(artifact));
  console.log(`Drought forecast ${status.toUpperCase()} | train/cal/test ${counts.train}/${counts.calibration}/${counts.test}`);
  deployment.forEach((item, index) => console.log(`+${item.month}M ${item.accepted ? "PASS" : "ABSTAIN"} | BA=${metrics[index].balancedAccuracy.toFixed(3)} F1=${metrics[index].f1.toFixed(3)} threshold=${metrics[index].threshold.toFixed(2)}`));
  if (!anyPass) process.exitCode = 2;
}

if (require.main === module) {
  try { main(); } catch (error) { console.error(error.message); process.exitCode = 1; }
}

module.exports = { loadRows, verifySplits };
