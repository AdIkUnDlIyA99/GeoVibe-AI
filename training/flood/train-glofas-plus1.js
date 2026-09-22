const fs = require("node:fs");
const path = require("node:path");

const DATASET = process.env.FLOOD_PLUS1_DATASET;
const ROOT = path.join(__dirname, "..", "..");
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const sigmoid = (value) => 1 / (1 + Math.exp(-clamp(value, -30, 30)));

function features(row) {
  const lead = row.forecast?.[0];
  const values = [lead?.mean, lead?.spread, lead?.p10, lead?.p50, lead?.p90, lead?.exceedanceProbability, lead?.thresholdRatio].map(Number);
  if (!values.every(Number.isFinite)) throw new Error("Flood row requires finite +1 month ensemble predictors");
  const month = new Date(row.originDate).getUTCMonth();
  const scale = (value) => Math.log1p(Math.max(0, value));
  return [1, scale(values[0]), scale(values[1]), scale(values[2]), scale(values[3]), scale(values[4]), clamp(values[5], 0, 1), clamp(values[6], 0, 10), Math.sin(2 * Math.PI * month / 12), Math.cos(2 * Math.PI * month / 12), clamp(Math.abs(Number(row.latitude)) / 90, 0, 1)];
}

function loadRows(file) {
  if (!file || !fs.existsSync(file)) throw new Error(`Missing +1 month GloFAS dataset: ${file || "set FLOOD_PLUS1_DATASET"}`);
  return fs.readFileSync(file, "utf8").split(/\r?\n/).filter(Boolean).map((line, index) => {
    const row = JSON.parse(line);
    if (row.forecast?.length !== 1 || row.forecast[0]?.leadMonth !== 1 || row.targets?.flood?.length !== 1 || ![0, 1].includes(row.targets.flood[0])) throw new Error(`Invalid +1 month row ${index + 1}`);
    if (!row.originDate || !row.region || !["train", "calibration", "test"].includes(row.split)) throw new Error(`Invalid metadata in row ${index + 1}`);
    return row;
  });
}

function raw(model, input) { return model.weights.reduce((sum, weight, index) => sum + weight * input[index], 0); }
function probability(model, row) { return sigmoid(model.calibration.a * raw(model, features(row)) + model.calibration.b); }

function train(rows) {
  const model = { schemaVersion: 1, architecture: "calibrated +1 month GloFAS ensemble logistic flood forecast", horizon: 1, weights: Array(11).fill(0), calibration: { a: 1, b: 0 }, threshold: 0.5 };
  const positives = rows.reduce((sum, row) => sum + row.targets.flood[0], 0);
  const positiveWeight = Math.sqrt((rows.length - positives) / Math.max(1, positives));
  for (let epoch = 0; epoch < 240; epoch += 1) {
    const rate = 0.02 / Math.sqrt(1 + epoch / 40);
    for (const row of rows) {
      const input = features(row), label = row.targets.flood[0];
      const error = (sigmoid(raw(model, input)) - label) * (label ? positiveWeight : 1);
      model.weights = model.weights.map((weight, index) => clamp(weight - rate * (error * input[index] + 0.0008 * weight), -8, 8));
    }
  }
  return model;
}

function metrics(probabilities, labels, threshold, climatology) {
  let tp = 0, tn = 0, fp = 0, fn = 0, brier = 0, baselineBrier = 0;
  probabilities.forEach((value, index) => {
    const label = labels[index], predicted = value >= threshold;
    brier += (value - label) ** 2; baselineBrier += (climatology - label) ** 2;
    if (predicted && label) tp += 1; else if (!predicted && !label) tn += 1; else if (predicted) fp += 1; else fn += 1;
  });
  const precision = tp / Math.max(1, tp + fp), recall = tp / Math.max(1, tp + fn), specificity = tn / Math.max(1, tn + fp);
  const brierScore = brier / probabilities.length, climatologyBrier = baselineBrier / probabilities.length;
  return { threshold, balancedAccuracy: (recall + specificity) / 2, precision, recall, f1: 2 * precision * recall / Math.max(1e-9, precision + recall), brierScore, climatologyBrier, brierSkill: 1 - brierScore / Math.max(1e-9, climatologyBrier), confusionMatrix: { tp, tn, fp, fn } };
}

function calibrate(model, rows) {
  let a = 1, b = 0;
  for (let epoch = 0; epoch < 250; epoch += 1) {
    let gradientA = 0, gradientB = 0;
    for (const row of rows) {
      const score = raw(model, features(row)), error = sigmoid(a * score + b) - row.targets.flood[0];
      gradientA += error * score; gradientB += error;
    }
    const rate = 0.08 / Math.sqrt(1 + epoch / 60);
    a -= rate * gradientA / rows.length; b -= rate * gradientB / rows.length;
  }
  model.calibration = { a, b };
  const labels = rows.map((row) => row.targets.flood[0]), probabilities = rows.map((row) => probability(model, row));
  const climatology = labels.reduce((sum, label) => sum + label, 0) / labels.length;
  let best = { threshold: 0.5, score: -Infinity };
  for (let threshold = 0.05; threshold <= 0.9; threshold += 0.01) {
    const result = metrics(probabilities, labels, threshold, climatology), score = result.balancedAccuracy + result.f1;
    if (score > best.score) best = { threshold, score };
  }
  model.threshold = Number(best.threshold.toFixed(2));
  return model;
}

function main() {
  const rows = loadRows(DATASET);
  const bySplit = (name) => rows.filter((row) => row.split === name);
  const trainRows = bySplit("train"), calibrationRows = bySplit("calibration"), testRows = bySplit("test");
  if (!trainRows.length || !calibrationRows.length || !testRows.length) throw new Error("Train, calibration, and test rows are required");
  const model = calibrate(train(trainRows), calibrationRows);
  const labels = testRows.map((row) => row.targets.flood[0]);
  const calibrationLabels = calibrationRows.map((row) => row.targets.flood[0]);
  const report = metrics(testRows.map((row) => probability(model, row)), labels, model.threshold, calibrationLabels.reduce((sum, label) => sum + label, 0) / calibrationLabels.length);
  const counts = { train: trainRows.length, calibration: calibrationRows.length, test: testRows.length };
  const accepted = counts.train >= 5000 && counts.calibration >= 750 && counts.test >= 1500 && report.balancedAccuracy >= 0.6 && report.f1 >= 0.5 && report.brierSkill > 0;
  const artifact = { ...model, deployment: { accepted, status: accepted ? "accepted" : "rejected" }, metrics: report, metadata: { trainedAt: new Date().toISOString(), dataset: path.resolve(DATASET), counts, target: "observed monthly GloFAS discharge exceeds the location-specific pre-2017 95th percentile", requirements: { train: 5000, calibration: 750, test: 1500, balancedAccuracy: 0.6, f1: 0.5, brierSkill: 0 } } };
  const directory = path.join(ROOT, "models", "flood");
  fs.mkdirSync(directory, { recursive: true });
  fs.writeFileSync(path.join(directory, "glofas-plus1.candidate.json"), JSON.stringify(artifact));
  if (accepted) fs.writeFileSync(path.join(directory, "glofas-plus1.json"), JSON.stringify(artifact));
  console.log(`GloFAS +1 month flood forecast ${accepted ? "ACCEPTED" : "REJECTED"} | train/cal/test ${counts.train}/${counts.calibration}/${counts.test}`);
  console.log(`+1M | BA=${report.balancedAccuracy.toFixed(3)} F1=${report.f1.toFixed(3)} BSS=${report.brierSkill.toFixed(3)} threshold=${report.threshold.toFixed(2)}`);
  if (!accepted) process.exitCode = 2;
}

if (require.main === module) {
  try { main(); } catch (error) { console.error(error.message); process.exitCode = 1; }
}

module.exports = { features, loadRows, train };
