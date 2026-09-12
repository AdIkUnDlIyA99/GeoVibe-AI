const fs = require("node:fs");
const path = require("node:path");
const { evaluateDroughtCnn, fitCalibration, trainDroughtCnn } = require("../src/ml/drought-cnn");

const DATASET = process.env.DROUGHT_DATASET || path.join(__dirname, "data", "drought-sequences.jsonl");

function loadRows(file) {
  if (!fs.existsSync(file)) throw new Error(`Missing SPEI-labelled dataset: ${file}`);
  return fs.readFileSync(file, "utf8").split(/\r?\n/).filter(Boolean).map((line, index) => {
    const item = JSON.parse(line);
    if (!item.region || !["train", "calibration", "test"].includes(item.split)) throw new Error(`Row ${index + 1} requires region and train/calibration/test split`);
    if (!Array.isArray(item.sequence) || item.sequence.length < 12 || !Number.isFinite(item.spei)) throw new Error(`Row ${index + 1} requires 12 monthly NDVI/NDWI values and SPEI`);
    return { ...item, label: item.spei <= -1 ? 1 : 0 };
  });
}

function main() {
  const rows = loadRows(DATASET);
  const split = (name) => rows.filter((row) => row.split === name);
  const train = split("train"), calibration = split("calibration"), test = split("test");
  if (!train.length || !calibration.length || !test.length) throw new Error("All three geographic splits must contain data");
  const regions = (items) => new Set(items.map((item) => item.region));
  const trainRegions = regions(train), heldOutRegions = new Set([...regions(calibration), ...regions(test)]);
  if ([...heldOutRegions].some((region) => trainRegions.has(region))) throw new Error("Geographic leakage detected: a region occurs in training and held-out data");
  const model = fitCalibration(trainDroughtCnn(train), calibration);
  const metrics = evaluateDroughtCnn(model, test);
  const artifact = { ...model, metrics, metadata: { status: "trained", task: "Drought occurrence classification", target: "SPEI_03_month <= -1.0", inputs: "12 monthly Sentinel-2 NDVI/NDWI observations", split: "Disjoint geographic regions", trainingRegions: [...trainRegions], calibrationRegions: [...regions(calibration)], testRegions: [...regions(test)], trainingSamples: train.length, calibrationSamples: calibration.length, testSamples: test.length, trainedAt: new Date().toISOString() } };
  const output = path.join(__dirname, "..", "models", "drought-cnn.json");
  fs.writeFileSync(output, JSON.stringify(artifact));
  console.log(`Drought model saved: balanced accuracy ${(metrics.balancedAccuracy * 100).toFixed(2)}% | F1 ${(metrics.f1 * 100).toFixed(2)}% | Brier ${metrics.brierScore.toFixed(3)}`);
}

try { main(); } catch (error) { console.error(error.message); process.exitCode = 1; }
