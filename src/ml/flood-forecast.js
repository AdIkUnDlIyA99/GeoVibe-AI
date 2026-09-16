const HORIZONS = [1, 3, 6];

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const sigmoid = (value) => 1 / (1 + Math.exp(-clamp(value, -30, 30)));

function features(row, horizonIndex) {
  const lead = row.forecast?.[horizonIndex];
  if (!lead) throw new Error(`Flood forecast requires lead month ${HORIZONS[horizonIndex]}`);
  const values = [lead.mean, lead.spread, lead.p10, lead.p50, lead.p90, lead.exceedanceProbability, lead.thresholdRatio].map(Number);
  if (!values.every(Number.isFinite)) throw new Error("Flood forecast requires finite ensemble predictors");
  const month = new Date(row.originDate).getUTCMonth();
  const scale = (value) => Math.log1p(Math.max(0, value));
  return [
    1,
    scale(values[0]), scale(values[1]), scale(values[2]), scale(values[3]), scale(values[4]),
    clamp(values[5], 0, 1), clamp(values[6], 0, 10),
    Math.sin(2 * Math.PI * month / 12), Math.cos(2 * Math.PI * month / 12),
    clamp(Math.abs(Number(row.latitude)) / 90, 0, 1)
  ];
}

function createFloodForecast() {
  return {
    schemaVersion: 1,
    architecture: "calibrated multi-horizon GloFAS ensemble flood forecast",
    horizons: HORIZONS,
    heads: HORIZONS.map(() => ({ weights: Array(11).fill(0) })),
    calibration: HORIZONS.map(() => ({ a: 1, b: 0 })),
    thresholds: HORIZONS.map(() => 0.5)
  };
}

function raw(head, input) {
  return head.weights.reduce((sum, weight, index) => sum + weight * input[index], 0);
}

function predictFloodForecast(model, row) {
  return model.heads.map((head, index) => {
    const score = raw(head, features(row, index));
    const calibration = model.calibration[index] || { a: 1, b: 0 };
    return sigmoid(calibration.a * score + calibration.b);
  });
}

function trainFloodForecast(rows, epochs = 240) {
  const model = createFloodForecast();
  const positiveWeights = HORIZONS.map((_, horizon) => {
    const positives = rows.reduce((sum, row) => sum + Number(row.targets.flood[horizon]), 0);
    return Math.sqrt((rows.length - positives) / Math.max(1, positives));
  });
  for (let epoch = 0; epoch < epochs; epoch += 1) {
    const rate = 0.02 / Math.sqrt(1 + epoch / 40);
    for (const row of rows) {
      model.heads.forEach((head, horizon) => {
        const input = features(row, horizon);
        const label = Number(row.targets.flood[horizon]);
        const error = (sigmoid(raw(head, input)) - label) * (label ? positiveWeights[horizon] : 1);
        head.weights.forEach((weight, index) => {
          head.weights[index] = clamp(weight - rate * (error * input[index] + 0.0008 * weight), -8, 8);
        });
      });
    }
  }
  return model;
}

function binaryMetrics(probabilities, labels, threshold, climatology) {
  let tp = 0, tn = 0, fp = 0, fn = 0, brier = 0, baselineBrier = 0;
  probabilities.forEach((probability, index) => {
    const predicted = probability >= threshold, label = labels[index];
    brier += (probability - label) ** 2;
    baselineBrier += (climatology - label) ** 2;
    if (predicted && label) tp += 1;
    else if (!predicted && !label) tn += 1;
    else if (predicted) fp += 1;
    else fn += 1;
  });
  const precision = tp / Math.max(1, tp + fp), recall = tp / Math.max(1, tp + fn), specificity = tn / Math.max(1, tn + fp);
  const brierScore = brier / probabilities.length, climatologyBrier = baselineBrier / probabilities.length;
  return { threshold, balancedAccuracy: (recall + specificity) / 2, precision, recall, f1: 2 * precision * recall / Math.max(1e-9, precision + recall), brierScore, climatologyBrier, brierSkill: 1 - brierScore / Math.max(1e-9, climatologyBrier), confusionMatrix: { tp, tn, fp, fn } };
}

function fitFloodCalibration(model, rows, epochs = 250) {
  model.calibration = model.heads.map((head, horizon) => {
    let a = 1, b = 0;
    for (let epoch = 0; epoch < epochs; epoch += 1) {
      let gradientA = 0, gradientB = 0;
      for (const row of rows) {
        const score = raw(head, features(row, horizon));
        const error = sigmoid(a * score + b) - Number(row.targets.flood[horizon]);
        gradientA += error * score;
        gradientB += error;
      }
      const rate = 0.08 / Math.sqrt(1 + epoch / 60);
      a -= rate * gradientA / rows.length;
      b -= rate * gradientB / rows.length;
    }
    return { a, b };
  });
  model.thresholds = model.heads.map((_, horizon) => {
    const probabilities = rows.map((row) => predictFloodForecast(model, row)[horizon]);
    const labels = rows.map((row) => Number(row.targets.flood[horizon]));
    const climatology = labels.reduce((sum, label) => sum + label, 0) / labels.length;
    let best = { threshold: 0.5, score: -Infinity };
    for (let threshold = 0.05; threshold <= 0.9; threshold += 0.01) {
      const metric = binaryMetrics(probabilities, labels, threshold, climatology);
      const score = metric.balancedAccuracy + metric.f1;
      if (score > best.score) best = { threshold, score };
    }
    return Number(best.threshold.toFixed(2));
  });
  return model;
}

function evaluateFloodForecast(model, rows, calibrationRows) {
  return HORIZONS.map((_, horizon) => {
    const probabilities = rows.map((row) => predictFloodForecast(model, row)[horizon]);
    const labels = rows.map((row) => Number(row.targets.flood[horizon]));
    const calibrationLabels = calibrationRows.map((row) => Number(row.targets.flood[horizon]));
    const climatology = calibrationLabels.reduce((sum, label) => sum + label, 0) / calibrationLabels.length;
    return binaryMetrics(probabilities, labels, model.thresholds[horizon], climatology);
  });
}

module.exports = { HORIZONS, createFloodForecast, evaluateFloodForecast, features, fitFloodCalibration, predictFloodForecast, trainFloodForecast };
