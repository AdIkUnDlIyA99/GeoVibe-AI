const HORIZONS = [1, 3, 6];

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const sigmoid = (value) => 1 / (1 + Math.exp(-clamp(value, -30, 30)));

function features(sequence, originDate) {
  if (!Array.isArray(sequence) || sequence.length !== 12) throw new Error("Drought forecast requires exactly 12 monthly observations");
  const history = sequence.map((item) => item.spei === null || item.spei === undefined || item.spei === "" ? NaN : Number(item.spei) / 3);
  if (!history.every(Number.isFinite)) throw new Error("Drought forecast requires finite SPEI values");
  const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;
  const slope = (values) => (values.at(-1) - values[0]) / Math.max(1, values.length - 1);
  const recent3 = history.slice(-3), recent6 = history.slice(-6);
  const month = new Date(originDate).getUTCMonth();
  return [
    1, ...history, history.at(-1), mean(recent3), mean(recent6),
    slope(recent3), slope(recent6), Math.min(...recent3), Math.min(...recent6),
    Math.sin(2 * Math.PI * month / 12), Math.cos(2 * Math.PI * month / 12)
  ];
}

function createDroughtForecast(featureCount) {
  return {
    schemaVersion: 1,
    architecture: "calibrated multi-horizon logistic SPEI-3 drought forecast",
    sequenceLength: 12,
    horizons: HORIZONS,
    heads: HORIZONS.map(() => ({ weights: Array(featureCount).fill(0) })),
    calibration: HORIZONS.map(() => ({ a: 1, b: 0 })),
    thresholds: HORIZONS.map(() => 0.5)
  };
}

function raw(head, input) {
  return head.weights.reduce((sum, weight, index) => sum + weight * input[index], 0);
}

function predictDroughtForecast(model, sequence, originDate) {
  const input = features(sequence, originDate);
  return model.heads.map((head, index) => {
    const calibration = model.calibration[index];
    return sigmoid(calibration.a * raw(head, input) + calibration.b);
  });
}

function trainDroughtForecast(rows, epochs = 300) {
  const sample = features(rows[0].input, rows[0].originDate);
  const model = createDroughtForecast(sample.length);
  const positiveWeights = HORIZONS.map((_, horizon) => {
    const positives = rows.reduce((sum, row) => sum + row.targets.drought[horizon], 0);
    return Math.sqrt((rows.length - positives) / Math.max(1, positives));
  });
  for (let epoch = 0; epoch < epochs; epoch += 1) {
    const rate = 0.025 / Math.sqrt(1 + epoch / 40);
    for (const row of rows) {
      const input = features(row.input, row.originDate);
      model.heads.forEach((head, horizon) => {
        const label = row.targets.drought[horizon];
        const error = (sigmoid(raw(head, input)) - label) * (label ? positiveWeights[horizon] : 1);
        head.weights.forEach((weight, index) => {
          head.weights[index] = clamp(weight - rate * (error * input[index] + 0.0008 * weight), -8, 8);
        });
      });
    }
  }
  return model;
}

function fitDroughtCalibration(model, rows, epochs = 300) {
  model.calibration = model.heads.map((head, horizon) => {
    let a = 1, b = 0;
    for (let epoch = 0; epoch < epochs; epoch += 1) {
      let gradientA = 0, gradientB = 0;
      for (const row of rows) {
        const score = raw(head, features(row.input, row.originDate));
        const error = sigmoid(a * score + b) - row.targets.drought[horizon];
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
    const scored = rows.map((row) => ({ probability: predictDroughtForecast(model, row.input, row.originDate)[horizon], label: row.targets.drought[horizon] }));
    let best = { threshold: 0.5, score: -Infinity };
    for (let threshold = 0.1; threshold <= 0.9; threshold += 0.01) {
      const metric = binaryMetrics(scored.map((item) => item.probability), scored.map((item) => item.label), threshold);
      const clearsMargin = metric.balancedAccuracy >= 0.62 && metric.f1 >= 0.55;
      const score = (clearsMargin ? 10 : 0) + metric.balancedAccuracy + metric.f1;
      if (score > best.score) best = { threshold, score };
    }
    return Number(best.threshold.toFixed(2));
  });
  return model;
}

function binaryMetrics(probabilities, labels, threshold) {
  let tp = 0, tn = 0, fp = 0, fn = 0, brier = 0;
  probabilities.forEach((probability, index) => {
    const predicted = probability >= threshold, label = labels[index];
    brier += (probability - label) ** 2;
    if (predicted && label) tp += 1;
    else if (!predicted && !label) tn += 1;
    else if (predicted) fp += 1;
    else fn += 1;
  });
  const precision = tp / Math.max(1, tp + fp), recall = tp / Math.max(1, tp + fn);
  const specificity = tn / Math.max(1, tn + fp);
  return { threshold, balancedAccuracy: (recall + specificity) / 2, precision, recall, f1: 2 * precision * recall / Math.max(1e-9, precision + recall), brierScore: brier / labels.length, confusionMatrix: { tp, tn, fp, fn } };
}

function evaluateDroughtForecast(model, rows) {
  return HORIZONS.map((_, horizon) => binaryMetrics(
    rows.map((row) => predictDroughtForecast(model, row.input, row.originDate)[horizon]),
    rows.map((row) => row.targets.drought[horizon]),
    model.thresholds[horizon]
  ));
}

module.exports = { HORIZONS, createDroughtForecast, evaluateDroughtForecast, features, fitDroughtCalibration, predictDroughtForecast, trainDroughtForecast };
