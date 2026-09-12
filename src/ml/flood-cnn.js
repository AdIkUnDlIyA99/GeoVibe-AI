function sigmoid(value) {
  return 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, value))));
}

function patchFeatures(ndvi, ndwi, width, height, x, y) {
  const features = [];
  for (const channel of [ndvi, ndwi]) {
    for (let dy = -1; dy <= 1; dy += 1) {
      for (let dx = -1; dx <= 1; dx += 1) {
        const px = Math.max(0, Math.min(width - 1, x + dx));
        const py = Math.max(0, Math.min(height - 1, y + dy));
        features.push(channel[py * width + px]);
      }
    }
  }
  return features;
}

function score(model, features) {
  return model.weights[0] + features.reduce((sum, value, index) => sum + value * model.weights[index + 1], 0);
}

function rawProbability(model, features) {
  return sigmoid(score(model, features));
}

function probability(model, features) {
  const raw = score(model, features);
  return sigmoid(model.calibration.a * raw + model.calibration.b);
}

function trainFloodCnn(rows, epochs = 180) {
  const weights = Array(19).fill(0);
  const positives = rows.reduce((sum, row) => sum + row.label, 0);
  const negatives = rows.length - positives;
  const positiveWeight = negatives / Math.max(1, positives);
  for (let epoch = 0; epoch < epochs; epoch += 1) {
    const gradient = Array(weights.length).fill(0);
    for (const row of rows) {
      const values = [1, ...row.features];
      const predicted = sigmoid(values.reduce((sum, value, index) => sum + value * weights[index], 0));
      const importance = row.label ? positiveWeight : 1;
      const error = (predicted - row.label) * importance;
      values.forEach((value, index) => { gradient[index] += error * value; });
    }
    const rate = 0.16 / Math.sqrt(1 + epoch / 20);
    weights.forEach((weight, index) => {
      const penalty = index === 0 ? 0 : 0.001 * weight;
      weights[index] -= rate * (gradient[index] / rows.length + penalty);
    });
  }
  return { architecture: "1-layer 3x3 fully convolutional NDVI/NDWI network", channels: ["NDVI", "NDWI"], kernelSize: 3, weights, calibration: { a: 1, b: 0 } };
}

function fitPlatt(model, rows, epochs = 300) {
  let a = 1;
  let b = 0;
  for (let epoch = 0; epoch < epochs; epoch += 1) {
    let gradientA = 0;
    let gradientB = 0;
    for (const row of rows) {
      const raw = score(model, row.features);
      const error = sigmoid(a * raw + b) - row.label;
      gradientA += error * raw;
      gradientB += error;
    }
    const rate = 0.08 / Math.sqrt(1 + epoch / 50);
    a -= rate * gradientA / rows.length;
    b -= rate * gradientB / rows.length;
  }
  model.calibration = { a, b };
  return model;
}

function evaluateFloodCnn(model, rows, threshold = 0.5) {
  let tp = 0, tn = 0, fp = 0, fn = 0, brier = 0;
  for (const row of rows) {
    const predicted = probability(model, row.features);
    const label = predicted >= threshold ? 1 : 0;
    brier += (predicted - row.label) ** 2;
    if (label === 1 && row.label === 1) tp += 1;
    else if (label === 0 && row.label === 0) tn += 1;
    else if (label === 1) fp += 1;
    else fn += 1;
  }
  const accuracy = (tp + tn) / Math.max(1, rows.length);
  const precision = tp / Math.max(1, tp + fp);
  const recall = tp / Math.max(1, tp + fn);
  return {
    samples: rows.length,
    accuracy,
    balancedAccuracy: (recall + tn / Math.max(1, tn + fp)) / 2,
    precision,
    recall,
    f1: 2 * precision * recall / Math.max(1e-9, precision + recall),
    iou: tp / Math.max(1, tp + fp + fn),
    brierScore: brier / Math.max(1, rows.length),
    confusionMatrix: { tp, tn, fp, fn }
  };
}

function inferGrid(model, samples) {
  const ordered = samples.slice().sort((a, b) => a.cell - b.cell);
  if (ordered.length !== 9) return null;
  const ndvi = ordered.map((sample) => sample.ndvi);
  const ndwi = ordered.map((sample) => sample.ndwi);
  const probabilities = ordered.map((_, index) => probability(model, patchFeatures(ndvi, ndwi, 3, 3, index % 3, Math.floor(index / 3))));
  return {
    probabilities,
    temporaryWaterFraction: probabilities.reduce((sum, value) => sum + value, 0) / probabilities.length,
    peakProbability: Math.max(...probabilities)
  };
}

module.exports = { evaluateFloodCnn, fitPlatt, inferGrid, patchFeatures, probability, trainFloodCnn };
