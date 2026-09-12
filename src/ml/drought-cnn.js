const FILTERS = 8;
const KERNEL = 3;
const CHANNELS = 3;

const sigmoid = (value) => 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, value))));

function createDroughtCnn() {
  const filters = Array.from({ length: FILTERS }, (_, filter) => ({
    bias: 0,
    weights: Array.from({ length: CHANNELS * KERNEL }, (_, index) => ((((filter + 1) * 17 + index * 13) % 29) - 14) / 140)
  }));
  return {
    architecture: "1D temporal CNN with global-average pooling",
    sequenceLength: 12,
    channels: ["NDVI", "NDWI", "VALID_MASK"],
    kernelSize: KERNEL,
    filters,
    output: { bias: 0, weights: Array(FILTERS).fill(0) },
    calibration: { a: 1, b: 0 }
  };
}

function normalizeSequence(sequence, length = 12) {
  if (!Array.isArray(sequence) || sequence.length < length) throw new Error(`Drought CNN requires at least ${length} monthly observations`);
  return sequence.slice(-length).map((item) => [
    Math.max(-1, Math.min(1, Number(item.ndvi) || 0)),
    Math.max(-1, Math.min(1, Number(item.ndwi) || 0)),
    item.valid === false ? 0 : 1
  ]);
}

function forward(model, sequence) {
  const input = normalizeSequence(sequence, model.sequenceLength);
  const positions = input.length - model.kernelSize + 1;
  const preActivations = model.filters.map((filter) => Array.from({ length: positions }, (_, position) => {
    let value = filter.bias;
    for (let offset = 0; offset < model.kernelSize; offset += 1) {
      for (let channel = 0; channel < CHANNELS; channel += 1) value += input[position + offset][channel] * filter.weights[offset * CHANNELS + channel];
    }
    return value;
  }));
  const pooled = preActivations.map((values) => values.reduce((sum, value) => sum + Math.max(0, value), 0) / positions);
  const raw = model.output.bias + pooled.reduce((sum, value, index) => sum + value * model.output.weights[index], 0);
  return { input, pooled, positions, preActivations, raw };
}

function probability(model, sequence) {
  const raw = forward(model, sequence).raw;
  return sigmoid(model.calibration.a * raw + model.calibration.b);
}

function trainDroughtCnn(rows, epochs = 80) {
  const model = createDroughtCnn();
  const positives = rows.reduce((sum, row) => sum + row.label, 0);
  const positiveWeight = (rows.length - positives) / Math.max(1, positives);
  for (let epoch = 0; epoch < epochs; epoch += 1) {
    const rate = 0.035 / Math.sqrt(1 + epoch / 15);
    for (const row of rows) {
      const state = forward(model, row.sequence);
      const importance = row.label ? positiveWeight : 1;
      const error = (sigmoid(state.raw) - row.label) * importance;
      const previousOutput = model.output.weights.slice();
      model.output.bias -= rate * error;
      model.output.weights.forEach((weight, index) => { model.output.weights[index] -= rate * (error * state.pooled[index] + 0.0005 * weight); });
      model.filters.forEach((filter, filterIndex) => {
        let biasGradient = 0;
        const gradients = Array(filter.weights.length).fill(0);
        state.preActivations[filterIndex].forEach((pre, position) => {
          if (pre <= 0) return;
          const activationGradient = error * previousOutput[filterIndex] / state.positions;
          biasGradient += activationGradient;
          for (let offset = 0; offset < model.kernelSize; offset += 1) {
            for (let channel = 0; channel < CHANNELS; channel += 1) gradients[offset * CHANNELS + channel] += activationGradient * state.input[position + offset][channel];
          }
        });
        filter.bias -= rate * biasGradient;
        filter.weights.forEach((weight, index) => { filter.weights[index] -= rate * (gradients[index] + 0.0005 * weight); });
      });
    }
  }
  return model;
}

function fitCalibration(model, rows, epochs = 250) {
  let a = 1, b = 0;
  for (let epoch = 0; epoch < epochs; epoch += 1) {
    let gradientA = 0, gradientB = 0;
    for (const row of rows) {
      const raw = forward(model, row.sequence).raw;
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

function evaluateDroughtCnn(model, rows, threshold = 0.5) {
  let tp = 0, tn = 0, fp = 0, fn = 0, brier = 0;
  for (const row of rows) {
    const predicted = probability(model, row.sequence);
    const label = predicted >= threshold ? 1 : 0;
    brier += (predicted - row.label) ** 2;
    if (label && row.label) tp += 1;
    else if (!label && !row.label) tn += 1;
    else if (label) fp += 1;
    else fn += 1;
  }
  const accuracy = (tp + tn) / Math.max(1, rows.length);
  const precision = tp / Math.max(1, tp + fp);
  const recall = tp / Math.max(1, tp + fn);
  const specificity = tn / Math.max(1, tn + fp);
  return { samples: rows.length, accuracy, balancedAccuracy: (recall + specificity) / 2, precision, recall, f1: 2 * precision * recall / Math.max(1e-9, precision + recall), brierScore: brier / Math.max(1, rows.length), confusionMatrix: { tp, tn, fp, fn } };
}

module.exports = { createDroughtCnn, evaluateDroughtCnn, fitCalibration, forward, normalizeSequence, probability, trainDroughtCnn };
