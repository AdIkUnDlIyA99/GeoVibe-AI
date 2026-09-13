const FILTERS = 10;
const CHANNELS = 3;
const KERNEL = 3;
const FUTURE_MONTHS = 6;
const DROUGHT_HORIZONS = [1, 3, 6];

const clamp = (value, min = -1, max = 1) => Math.max(min, Math.min(max, value));
const sigmoid = (value) => 1 / (1 + Math.exp(-clamp(value, -30, 30)));

function seededWeight(index) {
  return ((((index + 11) * 29) % 41) - 20) / 220;
}

function createForecastCnn() {
  const filters = Array.from({ length: FILTERS }, (_, filter) => ({
    bias: 0,
    weights: Array.from({ length: CHANNELS * KERNEL }, (_, index) => seededWeight(filter * 17 + index))
  }));
  const head = (count, offset) => Array.from({ length: count }, (_, output) => ({
    bias: 0,
    weights: Array.from({ length: FILTERS }, (_, index) => seededWeight(offset + output * 13 + index))
  }));
  return {
    schemaVersion: 1,
    architecture: "shared 1D temporal CNN with index-regression and drought-classification heads",
    sequenceLength: 12,
    channels: ["NDVI", "NDWI", "VALID_MASK"],
    filters,
    indexHeads: { ndvi: head(FUTURE_MONTHS, 100), ndwi: head(FUTURE_MONTHS, 300) },
    droughtHeads: head(DROUGHT_HORIZONS.length, 500),
    droughtHorizons: DROUGHT_HORIZONS,
    calibration: DROUGHT_HORIZONS.map(() => ({ a: 1, b: 0 }))
  };
}

function normalizeSequence(sequence) {
  if (!Array.isArray(sequence) || sequence.length !== 12) throw new Error("Forecast CNN requires exactly 12 monthly observations");
  return sequence.map((item) => [clamp(Number(item.ndvi) || 0), clamp(Number(item.ndwi) || 0), item.valid === false ? 0 : 1]);
}

function encode(model, sequence) {
  const input = normalizeSequence(sequence);
  const positions = input.length - KERNEL + 1;
  const pre = model.filters.map((filter) => Array.from({ length: positions }, (_, position) => {
    let value = filter.bias;
    for (let offset = 0; offset < KERNEL; offset += 1) {
      for (let channel = 0; channel < CHANNELS; channel += 1) value += input[position + offset][channel] * filter.weights[offset * CHANNELS + channel];
    }
    return value;
  }));
  const pooled = pre.map((values) => values.reduce((sum, value) => sum + Math.max(0, value), 0) / positions);
  return { input, positions, pre, pooled };
}

function linear(head, pooled) {
  return head.bias + pooled.reduce((sum, value, index) => sum + value * head.weights[index], 0);
}

function predictForecast(model, sequence) {
  const state = encode(model, sequence);
  const index = {};
  for (const key of ["ndvi", "ndwi"]) index[key] = model.indexHeads[key].map((head) => clamp(linear(head, state.pooled)));
  const drought = model.droughtHeads.map((head, index) => {
    const raw = linear(head, state.pooled);
    const calibration = model.calibration[index] || { a: 1, b: 0 };
    return sigmoid(calibration.a * raw + calibration.b);
  });
  return { index, drought };
}

function trainForecastCnn(rows, epochs = 100) {
  const model = createForecastCnn();
  const positiveWeight = DROUGHT_HORIZONS.map((_, index) => {
    const positives = rows.reduce((sum, row) => sum + row.targets.drought[index], 0);
    return (rows.length - positives) / Math.max(1, positives);
  });
  for (let epoch = 0; epoch < epochs; epoch += 1) {
    const rate = 0.025 / Math.sqrt(1 + epoch / 20);
    for (const row of rows) {
      const state = encode(model, row.input);
      const pooledGradient = Array(FILTERS).fill(0);
      for (const key of ["ndvi", "ndwi"]) {
        model.indexHeads[key].forEach((head, horizon) => {
          const error = clamp(linear(head, state.pooled)) - row.targets[key][horizon];
          const previous = head.weights.slice();
          head.bias -= rate * error;
          head.weights.forEach((weight, index) => { head.weights[index] -= rate * (error * state.pooled[index] + 0.0005 * weight); });
          previous.forEach((weight, index) => { pooledGradient[index] += 0.5 * error * weight; });
        });
      }
      model.droughtHeads.forEach((head, horizon) => {
        const label = row.targets.drought[horizon];
        const weight = label ? positiveWeight[horizon] : 1;
        const error = (sigmoid(linear(head, state.pooled)) - label) * weight;
        const previous = head.weights.slice();
        head.bias -= rate * error;
        head.weights.forEach((value, index) => { head.weights[index] -= rate * (error * state.pooled[index] + 0.0005 * value); });
        previous.forEach((value, index) => { pooledGradient[index] += error * value; });
      });
      model.filters.forEach((filter, filterIndex) => {
        let biasGradient = 0;
        const gradients = Array(filter.weights.length).fill(0);
        state.pre[filterIndex].forEach((pre, position) => {
          if (pre <= 0) return;
          const activationGradient = pooledGradient[filterIndex] / state.positions;
          biasGradient += activationGradient;
          for (let offset = 0; offset < KERNEL; offset += 1) {
            for (let channel = 0; channel < CHANNELS; channel += 1) gradients[offset * CHANNELS + channel] += activationGradient * state.input[position + offset][channel];
          }
        });
        filter.bias -= rate * biasGradient;
        filter.weights.forEach((value, index) => { filter.weights[index] -= rate * (gradients[index] + 0.0005 * value); });
      });
    }
  }
  return model;
}

function fitForecastCalibration(model, rows, epochs = 250) {
  model.calibration = model.droughtHeads.map((head, horizon) => {
    let a = 1, b = 0;
    for (let epoch = 0; epoch < epochs; epoch += 1) {
      let gradientA = 0, gradientB = 0;
      for (const row of rows) {
        const raw = linear(head, encode(model, row.input).pooled);
        const error = sigmoid(a * raw + b) - row.targets.drought[horizon];
        gradientA += error * raw;
        gradientB += error;
      }
      const rate = 0.08 / Math.sqrt(1 + epoch / 50);
      a -= rate * gradientA / rows.length;
      b -= rate * gradientB / rows.length;
    }
    return { a, b };
  });
  return model;
}

module.exports = { createForecastCnn, fitForecastCalibration, predictForecast, trainForecastCnn };
