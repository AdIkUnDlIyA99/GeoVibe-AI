const FILTERS = 10;
const CHANNELS = 3;
const KERNEL = 3;
const FUTURE_MONTHS = 6;
const DROUGHT_HORIZONS = [1, 3, 6];
const POSITIONS = 12 - KERNEL + 1;
const FEATURES = FILTERS * POSITIONS;
const CLIMATE_FEATURES = 18;

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
  const head = (count, offset, features = FEATURES, zero = false) => Array.from({ length: count }, (_, output) => ({
    bias: 0,
    weights: Array.from({ length: features }, (_, index) => zero ? 0 : seededWeight(offset + output * 131 + index) * 0.35)
  }));
  return {
    schemaVersion: 4,
    architecture: "order-preserving satellite CNN with calibrated index baselines and lagged-SPEI trend drought heads",
    sequenceLength: 12,
    channels: ["NDVI", "NDWI", "VALID_MASK"],
    filters,
    indexHeads: { ndvi: head(FUTURE_MONTHS, 100, FEATURES, true), ndwi: head(FUTURE_MONTHS, 300, FEATURES, true) },
    indexBlend: { ndvi: Array(FUTURE_MONTHS).fill(1), ndwi: Array(FUTURE_MONTHS).fill(1) },
    indexBaselineMix: { ndvi: Array(FUTURE_MONTHS).fill(0), ndwi: Array(FUTURE_MONTHS).fill(0) },
    droughtHeads: head(DROUGHT_HORIZONS.length, 500, FEATURES + CLIMATE_FEATURES),
    droughtHorizons: DROUGHT_HORIZONS,
    calibration: DROUGHT_HORIZONS.map(() => ({ a: 1, b: 0 })),
    droughtThresholds: DROUGHT_HORIZONS.map(() => 0.5)
  };
}

function normalizeSequence(sequence) {
  if (!Array.isArray(sequence) || sequence.length !== 12) throw new Error("Forecast CNN requires exactly 12 monthly observations");
  const input = sequence.map((item) => [clamp(Number(item.ndvi) || 0), clamp(Number(item.ndwi) || 0), item.valid === false ? 0 : 1]);
  const hasClimate = sequence.every((item) => Number.isFinite(Number(item.spei)));
  const climateHistory = sequence.map((item) => hasClimate ? clamp(Number(item.spei) / 3) : 0);
  const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;
  const slope = (values) => (values.at(-1) - values[0]) / Math.max(1, values.length - 1);
  const recent3 = climateHistory.slice(-3);
  const recent6 = climateHistory.slice(-6);
  const climate = [
    ...climateHistory,
    climateHistory.at(-1),
    mean(recent3),
    mean(recent6),
    slope(recent3),
    slope(recent6),
    Math.min(...recent6)
  ];
  return { input, climate, hasClimate };
}

function encode(model, sequence) {
  const { input, climate, hasClimate } = normalizeSequence(sequence);
  const positions = input.length - KERNEL + 1;
  const pre = model.filters.map((filter) => Array.from({ length: positions }, (_, position) => {
    let value = filter.bias;
    for (let offset = 0; offset < KERNEL; offset += 1) {
      for (let channel = 0; channel < CHANNELS; channel += 1) value += input[position + offset][channel] * filter.weights[offset * CHANNELS + channel];
    }
    return value;
  }));
  // Keep filter position in the feature vector; averaging here destroys month order.
  const pooled = pre.flatMap((values) => values.map((value) => Math.max(0, value)));
  return { input, climate, hasClimate, positions, pre, pooled };
}

function linear(head, pooled) {
  return head.bias + pooled.reduce((sum, value, index) => sum + value * head.weights[index], 0);
}

function predictForecast(model, sequence) {
  const state = encode(model, sequence);
  const index = {};
  for (const key of ["ndvi", "ndwi"]) index[key] = model.indexHeads[key].map((head, horizon) => {
    const channel = key === "ndvi" ? 0 : 1;
    const persistence = state.input.at(-1)[channel];
    const seasonal = horizon === 0 ? persistence : state.input[horizon][channel];
    const persistenceMix = model.indexBaselineMix?.[key]?.[horizon] ?? 0;
    const baseline = seasonal * (1 - persistenceMix) + persistence * persistenceMix;
    const blend = model.indexBlend?.[key]?.[horizon] ?? 1;
    return clamp(baseline + blend * linear(head, state.pooled));
  });
  const droughtFeatures = [...state.pooled, ...state.climate];
  const drought = model.droughtHeads.map((head, index) => {
    if (!state.hasClimate) return null;
    const raw = linear(head, droughtFeatures);
    const calibration = model.calibration[index] || { a: 1, b: 0 };
    return sigmoid(calibration.a * raw + calibration.b);
  });
  return { index, drought };
}

function trainForecastCnn(rows, epochs = 100) {
  const model = createForecastCnn();
  const positiveWeight = DROUGHT_HORIZONS.map((_, index) => {
    const positives = rows.reduce((sum, row) => sum + row.targets.drought[index], 0);
    return Math.sqrt((rows.length - positives) / Math.max(1, positives));
  });
  for (let epoch = 0; epoch < epochs; epoch += 1) {
    const rate = 0.004 / Math.sqrt(1 + epoch / 20);
    for (const row of rows) {
      const state = encode(model, row.input);
      const pooledGradient = Array(FEATURES).fill(0);
      for (const key of ["ndvi", "ndwi"]) {
        model.indexHeads[key].forEach((head, horizon) => {
          const channel = key === "ndvi" ? 0 : 1;
          const baseline = horizon === 0 ? state.input.at(-1)[channel] : state.input[horizon][channel];
          const error = clamp(baseline + linear(head, state.pooled)) - row.targets[key][horizon];
          const previous = head.weights.slice();
          head.bias = clamp(head.bias - rate * clamp(error, -2, 2), -3, 3);
          head.weights.forEach((weight, index) => { head.weights[index] = clamp(weight - rate * clamp(error * state.pooled[index] + 0.0005 * weight, -2, 2), -3, 3); });
          previous.forEach((weight, index) => { pooledGradient[index] += 0.5 * error * weight; });
        });
      }
      model.droughtHeads.forEach((head, horizon) => {
        if (!state.hasClimate) throw new Error("Drought training requires 12 historical SPEI values");
        const label = row.targets.drought[horizon];
        const weight = label ? positiveWeight[horizon] : 1;
        const droughtFeatures = [...state.pooled, ...state.climate];
        const error = (sigmoid(linear(head, droughtFeatures)) - label) * weight;
        const previous = head.weights.slice();
        head.bias = clamp(head.bias - rate * clamp(error, -2, 2), -3, 3);
        head.weights.forEach((value, index) => { head.weights[index] = clamp(value - rate * clamp(error * droughtFeatures[index] + 0.0005 * value, -2, 2), -3, 3); });
        previous.slice(0, FEATURES).forEach((value, index) => { pooledGradient[index] += error * value; });
      });
      model.filters.forEach((filter, filterIndex) => {
        let biasGradient = 0;
        const gradients = Array(filter.weights.length).fill(0);
        state.pre[filterIndex].forEach((pre, position) => {
          if (pre <= 0) return;
          const activationGradient = pooledGradient[filterIndex * state.positions + position];
          biasGradient += activationGradient;
          for (let offset = 0; offset < KERNEL; offset += 1) {
            for (let channel = 0; channel < CHANNELS; channel += 1) gradients[offset * CHANNELS + channel] += activationGradient * state.input[position + offset][channel];
          }
        });
        filter.bias = clamp(filter.bias - rate * clamp(biasGradient, -2, 2), -3, 3);
        filter.weights.forEach((value, index) => { filter.weights[index] = clamp(value - rate * clamp(gradients[index] + 0.0005 * value, -2, 2), -3, 3); });
      });
    }
  }
  return model;
}

function fitForecastCalibration(model, rows, epochs = 250) {
  model.indexBlend = { ndvi: [], ndwi: [] };
  model.indexBaselineMix = { ndvi: [], ndwi: [] };
  for (const key of ["ndvi", "ndwi"]) {
    for (let horizon = 0; horizon < FUTURE_MONTHS; horizon += 1) {
      let best = { blend: 0, persistenceMix: 0, mse: Infinity };
      for (let persistenceMix = 0; persistenceMix <= 1.0001; persistenceMix += 0.1) {
        for (let blend = 0; blend <= 1.5001; blend += 0.05) {
          let squaredError = 0;
          for (const row of rows) {
            const state = encode(model, row.input);
            const channel = key === "ndvi" ? 0 : 1;
            const persistence = state.input.at(-1)[channel];
            const seasonal = horizon === 0 ? persistence : state.input[horizon][channel];
            const baseline = seasonal * (1 - persistenceMix) + persistence * persistenceMix;
            const predicted = clamp(baseline + blend * linear(model.indexHeads[key][horizon], state.pooled));
            squaredError += (predicted - row.targets[key][horizon]) ** 2;
          }
          const mse = squaredError / rows.length;
          if (mse < best.mse) best = { blend, persistenceMix, mse };
        }
      }
      model.indexBlend[key][horizon] = Number(best.blend.toFixed(2));
      model.indexBaselineMix[key][horizon] = Number(best.persistenceMix.toFixed(2));
    }
  }
  model.calibration = model.droughtHeads.map((head, horizon) => {
    let a = 1, b = 0;
    for (let epoch = 0; epoch < epochs; epoch += 1) {
      let gradientA = 0, gradientB = 0;
      for (const row of rows) {
        const state = encode(model, row.input);
        const raw = linear(head, [...state.pooled, ...state.climate]);
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
  model.droughtThresholds = model.droughtHeads.map((_, horizon) => {
    const scored = rows.map((row) => ({ probability: predictForecast(model, row.input).drought[horizon], label: row.targets.drought[horizon] }));
    let best = { threshold: 0.5, score: -Infinity };
    for (let threshold = 0.1; threshold <= 0.9; threshold += 0.01) {
      let tp = 0, tn = 0, fp = 0, fn = 0;
      for (const item of scored) {
        const predicted = item.probability >= threshold;
        if (predicted && item.label) tp += 1;
        else if (!predicted && !item.label) tn += 1;
        else if (predicted) fp += 1;
        else fn += 1;
      }
      const recall = tp / Math.max(1, tp + fn), specificity = tn / Math.max(1, tn + fp);
      const precision = tp / Math.max(1, tp + fp), f1 = 2 * precision * recall / Math.max(1e-9, precision + recall);
      const balancedAccuracy = (recall + specificity) / 2;
      // Select on calibration only, with margin above both deployment floors.
      const clearsSafetyMargin = balancedAccuracy >= 0.62 && f1 >= 0.55;
      const score = (clearsSafetyMargin ? 10 : 0) + balancedAccuracy + f1;
      if (score > best.score) best = { threshold, score };
    }
    return best.threshold;
  });
  return model;
}

module.exports = { createForecastCnn, fitForecastCalibration, predictForecast, trainForecastCnn };
