const { inferGrid } = require("../ml/flood-cnn");

const clamp = (value, min = 0, max = 1) => Math.max(min, Math.min(max, value));
const direction = (value, threshold = 0.025) => Math.abs(value) < threshold ? 0 : Math.sign(value);

function median(values) {
  if (!values.length) return 0;
  const sorted = values.slice().sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function alignedChanges(before, after, key) {
  const baseline = new Map(before.spatialSamples.map((sample) => [sample.cell, sample[key]]));
  return after.spatialSamples.filter((sample) => baseline.has(sample.cell)).map((sample) => sample[key] - baseline.get(sample.cell));
}

function spatialAgreement(values) {
  const expected = direction(median(values));
  if (!expected) {
    const groups = [-1, 0, 1].map((candidate) => values.filter((value) => direction(value) === candidate).length);
    return Math.max(...groups) / Math.max(1, values.length);
  }
  return values.filter((value) => direction(value) === expected).length / Math.max(1, values.length);
}

function droughtSignal(observations) {
  const width = Math.max(1, Math.floor(observations.length / 3));
  const early = observations.slice(0, width);
  const late = observations.slice(-width);
  const deltaNdvi = median(late.map((item) => item.indices.ndvi)) - median(early.map((item) => item.indices.ndvi));
  const deltaNdwi = median(late.map((item) => item.indices.ndwi)) - median(early.map((item) => item.indices.ndwi));
  const score = clamp((-deltaNdvi * 2.4 - deltaNdwi * 1.6 + 0.08) / 0.55);
  return { label: score >= 0.67 ? "Elevated" : score >= 0.34 ? "Watch" : "Low", score: Math.round(score * 100), deltaNdvi: +deltaNdvi.toFixed(3), deltaNdwi: +deltaNdwi.toFixed(3) };
}

function buildPassport({ model, observations, beforeSample, afterSample }) {
  const beforeInference = inferGrid(model, beforeSample.spatialSamples);
  const afterInference = inferGrid(model, afterSample.spatialSamples);
  const stability = (spatialAgreement(alignedChanges(beforeSample, afterSample, "ndvi")) + spatialAgreement(alignedChanges(beforeSample, afterSample, "ndwi"))) / 2;
  const dateQuality = 1 - Math.min(1, (beforeSample.offsetDays + afterSample.offsetDays) / 80);
  const gridQuality = Math.min(beforeSample.sampleCount, afterSample.sampleCount) / 9;
  const cloudQuality = 1 - Math.min(1, (beforeSample.cloudCover + afterSample.cloudCover) / 90);
  const quality = Math.round(100 * (0.4 * gridQuality + 0.35 * dateQuality + 0.25 * cloudQuality));
  const waterChange = beforeInference && afterInference ? afterInference.temporaryWaterFraction - beforeInference.temporaryWaterFraction : null;
  const material = Math.abs(afterSample.indices.ndvi - beforeSample.indices.ndvi) >= 0.05 || Math.abs(afterSample.indices.ndwi - beforeSample.indices.ndwi) >= 0.05 || Math.abs(waterChange || 0) >= 0.08;
  let status = "uncertain";
  if (!afterInference || quality < 55) status = "insufficient";
  else if (material && stability >= 0.6) status = "supported";
  const conclusions = {
    supported: "Observed change is spatially consistent and supported by the available captures.",
    uncertain: "A change signal exists, but it is weak or spatially inconsistent.",
    insufficient: "The model abstained because capture or spatial evidence is incomplete."
  };
  return {
    schema: "GeoVibe Change Passport 1.0", status, conclusion: conclusions[status], quality,
    spatialStability: Math.round(stability * 100),
    indexChange: { ndvi: +(afterSample.indices.ndvi - beforeSample.indices.ndvi).toFixed(3), ndwi: +(afterSample.indices.ndwi - beforeSample.indices.ndwi).toFixed(3) },
    floodWater: afterInference ? { probabilities: afterInference.probabilities.map((value) => +value.toFixed(4)), temporaryWaterFraction: +afterInference.temporaryWaterFraction.toFixed(4), change: waterChange === null ? null : +waterChange.toFixed(4) } : null,
    drought: droughtSignal(observations),
    captures: [
      { role: "baseline", requestedDate: beforeSample.requestedDate, actualDate: beforeSample.date, offsetDays: beforeSample.offsetDays, cloudCover: beforeSample.cloudCover, spatialSamples: beforeSample.sampleCount },
      { role: "comparison", requestedDate: afterSample.requestedDate, actualDate: afterSample.date, offsetDays: afterSample.offsetDays, cloudCover: afterSample.cloudCover, spatialSamples: afterSample.sampleCount }
    ],
    caveats: ["Flood-water output is not an official flood warning.", "Drought is an index trend, not a trained event probability.", "Six-month trajectories are statistical projections."]
  };
}

module.exports = { buildPassport, droughtSignal, spatialAgreement };
