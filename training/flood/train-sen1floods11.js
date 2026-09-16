const fs = require("node:fs");
const path = require("node:path");
const { evaluateFloodCnn, fitPlatt, patchFeatures, trainFloodCnn } = require("../../src/ml/flood-cnn");

const BUCKET = "https://storage.googleapis.com/sen1floods11/v1.1";
const LIST_API = "https://storage.googleapis.com/storage/v1/b/sen1floods11/o";
const S2_PREFIX = "v1.1/data/flood_events/HandLabeled/S2Hand/";
const CHIPS_PER_COUNTRY = process.env.CHIPS_PER_COUNTRY ? Number(process.env.CHIPS_PER_COUNTRY) : Infinity;
const SIZE = Number(process.env.RASTER_SIZE || 512);
const ROWS_PER_CLASS_PER_CHIP = Number(process.env.ROWS_PER_CLASS_PER_CHIP || 512);

async function loadGeoTiff(url) {
  const { fromUrl } = await import("geotiff");
  const tiff = await fromUrl(url);
  return tiff.getImage();
}

async function chipRows(country, chip) {
  const root = `${BUCKET}/data/flood_events/HandLabeled`;
  const stem = `${country}_${chip}`;
  const [s2Image, labelImage, permanentImage] = await Promise.all([
    loadGeoTiff(`${root}/S2Hand/${stem}_S2Hand.tif`),
    loadGeoTiff(`${root}/LabelHand/${stem}_LabelHand.tif`),
    loadGeoTiff(`${root}/JRCWaterHand/${stem}_JRCWaterHand.tif`)
  ]);
  const [bands, labels, permanent] = await Promise.all([
    s2Image.readRasters({ width: SIZE, height: SIZE, interleave: false, resampleMethod: "bilinear" }),
    labelImage.readRasters({ width: SIZE, height: SIZE, interleave: true, resampleMethod: "nearest" }),
    permanentImage.readRasters({ width: SIZE, height: SIZE, interleave: true, resampleMethod: "nearest" })
  ]);
  const ratio = (a, b) => Math.abs(a + b) < 1e-9 ? 0 : Math.max(-1, Math.min(1, (a - b) / (a + b)));
  const ndvi = Array.from(bands[7], (nir, index) => ratio(nir, bands[3][index]));
  const ndwi = Array.from(bands[2], (green, index) => ratio(green, bands[7][index]));
  const positives = [];
  const negatives = [];
  for (let y = 1; y < SIZE - 1; y += 1) {
    for (let x = 1; x < SIZE - 1; x += 1) {
      const index = y * SIZE + x;
      if (labels[index] !== 0 && labels[index] !== 1) continue;
      const row = {
        features: patchFeatures(ndvi, ndwi, SIZE, SIZE, x, y),
        label: labels[index] === 1 && permanent[index] !== 1 ? 1 : 0,
        country,
        chip
      };
      (row.label ? positives : negatives).push(row);
    }
  }
  const selectAcross = (rows, limit) => {
    if (rows.length <= limit) return rows;
    const step = rows.length / limit;
    return Array.from({ length: limit }, (_, index) => rows[Math.floor(index * step)]);
  };
  return [
    ...selectAcross(positives, ROWS_PER_CLASS_PER_CHIP),
    ...selectAcross(negatives, ROWS_PER_CLASS_PER_CHIP)
  ];
}

async function mapLimit(values, limit, worker) {
  const output = Array(values.length);
  let cursor = 0;
  async function run() {
    while (cursor < values.length) {
      const index = cursor++;
      try { output[index] = await worker(values[index]); }
      catch (error) { console.warn(`Skipping ${values[index].country}/${values[index].chip}: ${error.message}`); output[index] = []; }
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, values.length) }, run));
  return output.flat();
}

async function listAllChips() {
  const groups = new Map();
  let pageToken;
  do {
    const endpoint = new URL(LIST_API);
    endpoint.searchParams.set("prefix", S2_PREFIX);
    endpoint.searchParams.set("maxResults", "1000");
    if (pageToken) endpoint.searchParams.set("pageToken", pageToken);
    const response = await fetch(endpoint);
    if (!response.ok) throw new Error(`Sen1Floods11 listing failed (${response.status})`);
    const listing = await response.json();
    for (const item of listing.items || []) {
      const match = item.name.match(/\/([^/]+)_([0-9]+)_S2Hand\.tif$/);
      if (!match) continue;
      if (!groups.has(match[1])) groups.set(match[1], new Set());
      groups.get(match[1]).add(match[2]);
    }
    pageToken = listing.nextPageToken;
  } while (pageToken);
  return groups;
}

async function main() {
  const groups = await listAllChips();
  const selected = [...groups].flatMap(([country, chipSet]) => [...chipSet].sort((a, b) => Number(a) - Number(b)).slice(0, CHIPS_PER_COUNTRY).map((chip) => ({ country, chip })));
  console.log(`Processing all ${selected.length} compatible full-resolution Sentinel-2 chips from ${groups.size} flood-event regions...`);
  console.log(`Raster ${SIZE}x${SIZE}; retaining up to ${ROWS_PER_CLASS_PER_CHIP} spatially distributed pixels per class and chip.`);
  if (process.env.INVENTORY_ONLY === "1") {
    console.log(JSON.stringify(Object.fromEntries([...groups].map(([country, chips]) => [country, chips.size])), null, 2));
    return;
  }
  const rows = await mapLimit(selected, 1, async ({ country, chip }) => {
    const result = await chipRows(country, chip);
    console.log(`[${country}/${chip}] ${result.length} training pixels retained`);
    return result;
  });
  const testCountries = ["India", "USA"];
  const calibrationCountries = ["Ghana"];
  const train = rows.filter((row) => !testCountries.includes(row.country) && !calibrationCountries.includes(row.country));
  const calibration = rows.filter((row) => calibrationCountries.includes(row.country));
  const test = rows.filter((row) => testCountries.includes(row.country));
  if (!train.length || !calibration.length || !test.length) throw new Error("Real-data geographic split is incomplete");
  const model = fitPlatt(trainFloodCnn(train), calibration);
  const metrics = evaluateFloodCnn(model, test);
  const artifact = {
    ...model,
    metrics,
    metadata: {
      task: "Temporary flood-water pixel classification",
      labelSource: "Sen1Floods11 v1.1 hand-labelled water minus JRC permanent-water mask",
      dataSource: "Real Sentinel-2 L1C image chips from documented flood events",
      split: "Geographic holdout by event country",
      trainingCountries: [...new Set(train.map((row) => row.country))],
      calibrationCountries,
      testCountries,
      trainingPixels: train.length,
      calibrationPixels: calibration.length,
      testPixels: test.length,
      chipsLoaded: selected.length,
      fullDatasetInventory: !Number.isFinite(CHIPS_PER_COUNTRY),
      sourceRasterSize: `${SIZE}x${SIZE}`,
      maximumRowsPerClassPerChip: ROWS_PER_CLASS_PER_CHIP,
      ingestion: "Every compatible Sen1Floods11 v1.1 hand-labelled Sentinel-2 chip; streamed directly from the public GCS bucket",
      weakLabelCaveat: "Temporary-water targets are derived by subtracting a permanent-water product from hand-labelled event water; they are not field-surveyed flood-depth measurements.",
      trainedAt: new Date().toISOString()
    }
  };
  const output = path.join(__dirname, "..", "..", "models", "flood", "flood-cnn.json");
  fs.writeFileSync(output, JSON.stringify(artifact));
  console.log(`Held-out India/USA accuracy ${(metrics.accuracy * 100).toFixed(2)}% | F1 ${(metrics.f1 * 100).toFixed(2)}% | IoU ${(metrics.iou * 100).toFixed(2)}% | Brier ${metrics.brierScore.toFixed(3)}`);
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
