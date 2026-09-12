# GeoVibe AI

GeoVibe AI is a global satellite environmental monitoring dashboard. It retrieves real Copernicus Sentinel-2 L2A observations, calculates NDVI and NDWI, compares selected dates, applies a calibrated flood-water CNN trained on real event imagery, and generates an auditable Change Passport.

## Features

- Worldwide place search, coordinate entry, map click, and draggable pin
- Date-matched Sentinel-2 baseline and comparison imagery
- NDVI vegetation monitoring
- NDWI surface-water monitoring
- Calibrated 3x3 flood-water CNN with a visible probability grid
- Change Passport with supported, uncertain, and insufficient-evidence decisions
- Capture ledger reporting requested dates, actual dates, offsets, cloud cover, and valid cells
- Honest held-out accuracy, balanced accuracy, F1, IoU, and Brier score
- Downloadable evidence JSON
- NDVI/NDWI-derived drought trend signal, explicitly separated from trained probabilities
- Six-month NDVI and NDWI statistical trajectories with uncertainty intervals
- Zoomable and draggable trajectory graph
- Cloud filtering, spatial median sampling, capture-offset reporting, and abstention on incomplete grids
- Responsive animated dashboard and loading sequence

## Data Source

GeoVibe AI uses the Copernicus Sentinel-2 L2A catalog exposed through Microsoft Planetary Computer. OpenStreetMap provides the interactive base map, and Nominatim provides worldwide location search.

## Tech Stack

- Frontend: HTML5, CSS3, vanilla JavaScript, SVG, Canvas, Leaflet
- Backend: Node.js HTTP server
- AI/ML: calibrated 3x3 fully convolutional NDVI/NDWI flood-water classifier and statistical time-series forecasting
- Data: Copernicus Sentinel-2 L2A

## Real Model Validation

The flood-water model is trained from 88 real Sentinel-2 flood-event chips in Sen1Floods11. Training, calibration, and testing are separated geographically. India and USA are held out for testing.

| Metric | Held-out result |
|---|---:|
| Accuracy | 87.62% |
| Balanced accuracy | 76.83% |
| Precision | 65.89% |
| Recall | 60.26% |
| F1 | 62.95% |
| IoU | 45.93% |
| Brier score | 0.097 |

The target is temporary flood water derived from Sen1Floods11 hand-labelled event water after removing JRC permanent water. It is not a field-surveyed flood-depth label. The drought signal is deterministic index analysis and has no claimed classifier accuracy.

## Run Locally

Requirements: Node.js 20 or newer and an active internet connection.

```bash
npm start
```

Open `http://127.0.0.1:8501` and keep the terminal running.

## Train Models

```bash
npm run train
```

By default, training enumerates every compatible hand-labelled Sentinel-2 chip in the complete Sen1Floods11 v1.1 public bucket, reads its 512x512 source and label rasters directly over HTTPS, and retains spatially distributed balanced pixels. It does not require Google Cloud Console credentials and does not store the 14 GB bucket locally.

Verify the complete remote inventory without training:

```bash
npm run train:inventory
```

For a smaller diagnostic run only, set `CHIPS_PER_COUNTRY`. A normal `npm run train` leaves this unset and therefore uses the full compatible inventory.

## Test

```bash
npm test
```

## How To Analyze

1. Search for a worldwide location or position the map pin.
2. Select a focused baseline and comparison range, preferably 3 to 36 months.
3. Keep both dates between 2015 and today.
4. Run the analysis. GeoVibe searches for clear Sentinel-2 imagery within 40 days of each selected date.
5. Interpret NDVI as vegetation response and NDWI as surface-water response.
6. Treat flood-water output as pixel classification evidence, not an official event warning.
7. Treat the drought score as an NDVI/NDWI trend signal, not a trained probability.
8. Treat the dashed six-month lines as statistical projections, not weather forecasts.

## Important Limitations

- Satellite imagery and location search require internet access.
- Cloud cover and missing acquisitions can prevent date matching.
- NDVI and NDWI summarize a small sampled area and cannot identify every environmental cause.
- The CNN generalizes from a small geographically separated research dataset; it is not an operational disaster-alert system.
- A complete 3x3 valid Sentinel grid is required for CNN inference; otherwise GeoVibe abstains.
- Forecasts extrapolate observed index behavior and do not ingest future weather.

## Project Structure

```text
geovibe-ai/
|-- public/                  # Browser application and static assets
|   `-- assets/              # Visual assets
|-- src/
|   |-- analysis/            # Change Passport and trajectory analysis
|   `-- ml/                  # Model architecture and inference
|-- training/                # Reproducible dataset and training pipelines
|-- models/                  # Deployable model artifacts
|-- tests/                   # Automated unit and API tests
|-- server.js                # API, Sentinel retrieval and static server
|-- package.json             # Commands and runtime dependencies
|-- package-lock.json        # Reproducible dependency lock
`-- README.md                # Project documentation
```

## License And Attribution

Satellite data: Copernicus Sentinel-2. Catalog and processing access: Microsoft Planetary Computer. Base map: OpenStreetMap contributors.
