# GeoVibe AI

GeoVibe AI is a global satellite environmental forecasting dashboard. It retrieves twelve months of real Copernicus Sentinel-2 L2A observations, calculates NDVI and NDWI, forecasts six months, and connects those trajectories to experimental flood and drought susceptibility models.

## Features

- Worldwide place search, coordinate entry, map click, and draggable pin
- One analysis date with a date-matched observed Sentinel-2 image
- NDVI vegetation monitoring
- NDWI surface-water monitoring
- Calibrated 3x3 flood-water CNN with a visible probability grid
- Change Passport with supported, uncertain, and insufficient-evidence decisions
- Capture ledger reporting requested dates, actual dates, offsets, cloud cover, and valid cells
- Honest held-out accuracy, balanced accuracy, F1, IoU, and Brier score
- Downloadable evidence JSON
- Trained drought CNN linked to projected NDVI/NDWI sequences, with its real held-out score disclosed
- Twelve observed months followed by six-month NDVI and NDWI forecasts with uncertainty intervals
- Connected +1, +3 and +6 month flood and drought susceptibility outlooks
- Zoomable and draggable trajectory graph
- Cloud filtering, spatial median sampling, capture-offset reporting, and abstention on incomplete grids
- Responsive animated dashboard and loading sequence

## Data Source

GeoVibe AI uses the Copernicus Sentinel-2 L2A catalog exposed through Microsoft Planetary Computer. Esri provides the interactive base map, and Nominatim provides worldwide location search.

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

The flood target is temporary water derived from Sen1Floods11 hand-labelled event water after removing JRC permanent water. It is not a field-surveyed flood-depth label. The drought CNN was trained on 286 samples, calibrated on 23, and tested on 32 geographically separated samples. Its current held-out balanced accuracy is 50%, so the interface labels drought susceptibility experimental rather than presenting it as a reliable warning.

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

### Drought CNN

`models/drought-cnn.json` contains the earlier present-state temporal CNN. Its 50% held-out balanced accuracy is below deployment quality. The new reproducible forecast pipeline supersedes it once an artifact passes all acceptance gates.

### Future Forecast CNN

The forecast pipeline builds real historical supervision in the form `12 observed months -> 6 later months`. One shared temporal CNN predicts six NDVI values, six NDWI values, and drought occurrence at +1, +3 and +6 months.

The trainer refuses deployment unless the candidate:

- Uses at least 1,000 training, 150 calibration and 300 test sequences
- Keeps regions disjoint across train, calibration and test
- Keeps every training origin earlier than the test period
- Beats persistence RMSE at every NDVI and NDWI horizon
- Reaches at least 60% balanced accuracy and 50% F1 at every drought horizon

Rejected candidates are written to `models/forecast-cnn.candidate.json` with a report, but never replace the runtime model. Accepted candidates are additionally written to `models/forecast-cnn.json` and are automatically loaded by the backend.

#### Google Colab

```python
!git clone https://github.com/AdIkUnDlIyA99/GeoVibe-AI.git /content/GeoVibe-AI
%cd /content/GeoVibe-AI
from google.colab import drive
drive.mount('/content/drive')
!pip install -r training/requirements-forecast.txt
%env SPEI_FILE=/content/drive/MyDrive/spei03.nc
%env FORECAST_OUTPUT=/content/drive/MyDrive/forecast-sequences.jsonl
%env FORECAST_SEQUENCES=5000
!npm run prepare:forecast
%env FORECAST_DATASET=/content/drive/MyDrive/forecast-sequences.jsonl
!npm run train:forecast
```

Preparation is resumable. Keep `forecast-sequences.jsonl` in Drive. After training, download `reports/forecast-evaluation.json` and either `models/forecast-cnn.json` when accepted or `models/forecast-cnn.candidate.json` when rejected.

## Test

```bash
npm test
```

## How To Analyze

1. Search for a worldwide location or position the map pin.
2. Select one analysis date between 2015 and today.
3. Run the analysis. GeoVibe retrieves twelve monthly observations ending near that date and matches the displayed capture within 40 days.
4. Read solid index lines as observations and dashed lines as the following six-month forecast.
5. Interpret NDVI as vegetation response and NDWI as surface-water response.
6. Treat flood-water output as pixel classification evidence, not an official event warning.
7. Treat drought and flood outputs as experimental model-linked susceptibility, not event predictions.
8. The hazard models were validated for present-state classification, not future-event forecasting.

## Important Limitations

- Satellite imagery and location search require internet access.
- Cloud cover and missing acquisitions can prevent date matching.
- NDVI and NDWI summarize a small sampled area and cannot identify every environmental cause.
- The CNN generalizes from a small geographically separated research dataset; it is not an operational disaster-alert system.
- A complete 3x3 valid Sentinel grid is required for CNN inference; otherwise GeoVibe abstains.
- Until `models/forecast-cnn.json` passes the acceptance gates, forecasts use a labelled statistical fallback and do not ingest future weather.

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

Satellite data: Copernicus Sentinel-2. Catalog and processing access: Microsoft Planetary Computer. Base map: Esri and its data contributors. Location search: OpenStreetMap Nominatim.
