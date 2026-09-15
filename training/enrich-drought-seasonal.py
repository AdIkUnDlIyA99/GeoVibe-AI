"""Join ECMWF seasonal ensemble summaries to SPEI drought sequences."""

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr


SOURCE = Path(os.environ.get("DROUGHT_DATASET", "/content/drive/MyDrive/drought-sequences-era5-spei.jsonl"))
SEASONAL_SOURCE = Path(os.environ.get("SEASONAL_SOURCE", "/content/drive/MyDrive/ecmwf-seasonal-drought-1981-2025"))
OUTPUT = Path(os.environ.get("DROUGHT_SEASONAL_OUTPUT", "/content/drive/MyDrive/drought-sequences-seasonal.jsonl"))
VARIABLES = ("t2m_mean", "t2m_spread", "tprate_mean", "tprate_spread")


def main():
    if not SOURCE.exists():
        raise RuntimeError(f"Drought dataset not found: {SOURCE}")
    if not SEASONAL_SOURCE.exists():
        raise RuntimeError(f"Seasonal source directory not found: {SEASONAL_SOURCE}")
    with SOURCE.open("r", encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    by_origin = defaultdict(list)
    for index, row in enumerate(rows):
        by_origin[row["originDate"][:7]].append(index)

    for number, (origin, indexes) in enumerate(sorted(by_origin.items()), 1):
        file = SEASONAL_SOURCE / f"seasonal-{origin}.nc"
        if not file.exists():
            raise RuntimeError(f"Missing seasonal forecast for origin {origin}: {file}")
        with xr.open_dataset(file) as dataset:
            missing = [name for name in VARIABLES if name not in dataset]
            if missing:
                raise RuntimeError(f"Missing variables in {file.name}: {missing}")
            latitudes = xr.DataArray([rows[index]["latitude"] for index in indexes], dims="points")
            longitudes = xr.DataArray([rows[index]["longitude"] for index in indexes], dims="points")
            selected = dataset[list(VARIABLES)].sel(latitude=latitudes, longitude=longitudes, method="nearest")
            for point, row_index in enumerate(indexes):
                forecasts = []
                for lead in range(6):
                    item = {name: float(selected[name].isel(points=point, step=lead).values) for name in VARIABLES}
                    if not all(np.isfinite(value) for value in item.values()):
                        raise RuntimeError(f"Non-finite seasonal value for row {row_index + 1}, lead {lead + 1}")
                    forecasts.append({"leadMonth": lead + 1, **{name: round(value, 8) for name, value in item.items()}})
                rows[row_index]["seasonalForecast"] = forecasts
        if number % 12 == 0 or number == len(by_origin):
            print(f"Enriched {number}/{len(by_origin)} forecast origins", flush=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, separators=(",", ":")) + "\n")
    temporary.replace(OUTPUT)
    print(f"Wrote {len(rows)} seasonally enriched rows to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
