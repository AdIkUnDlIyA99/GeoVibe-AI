"""Train calibrated boosted-tree SPEI-3 drought classifiers for +1/+3/+6 months."""

import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score, precision_score, recall_score


DATASET = Path(os.environ.get("DROUGHT_DATASET", "/content/drive/MyDrive/drought-sequences-seasonal.jsonl"))
MODEL_OUTPUT = Path(os.environ.get("DROUGHT_TREE_MODEL_OUTPUT", "models/drought/drought-tree.joblib"))
METRICS_OUTPUT = Path(os.environ.get("DROUGHT_TREE_METRICS_OUTPUT", "models/drought/drought-tree-metrics.json"))
HORIZONS = (1, 3, 6)


def features(row):
    history = np.asarray([float(item["spei"]) for item in row["input"]], dtype=float)
    forecasts = row["seasonalForecast"]
    if len(history) != 12 or len(forecasts) != 6 or not np.isfinite(history).all():
        raise ValueError("Requires 12 SPEI observations and six seasonal forecast leads")
    values = [
        *history,
        float(history[-1]),
        float(np.mean(history[-3:])),
        float(np.mean(history[-6:])),
        float(np.std(history[-3:])),
        float(np.std(history[-6:])),
        float(history[-1] - history[-3]),
        float(history[-1] - history[-6]),
        float(np.min(history[-3:])),
        float(np.min(history[-6:])),
    ]
    month = int(str(row["originDate"])[5:7]) - 1
    values.extend((float(np.sin(2 * np.pi * month / 12)), float(np.cos(2 * np.pi * month / 12))))
    cumulative_temperature = cumulative_precipitation = 0.0
    for forecast in forecasts:
        temperature = float(forecast["t2m_anomaly"])
        precipitation = float(forecast["tprate_anomaly"])
        temperature_spread = float(forecast["t2m_spread"])
        precipitation_spread = float(forecast["tprate_spread"])
        cumulative_temperature += temperature
        cumulative_precipitation += precipitation
        values.extend((
            np.clip(temperature, -6, 6),
            np.clip(precipitation, -6, 6),
            np.clip(temperature_spread / 10, 0, 4),
            np.clip(precipitation_spread * 86400 / 10, 0, 4),
            np.clip(max(0, temperature) * max(0, -precipitation), 0, 12),
            np.clip(cumulative_temperature / forecast["leadMonth"], -6, 6),
            np.clip(cumulative_precipitation / forecast["leadMonth"], -6, 6),
        ))
    return np.asarray(values, dtype=np.float64)


def load_data():
    if not DATASET.is_file():
        raise FileNotFoundError(f"Missing drought dataset: {DATASET}")
    result = {horizon: {"train": [], "calibration": [], "test": []} for horizon in HORIZONS}
    with DATASET.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            row = json.loads(line)
            split, targets = row.get("split"), row.get("targets", {}).get("drought", [])
            if split not in ("train", "calibration", "test") or len(targets) != 3:
                raise ValueError(f"Invalid split or targets in row {line_number}")
            vector = features(row)
            if not np.isfinite(vector).all():
                continue
            for horizon, target in zip(HORIZONS, targets):
                if target not in (0, 1):
                    raise ValueError(f"Invalid drought target in row {line_number}")
                result[horizon][split].append((vector, int(target)))
    return result


def unpack(entries):
    values, labels = zip(*entries)
    return np.asarray(values), np.asarray(labels, dtype=np.int8)


def threshold_for(probability, labels):
    thresholds = np.linspace(0.05, 0.95, 181)
    return float(max(thresholds, key=lambda value: f1_score(labels, probability >= value, zero_division=0)))


def fit_horizon(horizon, splits):
    x_train, y_train = unpack(splits["train"])
    x_calibration, y_calibration = unpack(splits["calibration"])
    x_test, y_test = unpack(splits["test"])
    if min(len(y_train), len(y_calibration), len(y_test)) == 0 or len(np.unique(y_train)) != 2:
        raise ValueError(f"+{horizon}M does not have both label classes in every split")
    positives, negatives = int(y_train.sum()), int((y_train == 0).sum())
    weights = np.where(y_train == 1, len(y_train) / (2 * max(1, positives)), len(y_train) / (2 * max(1, negatives)))
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.5,
        random_state=20260924,
    ).fit(x_train, y_train, sample_weight=weights)
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(classifier.predict_proba(x_calibration)[:, 1], y_calibration)
    calibrated = calibrator.predict(classifier.predict_proba(x_calibration)[:, 1])
    threshold = threshold_for(calibrated, y_calibration)
    probability = calibrator.predict(classifier.predict_proba(x_test)[:, 1])
    predicted = probability >= threshold
    reference_brier = brier_score_loss(y_test, np.full(len(y_test), y_calibration.mean()))
    brier = brier_score_loss(y_test, probability)
    metrics = {
        "balancedAccuracy": float(balanced_accuracy_score(y_test, predicted)),
        "f1": float(f1_score(y_test, predicted, zero_division=0)),
        "precision": float(precision_score(y_test, predicted, zero_division=0)),
        "recall": float(recall_score(y_test, predicted, zero_division=0)),
        "brierScore": float(brier),
        "brierSkill": float(1 - brier / reference_brier) if reference_brier else 0.0,
        "threshold": threshold,
        "counts": {"train": len(y_train), "calibration": len(y_calibration), "test": len(y_test)},
    }
    return {"classifier": classifier, "calibrator": calibrator, "threshold": threshold}, metrics


def main():
    data = load_data()
    models, metrics = {}, {}
    for horizon in HORIZONS:
        models[str(horizon)], metrics[str(horizon)] = fit_horizon(horizon, data[horizon])
        item = metrics[str(horizon)]
        print(f"+{horizon}M | BA={item['balancedAccuracy']:.3f} F1={item['f1']:.3f} P={item['precision']:.3f} R={item['recall']:.3f} BSS={item['brierSkill']:.3f} threshold={item['threshold']:.2f}", flush=True)
    deployment = {horizon: item["balancedAccuracy"] >= 0.6 and item["f1"] >= 0.5 and item["brierSkill"] > 0 for horizon, item in metrics.items()}
    artifact = {"schemaVersion": 1, "architecture": "calibrated histogram gradient-boosted drought trees", "horizons": HORIZONS, "models": models, "metrics": metrics, "deployment": deployment}
    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_OUTPUT)
    METRICS_OUTPUT.write_text(json.dumps({"metrics": metrics, "deployment": deployment}, indent=2), encoding="utf-8")
    print(f"Saved model: {MODEL_OUTPUT}")
    print(f"Saved metrics: {METRICS_OUTPUT}")


if __name__ == "__main__":
    main()
