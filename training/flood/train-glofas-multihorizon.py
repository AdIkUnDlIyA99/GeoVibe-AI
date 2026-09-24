"""Train calibrated gradient-boosted GloFAS classifiers for +1/+3/+6 months."""

import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score, precision_score, recall_score


DATASET = Path(os.environ.get("FLOOD_FORECAST_DATASET", "/content/drive/MyDrive/glofas-flood-sequences-1-3-6.jsonl"))
OUTPUT = Path(os.environ.get("FLOOD_TREE_MODEL_OUTPUT", "models/flood/glofas-multihorizon-tree.joblib"))
METRICS_OUTPUT = Path(os.environ.get("FLOOD_TREE_METRICS_OUTPUT", "models/flood/glofas-multihorizon-tree-metrics.json"))
HORIZONS = (1, 3, 6)
FEATURES = ("logMean", "logSpread", "logP10", "logP50", "logP90", "exceedanceProbability", "thresholdRatio")


def feature_vector(forecast):
    """Log-transform discharge values while retaining ensemble uncertainty features."""
    return [
        np.log1p(max(0.0, float(forecast["mean"]))),
        np.log1p(max(0.0, float(forecast["spread"]))),
        np.log1p(max(0.0, float(forecast["p10"]))),
        np.log1p(max(0.0, float(forecast["p50"]))),
        np.log1p(max(0.0, float(forecast["p90"]))),
        float(forecast["exceedanceProbability"]),
        float(forecast["thresholdRatio"]),
    ]


def load_rows():
    if not DATASET.is_file():
        raise FileNotFoundError(f"Missing dataset: {DATASET}")
    rows = {horizon: {"train": [], "calibration": [], "test": []} for horizon in HORIZONS}
    with DATASET.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            row = json.loads(line)
            split = row.get("split")
            forecasts, targets = row.get("forecast", []), row.get("targets", {}).get("flood", [])
            if split not in ("train", "calibration", "test") or len(forecasts) != 3 or len(targets) != 3:
                raise ValueError(f"Invalid row {line_number}")
            for forecast, target in zip(forecasts, targets):
                horizon = int(forecast["leadMonth"])
                if horizon not in rows or target not in (0, 1):
                    raise ValueError(f"Invalid lead or target in row {line_number}")
                values = feature_vector(forecast)
                if not np.isfinite(values).all():
                    continue
                rows[horizon][split].append((values, int(target)))
    return rows


def arrays(entries):
    x, y = zip(*entries)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.int8)


def best_threshold(probability, target):
    candidates = np.linspace(0.05, 0.95, 181)
    scores = [f1_score(target, probability >= threshold, zero_division=0) for threshold in candidates]
    return float(candidates[int(np.argmax(scores))])


def train_horizon(horizon, splits):
    x_train, y_train = arrays(splits["train"])
    x_cal, y_cal = arrays(splits["calibration"])
    x_test, y_test = arrays(splits["test"])
    if min(len(x_train), len(x_cal), len(x_test)) == 0 or len(np.unique(y_train)) < 2:
        raise ValueError(f"+{horizon}M has insufficient class coverage")

    # Balance rare flood labels without discarding the much larger no-flood history.
    positives = max(1, int(y_train.sum()))
    negatives = max(1, len(y_train) - positives)
    weights = np.where(y_train == 1, len(y_train) / (2 * positives), len(y_train) / (2 * negatives))
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=250,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )
    classifier.fit(x_train, y_train, sample_weight=weights)

    # Isotonic calibration and threshold choice use calibration data only.
    raw_cal = classifier.predict_proba(x_cal)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw_cal, y_cal)
    calibrated_cal = calibrator.predict(raw_cal)
    threshold = best_threshold(calibrated_cal, y_cal)
    probability = calibrator.predict(classifier.predict_proba(x_test)[:, 1])
    predicted = probability >= threshold
    climatology_brier = brier_score_loss(y_test, np.full(len(y_test), y_cal.mean()))
    brier = brier_score_loss(y_test, probability)
    metrics = {
        "balancedAccuracy": float(balanced_accuracy_score(y_test, predicted)),
        "f1": float(f1_score(y_test, predicted, zero_division=0)),
        "precision": float(precision_score(y_test, predicted, zero_division=0)),
        "recall": float(recall_score(y_test, predicted, zero_division=0)),
        "brierScore": float(brier),
        "brierSkill": float(1 - brier / climatology_brier) if climatology_brier else 0.0,
        "threshold": threshold,
        "counts": {"train": len(y_train), "calibration": len(y_cal), "test": len(y_test), "testPositive": int(y_test.sum())},
    }
    return {"classifier": classifier, "calibrator": calibrator, "threshold": threshold}, metrics


def main():
    rows = load_rows()
    models, metrics = {}, {}
    for horizon in HORIZONS:
        models[str(horizon)], metrics[str(horizon)] = train_horizon(horizon, rows[horizon])
        result = metrics[str(horizon)]
        print(
            f"+{horizon}M | BA={result['balancedAccuracy']:.3f} F1={result['f1']:.3f} "
            f"P={result['precision']:.3f} R={result['recall']:.3f} "
            f"BSS={result['brierSkill']:.3f} threshold={result['threshold']:.2f}",
            flush=True,
        )
    accepted = {horizon: value["balancedAccuracy"] >= 0.6 and value["f1"] >= 0.5 and value["brierSkill"] > 0 for horizon, value in metrics.items()}
    artifact = {"schemaVersion": 1, "modelType": "HistGradientBoostingClassifier with isotonic calibration", "features": FEATURES, "models": models, "metrics": metrics, "deployment": accepted}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, OUTPUT)
    METRICS_OUTPUT.write_text(json.dumps({"metrics": metrics, "deployment": accepted}, indent=2), encoding="utf-8")
    print(f"Saved model: {OUTPUT}")
    print(f"Saved metrics: {METRICS_OUTPUT}")


if __name__ == "__main__":
    main()
