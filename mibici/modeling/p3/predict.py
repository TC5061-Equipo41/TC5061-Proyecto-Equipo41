
from pathlib import Path

import joblib
import pandas as pd
import typer
from loguru import logger

from mibici.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


PATTERN_COLUMNS = [
    "salidas_manana",
    "salidas_tarde",
    "salidas_noche",
    "salidas_madrugada",
    "salidas_fin_semana",
]


def prepare_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Prepare station features using the same transformations as training."""
    data = df.copy()

    total = data["viajes_salida"].replace(0, 1)

    for column in PATTERN_COLUMNS:
        data[f"{column}_proporcion"] = data[column] / total

    for column in ["viajes_salida", "viajes_llegada"]:
        data[column] = data[column].clip(lower=0)
        data[f"log_{column}"] = __import__("numpy").log1p(data[column])

    data[feature_columns] = data[feature_columns].replace(
        [float("inf"), float("-inf")], float("nan")
    )
    data[feature_columns] = data[feature_columns].fillna(
        data[feature_columns].median()
    )

    return data[feature_columns]


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "p3_station_features.csv",
    model_path: Path = MODELS_DIR / "p3_kmeans.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "p3_station_clusters.csv",
):
    """Assign stations to clusters using the trained P3 model."""
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    bundle = joblib.load(model_path)
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_columns = bundle["feature_columns"]

    raw = pd.read_csv(features_path)
    X = prepare_features(raw, feature_columns)
    labels = model.predict(scaler.transform(X))

    predictions = raw.copy()
    predictions["cluster"] = labels

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False)

    logger.success(f"Predictions saved to {predictions_path}")
    logger.info(f"Stations processed: {len(predictions)}")
    logger.info(
        "Stations per cluster:\n"
        + predictions["cluster"].value_counts().sort_index().to_string()
    )


if __name__ == "__main__":
    app()

