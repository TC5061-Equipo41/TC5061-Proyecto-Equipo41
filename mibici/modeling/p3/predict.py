from pathlib import Path

import joblib
import numpy as np
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


def prepare_features(
    df: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    """Prepare station features using the same transformations as training."""
    data = df.copy()
    total = data["viajes_salida"].replace(0, 1)

    for column in PATTERN_COLUMNS:
        data[f"{column}_proporcion"] = data[column] / total

    for column in ["viajes_salida", "viajes_llegada"]:
        data[column] = data[column].clip(lower=0)
        data[f"log_{column}"] = np.log1p(data[column])

    missing_columns = [
        column for column in feature_columns if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Faltan columnas requeridas para predecir: {missing_columns}"
        )

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
    model_source: str = typer.Option(
        "local",
        "--model-source",
        help="Modelo que se utilizará: local o mlflow.",
    ),
):
    """Assign stations to clusters using the trained P3 model."""
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")


    if model_source not in ("local", "mlflow"):
        raise typer.BadParameter(
            "El origen debe ser 'local' o 'mlflow'."
        )

    if model_source == "local":
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        bundle = joblib.load(model_path)
        feature_columns = bundle["feature_columns"]
        model = bundle["model"]
        logger.info(f"Using local model: {model_path}")

    else:
        import mlflow
        from mibici.config import MLFLOW_TRACKING_URI

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_name = "mibici-equipo41-p3-grupos"
        model_uri = f"models:/{model_name}@champion"

        model_info = mlflow.models.get_model_info(model_uri)
        signature = model_info.signature

        if signature is None or signature.inputs is None:
            raise ValueError(
                f"El modelo de MLflow no tiene firma de entrada: {model_uri}"
            )

        feature_columns = [
            item.name for item in signature.inputs.inputs
        ]

        model = mlflow.sklearn.load_model(model_uri)
        logger.info(
            f"Using MLflow model: {model_uri}. "
            f"Features: {feature_columns}"
        )


    raw = pd.read_csv(features_path)
    X = prepare_features(raw, feature_columns)

    # Complete Pipeline: includes its own preprocessing.
    if hasattr(model, "predict"):
        labels = model.predict(X)

    # Legacy format: model and scaler were saved separately.
    elif model_source == "local" and "scaler" in bundle:
        labels = model.predict(bundle["scaler"].transform(X))

    else:
        raise TypeError(
            "El modelo cargado no permite predecir clústeres "
            "con el formato esperado."
        )

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
