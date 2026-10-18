

import hashlib
import os
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from mibici.config import MODELS_DIR, PROCESSED_DATA_DIR, FIGURES_DIR

app = typer.Typer()

PATTERN_COLUMNS = [
    "salidas_manana",
    "salidas_tarde",
    "salidas_noche",
    "salidas_madrugada",
    "salidas_fin_semana",
]


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build comparable usage-pattern features for station clustering."""
    df = df.copy()

    total = df["viajes_salida"].replace(0, 1)

    for column in PATTERN_COLUMNS:
        df[f"{column}_proporcion"] = df[column] / total

    feature_columns = [
        f"{column}_proporcion" for column in PATTERN_COLUMNS
    ]
    feature_columns += [
        "viajes_salida",
        "viajes_llegada",
        "duracion_promedio",
    ]

    for column in ["viajes_salida", "viajes_llegada"]:
        df[column] = df[column].clip(lower=0)
        df[f"log_{column}"] = np.log1p(df[column])
        feature_columns.remove(column)
        feature_columns.append(f"log_{column}")

    df[feature_columns] = df[feature_columns].replace(
        [float("inf"), float("-inf")], float("nan")
    )
    df[feature_columns] = df[feature_columns].fillna(
        df[feature_columns].median()
    )

    return df, feature_columns


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "p3_station_features.csv",
    model_path: Path = MODELS_DIR / "p3_kmeans.pkl",
    labels_path: Path = PROCESSED_DATA_DIR / "p3_station_clusters.csv",
    figure_path: Path = FIGURES_DIR / "p3_clusters.png",
    min_k: int = 2,
    max_k: int = 8,
    random_state: int = 42,
    experiment_name: str = "mibici-p3-clustering",
    date_range: str = "2026-06 a 2026-09",
):
    """Train and evaluate a KMeans station-clustering model."""
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    if min_k < 2 or max_k < min_k:
        raise ValueError("Require 2 <= min_k <= max_k.")

    raw = pd.read_csv(features_path)
    if len(raw) <= max_k:
        raise ValueError("There must be more stations than max_k.")

    data, feature_columns = prepare_features(raw)
    scaler = StandardScaler()
    X = scaler.fit_transform(data[feature_columns])

    data_hash = hashlib.sha256(features_path.read_bytes()).hexdigest()

    git_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    git_commit = git_result.stdout.strip() or "unknown"

    tracking_uri = os.getenv("MLFLOW_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        tracking_db = Path.cwd() / "mlflow.db"
        logger.warning(
            f"MLFLOW_URI is not set; using local SQLite tracking: {tracking_db}"
        )
        mlflow.set_tracking_uri(f"sqlite:///{tracking_db.as_posix()}")

    mlflow.set_experiment(experiment_name)

    scores = []
    best_model = None
    best_k = None
    best_score = float("-inf")
    best_labels = None

    model_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="p3-kmeans-selection"):
        mlflow.log_params(
            {
                "min_k": min_k,
                "max_k": max_k,
                "random_state": random_state,
                "scaler": "StandardScaler",
                "features": ",".join(feature_columns),
                "data_file": features_path.name,
                "data_sha256": data_hash,
                "date_range": date_range,
            }
        )

        mlflow.set_tags(
            {
                "team": "Equipo41",
                "project_part": "P3",
                "business_question": (
                    "Que grupos de estaciones comparten patrones de uso "
                    "y donde se ubican en la ciudad"
                ),
                "git_commit": git_commit,
            }
        )

        for k in range(min_k, max_k + 1):
            model = KMeans(
                n_clusters=k,
                random_state=random_state,
                n_init=10,
            )
            labels = model.fit_predict(X)
            score = silhouette_score(X, labels)

            scores.append({"k": k, "silhouette_score": score})
            logger.info(f"k={k}: silhouette_score={score:.4f}")

            if score > best_score:
                best_score = score
                best_k = k
                best_model = model
                best_labels = labels

        assert best_model is not None
        assert best_labels is not None
        assert best_k is not None

        import joblib

        joblib.dump(
            {
                "model": best_model,
                "scaler": scaler,
                "feature_columns": feature_columns,
            },
            model_path,
        )

        result = raw.copy()
        result["cluster"] = best_labels
        result.to_csv(labels_path, index=False)

        scores_df = pd.DataFrame(scores)
        scores_path = labels_path.with_name("p3_silhouette_scores.csv")
        scores_df.to_csv(scores_path, index=False)

        plt.figure(figsize=(8, 5))
        plt.plot(
            scores_df["k"],
            scores_df["silhouette_score"],
            marker="o",
        )
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Silhouette Score")
        plt.title("P3: KMeans model selection")
        plt.xticks(scores_df["k"])
        plt.tight_layout()
        plt.savefig(figure_path, dpi=150)
        plt.close()

        mlflow.log_param("selected_k", best_k)
        mlflow.log_metric("best_silhouette_score", best_score)
        mlflow.log_metrics(
            {
                f"silhouette_k_{row['k']}": row["silhouette_score"]
                for row in scores
            }
        )

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(labels_path))
        mlflow.log_artifact(str(scores_path))
        mlflow.log_artifact(str(figure_path))

        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="kmeans_model",
        )

        logger.success(
            f"Selected k={best_k} with silhouette score {best_score:.4f}"
        )
        logger.success(f"Model saved to {model_path}")
        logger.success(f"Cluster assignments saved to {labels_path}")
        logger.success(f"Plot saved to {figure_path}")


if __name__ == "__main__":
    app()

