import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from loguru import logger
from mlflow.models import infer_signature
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mibici.config import (
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
)

PATTERN_COLUMNS = [
    "salidas_manana",
    "salidas_tarde",
    "salidas_noche",
    "salidas_madrugada",
    "salidas_fin_semana",
]

MODEL_REGISTRY_NAME = "mibici-equipo41-p3-grupos"


def register_champion_model(run_id: str) -> str:
    """Register the selected MLflow model and assign the champion alias."""
    model_uri = f"runs:/{run_id}/candidate_model"

    registered = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_REGISTRY_NAME,
    )

    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(
        name=MODEL_REGISTRY_NAME,
        alias="champion",
        version=registered.version,
    )

    champion_uri = f"models:/{MODEL_REGISTRY_NAME}@champion"
    loaded_model = mlflow.sklearn.load_model(champion_uri)

    if not hasattr(loaded_model, "predict"):
        raise TypeError("El modelo champion registrado no permite predict().")

    logger.success(
        f"Modelo registrado: {MODEL_REGISTRY_NAME}, "
        f"versión {registered.version}, alias champion"
    )
    return champion_uri

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Prepare station usage features for clustering."""
    df = df.copy()
    total = df["viajes_salida"].replace(0, 1)

    for column in PATTERN_COLUMNS:
        df[f"{column}_proporcion"] = df[column] / total

    feature_columns = [f"{column}_proporcion" for column in PATTERN_COLUMNS]
    feature_columns += ["viajes_salida", "viajes_llegada", "duracion_promedio"]

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


def configure_mlflow(experiment_name: str) -> None:
    """Configure the tracking server and experiment."""
    tracking_uri = os.getenv("MLFLOW_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        tracking_db = PROJ_ROOT / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{tracking_db.as_posix()}")
        logger.warning(f"MLFLOW_URI is not set; using {tracking_db}")

    mlflow.set_experiment(experiment_name)


def get_git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJ_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def log_candidate_model(model, X: pd.DataFrame, labels: np.ndarray) -> None:
    """Log a candidate model with signature and input example."""
    example = X.head(5).copy()
    signature = infer_signature(example, model.predict(example))
    mlflow.sklearn.log_model(
        sk_model=model,
        name="candidate_model",
        signature=signature,
        input_example=example,
    )


def train_p3_model(
    features_path: Path = PROCESSED_DATA_DIR / "p3_station_features.csv",
    model_path: Path = MODELS_DIR / "p3_kmeans.pkl",
    labels_path: Path = PROCESSED_DATA_DIR / "p3_station_clusters.csv",
    figure_path: Path = FIGURES_DIR / "p3_clusters.png",
    min_k: int = 2,
    max_k: int = 8,
    random_state: int = 42,
    experiment_name: str = "mibici-equipo41",
    date_range: str = "2025-10 a 2026-09",
) -> dict:
    """Train and compare P3 clustering candidates using nested MLflow runs."""
    features_path = Path(features_path)
    model_path = Path(model_path)
    labels_path = Path(labels_path)
    figure_path = Path(figure_path)

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if min_k < 2 or max_k < min_k:
        raise ValueError("Require 2 <= min_k <= max_k.")

    raw = pd.read_csv(features_path)
    if len(raw) <= max_k:
        raise ValueError("There must be more stations than max_k.")

    data, feature_columns = prepare_features(raw)
    X = data[feature_columns].copy()
    data_hash = hashlib.sha256(features_path.read_bytes()).hexdigest()

    model_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    configure_mlflow(experiment_name)
    git_commit = get_git_commit()
    candidates = []

    with mlflow.start_run(run_name="P3 - Comparacion de modelos") as parent:
        mlflow.set_tags({
            "team": "41",
            "question": "P3",
            "task": "station_clustering",
            "model_family": "clustering",
            "stage_of_work": "search",
            "git_commit": git_commit,
            "date_range": date_range,
        })
        mlflow.log_params({
            "data_file": features_path.name,
            "data_sha256": data_hash,
            "n_stations": len(X),
            "random_state": random_state,
            "scaler": "StandardScaler",
            "features": ",".join(feature_columns),
        })

        # Nested search: one child run for each KMeans value of k.
        with mlflow.start_run(
            run_name="P3 - Busqueda KMeans k",
            nested=True,
        ):
            for k in range(min_k, max_k + 1):
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clusterer", KMeans(
                        n_clusters=k,
                        random_state=random_state,
                        n_init=10,
                    )),
                ])

                with mlflow.start_run(
                    run_name=f"KMeans k={k}",
                    nested=True,
                ) as candidate_run:
                    labels = pipeline.fit_predict(X)
                    score = float(silhouette_score(
                        pipeline.named_steps["scaler"].transform(X),
                        labels,
                    ))
                    mlflow.set_tags({
                        "question": "P3",
                        "model_family": "KMeans",
                        "candidate": "true",
                    })
                    mlflow.log_params({
                        "algorithm": "KMeans",
                        "n_clusters": k,
                        "random_state": random_state,
                        "n_init": 10,
                    })
                    mlflow.log_metric("silhouette_score", score)
                    log_candidate_model(pipeline, X, labels)
                    candidates.append({
                        "name": f"KMeans_k{k}",
                        "algorithm": "KMeans",
                        "n_clusters": k,
                        "silhouette_score": score,
                        "model": pipeline,
                        "labels": labels,
                        "run_id": candidate_run.info.run_id,
                    })
                    logger.info(f"KMeans k={k}: silhouette={score:.4f}")

        # Alternative algorithm: agglomerative clustering.
        for k in [2, 3, 4]:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clusterer", AgglomerativeClustering(n_clusters=k)),
            ])
            with mlflow.start_run(
                run_name=f"Agglomerative k={k}",
                nested=True,
            ):
                labels = model.fit_predict(X)
                scaled = model.named_steps["scaler"].transform(X)
                score = float(silhouette_score(scaled, labels))
                mlflow.set_tags({
                    "question": "P3",
                    "model_family": "AgglomerativeClustering",
                    "candidate": "true",
                })
                mlflow.log_params({
                    "algorithm": "AgglomerativeClustering",
                    "n_clusters": k,
                    "linkage": "ward",
                    "scaler": "StandardScaler",
                })
                mlflow.log_metric("silhouette_score", score)
                # AgglomerativeClustering has no predict method; its fitted
                # model is saved as an artifact for reproducibility.
                mlflow.log_dict(
                    {"algorithm": "AgglomerativeClustering", "n_clusters": k},
                    "candidate_config.json",
                )
                candidates.append({
                    "name": f"Agglomerative_k{k}",
                    "algorithm": "AgglomerativeClustering",
                    "n_clusters": k,
                    "silhouette_score": score,
                    "model": model,
                    "labels": labels,
                })
                logger.info(
                    f"Agglomerative k={k}: silhouette={score:.4f}"
                )

        # Alternative feature set: counts and duration only, with KMeans.
        alternative_columns = [
            "viajes_salida",
            "viajes_llegada",
            "duracion_promedio",
        ]
        X_alt = data[alternative_columns].copy()
        for k in [2, 3, 4]:
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clusterer", KMeans(
                    n_clusters=k,
                    random_state=random_state,
                    n_init=10,
                )),
            ])
            with mlflow.start_run(
                run_name=f"KMeans conteos k={k}",
                nested=True,
            ) as candidate_run:
                labels = pipeline.fit_predict(X_alt)
                scaled = pipeline.named_steps["scaler"].transform(X_alt)
                score = float(silhouette_score(scaled, labels))
                mlflow.set_tags({
                    "question": "P3",
                    "model_family": "KMeans",
                    "feature_set": "counts_duration",
                    "candidate": "true",
                })
                mlflow.log_params({
                    "algorithm": "KMeans",
                    "n_clusters": k,
                    "random_state": random_state,
                    "feature_set": "counts_duration",
                    "features": ",".join(alternative_columns),
                })
                mlflow.log_metric("silhouette_score", score)
                log_candidate_model(pipeline, X_alt, labels)
                candidates.append({
                    "name": f"KMeans_conteos_k{k}",
                    "algorithm": "KMeans",
                    "n_clusters": k,
                    "silhouette_score": score,
                    "model": pipeline,
                    "labels": labels,
                    "feature_columns": alternative_columns,
                    "run_id": candidate_run.info.run_id,
                })
                logger.info(
                    f"KMeans conteos k={k}: silhouette={score:.4f}"
                )

        # Select the best candidate that supports predictions for new stations.
        deployable_candidates = [
            item for item in candidates
            if hasattr(item["model"], "predict")
        ]

        if not deployable_candidates:
            raise RuntimeError(
                "No hay modelos candidatos compatibles con predict()."
            )

        best = max(
            deployable_candidates,
            key=lambda item: item["silhouette_score"],
        )
        best_model = best["model"]
        best_labels = best["labels"]
        best_columns = best.get("feature_columns", feature_columns)

        joblib.dump({
            "model": best_model,
            "feature_columns": best_columns,
            "algorithm": best["algorithm"],
            "n_clusters": best["n_clusters"],
        }, model_path)

        result = raw.copy()
        result["cluster"] = best_labels
        result.to_csv(labels_path, index=False)

        comparison = pd.DataFrame([
            {
                "model": item["name"],
                "algorithm": item["algorithm"],
                "n_clusters": item["n_clusters"],
                "silhouette_score": item["silhouette_score"],
            }
            for item in candidates
        ]).sort_values("silhouette_score", ascending=False)

        comparison_path = labels_path.with_name("p3_model_comparison.csv")
        comparison.to_csv(comparison_path, index=False)

        plots_dir = figure_path.parent / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        comparison_plot = plots_dir / "p3_model_comparison.png"

        fig, ax = plt.subplots(figsize=(10, 6))
        ordered = comparison.sort_values("silhouette_score")
        ax.barh(ordered["model"], ordered["silhouette_score"])
        ax.set_xlabel("Silhouette score")
        ax.set_title("P3 - Comparacion de modelos")
        fig.tight_layout()
        fig.savefig(comparison_plot, dpi=160)
        plt.close(fig)

        mlflow.log_metric("best_silhouette_score", best["silhouette_score"])
        mlflow.log_param("selected_model", best["name"])
        mlflow.set_tag("selected_algorithm", best["algorithm"])
        mlflow.set_tag("selected_n_clusters", str(best["n_clusters"]))

        if not best.get("run_id"):
            raise RuntimeError(
                f"El candidato ganador {best['name']} no tiene run_id registrado."
            )

        champion_uri = register_champion_model(best["run_id"])
        mlflow.log_param("registered_model_name", MODEL_REGISTRY_NAME)
        mlflow.log_param("registered_model_alias", "champion")

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(labels_path))
        mlflow.log_artifact(str(comparison_path))
        mlflow.log_artifact(str(comparison_plot), artifact_path="plots")

        mlflow.set_tag("selected_model", best["name"])
        logger.success(
            f"Selected {best['name']} with silhouette "
            f"{best['silhouette_score']:.4f}"
        )

    return {
        "model_name": best["name"],
        "algorithm": best["algorithm"],
        "n_clusters": best["n_clusters"],
        "silhouette_score": best["silhouette_score"],
        "model_path": str(model_path),
        "labels_path": str(labels_path),
        "comparison_path": str(comparison_path),
        "figure_path": str(comparison_plot),
        "registry_name": MODEL_REGISTRY_NAME,
        "model_uri": champion_uri,
        "registry_alias": "champion",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train P3 station clustering.")
    parser.add_argument("--features-path", type=Path,
                        default=PROCESSED_DATA_DIR / "p3_station_features.csv")
    parser.add_argument("--model-path", type=Path,
                        default=MODELS_DIR / "p3_kmeans.pkl")
    parser.add_argument("--labels-path", type=Path,
                        default=PROCESSED_DATA_DIR / "p3_station_clusters.csv")
    parser.add_argument("--figure-path", type=Path,
                        default=FIGURES_DIR / "p3_clusters.png")
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--experiment-name", default="mibici-equipo41")
    parser.add_argument("--date-range", default="2025-10 a 2026-09")
    args = parser.parse_args()

    result = train_p3_model(
        features_path=args.features_path,
        model_path=args.model_path,
        labels_path=args.labels_path,
        figure_path=args.figure_path,
        min_k=args.min_k,
        max_k=args.max_k,
        random_state=args.random_state,
        experiment_name=args.experiment_name,
        date_range=args.date_range,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
