import argparse
import json
import os
import subprocess
import sys

import mlflow
from dotenv import load_dotenv

from mibici.config import (
    DATA_METADATA,
    INTERIM_DATA_DIR,
    PROJ_ROOT,
)
from mibici.dataset import build_initial_dataset
from mibici.modeling.p1.train import train_p1_model
from mibici.modeling.p2.train import train_p2_model


load_dotenv()

EXPERIMENT_NAME = "mibici-equipo41"


def create_experiment():
    """
    Configura MLflow y selecciona el experimento del equipo.
    """

    mlflow_uri = (
        os.getenv("MLFLOW_TRACKING_URI")
        or os.getenv("MLFLOW_URI")
    )

    if mlflow_uri:
        mlflow.set_tracking_uri(
            mlflow_uri
        )
    else:
        mlflow.set_tracking_uri(
            f"sqlite:///{(PROJ_ROOT / 'mlflow.db').as_posix()}"
        )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )


def configure_project_parent_run():
    """
    Detecta la corrida creada automáticamente por MLflow Project.

    MLflow Projects define MLFLOW_RUN_ID para que el proceso interno
    utilice la corrida creada por `mlflow run`. Como los modelos del
    proyecto crean sus propias corridas, retiramos esa variable para
    evitar reutilizar accidentalmente el run del Project.

    El ID se conserva en una variable propia para utilizarlo como
    corrida padre de los experimentos internos.
    """

    project_run_id = os.environ.pop(
        "MLFLOW_RUN_ID",
        None,
    )

    if project_run_id:
        os.environ[
            "MIBICI_PROJECT_PARENT_RUN_ID"
        ] = project_run_id

        print(
            f"\nMLflow Project parent run: "
            f"{project_run_id}"
        )

    return project_run_id


def load_data_version_from_manifest():
    """
    Obtiene la versión reproducible del dataset desde
    el manifiesto generado durante la ingesta.
    """

    data_range = DATA_METADATA["data_range"]

    manifest_path = (
        INTERIM_DATA_DIR
        / (
            f"data_manifest-"
            f"{data_range['from']}-"
            f"{data_range['to']}.json"
        )
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            "No se encontró el manifiesto de datos. "
            "Ejecuta primero la ingesta con: "
            "mlflow run . -e ingest --env-manager local"
        )

    with open(
        manifest_path,
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    return manifest["data_version"]


def run_p1(
    data_version,
    date_range,
):
    """
    Ejecuta el entrenamiento correspondiente a P1.
    """

    print(
        "\nEjecutando SOLO P1..."
    )

    base_dataset = build_initial_dataset(
        DATA_METADATA
    )

    return train_p1_model(
        base_dataset,
        data_version,
        date_range,
    )


def run_p2(
    data_version,
    date_range,
    model="all",
):
    """
    Ejecuta el entrenamiento correspondiente a P2.

    Permite ejecutar todo el flujo o un modelo específico.
    """

    print(
        "\nEjecutando SOLO P2..."
    )

    print(
        f"Modelo seleccionado: {model}"
    )

    base_dataset = build_initial_dataset(
        DATA_METADATA
    )

    return train_p2_model(
        base_dataset,
        data_version,
        date_range,
        model=model,
    )


def run_p3():
    """
    Ejecuta preparación, features y entrenamiento de P3.
    """

    print(
        "\nEjecutando SOLO P3..."
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "mibici.P3.build_dataset",
        ],
        cwd=PROJ_ROOT,
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "mibici.P3.features",
        ],
        cwd=PROJ_ROOT,
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "mibici.modeling.p3.train",
            "--date-range",
            "2025-10 a 2026-09",
            "--experiment-name",
            EXPERIMENT_NAME,
        ],
        cwd=PROJ_ROOT,
        check=True,
    )


def execute_run(
    question,
    model="all",
):
    """
    Router principal de entrenamiento.

    Permite seleccionar P1, P2 o P3 desde línea
    de comandos y desde MLflow Project.
    """

    create_experiment()

    # Evita que las corridas internas reutilicen
    # accidentalmente el run creado por MLflow Project.
    configure_project_parent_run()

    data_version = (
        load_data_version_from_manifest()
    )

    date_range = DATA_METADATA[
        "data_range"
    ]

    print(
        f"\nData version utilizada: "
        f"{data_version}"
    )

    print(
        f"Pregunta seleccionada: "
        f"{question.upper()}"
    )

    if question == "p1":
        return run_p1(
            data_version=data_version,
            date_range=date_range,
        )

    if question == "p2":
        return run_p2(
            data_version=data_version,
            date_range=date_range,
            model=model,
        )

    if question == "p3":
        return run_p3()

    raise ValueError(
        f"Pregunta no válida: {question}"
    )


def parse_args():
    """
    Argumentos de línea de comandos del entrenamiento.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Entrenamiento reproducible del "
            "proyecto MiBici - Equipo 41."
        )
    )

    parser.add_argument(
        "--question",
        required=True,
        choices=[
            "p1",
            "p2",
            "p3",
        ],
        help=(
            "Pregunta del proyecto que se "
            "desea ejecutar."
        ),
    )

    parser.add_argument(
        "--model",
        default="all",
        help=(
            "Modelo a ejecutar. "
            "Por defecto ejecuta el flujo "
            "completo de la pregunta."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    execute_run(
        question=args.question,
        model=args.model,
    )