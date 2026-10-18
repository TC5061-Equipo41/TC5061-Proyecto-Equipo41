
import os
import subprocess
import sys

import mlflow
from dotenv import load_dotenv

from mibici.config import DATA_METADATA, PROJ_ROOT
from mibici.dataset import build_initial_dataset
from mibici.modeling.p1.train import train_p1_model
from mibici.modeling.p2.train import train_p2_model

load_dotenv()

EXPERIMENT_NAME = "mibici-equipo41"

def create_experiment():
    mlflow_uri = os.getenv("MLFLOW_URI")

    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
    else:
        mlflow.set_tracking_uri(
            f"sqlite:///{(PROJ_ROOT / 'mlflow.db').as_posix()}"
        )

    mlflow.set_experiment(EXPERIMENT_NAME)


def execute_runs():
    create_experiment()

    # P1
    base_dataset = build_initial_dataset(DATA_METADATA)
    data_version = "1.0.0"
    date_range = DATA_METADATA["data_range"]
    mlflow.set_experiment(EXPERIMENT_NAME)
    train_p1_model(
        base_dataset,
        data_version,
        DATA_METADATA["data_range"],
    )

    # P2: clasificación de riesgo
    mlflow.set_experiment(EXPERIMENT_NAME)
    train_p2_model(base_dataset, data_version, date_range)

    # P3
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


if __name__ == "__main__":
    execute_runs()
