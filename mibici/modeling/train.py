

import os

from dotenv import load_dotenv
import mlflow

from mibici.config import DATA_METADATA
from mibici.dataset import build_initial_dataset
from mibici.modeling.p1.train import train_p1_model

load_dotenv()

def create_experiment():
    mlflow_uri = os.getenv("MLFLOW_URI")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("mibici-equipo41")

def execute_runs():
    create_experiment()
    base_dataset = build_initial_dataset(DATA_METADATA)
    data_version = "1.0.0"

    train_p1_model(base_dataset, data_version, DATA_METADATA["data_range"])
    # Execute /p2/train.py (run)
    # Execute /p3/train.py (run)


if __name__ == '__main__':
    execute_runs()