import os
import shutil

import mlflow
from mlflow import MlflowClient

from mibici.config import MODELS_DIR

mlflow_client = MlflowClient()
def register_model(dir: str, model_name:str, model_alias: str):
    dst = MODELS_DIR / dir

    version = str(mlflow_client.get_model_version_by_alias(model_name, model_alias).version)
    shutil.rmtree(dst, ignore_errors=True)
    mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{model_name}/{version}", dst_path=str(dst))
    (dst / "VERSION").write_text(version)

def register_p3_champion():
    MODEL_NAME = os.getenv("P3_MODEL_NAME", "mibici-equipo41-p3-grupos")
    MODEL_ALIAS = os.getenv("P3_MODEL_ALIAS", "champion")

    dir = "p3_model"

    register_model(dir, MODEL_NAME, MODEL_ALIAS)

def register_p1_models_for_ab_test():
    CURRENT_MODEL_NAME = os.getenv("P1_CURRENT_MODEL", "mibici-equipo41-p1-demanda")
    CURRENT_MODEL_ALIAS = os.getenv("P1_CURRENT_ALIAS", "champion")
    CHALLENGER_MODEL_NAME = os.getenv("P1_CHALLENGER_MODEL", "mibici-equipo41-p1-demanda")
    CHALLENGER_MODEL_ALIAS = os.getenv("P1_CHALLENGER_ALIAS", "challenger")

    register_model("p1_current", CURRENT_MODEL_NAME, CURRENT_MODEL_ALIAS)
    register_model("p1_challenger", CHALLENGER_MODEL_NAME, CHALLENGER_MODEL_ALIAS)



if __name__ == '__main__':
    register_p3_champion()
    register_p1_models_for_ab_test()
