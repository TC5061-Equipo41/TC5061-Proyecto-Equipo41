import os
import shutil

import mlflow
from mlflow import MlflowClient

from mibici.config import MODELS_DIR

MODEL_NAME = os.getenv("MODEL_NAME", "mibici-equipo41-p3-grupos")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")

dst = MODELS_DIR / "p3_model"

version = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).version
shutil.rmtree(dst, ignore_errors=True)
mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{MODEL_NAME}/{version}", dst_path=str(dst))
(dst / "VERSION").write_text(version)