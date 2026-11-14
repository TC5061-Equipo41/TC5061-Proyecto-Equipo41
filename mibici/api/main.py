from contextlib import asynccontextmanager
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
import mlflow

from mibici.api.routes import health_check, predict

load_dotenv()

P3_MODEL_NAME = os.getenv("P3_MODEL_NAME")
P3_MODEL_ALIAS = os.getenv("P3_MODEL_ALIAS")
P3_MODEL_URI = os.getenv("P3_MODEL_URI")

P1_CURRENT_MODEL = os.getenv("P1_CURRENT_MODEL")
P1_CURRENT_ALIAS = os.getenv("P1_CURRENT_ALIAS")
P1_CURRENT_URI = os.getenv("P1_CURRENT_URI")

P1_CHALLENGER_MODEL = os.getenv("P1_CHALLENGER_MODEL")
P1_CHALLENGER_ALIAS = os.getenv("P1_CHALLENGER_ALIAS")
P1_CHALLENGER_URI = os.getenv("P1_CHALLENGER_URI")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield


def model_version(name: str, alias: str, uri: str) -> str:
    """Modelo exportado (Docker): archivo VERSION de export_model.py; si no, el registry."""
    version_file = Path(uri) / "VERSION"
    if version_file.is_file():
        return version_file.read_text().strip()
    return str(mlflow.MlflowClient().get_model_version_by_alias(name, alias).version)


def load_model(name: str, alias: str, uri: str) -> dict:
    return {
        "name": name,
        "alias": alias,
        "uri": uri,
        "loaded": mlflow.pyfunc.load_model(uri),
        "version": model_version(name, alias, uri),
    }


def load_models():
    app.state.models = {
        "p3": load_model(P3_MODEL_NAME, P3_MODEL_ALIAS, P3_MODEL_URI),
        "p1_current": load_model(P1_CURRENT_MODEL, P1_CURRENT_ALIAS, P1_CURRENT_URI),
        "p1_challenger": load_model(P1_CHALLENGER_MODEL, P1_CHALLENGER_ALIAS, P1_CHALLENGER_URI),
    }

        
app = FastAPI(
    title="MiBici API",
    description="API para predecir grupos de estaciones (P3) y viajes por estación y hora (P1) de MiBici.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health_check.router)
app.include_router(predict.router)
