from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
import mlflow

from mibici.api.routes import health_check, predict

MODEL_NAME = os.getenv("MODEL_NAME", "mibici-equipo41-p3-grupos")
MODEL_VERSION = os.getenv("MODEL_VERSION", "champion")

# Modelo del Model Registry (nombre + alias). La ubicación del registry la toma MLflow
# de la variable de entorno MLFLOW_TRACKING_URI.
MODEL_URI = os.getenv("MODEL_URI", f"models:/{MODEL_NAME}@champion")

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_name = MODEL_NAME
    app.state.model_version = MODEL_VERSION
    app.state.model_uri = MODEL_URI

    try:
        app.state.model = mlflow.pyfunc.load_model(MODEL_URI)
    except Exception as exc:
        raise RuntimeError(
            f"No fue posible cargar el modelo {MODEL_URI}: {exc}"
        ) from exc

    yield


app = FastAPI(
    title="MiBici API",
    description="API para predecir grupos de estaciones MiBici.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health_check.router)
app.include_router(predict.router)
