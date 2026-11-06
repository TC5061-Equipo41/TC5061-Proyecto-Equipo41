
import os
from pathlib import Path
from typing import Annotated

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_NAME = "mibici-equipo41-p3-grupos"
MODEL_VERSION = "3"

# Mantener el mismo tracking URI local utilizado para entrenar el modelo.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)

MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

app = FastAPI(
    title="MiBici P3 API",
    description="API para predecir grupos de estaciones MiBici.",
    version="1.0.0",
)

model = None


class PredictionRequest(BaseModel):
    viajes_salida: Annotated[int, Field(ge=0)]
    viajes_llegada: Annotated[int, Field(ge=0)]
    duracion_promedio: Annotated[float, Field(ge=0)]


class PredictionResponse(BaseModel):
    grupo_predicho: int
    modelo: str
    version: str


@app.on_event("startup")
def load_model():
    global model
    print("Tracking:", mlflow.get_tracking_uri())
    print("Registry:", mlflow.get_registry_uri())
    try:
        model = mlflow.pyfunc.load_model(MODEL_URI)
    except Exception as exc:
        raise RuntimeError(
            f"No fue posible cargar el modelo {MODEL_URI}: {exc}"
        ) from exc


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.get("/model-info")
def model_info():
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_uri": MODEL_URI,
        "tracking_uri": TRACKING_URI,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo todavía no está disponible.",
        )

    try:
        features = pd.DataFrame([request.model_dump()])
        prediction = model.predict(features)

        return PredictionResponse(
            grupo_predicho=int(prediction[0]),
            modelo=MODEL_NAME,
            version=MODEL_VERSION,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error al realizar la predicción: {exc}",
        ) from exc
