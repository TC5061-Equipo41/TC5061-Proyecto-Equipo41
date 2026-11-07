from fastapi import APIRouter, HTTPException, Request
import pandas as pd

from mibici.api.models.request.p3_prediction_request import P3PredictionRequest
from mibici.api.models.response.p3_prediction_response import P3PredictionResponse

router = APIRouter(prefix="/predict", tags=["predict"])


@router.get("/model-info")
def model_info(request: Request):
    state = request.app.state
    return {
        "model_name": state.model_name,
        "model_version": state.model_version,
        "model_uri": state.model_uri,
    }


@router.post("/p3", response_model=P3PredictionResponse)
def predict_p3(body: P3PredictionRequest, request: Request):
    state = request.app.state
    model = getattr(state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo todavía no está disponible.",
        )

    try:
        features = pd.DataFrame([body.model_dump()])
        prediction = model.predict(features)

        return P3PredictionResponse(
            grupo_predicho=int(prediction[0]),
            modelo=state.model_name,
            version=state.model_version,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error al realizar la predicción: {exc}",
        ) from exc
