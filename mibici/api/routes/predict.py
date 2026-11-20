from fastapi import APIRouter, HTTPException, Request
import pandas as pd

from mibici.api.ab_testing.p1_log import log_p1_prediction
from mibici.api.ab_testing.random_chooser import choose_model
from mibici.api.models.request.p1_prediction_request import P1PredictionRequest
from mibici.api.models.request.p3_prediction_request import P3PredictionRequest
from mibici.api.models.response.p1_prediction_response import P1PredictionResponse
from mibici.api.models.response.p3_prediction_response import P3PredictionResponse
from mibici.modeling.p3.train import prepare_features

router = APIRouter(prefix="/predict", tags=["predict"])


def model_input_names(model) -> list[str]:
    """Columnas que el modelo espera, según la firma registrada en MLflow."""
    return model.metadata.get_input_schema().input_names()


@router.post("/p3", response_model=P3PredictionResponse)
def predict_p3(body: P3PredictionRequest, request: Request):
    p3 = request.app.state.models["p3"]
    model = p3.get("loaded")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo todavía no está disponible.",
        )

    try:
        # Mismas transformaciones que en el entrenamiento (proporciones y logaritmos); luego se
        # entregan al modelo solo las columnas de su firma, sea cual sea el candidato campeón.
        features, _ = prepare_features(pd.DataFrame([body.model_dump()]))
        prediction = model.predict(features[model_input_names(model)])

        return P3PredictionResponse(
            grupo_predicho=int(prediction[0]),
            modelo=p3["name"],
            version=p3["version"],
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error al realizar la predicción: {exc}",
        ) from exc

@router.post("/p1/abtest", response_model=P1PredictionResponse)
def predict_p1_abtest(body: P1PredictionRequest, request: Request) -> P1PredictionResponse:
    model = choose_model(request.app.state.models)

    features = pd.DataFrame([{
        "Origen_Id": body.origen_id,
        "Hora": body.hora,
        "Dia": body.dia,
    }]).astype({"Hora": "int32"})  # la firma del modelo espera Hora como int32

    prediction = max(0.0, float(model["loaded"].predict(features)[0]))


    response = P1PredictionResponse(prediction=prediction, model=model["alias"])
    log_p1_prediction(body, response)

    return response
