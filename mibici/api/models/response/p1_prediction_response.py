

from pydantic import BaseModel


class P1PredictionResponse(BaseModel):
    prediction: float
    model: str