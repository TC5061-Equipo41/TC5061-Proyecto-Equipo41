
from pydantic import BaseModel


class P3PredictionResponse(BaseModel):
    grupo_predicho: int
    modelo: str
    version: str
