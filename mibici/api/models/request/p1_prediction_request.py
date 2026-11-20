from pydantic import BaseModel


class P1PredictionRequest(BaseModel):
    origen_id: int
    hora: int
    dia: str
    y: float