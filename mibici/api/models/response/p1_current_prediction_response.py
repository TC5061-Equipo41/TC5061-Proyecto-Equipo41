from pydantic import BaseModel, Field


class P1CurrentPredictionResponse(BaseModel):
    prediction: float = Field(ge=0, description="Viajes predichos (nunca negativos)")
    model: str = Field(description="Nombre del modelo registrado")
    alias: str = Field(description="Alias del modelo en el registry")
    version: str = Field(description="Versión del modelo registrado")
