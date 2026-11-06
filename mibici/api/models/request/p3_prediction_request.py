from typing import Annotated

from pydantic import BaseModel, Field


class P3PredictionRequest(BaseModel):
    viajes_salida: Annotated[int, Field(ge=0)]
    viajes_llegada: Annotated[int, Field(ge=0)]
    duracion_promedio: Annotated[float, Field(ge=0)]
