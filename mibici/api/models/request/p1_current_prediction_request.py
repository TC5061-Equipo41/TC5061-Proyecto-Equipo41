from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DiaSemana = Literal["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class P1CurrentPredictionRequest(BaseModel):
    """Estación, hora y día de la semana para predecir los viajes con el modelo actual de P1.

    ``dia`` usa los nombres en inglés de ``pandas.Series.dt.day_name()``, igual que en el
    entrenamiento.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"origen_id": 12, "hora": 8, "dia": "Monday"}]},
    )

    origen_id: int = Field(ge=0, description="Id de la estación de origen")
    hora: int = Field(ge=0, le=23, description="Hora del día (0-23)")
    dia: DiaSemana = Field(description="Día de la semana en inglés")
