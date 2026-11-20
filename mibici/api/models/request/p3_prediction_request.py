from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Conteo = Annotated[int, Field(ge=0)]


class P3PredictionRequest(BaseModel):
    """Uso agregado de una estación, con el mismo esquema que ``p3_station_features.csv``.

    La API calcula a partir de estos conteos las variables del modelo (proporciones por franja
    horaria y logaritmos de los viajes), igual que en el entrenamiento.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "viajes_salida": 856,
                    "viajes_llegada": 754,
                    "salidas_manana": 235,
                    "salidas_tarde": 229,
                    "salidas_noche": 380,
                    "salidas_madrugada": 12,
                    "salidas_fin_semana": 179,
                    "duracion_promedio": 13.4,
                }
            ]
        },
    )

    viajes_salida: Conteo = Field(description="Viajes que salen de la estación")
    viajes_llegada: Conteo = Field(description="Viajes que llegan a la estación")
    salidas_manana: Conteo = Field(description="Salidas entre 06:00 y 11:59")
    salidas_tarde: Conteo = Field(description="Salidas entre 12:00 y 17:59")
    salidas_noche: Conteo = Field(description="Salidas entre 18:00 y 23:59")
    salidas_madrugada: Conteo = Field(description="Salidas entre 00:00 y 05:59")
    salidas_fin_semana: Conteo = Field(description="Salidas en sábado o domingo")
    duracion_promedio: Annotated[float, Field(gt=0, le=1440)] = Field(
        description="Duración promedio de los viajes, en minutos"
    )

    @model_validator(mode="after")
    def salidas_consistentes(self) -> Self:
        franjas = (
            self.salidas_manana + self.salidas_tarde + self.salidas_noche + self.salidas_madrugada
        )
        if franjas > self.viajes_salida:
            raise ValueError("la suma de salidas por franja horaria supera viajes_salida")
        if self.salidas_fin_semana > self.viajes_salida:
            raise ValueError("salidas_fin_semana supera viajes_salida")
        return self
