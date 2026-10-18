
from pathlib import Path

import pandas as pd
from loguru import logger
import typer

from mibici.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR
    / "joined_trips_stations-2026_09-2026-06.csv",
    output_path: Path = PROCESSED_DATA_DIR / "p3_station_features.csv",
):
    """Generate station-level usage features for P3 clustering."""
    logger.info(f"Loading trips from {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    trips = pd.read_csv(
        input_path,
        usecols=[
            "Inicio_del_viaje",
            "Origen_Id",
            "Destino_Id",
            "Duracion_minutos",
            "Origen_Nombre",
            "Origen_Latitud",
            "Origen_Longitud",
            "Destino_Nombre",
            "Destino_Latitud",
            "Destino_Longitud",
        ],
        parse_dates=["Inicio_del_viaje"],
    )

    trips = trips.dropna(
        subset=["Inicio_del_viaje", "Origen_Id", "Destino_Id"]
    )

    # Departure features, including time-of-day and weekend patterns.
    trips["hora"] = trips["Inicio_del_viaje"].dt.hour
    trips["fin_semana"] = trips["Inicio_del_viaje"].dt.dayofweek >= 5

    trips["salidas_manana"] = trips["hora"].between(6, 11).astype(int)
    trips["salidas_tarde"] = trips["hora"].between(12, 17).astype(int)
    trips["salidas_noche"] = trips["hora"].between(18, 23).astype(int)
    trips["salidas_madrugada"] = trips["hora"].between(0, 5).astype(int)
    trips["salidas_fin_semana"] = trips["fin_semana"].astype(int)

    departures = trips.groupby("Origen_Id").agg(
        viajes_salida=("Origen_Id", "size"),
        salidas_manana=("salidas_manana", "sum"),
        salidas_tarde=("salidas_tarde", "sum"),
        salidas_noche=("salidas_noche", "sum"),
        salidas_madrugada=("salidas_madrugada", "sum"),
        salidas_fin_semana=("salidas_fin_semana", "sum"),
        duracion_promedio=("Duracion_minutos", "mean"),
        Nombre=("Origen_Nombre", "first"),
        Latitud=("Origen_Latitud", "first"),
        Longitud=("Origen_Longitud", "first"),
    )

    arrivals = trips.groupby("Destino_Id").size().rename("viajes_llegada")

    features = departures.join(arrivals, how="outer")
    features.index.name = "Estacion_Id"
    features = features.reset_index()

    count_columns = [
        "viajes_salida",
        "viajes_llegada",
        "salidas_manana",
        "salidas_tarde",
        "salidas_noche",
        "salidas_madrugada",
        "salidas_fin_semana",
    ]
    features[count_columns] = features[count_columns].fillna(0).astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)

    logger.success(
        f"Generated {len(features)} station records at {output_path}"
    )


if __name__ == "__main__":
    app()

