
import pandas as pd

from mibici.config import (
    CURRENT_MONTH_DATA,
    DEPRECATED_STATION_IDS_REPLACEMENTS,
    INTERIM_DATA_DIR,
    RAW_DATA_DIR,
    STATIONS_DATA,
    THREE_MONTH_DATA,
)

RENAMED_STATION_COLS = {
    "id": "Estacion_Id",
    "name": "Nombre",
    "latitude": "Latitud",
    "longitude": "Longitud",
    "dcapacity": "No_bicis",
}


def build_initial_dataset(
    csv_names: list[str],
    stations_csv_name: str,
    output_dataset_suffix: str | None = None,
    save_interim: bool = False,
) -> pd.DataFrame:
    """Clean and join trips with station metadata; optionally save as CSV."""
    trips = load_monthly_trips(csv_names)
    stations = get_clean_stations_df(stations_csv_name)
    result = join_station_metadata(trips, stations)

    if save_interim:
        result.to_csv(
            INTERIM_DATA_DIR / f"joined_trips_stations-{output_dataset_suffix}.csv",
            index=False,
        )

    return result


def load_monthly_trips(csv_names: list[str]) -> pd.DataFrame:
    """Read monthly CSV filenames, clean them, and remove duplicates."""
    frames = []
    for name in csv_names:
        frame = pd.read_csv(RAW_DATA_DIR / name)
        frames.append(clean_trips(frame))

    return _deduplicate_trips(pd.concat(frames, ignore_index=True))


def clean_trips(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove user fields and incomplete trips; parse dates and derive minutes."""
    required_columns = [
        "Inicio_del_viaje",
        "Fin_del_viaje",
        "Origen_Id",
        "Destino_Id",
    ]

    result = frame.drop(
        columns=["Usuario_Id", "Genero", "Año_de_nacimiento"],
        errors="ignore",
    )
    result = result.drop_duplicates()
    result = result.dropna(subset=required_columns).copy()

    for column in ("Inicio_del_viaje", "Fin_del_viaje"):
        result[column] = pd.to_datetime(result[column], errors="raise")

    result["Duracion_minutos"] = (
        result["Fin_del_viaje"] - result["Inicio_del_viaje"]
    ).dt.total_seconds() / 60

    if result["Duracion_minutos"].lt(0).any():
        raise ValueError("Some trips end before they start.")

    return result


def _deduplicate_trips(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove identical rows, including duplicates across monthly files."""
    return frame.drop_duplicates().reset_index(drop=True)


def get_clean_stations_df(station_df_name: str) -> pd.DataFrame:
    """Read and rename station metadata; exclude explicitly replaced IDs."""
    stations = pd.read_csv(RAW_DATA_DIR / station_df_name)
    stations = stations.rename(columns=RENAMED_STATION_COLS)

    replaced = stations["Estacion_Id"].isin(
        DEPRECATED_STATION_IDS_REPLACEMENTS.keys()
    )
    return stations.loc[~replaced].copy()


def join_station_metadata(
    trips: pd.DataFrame,
    stations: pd.DataFrame,
) -> pd.DataFrame:
    """Replace station aliases and attach prefixed origin/destination metadata."""
    result = trips.copy()
    station_columns = ["Origen_Id", "Destino_Id"]

    result[station_columns] = result[station_columns].replace(
        DEPRECATED_STATION_IDS_REPLACEMENTS
    )

    for station_column, prefix in (
        ("Origen_Id", "Origen"),
        ("Destino_Id", "Destino"),
    ):
        metadata = stations.rename(
            columns={
                column: (
                    station_column
                    if column == "Estacion_Id"
                    else f"{prefix}_{column}"
                )
                for column in stations.columns
            }
        )

        result = result.merge(
            metadata,
            on=station_column,
            how="left",
            validate="many_to_one",
            sort=False,
        )

    return result


if __name__ == "__main__":
    build_initial_dataset(CURRENT_MONTH_DATA, STATIONS_DATA, '2026_09', True)
    build_initial_dataset(CURRENT_MONTH_DATA+THREE_MONTH_DATA, STATIONS_DATA, '2026_09-2026-06', True)
    build_initial_dataset(THREE_MONTH_DATA, STATIONS_DATA, '2026_08-2026-06', True)

    
