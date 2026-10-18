
import pandas as pd

from mibici.config import (
    DATA_METADATA,
    DEPRECATED_STATION_IDS_REPLACEMENTS,
    INTERIM_DATA_DIR,
    RAW_DATA_DIR,
)
from mibici.features import (
    create_minutes_column,
    delete_duplicates_and_nulls,
    delete_duplicates_and_reset_idx,
    delete_user_data_columns,
    transform_datetime_columns,
)

RENAMED_STATION_COLS = {
    "id": "Estacion_Id",
    "name": "Nombre",
    "latitude": "Latitud",
    "longitude": "Longitud",
    "dcapacity": "No_bicis",
}

def build_initial_dataset(
    data_metadata: dict,
    save_interim: bool = False,
) -> pd.DataFrame:
    """Limpia y une los viajes con los metadatos de las estaciones; opcionalmente guarda el resultado en CSV."""
    csv_names = data_metadata['monthly']
    stations_csv_name = data_metadata['stations']
    output_dataset_suffix = f"{data_metadata["data_range"]["from"]}-{data_metadata["data_range"]["to"]}"

    trips = load_monthly_trips(csv_names)
    stations = get_clean_stations_df(stations_csv_name)
    result = join_station_metadata(trips, stations)

    if save_interim:
        result.to_parquet(
            INTERIM_DATA_DIR / f"joined_trips_stations-{output_dataset_suffix}.parquet",
            engine="pyarrow",
            compression="snappy",
            index=False,
        )

    return result


def load_monthly_trips(csv_names: list[str]) -> pd.DataFrame:
    """Lee los archivos CSV mensuales, limpia los datos y elimina los duplicados."""
    frames = []
    for name in csv_names:
        frame = pd.read_csv(RAW_DATA_DIR / name)
        frames.append(clean_trips(frame))

    return delete_duplicates_and_reset_idx(pd.concat(frames, ignore_index=True))


def clean_trips(frame: pd.DataFrame) -> pd.DataFrame:
    """Elimina los datos de los usuarios y los viajes incompletos; convierte las fechas y calcula la duración en minutos."""
    required_columns = [
        "Inicio_del_viaje",
        "Fin_del_viaje",
        "Origen_Id",
        "Destino_Id",
    ]

    result = delete_user_data_columns(frame, ["Usuario_Id", "Genero", "Año_de_nacimiento"])

    result = delete_duplicates_and_nulls(result, required_columns)

    result = transform_datetime_columns(result, ["Inicio_del_viaje", "Fin_del_viaje"])

    return create_minutes_column(result)




def get_clean_stations_df(station_df_name: str) -> pd.DataFrame:
    """Lee los metadatos de las estaciones, renombra sus columnas y excluye los identificadores reemplazados explícitamente."""
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
    """Reemplaza los identificadores alternativos de las estaciones y añade sus metadatos con prefijos de origen y destino."""    
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
    build_initial_dataset(DATA_METADATA, True)

    
