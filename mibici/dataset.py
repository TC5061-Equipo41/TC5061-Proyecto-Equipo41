import hashlib
import json
from datetime import datetime, timezone

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


def sha256_file(file_path, chunk_size=1024 * 1024):
    """Calcula el SHA-256 de un archivo sin cargarlo completo en memoria."""
    hasher = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def write_data_manifest(
    data_metadata: dict,
    dataset: pd.DataFrame,
    output_path,
):
    """
    Genera un manifiesto reproducible con la versión de los datos,
    archivos fuente, hashes y metadatos del dataset procesado.
    """
    source_paths = [
        RAW_DATA_DIR / name
        for name in data_metadata["monthly"]
    ]

    source_paths.append(
        RAW_DATA_DIR / data_metadata["stations"]
    )

    source_files = []

    for path in source_paths:
        source_files.append(
            {
                "file": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    # La versión del dataset depende de los archivos fuente
    # y del rango temporal utilizado.
    version_payload = {
        "data_range": data_metadata["data_range"],
        "source_files": source_files,
    }

    version_string = json.dumps(
        version_payload,
        sort_keys=True,
        ensure_ascii=False,
    )

    data_version = hashlib.sha256(
        version_string.encode("utf-8")
    ).hexdigest()

    manifest = {
        "dataset_name": "mibici_joined_trips_stations",
        "data_version": data_version,
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "data_range": data_metadata["data_range"],
        "rows": len(dataset),
        "columns": len(dataset.columns),
        "column_names": list(dataset.columns),
        "source_files": source_files,
        "processed_file": {
            "file": output_path.name,
            "size_bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        },
    }

    manifest_path = (
        INTERIM_DATA_DIR
        / (
            f"data_manifest-"
            f"{data_metadata['data_range']['from']}-"
            f"{data_metadata['data_range']['to']}.json"
        )
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(f"Manifest generado: {manifest_path}")
    print(f"Data version: {data_version}")

    return manifest_path


def build_initial_dataset(
    data_metadata: dict,
    save_interim: bool = False,
) -> pd.DataFrame:
    """
    Limpia y une los viajes con los metadatos de las estaciones.
    Opcionalmente guarda el resultado en Parquet y genera
    un manifiesto de versión de datos.
    """
    csv_names = data_metadata["monthly"]
    stations_csv_name = data_metadata["stations"]

    output_dataset_suffix = (
        f"{data_metadata['data_range']['from']}-"
        f"{data_metadata['data_range']['to']}"
    )

    trips = load_monthly_trips(csv_names)
    stations = get_clean_stations_df(stations_csv_name)
    result = join_station_metadata(trips, stations)

    if save_interim:
        INTERIM_DATA_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            INTERIM_DATA_DIR
            / f"joined_trips_stations-{output_dataset_suffix}.parquet"
        )

        result.to_parquet(
            output_path,
            engine="pyarrow",
            compression="snappy",
            index=False,
        )

        write_data_manifest(
            data_metadata=data_metadata,
            dataset=result,
            output_path=output_path,
        )

    return result


def load_monthly_trips(csv_names: list[str]) -> pd.DataFrame:
    """Lee los archivos CSV mensuales, limpia los datos y elimina los duplicados."""
    frames = []

    for name in csv_names:
        file_path = RAW_DATA_DIR / name

        try:
            frame = pd.read_csv(
                file_path,
                encoding="utf-8",
            )
        except UnicodeDecodeError:
            frame = pd.read_csv(
                file_path,
                encoding="cp1252",
            )

        frames.append(
            clean_trips(frame)
        )

    return delete_duplicates_and_reset_idx(
        pd.concat(
            frames,
            ignore_index=True,
        )
    )


def clean_trips(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina los datos de los usuarios y los viajes incompletos;
    convierte las fechas y calcula la duración en minutos.
    """
    required_columns = [
        "Inicio_del_viaje",
        "Fin_del_viaje",
        "Origen_Id",
        "Destino_Id",
    ]

    result = delete_user_data_columns(
        frame,
        [
            "Usuario_Id",
            "Genero",
            "Año_de_nacimiento",
        ],
    )

    result = delete_duplicates_and_nulls(
        result,
        required_columns,
    )

    result = transform_datetime_columns(
        result,
        [
            "Inicio_del_viaje",
            "Fin_del_viaje",
        ],
    )

    return create_minutes_column(result)


def get_clean_stations_df(
    station_df_name: str,
) -> pd.DataFrame:
    """
    Lee los metadatos de las estaciones, renombra sus columnas
    y excluye los identificadores reemplazados explícitamente.
    """
    stations = pd.read_csv(
        RAW_DATA_DIR / station_df_name
    )

    stations = stations.rename(
        columns=RENAMED_STATION_COLS
    )

    replaced = stations["Estacion_Id"].isin(
        DEPRECATED_STATION_IDS_REPLACEMENTS.keys()
    )

    return stations.loc[
        ~replaced
    ].copy()


def join_station_metadata(
    trips: pd.DataFrame,
    stations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reemplaza los identificadores alternativos de las estaciones
    y añade sus metadatos con prefijos de origen y destino.
    """
    result = trips.copy()

    station_columns = [
        "Origen_Id",
        "Destino_Id",
    ]

    result[station_columns] = result[
        station_columns
    ].replace(
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
    build_initial_dataset(
        DATA_METADATA,
        True,
    )
    
