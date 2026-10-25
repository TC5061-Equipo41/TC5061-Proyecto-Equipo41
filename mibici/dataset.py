
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import pandas as pd

from mibici.config import (
    DATA_METADATA,
    DATA_PIPELINE_VERSION,
    DEPRECATED_STATION_IDS_REPLACEMENTS,
    INTERIM_DATA_DIR,
    MANIFEST_PATH,
    PROJ_ROOT,
    RAW_DATA_DIR,
)
from mibici.features import (
    create_minutes_column,
    delete_duplicates_and_nulls,
    delete_duplicates_and_reset_idx,
    delete_user_data_columns,
    transform_datetime_columns,
)

def interim_dataset_path(data_metadata: dict) -> Path:
    """Build the path for the intermediate dataset."""
    data_range = data_metadata["data_range"]
    suffix = f'{data_range["from"]}-{data_range["to"]}'
    return INTERIM_DATA_DIR / f"joined_trips_stations-{suffix}.parquet"

RENAMED_STATION_COLS = {
    "id": "Estacion_Id",
    "name": "Nombre",
    "latitude": "Latitud",
    "longitude": "Longitud",
    "dcapacity": "No_bicis",
}

def sha256_file(file_path: Path) -> str:
    """Calculate the SHA-256 hash of a file."""
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()

def describe_raw_file(file_path: Path) -> dict:
    """Return metadata describing a raw data file."""
    return {
        "name": file_path.name,
        "size_bytes": file_path.stat().st_size,
        "sha256": sha256_file(file_path),
    }


def compute_data_version(raw_files: list[Path]) -> str:
    """Compute a reproducible version identifier for the raw data."""
    hashes = sorted(sha256_file(file_path) for file_path in raw_files)
    combined_hash = hashlib.sha256(
        "".join(hashes).encode("utf-8")
    ).hexdigest()

    return combined_hash[:16]

def build_manifest(data_metadata: dict) -> dict:
    """Build a manifest describing the raw data and pipeline version."""
    monthly_files = [
        RAW_DATA_DIR / name
        for name in data_metadata["monthly"]
    ]
    stations_file = RAW_DATA_DIR / data_metadata["stations"]
    raw_files = monthly_files + [stations_file]

    return {
        "data_version": compute_data_version(raw_files),
        "pipeline_version": DATA_PIPELINE_VERSION,
        "data_range": data_metadata["data_range"],
        "raw_files": [
            describe_raw_file(file_path)
            for file_path in raw_files
        ],
    }


def write_manifest(manifest: dict) -> None:
    """Write the data manifest as JSON."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(MANIFEST_PATH, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)

def run_ingest(data_metadata: dict = DATA_METADATA) -> tuple[pd.DataFrame, dict]:
    """Build the base dataset and save its manifest."""
    dataset = build_initial_dataset(
        data_metadata,
        save_interim=True,
    )

    manifest = build_manifest(data_metadata)
    write_manifest(manifest)

    return dataset, manifest


def load_base_dataset(
    data_metadata: dict = DATA_METADATA,
) -> tuple[pd.DataFrame, dict]:
    """Load the base dataset and its manifest."""
    return run_ingest(data_metadata)

def read_csv_any_encoding(file_path: Path) -> pd.DataFrame:
    """Read a CSV file using UTF-8 or CP1252 encoding."""
    try:
        return pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="cp1252")

def build_initial_dataset(
    data_metadata: dict,
    save_interim: bool = False,
) -> pd.DataFrame:
    """Limpia y une los viajes con los metadatos de las estaciones; opcionalmente guarda el resultado en CSV."""
    csv_names = data_metadata['monthly']
    stations_csv_name = data_metadata['stations']

    trips = load_monthly_trips(csv_names)
    stations = get_clean_stations_df(stations_csv_name)
    result = join_station_metadata(trips, stations)

    if save_interim:
        INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
        result.to_parquet(
            interim_dataset_path(data_metadata),
            engine="pyarrow",
            compression="snappy",
            index=False,
        )

    return result


def load_monthly_trips(csv_names: list[str]) -> pd.DataFrame:
    """Lee los archivos CSV mensuales, limpia los datos y elimina los duplicados."""
    frames = []

    for name in csv_names:
        file_path = RAW_DATA_DIR / name

        try:
            frame = pd.read_csv(file_path, encoding="utf-8")
        except UnicodeDecodeError:
            frame = pd.read_csv(file_path, encoding="cp1252")

        frames.append(clean_trips(frame))

    return delete_duplicates_and_reset_idx(
        pd.concat(frames, ignore_index=True)
    )


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

    
