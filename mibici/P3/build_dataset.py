from loguru import logger
import pandas as pd

from mibici.config import (
    INTERIM_DATA_DIR,
    RAW_DATA_DIR,
    STATIONS_DATA,
)
from mibici.dataset import (
    clean_trips,
    delete_duplicates_and_reset_idx,
    get_clean_stations_df,
    join_station_metadata,
)
from mibici.features import delete_duplicates_and_reset_idx

MONTHLY_FILES = [
    "datos_abiertos_2025_10.csv",
    "datos_abiertos_2025_11.csv",
    "datos_abiertos_2025_12.csv",
    "datos_abiertos_2026_01.csv",
    "datos_abiertos_2026_02.csv",
    "datos_abiertos_2026_03.csv",
    "datos_abiertos_2026_04.csv",
    "datos_abiertos_2026_05.csv",
    "datos_abiertos_2026_06.csv",
    "datos_abiertos_2026_07.csv",
    "datos_abiertos_2026_08.csv",
    "datos_abiertos_2026_09.csv",
]

def main():
    frames = []

    for name in MONTHLY_FILES:
        path = RAW_DATA_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"No se encontró: {path}")

        encoding = "cp1252" if name == "datos_abiertos_2026_01.csv" else "utf-8"
        logger.info(f"Leyendo {name} con {encoding}")

        frame = pd.read_csv(path, encoding=encoding)
        frames.append(clean_trips(frame))

    trips = delete_duplicates_and_reset_idx(pd.concat(frames, ignore_index=True))
    stations = get_clean_stations_df(STATIONS_DATA)
    result = join_station_metadata(trips, stations)

    output_path = (
        INTERIM_DATA_DIR
        / "joined_trips_stations-2026_09-2025_10.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    logger.success(f"Dataset anual guardado: {output_path}")
    print(f"Viajes procesados: {len(result):,}")
    print(
        result[["Inicio_del_viaje", "Fin_del_viaje"]]
        .agg(["min", "max"])
    )

if __name__ == "__main__":
    main()
