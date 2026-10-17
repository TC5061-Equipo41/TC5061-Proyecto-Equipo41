from pathlib import Path

import pandas as pd


# Configuración
PROJ_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJ_ROOT
    / "data"
    / "interim"
    / "joined_trips_stations-2026_09-2026-06.csv"
)

OUTPUT_FILE = (
    PROJ_ROOT
    / "data"
    / "processed"
    / "p2_station_hour.csv"
)

RISK_THRESHOLD = 0.25


def load_data():
    """Carga el consolidado de viajes y convierte las fechas."""
    df = pd.read_csv(INPUT_FILE)

    df["Inicio_del_viaje"] = pd.to_datetime(
        df["Inicio_del_viaje"], errors="coerce"
    )
    df["Fin_del_viaje"] = pd.to_datetime(
        df["Fin_del_viaje"], errors="coerce"
    )

    return df.dropna(
        subset=["Inicio_del_viaje", "Fin_del_viaje"]
    )


def build_station_hour(df):
    """Construye una tabla por estación y hora."""

    # Salidas
    departures = (
        df.assign(
            fecha_hora=df["Inicio_del_viaje"].dt.floor("h")
        )
        .groupby(["Origen_Id", "fecha_hora"])
        .size()
        .rename("departures")
        .reset_index()
        .rename(columns={"Origen_Id": "station_id"})
    )

    # Llegadas
    arrivals = (
        df.assign(
            fecha_hora=df["Fin_del_viaje"].dt.floor("h")
        )
        .groupby(["Destino_Id", "fecha_hora"])
        .size()
        .rename("arrivals")
        .reset_index()
        .rename(columns={"Destino_Id": "station_id"})
    )

    # Capacidad de cada estación
    origin_capacity = df[
        ["Origen_Id", "Origen_dpcapacity"]
    ].rename(
        columns={
            "Origen_Id": "station_id",
            "Origen_dpcapacity": "capacity",
        }
    )

    destination_capacity = df[
        ["Destino_Id", "Destino_dpcapacity"]
    ].rename(
        columns={
            "Destino_Id": "station_id",
            "Destino_dpcapacity": "capacity",
        }
    )

    stations = (
        pd.concat(
            [origin_capacity, destination_capacity],
            ignore_index=True,
        )
        .dropna()
        .groupby("station_id", as_index=False)["capacity"]
        .median()
    )

    stations = stations[stations["capacity"] > 0]

    # Rango horario completo
    start = min(
        df["Inicio_del_viaje"].min(),
        df["Fin_del_viaje"].min(),
    ).floor("h")

    end = max(
        df["Inicio_del_viaje"].max(),
        df["Fin_del_viaje"].max(),
    ).floor("h")

    hours = pd.date_range(start, end, freq="h")

    grid = pd.MultiIndex.from_product(
        [stations["station_id"], hours],
        names=["station_id", "fecha_hora"],
    ).to_frame(index=False)

    # Unión de actividad y capacidad
    station_hour = (
        grid
        .merge(
            departures,
            on=["station_id", "fecha_hora"],
            how="left",
        )
        .merge(
            arrivals,
            on=["station_id", "fecha_hora"],
            how="left",
        )
        .merge(
            stations,
            on="station_id",
            how="left",
        )
    )

    station_hour[["departures", "arrivals"]] = (
        station_hour[["departures", "arrivals"]]
        .fillna(0)
        .astype(int)
    )

    # Variables básicas
    station_hour["net_flow"] = (
        station_hour["arrivals"]
        - station_hour["departures"]
    )

    station_hour["hour"] = station_hour["fecha_hora"].dt.hour
    station_hour["day_of_week"] = (
        station_hour["fecha_hora"].dt.dayofweek
    )
    station_hour["is_weekend"] = (
        station_hour["day_of_week"] >= 5
    ).astype(int)
    station_hour["month"] = station_hour["fecha_hora"].dt.month

    return station_hour


def add_temporal_features(df):
    """Agrega historial reciente y construye el target de P2."""

    df = df.sort_values(
        ["station_id", "fecha_hora"]
    ).copy()

    grouped = df.groupby("station_id")

    # Información de la hora anterior
    df[
        ["departures_lag1", "arrivals_lag1", "net_flow_lag1"]
    ] = grouped[
        ["departures", "arrivals", "net_flow"]
    ].shift(1)

    # Información de la siguiente hora:
    # solo se utiliza para construir la etiqueta
    df[
        ["future_departures", "future_arrivals"]
    ] = grouped[
        ["departures", "arrivals"]
    ].shift(-1)

    df["future_net_flow"] = (
        df["future_arrivals"]
        - df["future_departures"]
    )

    df["future_pressure"] = (
        df["future_net_flow"].abs()
        / df["capacity"]
    )

    df["target_risk"] = (
        df["future_pressure"] >= RISK_THRESHOLD
    ).astype(int)

    return df


def main():
    df = load_data()
    print(f"Viajes cargados: {len(df):,}")

    station_hour = build_station_hour(df)
    station_hour = add_temporal_features(station_hour)

    print(
        f"Registros estación-hora: "
        f"{len(station_hour):,}"
    )

    print("\nDistribución target:")
    print(
        station_hour["target_risk"]
        .value_counts(normalize=True)
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    station_hour.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"\nDataset guardado en:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()