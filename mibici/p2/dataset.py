import pandas as pd

RISK_THRESHOLD = 0.25


def create_p2_dataset(original_data: pd.DataFrame) -> pd.DataFrame:
    """
    Construye el dataset específico de P2 a partir del
    dataset común generado por build_initial_dataset().
    """
    df = original_data.copy()

    # Asegurar formato datetime
    df["Inicio_del_viaje"] = pd.to_datetime(
        df["Inicio_del_viaje"],
        errors="coerce",
    )

    df["Fin_del_viaje"] = pd.to_datetime(
        df["Fin_del_viaje"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "Inicio_del_viaje",
            "Fin_del_viaje",
        ]
    )

    station_hour = build_station_hour(df)
    station_hour = add_temporal_features(station_hour)

    return station_hour


def get_capacity_columns(df: pd.DataFrame):
    """
    Identifica las columnas de capacidad generadas
    por el pipeline común.
    """

    if {
        "Origen_dpcapacity",
        "Destino_dpcapacity",
    }.issubset(df.columns):
        return (
            "Origen_dpcapacity",
            "Destino_dpcapacity",
        )

    if {
        "Origen_No_bicis",
        "Destino_No_bicis",
    }.issubset(df.columns):
        return (
            "Origen_No_bicis",
            "Destino_No_bicis",
        )

    raise ValueError(
        "No se encontraron columnas de capacidad de estaciones."
    )


def build_station_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Construye una tabla por estación y hora."""

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

    origin_capacity_col, destination_capacity_col = (
        get_capacity_columns(df)
    )

    origin_capacity = df[
        ["Origen_Id", origin_capacity_col]
    ].rename(
        columns={
            "Origen_Id": "station_id",
            origin_capacity_col: "capacity",
        }
    )

    destination_capacity = df[
        ["Destino_Id", destination_capacity_col]
    ].rename(
        columns={
            "Destino_Id": "station_id",
            destination_capacity_col: "capacity",
        }
    )

    stations = (
        pd.concat(
            [
                origin_capacity,
                destination_capacity,
            ],
            ignore_index=True,
        )
        .dropna()
        .groupby(
            "station_id",
            as_index=False,
        )["capacity"]
        .median()
    )

    stations = stations[
        stations["capacity"] > 0
    ]

    start = min(
        df["Inicio_del_viaje"].min(),
        df["Fin_del_viaje"].min(),
    ).floor("h")

    end = max(
        df["Inicio_del_viaje"].max(),
        df["Fin_del_viaje"].max(),
    ).floor("h")

    hours = pd.date_range(
        start,
        end,
        freq="h",
    )

    grid = pd.MultiIndex.from_product(
        [
            stations["station_id"],
            hours,
        ],
        names=[
            "station_id",
            "fecha_hora",
        ],
    ).to_frame(index=False)

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

    station_hour[
        ["departures", "arrivals"]
    ] = (
        station_hour[
            ["departures", "arrivals"]
        ]
        .fillna(0)
        .astype(int)
    )

    station_hour["net_flow"] = (
        station_hour["arrivals"]
        - station_hour["departures"]
    )

    station_hour["hour"] = (
        station_hour["fecha_hora"].dt.hour
    )

    station_hour["day_of_week"] = (
        station_hour["fecha_hora"].dt.dayofweek
    )

    station_hour["is_weekend"] = (
        station_hour["day_of_week"] >= 5
    ).astype(int)

    station_hour["month"] = (
        station_hour["fecha_hora"].dt.month
    )

    return station_hour


def add_temporal_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Agrega información de la hora anterior y construye
    el target de riesgo para la siguiente hora.
    """

    df = df.sort_values(
        [
            "station_id",
            "fecha_hora",
        ]
    ).copy()

    grouped = df.groupby("station_id")

    df[
        [
            "departures_lag1",
            "arrivals_lag1",
            "net_flow_lag1",
        ]
    ] = grouped[
        [
            "departures",
            "arrivals",
            "net_flow",
        ]
    ].shift(1)

    df[
        [
            "future_departures",
            "future_arrivals",
        ]
    ] = grouped[
        [
            "departures",
            "arrivals",
        ]
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
        df["future_pressure"]
        >= RISK_THRESHOLD
    ).astype(int)

    return df