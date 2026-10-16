

import pandas as pd

from mibici.config import CURRENT_MONTH_DATA, STATIONS_DATA, THREE_MONTH_DATA
from mibici.dataset import build_initial_dataset


def create_p1_dataset(consider_month: bool, original_data: pd.DataFrame | None = None) -> pd.DataFrame :
    if original_data is None:
        original_data = build_initial_dataset(CURRENT_MONTH_DATA+THREE_MONTH_DATA, STATIONS_DATA)
    for col in ['Inicio_del_viaje','Fin_del_viaje']:
        original_data[col] = pd.to_datetime(original_data[col])

    p1_df = original_data[["Origen_Id", "Inicio_del_viaje"]].copy()
    p1_df["Inicio_del_viaje"] = pd.to_datetime(p1_df["Inicio_del_viaje"])
    p1_df["Fecha"] = p1_df["Inicio_del_viaje"].dt.floor("h")

    counts = p1_df.groupby(["Origen_Id", "Fecha"]).size()

    index = pd.MultiIndex.from_product(
        [
            p1_df["Origen_Id"].unique(),
            pd.date_range(p1_df["Fecha"].min(), p1_df["Fecha"].max(), freq="h"),
        ],
        names=["Origen_Id", "Fecha"],
    )

    p1_df = (
        counts.reindex(index, fill_value=0)
        .rename("Viajes")
        .reset_index()
    )

    p1_df["Hora"] = p1_df["Fecha"].dt.hour
    p1_df["Dia"] = p1_df["Fecha"].dt.day
    if (consider_month):
        p1_df["Mes"] = p1_df["Fecha"].dt.month

    return p1_df


def get_train_test_split(df: pd.DataFrame, consider_month: bool):
    train_end = pd.Timestamp("2026-08-31")
    test_end = pd.Timestamp("2026-09-30")

    FEATURES = ["Origen_Id", "Hora", "Dia"]
    if (consider_month):
        FEATURES.append('Mes')

    TARGET = "Viajes"

    train = df[df["Fecha"] <= train_end]
    test = df[df["Fecha"] >= test_end]

    train.drop(columns=["Fecha"])
    test.drop(columns=["Fecha"])

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]
    return X_train, y_train, X_test, y_test


