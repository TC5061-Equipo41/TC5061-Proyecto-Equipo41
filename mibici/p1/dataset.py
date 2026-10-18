

import pandas as pd


def create_p1_dataset(data: pd.DataFrame) -> pd.DataFrame :
    for col in ['Inicio_del_viaje','Fin_del_viaje']:
        data[col] = pd.to_datetime(data[col])

    p1_df = data[["Origen_Id", "Inicio_del_viaje"]].copy()
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
    p1_df["Dia"] = p1_df["Fecha"].dt.day_name()

    return p1_df


def get_train_test_split(df: pd.DataFrame):
    TRAIN_END = pd.Timestamp("2026-08-01")
    TEST_END = pd.Timestamp("2026-09-01")

    FEATURES = ["Origen_Id", "Hora", "Dia"]

    TARGET = "Viajes"


    train = df[df["Fecha"] < TRAIN_END]
    test = df[(df["Fecha"] >= TRAIN_END) & (df["Fecha"] < TEST_END)] 


    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]
    return X_train, y_train, X_test, y_test


