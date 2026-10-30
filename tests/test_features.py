"""Pruebas de mibici/features.py (limpieza común de viajes)."""

import pandas as pd
import pytest

from mibici.features import (
    create_minutes_column,
    delete_duplicates_and_nulls,
    delete_user_data_columns,
    transform_datetime_columns,
)


def test_delete_user_data_columns_ignores_missing(raw_trips_df):
    result = delete_user_data_columns(raw_trips_df, ["Usuario_Id", "No_existe"])

    assert "Usuario_Id" not in result.columns
    assert len(result.columns) == len(raw_trips_df.columns) - 1


def test_delete_duplicates_and_nulls(raw_trips_df):
    result = delete_duplicates_and_nulls(raw_trips_df, required_columns=["Origen_Id"])

    assert len(result) == 2
    assert not result.duplicated().any()
    assert result["Origen_Id"].notna().all()


def test_transform_datetime_columns_converts_text(raw_trips_df):
    columns = ["Inicio_del_viaje", "Fin_del_viaje"]
    result = transform_datetime_columns(raw_trips_df, columns)

    for column in columns:
        assert pd.api.types.is_datetime64_any_dtype(result[column])
    # No debe modificar el DataFrame original
    assert not pd.api.types.is_datetime64_any_dtype(raw_trips_df["Inicio_del_viaje"])


def test_transform_datetime_columns_rejects_invalid_date():
    frame = pd.DataFrame({"Inicio_del_viaje": ["2026-07-01 08:00", "no es fecha"]})

    with pytest.raises((ValueError, TypeError)):
        transform_datetime_columns(frame, ["Inicio_del_viaje"])


def test_create_minutes_column(trips_df):
    result = create_minutes_column(trips_df)

    expected = (trips_df["Fin_del_viaje"] - trips_df["Inicio_del_viaje"]).dt.total_seconds() / 60
    pd.testing.assert_series_equal(result["Duracion_minutos"], expected, check_names=False)
    assert (result["Duracion_minutos"] >= 0).all()


def test_create_minutes_column_rejects_negative_duration(trips_df):
    trips_df.loc[0, "Fin_del_viaje"] = trips_df.loc[0, "Inicio_del_viaje"] - pd.Timedelta("5min")

    with pytest.raises(ValueError, match="end before they start"):
        create_minutes_column(trips_df)
