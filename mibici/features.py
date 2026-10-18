
import pandas as pd


def delete_user_data_columns(frame: pd.DataFrame, columns: list[str]):
    """Elimina las columnas indicadas de datos de usuarios e ignora las que no existen."""
    return frame.drop(
        columns,
        errors="ignore",
    )


def delete_duplicates_and_nulls(
    frame: pd.DataFrame,
    required_columns: list[str],
) -> pd.DataFrame:
    """Elimina las filas duplicadas y las que contienen valores nulos en las columnas requeridas."""
    result = frame.drop_duplicates()

    return result.dropna(subset=required_columns)


def delete_duplicates_and_reset_idx(frame: pd.DataFrame) -> pd.DataFrame:
    """Elimina las filas duplicadas y restablece el índice sin conservar el anterior."""
    return frame.drop_duplicates().reset_index(drop=True)


def transform_datetime_columns(frame: pd.DataFrame, columns: list[str]):
    """Convierte las columnas indicadas a fecha y hora; genera un error si algún valor no es válido."""
    frame = frame.copy()

    for column in columns:
        frame[column] = pd.to_datetime(frame[column], errors="raise")

    return frame


def create_minutes_column(frame: pd.DataFrame):
    """Calcula la duración de cada viaje en minutos y rechaza las duraciones negativas."""
    frame = frame.copy()
    frame["Duracion_minutos"] = (
        frame["Fin_del_viaje"] - frame["Inicio_del_viaje"]
    ).dt.total_seconds() / 60

    if frame["Duracion_minutos"].lt(0).any():
        raise ValueError("Some trips end before they start.")

    return frame
