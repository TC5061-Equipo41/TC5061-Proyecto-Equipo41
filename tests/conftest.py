"""Fixtures compartidas para todas las pruebas.

Las pruebas NO usan los datos reales de data/: trabajan con DataFrames
pequeños construidos aquí, donde sabemos de antemano el resultado correcto.
"""

import mlflow
import numpy as np
import pandas as pd
import pytest

SEED = 42


@pytest.fixture(autouse=True)
def isolated_mlflow(tmp_path, monkeypatch):
    """Envía cualquier registro de MLflow a una carpeta temporal, nunca a mlruns/ ni mlflow.db."""
    tracking_uri = f"file:{tmp_path / 'mlruns'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    yield
    if mlflow.active_run() is not None:
        mlflow.end_run()


@pytest.fixture
def trips_df() -> pd.DataFrame:
    """Viajes limpios con el esquema del dataset común (3 estaciones, 2 días)."""
    rng = np.random.default_rng(SEED)
    n = 60
    stations = np.array([1, 2, 3])
    capacity = {1: 10, 2: 15, 3: 20}

    start = pd.Timestamp("2026-07-31 06:00") + pd.to_timedelta(
        rng.integers(0, 48 * 60, size=n), unit="min"
    )
    duration = pd.to_timedelta(rng.integers(3, 45, size=n), unit="min")
    origin = rng.choice(stations, size=n)
    destination = rng.choice(stations, size=n)

    return pd.DataFrame(
        {
            "Viaje_Id": np.arange(1, n + 1),
            "Origen_Id": origin,
            "Destino_Id": destination,
            "Inicio_del_viaje": start,
            "Fin_del_viaje": start + duration,
            "Origen_dpcapacity": [capacity[s] for s in origin],
            "Destino_dpcapacity": [capacity[s] for s in destination],
        }
    )


@pytest.fixture
def raw_trips_df() -> pd.DataFrame:
    """Viajes "sucios" para casos límite: fechas como texto, un duplicado y un nulo."""
    return pd.DataFrame(
        {
            "Viaje_Id": [1, 2, 2, 3],
            "Usuario_Id": [100, 101, 101, 102],
            "Origen_Id": [1, 2, 2, None],
            "Destino_Id": [2, 3, 3, 1],
            "Inicio_del_viaje": [
                "2026-07-01 08:00:00",
                "2026-07-01 09:15:00",
                "2026-07-01 09:15:00",
                "2026-07-01 10:00:00",
            ],
            "Fin_del_viaje": [
                "2026-07-01 08:20:00",
                "2026-07-01 09:45:00",
                "2026-07-01 09:45:00",
                "2026-07-01 10:05:00",
            ],
        }
    )


@pytest.fixture
def raw_stations_df() -> pd.DataFrame:
    """Catálogo de estaciones con nombres crudos; 116 y 138 son IDs reemplazados."""
    return pd.DataFrame(
        {
            "id": [1, 2, 116, 138, 248, 249],
            "name": ["A", "B", "Vieja 116", "Vieja 138", "Nueva 248", "Nueva 249"],
            "latitude": [20.6, 20.7, 20.8, 20.9, 20.8, 20.9],
            "longitude": [-103.3, -103.4, -103.5, -103.6, -103.5, -103.6],
            "dcapacity": [10, 12, 8, 6, 8, 6],
        }
    )