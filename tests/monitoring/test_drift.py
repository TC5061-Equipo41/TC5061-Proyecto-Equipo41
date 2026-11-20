"""Pruebas del monitoreo de drift (mibici/monitoring/drift.py)."""

import numpy as np
import pandas as pd
import pytest

from mibici.monitoring.drift import (
    STATION_COUNTS,
    TRIP_CATEGORICAL,
    TRIP_NUMERICAL,
    Period,
    drift_table,
    rescale_counts,
    run_report,
    trip_features,
)


# ---------------------------------------------------------------------------
# Periodos
# ---------------------------------------------------------------------------
def test_period_of_training_covers_twelve_months_and_365_days():
    period = Period.parse("2025_10:2026_09")
    assert period.months[0] == "2025_10" and period.months[-1] == "2026_09"
    assert len(period.files) == 12
    assert period.days == 365


def test_single_month_period():
    period = Period.parse("2026_10")
    assert period.files == ["datos_abiertos_2026_10.csv"]
    assert period.days == 31


def test_inverted_period_is_rejected():
    with pytest.raises(ValueError):
        Period.parse("2026_09:2025_10")


# ---------------------------------------------------------------------------
# Preparación de los conjuntos
# ---------------------------------------------------------------------------
def test_rescale_counts_brings_one_month_to_training_scale():
    month = pd.DataFrame(
        {column: [10, 0] for column in STATION_COUNTS} | {"duracion_promedio": [12.5, 9.0]}
    )
    factor = Period.parse("2025_10:2026_09").days / Period.parse("2026_10").days
    scaled = rescale_counts(month, factor)
    assert scaled["viajes_salida"].tolist() == [round(10 * 365 / 31), 0]
    assert scaled["duracion_promedio"].tolist() == [12.5, 9.0]  # lo que no es conteo no cambia


def test_trip_features_have_expected_columns_and_no_personal_data():
    trips = pd.DataFrame(
        {
            "Inicio_del_viaje": pd.to_datetime(["2026-10-03 08:15", "2026-10-05 23:50"]),
            "Duracion_minutos": [12.345, 7.0],
            "Origen_Id": [10, 20],
            "Destino_Id": [11, 20],
            "Origen_Nombre": ["Estación 10", None],
            "Genero": ["F", "M"],
            "Año_de_nacimiento": [1990, 1985],
        }
    )
    features = trip_features(trips)
    assert list(features.columns) == TRIP_NUMERICAL + TRIP_CATEGORICAL
    assert features["hora"].tolist() == [8, 23]
    assert features["fin_de_semana"].tolist() == ["si", "no"]  # 3-oct-2026 es sábado
    assert features["viaje_circular"].tolist() == ["no", "si"]
    assert features["origen_en_catalogo"].tolist() == ["si", "no"]
    assert not {"Genero", "Año_de_nacimiento"} & set(features.columns)


# ---------------------------------------------------------------------------
# Evidently
# ---------------------------------------------------------------------------
def test_drift_table_reads_pvalues_and_distances():
    snapshot = {
        "metrics": [
            {
                "metric_name": "ValueDrift(column=a,method=K-S p_value,threshold=0.05)",
                "config": {"column": "a", "threshold": 0.05},
                "value": 0.01,
            },
            {
                "metric_name": "ValueDrift(column=b,method=Wasserstein distance (normed),threshold=0.1)",
                "config": {"column": "b", "threshold": 0.1},
                "value": 0.02,
            },
            {"metric_name": "RowCount()", "value": 100},
        ]
    }
    rows = {row["variable"]: row["drift"] for row in drift_table(snapshot)}
    assert rows == {"a": True, "b": False}


def test_report_detects_shifted_variable_only(tmp_path):
    rng = np.random.default_rng(42)
    n = 3000
    reference = pd.DataFrame({"estable": rng.normal(10, 2, n), "desplazada": rng.normal(10, 2, n)})
    current = pd.DataFrame({"estable": rng.normal(10, 2, n), "desplazada": rng.normal(14, 2, n)})
    result = run_report(reference, current, ["estable", "desplazada"], [], tmp_path / "r.html")
    drift = {row["variable"]: row["drift"] for row in result["variables"]}
    assert drift == {"estable": False, "desplazada": True}
    assert result["resumen_evidently"]["variables_con_drift"] == 1
    assert (tmp_path / "r.html").stat().st_size > 0
