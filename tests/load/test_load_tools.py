"""Pruebas de las herramientas de carga (load/): datos de entrada y lectura de docker stats."""

import math

from load import payloads
from load.monitor import parse_line, to_mib
import pytest

from mibici.api.models.request.p1_prediction_request import P1PredictionRequest
from mibici.api.models.request.p3_prediction_request import P3PredictionRequest


@pytest.mark.parametrize("use_real_stations", [True, False])
def test_payloads_pass_api_validation(monkeypatch, use_real_stations):
    """Lo que envía Locust debe ser una petición válida para la API (si no, mediríamos 422)."""
    if not use_real_stations:
        monkeypatch.setattr(payloads, "STATIONS", [])
    elif not payloads.STATIONS:
        pytest.skip("sin data/processed/p3_station_features.csv")
    payloads.rng.seed(42)
    for _ in range(300):
        P3PredictionRequest(**payloads.p3_payload())
        P1PredictionRequest(**payloads.p1_payload())


def test_synthetic_station_is_consistent():
    payloads.rng.seed(7)
    for _ in range(300):
        station = payloads.synthetic_station()
        franjas = sum(station[k] for k in payloads.FRANJAS)
        assert franjas == station["viajes_salida"]
        assert station["salidas_fin_semana"] <= station["viajes_salida"]
        assert 0 < station["duracion_promedio"] <= 1440


@pytest.mark.parametrize(
    "text,expected",
    [("795.1MiB", 795.1), ("1.5GiB", 1536.0), ("512KiB", 0.5), ("2.4GB", 2457.6)],
)
def test_to_mib_converts_units(text, expected):
    assert to_mib(text) == pytest.approx(expected)


def test_parse_docker_stats_line_with_screen_codes():
    line = (
        '\x1b[2J\x1b[H{"BlockIO":"0B / 0B","CPUPerc":"109.52%","Container":"mibici-api",'
        '"MemPerc":"9.91%","MemUsage":"795.1MiB / 7.836GiB","Name":"mibici-api","PIDs":"9"}'
    )
    row = parse_line(line)
    assert row == {
        "contenedor": "mibici-api",
        "cpu_pct": 109.52,
        "mem_mib": 795.1,
        "mem_pct": 9.91,
    }


def test_parse_podman_stats_line_and_ignores_noise():
    row = parse_line(
        '{"name":"mibici-mlflow","cpu_percent":"1.30%","mem_usage":"2.4GB / 8GB",'
        '"mem_percent":"30.95%"}'
    )
    assert row["contenedor"] == "mibici-mlflow" and row["cpu_pct"] == 1.3
    assert row["mem_mib"] == pytest.approx(2457.6)
    assert parse_line("CONTAINER ID   NAME") is None
    blank = parse_line('{"Name":"x","CPUPerc":"--","MemUsage":"-- / --","MemPerc":"--"}')
    assert math.isnan(blank["cpu_pct"])
