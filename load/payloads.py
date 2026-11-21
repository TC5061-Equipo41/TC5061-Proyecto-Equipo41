"""Datos de entrada para las pruebas de carga (sin dependencias de Locust, para poder probarlos).

Si existe ``data/processed/p3_station_features.csv`` se usan estaciones reales; si no, se generan
valores con la misma estructura.
"""

import csv
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
STATIONS_CSV = ROOT / "data" / "processed" / "p3_station_features.csv"
# Datos de prueba de carga, no criptografía: basta un generador pseudoaleatorio.
rng = random.Random()  # nosec B311

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
FRANJAS = ["salidas_manana", "salidas_tarde", "salidas_noche", "salidas_madrugada"]
P3_FIELDS = [
    "viajes_salida",
    "viajes_llegada",
    *FRANJAS,
    "salidas_fin_semana",
    "duracion_promedio",
]


def load_stations() -> list[dict]:
    """Estaciones reales de P3 (si existen) para que las peticiones tengan valores plausibles."""
    if not STATIONS_CSV.exists():
        return []
    with STATIONS_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    stations = []
    for row in rows:
        try:
            station = {k: int(float(row[k])) for k in P3_FIELDS if k != "duracion_promedio"}
            station["duracion_promedio"] = round(float(row["duracion_promedio"]), 2)
            station["Estacion_Id"] = int(row["Estacion_Id"])
        except KeyError, ValueError:
            continue
        if station["duracion_promedio"] > 0:
            stations.append(station)
    return stations


STATIONS = load_stations()


def synthetic_station() -> dict:
    """Estación con la estructura de p3_station_features.csv (si no hay datos reales)."""
    salidas = int(rng.lognormvariate(8.5, 1.0))
    weights = [rng.random() + 0.2 for _ in FRANJAS]
    parts = [int(salidas * w / sum(weights)) for w in weights]
    payload = dict(zip(FRANJAS, parts))
    payload["viajes_salida"] = sum(parts)
    payload["viajes_llegada"] = int(salidas * rng.uniform(0.85, 1.15))
    payload["salidas_fin_semana"] = int(payload["viajes_salida"] * rng.uniform(0.1, 0.35))
    payload["duracion_promedio"] = round(rng.uniform(6, 20), 2)
    return payload


def p3_payload() -> dict:
    if STATIONS:
        station = rng.choice(STATIONS)
        return {k: station[k] for k in P3_FIELDS}
    return synthetic_station()


def p1_payload() -> dict:
    origen = rng.choice(STATIONS)["Estacion_Id"] if STATIONS else rng.randint(1, 600)
    return {
        "origen_id": origen,
        "hora": rng.randint(5, 23),
        "dia": rng.choice(DAYS),
        "y": float(rng.randint(0, 15)),
    }
