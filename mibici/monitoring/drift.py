"""Drift de datos con Evidently: periodo de entrenamiento (referencia) contra un periodo posterior.

Compara dos niveles:

1. **Viajes** (lo que llega a producción): hora, día de la semana, duración, estación de origen y
   de destino, viajes circulares y estaciones fuera del catálogo. Se toma una muestra aleatoria
   de cada periodo (semilla fija) para que el reporte sea liviano.
2. **Estaciones** (la entrada del modelo P3): las variables de ``p3_station_features.csv``
   calculadas con el mismo código del entrenamiento (``mibici/P3/features.py``) y la misma
   transformación de la API (``prepare_features``). Los conteos se llevan a la escala del
   periodo de referencia (``conteo * días_referencia / días_actual``), porque el modelo se entrenó
   con totales de 12 meses. Se agrega el grupo que asigna el modelo campeón, para medir el
   drift de las predicciones.

Uso:
    python -m mibici.monitoring.drift --reference 2025_10:2026_09 --current 2026_10:2026_10

Salidas en ``reports/drift/``: ``trips_drift.html``, ``p3_stations_drift.html`` y
``drift_summary.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from loguru import logger
import numpy as np
import pandas as pd
import typer

from mibici.config import (
    INTERIM_DATA_DIR,
    MODELS_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    SEED,
    STATIONS_DATA,
)
from mibici.dataset import get_clean_stations_df, join_station_metadata, load_monthly_trips

DRIFT_DIR = REPORTS_DIR / "drift"
WORK_DIR = INTERIM_DATA_DIR / "drift"
SAMPLE_SIZE = 200_000

TRIP_NUMERICAL = ["hora", "duracion_minutos"]
TRIP_CATEGORICAL = [
    "dia_semana",
    "fin_de_semana",
    "origen_id",
    "destino_id",
    "viaje_circular",
    "origen_en_catalogo",
]
STATION_COUNTS = [
    "viajes_salida",
    "viajes_llegada",
    "salidas_manana",
    "salidas_tarde",
    "salidas_noche",
    "salidas_madrugada",
    "salidas_fin_semana",
]
P3_COLUMNS = [
    "Inicio_del_viaje",
    "Origen_Id",
    "Destino_Id",
    "Duracion_minutos",
    "Origen_Nombre",
    "Origen_Latitud",
    "Origen_Longitud",
    "Destino_Nombre",
    "Destino_Latitud",
    "Destino_Longitud",
]

app = typer.Typer()


# ---------------------------------------------------------------------------
# Periodos
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Period:
    """Rango de meses ``AAAA_MM:AAAA_MM`` (ambos incluidos)."""

    start: str
    end: str

    @classmethod
    def parse(cls, text: str) -> Period:
        start, _, end = text.partition(":")
        period = cls(start.strip(), (end or start).strip())
        if period.first_day > period.last_day:
            raise ValueError(f"Periodo invertido: {text}")
        return period

    @property
    def first_day(self) -> pd.Timestamp:
        return pd.Timestamp(self.start.replace("_", "-") + "-01")

    @property
    def last_day(self) -> pd.Timestamp:
        return pd.Timestamp(self.end.replace("_", "-") + "-01") + pd.offsets.MonthEnd(0)

    @property
    def months(self) -> list[str]:
        return [
            d.strftime("%Y_%m") for d in pd.date_range(self.first_day, self.last_day, freq="MS")
        ]

    @property
    def days(self) -> int:
        return (self.last_day - self.first_day).days + 1

    @property
    def files(self) -> list[str]:
        return [f"datos_abiertos_{month}.csv" for month in self.months]

    def __str__(self) -> str:
        return f"{self.start}:{self.end}"


def load_period(period: Period) -> pd.DataFrame:
    """Viajes del periodo con el mismo pipeline de la ingesta (limpieza + metadatos de estación)."""
    missing = [name for name in period.files if not (RAW_DATA_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"Faltan en data/raw: {', '.join(missing)}")
    trips = load_monthly_trips(period.files)
    trips = join_station_metadata(trips, get_clean_stations_df(STATIONS_DATA))
    inside = trips["Inicio_del_viaje"].between(
        period.first_day, period.last_day + pd.Timedelta(days=1), inclusive="left"
    )
    logger.info(f"{period}: {len(trips):,} viajes ({(~inside).sum():,} fuera del rango de fechas)")
    return trips.loc[inside].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Nivel 1: viajes
# ---------------------------------------------------------------------------
def trip_features(trips: pd.DataFrame) -> pd.DataFrame:
    """Variables por viaje que alimentan a P1, P2 y P3 (sin datos personales)."""
    start = trips["Inicio_del_viaje"]
    return pd.DataFrame(
        {
            "hora": start.dt.hour,
            "duracion_minutos": trips["Duracion_minutos"].round(2),
            "dia_semana": start.dt.dayofweek.astype(str),
            "fin_de_semana": (start.dt.dayofweek >= 5).map({True: "si", False: "no"}),
            "origen_id": trips["Origen_Id"].astype(int).astype(str),
            "destino_id": trips["Destino_Id"].astype(int).astype(str),
            "viaje_circular": (trips["Origen_Id"] == trips["Destino_Id"]).map(
                {True: "si", False: "no"}
            ),
            "origen_en_catalogo": trips["Origen_Nombre"].notna().map({True: "si", False: "no"}),
        }
    )


def sample(frame: pd.DataFrame, size: int = SAMPLE_SIZE, seed: int = SEED) -> pd.DataFrame:
    return frame.sample(n=min(size, len(frame)), random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Nivel 2: estaciones (entrada del modelo P3)
# ---------------------------------------------------------------------------
def station_features(trips: pd.DataFrame, name: str) -> pd.DataFrame:
    """Características por estación con el mismo código que generó el entrenamiento de P3."""
    from mibici.P3.features import main as build_p3_features

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    trips_csv = WORK_DIR / f"trips_{name}.csv"
    features_csv = WORK_DIR / f"p3_station_features_{name}.csv"
    trips[P3_COLUMNS].to_csv(trips_csv, index=False)
    build_p3_features(input_path=trips_csv, output_path=features_csv)
    trips_csv.unlink()  # con el periodo de referencia completo pesa cientos de MB
    return pd.read_csv(features_csv)


def rescale_counts(features: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Lleva los conteos a la escala de días del periodo de referencia."""
    result = features.copy()
    result[STATION_COUNTS] = (result[STATION_COUNTS] * factor).round().astype(int)
    return result


def load_champion(model_uri: str):
    import mlflow

    return mlflow.pyfunc.load_model(model_uri)


def model_view(stations: pd.DataFrame, model) -> pd.DataFrame:
    """Variables que ve el modelo (vía ``prepare_features``) más el grupo que predice."""
    from mibici.modeling.p3.train import prepare_features

    data, columns = prepare_features(stations)
    inputs = model.metadata.get_input_schema().input_names() if model is not None else []
    # Variables de prepare_features más las entradas exactas del campeón (p. ej. los conteos
    # sin logaritmo del modelo de conteos), sin repetir.
    view = data[["Estacion_Id", *dict.fromkeys([*inputs, *columns])]].copy()
    if model is not None:
        view["grupo_predicho"] = np.asarray(model.predict(data[inputs])).astype(int).astype(str)
    return view


# ---------------------------------------------------------------------------
# Evidently
# ---------------------------------------------------------------------------
def run_report(reference: pd.DataFrame, current: pd.DataFrame, numerical, categorical, path):
    definition = DataDefinition(numerical_columns=numerical, categorical_columns=categorical)
    snapshot = Report([DataDriftPreset(), DataSummaryPreset()], include_tests=False).run(
        reference_data=Dataset.from_pandas(reference, data_definition=definition),
        current_data=Dataset.from_pandas(current, data_definition=definition),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    snapshot.save_html(str(path))
    result = snapshot.dict()
    return {"variables": drift_table(result), "resumen_evidently": drifted_count(result)}


def drift_table(snapshot: dict) -> list[dict]:
    """Extrae del snapshot de Evidently el resultado por variable: método, valor y umbral."""
    rows = []
    for metric in snapshot["metrics"]:
        name = metric.get("metric_name", "")
        if not name.startswith("ValueDrift("):
            continue
        config = metric.get("config", {})
        column = config.get("column") or name.split("column=")[1].split(",")[0]
        method = name.split("method=")[1].split(",threshold")[0] if "method=" in name else ""
        threshold = config.get("threshold")
        if threshold is None and "threshold=" in name:
            threshold = float(name.split("threshold=")[1].rstrip(")"))
        value = float(metric["value"])
        # Evidently: con p-valor (K-S, chi²) hay drift si p < umbral; con distancias, si d >= umbral.
        is_pvalue = "p_value" in method.lower() or "test" in method.lower()
        drifted = value < threshold if is_pvalue else value >= threshold
        rows.append(
            {
                "variable": column,
                "metodo": method,
                "valor": round(value, 4),
                "umbral": threshold,
                "drift": bool(drifted),
            }
        )
    return sorted(rows, key=lambda r: (not r["drift"], r["variable"]))


def drifted_count(snapshot: dict) -> dict:
    """Conteo de variables con drift según Evidently (DriftedColumnsCount) y umbral de dataset."""
    for metric in snapshot["metrics"]:
        if metric.get("metric_name", "").startswith("DriftedColumnsCount("):
            share = metric["config"].get("drift_share", 0.5)
            return {
                "variables_con_drift": int(metric["value"]["count"]),
                "proporcion": round(float(metric["value"]["share"]), 4),
                "umbral_dataset": share,
                "drift_del_dataset": bool(metric["value"]["share"] >= share),
            }
    return {}


def compare_groups(reference_view: pd.DataFrame, current_view: pd.DataFrame) -> dict:
    if "grupo_predicho" not in reference_view:
        return {}
    joined = reference_view[["Estacion_Id", "grupo_predicho"]].merge(
        current_view[["Estacion_Id", "grupo_predicho"]],
        on="Estacion_Id",
        suffixes=("_referencia", "_actual"),
    )
    changed = joined["grupo_predicho_referencia"] != joined["grupo_predicho_actual"]
    shares = {
        name: view["grupo_predicho"].value_counts(normalize=True).round(4).sort_index().to_dict()
        for name, view in (("referencia", reference_view), ("actual", current_view))
    }
    return {
        "estaciones_en_ambos": len(joined),
        "estaciones_que_cambian_de_grupo": int(changed.sum()),
        "proporcion_que_cambia": round(float(changed.mean()), 4) if len(joined) else None,
        "estaciones_solo_en_referencia": int(len(reference_view) - len(joined)),
        "estaciones_solo_en_actual": int(len(current_view) - len(joined)),
        "distribucion_grupos": shares,
    }


def print_table(title: str, result: dict) -> None:
    rows, overall = result["variables"], result["resumen_evidently"]
    print(f"\n{title} · Evidently: {overall.get('variables_con_drift')} de {len(rows)} con drift")
    print(f"  {'variable':<32}{'método':<36}{'valor':>9}{'umbral':>9}  drift")
    for r in rows:
        mark = "SÍ" if r["drift"] else "no"
        print(
            f"  {r['variable']:<32}{r['metodo'][:35]:<36}{r['valor']:>9}{r['umbral']:>9}  {mark}"
        )


@app.command()
def main(
    reference: str = typer.Option("2025_10:2026_09", help="Periodo de entrenamiento"),
    current: str = typer.Option("2026_10:2026_10", help="Periodo posterior (producción)"),
    model_uri: str = typer.Option(str(MODELS_DIR / "p3_model"), help="Modelo campeón de P3"),
    sample_size: int = typer.Option(SAMPLE_SIZE, help="Viajes muestreados por periodo"),
):
    ref_period, cur_period = Period.parse(reference), Period.parse(current)
    if cur_period.first_day <= ref_period.last_day:
        logger.warning("El periodo actual no es posterior al de referencia")

    ref_trips, cur_trips = load_period(ref_period), load_period(cur_period)

    # Nivel 1 · viajes
    trips_result = run_report(
        sample(trip_features(ref_trips), sample_size),
        sample(trip_features(cur_trips), sample_size),
        TRIP_NUMERICAL,
        TRIP_CATEGORICAL,
        DRIFT_DIR / "trips_drift.html",
    )

    # Nivel 2 · estaciones (entrada y salida del modelo P3)
    factor = ref_period.days / cur_period.days
    ref_stations = station_features(ref_trips, "referencia")
    cur_stations = rescale_counts(station_features(cur_trips, "actual"), factor)
    model = load_champion(model_uri) if Path(model_uri).exists() or "://" in model_uri else None
    if model is None:
        logger.warning(f"No encuentro el modelo en {model_uri}: se omite el drift de predicciones")
    ref_view, cur_view = model_view(ref_stations, model), model_view(cur_stations, model)
    numerical = [c for c in ref_view.columns if c not in ("Estacion_Id", "grupo_predicho")]
    categorical = ["grupo_predicho"] if model is not None else []
    station_result = run_report(
        ref_view.drop(columns="Estacion_Id"),
        cur_view.drop(columns="Estacion_Id"),
        numerical,
        categorical,
        DRIFT_DIR / "p3_stations_drift.html",
    )

    summary = {
        "referencia": {
            "periodo": str(ref_period),
            "dias": ref_period.days,
            "viajes": len(ref_trips),
            "estaciones": len(ref_view),
        },
        "actual": {
            "periodo": str(cur_period),
            "dias": cur_period.days,
            "viajes": len(cur_trips),
            "estaciones": len(cur_view),
        },
        "viajes_muestreados_por_periodo": min(sample_size, len(ref_trips), len(cur_trips)),
        "factor_escala_conteos": round(factor, 4),
        "modelo": model_uri if model is not None else None,
        "viajes": trips_result,
        "estaciones_p3": station_result,
        "predicciones_p3": compare_groups(ref_view, cur_view),
        "calidad": quality(ref_trips, cur_trips),
    }
    out = DRIFT_DIR / "drift_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print_table("Drift por variable · viajes", trips_result)
    print_table("Drift por variable · estaciones (entrada del modelo P3)", station_result)
    print("\nPredicciones P3:", json.dumps(summary["predicciones_p3"], ensure_ascii=False))
    print("Calidad:", json.dumps(summary["calidad"], ensure_ascii=False))
    logger.success(f"Reportes en {DRIFT_DIR}")


def quality(ref_trips: pd.DataFrame, cur_trips: pd.DataFrame) -> dict:
    """Indicadores de calidad que no dependen del muestreo."""

    def indicators(trips: pd.DataFrame) -> dict:
        minutes = trips["Duracion_minutos"]
        return {
            "viajes": len(trips),
            "viajes_por_dia": None,
            "origen_fuera_de_catalogo": round(float(trips["Origen_Nombre"].isna().mean()), 4),
            "destino_fuera_de_catalogo": round(float(trips["Destino_Nombre"].isna().mean()), 4),
            "duracion_cero": round(float((minutes == 0).mean()), 4),
            "duracion_mayor_3h": round(float((minutes > 180).mean()), 4),
            "duracion_p50": round(float(minutes.median()), 2),
            "duracion_p99": round(float(minutes.quantile(0.99)), 2),
        }

    ref, cur = indicators(ref_trips), indicators(cur_trips)
    for values, trips in ((ref, ref_trips), (cur, cur_trips)):
        days = trips["Inicio_del_viaje"].dt.normalize().nunique()
        values["viajes_por_dia"] = round(len(trips) / max(days, 1), 1)
    return {"referencia": ref, "actual": cur}


if __name__ == "__main__":
    app()
