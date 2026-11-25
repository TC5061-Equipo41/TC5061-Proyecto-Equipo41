"""Reporte de drift para la demanda por estación y hora de P1."""

import json
from pathlib import Path

import pandas as pd

from mibici.monitoring.drift import Period, load_period, run_report, sample
from mibici.p1.dataset import create_p1_dataset


def main():
    # Construimos la misma tabla estación-hora que usa el entrenamiento de P1.
    period = Period.parse("2025_10:2026_10")
    columns = ["Origen_Id", "Inicio_del_viaje", "Fin_del_viaje"]
    trips = pd.concat(
        [load_period(Period.parse(month))[columns] for month in period.months],
        ignore_index=True,
    )
    hourly = create_p1_dataset(trips)
    del trips

    # El champion entrenó hasta agosto. Octubre es el mes de la prueba A/B.
    columns = ["Origen_Id", "Hora", "Dia", "Viajes"]
    reference = sample(hourly.loc[hourly["Fecha"] < "2026-09-01", columns])
    current = sample(hourly.loc[hourly["Fecha"] >= "2026-10-01", columns])

    for frame in (reference, current):
        frame["Origen_Id"] = frame["Origen_Id"].astype(str)
        frame["Hora"] = frame["Hora"].astype(str)

    output = Path(__file__).resolve().parents[2] / "reports" / "drift"
    results = run_report(
        reference,
        current,
        numerical=["Viajes"],
        categorical=["Origen_Id", "Hora", "Dia"],
        path=output / "p1_drift.html",
    )
    summary = {
        "modelo": "P1",
        "referencia_champion": "2025-10 a 2026-08",
        "actual": "2026-10",
        "filas_muestra_referencia": len(reference),
        "filas_muestra_actual": len(current),
        "entradas": ["Origen_Id", "Hora", "Dia"],
        "objetivo_observado": "Viajes",
        **results,
    }
    (output / "p1_drift_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Reporte generado en {output / 'p1_drift.html'}")


if __name__ == "__main__":
    main()
