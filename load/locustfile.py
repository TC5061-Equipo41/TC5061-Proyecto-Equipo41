"""Pruebas de carga de la API de MiBici con Locust.

Cada usuario simulado espera entre 0,5 y 2 s entre peticiones y reparte su carga entre los dos
endpoints de predicción de la API:

- ``POST /predict/p3`` (70 %): grupo de una estación a partir de su uso agregado.
- ``POST /predict/p1/abtest`` (30 %): viajes por estación, hora y día (prueba A/B de P1).

Los datos de cada petición cambian (``load/payloads.py``): estaciones reales de
``data/processed/p3_station_features.csv`` si existe, o valores generados con la misma estructura.

El escenario se elige con la variable ``LOAD_SCENARIO``:

- ``escalones``: 10 → 25 → 50 → 100 usuarios, 90 s cada escalón (capacidad en régimen estable).
- ``estres``: +10 usuarios cada 15 s hasta 500, y se detiene solo cuando el p95 supera
  ``BREAK_P95_MS`` o la tasa de error supera ``BREAK_ERROR_PCT`` en 3 mediciones seguidas.

Uso (lo orquesta ``python -m load.run``):
    LOAD_SCENARIO=escalones locust -f load/locustfile.py --headless --host http://localhost:8000
"""

import json
import os
from pathlib import Path
import time

from locust import HttpUser, LoadTestShape, between, task
from payloads import p1_payload, p3_payload  # locust agrega load/ al sys.path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.getenv("LOAD_OUT_DIR", ROOT / "reports" / "load"))

SCENARIO = os.getenv("LOAD_SCENARIO", "escalones")
STEPS = [(10, 90), (25, 90), (50, 90), (100, 90)]  # (usuarios, segundos)
SPAWN_RATE = 10
STRESS_START, STRESS_STEP, STRESS_EVERY, STRESS_MAX = 10, 10, 15, 500
BREAK_P95_MS = float(os.getenv("BREAK_P95_MS", "2000"))
BREAK_ERROR_PCT = float(os.getenv("BREAK_ERROR_PCT", "5"))
BREAK_CHECKS = 3  # mediciones seguidas fuera del umbral para declarar el quiebre
CHECK_EVERY = 5  # segundos entre mediciones


class MiBiciUser(HttpUser):
    """Cliente de la API: 70 % predicciones de P3 y 30 % de P1."""

    wait_time = between(0.5, 2)

    @task(7)
    def predict_p3(self):
        with self.client.post(
            "/predict/p3", json=p3_payload(), name="POST /predict/p3", catch_response=True
        ) as response:
            if response.status_code == 200 and "grupo_predicho" in response.text:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(3)
    def predict_p1(self):
        with self.client.post(
            "/predict/p1/abtest",
            json=p1_payload(),
            name="POST /predict/p1/abtest",
            catch_response=True,
        ) as response:
            if response.status_code == 200 and "prediction" in response.text:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")


class Escenario(LoadTestShape):
    """Escalones (régimen estable) o rampa de estrés con corte automático por quiebre."""

    def __init__(self):
        super().__init__()
        self.breaches = 0
        self.last_check = 0.0
        self.last_totals = (0, 0)
        self.stopped = False

    def tick(self):
        run_time = self.get_run_time()
        if SCENARIO == "escalones":
            elapsed = 0
            for users, seconds in STEPS:
                elapsed += seconds
                if run_time < elapsed:
                    return users, SPAWN_RATE
            return None

        # Estrés: rampa continua hasta el quiebre o hasta STRESS_MAX usuarios
        if self.stopped:
            return None
        users = min(STRESS_START + STRESS_STEP * int(run_time // STRESS_EVERY), STRESS_MAX)
        if run_time - self.last_check >= CHECK_EVERY:
            self.last_check = run_time
            if self.check_break(run_time, users):
                self.stopped = True
                return None
        if users >= STRESS_MAX and run_time > (STRESS_MAX / STRESS_STEP) * STRESS_EVERY + 60:
            self.save_break(run_time, users, "sin_quiebre", None, None)
            return None
        return users, SPAWN_RATE

    def check_break(self, run_time: float, users: int) -> bool:
        stats = self.runner.stats.total
        requests, failures = stats.num_requests, stats.num_failures
        new_requests = requests - self.last_totals[0]
        new_failures = failures - self.last_totals[1]
        self.last_totals = (requests, failures)
        if new_requests == 0:
            return False
        p95 = stats.get_current_response_time_percentile(0.95) or 0
        error_pct = 100 * new_failures / new_requests
        breach = p95 > BREAK_P95_MS or error_pct > BREAK_ERROR_PCT
        self.breaches = self.breaches + 1 if breach else 0
        if self.breaches >= BREAK_CHECKS:
            reason = "latencia" if p95 > BREAK_P95_MS else "errores"
            self.save_break(run_time, users, reason, p95, error_pct)
            return True
        return False

    def save_break(self, run_time, users, reason, p95, error_pct):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        stats = self.runner.stats.total
        (OUT_DIR / "estres_quiebre.json").write_text(
            json.dumps(
                {
                    "criterio": reason,
                    "segundo": round(run_time, 1),
                    "usuarios": users,
                    "p95_ms": round(p95, 1) if p95 is not None else None,
                    "error_pct": round(error_pct, 2) if error_pct is not None else None,
                    "rps_actual": round(stats.current_rps, 1),
                    "umbrales": {"p95_ms": BREAK_P95_MS, "error_pct": BREAK_ERROR_PCT},
                    "timestamp": time.time(),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
