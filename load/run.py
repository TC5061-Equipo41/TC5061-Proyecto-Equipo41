"""Corre un escenario de carga completo: espera a la API, monitorea recursos, ejecuta Locust y
genera las gráficas.

Uso (con los contenedores levantados con ``docker compose up -d``):
    python -m load.run --scenario escalones
    python -m load.run --scenario estres

Salidas en ``reports/load/``: reporte HTML de Locust, CSV de Locust y de recursos, gráficas y
resumen JSON.
"""

import argparse
import os
from pathlib import Path
import shutil
import signal

# Locust, python y docker con argumentos fijos
import subprocess  # nosec B404
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "load"
CONTAINERS = ["mibici-api", "mibici-mlflow"]


def wait_for_api(host: str, timeout: int = 120) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{host}/health", timeout=3) as response:  # nosec B310
                if response.status == 200:
                    print(f"API lista: {response.read().decode()}")
                    return
        except OSError:
            pass
        time.sleep(2)
    sys.exit(f"La API no respondió en {host}/health. ¿Está corriendo docker compose up -d?")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["escalones", "estres"], required=True)
    parser.add_argument("--host", default=os.getenv("API_HOST", "http://localhost:8000"))
    parser.add_argument("--engine", default=os.getenv("CONTAINER_ENGINE", "docker"))
    args = parser.parse_args()

    if not args.host.startswith(("http://", "https://")):
        sys.exit("--host debe empezar con http:// o https://")
    wait_for_api(args.host)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = OUT_DIR / args.scenario
    for old in OUT_DIR.glob(f"{args.scenario}_*"):
        old.unlink()
    if args.scenario == "estres":
        (OUT_DIR / "estres_quiebre.json").unlink(missing_ok=True)

    # Intérprete actual y argumentos fijos
    monitor = subprocess.Popen(  # nosec B603
        [
            sys.executable,
            "-m",
            "load.monitor",
            "--engine",
            args.engine,
            "--out",
            str(OUT_DIR / f"{args.scenario}_recursos.csv"),
            *CONTAINERS,
        ],
        cwd=ROOT,
    )
    time.sleep(3)  # línea base de recursos antes de la carga
    locust = shutil.which("locust") or sys.exit("Falta locust: pip install -r requirements.txt")
    env = {**os.environ, "LOAD_SCENARIO": args.scenario, "LOAD_OUT_DIR": str(OUT_DIR)}
    try:
        # Ejecutable absoluto y argumentos fijos
        status = subprocess.call(  # nosec B603
            [
                locust,
                "-f",
                str(ROOT / "load" / "locustfile.py"),
                "--headless",
                "--host",
                args.host,
                "--csv",
                str(prefix),
                "--csv-full-history",
                "--html",
                str(OUT_DIR / f"{args.scenario}_locust.html"),
                "--only-summary",
                "--exit-code-on-error",
                "0",
            ],
            cwd=ROOT,
            env=env,
        )
    finally:
        time.sleep(3)
        monitor.send_signal(signal.SIGTERM)
        try:
            monitor.wait(timeout=10)
        except subprocess.TimeoutExpired:
            monitor.kill()  # el CSV ya quedó escrito: se guarda fila por fila
    if status != 0:
        sys.exit(f"Locust terminó con código {status}")
    # Intérprete actual y argumentos fijos
    subprocess.call(  # nosec B603
        [sys.executable, "-m", "load.report", "--scenario", args.scenario], cwd=ROOT
    )


if __name__ == "__main__":
    main()
