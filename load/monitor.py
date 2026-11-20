"""Registra el consumo de CPU y memoria de los contenedores durante la prueba de carga.

Lee ``docker stats`` en modo continuo (una muestra por contenedor cada ~1-2 s) y escribe un CSV
con la marca de tiempo, el contenedor, el % de CPU (100 % = un núcleo), la memoria en MiB y el %
de memoria respecto del límite del contenedor. Se detiene con Ctrl+C o al recibir SIGTERM.

Uso:
    python -m load.monitor --out reports/load/escalones_recursos.csv mibici-api mibici-mlflow
"""

import argparse
import csv
import json
from pathlib import Path
import re
import shutil
import signal

# Solo docker/podman con argumentos fijos
import subprocess  # nosec B404
import sys
import time

UNITS = {
    "b": 1 / 2**20,
    "kib": 1 / 1024,
    "kb": 1 / 1024,
    "mib": 1,
    "mb": 1,
    "gib": 1024,
    "gb": 1024,
}


def to_mib(text: str) -> float:
    """'123.4MiB' -> 123.4; '1.2GiB' -> 1228.8."""
    match = re.match(r"\s*([\d.]+)\s*([A-Za-z]+)", text)
    if not match:
        return float("nan")
    value, unit = match.groups()
    return float(value) * UNITS.get(unit.lower(), float("nan"))


def parse_line(line: str) -> dict | None:
    """Una línea JSON de ``docker stats --format '{{json .}}'`` (o de podman) a una fila."""
    line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line).strip()  # códigos de pantalla
    if not line.startswith("{"):
        return None
    data = json.loads(line)
    name = data.get("Name") or data.get("name") or data.get("Container")
    cpu = str(data.get("CPUPerc") or data.get("cpu_percent") or "nan").rstrip("%")
    mem_usage = str(data.get("MemUsage") or data.get("mem_usage") or "nan")
    mem_pct = str(data.get("MemPerc") or data.get("mem_percent") or "nan").rstrip("%")
    return {
        "contenedor": name,
        "cpu_pct": float(cpu) if cpu not in ("", "--") else float("nan"),
        "mem_mib": round(to_mib(mem_usage.split("/")[0]), 1),
        "mem_pct": float(mem_pct) if mem_pct not in ("", "--") else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--engine", default="docker", choices=["docker", "podman"])
    parser.add_argument("containers", nargs="+")
    args = parser.parse_args()

    engine = shutil.which(args.engine)
    if engine is None:
        sys.exit(f"No encuentro {args.engine}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Ejecutable absoluto y argumentos fijos
    proc = subprocess.Popen(  # nosec B603
        [engine, "stats", "--format", "{{json .}}", *args.containers],
        stdout=subprocess.PIPE,
        text=True,
    )

    def stop(*_):
        # Interrumpe la lectura bloqueante: en macOS `docker stats` puede no cerrar su salida
        # al recibir SIGTERM, y el bucle quedaría esperando para siempre.
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["timestamp", "contenedor", "cpu_pct", "mem_mib", "mem_pct"]
        )
        writer.writeheader()
        try:
            for line in proc.stdout:
                for chunk in line.replace("}{", "}\n{").splitlines():
                    row = parse_line(chunk)
                    if row:
                        writer.writerow({"timestamp": round(time.time(), 2), **row})
                        f.flush()
        except KeyboardInterrupt:
            pass
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
