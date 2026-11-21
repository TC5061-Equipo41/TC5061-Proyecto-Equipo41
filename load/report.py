"""Gráficas y resumen de una prueba de carga a partir de los CSV de Locust y de ``docker stats``.

Genera en ``reports/load/``:

- ``<escenario>_desempeno.png``: usuarios, peticiones por segundo, latencia (p50/p95 total y p95
  por endpoint) y tasa de error en el tiempo.
- ``<escenario>_recursos.png``: CPU y memoria de cada contenedor junto al número de usuarios.
- ``<escenario>_resumen.json``: métricas por escalón (o por tramo de usuarios en estrés), totales
  por endpoint y, en estrés, el punto de quiebre.

Uso:
    python -m load.report --scenario escalones
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "load"
SLO_P95_MS = 500
SLO_ERROR_PCT = 1
WARMUP_S = 20  # se descartan los primeros segundos de cada escalón
API, MLFLOW = "mibici-api", "mibici-mlflow"
COLORS = {"total": "#1f2937", "p3": "#2563eb", "p1": "#d97706", API: "#2563eb", MLFLOW: "#16a34a"}


def load_history(prefix: Path) -> pd.DataFrame:
    history = pd.read_csv(f"{prefix}_stats_history.csv")
    history = history[history["Timestamp"] > 0].copy()
    history["t"] = history["Timestamp"] - history["Timestamp"].min()
    for column in ["50%", "95%", "99%", "Requests/s", "Failures/s"]:
        history[column] = pd.to_numeric(history[column], errors="coerce")
    return history


def aggregated(history: pd.DataFrame) -> pd.DataFrame:
    agg = history[history["Name"] == "Aggregated"].sort_values("t").reset_index(drop=True)
    # Tasa de error del intervalo a partir de los totales acumulados (más estable que el ratio
    # de las tasas instantáneas).
    requests = agg["Total Request Count"].diff().fillna(agg["Total Request Count"])
    failures = agg["Total Failure Count"].diff().fillna(agg["Total Failure Count"])
    window = 5
    agg["error_pct"] = (
        100
        * failures.rolling(window, min_periods=1).sum()
        / requests.rolling(window, min_periods=1).sum().replace(0, float("nan"))
    ).fillna(0)
    return agg


def load_resources(path: Path, start: float) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["t", "contenedor", "cpu_pct", "mem_mib"])
    resources = pd.read_csv(path)
    resources["t"] = resources["timestamp"] - start
    return resources[resources["t"] >= 0]


def endpoint(history: pd.DataFrame, name: str) -> pd.DataFrame:
    return history[history["Name"] == name].sort_values("t")


def plot_performance(agg, history, scenario, breakpoint, path):
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    axes[0].plot(agg["t"], agg["User Count"], color=COLORS["total"])
    axes[0].set_ylabel("Usuarios")

    axes[1].plot(agg["t"], agg["Requests/s"], color=COLORS["total"], label="Peticiones/s")
    axes[1].plot(agg["t"], agg["Failures/s"], color="#dc2626", label="Fallas/s")
    axes[1].set_ylabel("Peticiones por segundo")
    axes[1].legend(loc="upper left")

    axes[2].plot(agg["t"], agg["50%"], color=COLORS["total"], ls="--", label="p50 total")
    axes[2].plot(agg["t"], agg["95%"], color=COLORS["total"], label="p95 total")
    for name, key in (("POST /predict/p3", "p3"), ("POST /predict/p1/abtest", "p1")):
        data = endpoint(history, name)
        axes[2].plot(data["t"], data["95%"], color=COLORS[key], alpha=0.8, label=f"p95 {name}")
    axes[2].axhline(SLO_P95_MS, color="#6b7280", ls=":", label=f"Objetivo p95 {SLO_P95_MS} ms")
    axes[2].set_ylabel("Latencia (ms)")
    axes[2].set_yscale("log")
    axes[2].legend(loc="upper left", fontsize=8)

    axes[3].plot(agg["t"], agg["error_pct"], color="#dc2626")
    axes[3].axhline(SLO_ERROR_PCT, color="#6b7280", ls=":", label=f"Objetivo {SLO_ERROR_PCT} %")
    axes[3].set_ylabel("Tasa de error (%)")
    axes[3].set_xlabel("Segundos desde el inicio")
    axes[3].legend(loc="upper left")

    if breakpoint and breakpoint.get("criterio") not in (None, "sin_quiebre"):
        for ax in axes:
            ax.axvline(breakpoint["segundo"], color="#dc2626", ls="--", alpha=0.6)
        axes[0].annotate(
            f"Quiebre: {breakpoint['usuarios_activos']} usuarios ({breakpoint['criterio']})",
            (breakpoint["segundo"], 0.92),
            xycoords=("data", "axes fraction"),
            xytext=(-8, 0),
            textcoords="offset points",
            ha="right",
            va="top",
            color="#dc2626",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(f"Prueba de carga · escenario {scenario} · desempeño de la API")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_resources(agg, resources, scenario, path):
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(agg["t"], agg["User Count"], color=COLORS["total"])
    axes[0].set_ylabel("Usuarios")
    for name in (API, MLFLOW):
        data = resources[resources["contenedor"] == name]
        axes[1].plot(data["t"], data["cpu_pct"], color=COLORS[name], label=name)
        axes[2].plot(data["t"], data["mem_mib"], color=COLORS[name], label=name)
    axes[1].axhline(100, color="#6b7280", ls=":", label="1 núcleo")
    axes[1].set_ylabel("CPU (%)")
    axes[1].legend(loc="upper left")
    axes[2].set_ylabel("Memoria (MiB)")
    axes[2].set_xlabel("Segundos desde el inicio")
    axes[2].legend(loc="upper left")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(f"Prueba de carga · escenario {scenario} · recursos (docker stats)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def window_metrics(agg, resources, t0, t1) -> dict:
    rows = agg[(agg["t"] >= t0) & (agg["t"] < t1)]
    if rows.empty:
        return {}
    requests = rows["Total Request Count"].iloc[-1] - rows["Total Request Count"].iloc[0]
    failures = rows["Total Failure Count"].iloc[-1] - rows["Total Failure Count"].iloc[0]
    api = resources[(resources["contenedor"] == API) & resources["t"].between(t0, t1)]
    mlflow = resources[(resources["contenedor"] == MLFLOW) & resources["t"].between(t0, t1)]
    p95 = float(rows["95%"].median())
    error_pct = 100 * failures / requests if requests else 0.0
    return {
        "usuarios": int(rows["User Count"].median()),
        "rps": round(float(rows["Requests/s"].mean()), 1),
        "p50_ms": round(float(rows["50%"].median()), 1),
        "p95_ms": round(p95, 1),
        "error_pct": round(error_pct, 2),
        "cumple_objetivo": bool(p95 <= SLO_P95_MS and error_pct <= SLO_ERROR_PCT),
        "cpu_api_media_pct": round(float(api["cpu_pct"].mean()), 1) if len(api) else None,
        "cpu_api_max_pct": round(float(api["cpu_pct"].max()), 1) if len(api) else None,
        "mem_api_max_mib": round(float(api["mem_mib"].max()), 1) if len(api) else None,
        "cpu_mlflow_media_pct": round(float(mlflow["cpu_pct"].mean()), 1) if len(mlflow) else None,
        "mem_mlflow_max_mib": round(float(mlflow["mem_mib"].max()), 1) if len(mlflow) else None,
    }


def stage_windows(agg, scenario):
    """Escalones: cada nivel de usuarios sin el calentamiento. Estrés: tramos de 50 usuarios."""
    windows = []
    if scenario == "escalones":
        for users, rows in agg.groupby((agg["User Count"] != agg["User Count"].shift()).cumsum()):
            level = rows["User Count"].iloc[0]
            if level > 0 and len(rows) > WARMUP_S:
                windows.append((rows["t"].iloc[0] + WARMUP_S, rows["t"].iloc[-1] + 1))
    else:
        bins = (agg["User Count"] // 50) * 50
        for _, rows in agg.groupby(bins):
            windows.append((rows["t"].iloc[0], rows["t"].iloc[-1] + 1))
    return windows


def endpoint_totals(prefix: Path) -> list[dict]:
    stats = pd.read_csv(f"{prefix}_stats.csv")
    result = []
    for _, row in stats.iterrows():
        requests = int(row["Request Count"])
        result.append(
            {
                "endpoint": row["Name"],
                "peticiones": requests,
                "fallas": int(row["Failure Count"]),
                "error_pct": round(100 * row["Failure Count"] / requests, 2) if requests else 0,
                "promedio_ms": round(float(row["Average Response Time"]), 1),
                "p50_ms": float(row["50%"]),
                "p95_ms": float(row["95%"]),
                "p99_ms": float(row["99%"]),
                "rps_promedio": round(float(row["Requests/s"]), 1),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["escalones", "estres"], required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    prefix = args.out_dir / args.scenario
    history = load_history(prefix)
    agg = aggregated(history)
    start = float(history["Timestamp"].min())
    resources = load_resources(args.out_dir / f"{args.scenario}_recursos.csv", start)
    break_file = args.out_dir / "estres_quiebre.json"
    breakpoint = None
    if args.scenario == "estres" and break_file.exists():
        breakpoint = json.loads(break_file.read_text())
        breakpoint["segundo"] = round(breakpoint["timestamp"] - start, 1)
        before = agg[agg["t"] <= breakpoint["segundo"]]
        breakpoint["usuarios_activos"] = (
            int(before["User Count"].iloc[-1]) if len(before) else None
        )

    plot_performance(
        agg, history, args.scenario, breakpoint, args.out_dir / f"{args.scenario}_desempeno.png"
    )
    plot_resources(agg, resources, args.scenario, args.out_dir / f"{args.scenario}_recursos.png")

    stages = [
        m
        for t0, t1 in stage_windows(agg, args.scenario)
        if (m := window_metrics(agg, resources, t0, t1))
    ]
    peak = agg.loc[agg["Requests/s"].idxmax()]
    summary = {
        "escenario": args.scenario,
        "duracion_s": int(agg["t"].max()),
        "objetivo": {"p95_ms": SLO_P95_MS, "error_pct": SLO_ERROR_PCT},
        "tramos": stages,
        "pico_rps": {
            "rps": round(float(peak["Requests/s"]), 1),
            "usuarios": int(peak["User Count"]),
        },
        "por_endpoint": endpoint_totals(prefix),
        "quiebre": breakpoint,
    }
    (args.out_dir / f"{args.scenario}_resumen.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    print(
        f"\nEscenario {args.scenario} · objetivo: p95 ≤ {SLO_P95_MS} ms y errores ≤ {SLO_ERROR_PCT} %"
    )
    header = f"{'usuarios':>8} {'rps':>7} {'p50 ms':>8} {'p95 ms':>8} {'error %':>8} {'CPU API %':>10} {'mem API MiB':>12} {'CPU MLflow %':>13}  objetivo"
    print(header)
    for s in stages:
        print(
            f"{s['usuarios']:>8} {s['rps']:>7} {s['p50_ms']:>8} {s['p95_ms']:>8} {s['error_pct']:>8}"
            f" {s['cpu_api_media_pct']!s:>10} {s['mem_api_max_mib']!s:>12} {s['cpu_mlflow_media_pct']!s:>13}"
            f"  {'cumple' if s['cumple_objetivo'] else 'NO cumple'}"
        )
    print(
        f"\nPico: {summary['pico_rps']['rps']} peticiones/s con {summary['pico_rps']['usuarios']} usuarios"
    )
    for e in summary["por_endpoint"]:
        print(
            f"  {e['endpoint']:<26} {e['peticiones']:>7} peticiones · {e['error_pct']} % error"
            f" · p50 {e['p50_ms']} ms · p95 {e['p95_ms']} ms · p99 {e['p99_ms']} ms"
        )
    if breakpoint:
        print("Quiebre:", json.dumps(breakpoint, ensure_ascii=False))
    print(f"\nGráficas y resumen en {args.out_dir}")


if __name__ == "__main__":
    main()
