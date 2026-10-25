"""Despachador de entrenamiento: llama a ``train_pX_model(...)`` según ``question``.

Uso (desde la raíz del repositorio):

    python -m mibici.modeling.train                          # P1, P2 y P3 (como antes)
    python -m mibici.modeling.train --question p1 --params "alpha=0.5"
    mlflow run . -e train --experiment-name mibici-equipo41 -P question=p2

Contrato con cada pregunta (``mibici/modeling/pX/train.py``):

    def train_pX_model(original_data, data_version, date_range, model=None, params=None): ...

- ``original_data``: dataset común de ``dataset.build_initial_dataset`` (sin datos personales).
- ``data_version``: versión de ``data/manifest.json``; ``date_range``: ``DATA_METADATA``.
- ``model`` y ``params`` son opcionales: el despachador solo los pasa si la función los
  acepta. Sus valores por defecto viven en ``config.TRAIN_DEFAULTS``.

Cada ``train_pX_model`` abre sus propias corridas con ``mlflow.start_run``. Al terminar, el
despachador:
1. las cuelga como corridas hijas de la corrida del despachador (la que crea ``mlflow run``);
2. les agrega las etiquetas comunes que falten (``set_common_tags``);
3. registra la versión de datos y el dataset (``log_data_version``) y la configuración
   (``log_run_config``).
Así todas las corridas quedan con el mismo linaje aunque cada integrante escriba su código.
"""

from __future__ import annotations

import os
import re
import time

from loguru import logger
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import typer

from mibici.config import (
    DATA_METADATA,
    EXPERIMENT_NAME,
    INTERIM_DATA_DIR,
    MLFLOW_TRACKING_URI,
    PROCESSED_DATA_DIR,
    TRAIN_DEFAULTS,
)
from mibici.dataset import load_base_dataset
from mibici.modeling.tracking import (
    get_git_commit,
    log_data_version,
    log_run_config,
    set_common_tags,
    setup_tracking,
)

app = typer.Typer(help="Despachador de entrenamiento de MiBici (P1, P2, P3).")
QUESTIONS = ("p1", "p2", "p3")


# ---------------------------------------------------------------------------
# Parámetros
# ---------------------------------------------------------------------------
def _cast(value: str):
    v = value.strip()
    if v.lower() in ("none", "null"):
        return None
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    for kind in (int, float):
        try:
            return kind(v)
        except ValueError:
            pass
    return v


def parse_params(text: str) -> dict:
    """``"alpha=0.5,max_iter=2000"`` → ``{"alpha": 0.5, "max_iter": 2000}``."""
    if not text or text.strip().lower() == "default":
        return {}
    out = {}
    for pair in text.split(","):
        if not pair.strip():
            continue
        if "=" not in pair:
            raise typer.BadParameter(f"Formato esperado clave=valor, recibido: '{pair}'")
        key, value = pair.split("=", 1)
        out[key.strip()] = _cast(value)
    return out


def resolve_model_and_params(question: str, model: str, params: str) -> tuple[str, dict, dict]:
    """Combina los defaults de ``config.TRAIN_DEFAULTS`` con lo recibido por línea de comandos."""
    defaults = TRAIN_DEFAULTS[question]
    overrides = parse_params(params)
    chosen = defaults["model"] if model in ("", "default") else model
    return chosen, {**defaults["params"], **overrides}, overrides


def _call_question_fn(
    fn, question, base_dataset, data_version, date_range, model, params, overrides
):
    # Nombres de parámetros leídos del código (sin evaluar anotaciones de tipo, que en
    # algunos módulos no son evaluables, p. ej. `pd.Dataframe` en P1).
    code = fn.__code__
    accepted = set(code.co_varnames[: code.co_argcount + code.co_kwonlyargcount])
    kwargs = {}
    if "model" in accepted:
        kwargs["model"] = model
    if "params" in accepted:
        kwargs["params"] = params
    if (overrides or model != TRAIN_DEFAULTS[question]["model"]) and "params" not in accepted:
        logger.warning(
            f"{fn.__name__} aún no recibe model/params: se usan sus valores internos. "
            "Ver el contrato en mibici/modeling/train.py."
        )
    return fn(base_dataset, data_version, date_range, **kwargs)


# ---------------------------------------------------------------------------
# Preguntas
# ---------------------------------------------------------------------------
def run_p1(base_dataset, data_version, date_range, model, params, overrides):
    from mibici.modeling.p1.train import train_p1_model

    return _call_question_fn(
        train_p1_model, "p1", base_dataset, data_version, date_range, model, params, overrides
    )


def run_p2(base_dataset, data_version, date_range, model, params, overrides):
    from mibici.modeling.p2.train import train_p2_model

    return _call_question_fn(
        train_p2_model, "p2", base_dataset, data_version, date_range, model, params, overrides
    )


P3_FEATURES = PROCESSED_DATA_DIR / "p3_station_features.csv"
P3_INTERIM_CSV = INTERIM_DATA_DIR / "joined_trips_stations-2026_09-2025_10.csv"
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


def ensure_p3_features(base_dataset: pd.DataFrame, data_version: str) -> None:
    """Genera ``p3_station_features.csv`` con ``mibici/P3/features.py`` si falta o si los datos
    cambiaron (la versión se anota en un archivo junto al CSV)."""
    stamp = P3_FEATURES.with_suffix(".data_version")
    if P3_FEATURES.exists() and stamp.exists() and stamp.read_text().strip() == data_version:
        return
    from mibici.P3.features import main as build_p3_features

    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    base_dataset[P3_COLUMNS].to_csv(P3_INTERIM_CSV, index=False)
    build_p3_features(input_path=P3_INTERIM_CSV, output_path=P3_FEATURES)
    stamp.write_text(data_version)


def run_p3(base_dataset, data_version, date_range, model, params, overrides):
    from mibici.modeling.p3 import train as p3_train

    if hasattr(p3_train, "train_p3_model"):  # cuando P3 adopte el contrato común
        return _call_question_fn(
            p3_train.train_p3_model,
            "p3",
            base_dataset,
            data_version,
            date_range,
            model,
            params,
            overrides,
        )
    # Adaptador para el código actual de P3 (comando typer `main` que lee un CSV de features)
    ensure_p3_features(base_dataset, data_version)
    known = {"min_k", "max_k", "random_state"}
    return p3_train.main(
        features_path=P3_FEATURES,
        experiment_name=EXPERIMENT_NAME,
        date_range=f"{date_range['from'].replace('_', '-')} a {date_range['to'].replace('_', '-')}",
        **{k: v for k, v in params.items() if k in known},
    )


RUNNERS = {"p1": run_p1, "p2": run_p2, "p3": run_p3}


# ---------------------------------------------------------------------------
# Trazabilidad común
# ---------------------------------------------------------------------------
def _snake(text: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", text.replace(" ", "")).lower()


def _infer_stage(run, default: str) -> str:
    name = run.data.tags.get("mlflow.runName", "").lower()
    model = str(run.data.params.get("model", "")).lower()
    return "baseline" if any(w in name or w in model for w in ("dummy", "baseline")) else default


def _infer_family(run, default: str) -> str:
    model = run.data.params.get("model") or run.data.tags.get("modelo")
    return _snake(model) if model else default


def apply_common_tracking(
    run_ids: list[str],
    parent_id: str,
    question: str,
    model: str,
    stage: str,
    base_dataset: pd.DataFrame,
    run_config: dict,
) -> None:
    """Agrega linaje común a las corridas que creó ``train_pX_model`` (sin pisar sus etiquetas)."""
    client = MlflowClient()
    for run_id in run_ids:
        run = client.get_run(run_id)
        if "mlflow.parentRunId" not in run.data.tags:
            client.set_tag(run_id, "mlflow.parentRunId", parent_id)
        set_common_tags(
            question=question,
            model_family=_infer_family(run, model),
            stage_of_work=run.data.tags.get("stage_of_work") or _infer_stage(run, stage),
            run_id=run_id,
            extra={
                "launched_via": "mlflow_run" if run_config["launched_by_mlflow_run"] else "python"
            },
            overwrite=False,
        )
        log_data_version(dataset=base_dataset, run_id=run_id, name=f"{question}-base-dataset")
        log_run_config({**run_config, "question": question, "run_id": run_id}, run_id=run_id)


def _new_run_ids(experiment_id: str, since_ms: int, exclude: str) -> list[str]:
    runs = MlflowClient().search_runs(
        [experiment_id], filter_string=f"attributes.start_time >= {since_ms}", max_results=500
    )
    return [r.info.run_id for r in runs if r.info.run_id != exclude]


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------
def create_experiment() -> None:
    """Configura el servidor de tracking y el experimento del equipo (se conserva el nombre)."""
    os.environ["MLFLOW_URI"] = MLFLOW_TRACKING_URI  # el código de P3 lee esta variable
    setup_tracking(EXPERIMENT_NAME)


def dispatch(
    question: str = "all",
    model: str = "default",
    params: str = "default",
    stage_of_work: str = "default",
) -> None:
    question = question.lower()
    if question not in (*QUESTIONS, "all"):
        raise typer.BadParameter(f"question debe ser p1, p2, p3 o all (recibido: {question})")
    if question == "all" and (model != "default" or params != "default"):
        raise typer.BadParameter("model/params solo aplican a una pregunta: usa --question pX")
    stage = "candidate" if stage_of_work == "default" else stage_of_work
    launched = bool(os.environ.get("MLFLOW_RUN_ID"))
    create_experiment()
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    base_dataset, manifest = load_base_dataset(DATA_METADATA)
    data_version = manifest["data_version"]
    date_range = DATA_METADATA["data_range"]
    questions = list(QUESTIONS) if question == "all" else [question]

    # Corrida del despachador: retoma la que creó `mlflow run` (MLFLOW_RUN_ID) o crea una.
    with mlflow.start_run(run_name=f"dispatch-{question}-{data_version}") as parent:
        if parent.info.experiment_id != experiment.experiment_id:
            raise RuntimeError(
                f"La corrida quedó fuera de '{EXPERIMENT_NAME}'. Lanza con "
                f"`mlflow run . -e train --experiment-name {EXPERIMENT_NAME} ...`"
            )
        if not launched:
            mlflow.log_params(
                {
                    "question": question,
                    "model": model,
                    "params": params,
                    "stage_of_work": stage_of_work,
                }
            )
        set_common_tags(question=question, model_family="dispatcher", stage_of_work="dispatch")
        log_data_version()
        log_run_config(
            {
                "question": question,
                "model": model,
                "params": params,
                "data_version": data_version,
                "date_range": date_range,
                "experiment": EXPERIMENT_NAME,
                "git_commit": get_git_commit(),
            }
        )
        parent_id = parent.info.run_id
    os.environ.pop("MLFLOW_RUN_ID", None)  # las corridas de cada pregunta son nuevas (hijas)

    for q in questions:
        chosen, resolved, overrides = resolve_model_and_params(
            q,
            model if question != "all" else "default",
            params if question != "all" else "default",
        )
        logger.info(f"{q.upper()}: modelo={chosen} params={resolved}")
        t0 = int(time.time() * 1000) - 1000
        RUNNERS[q](base_dataset, data_version, date_range, chosen, resolved, overrides)
        mlflow.end_run()  # por si alguna función dejó una corrida abierta
        new_runs = _new_run_ids(experiment.experiment_id, t0, exclude=parent_id)
        run_config = {
            "model": chosen,
            "params": resolved,
            "params_overrides": overrides,
            "data_version": data_version,
            "date_range": date_range,
            "experiment": EXPERIMENT_NAME,
            "git_commit": get_git_commit(),
            "dispatcher_run_id": parent_id,
            "launched_by_mlflow_run": launched,
        }
        apply_common_tracking(new_runs, parent_id, q, chosen, stage, base_dataset, run_config)
        logger.success(f"{q.upper()}: {len(new_runs)} corrida(s) con linaje común")


def execute_runs() -> None:
    """Compatibilidad con la versión anterior: entrena P1, P2 y P3."""
    dispatch("all")


@app.command()
def main(
    question: str = typer.Option("all", help="p1, p2, p3 o all."),
    model: str = typer.Option("default", help="Modelo; default = config.TRAIN_DEFAULTS."),
    params: str = typer.Option("default", help='Hiperparámetros "clave=valor,clave=valor".'),
    stage_of_work: str = typer.Option("default", help="baseline | search | candidate."),
):
    """Entrena una pregunta (o todas) y registra las corridas en MLflow."""
    dispatch(question, model, params, stage_of_work)



if __name__ == "__main__":
    app()
