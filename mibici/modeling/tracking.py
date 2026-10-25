"""Utilidades comunes de MLflow para P1, P2 y P3.

Funciones principales (las usan ``modeling/train.py`` y cualquier ``train_pX_model``):

- ``set_common_tags()``: etiquetas obligatorias del equipo (team, git_commit, question,
  model_family, stage_of_work).
- ``log_data_version()``: manifiesto de datos como artefacto en ``data/``, etiqueta
  ``data_version`` y el conjunto de entrenamiento con ``mlflow.log_input``.
- ``log_run_config()``: configuración de la corrida en ``config/``.

Todas reciben ``run_id`` opcional: sin él actúan sobre la corrida activa (uso dentro de
``with mlflow.start_run():``); con él actúan sobre una corrida ya terminada. Ejemplo:

    from mibici.modeling.tracking import log_data_version, log_run_config, set_common_tags

    with mlflow.start_run(run_name="P1-candidato-ridge"):
        set_common_tags(question="p1", model_family="ridge", stage_of_work="candidate")
        log_data_version(dataset=X_train.assign(target=y_train), targets="target")
        log_run_config({"model": "ridge", "params": params})
        ...
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import warnings

import mlflow
from mlflow.entities import DatasetInput, InputTag
from mlflow.tracking import MlflowClient
import pandas as pd

from mibici.config import EXPERIMENT_NAME, MANIFEST_PATH, MLFLOW_TRACKING_URI, PROJ_ROOT, TEAM_NAME

STAGES = ("baseline", "search", "candidate", "ingest", "dispatch", "registry")
CONFIG_FILES = ("MLproject", "python_env.yaml", "requirements.txt", "mibici/config.py")

# Aviso informativo de MLflow al inferir esquemas con columnas enteras (ids de estación)
warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer column")


# ---------------------------------------------------------------------------
# Servidor, experimento y código
# ---------------------------------------------------------------------------
def setup_tracking(experiment_name: str = EXPERIMENT_NAME) -> str:
    """Apunta MLflow al servidor de ``config.MLFLOW_TRACKING_URI`` y activa el experimento."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    return MLFLOW_TRACKING_URI


def get_git_commit() -> str:
    """Commit actual de Git (``unknown`` si no hay repositorio)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except subprocess.CalledProcessError, FileNotFoundError:
        return "unknown"


def git_is_dirty() -> bool:
    """True si hay cambios sin commit en archivos versionados (el run no sería reproducible)."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=PROJ_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(out.stdout.strip())
    except subprocess.CalledProcessError, FileNotFoundError:
        return False


def _resolve_run_id(run_id: str | None) -> str:
    if run_id:
        return run_id
    active = mlflow.active_run()
    if active is None:
        raise RuntimeError(
            "No hay una corrida activa: pasa run_id o usa `with mlflow.start_run()`."
        )
    return active.info.run_id


# ---------------------------------------------------------------------------
# 1. Etiquetas comunes
# ---------------------------------------------------------------------------
def set_common_tags(
    question: str,
    model_family: str,
    stage_of_work: str,
    run_id: str | None = None,
    extra: dict | None = None,
    overwrite: bool = True,
) -> dict:
    """Etiqueta la corrida con equipo, commit, pregunta, familia del modelo y etapa del trabajo.

    ``stage_of_work`` es una de ``STAGES`` (baseline, search, candidate, ...). Con
    ``overwrite=False`` no reemplaza las etiquetas que la corrida ya tenga. También
    sobrescribe ``mlflow.user`` (expone el usuario del sistema operativo del integrante).
    """
    if stage_of_work not in STAGES:
        raise ValueError(f"stage_of_work debe ser uno de {STAGES}")
    run_id = _resolve_run_id(run_id)
    tags = {
        "team": TEAM_NAME,
        "git_commit": get_git_commit(),
        "git_dirty": str(git_is_dirty()).lower(),
        "question": question.upper(),
        "model_family": model_family,
        "stage_of_work": stage_of_work,
        "mlflow.user": TEAM_NAME,
        **(extra or {}),
    }
    client = MlflowClient()
    if not overwrite:
        current = client.get_run(run_id).data.tags
        tags = {k: v for k, v in tags.items() if k not in current or k == "mlflow.user"}
    for key, value in tags.items():
        client.set_tag(run_id, key, str(value))
    return tags


# ---------------------------------------------------------------------------
# 2. Versión de datos
# ---------------------------------------------------------------------------
def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    """Lee ``data/manifest.json``; falla con un mensaje claro si no se ha corrido la ingesta."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta la ingesta: mlflow run . -e ingest (ver README)."
        )
    return json.loads(Path(path).read_text(encoding="utf-8"))


def log_data_version(
    dataset: pd.DataFrame | None = None,
    run_id: str | None = None,
    name: str | None = None,
    targets: str | None = None,
    context: str = "training",
    manifest_path: Path = MANIFEST_PATH,
) -> dict:
    """Registra la versión de los datos de la corrida.

    - ``data/manifest.json`` como artefacto (archivos, filas, rango de fechas y hash).
    - Etiqueta ``data_version``.
    - Si se pasa ``dataset``, lo registra con ``mlflow.log_input`` (MLflow guarda esquema,
      número de filas y un digest; no sube las filas). No debe contener datos personales.

    Devuelve el manifiesto.
    """
    run_id = _resolve_run_id(run_id)
    manifest = load_manifest(manifest_path)
    client = MlflowClient()
    client.log_artifact(run_id, str(manifest_path), artifact_path="data")
    client.set_tag(run_id, "data_version", manifest["data_version"])
    if dataset is not None:
        source = f"data/interim#data_version={manifest['data_version']}"
        ds_name = name or f"mibici-{manifest['data_version']}"
        try:
            ds = mlflow.data.from_pandas(dataset, source=source, name=ds_name, targets=targets)
        except Exception:  # noqa: BLE001 - algunas versiones solo aceptan rutas como fuente
            ds = mlflow.data.from_pandas(dataset, name=ds_name, targets=targets)
        active = mlflow.active_run()
        if active is not None and active.info.run_id == run_id:
            mlflow.log_input(ds, context=context)
        else:
            tag = InputTag(key="mlflow.data.context", value=context)
            client.log_inputs(run_id, datasets=[DatasetInput(ds._to_mlflow_entity(), [tag])])
    return manifest


# ---------------------------------------------------------------------------
# 3. Configuración de la corrida
# ---------------------------------------------------------------------------
def log_run_config(
    run_config: dict, run_id: str | None = None, include_project_files: bool = True
) -> None:
    """Guarda en ``config/`` el ``run_config.json`` (pregunta, modelo, hiperparámetros, datos,
    commit…) y una copia de MLproject, python_env.yaml, requirements.txt y config.py."""
    run_id = _resolve_run_id(run_id)
    client = MlflowClient()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "run_config.json"
        path.write_text(
            json.dumps(run_config, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        client.log_artifact(run_id, str(path), artifact_path="config")
    if include_project_files:
        for rel in CONFIG_FILES:
            file = PROJ_ROOT / rel
            if file.exists():
                client.log_artifact(run_id, str(file), artifact_path="config")
