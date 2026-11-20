"""Chequeos de seguridad del repositorio como pruebas automáticas.

Verifican que el repositorio no exponga datos de MiBici, bases de MLflow, modelos ni
credenciales: ni en los archivos versionados hoy ni en ningún commit del historial, que las
rutas sensibles estén cubiertas por .gitignore, que .env.example no traiga valores secretos y
que detect-secrets no encuentre secretos nuevos respecto de .secrets.baseline.
"""

from pathlib import Path
import re
import shutil
import subprocess  # nosec B404 - solo git y detect-secrets, con argumentos fijos
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
GIT = shutil.which("git")

pytestmark = pytest.mark.skipif(
    GIT is None or not (ROOT / ".git").exists(), reason="requiere un clon de Git del proyecto"
)

# Archivos que nunca deben entrar al repositorio (datos, MLflow, modelos, credenciales)
FORBIDDEN = re.compile(
    r"(^|/)(data/(raw|interim|processed)/|mlruns/|mlartifacts/)"
    r"|\.(csv|parquet|db|sqlite|pkl|joblib|pem|key)$"
    r"|(^|/)\.env(\.(?!example$)[^/]*)?$"
    r"|(^|/)credentials[^/]*\.json$"
)


def _git(*args: str) -> str:
    out = subprocess.run(  # nosec B603 - ejecutable absoluto y argumentos fijos
        [GIT, *args], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return out.stdout


@pytest.mark.parametrize(
    "path",
    [
        "data/raw/datos_abiertos_2026_09.csv",
        "data/interim/joined_trips_stations-2025_10-2026_09.parquet",
        "data/processed/p3_station_features.csv",
        "mlflow.db",
        "mlartifacts/1/model.pkl",
        "mlruns/0/meta.yaml",
        "models/p3_kmeans.pkl",
        ".env",
        ".env.local",
    ],
)
def test_sensitive_paths_are_ignored(path):
    """Cada ruta sensible debe estar cubierta por .gitignore."""
    result = subprocess.run(  # nosec B603 - ejecutable absoluto y argumentos fijos
        [GIT, "check-ignore", "-q", "--no-index", path], cwd=ROOT, check=False
    )
    assert result.returncode == 0, f"{path} no está en .gitignore"


def test_manifest_and_env_example_are_versionable():
    """El manifiesto de datos y la plantilla de variables sí deben poder versionarse."""
    for path in ("data/manifest.json", ".env.example"):
        result = subprocess.run(  # nosec B603 - ejecutable absoluto y argumentos fijos
            [GIT, "check-ignore", "-q", "--no-index", path], cwd=ROOT, check=False
        )
        assert result.returncode != 0, f"{path} quedó ignorado"


def test_no_sensitive_files_tracked():
    """Ningún archivo versionado hoy corresponde a datos, MLflow, modelos o credenciales."""
    tracked = _git("ls-files").splitlines()
    offenders = [f for f in tracked if FORBIDDEN.search(f)]
    assert offenders == []


def test_history_never_contained_sensitive_files():
    """Ningún commit del historial (todas las ramas) agregó archivos sensibles."""
    history = set(_git("log", "--all", "--name-only", "--format=").split())
    offenders = sorted(f for f in history if FORBIDDEN.search(f))
    assert offenders == []


def test_env_example_has_no_secret_values():
    """.env.example documenta variables sin incluir valores secretos."""
    env_example = ROOT / ".env.example"
    assert env_example.exists(), "falta .env.example"
    secretish = re.compile(r"(PASSWORD|SECRET|TOKEN|API_KEY|ACCESS_KEY)", re.IGNORECASE)
    for line in env_example.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        if secretish.search(key):
            assert value.strip().strip("\"'") == "", f"{key} trae un valor en .env.example"


def test_detect_secrets_finds_nothing_new():
    """detect-secrets no encuentra secretos nuevos respecto de .secrets.baseline."""
    pytest.importorskip("detect_secrets")
    files = [f for f in _git("ls-files").splitlines() if (ROOT / f).is_file()]
    result = subprocess.run(  # nosec B603 - intérprete actual y argumentos fijos
        [sys.executable, "-m", "detect_secrets.pre_commit_hook", "--baseline",
         ".secrets.baseline", *files],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
