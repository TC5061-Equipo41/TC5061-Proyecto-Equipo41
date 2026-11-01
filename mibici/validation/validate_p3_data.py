
from pathlib import Path

import numpy as np

# Compatibilidad entre Deepchecks 0.19.1 y NumPy 2.x
if not hasattr(np, "Inf"):
    np.Inf = np.inf

if not hasattr(np, "NaN"):
    np.NaN = np.nan

import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity

from mibici.config import DATA_METADATA
from mibici.dataset import load_base_dataset
from mibici.p1.dataset import create_p1_dataset
from mibici.p2.dataset import create_p2_dataset


# --------------------------------------------------
# Rutas y configuración
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

P3_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "p3_station_features.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "validation"


# Variables utilizadas por los modelos
P1_FEATURES = ["Origen_Id", "Hora", "Dia"]
P1_TARGET = "Viajes"

P2_FEATURES = [
    "departures",
    "arrivals",
    "capacity",
    "net_flow",
    "hour",
    "day_of_week",
    "is_weekend",
    "month",
    "departures_lag1",
    "arrivals_lag1",
    "net_flow_lag1",
]
P2_TARGET = "target_risk"


# --------------------------------------------------
# Funciones de validación
# --------------------------------------------------

def run_validation(
    df: pd.DataFrame,
    project_name: str,
    label: str | None,
    categorical_features: list[str],
    features: list[str] | None = None,
    index_name: str | None = None,
    datetime_name: str | None = None,
):
    """Ejecuta Deepchecks y genera un reporte HTML."""

    if df.empty:
        raise ValueError(
            f"El dataset de {project_name} está vacío."
        )

    if label is not None and label not in df.columns:
        raise ValueError(
            f"No se encontró la variable objetivo "
            f"'{label}' en {project_name}."
        )

    missing_columns = [
        col for col in categorical_features
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Faltan columnas categóricas en {project_name}: "
            f"{missing_columns}"
        )

    print(f"\n{'=' * 60}")
    print(f"VALIDACIÓN {project_name}")
    print(f"{'=' * 60}")

    print(f"Dataset shape: {df.shape}")

    print("\nMissing values by column:")
    print(df.isnull().sum())

    dataset = Dataset(
        df,
        label=label,
        features=features,
        cat_features=categorical_features,
        index_name=index_name,
        datetime_name=datetime_name,
    )

    print("\nRunning Deepchecks data integrity suite...")

    suite = data_integrity()

    # Ajustar la tolerancia del PPS de identificadores.
    # Se conserva una tolerancia pequeña para evitar
    # alertas por valores numéricamente insignificantes.
    if index_name is not None or datetime_name is not None:
        from deepchecks.tabular.checks import IdentifierLabelCorrelation

        for check in suite.checks.values():
            if isinstance(check, IdentifierLabelCorrelation):
                check.clean_conditions()
                check.add_condition_pps_less_or_equal(max_pps=0.001)

    # Compatibilidad entre Deepchecks 0.19.1 y pandas 3.x.
    # Deepchecks espera que stack() descarte los valores NaN.
    original_stack = pd.DataFrame.stack

    def compatible_stack(self, *args, **kwargs):
        if "future_stack" not in kwargs and "dropna" not in kwargs:
            kwargs["future_stack"] = False

        return original_stack(self, *args, **kwargs)

    try:
        pd.DataFrame.stack = compatible_stack
        results = suite.run(dataset)
    finally:
        # Restauramos el comportamiento original de pandas.
        pd.DataFrame.stack = original_stack

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report_path = (
        REPORT_DIR
        / f"{project_name.lower()}_data_integrity.html"
    )

    # DeepChecks no sobrescribe: si el archivo existe crea "<nombre> (1).html". Se borra antes
    # para que cada ejecución reemplace el reporte en lugar de acumular copias.
    report_path.unlink(missing_ok=True)

    results.save_as_html(
        str(report_path),
        as_widget=False,
        requirejs=False,
        connected=False,
    )

    print("\nValidation completed.")
    print(f"HTML report saved at: {report_path}")


# --------------------------------------------------
# P1: Predicción de viajes por estación y hora
# --------------------------------------------------


def validate_p1(base_data: pd.DataFrame):
    """Valida la calidad del dataset de P1."""

    print("\nBuilding P1 dataset...")

    df = create_p1_dataset(base_data.copy())

    # Conservamos Fecha para identificar cada observación
    # temporal, pero no la utilizamos como predictor.
    df = df[["Fecha"] + P1_FEATURES + [P1_TARGET]].copy()

    # Fecha se utiliza como índice temporal de validación.
    df = df.set_index("Fecha")

    run_validation(
        df=df,
        project_name="P1",
        label=P1_TARGET,
        categorical_features=P1_FEATURES,
    )



# --------------------------------------------------
# P2: Clasificación de riesgo de estaciones
# --------------------------------------------------

def validate_p2(base_data: pd.DataFrame):
    """Valida la calidad del dataset de P2."""

    print("\nBuilding P2 dataset...")

    df = create_p2_dataset(base_data.copy())

    # Mismo criterio de eliminación de registros incompletos
    # utilizado en el entrenamiento de P2.
    required_columns = [
        "future_pressure",
        "departures_lag1",
        "arrivals_lag1",
        "net_flow_lag1",
    ]

    df = df.dropna(subset=required_columns)

    # Se conservan los identificadores para distinguir
    # cada observación de estación y hora.
    df = df[
        ["station_id", "fecha_hora"] + P2_FEATURES + [P2_TARGET]
    ].copy()

    run_validation(
        df=df,
        project_name="P2",
        label=P2_TARGET,
        categorical_features=[
            "hour",
            "day_of_week",
            "is_weekend",
            "month",
        ],
        features=P2_FEATURES,
        index_name="station_id",
        datetime_name="fecha_hora",
    )


# --------------------------------------------------
# P3: Características de estaciones
# --------------------------------------------------

def validate_p3():
    """Valida la calidad del dataset de P3."""

    print("\nLoading P3 station features...")

    if not P3_DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {P3_DATA_PATH}"
        )

    df = pd.read_csv(P3_DATA_PATH)

    run_validation(
        df=df,
        project_name="P3",
        label=None,
        categorical_features=["Nombre"],
    )


# --------------------------------------------------
# Ejecución principal
# --------------------------------------------------

def main():
    """Ejecuta las validaciones de P1, P2 y P3."""

    print("=" * 60)
    print("MIBICI - DATA QUALITY VALIDATION")
    print("=" * 60)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Dataset común para P1 y P2
    print("\nLoading common dataset...")

    base_data, manifest = load_base_dataset(DATA_METADATA)

    print(f"Base dataset shape: {base_data.shape}")
    print(f"Data version: {manifest.get('data_version')}")

    # Validaciones individuales
    validate_p1(base_data)
    validate_p2(base_data)
    validate_p3()

    print("\n" + "=" * 60)
    print("ALL VALIDATIONS COMPLETED")
    print("=" * 60)

    print(f"Reports directory: {REPORT_DIR}")


if __name__ == "__main__":
    main()
