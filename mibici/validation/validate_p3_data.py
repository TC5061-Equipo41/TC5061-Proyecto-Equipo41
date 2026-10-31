from pathlib import Path

import numpy as np

# Compatibilidad entre Deepchecks 0.19.1 y NumPy 2.x
if not hasattr(np, "Inf"):
    np.Inf = np.inf

import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity


# Rutas del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "p3_station_features.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "validation"
REPORT_PATH = REPORT_DIR / "p3_data_integrity.html"


def main():
    """Validate the quality and integrity of the P3 dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {DATA_PATH}"
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading P3 station features...", flush=True)

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}", flush=True)

    print("\nMissing values by column:", flush=True)
    print(df.isnull().sum(), flush=True)

    dataset = Dataset(
        df,
        label=None,
        cat_features=["Nombre"],
    )

    print("\nRunning Deepchecks data integrity suite...", flush=True)

    suite = data_integrity()
    results = suite.run(dataset)

    results.save_as_html(
    str(REPORT_PATH),
    as_widget=False,
    requirejs=False,
    connected=False,
    )

    print("\nValidation completed.", flush=True)
    print(f"HTML report saved at: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()