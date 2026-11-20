"""Prepara las entradas de la validación de datos con DeepChecks (entorno principal).

Se ejecuta con el entorno del proyecto (Python 3.14) antes de la validación:

    python -m mibici.validation.prepare_inputs      # o: make validate

1. Carga o reconstruye el dataset común (``data/interim/*.parquet``) y su manifiesto.
2. Genera ``data/processed/p3_station_features.csv`` si falta o si cambió ``data_version``.

Después, ``mibici/validation/validate_p3_data.py`` corre en el entorno de validación
(Python 3.12, ``requirements-validation.txt``) y lee esos archivos: así DeepChecks no necesita
instalarse junto al stack principal, con el que no es compatible.
"""

from mibici.config import DATA_METADATA
from mibici.dataset import load_base_dataset
from mibici.modeling.train import P3_FEATURES, ensure_p3_features


def main() -> None:
    base_dataset, manifest = load_base_dataset(DATA_METADATA)
    ensure_p3_features(base_dataset, manifest["data_version"])
    print(
        f"Entradas listas para DeepChecks · data_version={manifest['data_version']} · "
        f"{len(base_dataset):,} viajes · {P3_FEATURES.name}"
    )


if __name__ == "__main__":
    main()
