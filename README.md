# mibici

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project..

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         mibici and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── mibici   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mibici a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- Code to run model inference with trained models
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

---

### Setup inicial

1. Crear .venv `python -m venv .venv`
2. Activar .venv `source .venv/bin/activate`
3. Instalar dependencias `pip install -r requirements.txt`

---

### Como cargar el dataset inicial para uso en notebooks

1. Descarga los datos de nomenclatura y viajes mensuales https://mibici.net/es/datos-abiertos/
2. Revisa las declaraciones de `config.py`, modifica a partir de los datos que necesites

```

DATA_METADATA = {
    'monthly': YEARLY_DATA,
    'stations': STATIONS_DATA,
    'data_range': {'from': '2025_09', 'to': '2026_09'}
}
```

3. Ejecuta `dataset.py` para que se generen los datos procesados en /data/interim, observa los ejemplos propuestos en `dataset.py` y ajusta de ser necesario

---

### Como ejecutar el experimento

1. Correr el servidor de MLFlow local, ajusta to host y puerto de ser necesario

```
mlflow server --backend-store-uri sqlite:///mlflow.db  --host 127.0.0.1 --port 5000
```

2. Agregar las siguientes variables de ambiente a .env. Ajusta de ser necesario

```
MLFLOW_URI="http://127.0.0.1:5000"
```

3. Corre `mibici/modeling/train.py`, asegurate de antes haber completado el `paso 1` y `paso 2` de la sección `Como cargar el dataset inicial para uso en notebooks` de este README.md
4. Revisa resultados en http://127.0.0.1:5000

---
