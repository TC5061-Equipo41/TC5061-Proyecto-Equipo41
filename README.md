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

## Como cargar el dataset inicial

1. Descarga los datos de nomenclatura y viajes mensuales https://mibici.net/es/datos-abiertos/
2. Revisa las declaraciones de `config.py`, modifica a partir de los datos que necesites

```
    CURRENT_MONTH_DATA = ['datos_abiertos_2026_09.csv']
    THREE_MONTH_DATA = ['datos_abiertos_2026_08.csv', 'datos_abiertos_2026_07.csv', 'datos_abiertos_2026_06.csv']
    STATIONS_DATA = 'nomenclatura_2026_09.csv'
```

3. Ejecuta `dataset.py` para que se generen los datos procesados en /data/interim, observa los ejemplos propuestos en `dataset.py` y ajusta de ser necesario
## P3. Agrupamiento de estaciones por patrones de uso

### Pregunta de negocio

¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican en la ciudad?

### Objetivo

Identificar grupos de estaciones de MiBici con características de uso similares mediante un modelo de clustering. Los resultados pueden servir como apoyo para analizar la distribución de la demanda y explorar oportunidades de expansión del sistema.

### Datos y características

El análisis utiliza datos de viajes de MiBici y la información de nomenclatura de estaciones. Se construye un conjunto de características agregadas por estación, que incluye:

- Número de viajes de salida y llegada.
- Viajes de salida por franja horaria.
- Viajes de salida durante los fines de semana.
- Duración promedio de los viajes.
- Nombre y coordenadas geográficas de las estaciones.

Las variables numéricas utilizadas por el modelo se escalan antes de aplicar el algoritmo de clustering.

### Modelo y evaluación

Se utiliza K-Means para agrupar las estaciones. Se evalúan diferentes valores de K mediante el coeficiente de silueta (*silhouette score*), con el objetivo de comparar la separación y cohesión de los grupos.

El entrenamiento registra parámetros, métricas, etiquetas y artefactos en MLflow para facilitar la reproducibilidad y comparación de experimentos.

### Ejecución de P3

Desde la raíz del repositorio, ejecuta:

```bash
python -m mibici.modeling.p3.train
```

Para generar las asignaciones de grupos:

```bash
python -m mibici.modeling.p3.predict
```

Para generar el mapa de estaciones:

```bash
python -m mibici.modeling.p3.plot_clusters
```

### Resultados generados

- `models/p3_kmeans.pkl`: modelo entrenado.
- `data/processed/p3_station_features.csv`: características agregadas por estación.
- `data/processed/p3_station_clusters.csv`: asignación de las estaciones a sus grupos.
- `reports/figures/p3_clusters.png`: gráfica de evaluación del clustering.
- `reports/figures/p3_station_map.png`: mapa geográfico de las estaciones agrupadas.

### Interpretación

Los grupos permiten distinguir estaciones con diferentes niveles de actividad. La interpretación geográfica debe realizarse con el mapa y las características de cada grupo; no se debe asumir que los clústeres representan zonas geográficas claramente separadas sin evidencia adicional.

### Seguimiento de experimentos

Los experimentos de P3 se registran en MLflow bajo el experimento `mibici-p3-clustering`. Se guardan parámetros, métricas, etiquetas y artefactos para documentar y comparar las ejecuciones.

### Resultados e interpretación de los clústeres

El modelo K-Means seleccionó dos clústeres como resultado de la evaluación mediante el coeficiente de silueta. El conjunto analizado contiene 471 estaciones: 278 pertenecen al clúster 0 y 193 al clúster 1.

El clúster 0 presenta un promedio de 5,080.12 viajes de salida y 5,074.35 viajes de llegada por estación. En contraste, el clúster 1 registra 1,109.37 viajes de salida y 1,117.68 viajes de llegada por estación.

La duración promedio de los viajes es similar en ambos grupos: 10.94 minutos en el clúster 0 y 11.04 minutos en el clúster 1. Estos resultados sugieren que la principal diferencia entre los grupos está relacionada con el volumen de uso de las estaciones.

Para apoyar la interpretación espacial, se utiliza el mapa de estaciones agrupadas. La ubicación geográfica debe analizarse junto con los patrones de uso, sin asumir que cada clúster corresponde a una zona geográfica claramente delimitada.

Estos resultados pueden servir como punto de partida para estudiar la distribución de la demanda y detectar estaciones que podrían requerir análisis adicional para la planeación del sistema.