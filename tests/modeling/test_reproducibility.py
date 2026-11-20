"""Reproducibilidad del entrenamiento de P1, P2 y P3.

Cada prueba entrena dos veces el mismo modelo, con la misma semilla y los mismos datos, y
exige resultados idénticos: predicciones (P1), probabilidades y métricas (P2), etiquetas y
silhouette (P3). Las pruebas de control cambian la semilla y confirman que la comparación sí
detecta diferencias, para que una prueba que "siempre pasa" no oculte un error.

Se usan las funciones reales del proyecto (construcción de datasets, partición temporal y
pipelines) sobre datos sintéticos pequeños, sin MLflow ni archivos de data/.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import average_precision_score, f1_score, silhouette_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from mibici.modeling.p1.train import get_models
from mibici.modeling.p2.train import FEATURES as P2_FEATURES
from mibici.modeling.p2.train import TARGET as P2_TARGET
from mibici.modeling.p2.train import build_logistic_model
from mibici.modeling.p2.train import get_train_test_split as p2_split
from mibici.modeling.p3.train import prepare_features
from mibici.p1.dataset import create_p1_dataset
from mibici.p1.dataset import get_train_test_split as p1_split
from mibici.p2.dataset import create_p2_dataset

SEED = 42


# ---------------------------------------------------------------------------
# Datos sintéticos
# ---------------------------------------------------------------------------
@pytest.fixture
def trips_two_months() -> pd.DataFrame:
    """Viajes de julio y agosto de 2026 en 4 estaciones de capacidad pequeña.

    Julio queda en entrenamiento y agosto en prueba para las particiones temporales de P1
    (corte 2026-08-01) y P2 (último mes = agosto). La capacidad baja (4 anclajes) hace que haya
    horas con y sin riesgo, así P2 tiene ambas clases.
    """
    rng = np.random.default_rng(SEED)
    n = 3000
    stations = np.array([1, 2, 3, 4])
    start = pd.Timestamp("2026-07-01 06:00") + pd.to_timedelta(
        rng.integers(0, 61 * 24 * 60, size=n), unit="min"
    )
    origin = rng.choice(stations, size=n, p=[0.4, 0.3, 0.2, 0.1])
    destination = rng.choice(stations, size=n, p=[0.1, 0.2, 0.3, 0.4])
    return pd.DataFrame(
        {
            "Viaje_Id": np.arange(1, n + 1),
            "Origen_Id": origin,
            "Destino_Id": destination,
            "Inicio_del_viaje": start,
            "Fin_del_viaje": start + pd.to_timedelta(rng.integers(3, 40, size=n), unit="min"),
            "Origen_dpcapacity": 4,
            "Destino_dpcapacity": 4,
        }
    )


@pytest.fixture
def station_features() -> pd.DataFrame:
    """Características por estación con el esquema de p3_station_features.csv."""
    rng = np.random.default_rng(SEED)
    n = 60
    salidas = rng.integers(200, 6000, size=n)
    parts = rng.dirichlet([3, 2, 2, 1], size=n)
    return pd.DataFrame(
        {
            "Estacion_Id": np.arange(1, n + 1),
            "viajes_salida": salidas,
            "viajes_llegada": (salidas * rng.uniform(0.8, 1.2, size=n)).astype(int),
            "salidas_manana": (salidas * parts[:, 0]).astype(int),
            "salidas_tarde": (salidas * parts[:, 1]).astype(int),
            "salidas_noche": (salidas * parts[:, 2]).astype(int),
            "salidas_madrugada": (salidas * parts[:, 3]).astype(int),
            "salidas_fin_semana": (salidas * rng.uniform(0.1, 0.4, size=n)).astype(int),
            "duracion_promedio": rng.uniform(8, 15, size=n),
        }
    )


# ---------------------------------------------------------------------------
# P1 · Regresión de demanda
# ---------------------------------------------------------------------------
def _p1_fit_predict(trips: pd.DataFrame, model_name: str, seed: int = SEED) -> np.ndarray:
    spec = get_models()[model_name]
    params = dict(spec["base"])
    if model_name == "random_forest":
        params["n_jobs"] = 1  # un solo hilo: el orden de los árboles no depende del sistema
    pipeline = spec["pipeline"](params)
    pipeline.steps[-1][1].set_params(random_state=seed)
    X_train, y_train, X_test, _ = p1_split(create_p1_dataset(trips.copy()))
    pipeline.fit(X_train, y_train)
    return pipeline.predict(X_test)


@pytest.mark.parametrize("model_name", ["ridge", "random_forest", "hist_gb"])
def test_p1_same_seed_same_predictions(trips_two_months, model_name):
    first = _p1_fit_predict(trips_two_months, model_name)
    second = _p1_fit_predict(trips_two_months, model_name)
    assert len(first) > 0
    np.testing.assert_array_equal(first, second)


def test_p1_control_different_seed_changes_random_forest(trips_two_months):
    """Control: con otra semilla el bosque aleatorio cambia, así que la prueba anterior compara
    algo que de verdad depende de la semilla."""
    X_train, y_train, X_test, _ = p1_split(create_p1_dataset(trips_two_months.copy()))

    def fit(seed):
        model = make_pipeline(
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            RandomForestRegressor(
                n_estimators=20, max_features=0.5, max_samples=0.3, random_state=seed, n_jobs=1
            ),
        )
        return model.fit(X_train, y_train).predict(X_test)

    assert not np.array_equal(fit(SEED), fit(SEED + 1))


# ---------------------------------------------------------------------------
# P2 · Clasificación de riesgo
# ---------------------------------------------------------------------------
def _p2_splits(trips: pd.DataFrame):
    dataset = create_p2_dataset(trips.copy())
    X_train, y_train, X_test, y_test, _, _ = p2_split(dataset, {"from": "2026_07", "to": "2026_08"})
    return X_train, y_train, X_test, y_test


def test_p2_data_has_both_classes(trips_two_months):
    """Precondición: sin ambas clases la regresión logística no se puede entrenar."""
    _, y_train, X_test, _ = _p2_splits(trips_two_months)
    assert set(y_train.unique()) == {0, 1}
    assert len(X_test) > 0
    assert list(X_test.columns) == P2_FEATURES


def test_p2_same_seed_same_probabilities_and_metrics(trips_two_months):
    X_train, y_train, X_test, y_test = _p2_splits(trips_two_months)
    results = []
    for _ in range(2):
        model = build_logistic_model(c_value=1.0).fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        results.append(
            (
                proba,
                f1_score(y_test, model.predict(X_test), zero_division=0),
                average_precision_score(y_test, proba),
            )
        )
    np.testing.assert_array_equal(results[0][0], results[1][0])
    assert results[0][1:] == results[1][1:]


def test_p2_target_name_matches_dataset(trips_two_months):
    dataset = create_p2_dataset(trips_two_months.copy())
    assert P2_TARGET in dataset.columns


# ---------------------------------------------------------------------------
# P3 · Clustering de estaciones
# ---------------------------------------------------------------------------
def _p3_pipeline(k: int, seed: int) -> Pipeline:
    """Mismo pipeline que la búsqueda de k en mibici/modeling/p3/train.py."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clusterer", KMeans(n_clusters=k, random_state=seed, n_init=10)),
        ]
    )


def test_p3_same_seed_same_labels_and_silhouette(station_features):
    data, columns = prepare_features(station_features)
    X = data[columns]
    pipeline = _p3_pipeline(k=4, seed=SEED)
    labels_a = clone(pipeline).fit_predict(X)
    labels_b = clone(pipeline).fit_predict(X)
    np.testing.assert_array_equal(labels_a, labels_b)
    scaled = StandardScaler().fit_transform(X)
    assert silhouette_score(scaled, labels_a) == silhouette_score(scaled, labels_b)


def test_p3_control_single_init_depends_on_seed(station_features):
    """Control: con una sola inicialización, semillas distintas dan centroides distintos."""
    data, columns = prepare_features(station_features)
    X = StandardScaler().fit_transform(data[columns])

    def centroids(seed):
        model = KMeans(n_clusters=6, random_state=seed, n_init=1, init="random").fit(X)
        return np.sort(model.cluster_centers_, axis=0)

    assert not np.allclose(centroids(SEED), centroids(SEED + 7))
