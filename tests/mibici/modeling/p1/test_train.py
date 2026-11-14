import mlflow
import numpy as np
import pandas as pd
import pytest

from mibici.modeling.p1 import train

MODELS = list(train.P1_MODELS)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def make_split(n: int, seed: int):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "Origen_Id": rng.choice([1, 2, 3], size=n),
            "Hora": rng.integers(0, 24, size=n),
            "Dia": rng.integers(0, 7, size=n),
        }
    )
    y = pd.Series(rng.poisson(3, size=n).astype(float), name="Viajes")
    return X, y


@pytest.fixture
def splits():
    X_train, y_train = make_split(200, seed=0)
    X_test, y_test = make_split(50, seed=1)
    return X_train, y_train, X_test, y_test


@pytest.fixture(autouse=True)
def sqlite_mlflow(isolated_mlflow, tmp_path, monkeypatch):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)


@pytest.fixture(autouse=True)
def no_plots(monkeypatch):
    """Evita generar gráficas durante las pruebas."""
    monkeypatch.setattr(train, "plot_prediction_vs_real_data", lambda *a, **k: None)


# ---------------------------------------------------------------------------
# validate_model_and_params
# ---------------------------------------------------------------------------
def test_validate_rejects_unknown_model():
    with pytest.raises(ValueError):
        train.validate_model_and_params("xgboost", None)


def test_validate_rejects_params_with_all():
    with pytest.raises(ValueError):
        train.validate_model_and_params("all", {"alpha": 1.0})


# ---------------------------------------------------------------------------
# get_models
# ---------------------------------------------------------------------------
def test_get_models_covers_all_p1_models():
    models = train.get_models()
    assert set(models) == set(MODELS)
    for spec in models.values():
        assert {"pipeline", "grid", "base"} <= set(spec)


# ---------------------------------------------------------------------------
# Semilla
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", MODELS)
def test_model_uses_fixed_seed(name):
    spec = train.get_models()[name]
    estimator = spec["pipeline"](spec["base"]).steps[-1][1]
    assert estimator.random_state == train.SEED


@pytest.mark.parametrize("name", MODELS)
def test_model_is_reproducible_with_seed(name, splits):
    X_train, y_train, X_test, _ = splits
    spec = train.get_models()[name]

    preds = [
        spec["pipeline"](spec["base"]).fit(X_train, y_train).predict(X_test)
        for _ in range(2)
    ]
    np.testing.assert_allclose(preds[0], preds[1])


@pytest.mark.parametrize("name", MODELS)
def test_run_model_is_reproducible_with_seed(name, splits, monkeypatch):
    monkeypatch.setattr(train, "TRAIN_CHALLENGER", False)
    params = train.get_models()[name]["base"]
    for _ in range(2):
        train.run_model(None, "v1", {"from": "2026-07", "to": "2026-08"}, name, params, splits)

    runs = mlflow.search_runs(search_all_experiments=True)
    assert len(runs) == 2

    metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
    assert metric_cols
    first, second = runs[metric_cols].to_numpy()
    np.testing.assert_allclose(first, second)


# ---------------------------------------------------------------------------
# Entrenamiento y evaluación
# ---------------------------------------------------------------------------
def test_eval_model_returns_non_negative_rmse(splits):
    spec = train.get_models()["ridge"]
    model = spec["pipeline"](spec["base"]).fit(splits[0], splits[1])

    with mlflow.start_run():
        rmse = train.eval_model(model, splits)

    assert isinstance(rmse, float)
    assert rmse >= 0


def test_model_comparison_registers_best_model(splits, monkeypatch):
    registered = {}

    def fake_register(name, run_id, model, splits):
        registered["name"] = name

    monkeypatch.setattr(train, "register_champion", fake_register)

    train.run_model_comparison("v1", {"from": "2026-07", "to": "2026-08"}, splits)

    assert registered["name"] in MODELS
