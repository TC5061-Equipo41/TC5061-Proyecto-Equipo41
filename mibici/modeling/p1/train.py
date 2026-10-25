
from loguru import logger
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from mibici.modeling.p1.evaluate import log_metrics
from mibici.p1.dataset import create_p1_dataset, get_train_test_split
from mibici.p1.plots import plot_prediction_vs_real_data

P1_MODELS = ("ridge", "random_forest", "hist_gb")
RUN_GRID = False
SEED = 42
REGISTERED_MODEL_NAME = "mibici-equipo41-p1-demanda"
CHAMPION_ALIAS = "champion"

def train_p1_model(original_data: pd.DataFrame, data_version: str, date_range: dict, model: str = 'all', params: dict | None = None):
    validate_model_and_params(model, params)
    
    dataset = create_p1_dataset(original_data)
    splits = get_train_test_split(dataset)  # (X_train, y_train, X_test, y_test)

    if model == 'all':
        if RUN_GRID:
            run_model_grid_search(data_version, date_range, splits)
        else:
            run_model_comparison(data_version, date_range, splits)

    else:
        run_model(data_version, date_range, model, params, splits)

def run_model(data_version, date_range, model, params, splits):
    models = get_models()

    model_pipeline = models[model]["pipeline"](params)
    with mlflow.start_run(run_name=f"p1_model_{model}"):
        mlflow.set_tags({"data_version": data_version, "date_range": date_range })
        train_and_test_model(model, params, model_pipeline, splits)


def validate_model_and_params(model: str, params: dict | None):
    params = params or {}
    if model != "all" and model not in P1_MODELS:
        raise ValueError(f"model debe ser {P1_MODELS} o 'all' (recibido: {model})")
    if model == "all" and params:
        raise ValueError("params solo aplica a un modelo: usa -P model=ridge|random_forest|hist_gb")

def get_models():
    ordinal = lambda: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    return  {
        "ridge": {
            "pipeline": lambda p: make_pipeline(OneHotEncoder(handle_unknown="ignore"), Ridge(random_state=SEED, **p)),
            "grid": {'alpha': [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0], 'fit_intercept': [True, False], 'solver': ['auto', 'cholesky', 'sag', 'saga']},
            "base": {'alpha': 100.0,
                    'fit_intercept': False,
                    'solver': 'sag'
                    }
        },
        "random_forest": {
            "pipeline": lambda p: make_pipeline(ordinal(), RandomForestRegressor(random_state=SEED, **p)),
            "grid": {"min_samples_leaf": [5, 20, 50], "max_features": [0.5, 1.0], "max_samples": [0.3, 0.6], "n_jobs": [-1]},
            "base": {"max_features": 0.5, "max_samples": 0.3, "min_samples_leaf": 50, "n_jobs": -1 }
        },
        "hist_gb": {
            "pipeline": lambda p: make_pipeline(ordinal(), HistGradientBoostingRegressor(random_state=SEED, **p)),
            "grid": {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [31, 63], "min_samples_leaf": [20, 100],
                     "loss": ["squared_error", "poisson"]},
            "base": {"learning_rate": 0.05, "loss": "poisson", "max_leaf_nodes":31, "min_samples_leaf": 100}
        },
    }
    
def run_model_grid_search(data_version, date_range, splits):
    models = get_models()
    with mlflow.start_run(run_name="p1_grid_model_comparison"):
        mlflow.set_tags({"data_version": data_version, "date_range": date_range})
        for i, (name, spec) in enumerate(models.items(), start=1):
            logger.info(f"P1 [{i}/{len(models)}] {name}")

            with mlflow.start_run(run_name=name, nested=True):
                tune_model(name, spec, splits)

def run_model_comparison(data_version, date_range, splits):
    models = get_models()
    results = {}
    with mlflow.start_run(run_name="p1_model_comparison"):
        mlflow.set_tags({"data_version": data_version, "date_range": date_range})
        for i, (name, spec) in enumerate(models.items(), start=1):
            logger.info(f"P1 [{i}/{len(models)}] {name}")
            params = spec["base"]
            model_pipeline = spec["pipeline"](spec["base"])
            with mlflow.start_run(run_name=name, nested=True) as child:
                mlflow.set_tag("modelo", name)
                test_rmse = train_and_test_model(name, params, model_pipeline, splits)
                results[name] = (child.info.run_id, model_pipeline, test_rmse)

        best = min(results, key=lambda n: results[n][2])
        mlflow.log_metrics({f"{n}_test_rmse": r[2] for n, r in results.items()})
        mlflow.set_tag("best_model", best)
        logger.info(f"P1: mejor modelo = {best} (test_rmse={results[best][2]:.3f})")

        best_run_id, best_model, _ = results[best]
        register_champion(best, best_run_id, best_model, splits)


def register_champion(name, run_id, model, splits):
    _, _, X_test, _ = splits
    with mlflow.start_run(run_id=run_id, nested=True):
        info = mlflow.sklearn.log_model(
            sk_model=model,
            name="p1_model",
            input_example=X_test.head(5),
            registered_model_name=REGISTERED_MODEL_NAME,
            skops_trusted_types=['sklearn.tree._tree.Tree']
        )
    version = info.registered_model_version
    MlflowClient().set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version)
    mlflow.set_tags({"champion_model": name, "champion_version": version})
    logger.success(
        f"P1: {name} registrado como {REGISTERED_MODEL_NAME} v{version} @{CHAMPION_ALIAS}"
    )

        
def tune_model(name, spec, splits):
    mlflow.set_tags({"modelo": name, "stage_of_work": "candidate"})
    pipeline = spec["pipeline"]({})

    step = pipeline.steps[-1][0]
    grid = {f"{step}__{k}": v for k, v in spec["grid"].items()}

    search = GridSearchCV(
        pipeline,
        grid,
        cv=TimeSeriesSplit(n_splits=4),
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    )

    X_train, y_train, _, _ = splits
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    mlflow.log_params({k.split("__", 1)[1]: v for k, v in search.best_params_.items()})
    mlflow.log_metric("best_cv_rmse", -search.best_score_)
    eval_model(best_model, splits)

def train_and_test_model(name, params, model, splits):
    logger.info(f"P1 {name}: entrenando con {params}")
    mlflow.log_params(params)

    X_train, y_train, _, _ = splits
    model.fit(X_train, y_train)

    test_rmse = eval_model(model, splits)

    logger.success(f"P1 {name}: terminado")
    return test_rmse


def eval_model(model, splits):
    X_train, y_train, X_test, y_test = splits
    train_predictions = np.maximum(0, model.predict(X_train))
    test_predictions = np.maximum(0, model.predict(X_test))

    log_metrics(y_train, train_predictions, 'train')
    log_metrics(y_test, test_predictions, 'test')
    plot_prediction_vs_real_data(y_test, test_predictions)
    return float(np.sqrt(np.mean((y_test.to_numpy() - test_predictions) ** 2)))