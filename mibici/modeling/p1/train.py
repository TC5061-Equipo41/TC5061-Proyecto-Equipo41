


import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from mibici.modeling.p1.evaluate import log_metrics
from mibici.p1.dataset import create_p1_dataset, get_train_test_split
from mibici.p1.plots import plot_prediction_vs_real_data


def train_p1_model(original_data: pd.Dataframe, data_version: str, date_range: dict):
    dataset = create_p1_dataset(original_data)
    X_train, y_train, X_test, y_test = get_train_test_split(dataset)

    params = {'alpha': 0.1, 'fit_intercept': True, 'solver':"lsqr", 'tol':1e-4, 'max_iter':10_000}
    model = make_pipeline(
        OneHotEncoder(),
        Ridge(**params),
    )

    with mlflow.start_run(run_name="p1"):
        mlflow.set_tags({"data_version": data_version, "date_range": date_range })
        mlflow.log_params(params)
        model.fit(X_train, y_train)
        train_predictions = np.maximum(0, model.predict(X_train))
        test_predictions = np.maximum(0, model.predict(X_test))

        log_metrics(y_train, train_predictions, 'train')
        log_metrics(y_test, test_predictions, 'test')
        plot_prediction_vs_real_data(y_test, test_predictions)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="p1_model",
            input_example=X_test.head(5),
        )

