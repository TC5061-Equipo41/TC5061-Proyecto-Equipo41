

import argparse
import os

from dotenv import load_dotenv
import mlflow
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mibici.modeling.p1.evaluate import log_metrics
from mibici.p1.dataset import create_p1_dataset, get_train_test_split

load_dotenv()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--alpha", type=float, default=0.1, choices=[0.1,1.0,10.0,100.0])
    parser.add_argument("--fit_intercept", type=bool, default=True)
    parser.add_argument("--run_name", type=str, default="svc")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--consider_month", type=bool, default=False)
    args = parser.parse_args(argv)

    return args

def train_model():
    args = parse_args()
    dataset = create_p1_dataset(args.consider_month)
    X_train, y_train, X_test, y_test = get_train_test_split(dataset, args.consider_month)
    
    mlflow_uri = os.getenv("MLFLOW_URI")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("mibici-p1-training")

    params = {'alpha':args.alpha, 'fit_intercept':args.fit_intercept, 'solver':"lsqr", 'tol':1e-4, 'max_iter':10_000}
    model = make_pipeline(
        StandardScaler(),
        Ridge(**params),
    )

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(params)
        model.fit(X_train, y_train)
        train_predictions = np.maximum(0, model.predict(X_train))
        test_predictions = np.maximum(0, model.predict(X_test))

        log_metrics(y_train, train_predictions, 'train')
        log_metrics(y_test, test_predictions, 'test')

        mlflow.sklearn.log_model(
            sk_model=model,
            name=args.model_name,
            input_example=X_test.head(5),
        )


if __name__ == "__main__":
    train_model()
