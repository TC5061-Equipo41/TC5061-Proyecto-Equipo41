
import random

random.seed(42)


def choose_model(models_data: dict) -> dict:
    loaded_models = [models_data["p1_current"], models_data["p1_challenger"]]

    return random.choice(loaded_models)