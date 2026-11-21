import random

random.seed(42)


def choose_model(models_data: dict) -> dict:
    loaded_models = [models_data["p1_current"], models_data["p1_challenger"]]

    # Reparto A/B 50/50, no criptografía: basta un generador pseudoaleatorio.
    return random.choice(loaded_models)  # nosec B311
