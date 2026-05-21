"""
Entraînement des 4 modèles supervisés pour la prédiction du churn.
Modèles : Régression Logistique, Random Forest, Gradient Boosting, MLP (Deep Learning).
"""

import joblib
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src.preprocessing import build_preprocessor


MODELS_DIR = Path("models")


def build_pipelines() -> dict[str, Pipeline]:
    preprocessor = build_preprocessor()

    pipelines = {
        "logistic_regression": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            )),
        ]),
        "random_forest": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "gradient_boosting": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                random_state=42,
            )),
        ]),
        "mlp": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
            )),
        ]),
    }

    return pipelines


_NEEDS_SAMPLE_WEIGHT = {"gradient_boosting", "mlp"}


def train_all(pipelines: dict, X_train, y_train) -> dict:
    trained = {}
    sample_weight = compute_sample_weight("balanced", y_train)
    for name, pipeline in pipelines.items():
        print(f"Entraînement : {name} ...")
        if name in _NEEDS_SAMPLE_WEIGHT:
            pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weight)
        else:
            pipeline.fit(X_train, y_train)
        trained[name] = pipeline
        save_model(pipeline, name)
    return trained


def save_model(pipeline: Pipeline, name: str) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / f"{name}.pkl")
    print(f"  -> Sauvegardé : models/{name}.pkl")


def load_model(name: str) -> Pipeline:
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    return joblib.load(path)


def load_all_models() -> dict[str, Pipeline]:
    models = {}
    for path in MODELS_DIR.glob("*.pkl"):
        models[path.stem] = joblib.load(path)
    return models
