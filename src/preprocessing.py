"""
Pipeline de préparation des données pour la prédiction du churn.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


NUMERIC_FEATURES = [
    "age", "tenure_months", "monthly_fee", "total_revenue",
    "payment_failures", "support_tickets", "avg_session_time",
    "monthly_logins", "nps_score",
]

CATEGORICAL_FEATURES = ["gender", "contract_type"]

TARGET = "churn"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    return preprocessor


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # important pour les classes déséquilibrées
    )
    return X_train, X_test, y_train, y_test


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    num_names = NUMERIC_FEATURES
    cat_names = list(
        preprocessor.named_transformers_["cat"]
        .named_steps["encoder"]
        .get_feature_names_out(CATEGORICAL_FEATURES)
    )
    return num_names + cat_names
