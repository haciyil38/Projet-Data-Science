"""
Script d'entraînement principal.
Usage : python train.py
"""

from pathlib import Path
from src.preprocessing import load_data, split_data
from src.modeling import build_pipelines, train_all
from src.evaluation import compare_models

DATA_PATH = "data/raw/customer_churn.csv"


def main():
    if not Path(DATA_PATH).exists():
        print(f"ERREUR : dataset introuvable à {DATA_PATH}")
        print("Téléchargez customer_churn.csv depuis Kaggle et placez-le dans data/raw/")
        return

    print("Chargement des données...")
    df = load_data(DATA_PATH)
    print(f"  {len(df)} lignes, {df.shape[1]} colonnes")
    print(f"  Churn rate : {df['churn'].mean():.1%}")

    X_train, X_test, y_train, y_test = split_data(df)
    print(f"  Train : {len(X_train)} | Test : {len(X_test)}")

    print("\nConstruction des pipelines...")
    pipelines = build_pipelines()

    print("\nEntraînement des modèles...")
    trained_models = train_all(pipelines, X_train, y_train)

    print("\nÉvaluation comparative :")
    metrics = compare_models(trained_models, X_test, y_test)
    print(metrics.to_string())

    print("\nEntraînement terminé. Modèles sauvegardés dans models/")


if __name__ == "__main__":
    main()
