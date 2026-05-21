# Churn Prediction Platform — EFREI M1 Data Engineering

Plateforme intelligente de rétention client : prédiction du churn, comparaison multi-modèles, dashboard décisionnel.

## Structure

```
├── data/
│   ├── raw/            ← Placer customer_churn.csv ici
│   └── processed/      ← Données transformées (généré automatiquement)
├── models/             ← Modèles entraînés (.pkl)
├── notebooks/
│   ├── 01_EDA.ipynb    ← Analyse exploratoire
│   └── 02_modeling.ipynb ← Entraînement, évaluation, SHAP
├── src/
│   ├── preprocessing.py  ← Pipeline de préparation
│   ├── modeling.py       ← 4 modèles ML/DL
│   └── evaluation.py     ← Métriques et visualisations
├── dashboard/
│   └── app.py          ← Dashboard Streamlit
├── train.py            ← Script d'entraînement principal
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Dataset

Télécharger `customer_churn.csv` depuis [Kaggle](https://www.kaggle.com/datasets/miadul/customer-churn-prediction-business-dataset) et le placer dans `data/raw/`.

## Utilisation

### 1. Entraîner les modèles

```bash
python train.py
```

### 2. Lancer le dashboard

```bash
streamlit run dashboard/app.py
```

### 3. Explorer les notebooks

```bash
jupyter notebook notebooks/
```

## Modèles comparés

| Modèle | Type |
|--------|------|
| Régression Logistique | Baseline ML |
| Random Forest | ML — Ensemble |
| Gradient Boosting | ML — Boosting |
| MLP (MLPClassifier) | Deep Learning |

## Métriques d'évaluation

Accuracy, Precision, Recall, **F1-score**, **ROC-AUC** — avec focus sur F1 et Recall vu le déséquilibre des classes.
