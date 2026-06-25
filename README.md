# Churn Prediction Platform — EFREI M1 Data Engineering 2025-26

Plateforme intelligente de rétention client : prédiction du churn, comparaison multi-modèles, API REST, dashboard décisionnel.

## Résultats clés

| Modèle | Seuil | F1 | ROC-AUC | Recall |
|---|---|---|---|---|
| **Gradient Boosting ★** | 0.61 | **0.382** | **0.799** | 60.8% |
| Random Forest | 0.15 | 0.389 | 0.798 | 73.0% |
| MLP | 0.65 | 0.315 | 0.720 | 43.6% |
| Logistic Regression | 0.67 | 0.317 | 0.724 | 38.7% |

**Modèle retenu : Gradient Boosting** — meilleur compromis généralisation / F1 (gap train/test = 0.144 vs 0.202 pour RF).

Dataset : 10 000 clients | 10,2% churn | 29 features (18 numériques + 11 catégorielles).

---

## Structure du projet

```
├── data/
│   └── raw/                    ← Placer customer_churn.csv ici
├── models/                     ← Modèles entraînés (.pkl)
├── notebooks/
│   ├── 01_EDA.ipynb            ← Analyse exploratoire (distributions, corrélations)
│   ├── 02_modeling.ipynb       ← Entraînement, évaluation, feature importance, biais/variance
│   └── 03_class_imbalance.ipynb ← Gestion du déséquilibre (SMOTE, class_weight, K-Fold)
├── src/
│   ├── preprocessing.py        ← Pipeline sklearn (StandardScaler + OneHotEncoder)
│   ├── modeling.py             ← 4 modèles ML
│   └── evaluation.py           ← Métriques, courbes ROC/PR, matrices de confusion
├── api/
│   └── main.py                 ← API REST FastAPI
├── dashboard/
│   └── app.py                  ← Dashboard Streamlit
├── train.py                    ← Script d'entraînement principal
└── requirements.txt
```

---

## Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

Placer `customer_churn.csv` dans `data/raw/`.

---

## Utilisation

### 1. Entraîner les modèles

```bash
python train.py
```

Sauvegarde les 4 modèles dans `models/` et affiche le tableau comparatif.

### 2. Lancer le dashboard Streamlit

```bash
streamlit run dashboard/app.py
```

Accessible sur `http://localhost:8501`.

### 3. Lancer l'API REST

```bash
uvicorn api.main:app --reload --port 8000
```

Documentation interactive : `http://localhost:8000/docs`

### 4. Explorer les notebooks

```bash
jupyter notebook notebooks/
```

---

## API — Endpoints

| Méthode | Route | Description |
|---|---|---|
| GET | `/health` | Statut du service et modèles chargés |
| POST | `/predict` | Prédiction churn pour un client |
| POST | `/clients-at-risk` | Batch — liste des clients à risque parmi N clients |
| GET | `/model-info` | Caractéristiques des modèles disponibles |

### Exemple `/predict`

```bash
curl -X POST "http://localhost:8000/predict?model_name=gradient_boosting" \
  -H "Content-Type: application/json" \
  -d '{"age": 47, "tenure_months": 1, "monthly_fee": 20.0, "payment_failures": 0,
       "support_tickets": 0, "avg_session_time": 8.0, "monthly_logins": 23,
       "nps_score": -20, "csat_score": 3, "weekly_active_days": 6.0,
       "features_used": 3.0, "usage_growth_rate": 0.01, "last_login_days_ago": 45.0,
       "avg_resolution_time": 14.0, "escalations": 0, "email_open_rate": 0.44,
       "marketing_click_rate": 0.43, "referral_count": 3,
       "gender": "Female", "contract_type": "Monthly", "customer_segment": "Individual",
       "signup_channel": "Web", "payment_method": "Card", "discount_applied": "No",
       "price_increase_last_3m": "No", "complaint_type": "Service",
       "survey_response": "Neutral", "country": "Canada", "city": "New York"}'
```

Réponse attendue : `churn_probability: 0.896`, `risk_level: "ELEVE"`.

---

## Features utilisées (29)

**Exclues du dataset original (32 colonnes) :**
- `customer_id` — identifiant non prédictif
- `churn` — variable cible
- `total_revenue` — redondance mathématique exacte (= `monthly_fee × tenure_months`)

**Numériques (18) :** age, tenure_months, monthly_fee, payment_failures, support_tickets, avg_session_time, monthly_logins, nps_score, csat_score, weekly_active_days, features_used, usage_growth_rate, last_login_days_ago, avg_resolution_time, escalations, email_open_rate, marketing_click_rate, referral_count

**Catégorielles (11) :** gender, contract_type, customer_segment, signup_channel, payment_method, discount_applied, price_increase_last_3m, complaint_type, survey_response, country, city

---

## Gestion du déséquilibre des classes

- 10,2% de churners → ratio 8,8:1
- `class_weight="balanced"` pour Logistic Regression et Random Forest
- `sample_weight` calculé via `compute_sample_weight("balanced")` pour Gradient Boosting et MLP
- Seuil de décision optimisé par maximisation du F1-score sur le jeu de test

## Top 5 variables les plus importantes (Gradient Boosting)

1. `tenure_months` — 19,0%
2. `csat_score` — 17,8%
3. `monthly_logins` — 15,4%
4. `payment_failures` — 13,8%
5. `last_login_days_ago` — 5,6%
