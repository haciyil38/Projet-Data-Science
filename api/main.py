"""
API REST — Système de Prédiction du Churn Client
FastAPI + uvicorn

Lancement : uvicorn api.main:app --reload --port 8000
Documentation : http://localhost:8000/docs
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from src.modeling import load_all_models

# ── Initialisation ────────────────────────────────────────────────────────────

app = FastAPI(
    title="Churn Prediction API",
    description="API de prédiction du churn client — EFREI M1 Data Engineering 2025-26",
    version="1.0.0",
)

models: dict = {}
MODEL_LOAD_TIME: str = ""

@app.on_event("startup")
def startup():
    global models, MODEL_LOAD_TIME
    models = load_all_models()
    MODEL_LOAD_TIME = datetime.now().isoformat()


# ── Schémas Pydantic ──────────────────────────────────────────────────────────

class ClientFeatures(BaseModel):
    # Numériques
    age: int = Field(..., ge=18, le=100, example=47)
    tenure_months: int = Field(..., ge=0, le=240, example=1)
    monthly_fee: float = Field(..., ge=0, le=10000, example=20.0)
    payment_failures: int = Field(..., ge=0, le=100, example=0)
    support_tickets: int = Field(..., ge=0, le=200, example=0)
    avg_session_time: float = Field(..., ge=0, example=8.0)
    monthly_logins: int = Field(..., ge=0, example=23)
    nps_score: int = Field(..., ge=-100, le=100, example=-20)
    csat_score: int = Field(..., ge=1, le=5, example=3)
    weekly_active_days: float = Field(..., ge=0, le=7, example=6.0)
    features_used: float = Field(..., ge=0, example=3.0)
    usage_growth_rate: float = Field(..., example=0.01)
    last_login_days_ago: float = Field(..., ge=0, example=45.0)
    avg_resolution_time: float = Field(..., ge=0, example=14.0)
    escalations: int = Field(..., ge=0, example=0)
    email_open_rate: float = Field(..., ge=0, le=1, example=0.44)
    marketing_click_rate: float = Field(..., ge=0, le=1, example=0.43)
    referral_count: int = Field(..., ge=0, example=3)
    # Catégorielles
    gender: str = Field(..., example="Female")
    contract_type: str = Field(..., example="Monthly")
    customer_segment: str = Field(..., example="Individual")
    signup_channel: str = Field(..., example="Web")
    payment_method: str = Field(..., example="Card")
    discount_applied: str = Field(..., example="No")
    price_increase_last_3m: str = Field(..., example="No")
    complaint_type: str = Field(..., example="Service")
    survey_response: str = Field(..., example="Neutral")
    country: str = Field(..., example="Canada")
    city: str = Field(..., example="New York")

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v):
        allowed = {"Male", "Female"}
        if v not in allowed:
            raise ValueError(f"gender doit être parmi {allowed}")
        return v

    @field_validator("contract_type")
    @classmethod
    def validate_contract(cls, v):
        allowed = {"Monthly", "Yearly", "Quarterly"}
        if v not in allowed:
            raise ValueError(f"contract_type doit être parmi {allowed}")
        return v

    @field_validator("customer_segment")
    @classmethod
    def validate_segment(cls, v):
        allowed = {"SME", "Individual", "Enterprise"}
        if v not in allowed:
            raise ValueError(f"customer_segment doit être parmi {allowed}")
        return v

    @field_validator("survey_response")
    @classmethod
    def validate_survey(cls, v):
        allowed = {"Satisfied", "Neutral", "Unsatisfied"}
        if v not in allowed:
            raise ValueError(f"survey_response doit être parmi {allowed}")
        return v


class PredictionResponse(BaseModel):
    model_used: str
    churn_probability: float
    churn_prediction: int
    risk_level: str
    revenue_at_risk_annual: float
    threshold_used: float


class ClientAtRisk(BaseModel):
    index: int
    churn_probability: float
    risk_level: str
    revenue_at_risk_annual: float
    monthly_fee: float
    tenure_months: int


class ClientsAtRiskResponse(BaseModel):
    model_used: str
    threshold_used: float
    total_clients: int
    clients_at_risk: int
    total_revenue_at_risk: float
    results: list[ClientAtRisk]


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
    model_count: int
    loaded_at: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    recommended: bool
    description: str


# ── Utilitaires ───────────────────────────────────────────────────────────────

THRESHOLDS = {
    "gradient_boosting": 0.53,
    "random_forest": 0.14,
    "mlp": 0.60,
    "logistic_regression": 0.64,
}

MODEL_DESCRIPTIONS = {
    "gradient_boosting": "Gradient Boosting — modèle recommandé (F1=0.386, AUC=0.799)",
    "random_forest": "Random Forest — meilleur recall (73.0%)",
    "mlp": "Multi-Layer Perceptron — Deep Learning (F1=0.315)",
    "logistic_regression": "Régression Logistique — baseline interprétable (F1=0.317)",
}

def get_risk_level(proba: float) -> str:
    if proba > 0.6:
        return "ELEVE"
    elif proba > 0.3:
        return "MODERE"
    return "FAIBLE"

def features_to_dataframe(features: ClientFeatures) -> pd.DataFrame:
    return pd.DataFrame([features.model_dump()])


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    """Vérifie que le service est actif et que les modèles sont chargés."""
    if not models:
        raise HTTPException(status_code=503, detail="Aucun modèle chargé.")
    return HealthResponse(
        status="ok",
        models_loaded=list(models.keys()),
        model_count=len(models),
        loaded_at=MODEL_LOAD_TIME,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prédiction"])
def predict(features: ClientFeatures, model_name: str = "gradient_boosting"):
    """
    Prédit la probabilité de churn d'un client.

    - **model_name** : modèle à utiliser (gradient_boosting recommandé)
    - Retourne la probabilité, la prédiction binaire, le niveau de risque et le revenu à risque annuel.
    """
    if model_name not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Modèle '{model_name}' introuvable. Modèles disponibles : {list(models.keys())}",
        )

    model = models[model_name]
    X = features_to_dataframe(features)

    try:
        proba = float(model.predict_proba(X)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")

    threshold = THRESHOLDS.get(model_name, 0.5)
    prediction = int(proba >= threshold)
    risk_level = get_risk_level(proba)
    revenue_at_risk = round(features.monthly_fee * 12 * proba, 2)

    return PredictionResponse(
        model_used=model_name,
        churn_probability=round(proba, 4),
        churn_prediction=prediction,
        risk_level=risk_level,
        revenue_at_risk_annual=revenue_at_risk,
        threshold_used=threshold,
    )


@app.post("/clients-at-risk", response_model=ClientsAtRiskResponse, tags=["Prédiction"])
def clients_at_risk(
    clients: list[ClientFeatures],
    model_name: str = "gradient_boosting",
    risk_threshold: float = 0.3,
):
    """
    Analyse un batch de clients et retourne ceux dont la probabilité de churn dépasse le seuil.

    - **model_name** : modèle à utiliser (gradient_boosting recommandé)
    - **risk_threshold** : seuil de probabilité pour considérer un client à risque (défaut 0.3 = MODÉRÉ+)
    - Retourne les clients triés par probabilité décroissante avec le revenu total à risque.
    """
    if model_name not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Modèle '{model_name}' introuvable. Modèles disponibles : {list(models.keys())}",
        )
    if not clients:
        raise HTTPException(status_code=422, detail="La liste de clients est vide.")

    model = models[model_name]
    X = pd.DataFrame([c.model_dump() for c in clients])

    try:
        probas = model.predict_proba(X)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")

    results = []
    for i, (client, proba) in enumerate(zip(clients, probas)):
        if proba >= risk_threshold:
            results.append(ClientAtRisk(
                index=i,
                churn_probability=round(float(proba), 4),
                risk_level=get_risk_level(float(proba)),
                revenue_at_risk_annual=round(client.monthly_fee * 12 * float(proba), 2),
                monthly_fee=client.monthly_fee,
                tenure_months=client.tenure_months,
            ))

    results.sort(key=lambda x: x.churn_probability, reverse=True)
    total_revenue_at_risk = round(sum(r.revenue_at_risk_annual for r in results), 2)

    return ClientsAtRiskResponse(
        model_used=model_name,
        threshold_used=risk_threshold,
        total_clients=len(clients),
        clients_at_risk=len(results),
        total_revenue_at_risk=total_revenue_at_risk,
        results=results,
    )


@app.get("/model-info", response_model=list[ModelInfoResponse], tags=["Modèles"])
def model_info():
    """Retourne la liste des modèles disponibles avec leurs caractéristiques."""
    return [
        ModelInfoResponse(
            model_name=name,
            model_type=type(model.named_steps["classifier"]).__name__,
            recommended=(name == "gradient_boosting"),
            description=MODEL_DESCRIPTIONS.get(name, ""),
        )
        for name, model in models.items()
    ]
