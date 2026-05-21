"""
Dashboard Streamlit — Plateforme de rétention client.
Orienté utilisateur métier (CRM / Marketing).
"""

import sys
from pathlib import Path

# Permet d'importer src/ depuis le dossier racine
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests

from src.modeling import load_all_models
from src.preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, load_data, split_data
from src.evaluation import compare_models, plot_confusion_matrix, plot_roc_curves, get_feature_importance

# ─── Configuration page ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction Platform",
    page_icon="📊",
    layout="wide",
)

# ─── Chargement des données et modèles ────────────────────────────────────────
@st.cache_data
def load_dataset():
    path = Path("data/raw/customer_churn.csv")
    if not path.exists():
        st.error("Dataset introuvable : placez customer_churn.csv dans data/raw/")
        st.stop()
    return load_data(str(path))


@st.cache_resource
def load_models():
    models = load_all_models()
    if not models:
        st.error("Aucun modèle trouvé. Lancez d'abord train.py pour entraîner les modèles.")
        st.stop()
    return models


df = load_dataset()
models = load_models()
X_train, X_test, y_train, y_test = split_data(df)

# ─── Sidebar navigation ───────────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Section",
    ["Vue d'ensemble", "Analyse Clients", "Comparaison Modèles", "Simulation Client"],
)

# ─── Page : Vue d'ensemble ────────────────────────────────────────────────────
if page == "Vue d'ensemble":
    st.title("Plateforme de Rétention Client")
    st.markdown("Tableau de bord décisionnel pour anticiper le churn et estimer le revenu à risque.")

    total_clients = len(df)
    churned = df[TARGET].sum()
    churn_rate = churned / total_clients * 100
    revenue_at_risk = df.loc[df[TARGET] == 1, "total_revenue"].sum()
    avg_monthly = df.loc[df[TARGET] == 1, "monthly_fee"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Clients total", f"{total_clients:,}")
    col2.metric("Clients à risque", f"{churned:,}", f"{churn_rate:.1f}%")
    col3.metric("Revenu à risque (€)", f"{revenue_at_risk:,.0f}")
    col4.metric("Charge mensuelle moy. (churners)", f"{avg_monthly:.2f} €")

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.pie(
            names=["Fidèles", "Churners"],
            values=[total_clients - churned, churned],
            title="Répartition Churn / Fidèles",
            color_discrete_sequence=["#2ecc71", "#e74c3c"],
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        rev_by_contract = (
            df.groupby(["contract_type", TARGET])["monthly_fee"]
            .sum()
            .reset_index()
        )
        rev_by_contract[TARGET] = rev_by_contract[TARGET].map({0: "Fidèle", 1: "Churner"})
        fig2 = px.bar(
            rev_by_contract,
            x="contract_type",
            y="monthly_fee",
            color=TARGET,
            barmode="group",
            title="Charges mensuelles par type de contrat",
            color_discrete_map={"Fidèle": "#2ecc71", "Churner": "#e74c3c"},
        )
        st.plotly_chart(fig2, use_container_width=True)

# ─── Page : Analyse Clients ───────────────────────────────────────────────────
elif page == "Analyse Clients":
    st.title("Analyse des Profils Clients")

    feature = st.selectbox("Variable à analyser", NUMERIC_FEATURES)
    fig = px.histogram(
        df, x=feature, color=TARGET,
        barmode="overlay",
        labels={TARGET: "Churn"},
        color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
        title=f"Distribution de {feature} selon le churn",
        opacity=0.7,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Matrice de corrélation")
    corr = df[NUMERIC_FEATURES + [TARGET]].corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Corrélations entre variables numériques",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# ─── Page : Comparaison Modèles ───────────────────────────────────────────────
elif page == "Comparaison Modèles":
    st.title("Comparaison des Modèles")

    with st.spinner("Évaluation en cours..."):
        metrics_df = compare_models(models, X_test, y_test)

    st.dataframe(
        metrics_df.style.highlight_max(axis=0, color="#d4efdf").format("{:.4f}"),
        use_container_width=True,
    )

    fig = px.bar(
        metrics_df.reset_index(),
        x="model",
        y=["accuracy", "precision", "recall", "f1", "roc_auc"],
        barmode="group",
        title="Métriques par modèle",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Courbes ROC")
    roc_fig = plot_roc_curves(models, X_test, y_test)
    st.pyplot(roc_fig)

    st.subheader("Matrices de Confusion")
    cm_cols = st.columns(2)
    for idx, (name, model) in enumerate(models.items()):
        with cm_cols[idx % 2]:
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
            plot_confusion_matrix(model, X_test, y_test, model_name=name, ax=ax_cm)
            st.pyplot(fig_cm)
            plt.close(fig_cm)

    # Feature importance pour les modèles arbres
    tree_models = {k: v for k, v in models.items() if k in ("random_forest", "gradient_boosting")}
    if tree_models:
        st.subheader("Importance des Variables")
        selected = st.selectbox("Modèle", list(tree_models.keys()))
        model = tree_models[selected]
        preprocessor = model.named_steps["preprocessor"]
        preprocessor.fit(X_train, y_train)  # pour récupérer les noms
        from src.preprocessing import get_feature_names
        feature_names = get_feature_names(preprocessor)
        importance = get_feature_importance(model, feature_names)
        fig_imp = px.bar(
            importance.head(15).reset_index(),
            x="index", y=0,
            labels={"index": "Variable", "0": "Importance"},
            title=f"Top 15 variables — {selected}",
        )
        st.plotly_chart(fig_imp, use_container_width=True)

# ─── Page : Simulation Client ─────────────────────────────────────────────────
elif page == "Simulation Client":
    st.title("Simulation — Probabilité de Churn d'un Client")
    st.markdown("Modifiez les paramètres ci-dessous pour estimer le risque d'un client spécifique.")

    model_keys = list(models.keys())
    default_idx = model_keys.index("gradient_boosting") if "gradient_boosting" in model_keys else 0
    model_name = st.selectbox("Modèle à utiliser", model_keys, index=default_idx,
                              help="Le Gradient Boosting est le modèle recommandé (meilleur F1 et ROC-AUC).")
    model = models[model_name]

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Âge", 18, 80, 35)
        tenure_months = st.slider("Ancienneté (mois)", 0, 72, 12)
        monthly_fee = st.number_input("Charges mensuelles (€)", 10.0, 500.0, 80.0)
    with col2:
        total_revenue = st.number_input("Revenu total (€)", 0.0, 50000.0, 960.0)
        payment_failures = st.slider("Échecs de paiement", 0, 20, 0)
        support_tickets = st.slider("Tickets support", 0, 30, 2)
    with col3:
        avg_session_time = st.slider("Durée session (min)", 0, 120, 30)
        monthly_logins = st.slider("Connexions/mois", 0, 100, 15)
        nps_score = st.slider("NPS Score", -100, 100, 20)
        gender = st.selectbox("Genre", ["Male", "Female"])
        contract_type = st.selectbox("Type de contrat", ["Monthly", "Yearly", "Quarterly"])

    input_data = pd.DataFrame([{
        "age": age, "tenure_months": tenure_months, "monthly_fee": monthly_fee,
        "total_revenue": total_revenue, "payment_failures": payment_failures,
        "support_tickets": support_tickets, "avg_session_time": avg_session_time,
        "monthly_logins": monthly_logins, "nps_score": nps_score,
        "gender": gender, "contract_type": contract_type,
    }])

    API_URL = "http://localhost:8000"
    api_available = False
    try:
        r = requests.get(f"{API_URL}/health", timeout=1)
        api_available = r.status_code == 200
    except Exception:
        pass

    if api_available:
        st.info("Connecté à l'API REST (http://localhost:8000)", icon="🔗")
    else:
        st.warning("API non démarrée — prédiction locale. Lancez : `uvicorn api.main:app --port 8000`", icon="⚠️")

    if st.button("Calculer le risque de churn", type="primary"):
        if api_available:
            payload = {
                "age": age, "tenure_months": tenure_months, "monthly_fee": monthly_fee,
                "total_revenue": total_revenue, "payment_failures": payment_failures,
                "support_tickets": support_tickets, "avg_session_time": avg_session_time,
                "monthly_logins": monthly_logins, "nps_score": nps_score,
                "gender": gender, "contract_type": contract_type,
            }
            resp = requests.post(f"{API_URL}/predict?model_name={model_name}", json=payload, timeout=5)
            if resp.status_code == 200:
                result = resp.json()
                proba = result["churn_probability"]
                revenue_at_risk = result["revenue_at_risk_annual"]
            else:
                st.error(f"Erreur API : {resp.json().get('detail', 'Erreur inconnue')}")
                st.stop()
        else:
            proba = model.predict_proba(input_data)[0, 1]
            revenue_at_risk = monthly_fee * 12 * proba

        risk_label = "ÉLEVÉ" if proba > 0.6 else "MODÉRÉ" if proba > 0.3 else "FAIBLE"
        risk_color = "#e74c3c" if proba > 0.6 else "#f39c12" if proba > 0.3 else "#2ecc71"
        source = "via API REST" if api_available else "via modèle local"

        st.markdown(f"""
        <div style='padding:20px; border-radius:10px; background:{risk_color}20; border-left: 5px solid {risk_color}'>
        <h2 style='color:{risk_color}'>Risque {risk_label}</h2>
        <h3>Probabilité de churn : <strong>{proba:.1%}</strong></h3>
        <p>Revenu à risque estimé : <strong>{revenue_at_risk:.0f} €/an</strong></p>
        <p style='font-size:0.8em; color:gray'>Prédiction {source}</p>
        </div>
        """, unsafe_allow_html=True)

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            title={"text": "Probabilité de Churn (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": risk_color},
                "steps": [
                    {"range": [0, 30], "color": "#d4efdf"},
                    {"range": [30, 60], "color": "#fdebd0"},
                    {"range": [60, 100], "color": "#fadbd8"},
                ],
            },
        ))
        st.plotly_chart(gauge, use_container_width=True)
