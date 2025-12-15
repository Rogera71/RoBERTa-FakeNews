import streamlit as st
import pandas as pd
import os
import numpy as np

#st.set_page_config(page_title="FakeNews Detector", page_icon="📰", layout="wide")
st.set_page_config(
    page_title="FakeNews Detector",
    page_icon="📰",
    layout="wide",
)
#if "page" not in st.session_state:
    #st.session_state["page"] = "dashboard"
## Style Dashboard 
st.markdown(
    """
    <style>
    /* Fond général */
    .stApp {
        background-color: #757575; /* fond blanc */
        max-width: 1200px; 
        margin-left: auto;     /* centre horizontalement */
        margin-right: auto;
        overflow-x: hidden;    /* évite la barre horizontale */
    }
    .block-container {
        padding-top: 1.5rem;
    }
    /* Cartes KPI */
    .kpi-card {
        background-color: #ffffff;
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.20);
    }
    .kpi-title {
        font-size: 0.9rem;
        color: black;
        margin-bottom: 0.3rem;
        font-weight: 700;
    }
    .kpi-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #12355b; /* bleu principal */
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #777777;
    }

    /* Boutons */
    .stButton > button {
        background-color: #12355b;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 0.8rem;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #1d4f8f;  /* couleur de survol */
        color: white;
    }

    /* Carte pipeline */
    .pipeline-card {
        background-color: #ffffff;
        padding: 0.9rem 1.1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.20);
        margin-top: 0.8rem;
    }

    /* Blocs du pipeline : même taille */
    .pipeline-row {
        display: flex;
        justify-content: center;
        gap: 0.6rem;
        margin-bottom: 0.7rem;
    }
    .pipeline-block {
        min-width: 170px;
        max-width: 170px;
        text-align: center;
        padding: 0.4rem 0.5rem;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #ffffff;
        box-shadow: 0 1px 4px rgba(0,0,0,0.25);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .step-raw  { background-color: #12355b; }  /* bleu */
    .step-model{ background-color: #12355b; }  /* vert */
    .step-exp  { background-color: #12355b; }  /* orange */

    .pipeline-arrow {
        align-self: center;
        font-size: 1.0rem;
        color: #1d4f8f;
    }
    </style>
    """,
    unsafe_allow_html=True
)
## Titre
st.markdown("<h1 style='color: #FAA15D'>🏠 Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #FCFCFC'>Vue d’ensemble de FakeNews Detector</p>", unsafe_allow_html=True)

ROOT_DIR = os.path.dirname(__file__)
CSV_FILE = os.path.join(ROOT_DIR, "analysis_history.csv")

total_analyses = 0
avg_sentiment = "—"
pct_real = "—"
pct_fake = "—"

if os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0:
    hist = pd.read_csv(CSV_FILE)
    if not hist.empty:
        total_analyses = len(hist)
        # % Real / Fake
        n_real = int((hist["label"] == "Real").sum())
        n_fake = int((hist["label"] == "Fake").sum())
        pct_real = f"{(n_real / total_analyses) * 100:.1f}%"
        pct_fake = f"{(n_fake / total_analyses) * 100:.1f}%"
        # score de sentiment moyen
        if "sentiment" in hist.columns:
            avg_sentiment_val = hist["sentiment"].mean()
            avg_sentiment = f"{avg_sentiment_val:.3f}"
# KPI avec valeurs
row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)
# KPI
row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">Textes analysés (session)</div>
          <div class="kpi-value">{total_analyses}</div>
          <div class="kpi-sub">Nombre de requêtes traitées dans application.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with row1_col2:
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">% Fake / Real (test)</div>
          <div class="kpi-value">Fake {pct_fake} / Real {pct_real}</div>
          <div class="kpi-sub">Répartition approximative des classes dans le jeu de test.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
st.markdown("<br>", unsafe_allow_html=True)
row2_col1, row2_col2 = st.columns(2)
with row2_col1:
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">Score de sentiment moyen</div>
          <div class="kpi-value">{avg_sentiment}</div>
          <div class="kpi-sub">Sentiment moyen des derniers textes analysés.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with row2_col2:
    st.markdown(
        """
        <div class="kpi-card">
          <div class="kpi-title">Performance du modèle</div>
          <div class="kpi-value">F1 ≈ 0.999</div>
          <div class="kpi-sub">Résultats obtenus sur le jeu de test (accuracy / F1).</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
st.markdown("<br>", unsafe_allow_html=True)
# Lecture du fichier CSV
st.markdown("<h3 style='color: #FFA22A;'>Historique des dernières analyses</h3>", unsafe_allow_html=True)

if os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0:
    hist = pd.read_csv(CSV_FILE)
    if not hist.empty:
        hist = hist.sort_values("timestamp", ascending=False)

        # petit résumé
        n_total = len(hist)
        n_fake = int((hist["label"] == "Fake").sum())
        n_real = int((hist["label"] == "Real").sum())
        st.markdown(
            f"<p style='color:#ffffff;'>Total en historique : <b>{n_total}</b> · Real : <b>{n_real}</b> · Fake : <b>{n_fake}</b></p>",
            unsafe_allow_html=True,
        )
        # sélection des colonnes + renommage pour un tableau plus clair
        df_view = hist[["timestamp", "source_page", "label", "prob_fake", "prob_real", "text_snippet"]].head(3)
        df_view = df_view.rename(columns={
            "timestamp": "Date/heure",
            "source_page": "Page",
            "label": "Classe",
            "prob_fake": "P(Fake)",
            "prob_real": "P(Real)",
            "text_snippet": "Extrait du texte",
        })
        # style du tableau avec lignes zébrées
        def zebra_rows(df):
            styles = []
            for i in range(len(df)):
                if i % 2 == 0:
                    styles.append(['background-color: white'] * df.shape[1])  # ligne paire
                else:
                    styles.append(['background-color: #BFBFBF'] * df.shape[1])  # ligne impaire
            return pd.DataFrame(styles, index=df.index, columns=df.columns)


        styled = (
            df_view.style
            .apply(zebra_rows, axis=None)
            .set_properties(**{"color": "black", "font-size": "12px"})
            .set_table_styles([
                {
                    "selector": "th.col_heading",
                    "props": [
                        ("background-color","#2e2f2f"),
                        ("color", "white"),
                        ("font-weight", "bold"),
                        ("font-size", "14px"),
                    ],
                },
                # En-tête de l’index
                {
                    "selector": "th.row_heading",
                    "props": [
                        ("background-color", "#2e2f2f"),
                        ("color", "white"),
                        ("font-weight", "bold"),
                        ("font-size", "14px"),
                    ],
                },
                {"selector": "th.col0", "props": [("min-width", "120px")]},  # Date/heure
                {"selector": "th.col1", "props": [("min-width", "120px")]},  # Page
                {"selector": "th.col2", "props": [("min-width", "70px")]},   # Classe
                {"selector": "th.col3", "props": [("min-width", "90px")]},   # P(Fake)
                {"selector": "th.col4", "props": [("min-width", "90px")]},   # P(Real)
                {"selector": "th.col5", "props": [("min-width", "300px")]},  # Extrait du texte
            ])
        )

        st.table(styled)
        #st.dataframe(df_view, use_container_width=True)
    else:
        st.info("Aucune analyse enregistrée pour le moment.")
else:
    st.info("Le fichier `analysis_history.csv` n’existe pas encore ou est vide.")

# ACTIONS RAPIDES
st.markdown("<h3 style='color: #FFA22A;'>Actions rapides</h3>", unsafe_allow_html=True)
c1, c2 = st.columns(2)

with c1:
    if st.button(" Analyser un nouveau texte", use_container_width=True):
        st.session_state["page"] = "single_text"
        st.rerun()

with c2:
    if st.button(" Analyser un fichier ", use_container_width=True):
        st.session_state["page"] = "batch"
        st.rerun()

# PIPELINE SOUS FORME DE FIGURE LINÉAIRE
st.markdown("<h3 style='color: #FFA22A;'>Pipeline simplifié</h3>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="pipeline-card">
      <!-- Ligne 1 -->
      <div class="pipeline-row">
        <div class="pipeline-block step-raw">Texte brut</div>
        <div class="pipeline-arrow">➜</div>
        <div class="pipeline-block step-raw">Prétraitement</div>
        <div class="pipeline-arrow">➜</div>
        <div class="pipeline-block step-model">RoBERTa</div>
      </div>
      <div class="pipeline-row">
      <div class="pipeline-arrow"> . </div>
      </div>
      <!-- Ligne 2 -->
      <div class="pipeline-row">
        <div class="pipeline-block step-model">Prédiction Real/Fake</div>
        <div class="pipeline-arrow">➜</div>
        <div class="pipeline-block step-exp">LIME</div>
        <div class="pipeline-arrow">➜</div>
        <div class="pipeline-block step-exp">Sentiment & Style</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <hr style='border: 0.8px solid black; margin-top: 1.5rem; margin-bottom: 0.5rem;'>

    <div style='text-align: center; color: #FCFCFC; font-size: 14px;'>
        FakeNews Detector · Projet académique · 2025
    </div>
    """,
    unsafe_allow_html=True
)
