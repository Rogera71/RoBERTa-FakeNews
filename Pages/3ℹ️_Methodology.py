import streamlit as st

st.session_state["page"] = "about"

## STYLE GLOBAL (même template) 
st.markdown(
    """
    <style>
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
    .about-zone {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.8rem;
        border: 1px solid #bbbbbb;
    }
    .about-zone-title {
        font-weight: 700;
        color: #1d4f8f;
        margin-bottom: 0.2rem;
        font-size: 18px;
    }
    .about-zone-text {
        color: #1F1F1F;
        font-size: 14px;
        line-height: 1.4;
        white-space: pre-line;  /* pour respecter les sauts de ligne */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='color: #FAA15D'>ℹ️ METHODOLOGIE</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #F0F0F0'>Description du projet, du modèle et du pipeline de détection de Fake News.</p>", unsafe_allow_html=True)
st.caption("")
# Objectif
st.markdown(
    """
    <div class="about-zone">
      <div class="about-zone-title">ObjectifS</div>
      <div class="about-zone-text">
        - Détecter automatiquement si un article ou un message est Real ou Fake.
        - Fournir des informations complémentaires : sentiment, style d’écriture, explication LIME.
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Modèle & données
st.markdown(
    """
    <div class="about-zone">
      <div class="about-zone-title">Modèle & Données</div>
      <div class="about-zone-text">
        - Modèle : RoBERTa (roberta-base) fine-tuné pour une classification binaire Real / Fake.
        - Données : fichiers Fake.csv et True.csv (titre, texte, sujet, date).
        - Prétraitement : fusion, ajout du label, création de content = title + text, nettoyage.
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Pipeline
st.markdown(
    """
    <div class="about-zone">
      <div class="about-zone-title">Pipeline de traitement</div>
      <div class="about-zone-text">
        1. Prétraitement : concaténation titre + texte, nettoyage.
        2. Tokenisation : RobertaTokenizerFast, tronquage/padding à 512 tokens.
        3. Entraînement : fine-tuning de RoBERTa avec early stopping sur la F1.
        4. Inférence : prédiction Real / Fake + probabilités.
        5. Explicabilité : LIME + calcul du score de sentiment et des mesures de style.
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Limitations & pistes futures
st.markdown(
    """
    <div class="about-zone">
      <div class="about-zone-title">Limitations & pistes futures</div>
      <div class="about-zone-text">
        - Langue principalement anglaise (extension multilingue possible).
        - Ajout futur de l’analyse d’images (OCR) pour les captures d’écran.
        - Amélioration de la robustesse sur les textes très courts / très longs.
        - Mise à jour du modèle avec de nouvelles données de fake news.
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <hr style='border: 0.8px solid black; margin-top: 1.5rem; margin-bottom: 0.5rem;'>

    <div style='text-align: center; color: #F0F0F0; font-size: 14px;'>
        FakeNews Detector · Projet académique · 2025
    </div>
    """,
    unsafe_allow_html=True
)