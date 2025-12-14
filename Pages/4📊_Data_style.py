import streamlit as st
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from PIL import Image
st.session_state["page"] = "insights"

# STYLE GLOBAL
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
    .main-card {
        background-color: #BCF6F3;
        padding: 0.3rem 0.4rem;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    .section-title {
        font-weight: 700;
        color: #F0F0F0;
        margin-bottom: 0.5rem;
        font-size: 22px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='color: #FAA15D'>📊 Data & Style</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #F0F0F0'>Statistiques globales sur le jeu de données et les performances du modèle.</p>", unsafe_allow_html=True)

# visualisations images dataset
ROOT_DIR = os.path.dirname(os.path.dirname("./Images0"))  # depuis Pages/
IMG_DIR = os.path.join(ROOT_DIR, "Images0")

image_files = [
    os.path.join(IMG_DIR, f)
    for f in os.listdir(IMG_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

st.markdown("<h3 style='color:#FAA15D;'>Visualisations (images Dataset_Original)</h3>", unsafe_allow_html=True)

# Parcours par pas de 2
for i in range(0, len(image_files), 2):
    row_imgs = image_files[i:i+2]
    col1, col2 = st.columns(2)

    with col1:
        st.image(row_imgs[0])

    if len(row_imgs) > 1:
        with col2:
            st.image(row_imgs[1])

# VISUALISATIONS (IMAGES)
ROOT_DIR = os.path.dirname(os.path.dirname("./Images"))  # depuis Pages/
IMG_DIR = os.path.join(ROOT_DIR, "Images")

image_files = [
    os.path.join(IMG_DIR, f)
    for f in os.listdir(IMG_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]


st.markdown("<h3 style='color:#FAA15D;'>Visualisations (images Résultats)</h3>", unsafe_allow_html=True)

# Parcours par pas de 2
for i in range(0, len(image_files), 2):
    row_imgs = image_files[i:i+2]
    col1, col2 = st.columns(2)

    with col1:
        st.image(row_imgs[0])

    if len(row_imgs) > 1:
        with col2:
            st.image(row_imgs[1])


st.markdown(
    """
    <hr style='border: 0.8px solid black; margin-top: 1.5rem; margin-bottom: 0.5rem;'>

    <div style='text-align: center; color: #F0F0F0; font-size: 14px;'>
        FakeNews Detector · Projet académique · 2025
    </div>
    """,
    unsafe_allow_html=True
)