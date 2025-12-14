import streamlit as st
import pandas as pd
import torch
import numpy as np
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast

st.session_state["page"] = "batch"

# ==== STYLE GLOBAL (copié / adapté) ====
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
    /* Si tu as des éléments trop larges, on limite leur largeur */
    .block-container > div {
        max-width: 100%;
    }
    .main-card {
        background-color: #BCF6F3;
        padding: 0.3rem 0.4rem;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    
    /* Boutons */
    .stButton > button {
        background-color: #577FB2;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.4rem 0.8rem;
    }
    /* Label du file_uploader */
    div[data-testid="stFileUploader"] > label {
        color: #FCFCFC;           /* couleur du texte "Uploader un fichier CSV" */
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #1d4f8f;
        color: white;
    }
    .section-title {
        font-weight: 700;
        color: #1d4f8f;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='color: #FAA15D'>📂 Analyse Batch</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #FCFCFC'>Analyser plusieurs articles en une seule fois (fichier).</p>", unsafe_allow_html=True)

## MODELES 
MODEL_DIR = "roberta_fakenews_model"

@st.cache_resource
def load_model():
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_DIR)
    model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device); model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model()

def predict_batch(texts):
    enc = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    labels = probs.argmax(axis=1)
    return labels, probs

## LAYOUT
col_left, col_right = st.columns([1.4, 2])

## Colonne gauche 
with col_left:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title" style="font-size: 25px">FICHIER</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Uploader un fichier CSV", type="csv")
    st.markdown("<p style='color: #1F1F1F'>Assurez-vous que le fichier contient au moins une colonne text ou title + text.</p>", unsafe_allow_html=True)
    use_concat = st.checkbox("Concaténer title + text si les deux colonnes existent.", value=True)
    run_btn = st.button(" Lancer l’analyse batch", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
## Colonne droite : RÉSULTATS
with col_right:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title" style="font-size: 25px; margin-bottom: 2.2rem">RÉSULTATS</p>', unsafe_allow_html=True)

    if uploaded is not None and run_btn:
        df = pd.read_csv(uploaded)

        # Texte à analyser
        if use_concat and {"title", "text"}.issubset(df.columns):
            texts = (df["title"].fillna("") + " . " + df["text"].fillna("")).tolist()
        elif "text" in df.columns:
            texts = df["text"].fillna("").tolist()
        else:
            st.error("Aucune colonne `text` trouvée dans le CSV.")
            st.markdown('</div>', unsafe_allow_html=True)
            st.stop()

        labels, probs = predict_batch(texts)
        df["pred_label"] = labels
        df["pred_name"] = df["pred_label"].map({0: "Real", 1: "Fake"})
        df["prob_real"] = probs[:, 0]
        df["prob_fake"] = probs[:, 1]

        # Petit résumé
        n_total = len(df)
        n_fake = int((df["pred_name"] == "Fake").sum())
        n_real = int((df["pred_name"] == "Real").sum())
        st.write(f"Total textes : **{n_total}** | Real : **{n_real}** | Fake : **{n_fake}**")

        st.dataframe(
            df[["pred_name", "prob_real", "prob_fake"] +
               [c for c in df.columns if c not in ["pred_name", "prob_real", "prob_fake", "pred_label"]]],
            use_container_width=True
        )
    elif uploaded is None:
        st.info("Charge un fichier CSV à gauche, puis clique sur **Lancer l’analyse batch**.")
    else:
        st.info("Clique sur **Lancer l’analyse batch** pour démarrer.")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    """
    <hr style='border: 0.8px solid black; margin-top: 1.5rem; margin-bottom: 0.5rem;'>

    <div style='text-align: center; color: #FCFCFC; font-size: 14px;'>
        FakeNews Detector · Projet académique · 2025
    </div>
    """,
    unsafe_allow_html=True
)