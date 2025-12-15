import streamlit as st
import torch
import numpy as np
import re
import os
import pandas as pd
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast, pipeline
from lime.lime_text import LimeTextExplainer

st.session_state["Page"] = "Text Analysis"

# STYLE GLOBAL
st.markdown(
    """
    <style>
    /* Fond global de la page */
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
    /* Zone de texte (input article) */
    .stTextArea textarea {
        background-color: #ffffff;  /* fond de la zone */
        color: black;             /* couleur du texte */
        border-radius: 6px;
        border: 1px solid #black; /* bordure noire */
        caret-color: #000000; 
    }
    .stTextArea textarea:focus {
        border-color: #DBAB04;      /* bord bleu quand on clique */
        box-shadow: 0 0 0 1px #12355b;
    }
    .stTextArea textarea::placeholder {
        color: #525252;              /* mets ici la couleur que tu veux */
        opacity: 1;                  /* pour être sûr qu’il soit bien visible */
    }
    .main-card {
        background-color: #BCF6F3;
        padding: 0.3rem 0.4rem;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    .badge-real {
        background-color: #2ecc71;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge-fake {
        background-color: #e67e22;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    /* Boutons principaux */
    .stButton > button {
        background-color: #1d4f8f;   /* fond */
        color: white;                /* texte */
        border-radius: 6px;
        border: none;
        padding: 0.4rem 0.8rem;
    }
    .stButton > button:hover {
        background-color: #44A4F0;   /* couleur au survol */
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

st.markdown("<h1 style='color: #FAA15D'>📝 Analyse des Articles</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #FCFCFC'>Modèle de prédiction des Fake News avec RoBERTa.</p>", unsafe_allow_html=True)

# MODELES & FONCTIONS 
MODEL_DIR = "roberta_fakenews_model"

@st.cache_resource
def load_models():
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_DIR)
    model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    return tokenizer, model, device, sentiment_model

tokenizer, model, device, sentiment_model = load_models()
explainer = LimeTextExplainer(class_names=["Real", "Fake"])

def basic_style_features(text):
    words = text.split()
    len_words = len(words)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    punct_count = len(re.findall(r"[!?.,;:]", text))
    upper_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    return len_words, avg_word_len, punct_count, upper_ratio

def get_sentiment_score(text):
    res = sentiment_model(text[:512])[0]
    sign = 1 if res["label"] == "POSITIVE" else -1
    return sign * res["score"]

def predict_roberta(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    label = int(probs.argmax())
    return label, probs

def lime_predict_streamlit(texts):
    inputs = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
    return probs

# LAYOUT PRINCIPAL 
col_input, col_result = st.columns([2, 1])
# Colonne gauche : ARTICLES
with col_input:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title" style="font-size: 25px">ARTICLES</p>', unsafe_allow_html=True)

    user_text = st.text_area(
        "",
        height=220,
        
        placeholder="Collez ici l’article, le post ou le message à analyser..."
    )
    c1, c2 = st.columns(2)
    with c1:
        analyze_btn = st.button("Analyser", use_container_width=True)
    with c2:
        reset_btn = st.button(" Réinitialiser", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if reset_btn:
        st.experimental_rerun()

# Colonne droite : RESULTATS 
with col_result:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title" style="font-size: 25px; margin-bottom: 2.2rem">RESULTATS</p>', unsafe_allow_html=True)

    if analyze_btn and user_text.strip():
        label_idx, probs = predict_roberta(user_text)
        classes = ["Real", "Fake"]
        label_name = classes[label_idx]

        sentiment = get_sentiment_score(user_text)
        len_words, avg_word_len, punct_count, upper_ratio = basic_style_features(user_text)
        # Badge de prédiction
        if label_name == "Real":
            st.markdown("**Prédiction :** <span class='badge-real'>REAL</span>", unsafe_allow_html=True)
        else:
            st.markdown("**Prédiction :** <span class='badge-fake'>FAKE</span>", unsafe_allow_html=True)

        st.write(f"Probabilité Real : {probs[0]:.6f}")
        st.write(f"Probabilité Fake : {probs[1]:.6f}")
        st.progress(float(probs[1]))

        st.markdown("---")
        st.markdown("**Sentiment** *(−1 = très négatif, +1 = très positif)*")
        st.slider("Score de sentiment", -1.0, 1.0, float(sentiment), 0.01, disabled=True)

        st.markdown("---")
        st.markdown("**Style du texte**")
        st.write(f"- Longueur (mots) : {len_words}")
        st.write(f"- Longueur moyenne des mots : {avg_word_len:.2f}")
        st.write(f"- Ponctuation : {punct_count}")
        st.write(f"- Ratio majuscules : {upper_ratio:.6f}")
    else:
        st.info("Collez un texte à gauche puis cliquez sur **Analyser**.")

    st.markdown('</div>', unsafe_allow_html=True)

#  BLOC LIME 
if analyze_btn and user_text.strip():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Explication LIME</p>', unsafe_allow_html=True)

    short_text = " ".join(user_text.split()[:300])
    exp = explainer.explain_instance(
        short_text,
        lime_predict_streamlit,
        num_features=10,
        num_samples=500
    )
    st.pyplot(exp.as_pyplot_figure())
    st.caption("Orange : mots qui poussent vers Fake. Vert : mots qui poussent vers Real.")
    st.markdown('</div>', unsafe_allow_html=True)

## Historique dans la session
import datetime

if "history" not in st.session_state:
    st.session_state["history"] = []

# après avoir calculé label_name, probs, sentiment, etc.
if analyze_btn and user_text.strip():
    result = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "text": user_text[:200],       # on garde un extrait
        "label": label_name,
        "prob_real": float(probs[0]),
        "prob_fake": float(probs[1]),
        "sentiment": float(sentiment),
    }
    st.session_state["history"].append(result)

    # affichage d’un petit historique en bas de la page
    st.markdown("### Historique (session)")
    for h in reversed(st.session_state["history"][-5:]):
        st.write(f"- {h['timestamp']} ·{h['label']}={h['prob_real']:.6f} · Fake={h['prob_fake']:.6f}")

# Sauvegarde dans un fichier CSV
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
CSV_FILE = os.path.join(ROOT_DIR, "analysis_history.csv")
# Initialisation robuste
if os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0:
    history_df = pd.read_csv(CSV_FILE)
else:
    history_df = pd.DataFrame(columns=[
        "timestamp", "source_page", "text_snippet",
        "label", "prob_real", "prob_fake", "sentiment"
    ])

# après la prédiction
if analyze_btn and user_text.strip():
    # ... calcul des résultats ...
    new_row = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_page": "Text Analysis",
        "text_snippet": user_text[:200],
        "label": label_name,
        "prob_real": float(probs[0]),
        "prob_fake": float(probs[1]),
        "sentiment": float(sentiment),
    }
    history_df.loc[len(history_df)] = new_row
    history_df.to_csv(CSV_FILE, index=False)

## history_df.to_csv(CSV_FILE, index=False)
st.markdown(
    """
    <hr style='border: 0.8px solid black; margin-top: 1.5rem; margin-bottom: 0.5rem;'>

    <div style='text-align: center; color: #FCFCFC; font-size: 14px;'>
        FakeNews Detector · Projet académique · 2025
    </div>
    """,
    unsafe_allow_html=True
)
