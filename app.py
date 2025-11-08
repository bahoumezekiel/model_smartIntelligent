# app.py
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import io
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------------- CONFIGURATION ----------------
MODEL_PATH = "best_egg_model.keras"
IMG_SIZE = (224, 224)
LABEL_MAP = {
    'dead': 0,
    'fertile': 1,
    'infertile': 2
}
CLASS_NAMES = {v: k for k, v in LABEL_MAP.items()}
# ------------------------------------------------

# Configuration de la page Streamlit
st.set_page_config(
    page_title="🐣 Egg Classifier - MobileNetV2", 
    layout="centered",
    page_icon="🥚"
)

# CSS personnalisé pour améliorer le design
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 3rem;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #3498db;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .probability-bar {
        background-color: #ecf0f1;
        border-radius: 10px;
        margin: 10px 0;
        overflow: hidden;
    }
    .probability-fill {
        padding: 8px;
        color: white;
        text-align: center;
        border-radius: 10px;
         min-width: 80px;
    }
    .fertile { background-color: #2ecc71; }
    .infertile { background-color: #f39c12; }
    .dead { background-color: #e74c3c; }
    .stProgress > div > div > div > div {
        background-color: #2ecc71;
    }
</style>
""", unsafe_allow_html=True)

# En-tête principale
st.markdown('<h1 class="main-header">🔬 Classificateur Intelligent d\'Œufs</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analyse de mirage d\'œufs par Intelligence Artificielle - Détection de fertilité</p>', unsafe_allow_html=True)

# Section d'information
with st.expander("ℹ️ Instructions d'utilisation"):
    st.markdown("""
    1. **Téléchargez** une image claire d'un œuf miragé
    2. **Attendez** le traitement automatique de l'image
    3. **Consultez** les résultats détaillés de l'analyse
    4. **Interprétez** les probabilités de classification
    
    **Conseils :** Utilisez des images bien éclairées avec un fond neutre pour de meilleurs résultats.
    """)

# Chargement du modèle avec mise en cache
@st.cache_resource(show_spinner=True)
def load_model(path):
    model = tf.keras.models.load_model(path)
    return model

model = load_model(MODEL_PATH)

# ----------- FONCTIONS DE TRAITEMENT ------------
def preprocess_image(image: Image.Image, target_size=IMG_SIZE):
    """Prépare l'image avant prédiction."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = ImageOps.exif_transpose(image)
    image = image.resize(target_size)
    arr = np.array(image).astype(np.float32)
    arr = preprocess_input(arr)  # prétraitement MobileNetV2
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict(image: Image.Image):
    x = preprocess_image(image)
    preds = model.predict(x, verbose=0)[0]
    top_idx = np.argmax(preds)
    label = CLASS_NAMES[int(top_idx)]
    proba = float(preds[int(top_idx)])
    topk = sorted(enumerate(preds), key=lambda x: x[1], reverse=True)[:3]
    topk = [(CLASS_NAMES[int(i)], float(p)) for i, p in topk]
    return label, proba, topk, preds
# ------------------------------------------------

# Section de téléchargement
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown("### 📸 Téléversement d'Image")
st.markdown("Glissez-déposez ou sélectionnez une image d'œuf à analyser")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    # Affichage de l'image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 🖼️ Image Originale")
        image = Image.open(io.BytesIO(uploaded_file.read()))
        st.image(image, caption="Image téléversée", use_container_width=True)
    
    with col2:
        st.markdown("#### 🔍 Analyse")
        if st.button("Lancer l'Analyse", use_container_width=True):
            with st.spinner("🔬 Analyse en cours... Veuillez patienter"):
                label, proba, topk, preds = predict(image)

            # Affichage des résultats principaux
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            
            # Icônes différentes selon la classe
            icons = {'fertile': '🐣', 'infertile': '🥚', 'dead': '💀'}
            colors = {'fertile': '#2ecc71', 'infertile': '#f39c12', 'dead': '#e74c3c'}
            
            st.markdown(f"### {icons[label]} Résultat Principal")
            st.markdown(f"**Niveau de confiance :** {proba*100:.2f}%")
            
            # Barre de progression
            progress_value = proba
            st.progress(progress_value)
            
            st.markdown('</div>', unsafe_allow_html=True)

            # Détails des probabilités
            st.markdown("### Analyse Détaillée")
            
            for cls, p in topk:
                width = p * 100
                st.markdown(f"**{cls.capitalize()}**")
                st.markdown(
                    f'<div class="probability-bar">'
                    f'<div class="probability-fill {cls}" style="width: {width}%">'
                    f'{p*100:.2f}%</div></div>', 
                    unsafe_allow_html=True
                )

            # Tableau des probabilités complètes
            st.markdown("### 📈 Probabilités Complètes")
            probs_dict = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(preds))}
            
            # Affichage sous forme de métriques
            cols = st.columns(3)
            for i, (cls, prob) in enumerate(probs_dict.items()):
                with cols[i]:
                    st.metric(
                        label=f"{icons.get(cls, '📊')} {cls.capitalize()}",
                        value=f"{prob*100:.2f}%"
                    )

else:
    # Section d'information quand aucune image n'est téléchargée
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🥚 Fertile")
        st.markdown("Œuf fécondé avec embryon viable")
        
    with col2:
        st.markdown("### 🔅 Infertile")
        st.markdown("Œuf non fécondé ou non viable")
        
    with col3:
        st.markdown("### 💀 Mort")
        st.markdown("Embryon décédé pendant l'incubation")
    
    st.info("👆 Commencez par téléverser une image d'œuf ci-dessus pour l'analyse.")

# Pied de page
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #7f8c8d;'>"
    "Système expert d'analyse de mirage d'œufs - Technologies IA Avancée"
    "</div>", 
    unsafe_allow_html=True
)