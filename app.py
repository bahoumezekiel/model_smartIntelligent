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
st.set_page_config(page_title="🐣 Egg Classifier - MobileNetV2", layout="centered")

st.title("🐣 Bienvenu sur best_egg_model (classificateur d'oeuf) ")
st.markdown(
    "Ce modèle prédit si un œuf est **mort**, **fertile** ou **infertile** à partir d'une image de mirage."
)

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

# Interface utilisateur
uploaded_file = st.file_uploader("📸 Téléverser une image (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.read()))
    st.image(image, caption="Image téléversée",  use_container_width=True)

    if st.button("🔍 Lancer la prédiction"):
        with st.spinner("Analyse de l'image en cours..."):
            label, proba, topk, preds = predict(image)

        # Résultat principal
        st.success(f"**Classe prédite : {label.upper()}** ({proba*100:.2f} % de confiance)")

        # Détails top-3
        st.markdown("### 🔢 Top 3 des probabilités :")
        for cls, p in topk:
            st.write(f"- **{cls}** : {p*100:.2f} %")

        # Tableau complet des probabilités
        st.markdown("---")
        st.write("### 📊 Probabilités complètes")
        probs_dict = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(preds))}
        st.table(probs_dict)

else:
    st.info("👆 Charge une image d'œuf (par exemple depuis ton dossier `testing`).")
