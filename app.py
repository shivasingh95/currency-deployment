import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(page_title="Currency Classifier", layout="centered")
st.title("💵 Indian Currency Classification")
st.write("Upload a currency note image")

# =========================
# LOAD MODEL (.keras)
# =========================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("final_currency_model.keras")

model = load_model()

# =========================
# CLASS LABELS
# =========================

CLASS_NAMES = ['10','100','20','200','5','50','500','None']
IMG_SIZE = 224

# =========================
# FILE UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    img = image.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img)

    # Same preprocessing used during training
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    with st.spinner("🔍 Predicting..."):
        predictions = model.predict(img)
        predicted_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

    st.success(f"Prediction: ₹ {CLASS_NAMES[predicted_index]}")
    st.write(f"Confidence: {confidence*100:.2f}%")
