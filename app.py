import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

st.set_page_config(page_title="Currency Classifier", layout="centered")

st.title("💵 Currency Classification System")
st.write("Upload a currency note image to classify")

# -----------------------
# Rebuild Architecture
# -----------------------
@st.cache_resource
def load_model():

    NUM_CLASSES = 6  # adjust if needed

    base_model = MobileNetV2(
        weights=None,
        include_top=False,
        input_shape=(224,224,3)
    )

    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    # Load weights only
    model.load_weights("final_weights.h5")

    return model

model = load_model()

CLASS_NAMES = ["10", "20", "50", "100", "200", "500"]

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    img = image.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img,0)

    with st.spinner("🔍 Predicting..."):
        preds = model.predict(img)
        idx = int(np.argmax(preds))
        conf = float(np.max(preds))

    st.success(f"Prediction: ₹ {CLASS_NAMES[idx]}")
    st.write(f"Confidence: {conf:.4f}")
