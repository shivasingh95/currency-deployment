import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page settings
st.set_page_config(page_title="Currency Classifier", layout="centered")

st.title("💵 Currency Classification System")
st.write("Upload a currency note image to classify")

# Load model once
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("converted_model.h5", compile=False)
    return model

model = load_model()

# IMPORTANT: Update according to your training folder order
CLASS_NAMES = ["10", "20", "50", "100", "200", "500"]

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess (adjust size if your model used different input size)
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("🔍 Predicting..."):
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)
        confidence = float(np.max(predictions))

    predicted_label = CLASS_NAMES[predicted_index]

    st.success(f"Prediction: ₹ {predicted_label}")
    st.write(f"Confidence: {confidence:.4f}")
