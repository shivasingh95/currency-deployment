import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page config
st.set_page_config(page_title="Currency Classifier", layout="centered")

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("final_currency_model.keras")

model = load_model()

# Update according to your training class order
CLASS_NAMES = ["10", "20", "50", "100", "200", "500"]

st.title("💵 Currency Classification System")
st.write("Upload an image of currency note to classify.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    with st.spinner("🔍 Predicting..."):
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)
        confidence = float(np.max(predictions))

    label = CLASS_NAMES[predicted_index]

    st.success(f"Prediction: ₹ {label}")
    st.write(f"Confidence: {confidence:.4f}")
