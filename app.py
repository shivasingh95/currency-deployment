import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------
# Page Configuration
# -----------------------
st.set_page_config(
    page_title="Currency Classifier",
    layout="centered"
)

st.title("💵 Currency Classification System")
st.write("Upload a currency note image to classify")

# -----------------------
# Load Clean Keras Model
# -----------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("final_model.keras")

model = load_model()

# IMPORTANT:
# Make sure this order matches your training folders
CLASS_NAMES = ["10", "20", "50", "100", "200", "500"]

# -----------------------
# File Upload
# -----------------------
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocessing
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    with st.spinner("🔍 Predicting..."):
        predictions = model.predict(img_array)
        predicted_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

    predicted_label = CLASS_NAMES[predicted_index]

    st.success(f"Prediction: ₹ {predicted_label}")
    st.write(f"Confidence: {confidence:.4f}")
