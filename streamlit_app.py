import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "final_currency_model.keras"
IMG_SIZE = 224

class_names = ['10','100','20','200','5','50','500','None']

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

def predict_image(image):
    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]
    idx = np.argmax(preds)
    conf = float(np.max(preds))

    return class_names[idx], conf

st.title("💰 Smart Currency Classifier")

uploaded_file = st.file_uploader("Upload Note Image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    label, confidence = predict_image(image)

    st.success(f"Prediction: ₹{label}")
    st.write(f"Confidence: {confidence*100:.2f}%")
