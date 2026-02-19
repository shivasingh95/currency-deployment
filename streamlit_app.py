import streamlit as st
import tensorflow as tf
import numpy as np
import time
from PIL import Image

st.set_page_config(page_title="Smart DaanPeti", layout="centered")
st.title("💰 Smart Automatic Donation System")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("final_currency_model.keras")

model = load_model()

CLASS_NAMES = ['10','100','20','200','5','50','500','None']
IMG_SIZE = 224
CONF_THRESHOLD = 0.85
COOLDOWN = 3

if "total_amount" not in st.session_state:
    st.session_state.total_amount = 0

if "last_detection_time" not in st.session_state:
    st.session_state.last_detection_time = 0

st.subheader(f"💵 Total Donation: ₹{st.session_state.total_amount}")

mode = st.radio("Select Mode:", ["Upload Image", "Webcam Auto Scan"])

def predict_image(image):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = np.expand_dims(img, 0)

    preds = model.predict(img)
    idx = int(np.argmax(preds))
    conf = float(np.max(preds))
    return CLASS_NAMES[idx], conf

# ================= Upload Mode =================

if mode == "Upload Image":

    uploaded_file = st.file_uploader("Upload currency image", type=["jpg","jpeg","png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)

        if st.button("Predict"):
            label, confidence = predict_image(image)

            st.success(f"Prediction: ₹{label}")
            st.info(f"Confidence: {confidence*100:.2f}%")

            if confidence > CONF_THRESHOLD and label != "None":
                try:
                    st.session_state.total_amount += int(label)
                    st.success(f"₹{label} Added to Total!")
                except:
                    pass

# ================= Webcam Mode =================

elif mode == "Webcam Auto Scan":

    st.write("📷 Show the note clearly in front of camera.")
    picture = st.camera_input("Camera")

    if picture:
        current_time = time.time()
        image = Image.open(picture).convert("RGB")

        label, confidence = predict_image(image)

        st.write(f"Detected: ₹{label}")
        st.write(f"Confidence: {confidence*100:.2f}%")

        if (
            confidence > CONF_THRESHOLD
            and label != "None"
            and current_time - st.session_state.last_detection_time > COOLDOWN
        ):
            try:
                st.session_state.total_amount += int(label)
                st.session_state.last_detection_time = current_time
                st.success(f"✅ ₹{label} Added Automatically!")
            except:
                pass

        time.sleep(1)
        st.rerun()

# Reset

if st.button("🔄 Reset Total"):
    st.session_state.total_amount = 0
    st.session_state.last_detection_time = 0
    st.success("Total Reset Successfully!")
