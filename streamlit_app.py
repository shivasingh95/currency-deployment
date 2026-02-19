import streamlit as st
import requests
import time
from PIL import Image

API_URL = "http://127.0.0.1:5000/predict"

st.set_page_config(page_title="Smart DaanPeti", layout="centered")

st.title("💰 Smart Automatic Donation System")

# ---------------------------
# Session State Initialization
# ---------------------------
if "total_amount" not in st.session_state:
    st.session_state.total_amount = 0

if "last_detection_time" not in st.session_state:
    st.session_state.last_detection_time = 0

if "auto_mode" not in st.session_state:
    st.session_state.auto_mode = False

CONF_THRESHOLD = 0.85
COOLDOWN = 3  # seconds

st.subheader(f"💵 Total Donation: ₹{st.session_state.total_amount}")

# ---------------------------
# Mode Selection
# ---------------------------
mode = st.radio("Select Mode:", ["Upload Image", "Webcam Auto Scan"])

# ===========================
# 1️⃣ IMAGE UPLOAD MODE
# ===========================
if mode == "Upload Image":

    uploaded_file = st.file_uploader("Upload currency image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Predict"):
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(API_URL, files=files).json()

            if "prediction" in response:
                label = response["prediction"]
                confidence = response["confidence"]

                st.success(f"Prediction: ₹{label}")
                st.info(f"Confidence: {confidence*100:.2f}%")

                if confidence > CONF_THRESHOLD and label != "None":
                    try:
                        value = int(label)
                        st.session_state.total_amount += value
                        st.success(f"₹{value} Added to Total!")
                    except:
                        pass

# ===========================
# 2️⃣ WEBCAM AUTO MODE
# ===========================
elif mode == "Webcam Auto Scan":

    st.write("📷 Show the note clearly in front of camera.")

    picture = st.camera_input("Camera")

    if picture:
        current_time = time.time()

        files = {"file": picture.getvalue()}
        response = requests.post(API_URL, files=files).json()

        if "prediction" in response:
            label = response["prediction"]
            confidence = response["confidence"]

            st.write(f"Detected: ₹{label}")
            st.write(f"Confidence: {confidence*100:.2f}%")

            if (
                confidence > CONF_THRESHOLD
                and label != "None"
                and current_time - st.session_state.last_detection_time > COOLDOWN
            ):
                try:
                    value = int(label)
                    st.session_state.total_amount += value
                    st.session_state.last_detection_time = current_time
                    st.success(f"✅ ₹{value} Added Automatically!")
                except:
                    pass

        time.sleep(1)
        st.rerun()

# ---------------------------
# Reset Button
# ---------------------------
if st.button("🔄 Reset Total"):
    st.session_state.total_amount = 0
    st.session_state.last_detection_time = 0
    st.success("Total Reset Successfully!")
