import tensorflow as tf
import numpy as np
import cv2

MODEL_PATH = "final_currency_model.keras"
IMG_SIZE = 224

class_names = ['10','100','20','200','5','50','500','None']

# Load model once
model = tf.keras.models.load_model(MODEL_PATH)

def predict_from_image(image):
    """
    Receives image in BGR format (OpenCV)
    Returns:
        predicted label
        confidence score
    """

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))

    # Preprocess
    pre = tf.keras.applications.mobilenet_v2.preprocess_input(img_resized)
    pre = np.expand_dims(pre, axis=0)

    preds = model.predict(pre)[0]

    idx = np.argmax(preds)
    conf = float(np.max(preds))

    return class_names[idx], conf
