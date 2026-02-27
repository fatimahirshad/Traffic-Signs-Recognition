# streamlit_app.py - Traffic Sign Recognition (TFLite, Upload-Only UI)

import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image

# -----------------------------
# 1. Page Config
# -----------------------------
st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="centered"
)

st.title("🚦 Traffic Sign Recognition App")
st.markdown("""
Upload a traffic sign image, and the app will predict its class with confidence using a **MobileNetV2-based CNN (TFLite optimized)**.
""")

# -----------------------------
# 2. Load TFLite Model
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

# Provide path to your optimized TFLite model
interpreter, input_details, output_details = load_tflite_model("traffic_sign_model_optimized.tflite")

# -----------------------------
# 3. Define Class Names
# -----------------------------
# Replace with your actual class names in order used during training
class_names = [
    "Bridge Ahead", "Cross Roads", "Give Way", "Left bend", "No Horns", 
    "No Mobile Allowed", "No Overtaking", "No Parking", "No U-Turn", 
    "No left turn", "No right turn", "Parking", "Pedestrians", "Railway Crossing", 
    "Right bend", "Control", "Road Divides", "Roundabout Ahead", "Sharp Right Turn", 
    "Slow", "Speed Breaker Ahead", "Speed Limit (20 kmph)", "Speed Limit (25 kmph)", 
    "Speed Limit (30 kmph)", "Speed Limit (40 kmph)", "Speed Limit (45 kmph)", 
    "Speed Limit (50 kmph)", "Speed Limit (60 kmph)", "Speed Limit (65 kmph)", 
    "Speed Limit (70 kmph)", "Speed Limit (80 kmph)", "Steep Descent", "Stop 1", 
    "Stop 2", "U-Turn", "Zigzag Road Ahead"
]

# -----------------------------
# 4. Prediction Function
# -----------------------------
def predict_tflite(img: Image.Image):
    img_resized = img.resize((224,224))
    x = np.array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = x.astype(np.uint8)  # TFLite expects uint8 for integer quantized models
    
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])[0]
    
    pred_class = class_names[np.argmax(pred)]
    confidence = np.max(pred)
    return pred_class, confidence

# -----------------------------
# 5. Sidebar - Upload Image
# -----------------------------
st.sidebar.header("Upload Traffic Sign Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    pred_class, confidence = predict_tflite(img)
    st.success(f"Predicted Class: **{pred_class}**")
    st.info(f"Confidence: **{confidence*100:.2f}%**")
    
    st.markdown("---")
    st.markdown("✅ **Upload another image to get prediction.**")
