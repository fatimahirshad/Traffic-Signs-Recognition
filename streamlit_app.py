# streamlit_app.py - TFLite Traffic Sign Recognition (Click-to-Test Images)
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO

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
Upload a traffic sign image or **click on a demo image** below to see the prediction.
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

interpreter, input_details, output_details = load_tflite_model("traffic_sign_model_optimized.tflite")

# -----------------------------
# 3. Class Names
# -----------------------------
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
    x = np.array(img_resized, dtype=np.uint8)
    x = np.expand_dims(x, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    
    pred = interpreter.get_tensor(output_details[0]['index'])[0]
    pred_class = class_names[np.argmax(pred)]
    return pred_class

# -----------------------------
# 5. Sidebar - Upload Image
# -----------------------------
st.sidebar.header("Upload Traffic Sign Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
selected_image = None

if uploaded_file:
    selected_image = Image.open(uploaded_file)

# -----------------------------
# 6. Demo Images
# -----------------------------
st.sidebar.header("Or try Demo Images")
demo_images = {
    
    "image1": "Visuals/IMG_4235.jpg",
    "image2": "Visuals/IMG_4311.jpg",
    "image3": "Visuals/IMG_4462.jpg"
}

for label, url in demo_images.items():
    if st.sidebar.button(label):
        response = requests.get(url)
        selected_image = Image.open(BytesIO(response.content))

# -----------------------------
# 7. Show Selected Image and Predict
# -----------------------------
if selected_image:
    st.image(selected_image, caption="Selected Image", use_column_width=True)
    pred_class = predict_tflite(selected_image)
    st.success(f"Predicted Class: **{pred_class}**")
