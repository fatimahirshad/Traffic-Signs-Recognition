# app.py - Streamlit App for Traffic Sign Recognition

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
import os

# -----------------------------
# 1. Page Config
# -----------------------------
st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚦 Traffic Sign Recognition App")
st.markdown("""
This app allows you to **upload a traffic sign image** and predicts its class using a **MobileNetV2-based CNN**.
You can also **explore confusion matrix and demo predictions**.
""")

# -----------------------------
# 2. Load Model
# -----------------------------
@st.cache_data(show_spinner=True)
def load_trained_model(model_path):
    model = load_model(model_path)
    return model

model = load_trained_model("traffic_sign_model_optimized.tflite")

# -----------------------------
# 3. Load Class Names
# -----------------------------
# Ensure these match your training generator
class_names = sorted(os.listdir("traffic_sign_dataset/train"))

# -----------------------------
# 4. Sidebar - Upload Image
# -----------------------------
st.sidebar.header("Upload Traffic Sign Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess
    img_resized = img.resize((224,224))
    x = np.array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # Predict
    pred_prob = model.predict(x)
    pred_class = class_names[np.argmax(pred_prob)]
    confidence = np.max(pred_prob)*100
    
    st.success(f"Predicted Class: **{pred_class}** ({confidence:.2f}% confidence)")

# -----------------------------
# 5. Sidebar - Show Demo Confusion Matrix
# -----------------------------
if st.sidebar.checkbox("Show Confusion Matrix"):
    st.subheader("Confusion Matrix - Test Set")
    
    # Load test data (make sure test folder exists)
    import tensorflow as tf
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)
    test_generator = test_datagen.flow_from_directory(
        "traffic_sign_dataset/test",
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    y_true = test_generator.classes
    y_pred = np.argmax(model.predict(test_generator), axis=1)
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Display heatmap
    plt.figure(figsize=(12,12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_generator.class_indices.keys())
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title("Confusion Matrix - Traffic Sign Classes", fontsize=16)
    st.pyplot(plt)
    
    st.markdown("**Note:** Diagonal values show correct predictions. Off-diagonal shows misclassifications.")

# -----------------------------
# 6. Sidebar - Show Random Test Samples
# -----------------------------
if st.sidebar.checkbox("Show Sample Predictions"):
    st.subheader("Random Test Samples")
    
    test_files = []
    for cls in os.listdir("traffic_sign_dataset/test"):
        cls_folder = os.path.join("traffic_sign_dataset/test", cls)
        for f in os.listdir(cls_folder):
            test_files.append((os.path.join(cls_folder, f), cls))
    
    np.random.shuffle(test_files)
    
    cols = st.columns(3)
    for idx, (fpath, true_cls) in enumerate(test_files[:9]):
        img = Image.open(fpath).resize((224,224))
        x = np.array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        pred_cls = class_names[np.argmax(model.predict(x))]
        
        with cols[idx%3]:
            st.image(img, caption=f"T: {true_cls}\nP: {pred_cls}", use_column_width=True)
