# streamlit_app.py - TFLite version for Traffic Sign Recognition

import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
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
You can also **explore confusion matrix and sample predictions**.
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
# 3. Load Class Names
# -----------------------------
class_names = sorted(os.listdir("traffic_sign_dataset/train"))

# -----------------------------
# 4. Prediction Function
# -----------------------------
def predict_tflite(img: Image.Image):
    img_resized = img.resize((224,224))
    x = np.array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = x.astype(np.uint8)  # TFLite integer quantized
    
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
    st.success(f"Predicted Class: **{pred_class}** ({confidence*100:.2f}% confidence)")

# -----------------------------
# 6. Sidebar - Show Confusion Matrix
# -----------------------------
if st.sidebar.checkbox("Show Confusion Matrix"):
    st.subheader("Confusion Matrix - Test Set")
    
    test_dir = "traffic_sign_dataset/test"
    test_files, y_true, y_pred = [], [], []
    
    # Load test images and predict
    for idx, cls in enumerate(class_names):
        cls_folder = os.path.join(test_dir, cls)
        if not os.path.exists(cls_folder):
            continue
        for f in os.listdir(cls_folder):
            fpath = os.path.join(cls_folder, f)
            img = Image.open(fpath)
            pred_cls, _ = predict_tflite(img)
            
            test_files.append(fpath)
            y_true.append(idx)
            y_pred.append(class_names.index(pred_cls))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12,12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title("Confusion Matrix - Traffic Sign Classes", fontsize=16)
    st.pyplot(plt)

# -----------------------------
# 7. Sidebar - Show Random Test Samples
# -----------------------------
if st.sidebar.checkbox("Show Sample Predictions"):
    st.subheader("Random Test Samples")
    
    sample_files = []
    for cls in class_names:
        cls_folder = os.path.join("traffic_sign_dataset/test", cls)
        if not os.path.exists(cls_folder):
            continue
        for f in os.listdir(cls_folder):
            sample_files.append((os.path.join(cls_folder, f), cls))
    
    np.random.shuffle(sample_files)
    cols = st.columns(3)
    
    for idx, (fpath, true_cls) in enumerate(sample_files[:9]):
        img = Image.open(fpath)
        pred_cls, _ = predict_tflite(img)
        
        with cols[idx%3]:
            st.image(img.resize((224,224)), caption=f"T: {true_cls}\nP: {pred_cls}", use_column_width=True)
