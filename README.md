
# 🚦 Traffic Sign Recognition using Deep Learning

A Computer Vision project that recognizes and classifies Pakistani traffic signs using a deep learning model and an interactive Streamlit application.

The system allows users to upload a traffic sign image or select a demo image and receive a prediction of the sign class.

---

## 📌 Project Overview

Traffic sign recognition is an important application in intelligent transportation systems and autonomous driving.  
This project builds a **deep learning image classification model** trained on Pakistani traffic sign images and deploys it through an easy-to-use **Streamlit web application**.

The model is optimized and converted to **TensorFlow Lite** to reduce size and enable fast inference in web environments.

---

## 🎯 Features

- 🚦 Traffic sign classification
- 📤 Upload your own traffic sign image
- 🖼 Demo images available in the sidebar
- ⚡ Optimized lightweight model (TFLite)
- 🎨 Clean and interactive Streamlit UI
- 📊 Deep learning based prediction system

---

## 🧠 Model Architecture

The model is built using **Transfer Learning** with:

- MobileNetV2 (pretrained backbone)
- Custom classification head
- Image preprocessing and normalization
- TensorFlow Lite optimization for deployment

---

## 📂 Dataset

The model is trained using the following dataset:

Dataset Link:  
https://www.kaggle.com/datasets/mexwell/pakistani-traffic-sign-recognition-dataset

Dataset contains multiple classes of Pakistani traffic signs such as:

- Stop
- Give Way
- No Parking
- Speed Limits
- U-Turn
- Railway Crossing
- Pedestrian Crossing
- And many more

Total classes used in this project: **36 traffic sign categories**

---

## 🖥 Demo Application

You can try the live application here:

Streamlit Demo:  
**[https://traffic-signs-recognition-8uglld2twkciaioicy9z8e.streamlit.app]**

Users can:

1. Upload a traffic sign image
2. Click demo images from the sidebar
3. Receive the predicted traffic sign class instantly

---

## 📁 Project Structure

```

Traffic-Signs-Recognition
│
├── streamlit_app.py        # Streamlit UI application
├── traffic_sign_model.tflite  # Optimized trained model
├── requirements.txt        # Python dependencies
├── runtime.txt             # Python runtime for deployment
├── README.md               # Project documentation
│
└── Visuals
├── IMG_4311.jpg
├── IMG_4462.jpg
└── IMG_4235.jpg

```

---

## ⚙️ Installation

Clone the repository:

```

git clone https://github.com/yourusername/Traffic-Signs-Recognition.git

cd Traffic-Signs-Recognition

```

Install dependencies:

```

pip install -r requirements.txt

```

---

## ▶️ Run the Streamlit App

```

streamlit run streamlit_app.py

```

The application will open in your browser.

---

## 📦 Requirements

Main libraries used:

- Python
- TensorFlow
- TensorFlow Lite
- Streamlit
- NumPy
- Pillow
- Requests

All dependencies are listed in **requirements.txt**.

---

## 📊 Model Optimization

To ensure smooth deployment on Streamlit Cloud, the trained model was optimized using:

- TensorFlow Lite conversion
- Model quantization
- Reduced model size (under 25MB)

This enables faster loading and inference in the deployed application.

---

## 🚀 Future Improvements

Possible enhancements include:

- Real-time traffic sign detection from camera
- Object detection using YOLO
- Mobile application integration
- Larger and more diverse dataset
- Higher model accuracy

---

## 👩‍💻 Author

Fatima Irshad  

Computer Science Student | AI & Computer Vision Enthusiast

---

## ⭐ Acknowledgments

Dataset provided by Kaggle community contributors.

This project was developed as part of a **Computer Vision Internship Task**.
```
