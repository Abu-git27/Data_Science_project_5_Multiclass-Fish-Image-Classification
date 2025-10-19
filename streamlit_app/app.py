# ========================================
# Import Libraries
# ========================================
import streamlit as st
import tensorflow as tf
import numpy as np
import os
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

# ========================================
# Straemlit Configuration
# ========================================
st.set_page_config(page_title="Fish Image Classification", layout="wide")

# Directory paths
MODEL_DIRS = [
    os.path.join("..", "notebooks", "best_model"),
    os.path.join("..", "notebooks", "models")
]

# Fish classes
CLASS_NAMES = [
    'animal fish', 'animal fish bass', 'black_sea_sprat', 'gilt_head_bream',
    'hourse_mackerel', 'red_mullet', 'red_sea_bream', 'sea_bass',
    'shrimp', 'striped_red_mullet', 'trout'
]

# ========================================
# Load Models
# ========================================
@st.cache_resource
def load_all_models():
    models = {}
    for model_dir in MODEL_DIRS:
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                if file.endswith(".h5"):
                    model_name = file.replace("_best_model.h5", "").replace(".h5", "")
                    try:
                        models[model_name] = load_model(os.path.join(model_dir, file))
                    except Exception as e:
                        st.warning(f"Error loading {file}: {e}")
    return models

models = load_all_models()

# ========================================
# Model Comparison Data 
# ========================================
model_metrics = pd.DataFrame({
    "Model": ["Custom CNN", "VGG16", "MobileNet", "EfficientNetV2B0", "InceptionV3", "ResNet50"],
    "Training Accuracy (%)": [92.12, 94.87, 98.90, 17.12, 98.81, 69.23],
    "Test Accuracy (%)": [92.47, 97.05, 99.18, 16.32, 99.34, 69.72]
})

# Identify best model
best_model = model_metrics.iloc[model_metrics['Test Accuracy (%)'].idxmax()]

# ========================================
# Image Prediction
# ========================================
def predict_image(model, img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_class, confidence

# ========================================
# Sidebar Navigation
# ========================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Model Comparison", "Prediction", "About"])

# ========================================
# Home Page
# ========================================
if page == "Home":
    st.title("Multiclass Fish Image Classification")
    st.write("""
    This project classifies fish species into multiple categories using **deep learning models**.
    It evaluates six architectures — Custom CNN, VGG16, MobileNet, EfficientNetV2B0, InceptionV3, and ResNet50 —
    on both training and test datasets.
    """)
    st.markdown("---")
    st.subheader("Available Models")
    if models:
        for name in models.keys():
            st.write(f"- {name}")
    else:
        st.error("No models found! Please add trained `.h5` models in the `notebooks/best_model` or `notebooks/models` folders.")

# ========================================
# Model Comparison Page
# ========================================
elif page == "Model Comparison":
    st.title("Model Comparison Report")
    st.write("Comparison of all models based on **Training and Test Accuracy**.")

    # Table display
    st.dataframe(model_metrics, use_container_width=True)

    # Bar chart for visual comparison
    st.write("### Accuracy Comparison")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(model_metrics["Model"], model_metrics["Training Accuracy (%)"], label="Training Accuracy", alpha=0.7)
    ax.bar(model_metrics["Model"], model_metrics["Test Accuracy (%)"], label="Test Accuracy", alpha=0.7)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Model")
    ax.legend()
    ax.set_title("Training vs Test Accuracy")
    plt.xticks(rotation=15)
    st.pyplot(fig)

    # Best model info
    st.markdown("---")
    st.markdown(f"""
    ### Best Model
    **{best_model['Model']}**  
    - Training Accuracy: **{best_model['Training Accuracy (%)']}%**  
    - Test Accuracy: **{best_model['Test Accuracy (%)']}%**
    """)

    st.info("""
    **InceptionV3** achieved the best overall accuracy on the test set (99.34%), 
    showing exceptional generalization and robust feature extraction performance.
    """)

# ========================================
# Prediction Page
# ========================================
elif page == "Prediction":
    st.title("Fish Species Prediction")

    if not models:
        st.error("No models found! Please ensure `.h5` files exist in the model directories.")
    else:
        model_name = st.selectbox("Select a model for prediction", list(models.keys()))
        model = models[model_name]

        uploaded_file = st.file_uploader("Upload a fish image (JPG/PNG)", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, caption="Uploaded Image", width=300)

            if st.button("Predict"):
                with st.spinner("Predicting..."):
                    predicted_class, confidence = predict_image(model, img)
                st.success(f"Predicted Fish Species: **{predicted_class}**")
                st.write(f"Confidence: **{confidence*100:.2f}%**")

# ========================================
# About Page
# ========================================
elif page == "About":
    st.title("About This Project")
    st.write("""
    ### Project Overview
    The **Multiclass Fish Image Classification** project aims to automatically identify
    fish species from images using deep learning models trained on a dataset of 11 categories.

    ### Models Used
    - Custom CNN  
    - VGG16 (Transfer Learning)  
    - MobileNet (Transfer Learning)  
    - EfficientNetV2B0  
    - InceptionV3  
    - ResNet50  

    ### Performance Summary
    - **Best Model:** InceptionV3 (99.34% Test Accuracy)  
    - **Runner-Up:** MobileNet (99.18% Test Accuracy)

    ### Technology Stack
    - Python, TensorFlow/Keras
    - Streamlit for Dashboard
    - Matplotlib, NumPy, Pandas
    - Jupyter Notebook for Training

    ### Developed By
    **Abu Shakeer A W**  
    Data Science Enthusiast | Python | Deep Learning 
    """)

