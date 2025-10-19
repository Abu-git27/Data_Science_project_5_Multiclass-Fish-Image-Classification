## 🐠 Multiclass Fish Image Classification

Automated fish species recognition using deep learning
Author: Abu Shakeer A W

---

### 🔍 Project Overview

This project focuses on building a robust deep learning system to classify multiple fish species from images. It leverages convolutional neural networks (CNNs) and transfer learning to achieve high accuracy and deploys a user-friendly web interface for real-time predictions.

The project demonstrates end-to-end AI workflow: from data preprocessing and model training to deployment, evaluation, and interactive visualization.

---

### ⚡ Features

- Supports multiple CNN architectures: Custom CNN, VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetV2B0.

- Data augmentation for improved generalization.

- Transfer learning using pre-trained ImageNet models.

- Top-3 prediction visualization with probability scores.

- Streamlit dashboard for real-time inference and CSV download.

- Modular and extensible for future datasets or projects.

---

### 📂 Project Structure

Multiclass_Fish_Image_Classification/
│
├── data/                 # Train, validation, and test datasets organized by class
├── models/               # Trained model files (.h5 / .keras)
├── notebooks/            # Jupyter notebooks for exploration & training
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb
│   └── evaluation.ipynb
├── Custom_CNN/           # Custom CNN model scripts
├── PREDICT.PY            # Prediction script for single images
├── Class_Label.json      # Mapping of class indices to fish names
├── app.py                # Streamlit app for interactive predictions
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation

---

### 🧠 Dataset & Preprocessing

- Train/Validation/Test Split: Ensures proper evaluation.

- Augmentation: Rotation, flip, zoom, and brightness adjustments to increase dataset diversity.

- Normalization: Rescaled pixel values (1/255) for consistent input distribution.

- Label Mapping: JSON file maps numeric indices to class names for easy inference.

---

### 🖥 Model Training

- Custom CNN: Baseline architecture to understand dataset patterns.

- Pre-trained Models: VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetV2B0.

- Techniques Used: Dropout, batch normalization, fine-tuning last layers, early stopping.

- Evaluation Metrics: Accuracy, loss, and confusion matrix.

| Model             | Training Accuracy (%) | Test Accuracy (%) | Notes                             |
|------------------|--------------------|-----------------|-----------------------------------|
| Custom CNN        | 92.12              | 92.47           | Baseline                          |
| VGG16             | 94.87              | 97.05           | High accuracy, heavier model      |
| MobileNet         | 98.90              | 99.18           | Best model: lightweight & fast |
| EfficientNetV2B0  | 17.12              | 16.32           | Overfitting / low accuracy        |
| InceptionV3       | 98.81              | 99.34           | Slightly heavier than MobileNet   |
| ResNet50          | 69.23              | 69.72           | Gradient instability              |

---
 
### 📊 Evaluation & Visualization

- Plotted training vs validation curves for all models.

- Confusion matrices to understand misclassifications.

- Implemented Top-3 predictions in the Streamlit app for better interpretability.

- Performance comparison guided MobileNet selection as production-ready.

---


### 🌐 Streamlit Web Application

Features:

- Upload images and get instant class predictions.

- Select any trained model for prediction.

- Bar chart visualization for Top-3 predictions.

- Download CSV results of predictions.

- Interactive and mobile-friendly dashboard for users.

---

## 💡 Applications

- Marine Research: Automated species recognition for biodiversity studies.

- Seafood Industry: Sorting and quality control.

- Edge AI Deployment: MobileNet model suitable for embedded systems and cameras.

- General AI Framework: Can be extended to other image classification datasets.

---

### 🛠 Technology Stack

Programming: Python 3

Deep Learning: TensorFlow, Keras, CNNs, Transfer Learning

Data Handling: NumPy, Pandas, Pillow, ImageDataGenerator

Visualization: Matplotlib, Seaborn, Altair

Deployment: Streamlit, JSON, CSV export

Environment: Virtualenv (tf_env), requirements.txt

---

### 📌 Key Takeaways

- Practical experience with CNNs and transfer learning.

- Built a modular deep learning pipeline from data to deployment.

- Learned to evaluate, compare, and select models for production.

- Developed a real-time interactive application for end-users.

---

📘 Author: Abu Shakeer A W
🔗 Ideal for AI research, marine species classification, and deployment-ready computer vision projects