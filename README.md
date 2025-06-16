"# Deep-Learning-Based-TB-Detection-System" 
# 🫁 Tuberculosis (TB) Detection from Chest X-ray using CNN and Flask

This project uses **Convolutional Neural Networks (CNNs)** to detect **Tuberculosis (TB)** from chest X-ray images. The trained deep learning model is integrated into a simple **web interface using Flask**, where users can upload X-ray images and receive predictions in real-time.

---

## 📌 Features

- 🧠 Deep learning model (CNN) trained to classify X-ray images as TB or Normal
- 🖼️ Image preprocessing with CLAHE (Contrast Limited Adaptive Histogram Equalization)
- 🌐 Web interface using Flask
- 📁 File upload support and result display
- 📊 Accuracy and validation tracking
- 🔒 Secure file handling

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Flask
- OpenCV & PIL for image processing
- HTML/CSS (frontend)
- NumPy, Matplotlib
- Scikit-learn

---

## 📂 Project Structure

TB_Detection_Project/
│
├── static/ # Stores uploaded X-ray images
├── templates/
│ ├── index.html # Upload page
│ └── result.html # Result page
├── tb_detection_model.h5 # Trained CNN model
├── app.py # Main Flask application
├── train_model.ipynb # Jupyter Notebook for model training
├── README.md # This file
└── requirements.txt # Python dependencies
