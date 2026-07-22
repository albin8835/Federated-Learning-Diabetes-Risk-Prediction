# Federated Learning-Based Diabetes Risk Prediction System for Edge Devices

## Overview

This project presents a privacy-preserving diabetes risk prediction system using **Federated Learning**, **TensorFlow Lite**, and **Raspberry Pi 4**.

A lightweight Artificial Neural Network (ANN) is trained locally on multiple clients using the Flower Federated Learning framework. Instead of sharing patient data, only model parameters are exchanged with the server using the FedAvg algorithm, ensuring data privacy.

The trained global model is converted into TensorFlow Lite format and deployed on Raspberry Pi 4 for real-time diabetes risk prediction through a graphical user interface developed using CustomTkinter.

---

# Features

- Federated Learning using Flower Framework
- Privacy-preserving distributed training
- Lightweight ANN model
- FedAvg aggregation algorithm
- TensorFlow Lite conversion
- Raspberry Pi 4 deployment
- Modern CustomTkinter GUI
- Real-time diabetes risk prediction
- Health recommendations based on prediction

---

# Project Architecture

![Architecture](Images/architecture.png)

---

# Technologies Used

- Python
- TensorFlow
- TensorFlow Lite
- Flower Framework
- Scikit-learn
- Pandas
- NumPy
- Joblib
- CustomTkinter
- Raspberry Pi 4

---

# Dataset

**Dataset Name**

Diabetes Prediction Dataset

**Source**

https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

---

# Model Architecture

Input Layer

↓

Dense (64, ReLU)

↓

Batch Normalization

↓

Dropout (0.3)

↓

Dense (32, ReLU)

↓

Batch Normalization

↓

Dropout (0.2)

↓

Dense (16, ReLU)

↓

Output Layer (Sigmoid)

---

# Federated Learning Workflow

Dataset

↓

Data Preprocessing

↓

Client 1 Local Training

Client 2 Local Training

↓

Flower Server

↓

FedAvg Aggregation

↓

Global ANN Model

↓

TensorFlow Lite Conversion

↓

Raspberry Pi Deployment

↓

GUI Prediction

---

# Performance

| Metric | Value |
|---------|--------|
| Accuracy | 95.12% |
| Precision | 69.38% |
| Recall | 80.01% |
| F1 Score | 74.32% |
| AUC | 97.45% |

---

# Raspberry Pi Deployment

The final trained global model was converted into TensorFlow Lite format and deployed on Raspberry Pi 4 for efficient edge inference.

The Raspberry Pi application provides:

- Patient Information Input
- Diabetes Risk Score
- Risk Percentage
- Risk Level
- Health Recommendations

---

# Folder Structure

```
Training/
    server.py
    client.py
    model.py
    preprocess.py
    evaluate.py
    predict.py
    global_model.keras

RaspberryPi/
    predict_pi.py
    model.tflite
    scaler.pkl

Images/
    architecture.png
    gui.png
    results.png

requirements.txt
README.md
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Federated-Learning-Diabetes-Risk-Prediction.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Flower server

```bash
python server.py
```

Run client

```bash
python client.py
```

Run Raspberry Pi application

```bash
python predict_pi.py
```

---

# Future Improvements

- Mobile Application
- Cloud Deployment
- Multi-client Federated Learning
- Explainable AI
- IoT Sensor Integration

---

# Author

**Albin B**

Artificial Intelligence & Data Science

Amal Jyothi College of Engineering

---

# License

This project is developed for educational and research purposes.
