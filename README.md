# 🛡️ FL-DIAPRED: A Federated Learning Framework for Privacy-Preserving Diabetes Risk Prediction Using Artificial Neural Networks

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow 2.13](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)](https://tensorflow.org/)
[![Flower 1.4](https://img.shields.io/badge/Flower-1.4%20(FedAvg)-pink.svg)](https://flower.dev/)
[![Hardware](https://img.shields.io/badge/Target-Raspberry%20Pi%204%20%7C%20Edge%20AI-red.svg)](https://www.raspberrypi.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, privacy-preserving clinical decision support system that predicts diabetes risk using **Federated Learning (Flower FedAvg)**, **Artificial Neural Networks**, and **Explainable AI (XAI)**, deployed on **Edge Hardware (Raspberry Pi 4 / TensorFlow Lite)** with an interactive clinical desktop dashboard.

---

## 📌 The Problem
Healthcare artificial intelligence requires massive, diverse datasets to detect chronic conditions accurately. However, strict data governance regulations (**HIPAA, GDPR**) strictly forbid medical institutions from sharing or centralizing private patient health information (PHI). 

## 💡 The Solution
**FL-DIAPRED** addresses this challenge by decentralizing model training. Instead of aggregating sensitive patient records onto a central cloud server:
1. Patient data remains strictly on local hospital nodes.
2. Only model weights are shared and aggregated using the **Federated Averaging (FedAvg)** algorithm.
3. The resulting global neural network is compressed via **TensorFlow Lite** to run inference locally on edge devices (Raspberry Pi 4) without requiring internet connectivity.

---

## 🚀 Key Features (Version 2.0)

- **🔒 Decentralized Federated Learning**: Multi-client collaborative training orchestrated via Flower (`flwr`) preserving 100% patient data confidentiality.
- **📊 Comparative Benchmarking Suite**: Proves empirically that collaborative federated training outperforms isolated local hospital models.
- **💡 Explainable AI (XAI)**: Feature attribution breakdown providing percentage contributions for patient vitals (e.g. Blood Glucose, HbA1c, BMI, Age).
- **📱 Edge Deployment (Raspberry Pi 4)**: Quantized into Float32 TensorFlow Lite (`.tflite`) for sub-10ms inference on resource-constrained hardware.
- **🩺 Modern Clinical Dashboard**: High-tech medical SaaS desktop interface built with CustomTkinter featuring real-time risk gauges and 1-click patient demo presets.
- **📄 Clinical PDF Report Export**: Generates 1-page clinical reports complete with patient vitals, risk level, XAI factor breakdown, and lifestyle advice.
- **🏥 Realistic Hospital Skew Simulation**: Simulates heterogeneous hospital demographics using a Dirichlet distribution ($\alpha = 0.5$).

---

## 📈 Benchmark Results

Evaluated on 19,230 holdout clinical test records:

| Model Paradigm | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Privacy Guarantee |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Federated FedAvg (FL-DIAPRED)** | **94.92%** | **67.72%** | **81.13%** | **0.7382** | **0.9750** | **100% Data Confidentiality** |
| Centralized Baseline (Pooled Data) | 94.15% | 63.19% | 80.66% | 0.7086 | 0.9727 | ❌ None (Data Centralized) |
| Isolated Local Models (Average) | 93.97% | 62.84% | 78.14% | 0.6961 | 0.9675 | Local Only (Sub-optimal Recall) |

> **Key Takeaway**: Federated Learning achieves a **+3.0% higher recall** than isolated local hospital training, demonstrating superior screening sensitivity without compromising patient confidentiality.

---

## 🏗️ System Architecture

```
[ Hospital 1 (Local Data) ] ---\
[ Hospital 2 (Local Data) ] ----\   (Model Weights Only)
[ Hospital 3 (Local Data) ] ------> [ Flower Server (FedAvg) ] ---> [ Global ANN Model ]
[ Hospital 4 (Local Data) ] ---/                                          |
                                                               (TFLite Converter)
                                                                          v
[ Clinical PDF Report ] <--- [ CustomTkinter Dashboard ] <--- [ Raspberry Pi 4 Edge ]
```

---

## 📂 Project Directory Structure

```
├── RaspberryPi/
│   └── reports/
│       ├── predict_pi.py           # Modern Clinical Desktop Dashboard
│       ├── model.tflite             # Edge TensorFlow Lite Model
│       ├── scaler.pkl               # Standard Scaler for inference
│       └── generated_reports/       # Exported Clinical PDF Reports
├── Training/
│   ├── benchmark.py                 # Centralized vs Local vs Federated Benchmark
│   ├── explain.py                   # Explainable AI (XAI) Attribution Module
│   ├── partition_data.py            # Dirichlet Non-IID Hospital Data Simulator
│   ├── server.py                    # Flower Federated Learning Server
│   ├── client.py                    # Flower Federated Learning Client Node
│   ├── model.py                     # Artificial Neural Network Architecture
│   ├── evaluate.py                  # Evaluation on holdout test data
│   ├── convert_tflite.py            # Keras to TFLite Float32 Converter
│   └── diabetes_prediction_dataset.csv
├── requirements.txt
└── README.md
```

---

## ⚡ Quickstart & Installation

### 1. Clone & Setup Environment
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install -r requirements.txt
```

### 2. Launch the Clinical Desktop GUI
```bash
cd RaspberryPi/reports
python predict_pi.py
```
*Click **"🔴 High Risk Patient"** or **"🟢 Healthy Baseline"** to test live AI inference and click **"📄 Export Clinical Report (PDF)"** to generate a patient report.*

### 3. Run Benchmarking Evaluation
```bash
cd Training
python benchmark.py
```

### 4. Run Federated Training
In Terminal 1 (Start Server):
```bash
cd Training
python server.py
```
In Terminals 2 to 5 (Start 4 Clients):
```bash
python client.py 1
python client.py 2
python client.py 3
python client.py 4
```

---

## 🛡️ License
This project is licensed under the MIT License.
