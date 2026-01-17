Credit Card Fraud Detection System 💳🚨

🔗 Live Demo:
https://credit-card-fraud-detection-4apfrgcdumarksxeiayxvw.streamlit.app/

📌 Project Overview

This project is an end-to-end Machine Learning–based Credit Card Fraud Detection System designed to identify fraudulent transactions with high reliability.
It uses a Random Forest classifier trained on historical transaction data and is deployed as an interactive Streamlit web application that can be accessed from any network using a single public link.

The system focuses on probability-based fraud detection and allows dynamic adjustment of the fraud decision threshold to balance recall and false positives, which is critical in real-world financial systems.

🎯 Key Objectives

Detect fraudulent credit card transactions accurately

Handle highly imbalanced datasets

Provide probability-based predictions instead of hard labels

Allow threshold tuning for business risk control

Deploy a pre-trained model for real-time inference (no retraining)

Make the system publicly accessible via a single URL

🛠️ What We Have Done

Trained a Random Forest classifier for fraud detection

Handled class imbalance using appropriate model settings

Scaled sensitive numerical features (Time, Amount)

Saved the trained model and preprocessor using joblib

Built an interactive Streamlit UI with:

Batch prediction via CSV upload

Single transaction prediction

Fraud probability threshold slider

Implemented inference-only deployment (no training during runtime)

Resolved real-world deployment issues related to:

Python version compatibility

scikit-learn model serialization

Binary wheel vs source builds

Successfully deployed the application on Streamlit Community Cloud

⚙️ Technologies & Tools Used
Programming & Libraries

Python

NumPy

Pandas

scikit-learn

Joblib

Matplotlib

Seaborn

Machine Learning

Random Forest Classifier

Probability-based prediction (predict_proba)

Threshold-based decisioning

Web & Deployment

Streamlit

Streamlit Community Cloud

GitHub for version control

🧠 How the System Works

The trained model predicts the probability of fraud for each transaction.

A fraud probability threshold is applied:

If probability ≥ threshold → Fraud

Else → Legitimate

Users can adjust the threshold to control:

Higher recall (catch more fraud)

Lower false positives

The app supports:

Batch prediction (CSV upload)

Single transaction prediction

📂 Project Structure
credit-card-fraud-detection/
│
├── models/
│   ├── fraud_model.pkl
│   └── preprocessor.pkl
│
├── app.py
├── model_training.py
├── requirements.txt
├── testcreditcard.csv
├── feature_importance.png
└── README.md
