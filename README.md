# 💳 Credit Card Fraud Detection System

An end-to-end Machine Learning application that detects fraudulent credit card transactions using a **Random Forest Classifier**. The project includes a complete ML pipeline, an interactive **Streamlit** web application, and comprehensive model evaluation to identify fraudulent transactions in highly imbalanced financial data.

---

## 🚀 Features

- Detects fraudulent credit card transactions using Machine Learning.
- Interactive Streamlit web application for easy prediction.
- Batch prediction using uploaded CSV files.
- Single transaction analysis with manual input or randomly generated transaction.
- Adjustable fraud probability threshold.
- Fraud probability prediction for every transaction.
- Model evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix
- Feature Importance visualization.
- Download prediction results as CSV.

---

## 📂 Project Structure

```
Credit-Card-Fraud-Detection/
│
├── app.py                      # Streamlit application
├── model_training.py           # Model training script
├── creditcard.csv              # Original dataset
├── testcreditcard.csv          # Unseen test dataset
│
├── models/
│   ├── fraud_model.pkl         # Trained Random Forest model
│   └── preprocessor.pkl        # Saved StandardScaler
│
├── requirements.txt
├── README.md
```

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Joblib

---

## ⚙️ Machine Learning Pipeline

```
Dataset
   │
   ▼
Data Preprocessing
   │
   ▼
Train-Test Split
   │
   ▼
Feature Scaling
(StandardScaler)
   │
   ▼
Random Forest Classifier
   │
   ▼
Model Evaluation
   │
   ▼
Model Saved (.pkl)
   │
   ▼
Streamlit Deployment
```

---

## 📊 Dataset

The project uses the **Credit Card Fraud Detection Dataset**, which contains anonymized transaction features.

### Features

- Time
- V1 – V28 (PCA-transformed features)
- Amount

Target Variable

- Class
    - 0 → Legitimate Transaction
    - 1 → Fraudulent Transaction

---

## 🤖 Model

Algorithm Used:

- Random Forest Classifier

### Why Random Forest?

- Handles high-dimensional data effectively.
- Works well with imbalanced datasets using class balancing.
- Reduces overfitting through ensemble learning.
- Provides Feature Importance for model interpretation.

---

## 📈 Model Evaluation

The trained model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Precision-Recall AUC
- Confusion Matrix

These metrics provide a better evaluation for highly imbalanced fraud detection problems than accuracy alone.

---

## 🌐 Streamlit Application

The application supports two prediction modes.

### 1️⃣ Batch Prediction

- Upload transaction CSV file.
- Predict fraud for every transaction.
- View fraud probability.
- View evaluation metrics (if labels are available).
- Download prediction results.

### 2️⃣ Single Transaction Prediction

- Generate a random transaction.
- Enter feature values manually.
- Predict fraud probability.
- View prediction instantly.

---

## 📊 Visualizations

The application includes:

- Fraud Probability Distribution
- Fraud vs Legitimate Prediction Count
- Feature Importance Chart
- Confusion Matrix

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
```

Move into the project directory

```bash
cd credit-card-fraud-detection
```

Create virtual environment

```bash
python -m venv venv
```

Activate environment

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Train the Model

```bash
python model_training.py
```

This generates:

```
models/
    fraud_model.pkl
    preprocessor.pkl
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 👨‍💻 Author

**Arpit Uniyal**

B.Tech Computer Science Engineering


---
