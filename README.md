# Credit Card Fraud Detection

An end-to-end machine learning project that identifies potentially fraudulent credit card transactions using **XGBoost** and provides an interactive **Streamlit** dashboard with **SHAP-based Explainable AI (XAI)** for batch and single-transaction predictions.

## 🚀 Live Demo

**Try the deployed application here:**

👉 [Credit Card Fraud Detection – Live App](https://credit-card-fraud-detection-du4qrhquskcolz5kyufmee.streamlit.app/)

## Project Overview

Credit card fraud detection is an imbalanced binary-classification problem: fraudulent transactions are very rare compared with legitimate transactions. This project explores the data, trains and compares Random Forest and XGBoost classifiers, tunes model hyperparameters, evaluates fraud-detection performance, and uses the selected XGBoost model in a Streamlit application.

The project also integrates **SHAP (SHapley Additive exPlanations)** to interpret model predictions and understand how individual features influence fraud detection.

The final dashboard supports:

- Batch predictions from an uploaded CSV file.
- Single-transaction predictions through a form.
- An adjustable fraud-probability threshold.
- Evaluation metrics and a confusion matrix when the uploaded CSV includes actual `Class` labels.
- SHAP global feature-importance visualization.
- SHAP beeswarm plots to understand feature impact and direction.
- SHAP waterfall plots to explain individual transaction predictions.
- Downloading prediction results as a CSV file.

## Dataset

The project uses the Kaggle **Credit Card Fraud Detection** dataset, containing 284,807 transactions and 30 input features plus the target column `Class`.

- `Class = 0`: legitimate transaction
- `Class = 1`: fraudulent transaction
- Input features: `Time`, `V1`–`V28`, and `Amount`
- The `V1`–`V28` columns are anonymized PCA-transformed features.

The target is highly imbalanced: approximately 0.1727% of the original transactions are fraudulent.

**Dataset note:** The dataset is not included in this repository. Download it from Kaggle and place `creditcard.csv` in the `data/` directory if you want to reproduce the notebook workflow. The Streamlit dashboard does not need the original training CSV; it loads the saved model artifacts from `models/`.

## Repository Structure

```text
credit-card-fraud-detection/
├── data/
│   └── creditcard.csv                 # Download separately; not committed
├── models/
│   ├── xgboost.pkl                    # Trained XGBoost model
│   └── model_metadata.json            # Feature names and default threshold
├── notebooks/
│   └── fraud_detection.ipynb          # EDA, training, evaluation and SHAP
├── outputs/
│   ├── plots/                         # Saved plots (if generated)
│   └── results/                       # Saved evaluation results (if generated)
├── app.py                             # Streamlit dashboard with SHAP
├── requirements.txt                   # Project dependencies
├── .gitignore
└── README.md
```

## Machine Learning Workflow

1. Load the transaction dataset and inspect its structure.
2. Check missing values, duplicates, and class distribution.
3. Remove duplicate rows.
4. Separate features (`X`) and target (`y`).
5. Split the data into training and test sets using an 80/20 stratified split.
6. Train baseline Random Forest and XGBoost classifiers.
7. Use 3-fold `StratifiedKFold` cross-validation.
8. Tune hyperparameters with `RandomizedSearchCV`, using Average Precision as the search metric.
9. Compare models using precision, recall, F1-score, ROC-AUC, Average Precision, balanced accuracy, and confusion matrices.
10. Select XGBoost and explore classification thresholds to study the precision-recall trade-off.
11. Apply SHAP TreeExplainer to interpret the trained XGBoost model.
12. Generate SHAP global feature-importance, beeswarm, and individual waterfall plots.
13. Save the selected model and integrate prediction and SHAP explanations into the Streamlit dashboard.

## Model Selection and Results

XGBoost was selected because it achieved stronger recall, F1-score, ROC-AUC, Average Precision, and balanced accuracy than the baseline/tuned Random Forest comparison in this project.

### Tuned XGBoost test-set results

The following results were observed at a probability threshold of **0.40**:

| Metric | Result |
|---|---:|
| Accuracy | 99.95% |
| Precision | 88.51% |
| Recall | 81.05% |
| F1-score | 84.62% |
| ROC-AUC | 97.71% |
| Average Precision | 81.95% |

The threshold changes the predicted class labels, and therefore affects precision, recall, F1-score, and the confusion matrix. ROC-AUC and Average Precision are calculated from predicted probabilities and do not depend on the selected classification threshold.

### Threshold selection caveat

Threshold 0.40 was explored after inspecting predictions on the held-out test set. Therefore, the reported threshold-specific test metrics should be treated as project results rather than an unbiased final estimate of production performance. For a rigorous final evaluation, select the threshold using a separate validation set and evaluate it once on a test set that has not influenced model or threshold selection.

## Explainable AI with SHAP

This project uses **SHAP (SHapley Additive exPlanations)** to interpret the predictions made by the trained XGBoost classifier.

SHAP is an Explainable AI technique based on Shapley values from cooperative game theory. It estimates how much each feature contributes to a model's prediction.

The project uses **TreeSHAP**, which is designed for tree-based models such as XGBoost.

### SHAP Visualizations

**1. Global Feature Importance (SHAP Bar Plot)**

- Displays the mean absolute SHAP value of each feature.
- Identifies the features that have the greatest average influence on model predictions.
- Helps understand which features the model relies on most.

**2. SHAP Beeswarm Plot**

- Displays feature contributions across multiple transactions.
- Shows how high and low feature values influence model output.
- Positive SHAP values push predictions towards fraud, while negative values push predictions away from fraud.

**3. SHAP Waterfall Plot**

- Explains an individual transaction prediction.
- Displays the base value and the contributions of individual features.
- Red bars represent positive contributions towards the fraud class.
- Blue bars represent negative contributions away from the fraud class.

### SHAP Integration in Streamlit

The dashboard includes SHAP-based explanations in two areas:

- **Batch Prediction:** Generates global SHAP feature-importance and beeswarm plots using a sample of uploaded transactions.
- **Single Transaction:** Generates an individual SHAP waterfall plot to explain why the model classified a transaction as fraudulent or legitimate.

SHAP explanations describe the model's learned behaviour; they do not establish causal relationships or prove that a transaction is fraudulent.

**Note:** The model's SHAP TreeExplainer uses raw output (log-odds) by default. Therefore, SHAP values represent contributions in the model's raw output space, not direct percentage changes in fraud probability.

Since `V1`–`V28` are PCA-transformed features, SHAP explains their influence on predictions but does not recover their original financial meaning.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ArpitUniyal/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Create and activate a virtual environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The requirements file includes SHAP along with the machine learning, data processing, visualization, and dashboard libraries.

## Run the Notebook

1. Download the Kaggle dataset.
2. Place `creditcard.csv` in the `data/` directory.
3. Install the dependencies.
4. Start Jupyter:

```bash
jupyter notebook
```

5. Open `notebooks/fraud_detection.ipynb` and run the cells to reproduce data analysis, model training, evaluation, and SHAP explanations.

## Run the Streamlit Dashboard

Make sure these files exist:

```text
models/
├── xgboost.pkl
└── model_metadata.json
```

Start the dashboard from the repository root:

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal, usually `http://localhost:8501`.

## Using the Dashboard

### Batch Prediction

1. Open the **Batch prediction** tab.
2. Upload a CSV containing the model's input features: `Time`, `V1`–`V28`, and `Amount`.
3. The `Class` column is optional. If present, the app displays evaluation metrics and a confusion matrix; if absent, it performs predictions without claiming evaluation metrics.
4. Review predicted fraud probabilities and labels.
5. Optionally generate SHAP global feature-importance and beeswarm plots to understand feature contributions across the uploaded transactions.
6. Download the prediction results CSV.

### Single-Transaction Prediction

1. Open the **Single transaction** tab.
2. Enter the feature values.
3. Click **Analyze transaction** to see the predicted fraud probability and class.
4. Review the SHAP waterfall plot to understand which features contributed towards or away from the prediction.

The dashboard's threshold slider controls how probabilities are converted into class predictions. Lowering the threshold generally increases the number of transactions flagged as fraud and may increase recall at the cost of precision.

**Input compatibility:** Uploaded data must use the same feature definitions and preprocessing as the training data. This project trains on the provided numeric features without a separate scaler.

## Technologies Used

- **Programming:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost
- **Explainable AI:** SHAP (TreeSHAP)
- **Visualization:** Matplotlib, Seaborn, SHAP
- **Development:** Jupyter Notebook
- **Dashboard:** Streamlit
- **Model Persistence:** Joblib

## Future Improvements

- Add SHAP explanations for more transaction-level analysis.
- Evaluate the model using a separate validation set for threshold selection.
- Explore additional fraud-detection strategies and model calibration.
- Improve interpretability of transaction patterns using domain-specific features.

## Author

[Arpit Uniyal](https://github.com/ArpitUniyal)

## GitHub Repository

[Credit Card Fraud Detection – Source Code](https://github.com/ArpitUniyal/credit-card-fraud-detection)
