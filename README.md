# Credit Card Fraud Detection

An end-to-end machine learning project that identifies potentially
fraudulent credit card transactions using **XGBoost** and provides an
interactive **Streamlit** dashboard for batch and single-transaction
predictions.

## Project Overview

Credit card fraud detection is an imbalanced binary-classification
problem: fraudulent transactions are very rare compared with legitimate
transactions. This project explores the data, trains and compares Random
Forest and XGBoost classifiers, tunes model hyperparameters, evaluates
fraud-detection performance, and uses the selected XGBoost model in a
Streamlit application.

The final dashboard supports: - Batch predictions from an uploaded CSV
file. - Single-transaction predictions through a form. - An adjustable
fraud-probability threshold. - Evaluation metrics and a confusion matrix
when the uploaded CSV includes actual `Class` labels. -
Feature-importance visualization. - Downloading prediction results as a
CSV file.

## Dataset

The project uses the Kaggle **Credit Card Fraud Detection** dataset,
containing 284,807 transactions and 30 input features plus the target
column `Class`.

-   `Class = 0`: legitimate transaction
-   `Class = 1`: fraudulent transaction
-   Input features: `Time`, `V1`--`V28`, and `Amount`
-   The `V1`--`V28` columns are anonymized PCA-transformed features.

The target is highly imbalanced: approximately 0.1727% of the original
transactions are fraudulent.

**Dataset note:** The dataset is not included in this repository.
Download it from Kaggle and place `creditcard.csv` in the `data/`
directory if you want to reproduce the notebook workflow. The Streamlit
dashboard does not need the original training CSV; it loads the saved
model artifacts from `models/`.

## Repository Structure

``` text
credit-card-fraud-detection/
├── data/
│   └── creditcard.csv                 # Download separately; not committed
├── models/
│   ├── xgboost.pkl                    # Trained XGBoost model
│   └── model_metadata.json            # Feature names and default threshold
├── notebooks/
│   └── fraud_detection.ipynb          # Data analysis, training and evaluation
├── outputs/
│   ├── plots/                         # Saved plots (if generated)
│   └── results/                       # Saved evaluation results (if generated)
├── app.py                             # Streamlit dashboard
├── requirements.txt
├── requirements_streamlit.txt
├── .gitignore
└── README.md
```

## Machine Learning Workflow

1.  Load the transaction dataset and inspect its structure.
2.  Check missing values, duplicates, and class distribution.
3.  Remove duplicate rows.
4.  Separate features (`X`) and target (`y`).
5.  Split the data into training and test sets using an 80/20 stratified
    split.
6.  Train baseline Random Forest and XGBoost classifiers.
7.  Use 3-fold `StratifiedKFold` cross-validation.
8.  Tune hyperparameters with `RandomizedSearchCV`, using Average
    Precision as the search metric.
9.  Compare models using precision, recall, F1-score, ROC-AUC, Average
    Precision, balanced accuracy, and confusion matrices.
10. Select XGBoost and explore classification thresholds to study the
    precision-recall trade-off.
11. Save the selected model and use it in the Streamlit dashboard.

## Model Selection and Results

XGBoost was selected because it achieved stronger recall, F1-score,
ROC-AUC, Average Precision, and balanced accuracy than the
baseline/tuned Random Forest comparison in this project.

### Tuned XGBoost test-set results

The following results were observed at a probability threshold of
**0.40**:

  Metric                Result
  ------------------- --------
  Accuracy              99.95%
  Precision             88.51%
  Recall                81.05%
  F1-score              84.62%
  ROC-AUC               97.71%
  Average Precision     81.95%

The threshold changes the predicted class labels, and therefore affects
precision, recall, F1-score, and the confusion matrix. ROC-AUC and
Average Precision are calculated from predicted probabilities and do not
depend on the selected classification threshold.

### Threshold selection caveat

Threshold 0.40 was explored after inspecting predictions on the held-out
test set. Therefore, the reported threshold-specific test metrics should
be treated as project results rather than an unbiased final estimate of
production performance. For a rigorous final evaluation, select the
threshold using a separate validation set and evaluate it once on a test
set that has not influenced model or threshold selection.

## Installation

### 1. Clone the repository

``` bash
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Create and activate a virtual environment

**Windows:**

``` bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**

``` bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

If you only need to run the dashboard, you can install:

``` bash
pip install -r requirements_streamlit.txt
```

## Run the Notebook

1.  Download the Kaggle dataset.
2.  Place `creditcard.csv` in the `data/` directory.
3.  Start Jupyter:

``` bash
jupyter notebook
```

4.  Open `notebooks/fraud_detection.ipynb` and run the cells.

## Run the Streamlit Dashboard

Make sure these files exist:

``` text
models/
├── xgboost.pkl
└── model_metadata.json
```

Start the dashboard from the repository root:

``` bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal, usually
`http://localhost:8501`.

## Using the Dashboard

### Batch prediction

1.  Open the **Batch prediction** tab.
2.  Upload a CSV containing the model's input features: `Time`,
    `V1`--`V28`, and `Amount`.
3.  The `Class` column is optional. If present, the app displays
    evaluation metrics and a confusion matrix; if absent, it performs
    predictions without claiming evaluation metrics.
4.  Review predicted fraud probabilities and labels.
5.  Download the results CSV.

### Single-transaction prediction

1.  Open the **Single transaction** tab.
2.  Enter the feature values.
3.  Click **Analyze transaction** to see the predicted fraud probability
    and class.

The dashboard's threshold slider controls how probabilities are
converted into class predictions. Lowering the threshold generally
increases the number of transactions flagged as fraud and may increase
recall at the cost of precision.

**Input compatibility:** Uploaded data must use the same feature
definitions and preprocessing as the training data. This project trains
on the provided numeric features without a separate scaler.

## Technologies Used

-   Python
-   Pandas and NumPy
-   Scikit-learn
-   XGBoost
-   Matplotlib and Seaborn
-   Jupyter Notebook
-   Streamlit
-   Joblib

