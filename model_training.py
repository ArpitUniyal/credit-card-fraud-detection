import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Create directories for saving models and data
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('archive (1)/creditcard.csv')

# Display basic information
print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\n{df['Class'].value_counts()}")
print(f"Fraud percentage: {df['Class'].mean() * 100:.4f}%")

# Check for missing values
print(f"Missing values: {df.isnull().sum().sum()}")

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"Dataset shape after removing duplicates: {df.shape}")

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Scale the Time and Amount features
scaler = StandardScaler()
X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

# Split the data into training and testing sets (stratified to maintain class distribution)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Train a Random Forest model with balanced class weights
print("Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Calculate PR-AUC
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall_curve, precision_curve)

print("\nModel Evaluation:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# Save the model and scaler
print("Saving model and preprocessor...")
joblib.dump(model, 'models/fraud_model.pkl')
joblib.dump(scaler, 'models/preprocessor.pkl')

# Create a new dataset for testing (without Class column)
print("Creating test dataset...")
test_data = X.iloc[:500].copy()  # Take first 500 rows
test_data.to_csv('testcreditcard.csv', index=False)
print(f"Test dataset created: testcreditcard.csv with {len(test_data)} rows")

print("Model training and evaluation completed successfully!")

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
plt.title('Top 15 Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')