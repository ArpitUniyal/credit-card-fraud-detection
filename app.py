import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Function to load model and scaler
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/fraud_model.pkl')
        scaler = joblib.load('models/preprocessor.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please run model_training.py first.")
        return None, None

# Function to make predictions
def predict(data, model, scaler,threshold):
    # Ensure Time and Amount are scaled
    if 'Time' in data.columns and 'Amount' in data.columns:
        data[['Time', 'Amount']] = scaler.transform(data[['Time', 'Amount']])
    
    # Make predictions
    probs = model.predict_proba(data)[:, 1]
    predictions = (probs >= threshold).astype(int)
    
    return predictions, probs

# Function to generate random transaction
def generate_random_transaction(df_sample):
    # Get min and max values for each feature
    min_vals = df_sample.min()
    max_vals = df_sample.max()
    
    # Generate random values within the range
    random_transaction = {}
    for col in df_sample.columns:
        random_transaction[col] = np.random.uniform(min_vals[col], max_vals[col])
    
    return pd.DataFrame([random_transaction])

# Main function
def main():
    st.title("Credit Card Fraud Detection System")
    st.write("This application helps detect fraudulent credit card transactions.")
    
    # Load model and scaler
    model, scaler = load_model()
    
    if model is None or scaler is None:
        st.warning("Please run the model_training.py script first to train and save the model.")
        if st.button("Show Instructions"):
            st.code("python model_training.py")
        return
    
    # Threshold selection (business-aware prediction)
    st.sidebar.header("Prediction Settings")
    threshold = st.sidebar.slider(
        "Fraud Probability Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.05
    )


    # Create tabs for different modes
    tab1, tab2 = st.tabs(["Batch Prediction (Upload CSV)", "Single Transaction Prediction"])
    
    # Store active tab in session state
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0

    # Mode: Upload CSV File
    with tab1:
        st.header("Upload Transactions File")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="uploader")

        if uploaded_file is not None:
            # Load data
            df = pd.read_csv(uploaded_file)

            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(df.head())

            # Check if required columns exist
            required_cols = ['Time', 'Amount', 'V1', 'V2', 'V3', 'V4']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                # Make predictions
                predictions, probabilities = predict(df, model, scaler,threshold)

                # Add predictions to dataframe
                results = df.copy()
                results['Fraud_Probability'] = probabilities
                results['Prediction'] = predictions

                # Display results
                st.subheader("Prediction Results")

                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Transactions", len(results))
                with col2:
                    st.metric("Flagged as Fraud", results['Prediction'].sum())
                with col3:
                    st.metric("Fraud Percentage", f"{results['Prediction'].mean()*100:.2f}%")

                # Show top suspicious transactions
                st.subheader("Top Suspicious Transactions")
                suspicious = results.sort_values('Fraud_Probability', ascending=False).head(10)
                st.dataframe(suspicious)

                # Visualizations
                st.subheader("Visualizations")

                col1, col2 = st.columns(2)

                with col1:
                    # Fraud probability distribution
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(results['Fraud_Probability'], bins=50, kde=True)
                    plt.title('Fraud Probability Distribution')
                    plt.xlabel('Fraud Probability')
                    plt.ylabel('Count')
                    st.pyplot(fig)

                with col2:
                    # Prediction counts
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.countplot(x='Prediction', data=results)
                    plt.title('Prediction Counts')
                    plt.xlabel('Prediction (1=Fraud, 0=Normal)')
                    plt.ylabel('Count')
                    for p in ax.patches:
                        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width()/2., p.get_height()),
                                    ha='center', va='bottom')
                    st.pyplot(fig)

                # Feature importance
                st.subheader("Feature Importance")
                feature_importance = pd.DataFrame({
                    'Feature': df.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)

                fig, ax = plt.subplots(figsize=(12, 8))
                sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
                plt.title('Top 15 Feature Importance')
                plt.tight_layout()
                st.pyplot(fig)

                # Download results
                csv = results.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="fraud_detection_results.csv",
                    mime="text/csv"
                )
    
    # Mode: Single Transaction
    with tab2:
        st.header("Single Transaction Analysis")

        # Load sample data for feature ranges
        try:
            sample_data = pd.read_csv('testcreditcard.csv')
        except FileNotFoundError:
            st.error("Sample data not found. Please run model_training.py first.")
            return

        # Option to generate random transaction (persist mode on click)
        if st.button("Generate Random Transaction", key="generate_random_btn"):
            transaction = generate_random_transaction(sample_data)
            st.session_state.transaction = transaction
            # Set active tab to Single Transaction
            st.session_state.active_tab = 1

        # Create input form
        st.subheader("Transaction Details")

        # Initialize transaction in session state if not exists
        if 'transaction' not in st.session_state:
            # Default values (all zeros)
            default_transaction = {col: 0.0 for col in sample_data.columns}
            st.session_state.transaction = pd.DataFrame([default_transaction])

        # Create a form for manual input
        with st.form("transaction_form"):
            # Create columns for better layout
            cols = st.columns(3)

            # Create input fields for each feature
            updated_values = {}
            for i, col in enumerate(sample_data.columns):
                col_idx = i % 3
                with cols[col_idx]:
                    current_value = st.session_state.transaction[col].values[0]
                    updated_values[col] = st.number_input(
                        f"{col}",
                        value=float(current_value),
                        format="%.6f",
                        key=f"num_{col}"
                    )

            # Submit button
            submitted = st.form_submit_button("Analyze Transaction", use_container_width=True)

            if submitted:
                # Update transaction with form values
                st.session_state.transaction = pd.DataFrame([updated_values])
                # Set active tab to Single Transaction
                st.session_state.active_tab = 1

        # Display current transaction
        st.subheader("Current Transaction Values")
        st.dataframe(st.session_state.transaction)

        # Analyze button outside the form
        if st.button("Analyze Current Transaction", key="analyze_current_btn"):
            # Set active tab to Single Transaction
            st.session_state.active_tab = 1
            # Make prediction
            prediction, probability = predict(st.session_state.transaction, model, scaler,threshold)

            # Display result
            st.subheader("Analysis Result")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Fraud Probability", f"{probability[0]:.4f}")
            with col2:
                result = "Fraudulent" if prediction[0] == 1 else "Legitimate"
                st.metric("Prediction", result)
            


if __name__ == "__main__":
    main()
