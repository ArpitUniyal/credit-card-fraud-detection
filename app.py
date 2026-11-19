import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)

# -------------------- Page configuration --------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)

MODEL_PATH = Path("models/xgboost.pkl")
METADATA_PATH = Path("models/model_metadata.json")
DEFAULT_THRESHOLD = 0.40

st.title("💳 Credit Card Fraud Detection")
st.caption("XGBoost-powered transaction screening dashboard")

# -------------------- Load model and metadata --------------------
@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Save your trained XGBoost model first."
        )

    model = joblib.load(MODEL_PATH)

    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    feature_names = metadata.get("feature_names")
    if not feature_names:
        feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        raise ValueError(
            "Feature names are missing. Save feature_names in model_metadata.json."
        )

    threshold = float(metadata.get("threshold", DEFAULT_THRESHOLD))
    return model, list(feature_names), threshold


@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model)


try:
    model, FEATURE_NAMES, saved_threshold = load_artifacts()
except Exception as exc:
    st.error(str(exc))
    st.info(
        "In your training notebook, save the model and metadata using the code "
        "provided with this dashboard."
    )
    st.stop()

# -------------------- Sidebar settings --------------------
st.sidebar.header("Prediction settings")
threshold = st.sidebar.slider(
    "Fraud probability threshold",
    min_value=0.01,
    max_value=0.99,
    value=float(saved_threshold),
    step=0.01,
    help="Lower thresholds usually catch more fraud but can increase false alarms.",
)
st.sidebar.write(f"Active threshold: **{threshold:.2f}**")
st.sidebar.caption(
    "The saved default is 0.40. This is a configurable operating threshold, "
    "not a universal optimum."
)

# -------------------- Prediction helpers --------------------
def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and order input columns exactly as used during training."""
    missing = [col for col in FEATURE_NAMES if col not in data.columns]
    extra = [col for col in data.columns if col not in FEATURE_NAMES and col != "Class"]

    if missing:
        raise ValueError(f"CSV is missing required feature columns: {missing}")

    features = data[FEATURE_NAMES].copy()
    for col in FEATURE_NAMES:
        features[col] = pd.to_numeric(features[col], errors="coerce")

    if features.isna().any().any():
        bad_cols = features.columns[features.isna().any()].tolist()
        raise ValueError(f"Missing or non-numeric values found in: {bad_cols}")

    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Input contains infinite values.")

    return features


def predict_transactions(features: pd.DataFrame):
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return predictions, probabilities


def show_evaluation(actual, predicted, probabilities):
    st.subheader("Evaluation (actual Class labels supplied)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{accuracy_score(actual, predicted):.4f}")
    c2.metric("Precision", f"{precision_score(actual, predicted, zero_division=0):.4f}")
    c3.metric("Recall", f"{recall_score(actual, predicted, zero_division=0):.4f}")
    c4.metric("F1 score", f"{f1_score(actual, predicted, zero_division=0):.4f}")

    cm = confusion_matrix(actual, predicted, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    st.pyplot(fig)
    plt.close(fig)

    tn, fp, fn, tp = cm.ravel()
    a, b, c, d = st.columns(4)
    a.metric("True negatives", f"{tn:,}")
    b.metric("False positives", f"{fp:,}")
    c.metric("False negatives", f"{fn:,}")
    d.metric("True positives", f"{tp:,}")

    if len(np.unique(actual)) == 2:
        st.write(f"**ROC-AUC:** {roc_auc_score(actual, probabilities):.4f}")
        st.write(f"**Average precision:** {average_precision_score(actual, probabilities):.4f}")


# -------------------- Dashboard tabs --------------------
batch_tab, single_tab, info_tab = st.tabs(
    ["📁 Batch prediction", "🔎 Single transaction", "ℹ️ Model information"]
)

with batch_tab:
    st.subheader("Upload a transaction CSV")
    st.write(
        "Upload a CSV with the same feature columns used to train the model "
        "(Time, V1–V28, Amount for the standard Kaggle dataset). "
        "The `Class` column is optional and is used only for evaluation."
    )
    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            st.write("**Data preview**")
            st.dataframe(raw_df.head(10), use_container_width=True)

            features = prepare_features(raw_df)
            predictions, probabilities = predict_transactions(features)

            results = raw_df.copy()
            results["Fraud_Probability"] = probabilities
            results["Prediction"] = predictions
            results["Prediction_Label"] = np.where(
                predictions == 1, "Fraud", "Legitimate"
            )

            st.subheader("Prediction summary")
            total = len(results)
            flagged = int(predictions.sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("Transactions", f"{total:,}")
            m2.metric("Flagged as fraud", f"{flagged:,}")
            m3.metric("Flagged percentage", f"{(flagged / total * 100):.2f}%")

            st.subheader("Prediction results")
            st.dataframe(
                results.sort_values("Fraud_Probability", ascending=False),
                use_container_width=True,
            )

            if "Class" in raw_df.columns:
                actual = pd.to_numeric(raw_df["Class"], errors="coerce")
                if actual.isna().any() or not actual.isin([0, 1]).all():
                    st.warning("Class must contain only 0 and 1 for evaluation.")
                else:
                    show_evaluation(actual.astype(int), predictions, probabilities)

            st.subheader("Explain predictions with SHAP")
            st.caption(
                "Global explanations show how features influence predictions "
                "across a sample of uploaded transactions. SHAP values use the "
                "model's raw output (log-odds) by default."
            )
            show_global_shap = st.checkbox(
                "Generate SHAP global explanations for this batch",
                value=False,
                key="batch_global_shap",
            )
            if show_global_shap:
                with st.spinner("Calculating SHAP values for the uploaded transactions..."):
                    shap_sample = features.sample(
                        n=min(500, len(features)),
                        random_state=42,
                    )
                    batch_explainer = get_shap_explainer(model)
                    batch_shap_values = batch_explainer(shap_sample)

                st.write("**Global SHAP feature importance**")
                shap.plots.bar(batch_shap_values, max_display=15, show=False)
                st.pyplot(plt.gcf(), clear_figure=True)
                plt.close("all")

                st.write("**SHAP summary (beeswarm) plot**")
                shap.plots.beeswarm(batch_shap_values, max_display=15, show=False)
                st.pyplot(plt.gcf(), clear_figure=True)
                plt.close("all")


            csv_bytes = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download prediction results",
                data=csv_bytes,
                file_name="fraud_detection_predictions.csv",
                mime="text/csv",
            )

        except Exception as exc:
            st.error(f"Could not process this file: {exc}")

with single_tab:
    st.subheader("Analyze one transaction")
    st.caption(
        "Enter the transaction's feature values. V1–V28 are PCA-transformed "
        "features in the standard Kaggle dataset."
    )

    with st.form("single_transaction_form"):
        values = {}
        columns = st.columns(3)
        for idx, feature in enumerate(FEATURE_NAMES):
            with columns[idx % 3]:
                values[feature] = st.number_input(
                    feature,
                    value=0.0,
                    format="%.6f",
                    key=f"single_{feature}",
                )
        submitted = st.form_submit_button(
            "Analyze transaction", use_container_width=True
        )

    if submitted:
        one_row = pd.DataFrame([values], columns=FEATURE_NAMES)
        one_pred, one_prob = predict_transactions(one_row)
        is_fraud = bool(one_pred[0])

        col1, col2 = st.columns(2)
        col1.metric("Fraud probability", f"{one_prob[0]:.2%}")
        col2.metric("Prediction", "🚨 Fraudulent" if is_fraud else "✅ Legitimate")

        if is_fraud:
            st.error("This transaction is flagged for review.")
        else:
            st.success("This transaction is classified as legitimate.")

        st.subheader("Why did the model make this prediction?")
        st.caption(
            "The waterfall plot explains this transaction's feature contributions. "
            "Red bars push the raw model output towards fraud; blue bars push it "
            "away from fraud. These are log-odds contributions, not percentage points."
        )
        with st.spinner("Generating individual SHAP explanation..."):
            single_explainer = get_shap_explainer(model)
            single_shap_values = single_explainer(one_row)
            shap.plots.waterfall(single_shap_values[0], max_display=15, show=False)
            st.pyplot(plt.gcf(), clear_figure=True)
            plt.close("all")

        st.caption(
            "This is a model prediction, not a definitive determination. "
            "Real payment decisions should include additional controls and review."
        )

with info_tab:
    st.subheader("Model information")
    st.write("**Model:** XGBoost classifier")
    st.write("**Explainability:** SHAP (TreeSHAP) global and individual explanations")
    st.write(f"**Number of input features:** {len(FEATURE_NAMES)}")
    st.write(f"**Default saved threshold:** {saved_threshold:.2f}")
    st.write("**Input feature order:**")
    st.code(", ".join(FEATURE_NAMES))
    st.warning(
        "The app expects the same feature definitions and preprocessing used "
        "during model training. This app does not apply scaling. That matches "
        "a model trained directly on the unscaled dataset."
    )
