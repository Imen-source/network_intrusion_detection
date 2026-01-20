import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

st.set_page_config("Network Intrusion Detection", layout="wide")
st.title("🛡️ Network Intrusion Detection System")

# -----------------------------
# Load assets
# -----------------------------
@st.cache_resource
def load_assets():
    models = {
        "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
        "Random Forest": joblib.load("models/random_forest.pkl"),
        "XGBoost": joblib.load("models/xgboost.pkl"),
    }

    scaler = joblib.load("models/scaler.pkl")
    features = joblib.load("models/training_features.pkl")
    top_features = joblib.load("models/top_features.pkl")

    le_protocol = joblib.load("models/le_protocol_type.pkl")
    le_service = joblib.load("models/le_service.pkl")
    le_flag = joblib.load("models/le_flag.pkl")
    le_target = joblib.load("models/le_target.pkl")

    return models, scaler, features, top_features, le_protocol, le_service, le_flag, le_target


models, scaler, FEATURES, TOP2, le_protocol, le_service, le_flag, le_target = load_assets()

# -----------------------------
# Mode selection
# -----------------------------
binary_mode = st.checkbox("Binary Mode (Attack vs Normal)")

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Connection Features")

inputs = {f: 0 for f in FEATURES}

inputs["protocol_type"] = st.sidebar.selectbox("Protocol", le_protocol.classes_)
inputs["service"] = st.sidebar.selectbox("Service", le_service.classes_)
inputs["flag"] = st.sidebar.selectbox("Flag", le_flag.classes_)

for f in FEATURES:
    if f not in ["protocol_type", "service", "flag"]:
        inputs[f] = st.sidebar.number_input(f, value=0.0)

# -----------------------------
# Prediction function
# -----------------------------
def preprocess(df):
    df["protocol_type"] = le_protocol.transform(df["protocol_type"])
    df["service"] = le_service.transform(df["service"])
    df["flag"] = le_flag.transform(df["flag"])
    df = df[FEATURES]
    return scaler.transform(df)

# -----------------------------
# Single prediction
# -----------------------------
if st.button("🚀 Predict"):

    df = pd.DataFrame([inputs])
    X_scaled = preprocess(df)

    st.subheader("Predictions")

    for name, model in models.items():
        probs = model.predict_proba(X_scaled)[0]
        idx = np.argmax(probs)
        label = le_target.inverse_transform([idx])[0]
        conf = probs[idx] * 100

        if binary_mode:
            label = "Normal" if label == "normal" else "Attack"

        st.write(f"**{name}:** {label} — {conf:.2f}%")

# -----------------------------
# Batch CSV prediction
# -----------------------------
st.subheader("📂 Batch Prediction")

file = st.file_uploader("Upload CSV", type="csv")

if file:
    df = pd.read_csv(file)
    X_scaled = preprocess(df)

    model = models["Random Forest"]
    preds = model.predict(X_scaled)
    labels = le_target.inverse_transform(preds)

    if binary_mode:
        labels = ["Normal" if l == "normal" else "Attack" for l in labels]

    df["Prediction"] = labels
    st.dataframe(df)

# -----------------------------
# Feature importance display
# -----------------------------
st.subheader("🔥 Top Important Features")

for f in TOP2:
    st.write(f"• {f}")

# -----------------------------
# SHAP explanation
# -----------------------------
st.subheader("🧠 Explain Prediction (SHAP)")

if st.button("Explain with SHAP"):
    model = models["Random Forest"]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.summary_plot(shap_values, X_scaled, feature_names=FEATURES, plot_type="bar")
    st.pyplot()

st.markdown("---")
st.markdown("Built with ❤️ – AI Powered Network Security System")
