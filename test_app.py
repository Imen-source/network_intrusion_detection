# test_app.py
import numpy as np
import pandas as pd
import joblib

# Load models and scaler
models = {
    "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
    "Random Forest": joblib.load("models/random_forest.pkl"),
    "XGBoost": joblib.load("models/xgboost.pkl")
}
scaler = joblib.load("models/scaler.pkl")

# Load label encoders
le_protocol = joblib.load("models/le_protocol.pkl")
le_service = joblib.load("models/le_service.pkl")
le_flag = joblib.load("models/le_flag.pkl")

# Dummy input (replace with any realistic sample)
sample = {
    "duration": 0,
    "protocol_type": "tcp",
    "service": "http",
    "flag": "SF",
    "src_bytes": 181,
    "dst_bytes": 5450,
    "land": 0,
    "wrong_fragment": 0,
    "urgent": 0,
    "hot": 0,
    "num_failed_logins": 0,
    "logged_in": 1,
    "num_compromised": 0,
    "root_shell": 0,
    "su_attempted": 0,
    "num_root": 0,
    "num_file_creations": 0,
    "num_shells": 0,
    "num_access_files": 0,
    "num_outbound_cmds": 0,
    "is_host_login": 0,
    "is_guest_login": 0,
    "count": 9,
    "srv_count": 9,
    "serror_rate": 0.0,
    "srv_serror_rate": 0.0,
    "rerror_rate": 0.0,
    "srv_rerror_rate": 0.0,
    "same_srv_rate": 1.0,
    "diff_srv_rate": 0.0,
    "srv_diff_host_rate": 0.0,
    "dst_host_count": 9,
    "dst_host_srv_count": 9,
    "dst_host_same_srv_rate": 1.0,
    "dst_host_diff_srv_rate": 0.0,
    "dst_host_same_src_port_rate": 1.0,
    "dst_host_srv_diff_host_rate": 0.0,
    "dst_host_serror_rate": 0.0,
    "dst_host_srv_serror_rate": 0.0,
    "dst_host_rerror_rate": 0.0,
    "dst_host_srv_rerror_rate": 0.0,
}

# Convert to DataFrame
df = pd.DataFrame([sample])

# Encode categorical features
df["protocol_type"] = le_protocol.transform(df["protocol_type"])
df["service"] = le_service.transform(df["service"])
df["flag"] = le_flag.transform(df["flag"])

# Scale features
X = scaler.transform(df)

# Predict using each model
for name, model in models.items():
    pred = model.predict(X)
    print(f"{name} prediction: {pred[0]}")
