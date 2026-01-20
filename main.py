# main.py - Final Enhanced Version

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# -----------------------------
# Setup
# -----------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# -----------------------------
# Load dataset
# -----------------------------
cols = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty"
]

df = pd.read_csv("data/KDDTrain+.txt", names=cols)

# -----------------------------
# Encode categoricals
# -----------------------------
categorical_cols = ["protocol_type", "service", "flag"]
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    joblib.dump(le, f"models/le_{col}.pkl")

X = df.drop(columns=["label", "difficulty"])
y = df["label"]

# -----------------------------
# SMOTE
# -----------------------------
print("Original:", Counter(y))
min_class_size = min(Counter(y).values())
k = min(5, min_class_size - 1)

if k >= 1:
    smote = SMOTE(random_state=42, k_neighbors=k)
    X_res, y_res = smote.fit_resample(X, y)
else:
    X_res, y_res = X, y

# Encode target
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y_res)
joblib.dump(le_target, "models/le_target.pkl")

# -----------------------------
# Sample
# -----------------------------
sample_size = min(100_000, len(X_res))
idx = np.random.choice(len(X_res), sample_size, replace=False)
X_res = X_res.iloc[idx]
y_encoded = y_encoded[idx]

# -----------------------------
# Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# -----------------------------
# Scale
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "models/scaler.pkl")

# Save feature order
training_features = X.columns.tolist()
joblib.dump(training_features, "models/training_features.pkl")

# -----------------------------
# Train models
# -----------------------------
models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=200),
    "xgboost": XGBClassifier(eval_metric="mlogloss")
}

for name, model in models.items():
    print(f"Training {name}")
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, f"models/{name}.pkl")

# -----------------------------
# Feature importance (top 2)
# -----------------------------
rf = models["random_forest"]
importances = pd.Series(rf.feature_importances_, index=training_features)
top2 = importances.sort_values(ascending=False).head(2).index.tolist()

joblib.dump(top2, "models/top_features.pkl")

print("Top 2 important features:", top2)

print("✅ Training complete. All assets saved.")
