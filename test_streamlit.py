# test_streamlit.py
import joblib
import pandas as pd
import numpy as np

print("🔹 Testing Streamlit App Components...")

try:
    # Load models
    models = {
        "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
        "Random Forest": joblib.load("models/random_forest.pkl"),
        "XGBoost": joblib.load("models/xgboost.pkl"),
    }
    print("✅ Models loaded successfully")
except Exception as e:
    print("❌ Failed to load models:", e)

try:
    # Load scaler
    scaler = joblib.load("models/scaler.pkl")
    print("✅ Scaler loaded successfully")
except Exception as e:
    print("❌ Failed to load scaler:", e)

try:
    # Load encoders
    le_protocol = joblib.load("models/le_protocol_type.pkl")
    le_service = joblib.load("models/le_service.pkl")
    le_flag = joblib.load("models/le_flag.pkl")
    le_target = joblib.load("models/le_target.pkl")
    print("✅ Encoders loaded successfully")
except Exception as e:
    print("❌ Failed to load encoders:", e)

try:
    # Load training features
    training_features = joblib.load("models/training_features.pkl")
    print("✅ Training feature order loaded")
except Exception as e:
    print("❌ Failed to load training features:", e)

# Create a sample input with zeros / defaults
sample_input = pd.DataFrame([{col: 0 for col in training_features}])

# Replace categorical defaults with first class
sample_input["protocol_type"] = le_protocol.classes_[0]
sample_input["service"] = le_service.classes_[0]
sample_input["flag"] = le_flag.classes_[0]

# Encode categorical features
sample_input["protocol_type"] = le_protocol.transform(sample_input["protocol_type"])
sample_input["service"] = le_service.transform(sample_input["service"])
sample_input["flag"] = le_flag.transform(sample_input["flag"])

# Scale features
try:
    X_scaled = scaler.transform(sample_input)
    print("✅ Sample input scaled successfully")
except Exception as e:
    print("❌ Failed to scale input:", e)

# Make predictions
for name, model in models.items():
    try:
        pred_encoded = model.predict(X_scaled)[0]
        pred_label = le_target.inverse_transform([pred_encoded])[0]
        print(f"✅ {name} prediction successful: {pred_label}")
    except Exception as e:
        print(f"❌ {name} prediction failed:", e)

print("\n🎯 Test completed. If all ✅, your Streamlit app is functional!")
