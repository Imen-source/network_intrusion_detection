# 🛡️ Network Intrusion Detection System (AI-Powered)

An end-to-end Machine Learning system for detecting cyber attacks in network traffic using the NSL-KDD dataset.  
Includes model training, evaluation, explainability, and a full interactive Streamlit web application.

---

## 🚀 Features

- Multi-class attack classification (23 attack types)
- Binary detection mode (Attack vs Normal)
- Real-time prediction via web interface
- Batch prediction using CSV upload
- Confidence scores for each prediction
- Automatic feature scaling & encoding
- Class imbalance handling using SMOTE
- Model explainability using SHAP
- Feature importance visualization
- Three trained models:
  - Logistic Regression
  - Random Forest
  - XGBoost

---

## 🧠 Machine Learning Pipeline

1. Data loading (NSL-KDD)
2. Categorical encoding (LabelEncoder)
3. Class balancing (SMOTE)
4. Feature scaling (StandardScaler)
5. Model training & evaluation
6. Feature importance extraction
7. Model serialization
8. Web deployment with Streamlit

---

## 🏗️ Project Structure

network_intrusion_detection/
│
├── data/
│ └── KDDTrain+.txt
│
├── models/
│ ├── logistic_regression.pkl
│ ├── random_forest.pkl
│ ├── xgboost.pkl
│ ├── scaler.pkl
│ ├── le_protocol_type.pkl
│ ├── le_service.pkl
│ ├── le_flag.pkl
│ ├── le_target.pkl
│ ├── training_features.pkl
│ └── top_features.pkl
│
├── reports/
│ └── confusion_matrix & feature importance plots
│
├── main.py
├── app.py
├── requirements.txt
└── README.md


---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/network_intrusion_detection.git
cd network_intrusion_detection
2. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # Linux/Mac
3. Install dependencies
pip install -r requirements.txt
🏋️ Train the Models
python main.py
This will:

Train all models

Generate reports

Save all assets to /models

🌐 Run the Web Application
streamlit run app.py
Open in browser:

http://localhost:8501
📊 Dataset
NSL-KDD (Improved version of KDD Cup 99)

41 network traffic features

23 attack classes + normal traffic

🧪 Example Use Cases
Intrusion detection systems (IDS)

SOC automation

Network monitoring tools

Cybersecurity research

ML portfolio project

🧩 Model Explainability
This project integrates:

Random Forest feature importance

SHAP value explanations for predictions

This allows understanding why a connection was classified as an attack.

🛡️ Technologies Used
Python

Scikit-learn

XGBoost

Streamlit

SHAP

Pandas / NumPy

Matplotlib / Seaborn

Imbalanced-learn (SMOTE)

📈 Future Improvements
Live packet capture integration

Deep learning models (LSTM / Transformer)

REST API

Docker deployment

Real-time streaming detection

User authentication dashboard