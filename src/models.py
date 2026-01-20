from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_logistic_regression(X, y):
    lr = LogisticRegression(max_iter=500)
    lr.fit(X, y)
    return lr

def train_random_forest(X, y, n_estimators=100):
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf.fit(X, y)
    return rf

def train_xgboost(X, y):
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    xgb.fit(X, y)
    return xgb
