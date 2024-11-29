import numpy as np
from src.ensemble.isolation_forest import IsolationForest
from src.preprocessing.smote import SMOTE
from src.ensemble.adaboost import AdaBoostClassifier

def run_advanced_fraud_demo():
    print("Running Advanced Fraud Detection...")
    # Mock data
    X = np.random.rand(1000, 10)
    y = np.zeros(1000)
    y[:10] = 1 # 1% fraud
    
    print("1. Anomaly Detection with Isolation Forest")
    iso = IsolationForest(random_state=42)
    iso.fit(X)
    preds = iso.predict(X)
    print(f"Detected {np.sum(preds == -1)} anomalies")
    
    print("\n2. Oversampling with SMOTE")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"Original shape: {X.shape}, Resampled shape: {X_res.shape}")
    
    print("\n3. Training AdaBoost on Resampled Data")
    clf = AdaBoostClassifier(n_estimators=10)
    clf.fit(X_res, y_res)
    print("Training complete.")

if __name__ == "__main__":
    run_advanced_fraud_demo()
