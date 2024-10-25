import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from src.ensemble.balanced import BalancedBaggingClassifier
from src.ensemble.bagging import BaggingClassifier

def run_fraud_detection_demo():
    print("Generating synthetic imbalanced fraud data...")
    # 99% legitimate, 1% fraud
    X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, 
                               n_redundant=5, weights=[0.99], flip_y=0, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Fraud cases in training: {sum(y_train)}")
    
    print("\n--- Training Standard Bagging ---")
    clf_std = BaggingClassifier(n_estimators=10, random_state=42)
    clf_std.fit(X_train, y_train)
    y_pred_std = clf_std.predict(X_test)
    print("Standard Bagging Results:")
    print(confusion_matrix(y_test, y_pred_std))
    print(classification_report(y_test, y_pred_std))
    
    print("\n--- Training Balanced Bagging ---")
    clf_bal = BalancedBaggingClassifier(n_estimators=10, random_state=42)
    clf_bal.fit(X_train, y_train)
    y_pred_bal = clf_bal.predict(X_test)
    print("Balanced Bagging Results:")
    print(confusion_matrix(y_test, y_pred_bal))
    print(classification_report(y_test, y_pred_bal))

if __name__ == "__main__":
    run_fraud_detection_demo()
