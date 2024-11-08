import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.ensemble.forest import RandomForestClassifier

def run_churn_prediction_demo():
    print("Generating synthetic churn data...")
    # Simulating telecom data features
    X, y = make_classification(n_samples=5000, n_features=15, n_informative=10, 
                               random_state=42)
    
    feature_names = [f"feat_{i}" for i in range(15)]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\n--- Training Random Forest ---")
    clf = RandomForestClassifier(n_estimators=50, max_features="sqrt", random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    # Feature Importance (Simulated by aggregating base estimators)
    # Since we use sklearn trees, we can access their feature_importances_
    # But we need to map them back if we used feature subsampling in Bagging (Random Subspaces)
    # In our RF implementation, we pass all features to the tree and let the tree subsample at split.
    # So tree.feature_importances_ is valid for all features.
    
    importances = np.mean([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    
    print("\nFeature Ranking:")
    for f in range(X.shape[1]):
        print(f"{f+1}. {feature_names[indices[f]]} ({importances[indices[f]]:.4f})")

if __name__ == "__main__":
    run_churn_prediction_demo()
