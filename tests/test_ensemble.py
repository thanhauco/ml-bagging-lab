import numpy as np
import pytest
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.ensemble.bagging import BaggingClassifier
from src.ensemble.forest import RandomForestClassifier

@pytest.fixture
def data():
    return load_iris(return_X_y=True)

def test_bagging_classifier(data):
    X, y = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = BaggingClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    assert acc > 0.8
    assert len(clf.estimators_) == 10

def test_random_forest_classifier(data):
    X, y = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    assert acc > 0.8
    assert len(clf.estimators_) == 10

def test_random_subspaces():
    # Create dataset with redundant features
    X, y = make_classification(n_samples=100, n_features=20, n_informative=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Use only 50% of features per estimator
    clf = BaggingClassifier(n_estimators=10, max_features=0.5, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    assert acc > 0.6
    # Check that features were subsampled
    assert len(clf.estimators_features_[0]) == 10
