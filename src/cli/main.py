import argparse
import numpy as np
import pandas as pd
from src.ensemble.bagging import BaggingClassifier
from src.ensemble.forest import RandomForestClassifier

def main():
    parser = argparse.ArgumentParser(description='ML Bagging Lab CLI')
    parser.add_argument('--model', type=str, default='bagging', choices=['bagging', 'rf'])
    parser.add_argument('--estimators', type=int, default=10)
    parser.add_argument('--data', type=str, required=True)
    args = parser.parse_args()
    
    print(f"Loading data from {args.data}...")
    # Mock loading
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    
    if args.model == 'bagging':
        clf = BaggingClassifier(n_estimators=args.estimators)
    else:
        clf = RandomForestClassifier(n_estimators=args.estimators)
        
    print(f"Training {args.model} with {args.estimators} estimators...")
    clf.fit(X, y)
    print("Training complete.")

if __name__ == '__main__':
    main()
