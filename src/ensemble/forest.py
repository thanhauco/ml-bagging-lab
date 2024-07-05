from sklearn.tree import DecisionTreeClassifier
from src.ensemble.bagging import BaggingClassifier

class RandomForestClassifier(BaggingClassifier):
    """
    Random Forest Classifier.
    
    A Random Forest is a meta estimator that fits a number of decision tree classifiers
    on various sub-samples of the dataset and uses averaging to improve the predictive
    accuracy and control over-fitting.
    
    Parameters:
        n_estimators (int): The number of trees in the forest.
        max_features (int, float or {"sqrt", "log2"}): The number of features to consider when looking for the best split.
        max_depth (int): The maximum depth of the tree.
        n_jobs (int): The number of jobs to run in parallel.
        random_state (int): Seed for reproducibility.
    """
    def __init__(self, n_estimators=100, max_features="sqrt", max_depth=None, n_jobs=1, random_state=None):
        # Initialize base estimator with max_features for split-level sampling
        base_estimator = DecisionTreeClassifier(
            max_features=max_features,
            max_depth=max_depth
        )
        
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            max_samples=1.0,  # Bootstrap full size
            max_features=1.0, # All features available to the tree (it selects subset at split)
            bootstrap=True,
            bootstrap_features=False,
            n_jobs=n_jobs,
            random_state=random_state
        )
