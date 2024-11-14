from src.ensemble.bagging import BaggingClassifier

class PastingClassifier(BaggingClassifier):
    def __init__(self, base_estimator=None, n_estimators=10, n_jobs=1, random_state=None):
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            bootstrap=False,  # Pasting = Sampling without replacement
            n_jobs=n_jobs,
            random_state=random_state
        )
