"""Teacher ensemble and student MLP builder for KARAT."""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin


class KATSEnsemble(BaseEstimator, ClassifierMixin):
    """
    Soft-voting ensemble: GBM + RF + LR.
    alpha controls ensemble weight toward the stronger GBM component.
    """

    def __init__(self, alpha: float = 5.0, seed: int = 42):
        self.alpha = alpha
        self.seed  = seed
        self.gbm_  = None
        self.rf_   = None
        self.lr_   = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.gbm_ = GradientBoostingClassifier(
            n_estimators=120, max_depth=5,
            learning_rate=0.08, random_state=self.seed
        ).fit(X, y)
        self.rf_ = RandomForestClassifier(
            n_estimators=80, max_depth=6,
            random_state=self.seed
        ).fit(X, y)
        self.lr_ = Pipeline([
            ('sc', StandardScaler()),
            ('clf', LogisticRegression(C=1.0, max_iter=300,
                                       random_state=self.seed))
        ]).fit(X, y)
        return self

    def predict_proba(self, X):
        w = self.alpha
        p = (w * self.gbm_.predict_proba(X) +
             1.0 * self.rf_.predict_proba(X) +
             1.0 * self.lr_.predict_proba(X)) / (w + 2.0)
        return p

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def build_kats_ensemble(alpha: float = 5.0, seed: int = 42) -> KATSEnsemble:
    """Return an unfitted KATSEnsemble."""
    return KATSEnsemble(alpha=alpha, seed=seed)


def build_student(seed: int = 42,
                  hidden: tuple = (16,),
                  lr: float = 0.008,
                  alpha: float = 0.05,
                  max_iter: int = 60) -> MLPClassifier:
    """Return an unfitted shallow MLP student."""
    return MLPClassifier(
        hidden_layer_sizes=hidden,
        activation='relu',
        max_iter=max_iter,
        random_state=seed,
        learning_rate_init=lr,
        alpha=alpha
    )
