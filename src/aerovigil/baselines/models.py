"""Model zoo, tuning grids and metrics for the Module 05 baselines."""
import itertools

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SUPERVISED = ("logistic_regression", "naive_bayes", "knn", "decision_tree", "random_forest")
UNSUPERVISED = ("isolation_forest",)
STOCHASTIC = ("decision_tree", "random_forest", "isolation_forest")


def grid(params):
    keys = list(params)
    return [dict(zip(keys, values)) for values in itertools.product(*(params[k] for k in keys))]


def build_model(name, params, seed):
    """Imputer + scaler + estimator. Both preprocessing steps are fitted on the
    training windows only, because the whole pipeline is fitted on them."""
    if name == "logistic_regression":
        clf = LogisticRegression(class_weight="balanced", max_iter=2000, **params)
    elif name == "naive_bayes":
        clf = GaussianNB(**params)
    elif name == "knn":
        clf = KNeighborsClassifier(n_jobs=-1, **params)
    elif name == "decision_tree":
        clf = DecisionTreeClassifier(class_weight="balanced", random_state=seed, **params)
    elif name == "random_forest":
        clf = RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1,
                                     random_state=seed, **params)
    elif name == "isolation_forest":
        clf = IsolationForest(random_state=seed, n_jobs=-1, **params)
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()), ("model", clf)])


def fit_model(name, pipe, x, y):
    if name in UNSUPERVISED:            # one-class: normal windows only
        n_normal = int((y == 0).sum())
        ms = pipe[-1].max_samples
        if isinstance(ms, int) and ms > n_normal:
            pipe[-1].set_params(max_samples=n_normal)
        pipe.fit(x[y == 0])
    else:
        pipe.fit(x, y)
    return pipe


def score_model(name, pipe, x):
    """Attack score: higher means more attack-like."""
    if name in UNSUPERVISED:
        return -pipe[-1].score_samples(pipe[:-1].transform(x))
    return pipe.predict_proba(x)[:, 1]


def tpr_at_fpr(y_true, score, target_fpr):
    fpr, tpr, _ = roc_curve(y_true, score)
    ok = fpr <= target_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def evaluate(y_true, score, threshold, target_fpr=0.01):
    """Window-level metrics. Returns None if the set lacks either class."""
    y_true = np.asarray(y_true).astype(int)
    n_pos, n_neg = int((y_true == 1).sum()), int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    pred = (np.asarray(score) >= threshold).astype(int)
    return {"n_pos": n_pos, "n_neg": n_neg,
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, score),
            "tpr_at_fpr": tpr_at_fpr(y_true, score, target_fpr)}