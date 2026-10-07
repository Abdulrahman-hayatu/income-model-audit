"""Group fairness metrics and a proxy-leakage check."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

from src.models import make_preprocessor

RATE_COLS = ["base_rate", "selection_rate", "TPR", "FPR"]

# a function to compute per-group rates at a fixed threshold
def group_report(y_true, proba, group, threshold=0.5):
    """Per-group rates at a fixed threshold. All inputs must be in the same row order."""
    y, p, g = np.asarray(y_true), np.asarray(proba), np.asarray(group)
    if not (len(y) == len(p) == len(g)):
        raise ValueError("y_true, proba and group must have the same length")
    pred = (p >= threshold).astype(int)
    rows = {}
    for lvl in pd.unique(g):
        m = g == lvl
        pos, neg = y[m] == 1, y[m] == 0
        rows[lvl] = {
            "n": int(m.sum()),
            "base_rate": y[m].mean(),
            "selection_rate": pred[m].mean(),
            "TPR": pred[m][pos].mean() if pos.any() else np.nan,
            "FPR": pred[m][neg].mean() if neg.any() else np.nan,
            "AUC": roc_auc_score(y[m], p[m]) if pos.any() and neg.any() else np.nan,
        }
    return pd.DataFrame(rows).T.astype({"n": int})

# a function to compute the difference in rates between two groups
def gaps(report, a, b):
    """Rate differences, group a minus group b."""
    return {f"{c} gap": float(report.loc[a, c] - report.loc[b, c]) for c in RATE_COLS}

# a function to check whether a binary attribute is predictable from the remaining features
def attribute_predictability(X, attr, positive, drop, cv_splits=5, seed=42):
    """Cross-validated ROC-AUC for predicting a binary attribute from the remaining features."""
    y = (X[attr] == positive).astype(int)
    Xr = X.drop(columns=list(set(drop) | {attr}))
    model = make_pipeline(make_preprocessor(Xr), LogisticRegression(max_iter=1000))
    cv = StratifiedKFold(cv_splits, shuffle=True, random_state=seed)
    return cross_val_score(model, Xr, y, cv=cv, scoring="roc_auc", n_jobs=1)


if __name__ == "__main__":
    from sklearn.metrics import recall_score
    from src.data import load_adult
    from src.explain import load_tuned_params, fit_xgb

    # 1. Hand-worked example (answers computed on paper, not by the code under test)
    y = [1, 1, 0, 0, 1, 0, 0, 0]
    p = [0.9, 0.2, 0.8, 0.1, 0.7, 0.6, 0.3, 0.1]
    g = ["a"] * 4 + ["b"] * 4
    r = group_report(y, p, g)
    expected = {"a": dict(n=4, base_rate=0.5, selection_rate=0.5, TPR=0.5, FPR=0.5, AUC=0.75),
                "b": dict(n=4, base_rate=0.25, selection_rate=0.5, TPR=1.0, FPR=1 / 3, AUC=1.0)}
    for lvl, vals in expected.items():
        for col, v in vals.items():
            assert np.isclose(r.loc[lvl, col], v), f"{lvl}/{col}: got {r.loc[lvl, col]}, want {v}"
    gp = gaps(r, "a", "b")
    assert np.isclose(gp["TPR gap"], -0.5) and np.isclose(gp["FPR gap"], 0.5 - 1 / 3)
    print("hand-worked example: OK")

    # 2. Pooled TPR must match sklearn
    rng = np.random.default_rng(0)
    yy = rng.integers(0, 2, 500)
    pp = rng.random(500)
    pooled = group_report(yy, pp, ["all"] * 500)
    assert np.isclose(pooled.loc["all", "TPR"], recall_score(yy, (pp >= 0.5).astype(int)))
    print("matches sklearn recall: OK")

    # 3. Real model on a small slice (test set deliberately ignored)
    X_train, _, y_train, _ = load_adult()
    pipe = fit_xgb(X_train.iloc[:5000], y_train.iloc[:5000], load_tuned_params())
    Xv, yv = X_train.iloc[5000:7000], y_train.iloc[5000:7000]
    rep = group_report(yv, pipe.predict_proba(Xv)[:, 1], Xv["sex"])
    assert rep["n"].sum() == len(Xv)
    print(rep.round(3))

    # 4. Proxy leakage on a slice (3 folds): printed, not asserted
    Xs = X_train.iloc[:5000]
    for name, drop in {"drop sex, race": ["sex", "race"],
                       "drop sex, race, relationship, marital-status":
                           ["sex", "race", "relationship", "marital-status"]}.items():
        auc = attribute_predictability(Xs, "sex", "Male", drop, cv_splits=3)
        print(f"sex predictable from remaining features [{name}]: AUC {auc.mean():.3f}")
    print("OK")