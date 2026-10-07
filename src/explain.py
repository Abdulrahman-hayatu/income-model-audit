"""SHAP explanations for the tuned XGBoost pipeline."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from src.models import build_models

# Configuration
XGB_KEY = "XGBoost (boosting)"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# loading tuned parameters from the results directory
def load_tuned_params(path=None):
    path = Path(path) if path else RESULTS_DIR / "best_xgb_params.json"
    with open(path) as f:
        return json.load(f)

# Fit the XGBoost model with tuned parameters
def fit_xgb(X, y, params=None, seed=42):
    model = build_models(X, seed=seed, xgb_params=params)[XGB_KEY]
    return model.fit(X, y)

# Encode the input data using the fitted preprocessor
def encode(pipe, X):
    """Apply the fitted preprocessor; return (dense array, encoded column names)."""
    pre = pipe[0]
    Z = pre.transform(X)
    if hasattr(Z, "toarray"):
        Z = Z.toarray()
    return np.asarray(Z, dtype=float), list(pre.get_feature_names_out())

# Map encoded column names back to original features
def _feature_map(pipe):
    """Map each encoded column name back to its original feature."""
    pre = pipe[0]
    mapping = {}
    for name, _, cols in pre.transformers_:
        if name == "num":
            for c in cols:
                mapping[f"num__{c}"] = c
        elif name == "cat":
            cats = pre.named_transformers_["cat"].categories_
            for c, levels in zip(cols, cats):
                for lv in levels:
                    mapping[f"cat__{c}_{lv}"] = c
    return mapping

# Explain SHAP values
def explain_shap(pipe, X):
    """SHAP values per encoded column and summed per original feature (log-odds)."""
    Z, names = encode(pipe, X)
    clf = pipe[-1]
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(Z)
    if isinstance(sv, list):
        sv = sv[1]
    base = float(np.ravel(explainer.expected_value)[0])
    proba_shap = 1 / (1 + np.exp(-(base + sv.sum(axis=1))))
    gap = float(np.abs(proba_shap - pipe.predict_proba(X)[:, 1]).max())
    if gap > 1e-3:
        raise RuntimeError(
            f"SHAP explains a different function than the pipeline (gap {gap:.3f})")

    fmap = _feature_map(pipe)
    owners = [fmap[n] for n in names]  # KeyError here = mapping is broken
    grouped = (pd.DataFrame(sv, columns=owners, index=X.index)
                 .T.groupby(level=0, sort=False).sum().T)
    return {"values": sv, "names": names, "Z": Z,
            "grouped": grouped, "base_value": base}

# Calculate global importance of features
def global_importance(grouped):
    """Mean |SHAP| per original feature (sum within feature first, then abs)."""
    return grouped.abs().mean().sort_values(ascending=False)


if __name__ == "__main__":
    from src.data import load_adult

    X_train, _, y_train, _ = load_adult()  # test set deliberately ignored
    params = load_tuned_params()
    Xa, ya = X_train.iloc[:5000], y_train.iloc[:5000]
    Xb = X_train.iloc[5000:5200]

    pipe = fit_xgb(Xa, ya, params)
    out = explain_shap(pipe, Xb)
    grouped = out["grouped"]

    proba_pipe = pipe.predict_proba(Xb)[:, 1]
    proba_shap = 1 / (1 + np.exp(-(out["base_value"] + out["values"].sum(axis=1))))
    gap = float(np.abs(proba_pipe - proba_shap).max())
    print(f"max gap vs pipeline.predict_proba: {gap:.2e}")
    assert gap < 1e-3, "SHAP does not match the pipeline's predictions"

    assert set(grouped.columns) == set(Xb.columns), "grouping lost or added features"
    assert np.allclose(grouped.sum(axis=1), out["values"].sum(axis=1), atol=1e-5), \
        "grouping changed the total attribution"

    print("encoded columns:", len(out["names"]), "-> original features:", grouped.shape[1])
    print(global_importance(grouped).head(5).round(3))
    print("OK")