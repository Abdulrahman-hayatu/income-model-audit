"""Model definitions: baseline and ensembles, each wrapped with its preprocessing."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

SEED = 42


def make_preprocessor(X):
    num = X.select_dtypes(include="number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])


def _lr():
    return LogisticRegression(max_iter=1000)


def _rf(seed):
    return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)


def _xgb(seed, params=None):
    cfg = dict(n_estimators=300, eval_metric="logloss", n_jobs=-1, random_state=seed)
    cfg.update(params or {})
    return XGBClassifier(**cfg)


def build_models(X, seed=SEED, xgb_params=None):
    """Return {name: sklearn Pipeline}. X is only used to detect column types."""
    def pipe(clf):
        return make_pipeline(make_preprocessor(X), clf)

    stack = StackingClassifier(
        estimators=[("rf", _rf(seed)), ("xgb", _xgb(seed, xgb_params))],
        final_estimator=_lr(),
        cv=5,
    )
    return {
        "Logistic Regression (baseline)": pipe(_lr()),
        "Random Forest (bagging)": pipe(_rf(seed)),
        "XGBoost (boosting)": pipe(_xgb(seed, xgb_params)),
        "Stacking (RF + XGB -> LR)": pipe(stack),
    }


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score
    from src.data import load_adult

    X_train, _, y_train, _ = load_adult()  # test set deliberately ignored
    Xa, ya = X_train.iloc[:5000], y_train.iloc[:5000]
    Xb, yb = X_train.iloc[5000:7000], y_train.iloc[5000:7000]

    for name, model in build_models(Xa).items():
        model.fit(Xa, ya)
        proba = model.predict_proba(Xb)
        assert proba.shape == (len(Xb), 2), "unexpected probability shape"
        print(f"{name:35s} AUC={roc_auc_score(yb, proba[:, 1]):.3f}")