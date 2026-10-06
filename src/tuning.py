"""Hyperparameter optimization for XGBoost with Optuna."""
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.models import build_models

TUNE_SEED = 7  # differs from the CV seed (42) used for the Step 3 comparison
XGB_KEY = "XGBoost (boosting)"
PARAM_KEYS = {"n_estimators", "learning_rate", "max_depth", "min_child_weight",
              "subsample", "colsample_bytree", "reg_lambda", "reg_alpha"}

# parameters to tune for XGBoost, with ranges and distributions
def suggest_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }

# Function to evaluate the performance of a model with given parameters
def _score(X, y, params, cv, seed):
    model = build_models(X, seed=seed, xgb_params=params)[XGB_KEY]
    return cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=1).mean()

# Function to perform hyperparameter tuning for XGBoost using Optuna
def tune_xgb(X, y, n_trials=40, cv_splits=5, seed=42):
    """Return (study, default_score), both measured on the same tuning folds."""
    cv = StratifiedKFold(cv_splits, shuffle=True, random_state=TUNE_SEED)

    def objective(trial):
        return _score(X, y, suggest_params(trial), cv, seed)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials)
    default_score = _score(X, y, {}, cv, seed)
    return study, default_score

# Run a simple test of the tuning function when this file is executed directly
if __name__ == "__main__":
    from src.data import load_adult

    X_train, _, y_train, _ = load_adult()  # test set deliberately ignored
    Xs, ys = X_train.iloc[:5000], y_train.iloc[:5000]

    study, default_score = tune_xgb(Xs, ys, n_trials=3, cv_splits=3)
    print(f"default XGB CV AUC (same folds): {default_score:.4f}")
    print(f"best of 3 trials:                {study.best_value:.4f}")
    print("best params:", study.best_params)

    assert len(study.trials) == 3
    assert set(study.best_params) == PARAM_KEYS
    print("OK")