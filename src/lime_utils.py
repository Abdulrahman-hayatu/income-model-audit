"""LIME helpers: bridge between LIME's numeric space and the sklearn pipeline."""
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

# Build a LIME explainer for the fitted pipeline
def build_lime(X_train, pipe, seed=0):
    """Return (explainer, predict_fn, encode_rows) for the fitted pipeline."""
    cols = list(X_train.columns)
    cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()
    cat_idx = [cols.index(c) for c in cat_cols]
    levels = {c: sorted(X_train[c].unique()) for c in cat_cols}
    lookup = {c: {lv: i for i, lv in enumerate(levels[c])} for c in cat_cols}
# a method to encode rows of the original dataframe into LIME's numeric space
    def encode_rows(df):
        out = df[cols].copy()
        for c in cat_cols:
            out[c] = out[c].map(lookup[c])
        return out.to_numpy(dtype=float)
# a method to predict probabilities from LIME's numeric space back to the original pipeline
    def predict_fn(arr):
        df = pd.DataFrame(arr, columns=cols)
        for c in cat_cols:
            codes = df[c].round().astype(int).to_numpy()
            df[c] = np.asarray(levels[c], dtype=object)[codes]
        return pipe.predict_proba(df)

    explainer = LimeTabularExplainer(
        encode_rows(X_train),
        feature_names=cols,
        categorical_features=cat_idx,
        categorical_names={i: levels[cols[i]] for i in cat_idx},
        class_names=["<=50K", ">50K"],
        discretize_continuous=True,
        random_state=seed,
    )
    return explainer, predict_fn, encode_rows

# Explain a single instance with LIME and return weights per original feature
def lime_weights(explainer, predict_fn, encoded_row, columns, num_samples=5000):
    """Return (weights per original feature for P(>50K), surrogate R2, surrogate prediction)."""
    exp = explainer.explain_instance(
        encoded_row, predict_fn, num_features=len(columns), num_samples=num_samples
    )
    w = pd.Series(dict(exp.as_map()[1]))
    w.index = [columns[i] for i in w.index]
    score = exp.score[1] if isinstance(exp.score, dict) else exp.score
    lp = exp.local_pred[1] if isinstance(exp.local_pred, dict) else exp.local_pred
    return w.reindex(columns), float(score), float(np.ravel(lp)[0])


if __name__ == "__main__":
    import json
    from src.data import load_adult
    from src.explain import load_tuned_params, fit_xgb

    X_train, _, y_train, _ = load_adult()  # test set deliberately ignored
    pipe = fit_xgb(X_train, y_train, load_tuned_params())
    explainer, predict_fn, encode_rows = build_lime(X_train, pipe)
    cols = list(X_train.columns)

    # 1. Round trip: LIME's encoding must reproduce the pipeline's predictions
    rows = X_train.sample(50, random_state=0)
    gap = float(np.abs(predict_fn(encode_rows(rows)) - pipe.predict_proba(rows)).max())
    print(f"round-trip max gap vs pipe.predict_proba: {gap:.2e}")
    assert gap < 1e-6, "LIME bridge does not reproduce the pipeline"

    # 2. The row saved in notebook 03 is the row we explain
    saved = json.load(open("results/explained_instance.json"))
    idx = saved["train_row_index"]
    p_pipe = float(pipe.predict_proba(X_train.loc[[idx]])[0, 1])
    print(f"row {idx}: P(>50K) = {p_pipe:.4f} (saved: {saved['proba']:.4f})")
    assert abs(p_pipe - saved["proba"]) < 1e-4, "refit model differs from the notebook's"

    # 3. One LIME explanation of that row
    w, r2, local_pred = lime_weights(explainer, predict_fn,
                                     encode_rows(X_train.loc[[idx]])[0], cols)
    assert len(w) == 13 and w.notna().all(), "expected one weight per original feature"
    print(f"surrogate R2: {r2:.3f} | surrogate prediction: {local_pred:.3f} | pipeline: {p_pipe:.3f}")
    print(w.sort_values(key=abs, ascending=False).round(4))
    print("OK")