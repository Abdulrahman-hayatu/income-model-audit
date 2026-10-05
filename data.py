"""Data loading for the Adult Income audit."""
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

SEED = 42
POSITIVE_LABEL = ">50K"
DROP_COLS = ["fnlwgt"]


def load_adult(seed: int = SEED, test_size: float = 0.2):
    """Return X_train, X_test, y_train, y_test (y: 1 if income > 50K)."""
    X, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)

    y = y.astype(str).str.strip().eq(POSITIVE_LABEL).astype(int)
    if y.sum() == 0:
        raise ValueError("No positive labels found; check target encoding.")

    X = X.drop(columns=DROP_COLS)
    cat_cols = X.select_dtypes(include=["category", "object"]).columns
    X[cat_cols] = X[cat_cols].astype("object").fillna("Missing")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    return (X_train.reset_index(drop=True), X_test.reset_index(drop=True),
            y_train.reset_index(drop=True), y_test.reset_index(drop=True))


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_adult()
    print("train/test shapes:", X_train.shape, X_test.shape)
    print("positive rate train/test:", round(y_train.mean(), 4), round(y_test.mean(), 4))
    print("remaining NaNs:", int(X_train.isna().sum().sum()))
    print("categorical:", X_train.select_dtypes("object").columns.tolist())
    print("numeric:", X_train.select_dtypes(exclude="object").columns.tolist())