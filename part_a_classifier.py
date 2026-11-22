"""Part A - Dish category classifier (Kharchify AI/ML Intern task).

Run:  python part_a_classifier.py
Outputs: console report + outputs/confusion_matrix.png + outputs/dish_classifier.joblib

Design notes
* TF-IDF (word + character n-grams) -> Logistic Regression is the primary model.
* Hyper-parameter C is tuned with cross-validation on the TRAIN split only;
  the test split is touched exactly once, for the final report.
* With only ~130 rows a 20 % test set is 26 dishes (1 error = ~3.8 pts), so
  I also report repeated stratified CV, which is a far steadier estimate.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_val_score, train_test_split)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

SEED = 42
BASE = Path(__file__).parent
DATA = BASE / "data" / "dishes.csv"
OUT = BASE / "outputs"
MODEL_PATH = OUT / "dish_classifier.joblib"
LOW_CONFIDENCE = 0.50  # below this, the suggestion is flagged "needs review"


# --------------------------------------------------------------------------- #
# 1. Load + clean
# --------------------------------------------------------------------------- #
def clean_dishes(raw: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Trim/normalise text, drop missing rows, resolve duplicates."""
    df = raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    n0 = len(df)

    df = df.dropna(subset=["dish_name", "category"])
    for col in ("dish_name", "category"):
        df[col] = df[col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[(df["dish_name"] != "") & (df["category"] != "")]
    df["category"] = df["category"].str.title()          # "main course" -> "Main Course"
    df["key"] = df["dish_name"].str.lower()               # case-insensitive identity

    n_missing = n0 - len(df)
    # Same dish, different labels -> ambiguous: drop all copies rather than guess.
    conflicts = df.groupby("key")["category"].nunique()
    conflict_keys = conflicts[conflicts > 1].index
    df = df[~df["key"].isin(conflict_keys)]
    n_conflict = len(conflict_keys)
    # Same dish, same label -> keep first.
    before = len(df)
    df = df.drop_duplicates("key", keep="first")
    n_dupe = before - len(df)

    df = df.drop(columns="key").reset_index(drop=True)
    if verbose:
        print(f"[clean] rows in: {n0} | missing/blank dropped: {n_missing} | "
              f"exact duplicates dropped: {n_dupe} | conflicting-label dishes dropped: {n_conflict} "
              f"| rows out: {len(df)}")
    return df


def normalise_text(s: str) -> str:
    """lower-case, strip punctuation, collapse spaces (used for training AND prediction)."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()


# --------------------------------------------------------------------------- #
# 2. Models
# --------------------------------------------------------------------------- #
def _word_tfidf():
    return TfidfVectorizer(preprocessor=normalise_text, ngram_range=(1, 2), sublinear_tf=True)


def _word_char_tfidf():
    # char_wb n-grams let the model recognise sub-words ("-naan", "biryan-") and
    # survive spelling variants (Roomali / Rumali) that word features miss.
    return FeatureUnion([
        ("word", _word_tfidf()),
        ("char", TfidfVectorizer(preprocessor=normalise_text, analyzer="char_wb",
                                 ngram_range=(2, 4), sublinear_tf=True)),
    ])


def build_models() -> dict[str, Pipeline]:
    return {
        "LogReg + word TF-IDF": Pipeline([
            ("tfidf", _word_tfidf()),
            ("clf", LogisticRegression(C=10, max_iter=2000, random_state=SEED))]),
        "LogReg + word+char TF-IDF (primary)": Pipeline([
            ("tfidf", _word_char_tfidf()),
            ("clf", LogisticRegression(C=10, max_iter=2000, random_state=SEED))]),
        "LinearSVC + word+char TF-IDF": Pipeline([
            ("tfidf", _word_char_tfidf()),
            ("clf", LinearSVC(C=1.0, random_state=SEED))]),
    }


PRIMARY = "LogReg + word+char TF-IDF (primary)"


# --------------------------------------------------------------------------- #
# 3. Prediction API
# --------------------------------------------------------------------------- #
_MODEL: Pipeline | None = None


def _get_model() -> Pipeline:
    """Load the saved model, or train it (on all clean data) if it isn't there yet."""
    global _MODEL
    if _MODEL is None:
        if MODEL_PATH.exists():
            _MODEL = joblib.load(MODEL_PATH)
        else:
            df = clean_dishes(pd.read_csv(DATA), verbose=False)
            _MODEL = build_models()[PRIMARY].fit(df["dish_name"], df["category"])
    return _MODEL


def predict_category(dish_name: str, top_k: int = 3) -> dict:
    """Suggest a category for a new dish.

    Returns {'dish', 'category', 'confidence', 'needs_review', 'top'} where 'top' lists
    the top_k (category, probability) pairs. needs_review=True when confidence is low,
    so the UI can ask the owner to confirm instead of silently auto-filling.
    """
    if not isinstance(dish_name, str) or not normalise_text(dish_name):
        return {"dish": dish_name, "category": None, "confidence": 0.0,
                "needs_review": True, "top": []}
    model = _get_model()
    proba = model.predict_proba([dish_name])[0]
    order = np.argsort(proba)[::-1][:top_k]
    top = [(model.classes_[i], float(proba[i])) for i in order]
    return {"dish": dish_name.strip(), "category": top[0][0], "confidence": top[0][1],
            "needs_review": top[0][1] < LOW_CONFIDENCE, "top": top}


# --------------------------------------------------------------------------- #
# 4. Experiment
# --------------------------------------------------------------------------- #
def main() -> None:
    OUT.mkdir(exist_ok=True)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    df = clean_dishes(pd.read_csv(DATA))
    print("\nClass balance:\n" + df["category"].value_counts().to_string())

    # Stratified split so every category appears in test; fixed seed = reproducible.
    X_tr, X_te, y_tr, y_te = train_test_split(
        df["dish_name"], df["category"], test_size=0.2, stratify=df["category"], random_state=SEED)
    print(f"\nTrain: {len(X_tr)} | Test: {len(X_te)}")

    # --- tune C for the primary model on TRAIN only (5-fold CV) ---
    models = build_models()
    grid = GridSearchCV(models[PRIMARY], {"clf__C": [1, 3, 10, 30, 100]},
                        cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="accuracy")
    grid.fit(X_tr, y_tr)
    print(f"Best C for primary (train-only CV): {grid.best_params_['clf__C']} "
          f"(CV acc {grid.best_score_:.3f})")
    models[PRIMARY] = grid.best_estimator_

    # --- fit everything on train, evaluate ONCE on test ---
    print("\n=== Test-set accuracy (26 unseen dishes) ===")
    for name, m in models.items():
        if name != PRIMARY:
            m.fit(X_tr, y_tr)
        print(f"{name:42s} {accuracy_score(y_te, m.predict(X_te)):.3f}")

    # --- steadier estimate: repeated stratified 5-fold CV over all data ---
    print("\n=== Repeated 5-fold CV over all 130 dishes (3 repeats) ===")
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    for name, m in build_models().items():
        s = cross_val_score(m, df["dish_name"], df["category"], cv=rcv, scoring="accuracy")
        print(f"{name:42s} {s.mean():.3f} +/- {s.std():.3f}")

    # --- per-category report for the primary model ---
    best = models[PRIMARY]
    pred = best.predict(X_te)
    print(f"\n=== Primary model: {PRIMARY} ===")
    print(f"Test accuracy: {accuracy_score(y_te, pred):.3f}\n")
    print(classification_report(y_te, pred, zero_division=0))
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y_te, pred, ax=ax, xticks_rotation=45, cmap="Blues",
                                            colorbar=False)
    ax.set_title("Confusion matrix - test set (primary model)")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=130)
    plt.close(fig)

    # --- error analysis: test errors + out-of-fold errors on all data (more samples) ---
    print("=== Test-set mistakes ===")
    wrong = pd.DataFrame({"dish": X_te.values, "true": y_te.values, "pred": pred})
    print(wrong[wrong.true != wrong.pred].to_string(index=False) or "none")

    oof = cross_val_predict(build_models()[PRIMARY], df["dish_name"], df["category"],
                            cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    err = df.assign(pred=oof)
    err = err[err.category != err.pred]
    print(f"\n=== Out-of-fold mistakes over all data ({len(err)}/{len(df)}) ===")
    print(err.to_string(index=False))

    # --- final model: refit on ALL cleaned data for deployment + save ---
    final = build_models()[PRIMARY].set_params(clf__C=grid.best_params_["clf__C"])
    final.fit(df["dish_name"], df["category"])
    joblib.dump(final, MODEL_PATH)
    global _MODEL
    _MODEL = final

    # --- predictions for dishes NOT in the dataset ---
    new_dishes = ["Butter Kulcha", "Chocolate Lava Cake", "Prawn Biryani", "Vanilla Milkshake",
                  "Paneer Tikka Masala", "Cheese Garlic Bread", "Chicken Lollipop",
                  "Tandoori Paneer Tikka", "Lemon Iced Tea", "Rava Dosa",
                  "  panner butter masla ", "Hakka Noodles"]
    assert not set(map(str.lower, map(str.strip, new_dishes))) & set(df["dish_name"].str.lower())
    print("\n=== predict_category() on dishes not in the dataset ===")
    for d in new_dishes:
        r = predict_category(d)
        flag = "  <-- needs review" if r["needs_review"] else ""
        alts = ", ".join(f"{c} {p:.2f}" for c, p in r["top"])
        print(f"{r['dish']:26s} -> {r['category']:12s} ({r['confidence']:.2f})  [{alts}]{flag}")


if __name__ == "__main__":
    main()
