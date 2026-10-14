import os
import json
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("Loading dataset...")
df = pd.read_csv("data/bank-full.csv", sep=";")
print("Dataset shape:", df.shape)

# Feature engineering: remove duration to avoid a predictor that is
# only known after a call; create a balance-per-contact feature.
df["balance_per_campaign"] = df["balance"] / (df["campaign"] + 1)
df = df.drop(columns=["duration"])

X = df.drop(columns=["y"])
y = (df["y"] == "yes").astype(int)

# Keep a stratified sample to make tuning finish faster.

if len(df) > 12000:
    sample_idx = (
        df.groupby("y", group_keys=False)
          .sample(frac=12000 / len(df), random_state=42)
          .index
    )
    X = X.loc[sample_idx]
    y = y.loc[sample_idx]


    print("Using stratified sample:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", categorical_pipe, categorical_features)
])

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        {"model__C": [0.1, 1.0, 3.0]}
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        {"model__max_depth": [5, 10, None],
         "model__min_samples_leaf": [2, 10]}
    ),
    "Random Forest": (
        RandomForestClassifier(
            random_state=42, class_weight="balanced", n_jobs=-1
        ),
        {"model__n_estimators": [100],
         "model__max_depth": [10, None],
         "model__min_samples_leaf": [2]}
    )
}

results = []
best_score = -1
best_pipeline = None
best_name = None

for name, (estimator, params) in models.items():
    print(f"\nTuning {name}...")
    pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    search = GridSearchCV(
        pipe, params, scoring="f1", cv=cv, n_jobs=-1, refit=True
    )
    start = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start

    pred = search.predict(X_test)
    prob = search.predict_proba(X_test)[:, 1]
    row = {
        "model": name,
        "best_cv_f1": round(search.best_score_, 4),
        "test_accuracy": round(accuracy_score(y_test, pred), 4),
        "test_precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "test_recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "test_f1": round(f1_score(y_test, pred, zero_division=0), 4),
        "test_roc_auc": round(roc_auc_score(y_test, prob), 4),
        "best_params": str(search.best_params_),
        "tuning_seconds": round(elapsed, 1)
    }
    results.append(row)
    print(row)

    if search.best_score_ > best_score:
        best_score = search.best_score_
        best_pipeline = search.best_estimator_
        best_name = name
        best_pred = pred
        best_prob = prob

pd.DataFrame(results).sort_values("best_cv_f1", ascending=False).to_csv(
    "results/model_comparison.csv", index=False
)
joblib.dump(best_pipeline, "models/best_model.joblib")

with open("models/model_metadata.json", "w") as f:
    json.dump({
        "model_name": best_name,
        "features": X.columns.tolist(),
        "target": "y",
        "positive_class": "yes",
        "sample_size": len(X),
        "test_size": len(X_test)
    }, f, indent=2)

# Save a confusion matrix for the selected model on its test split.
plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, best_pred, display_labels=["No", "Yes"], cmap="Blues"
)
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=150)
plt.close()

with open("results/classification_report.txt", "w") as f:
    f.write(f"Selected model: {best_name}\n\n")
    f.write(classification_report(
        y_test, best_pred, target_names=["No", "Yes"], zero_division=0
    ))

print("\nTraining complete.")
print("Selected model:", best_name)
print("Saved: models/best_model.joblib")
print("Saved: results/model_comparison.csv")
print("Saved: results/confusion_matrix.png")
print("Saved: results/classification_report.txt")
