import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score,
    learning_curve
)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.metrics import (
    roc_curve,
    confusion_matrix,
    precision_recall_curve,
    brier_score_loss
)

# -------------------------------------------------------
# LOAD DATA  (FROM YOUR ORIGINAL FILE)  :contentReference[oaicite:1]{index=1}
# -------------------------------------------------------
DATA_FILE = r"query (2).csv"

df = pd.read_csv(DATA_FILE)
df = df.drop_duplicates()

mag_col = "mag"
df = df.dropna(subset=[mag_col])

# Simplified features compatible with GUI
features = ["latitude", "longitude", "depth", "gap", "rms", "nst", "dmin"]
df[features] = df[features].apply(pd.to_numeric, errors="coerce")

X = df[features]
y = df[mag_col]

mask = ~y.isna()
X, y = X[mask], y[mask]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), list(range(len(features))))
    ]
)

# -------------------------------------------------------
# TRAIN REGRESSION MODEL
# -------------------------------------------------------
reg_model = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestRegressor(n_estimators=200, random_state=42))
])

reg_model.fit(X_train, y_train)
y_pred = reg_model.predict(X_test)

# -------------------------------------------------------
# TRAIN CLASSIFICATION MODEL
# -------------------------------------------------------
threshold = 4.0
labels = (df[mag_col] >= threshold).astype(int)

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

cls_model = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestClassifier(n_estimators=300, random_state=42))
])

cls_model.fit(Xc_train, yc_train)

# -------------------------------------------------------
# SAVE MODEL BUNDLE
# -------------------------------------------------------
bundle = {
    "features": features,
    "threshold": threshold,
    "reg_pipeline": reg_model,
    "cls_pipeline": cls_model
}

joblib.dump(bundle, "eq_model_bundle_new.pkl")
print("Saved updated model → eq_model_bundle_new.pkl")

# -------------------------------------------------------
# CREATE RESULTS FOLDER
# -------------------------------------------------------
os.makedirs("results_new", exist_ok=True)


# =======================================================
# 1️⃣ RESIDUAL PLOT
# =======================================================
residuals = y_test - y_pred

plt.figure(figsize=(6,4))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, color="red")
plt.xlabel("Predicted Magnitude")
plt.ylabel("Residual Error")
plt.title("Residual Plot")
plt.savefig("results_new/residual_plot.png")
plt.close()


# =======================================================
# 2️⃣ LEARNING CURVE
# =======================================================
train_sizes, train_scores, test_scores = learning_curve(
    reg_model, X, y, cv=5, scoring="r2"
)

plt.figure(figsize=(6,4))
plt.plot(train_sizes, train_scores.mean(axis=1), label="Train Score")
plt.plot(train_sizes, test_scores.mean(axis=1), label="Validation Score")
plt.xlabel("Training Samples")
plt.ylabel("R² Score")
plt.title("Learning Curve")
plt.legend()
plt.savefig("results_new/learning_curve.png")
plt.close()


# =======================================================
# 3️⃣ K-FOLD CROSS VALIDATION
# =======================================================
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(reg_model, X, y, cv=kfold, scoring="r2")

with open("results_new/cross_validation_scores.txt", "w") as f:
    f.write("K-FOLD CROSS VALIDATION SCORES\n")
    f.write(str(cv_scores) + "\n")
    f.write("Mean: " + str(np.mean(cv_scores)) + "\n")
    f.write("Std: " + str(np.std(cv_scores)) + "\n")


# =======================================================
# 4️⃣ PRECISION–RECALL CURVE (CLASSIFICATION)
# =======================================================
y_proba = cls_model.predict_proba(Xc_test)[:, 1]
precision, recall, _ = precision_recall_curve(yc_test, y_proba)

plt.figure(figsize=(6,4))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.savefig("results_new/precision_recall_curve.png")
plt.close()


# =======================================================
# 5️⃣ CALIBRATION CURVE
# =======================================================
# Brier Score
brier = brier_score_loss(yc_test, y_proba)

# Perfect calibration line
plt.figure(figsize=(6,4))
plt.plot([0,1], [0,1], "k--", label="Perfect Calibration")

# Your model curve
sorted_idx = np.argsort(y_proba)
plt.plot(y_proba[sorted_idx], yc_test.iloc[sorted_idx], label="Model Calibration")

plt.xlabel("Predicted Probability")
plt.ylabel("Actual Outcome")
plt.title(f"Calibration Curve (Brier Score={brier:.3f})")
plt.legend()
plt.savefig("results_new/calibration_curve.png")
plt.close()

print("All evaluation graphs saved in results_new/")
