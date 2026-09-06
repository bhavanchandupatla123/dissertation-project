import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

combined = pd.read_csv("combined_dataset.csv")

target_col = "Estimate; Total: - With self-employment income"
if target_col not in combined.columns:
    raise KeyError(
        f"Target column not found in DataFrame: {target_col}. "
        "Replace `target_col` with the correct exact column name."
    )


def to_numeric_series(s):
    return pd.to_numeric(
        s.astype(str).str.replace(",", "").str.replace(" ", ""),
        errors="coerce",
    )


y_raw = to_numeric_series(combined[target_col])
median_val = np.nanmedian(y_raw)
y = (y_raw > median_val).astype(int)

drop_cols = ["Neighborhood", "Id", target_col]
X = combined.drop(
    columns=[c for c in drop_cols if c in combined.columns],
    errors="ignore",
)

X_numeric = X.apply(
    lambda col: to_numeric_series(col)
    if col.dtype == object or col.dtype.name.startswith("str")
    else pd.to_numeric(col, errors="coerce")
)

non_numeric = [
    c for c in X_numeric.columns
    if not pd.api.types.is_numeric_dtype(X_numeric[c])
]
if non_numeric:
    X_numeric = X_numeric.drop(columns=non_numeric)

imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(
    imputer.fit_transform(X_numeric),
    columns=X_numeric.columns,
    index=X_numeric.index,
)

scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X_imputed),
    columns=X_imputed.columns,
    index=X_imputed.index,
)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.25,
    stratify=y,
    random_state=42,
)

# Random Forest Classification
clf_rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
)
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

print(f"\nAccuracy of Random Forest: {acc_rf:.2f}\n")
print("Classification Report of Random Forest:\n")
print(classification_report(y_test, y_pred_rf, digits=2))

cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(5, 4))
plt.imshow(cm_rf, interpolation="nearest", cmap="Blues")
plt.title("Confusion Matrix of Random Forest Classification")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
thresh = cm_rf.max() / 2.0
for i in range(cm_rf.shape[0]):
    for j in range(cm_rf.shape[1]):
        plt.text(
            j,
            i,
            format(cm_rf[i, j], "d"),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
            color="white" if cm_rf[i, j] > thresh else "black",
        )
plt.xticks([0, 1])
plt.yticks([0, 1])
plt.tight_layout()
plt.show()

y_prob_rf = clf_rf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)
plt.figure(figsize=(6, 5))
plt.plot(fpr_rf, tpr_rf)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title(f"ROC curve of Random Forest Classification (AUC = {roc_auc_rf:.2f})")
plt.grid(True)
plt.tight_layout()
plt.show()

# Gradient Boosting Classification
clf_gb = GradientBoostingClassifier(
    n_estimators=50,
    learning_rate=0.05,
    max_depth=2,
    min_samples_split=10,
    random_state=42,
)
clf_gb.fit(X_train, y_train)
y_pred_gb = clf_gb.predict(X_test)
acc_gb = accuracy_score(y_test, y_pred_gb)

print(f"\nAccuracy of Gradient Boosting: {acc_gb:.2f}\n")
print("Classification Report of Gradient Boosting:\n")
print(classification_report(y_test, y_pred_gb, digits=2))

cm_gb = confusion_matrix(y_test, y_pred_gb)
plt.figure(figsize=(5, 4))
plt.imshow(cm_gb, interpolation="nearest", cmap="Greens")
plt.title("Confusion Matrix of Gradient Boosting Classification")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
thresh = cm_gb.max() / 2.0
for i in range(cm_gb.shape[0]):
    for j in range(cm_gb.shape[1]):
        plt.text(
            j,
            i,
            format(cm_gb[i, j], "d"),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
            color="white" if cm_gb[i, j] > thresh else "black",
        )
plt.xticks([0, 1])
plt.yticks([0, 1])
plt.tight_layout()
plt.show()

y_prob_gb = clf_gb.predict_proba(X_test)[:, 1]
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_prob_gb)
roc_auc_gb = auc(fpr_gb, tpr_gb)
plt.figure(figsize=(6, 5))
plt.plot(fpr_gb, tpr_gb)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title(f"ROC curve of Gradient Boosting Classification (AUC = {roc_auc_gb:.2f})")
plt.grid(True)
plt.tight_layout()
plt.show()

# Classification Comparison of Model Accuracies
models = ["Random Forest Classification", "Gradient Boosting Classification"]
accuracies = [acc_rf, acc_gb]
plt.figure(figsize=(7, 5))
colors = plt.cm.cool(np.linspace(0, 1, len(models)))
bars = plt.bar(models, accuracies, color=colors)
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.01,
        f"{height:.2f}",
        ha="center",
        fontsize=12,
    )
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Classification Comparison of Model Accuracies")
plt.tight_layout()
plt.show()
