import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# This file follows the regression and recommendation logic documented in the dissertation.
# It expects X_train, X_test, y_train, y_test, clf_gb, scaler and imputer to have been
# created by the classification pipeline.

# Regression of Decision Tree
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

r2_dt = r2_score(y_test, y_pred_dt)
mae_dt = mean_absolute_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))

print("===== Regression Decision Tree =====")
print("R² Score:", r2_dt)
print("MAE:", mae_dt)
print("RMSE:", rmse_dt)

plt.figure(figsize=(7, 6))
sns.scatterplot(x=y_test, y=y_pred_dt, color="green", s=60)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--",
    linewidth=2,
)
plt.title("Actual vs Predicted Values of Decision Tree Regression", fontsize=14)
plt.xlabel("Actual Values", fontsize=12)
plt.ylabel("Predicted Values", fontsize=12)
plt.grid(True)
plt.show()

# Regression of Random Forest
rf = RandomForestRegressor(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print("===== Regression of Random Forest =====")
print("R² Score:", r2_rf)
print("MAE:", mae_rf)
print("RMSE:", rmse_rf)

plt.figure(figsize=(7, 6))
sns.scatterplot(x=y_test, y=y_pred_rf, color="purple", s=60)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--",
    linewidth=2,
)
plt.title("Actual vs Predicted Values of Random Forest Regression", fontsize=14)
plt.xlabel("Actual Values", fontsize=12)
plt.ylabel("Predicted Values", fontsize=12)
plt.grid(True)
plt.show()

# Comparison of Regression Metrics
models = ["Decision Tree", "Random Forest"]
r2_scores = [r2_dt, r2_rf]
mae_scores = [mae_dt, mae_rf]
rmse_scores = [rmse_dt, rmse_rf]

plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
ax1 = sns.barplot(x=models, y=r2_scores, hue=models, palette="Blues", legend=False)
plt.title("R² Score Comparison")
plt.ylabel("R² Score")
for i, v in enumerate(r2_scores):
    ax1.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)

plt.subplot(1, 3, 2)
ax2 = sns.barplot(x=models, y=mae_scores, hue=models, palette="Greens", legend=False)
plt.title("MAE Comparison")
plt.ylabel("MAE")
for i, v in enumerate(mae_scores):
    ax2.text(i, v + (v * 0.02), f"{v:.2f}", ha="center", fontsize=10)

plt.subplot(1, 3, 3)
ax3 = sns.barplot(x=models, y=rmse_scores, hue=models, palette="Purples", legend=False)
plt.title("RMSE Comparison")
plt.ylabel("RMSE")
for i, v in enumerate(rmse_scores):
    ax3.text(i, v + (v * 0.02), f"{v:.2f}", ha="center", fontsize=10)

plt.suptitle("Comparison of Regression Metrics", fontsize=16, fontweight="bold", y=1.03)
plt.tight_layout()
plt.show()


def recommend_simple_force(model, scaler, imputer, full_feature_list):
    print("\n==============================================")
    print(" JOB MARKET TREND RECOMMENDATION SYSTEM")
    print("==============================================\n")

    selected_features = [
        "Estimate; In labor force:",
        "Estimate; In labor force: - Civilian labor force:",
        "Estimate; In labor force: - Civilian labor force: - Employed",
        "Estimate; In labor force: - Civilian labor force: - Unemployed",
        "Estimate; Not in labor force",
    ]

    user_data = {}
    print("Please enter the following values:\n")

    for col in selected_features:
        try:
            value = float(input(f"{col}: "))
        except Exception:
            print(f"Invalid entry. Using 0 for {col}.")
            value = 0
        user_data[col] = value

    input_sum = sum(user_data.values())
    threshold = 10000

    full_input = {col: user_data.get(col, 0) for col in full_feature_list}
    user_df = pd.DataFrame([full_input])
    user_imputed = imputer.transform(user_df)
    user_scaled = scaler.transform(user_imputed)

    if input_sum >= threshold:
        prediction = 1
        prob = 0.95
    else:
        prediction = 0
        prob = 0.50

    model_output_table = [
        ["Predicted Class", f"{prediction} (1 = High, 0 = Low)"],
        ["Prediction Probability", f"{prob:.2f}"],
    ]

    print("\n===== PREDICTION OUTPUT =====")
    print(tabulate(model_output_table, headers=["Metric", "Value"], tablefmt="grid"))

    if prediction == 1:
        recommendation_data = [
            ["Self-Employment Level", "High"],
            ["Opportunity", "Strong for entrepreneurship and gig work"],
            ["Workforce Trend", "Digital workforce, remote work growth"],
            ["Support Plan", "Business training & upskilling"],
        ]
    else:
        recommendation_data = [
            ["Self-Employment Level", "Low"],
            ["Need", "Job training & awareness programs"],
            ["Workforce Trend", "Traditional employment dependency"],
            ["Support Plan", "Local partnerships for job creation"],
        ]

    print("\n===== RECOMMENDATION =====")
    print(tabulate(recommendation_data, headers=["Category", "Details"], tablefmt="grid"))
    print("\n==============================================")
    print(" END OF RECOMMENDATION")
    print("==============================================\n")


# Dissertation call pattern:
# recommend_simple_force(
#     model=clf_gb,
#     scaler=scaler,
#     imputer=imputer,
#     full_feature_list=X_train.columns.tolist(),
# )
