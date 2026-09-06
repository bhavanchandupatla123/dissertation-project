import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

combined = pd.read_csv("combined_dataset.csv")

# Correlation Matrix
numeric_cols = combined.select_dtypes(include="number")
valid_numeric = numeric_cols.loc[:, numeric_cols.apply(lambda x: x.nunique() > 1)]
selected_cols = valid_numeric.iloc[:, :10]
corr_matrix = selected_cols.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="viridis", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# Distribution of Master's Degree Holders in Leading Neighborhoods
masters_col = "Estimate; Total: - Master's degree"
colors = plt.cm.rainbow(np.linspace(0, 1, len(combined)))
top_neighborhoods = combined[["Neighborhood", masters_col]].sort_values(
    masters_col, ascending=False
).head(10)

plt.figure(figsize=(12, 6))
bars = plt.bar(
    top_neighborhoods["Neighborhood"],
    top_neighborhoods[masters_col],
    color=colors[: len(top_neighborhoods)],
)
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval + 2,
        int(yval),
        ha="center",
        va="bottom",
        fontsize=10,
    )
plt.xticks(rotation=45, ha="right")
plt.title("Distribution of Master's Degree Holders in Leading Neighborhoods")
plt.xlabel("Neighborhood")
plt.ylabel("Number of Master's Degree Holders")
plt.tight_layout()
plt.show()

# Unemployment Count by Neighborhood
unemp_col = "Estimate; In labor force: - Civilian labor force: - Unemployed"
sorted_df = combined.sort_values("Neighborhood")
plt.figure(figsize=(14, 6))
plt.plot(sorted_df["Neighborhood"], sorted_df[unemp_col], marker="o", color="#FF00FF")
plt.xticks(rotation=90)
plt.title("Unemployment Count by Neighborhood")
plt.xlabel("Neighborhood")
plt.ylabel("Unemployment Count")
plt.tight_layout()
plt.show()

# Distribution of Self-Employment Income
self_emp_col = "Estimate; Total: - With self-employment income"
top10 = combined[["Neighborhood", self_emp_col]].sort_values(
    self_emp_col, ascending=False
).head(10)

plt.figure(figsize=(8, 8))
plt.pie(
    top10[self_emp_col],
    labels=top10["Neighborhood"],
    autopct="%1.1f%%",
    startangle=90,
)
plt.title("Distribution of Self-Employment Income")
plt.tight_layout()
plt.show()

# Frequency of Educational Total
column = "Estimate; Total:_edu"
plt.figure(figsize=(10, 6))
plt.hist(combined[column].dropna(), bins=20, edgecolor="black", color="#BBF90F")
plt.title("Frequency of Educational Total")
plt.xlabel("Educational Total")
plt.ylabel("Frequency")
plt.show()
