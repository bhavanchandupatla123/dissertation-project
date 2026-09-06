import numpy as np
import pandas as pd

# Loading the datasets
employment_status = pd.read_csv(
    "employment-status-for-the-population-16-years-and-over.csv"
)
educational_attainment = pd.read_csv(
    "educational-attainment-for-the-population-25-years-and-over.csv"
)
self_employment_income = pd.read_csv(
    "self-employment-income-in-the-past-12-months-for-households.csv"
)
poverty_status = pd.read_csv(
    "poverty-status-of-individuals-in-the-past-12-months-by-living-arrangement.csv"
)

# Showing the datasets
print(employment_status.head())
print(educational_attainment.head())
print(self_employment_income.head())
print(poverty_status.head())

# Ensuring consistent column names in case of slight variations
poverty_status = poverty_status.rename(columns={"hborhood": "Neighborhood"})

# Merging all datasets on Neighborhood and Id
combined = (
    employment_status
    .merge(
        educational_attainment,
        on=["Neighborhood", "Id"],
        how="left",
        suffixes=("", "_edu"),
    )
    .merge(
        self_employment_income,
        on=["Neighborhood", "Id"],
        how="left",
        suffixes=("", "_selfemp"),
    )
    .merge(
        poverty_status,
        on=["Neighborhood", "Id"],
        how="left",
        suffixes=("", "_poverty"),
    )
)

# Viewing and saving the combined dataframe
print(combined.head())
combined.to_csv("combined_dataset.csv", index=False)

# Data-quality checks
combined.drop_duplicates(inplace=True)
print(combined.isna().sum())
print(combined.describe())
