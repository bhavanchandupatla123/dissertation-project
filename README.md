# AI-Powered Job Market Trend Prediction

**MSc Data Science Individual Project | Coventry University**

A machine-learning project focused on analysing and forecasting labour-market trends by combining multiple socio-economic datasets into a single analytical framework.

## Project overview

The project investigated how factors such as **employment, education, poverty and self-employment income** relate to labour-market outcomes. I built an end-to-end analytical workflow in Python covering data preparation, exploratory analysis, modelling, evaluation and interpretation.

The goal was not simply to produce a prediction, but to create an evidence-based decision-support approach that could help explain workforce patterns and support planning decisions.

## Tech stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Exploratory Data Analysis (EDA)
- Statistical analysis
- Machine Learning
- Data visualisation
- Data quality and preprocessing

## What I built

- Integrated multiple public socio-economic datasets into one analysis-ready dataset.
- Cleaned missing values, duplicate records and outliers.
- Standardised variables before modelling.
- Used descriptive statistics, distributions and correlation analysis to identify trends and influential features.
- Developed and compared supervised machine-learning models including **Random Forest, Gradient Boosting and Decision Tree** approaches.
- Used cross-validation and structured model evaluation to reduce the risk of overfitting.
- Evaluated classification performance using accuracy, precision, recall, F1 and ROC/AUC.
- Evaluated regression performance using R², MAE and RMSE.
- Interpreted model results in the context of workforce planning and responsible decision-making.

## Analytical workflow

```text
Public socio-economic data
        ↓
Data validation & cleaning
        ↓
Dataset integration
        ↓
Exploratory analysis
        ↓
Feature preparation
        ↓
Model development
        ↓
Model comparison
        ↓
Evaluation & interpretation
        ↓
Decision-support insights
```

## Source code

The `src/` directory contains the Python implementation recovered from the submitted dissertation code appendix and reformatted into readable source files without changing the core modelling logic or project parameters.

- [`src/data_preparation.py`](src/data_preparation.py) — loads, merges and validates the four socio-economic datasets.
- [`src/eda.py`](src/eda.py) — correlation analysis and visual exploration of education, unemployment and self-employment indicators.
- [`src/classification_models.py`](src/classification_models.py) — Random Forest and Gradient Boosting classification, confusion matrices, ROC curves and model comparison.
- [`src/regression_and_recommendation.py`](src/regression_and_recommendation.py) — Decision Tree and Random Forest regression plus the Job Market Trend Recommendation System.

## Selected results

| Model / Metric | Result |
|---|---:|
| Gradient Boosting classification accuracy | **96%** |
| Random Forest classification accuracy | **92%** |
| Gradient Boosting ROC AUC | **0.99** |
| Decision Tree regression R² | **0.82** |
| Decision Tree regression MAE | **0.04** |
| Decision Tree regression RMSE | **0.21** |

The results showed Gradient Boosting performing strongly in the classification task, while the Decision Tree regression model explained approximately 82% of the variance in its target.

## Responsible analytics

The project also considered:

- Data privacy
- Fairness and bias
- Transparency and explainability
- Model robustness
- Traceability
- Human judgement in AI-supported decisions

A key principle of the project was that automated predictions should **support rather than replace informed human decision-making**.

## Business relevance

Although completed as an academic Data Science project, the workflow maps directly to practical analyst work:

- **Data quality:** validating and cleaning multiple data sources before analysis.
- **Trend analysis:** identifying meaningful patterns and unusual behaviour.
- **Performance measurement:** comparing alternative models using objective metrics.
- **Decision support:** translating technical results into clear interpretations.
- **Risk awareness:** considering robustness, bias, privacy and governance.

## Repository structure

```text
dissertation-project/
├── README.md
├── src/
│   ├── data_preparation.py
│   ├── eda.py
│   ├── classification_models.py
│   └── regression_and_recommendation.py
├── docs/
│   ├── methodology.md
│   └── results.md
└── requirements.txt
```

## Key takeaway

This project demonstrates my ability to take a complex analytical problem from **raw data → cleaning → analysis → modelling → evaluation → communication**, while maintaining a strong focus on data quality and responsible use of AI.

---

**Bhavan Chandupatla**  
MSc Data Science, Coventry University  
Portfolio: [bhavananalyst.com](https://bhavananalyst.com)  
GitHub: [bhavanchandupatla123](https://github.com/bhavanchandupatla123)
