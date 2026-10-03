# Task 2: Exploratory Data Analysis (EDA)
**AI & ML Internship — Elevate Labs**

## Objective
Understand data using statistics and visualizations.

## Tools Used
Python, Pandas, Matplotlib, Seaborn
(Plotly was in the suggested tool list — the plots here are done in
Matplotlib/Seaborn since they render directly to static image files for the
repo; swapping in Plotly for interactive versions is a drop-in change if
you'd like that instead.)

## Dataset
Titanic dataset (`data/titanic.csv`) — same dataset used in Task 1.

> **Note on the data file:** generated locally with `generate_dataset.py`
> to match the real Kaggle Titanic dataset's columns and distributions,
> since this environment has no internet access to download the original.
> Swap in the real `train.csv` from
> [Kaggle's Titanic competition](https://www.kaggle.com/competitions/titanic/data)
> if you'd like — `eda.py` needs no changes since the columns match exactly.

## Project Structure
```
├── data/
│   └── titanic.csv                  # dataset used for analysis
├── images/
│   ├── histograms_numeric_features.png
│   ├── boxplots_numeric_features.png
│   ├── categorical_counts.png
│   ├── survival_rate_by_sex_pclass.png
│   ├── correlation_matrix.png
│   └── pairplot.png
├── generate_dataset.py              # creates the dataset
├── eda.py                           # full EDA script
├── eda_findings.txt                 # auto-generated written findings
├── requirements.txt
└── README.md
```

## How to Run
```bash
pip install -r requirements.txt
python generate_dataset.py    # only needed if data/titanic.csv doesn't exist
python eda.py
```

## What Was Done

1. **Summary statistics** — `.describe()`, mean, median, std, skewness, and
   kurtosis for all numeric features (`Age`, `SibSp`, `Parch`, `Fare`), plus
   survival rate breakdowns by `Sex` and `Pclass`.
2. **Histograms & boxplots** — distribution shape and outliers visualized
   for every numeric feature, plus count plots for categorical features
   (`Survived`, `Pclass`, `Sex`) and survival-rate bar charts split by
   `Sex` and `Pclass`.
3. **Correlation matrix & pairplot** — a heatmap of pairwise correlations
   between numeric features, and a Seaborn pairplot colored by `Survived`
   to visually inspect relationships and separability.
4. **Patterns, trends & anomalies** — see `eda_findings.txt` for the full
   auto-generated write-up (regenerated fresh every run to match your data).
5. **Feature-level inferences** — summarized below.

## Key Findings
- Overall survival rate is ~52%. **Sex is the strongest visual predictor**
  — women survive at a much higher rate than men.
- **Pclass matters**: 1st class passengers survive at a noticeably higher
  rate than 3rd class, consistent with better access to lifeboats.
- **Fare is heavily right-skewed** with a long tail of high-value outliers
  — a handful of passengers paid far more than the median fare.
- **Age is roughly normal** with a slight right skew, concentrated in the
  20s-30s range.
- **Fare and Pclass are strongly negatively correlated** (lower class
  number = higher fare, which is expected since Pclass 1 is the *most*
  expensive class).
- No pair of raw numeric features shows correlation above 0.8, so there's
  no strong multicollinearity concern among them in this dataset.

(Full numeric detail is in the console output of `eda.py` and in
`eda_findings.txt`.)

---

## Interview Questions & Answers

**1. What is the purpose of EDA?**
Exploratory Data Analysis is the process of investigating a dataset before
modeling — summarizing its main characteristics, spotting missing values,
outliers, distributions, and relationships between variables, usually with
statistics and visualizations. Its purpose is to build intuition about the
data, catch data-quality issues early, and guide decisions about
preprocessing and feature engineering before any model is trained.

**2. How do boxplots help in understanding a dataset?**
A boxplot visually summarizes a feature's distribution using its quartiles
(Q1, median, Q3), whiskers, and outlier points. It quickly shows the
spread and skew of the data, and flags outliers (points beyond 1.5×IQR
from Q1/Q3) at a glance — much faster than scanning raw numbers.

**3. What is correlation and why is it useful?**
Correlation measures the strength and direction of a linear relationship
between two numeric variables, typically ranging from -1 (perfect negative)
to +1 (perfect positive). It's useful because it helps identify which
features move together, which features might be redundant
(multicollinear), and which features are likely to be useful predictors
of the target variable.

**4. How do you detect skewness in data?**
Visually, via histograms or density plots — a long tail on one side
indicates skew. Numerically, via the skewness statistic (`.skew()` in
pandas): values near 0 indicate a symmetric distribution, positive values
indicate a right-skewed (long right tail) distribution, and negative
values indicate a left-skewed distribution.

**5. What is multicollinearity?**
Multicollinearity is when two or more independent (predictor) variables in
a dataset are highly correlated with each other. It's a problem for
models like linear/logistic regression because it makes it hard to
isolate the individual effect of each correlated variable on the target,
inflates the variance of coefficient estimates, and can make the model
unstable. It's typically detected via a correlation matrix or the
Variance Inflation Factor (VIF).

**6. What tools do you use for EDA?**
Pandas (data manipulation and summary statistics), Matplotlib and Seaborn
(static visualizations — histograms, boxplots, heatmaps, pairplots),
Plotly (interactive visualizations), and sometimes automated EDA libraries
like `ydata-profiling` (formerly pandas-profiling) or `Sweetviz` for a
quick first-pass report.

**7. Can you explain a time when EDA helped you find a problem?**
In this task, plotting the boxplot for `Fare` immediately revealed a
handful of extreme outliers (fares far above the typical range) that
weren't obvious from `.describe()` alone since the mean can be skewed by
a few large values. That visual check is what justified applying the IQR-
based outlier removal in the preprocessing step, rather than leaving those
extreme values in the training data unexamined.

**8. What is the role of visualization in ML?**
Visualization makes patterns, relationships, and anomalies in data
intuitively visible in a way raw numbers often aren't. It's used before
modeling (to guide cleaning and feature engineering, as in this task),
during modeling (to understand feature importance, decision boundaries,
or model convergence via loss curves), and after modeling (to interpret
predictions, evaluate errors via confusion matrices/ROC curves, and
communicate results to non-technical stakeholders).
