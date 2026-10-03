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
