"""
Task 2: Exploratory Data Analysis (EDA)
AI & ML Internship - Elevate Labs

Objective: Understand data using statistics and visualizations.
Dataset: Titanic dataset (data/titanic.csv)

Steps covered:
1. Generate summary statistics (mean, median, std, etc.)
2. Create histograms and boxplots for numeric features
3. Use pairplot/correlation matrix for feature relationships
4. Identify patterns, trends, or anomalies in the data
5. Make basic feature-level inferences from visuals
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

df = pd.read_csv("data/titanic.csv")

# ---------------------------------------------------------------
# STEP 1: Summary statistics
# ---------------------------------------------------------------
print("=" * 60)
print("STEP 1: SUMMARY STATISTICS")
print("=" * 60)

print("\nShape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())

numeric_cols = ["Age", "SibSp", "Parch", "Fare"]
print("\nDescribe (numeric features):\n", df[numeric_cols].describe())

print("\nMean:\n", df[numeric_cols].mean())
print("\nMedian:\n", df[numeric_cols].median())
print("\nStd Dev:\n", df[numeric_cols].std())
print("\nSkewness:\n", df[numeric_cols].skew())
print("\nKurtosis:\n", df[numeric_cols].kurt())

print("\nSurvival rate overall:", round(df["Survived"].mean(), 3))
print("\nSurvival rate by Sex:\n", df.groupby("Sex")["Survived"].mean())
print("\nSurvival rate by Pclass:\n", df.groupby("Pclass")["Survived"].mean())

# ---------------------------------------------------------------
# STEP 2: Histograms and boxplots for numeric features
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: HISTOGRAMS & BOXPLOTS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, col in zip(axes.flatten(), numeric_cols):
    sns.histplot(df[col].dropna(), kde=True, ax=ax, color="steelblue")
    ax.set_title(f"Distribution of {col}")
plt.tight_layout()
plt.savefig("images/histograms_numeric_features.png", dpi=120)
plt.close()
print("\nSaved images/histograms_numeric_features.png")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, col in zip(axes.flatten(), numeric_cols):
    sns.boxplot(y=df[col], ax=ax, color="lightcoral")
    ax.set_title(f"Boxplot of {col}")
plt.tight_layout()
plt.savefig("images/boxplots_numeric_features.png", dpi=120)
plt.close()
print("Saved images/boxplots_numeric_features.png")

# Categorical count plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.countplot(data=df, x="Survived", hue="Survived", ax=axes[0], palette="Set2", legend=False)
axes[0].set_title("Survival Count (0 = No, 1 = Yes)")
sns.countplot(data=df, x="Pclass", hue="Pclass", ax=axes[1], palette="Set2", legend=False)
axes[1].set_title("Passenger Class Count")
sns.countplot(data=df, x="Sex", hue="Sex", ax=axes[2], palette="Set2", legend=False)
axes[2].set_title("Sex Count")
plt.tight_layout()
plt.savefig("images/categorical_counts.png", dpi=120)
plt.close()
print("Saved images/categorical_counts.png")

# Survival rate broken down by Sex and Pclass
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.barplot(data=df, x="Sex", y="Survived", hue="Sex", ax=axes[0], palette="Set1", legend=False)
axes[0].set_title("Survival Rate by Sex")
sns.barplot(data=df, x="Pclass", y="Survived", hue="Pclass", ax=axes[1], palette="Set1", legend=False)
axes[1].set_title("Survival Rate by Passenger Class")
plt.tight_layout()
plt.savefig("images/survival_rate_by_sex_pclass.png", dpi=120)
plt.close()
print("Saved images/survival_rate_by_sex_pclass.png")

# ---------------------------------------------------------------
# STEP 3: Correlation matrix and pairplot
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: CORRELATION MATRIX & PAIRPLOT")
print("=" * 60)

corr_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
corr_matrix = df[corr_cols].corr()
print("\nCorrelation matrix:\n", corr_matrix.round(2))

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation Matrix (Numeric Features)")
plt.tight_layout()
plt.savefig("images/correlation_matrix.png", dpi=120)
plt.close()
print("\nSaved images/correlation_matrix.png")

pairplot_fig = sns.pairplot(
    df[corr_cols + ["Sex"]].dropna(),
    hue="Survived",
    palette="husl",
    diag_kind="kde",
    plot_kws={"alpha": 0.5, "s": 20},
)
pairplot_fig.fig.suptitle("Pairplot of Numeric Features by Survival", y=1.02)
pairplot_fig.savefig("images/pairplot.png", dpi=120)
plt.close()
print("Saved images/pairplot.png")

# ---------------------------------------------------------------
# STEP 4 & 5: Patterns, trends, anomalies + feature-level inferences
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4 & 5: PATTERNS, TRENDS, ANOMALIES & INFERENCES")
print("=" * 60)

findings = f"""
Key observations from this EDA run:

1. Survival: Overall survival rate is {round(df['Survived'].mean()*100, 1)}%.
   Females survive at a much higher rate ({round(df.groupby('Sex')['Survived'].mean()['female']*100,1)}%)
   than males ({round(df.groupby('Sex')['Survived'].mean()['male']*100,1)}%) -> Sex is a strong predictor.

2. Passenger class: 1st class passengers survive at a higher rate than 3rd class
   ({round(df.groupby('Pclass')['Survived'].mean()[1]*100,1)}% vs
   {round(df.groupby('Pclass')['Survived'].mean()[3]*100,1)}%), suggesting Pclass
   (likely a proxy for socio-economic status / cabin location) matters for survival.

3. Fare is heavily right-skewed (skew = {round(df['Fare'].skew(), 2)}) with a long tail of
   high-fare outliers, visible clearly in both the histogram and the boxplot.

4. Age is roughly bell-shaped but has a slight right skew (skew = {round(df['Age'].skew(), 2)}),
   with a concentration of passengers in their 20s-30s.

5. Correlation matrix: Fare and Pclass are negatively correlated
   ({round(corr_matrix.loc['Fare','Pclass'], 2)}), which makes sense since lower Pclass
   number (1st class) pays a higher fare. Survived correlates negatively with Pclass
   ({round(corr_matrix.loc['Survived','Pclass'], 2)}), reinforcing observation #2.

6. SibSp and Parch show a {'positive' if corr_matrix.loc['SibSp','Parch'] >= 0 else 'negative'}
   correlation with each other ({round(corr_matrix.loc['SibSp','Parch'], 2)}). In the real
   Titanic dataset these two are positively correlated since both describe family
   relationships aboard; the small magnitude here (and its sign) can vary with sampling.

7. No single pair of numeric features shows extremely high correlation (>0.8), so there's
   no obvious multicollinearity issue among the raw numeric features in this dataset.
"""
print(findings)

with open("eda_findings.txt", "w") as f:
    f.write(findings)
print("Saved eda_findings.txt")
