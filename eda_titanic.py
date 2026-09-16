"""
Task 2: Exploratory Data Analysis (EDA) - Titanic Dataset
AI & ML Internship - Elevate Labs

Objective: Understand data using statistics and visualizations.
Tools: Pandas, Matplotlib, Seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

DATA_PATH = "../data/titanic.csv"
OUT_DIR = "../visuals"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("1. BASIC DATASET OVERVIEW")
print("=" * 60)
print(f"Shape: {df.shape}")
print("\nColumn dtypes:\n", df.dtypes)
print("\nFirst 5 rows:\n", df.head())

# ---------------------------------------------------------
# 2. Missing values
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("2. MISSING VALUES")
print("=" * 60)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_table = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
missing_table = missing_table[missing_table.missing_count > 0].sort_values(
    "missing_count", ascending=False
)
print(missing_table)

# ---------------------------------------------------------
# 3. Summary statistics
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("3. SUMMARY STATISTICS (numeric features)")
print("=" * 60)
numeric_cols = ["Age", "Fare", "SibSp", "Parch", "Pclass"]
summary = df[numeric_cols].describe().T
summary["median"] = df[numeric_cols].median()
summary["skew"] = df[numeric_cols].skew()
print(summary)

print("\nSurvival rate overall: {:.2%}".format(df["Survived"].mean()))
print("\nSurvival rate by Sex:\n", df.groupby("Sex")["Survived"].mean())
print("\nSurvival rate by Pclass:\n", df.groupby("Pclass")["Survived"].mean())

# ---------------------------------------------------------
# 4. Histograms for numeric features
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
hist_cols = ["Age", "Fare", "SibSp", "Parch", "Pclass"]
for ax, col in zip(axes.flatten(), hist_cols):
    sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#4C72B0")
    ax.set_title(f"Distribution of {col}")
axes.flatten()[-1].axis("off")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_histograms.png")
plt.close()
print("\nSaved: 01_histograms.png")

# ---------------------------------------------------------
# 5. Boxplots (outlier detection)
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(y=df["Age"], ax=axes[0], color="#55A868")
axes[0].set_title("Boxplot: Age")
sns.boxplot(y=df["Fare"], ax=axes[1], color="#C44E52")
axes[1].set_title("Boxplot: Fare")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_boxplots.png")
plt.close()
print("Saved: 02_boxplots.png")

# Boxplot of Fare/Age split by Survived
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(x="Survived", y="Age", data=df, ax=axes[0], palette="Set2")
axes[0].set_title("Age by Survival")
sns.boxplot(x="Survived", y="Fare", data=df, ax=axes[1], palette="Set2")
axes[1].set_title("Fare by Survival")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_boxplots_by_survival.png")
plt.close()
print("Saved: 03_boxplots_by_survival.png")

# ---------------------------------------------------------
# 6. Correlation matrix / heatmap
# ---------------------------------------------------------
corr_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
corr = df[corr_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_correlation_heatmap.png")
plt.close()
print("Saved: 04_correlation_heatmap.png")

# ---------------------------------------------------------
# 7. Pairplot
# ---------------------------------------------------------
pair_df = df[corr_cols].dropna()
g = sns.pairplot(pair_df, hue="Survived", diag_kind="hist", palette="husl")
g.fig.suptitle("Pairplot of Key Features by Survival", y=1.02)
g.savefig(f"{OUT_DIR}/05_pairplot.png")
plt.close()
print("Saved: 05_pairplot.png")

# ---------------------------------------------------------
# 8. Categorical breakdowns
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.countplot(x="Sex", hue="Survived", data=df, ax=axes[0], palette="Set1")
axes[0].set_title("Survival Count by Sex")
sns.countplot(x="Pclass", hue="Survived", data=df, ax=axes[1], palette="Set1")
axes[1].set_title("Survival Count by Pclass")
sns.countplot(x="Embarked", hue="Survived", data=df, ax=axes[2], palette="Set1")
axes[2].set_title("Survival Count by Embarked")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/06_categorical_survival.png")
plt.close()
print("Saved: 06_categorical_survival.png")

print("\n" + "=" * 60)
print("EDA COMPLETE. All visuals saved to:", OUT_DIR)
print("=" * 60)
