# Task 2: Exploratory Data Analysis (EDA) — Titanic Dataset

**Internship:** AI & ML Internship — Elevate Labs
**Objective:** Understand data using statistics and visualizations.
**Tools used:** Pandas, Matplotlib, Seaborn

## Repo structure
```
.
├── data/
│   └── titanic.csv              # Dataset (891 passengers, 12 columns)
├── notebook/
│   └── eda_titanic.py           # Full EDA script
├── visuals/
│   ├── 01_histograms.png
│   ├── 02_boxplots.png
│   ├── 03_boxplots_by_survival.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_pairplot.png
│   └── 06_categorical_survival.png
└── README.md
```

## How to run
```bash
pip install pandas numpy matplotlib seaborn
cd notebook
python eda_titanic.py
```

## 1. Summary statistics
- 891 rows, 12 columns. Numeric features analyzed: `Age`, `Fare`, `SibSp`, `Parch`, `Pclass`.
- **Age**: mean ≈ 29.7, median 28, std ≈ 14.5 — roughly symmetric (skew ≈ 0.39).
- **Fare**: mean ≈ 32.2, median 14.45, std ≈ 49.7 — heavily right-skewed (skew ≈ 4.79), driven by a few very expensive first-class tickets (max ≈ 512).
- **SibSp / Parch**: most passengers traveled alone (median 0 for both), with long right tails.
- Overall survival rate: **38.4%**.

## 2. Missing values
| Column | Missing | % |
|---|---|---|
| Cabin | 687 | 77.1% |
| Age | 177 | 19.9% |
| Embarked | 2 | 0.2% |

`Cabin` is too sparse to use directly (would need to be engineered, e.g. "has cabin" flag). `Age` has meaningful gaps typically imputed with median/group-median.

## 3. Key visual findings

**Histograms (`01_histograms.png`)**
- Age is fairly bell-shaped, centered in the 20–35 range, with a small bump for young children.
- Fare is extremely right-skewed — most fares are low, with a long tail of expensive tickets.

**Boxplots (`02_boxplots.png`, `03_boxplots_by_survival.png`)**
- Fare shows many outliers above ~$65, consistent with 1st class fares.
- Age has a few outliers at the high end (elderly passengers) and low end (infants).
- Survivors skew slightly younger and paid noticeably higher fares on average than non-survivors — a visible spread in the Fare-by-Survived boxplot.

**Correlation heatmap (`04_correlation_heatmap.png`)**
- `Pclass` and `Fare` are strongly negatively correlated (**-0.55**) — makes sense, lower class number (1st class) pays more.
- `Survived` correlates negatively with `Pclass` (**-0.34**) and positively with `Fare` (**0.26**) — wealthier, higher-class passengers were more likely to survive.
- `SibSp` and `Parch` are moderately correlated (**0.41**) — family members tend to travel together.
- No signs of severe multicollinearity among predictors (no pair besides Pclass/Fare exceeds ~0.5).

**Pairplot (`05_pairplot.png`)**
- Confirms the class/fare/survival relationship visually — survivors (orange/hue) cluster at lower Pclass and higher Fare.

**Categorical breakdown (`06_categorical_survival.png`)**
- **Sex** is the single strongest survival signal: women survived at **74.2%** vs men at **18.9%**.
- **Pclass**: 1st class 63.0% survived, 2nd class 47.3%, 3rd class 24.2% — clear class gradient.
- **Embarked**: passengers boarding at Cherbourg (C) had a noticeably higher survival rate than Southampton (S) or Queenstown (Q), likely confounded with class/fare.

## 4. Feature-level inferences
- **Sex** and **Pclass** are the two most powerful predictors of survival ("women and children first," plus class-based access to lifeboats).
- **Fare** is a proxy for class and correlates positively with survival, but it's highly skewed, so a model would benefit from a log-transform.
- **Age** has too many missing values to use raw; median imputation (ideally grouped by Pclass/Sex) is standard practice.
- **Cabin** is mostly missing but could be engineered into a binary "has_cabin" feature as a proxy for class/wealth.
- **SibSp/Parch** can be combined into a "family size" feature; very large families (SibSp/Parch outliers) tend to have lower survival, possibly due to difficulty staying together while evacuating.

---

## Interview Questions & Answers

**1. What is the purpose of EDA?**
EDA (Exploratory Data Analysis) is the process of examining a dataset before modeling — using statistics and visualizations — to understand its structure, spot missing values or errors, detect patterns/relationships between variables, and form hypotheses. It guides decisions about cleaning, feature engineering, and which models make sense.

**2. How do boxplots help in understanding a dataset?**
A boxplot summarizes a numeric variable's distribution using its median, quartiles (IQR), and whiskers, and flags points beyond 1.5×IQR as outliers. It makes it quick to spot skewness, spread, and outliers, and to compare distributions across categories (e.g., Fare by Survived).

**3. What is correlation and why is it useful?**
Correlation measures the strength and direction of the linear relationship between two numeric variables, ranging from -1 to +1. It's useful for spotting which features move together, guiding feature selection, and flagging redundant (highly correlated) features before modeling.

**4. How do you detect skewness in data?**
Visually, via histograms or KDE plots (a long tail on one side) or boxplots (asymmetric whisker lengths / many outliers on one side). Numerically, via the `skew()` statistic — close to 0 means roughly symmetric; a large positive value means a right (long tail toward high values) skew, as seen with `Fare` here (skew ≈ 4.79).

**5. What is multicollinearity?**
Multicollinearity is when two or more predictor variables are highly correlated with each other, making it hard for a model (especially linear/logistic regression) to isolate each variable's individual effect, inflating coefficient variance. It's detected via correlation matrices or VIF (Variance Inflation Factor) and addressed by dropping/combining redundant features.

**6. What tools do you use for EDA?**
Pandas for data loading, cleaning, and summary statistics; Matplotlib and Seaborn for static visualizations (histograms, boxplots, heatmaps, pairplots); Plotly for interactive visualizations; and Jupyter notebooks as the working environment.

**7. Can you explain a time when EDA helped you find a problem?**
In this Titanic EDA, checking missing values upfront revealed that `Cabin` is 77% empty (unusable as-is) and `Age` is ~20% missing — without that check, a model could silently drop most rows or be trained on badly imputed data. Similarly, the correlation heatmap surfaced that `Pclass` and `Fare` are strongly correlated, a multicollinearity risk that would need addressing before using both in a linear model.

**8. What is the role of visualization in ML?**
Visualization turns raw numbers into patterns a human can quickly interpret — it reveals distributions, outliers, class imbalance, and relationships between features and the target that summary statistics alone can miss. It supports feature engineering decisions, sanity-checks assumptions, and later helps communicate model results and errors to stakeholders.
