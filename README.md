# DA-machine-learning-task
Data on Notebook:
https://colab.research.google.com/drive/1sLzWAqUTjloAG8GtAIOeGRC8_tnkEK0Z?usp=sharing
# 📱 Customer Churn Prediction — Classification Model

> **Level 3 Advanced ML Task** — Predict whether a telecom customer will cancel their subscription using Machine Learning.

---

## 📋 Project Overview

A telecom company wants to know: **which customers are likely to leave?**

This is a **Classification problem** — the model predicts one of two outcomes:
- `1` → Customer will churn ❌
- `0` → Customer will stay ✅

---

## 📂 Dataset

| File | Purpose | Size |
|------|---------|------|
| `churn-bigml-80.csv` | Training data | 80% of data |
| `churn-bigml-20.csv` | Testing data | 20% of data |

### Key Features

| Column | Description |
|--------|-------------|
| `State` | Customer's state |
| `Account length` | Days since account opened |
| `International plan` | Has international plan? Yes/No |
| `Voice mail plan` | Has voicemail plan? Yes/No |
| `Total day minutes` | Total daytime call minutes |
| `Total eve minutes` | Total evening call minutes |
| `Total night minutes` | Total night call minutes |
| `Total intl minutes` | Total international call minutes |
| `Customer service calls` | Number of calls to customer service |
| `Churn` | **Target** — Did customer leave? True/False |

---

## 🛠️ Tools & Libraries

```python
pandas
scikit-learn
matplotlib
```

---

## 🔄 Project Pipeline

```
Raw CSV Data
      ↓
1. Preprocessing
      ↓
2. Train 3 Models (on 80%)
      ↓
3. Grid Search / Hyperparameter Tuning (on 80%)
      ↓
4. Predict (on 20%)
      ↓
5. Evaluate with 4 Metrics (on 20%)
      ↓
Final Result 🎯
```

---

## ⚙️ Step 1 — Preprocessing

### Encoding (Text → Numbers)
The model only understands numbers, so we convert text columns:
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

X['State'] = le.fit_transform(X['State'])
X['International plan'] = le.fit_transform(X['International plan'])
X['Voice mail plan'] = le.fit_transform(X['Voice mail plan'])
```

### Feature Scaling
Normalize all numbers to the same range so no feature dominates:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)   # transform only — not fit!
```

> ⚠️ On test data we use `.transform()` only — the scaler already learned from the training data.

---

## 🤖 Step 2 — Train 3 Models

### 1. Decision Tree
Makes decisions using a series of if/else questions learned from data.
```python
from sklearn.tree import DecisionTreeClassifier
Dec_tree = DecisionTreeClassifier()
Dec_tree.fit(X_train, y_train)
```

### 2. Logistic Regression
Draws a decision boundary to separate churners from non-churners.
```python
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
```

### 3. Random Forest
Combines 100+ decision trees and takes a majority vote.
```python
from sklearn.ensemble import RandomForestClassifier
Ran_for = RandomForestClassifier()
Ran_for.fit(X_train, y_train)
```

---

## 🔧 Step 3 — Hyperparameter Tuning (Grid Search)

Automatically tests all parameter combinations to find the best settings:

```python
from sklearn.model_selection import GridSearchCV

# Decision Tree
dt_params = {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]}
dt_grid = GridSearchCV(DecisionTreeClassifier(), dt_params, cv=5)
dt_grid.fit(X_train, y_train)

# Logistic Regression
lr_params = {'C': [0.1, 1, 10], 'max_iter': [100, 200]}
lr_grid = GridSearchCV(LogisticRegression(), lr_params, cv=5)
lr_grid.fit(X_train, y_train)

# Random Forest
rf_params = {'n_estimators': [50, 100], 'max_depth': [3, 5, 10]}
rf_grid = GridSearchCV(RandomForestClassifier(), rf_params, cv=5)
rf_grid.fit(X_train, y_train)
```

---

## 📊 Step 4 — Evaluate on Test Data (20%)

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

models = {
    'Decision Tree': dt_grid.predict(X_test),
    'Logistic Regression': lr_grid.predict(X_test),
    'Random Forest': rf_grid.predict(X_test)
}

for name, pred in models.items():
    print(f"--- {name} ---")
    print(f"Accuracy:  {accuracy_score(y_test, pred):.2f}")
    print(f"Precision: {precision_score(y_test, pred):.2f}")
    print(f"Recall:    {recall_score(y_test, pred):.2f}")
    print(f"F1-Score:  {f1_score(y_test, pred):.2f}")
```

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Decision Tree** | **0.95** | 0.85 | 0.77 | **0.81** |
| Logistic Regression | 0.85 | 0.44 | 0.18 | 0.25 |
| **Random Forest** | 0.94 | **0.95** | 0.62 | 0.75 |

### 🏆 Best Model: Decision Tree
- Highest overall accuracy (95%)
- Best F1-Score (0.81)
- Strong balance between Precision and Recall

### ❌ Worst Model: Logistic Regression
- Low Recall (0.18) — misses most customers who actually churn
- Low F1-Score (0.25)

---

## 📖 Metrics Explained

| Metric | Meaning |
|--------|---------|
| **Accuracy** | Out of all customers, how many did we predict correctly? |
| **Precision** | When we said "will churn", how often were we right? |
| **Recall** | Out of all customers who actually churned, how many did we catch? |
| **F1-Score** | Balanced average of Precision and Recall |

> **Why not just use Accuracy?** If only 5% of customers churn, a model that predicts "nobody churns" gets 95% accuracy but catches zero churners. Precision, Recall & F1 reveal the real story.

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/your-username/churn-prediction.git

# 2. Install dependencies
pip install pandas scikit-learn matplotlib

# 3. Open the notebook
jupyter notebook "Predictive Modelling Task.ipynb"
```

---

## 📁 Project Structure

```
churn-prediction/
│
├── churn-bigml-80.csv          # Training data
├── churn-bigml-20.csv          # Testing data
├── Predictive Modelling Task.ipynb  # Main notebook
└── README.md                   # This file
```
